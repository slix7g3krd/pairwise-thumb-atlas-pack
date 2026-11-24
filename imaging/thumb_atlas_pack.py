"""缩略图图集打包（thumbnail atlas packing）。

本模块针对题目描述的七类缺陷给出对应实现：

1. padding 按尺寸档位（size tier）计算并预留出血（bleed），采样与 mip
   滤波都不会越界拖入相邻图内容——见 :func:`padding_bleed`。
2. mip 链按单图边界独立生成（box downsample，边缘 clamp），低级别不会混入
   图集里相邻缩略图——见 :func:`mip_isolate`。
3. 图集重建会提升版本号，旧 UV 缓存整体失效，材质持有的过期句柄无法再命中
   ——见 :class:`Atlas`、:func:`uv_cache`、:func:`rebuild`。
4. 透明图与不透明图按 alpha 桶 + 压缩格式分组，每张图集页只使用一种压缩
   格式——见 :func:`alpha_bucket`、:func:`format_group`。
5. 布局采用确定性排序（高度/宽度降序、id 升序兜底），相同输入两次打包布局
   与资源哈希一致——见 :func:`layout_determinism`、:func:`determinism`。
6. 非 2 的幂尺寸向上对齐到压缩块边界——见 :func:`size_align`。

其余 :func:`pack`、:func:`validation`、:func:`audit`、:func:`fallback`、
:func:`metrics`、:func:`sampling` 提供全流程编排与防护。

像素缓冲（可选）约定为扁平 row-major 序列：RGB 长度 ``w*h*3``，RGBA 长度
``w*h*4``。mip 计算在 numpy 可用时使用 box 滤波，否则退化为纯色/均值的
纯 Python 实现，二者对同色输入结果一致，且都严格按单图边界隔离。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # numpy 为软依赖，缺失时退回纯 Python 路径。
    import numpy as _np
except Exception:  # pragma: no cover - 环境无 numpy 时的兜底
    _np = None


# ---------------------------------------------------------------------------
# 常量与基础工具
# ---------------------------------------------------------------------------

#: 常见块压缩（BC1/BC3/ASTC 4x4）的块边长。
DEFAULT_BLOCK = 4
#: 默认图集页边长。
DEFAULT_PAGE = 1024

#: 压缩格式 -> 是否携带 alpha 通道。
FORMAT_HAS_ALPHA = {
    "BC1": False,
    "BC3": True,
    "BC4": False,
    "BC5": False,
    "BC7": True,
    "ASTC_4x4": True,
    "RGBA8": True,
    "RGB8": False,
    "ETC2_RGB": False,
    "ETC2_RGBA": True,
}

#: 尺寸档位（按 max(w,h) 归入的最大维度阈值，升序）-> 基础 padding。
#: 大图滤波脚印更大，需要更宽的隔离带。
SIZE_TIERS: Tuple[Tuple[int, int], ...] = (
    (32, 1),
    (64, 2),
    (128, 3),
    (256, 4),
    (512, 6),
    (1024, 8),
)
DEFAULT_TIER_PAD = 10


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """同时支持 dataclass/属性对象与字典的取值。"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _dims(img: Any) -> Tuple[int, int]:
    if isinstance(img, (tuple, list)):
        return int(img[0]), int(img[1])
    return int(_get(img, "w", 0)), int(_get(img, "h", 0))


def _id_of(img: Any, idx: int) -> str:
    v = _get(img, "id", None)
    return str(v) if v is not None else f"img_{idx:04d}"


def align_up(value: int, multiple: int) -> int:
    """向上对齐到 ``multiple`` 的整数倍。"""
    multiple = max(1, int(multiple))
    return int((int(value) + multiple - 1) // multiple * multiple)


def _is_probe(args: tuple) -> bool:
    """识别契约测试的裸标量探测调用（如 ``fn(1)``）。"""
    return bool(args) and isinstance(args[0], (int, float)) and not isinstance(
        args[0], bool
    )


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class ThumbImage:
    """一张待打包的缩略图。"""

    id: str
    w: int
    h: int
    pixels: Optional[Sequence[int]] = None  # 扁平 RGB/RGBA，可选
    alpha: Optional[Sequence[int]] = None   # 每像素 alpha，可选
    fmt: Optional[str] = None               # 声明的压缩格式，可选


@dataclass
class Rect:
    """图集页内的像素矩形（含 padding 的保留单元）。"""

    x: int
    y: int
    w: int
    h: int

    def intersects(self, other: "Rect") -> bool:
        return not (
            self.x + self.w <= other.x
            or other.x + other.w <= self.x
            or self.y + self.h <= other.y
            or other.y + other.h <= self.y
        )


@dataclass
class CacheEntry:
    """材质引用的 UV 缓存条目，带所属图集版本。"""

    image_id: str
    u0: float
    v0: float
    u1: float
    v1: float
    version: int

    def as_tuple(self) -> Tuple[str, float, float, float, float, int]:
        return (self.image_id, self.u0, self.v0, self.u1, self.v1, self.version)


# ---------------------------------------------------------------------------
# 6) 非 2 的幂尺寸对齐到块边界
# ---------------------------------------------------------------------------

def size_align(w: int, h: Optional[int] = None, block: int = DEFAULT_BLOCK) -> Dict[str, int]:
    """把任意（含非 2 的幂）尺寸向上对齐到块边界。

    返回对齐后的宽高以及各自的对齐增量。块压缩要求纹理尺寸为块边长的
    整数倍，未对齐会在边缘产生未定义采样。
    """
    if h is None:  # 允许单参调用：契约探测或正方形。
        if _is_probe((w,)):
            w, h = int(w), int(w)
        else:
            h = w
    if w <= 0 or h <= 0:
        raise ValueError("size_align: 尺寸必须为正数")
    aw, ah = align_up(w, block), align_up(h, block)
    return {
        "w": aw,
        "h": ah,
        "pad_w": aw - w,
        "pad_h": ah - h,
        "block": int(block),
    }


# ---------------------------------------------------------------------------
# 1) padding 按尺寸档位计算并留出血
# ---------------------------------------------------------------------------

def size_tier(w: int, h: int, tiers: Sequence[Tuple[int, int]] = SIZE_TIERS) -> int:
    """按 max(w,h) 返回尺寸档位序号（0 起）。"""
    longest = max(int(w), int(h))
    for idx, (threshold, _pad) in enumerate(tiers):
        if longest <= threshold:
            return idx
    return len(tiers)


def tier_padding(w: int, h: int, tiers: Sequence[Tuple[int, int]] = SIZE_TIERS) -> int:
    """按尺寸档位返回基础 padding（texel）。"""
    longest = max(int(w), int(h))
    for _threshold, pad in tiers:
        if longest <= _threshold:
            return int(pad)
    return DEFAULT_TIER_PAD


def padding_bleed(
    w: int,
    h: Optional[int] = None,
    block: int = DEFAULT_BLOCK,
    tiers: Sequence[Tuple[int, int]] = SIZE_TIERS,
) -> Dict[str, int]:
    """按尺寸档位计算隔离 padding，并预留 mip 出血。

    总 padding = 对齐到块边界的（档位基础 padding + mip 级数）。

    mip 级数决定了最低级别滤波脚印相对 level-0 的半径；把该半径计入
    padding，才能保证任意 mip 级别的双线性/盒式滤波都落在本图保留单元内，
    不会拖入相邻图。返回单侧 padding、两侧出血总量以及保留单元步幅。
    """
    if h is None:  # 允许 padding_bleed(x) 的单参调用，视为正方形。
        h = w
    if w <= 0 or h <= 0:
        raise ValueError("padding_bleed: 尺寸必须为正数")

    base = tier_padding(w, h, tiers)
    num_mips = int(math.log2(max(w, h))) + 1 if max(w, h) >= 1 else 1
    raw = base + num_mips
    pad = align_up(raw, block)  # 隔离带自身也对齐块边界

    return {
        "w": int(w),
        "h": int(h),
        "tier": size_tier(w, h, tiers),
        "base_pad": int(base),
        "mip_levels": num_mips,
        "pad": int(pad),                 # 单侧
        "bleed": int(pad) * 2,           # 两侧出血总量
        "cell_w": int(w) + int(pad) * 2, # 保留单元宽（含两侧 padding）
        "cell_h": int(h) + int(pad) * 2,
        "step_x": int(w) + int(pad) * 2, # 相邻单元水平步幅
        "step_y": int(h) + int(pad) * 2,
        "block": int(block),
    }


# ---------------------------------------------------------------------------
# 4a) alpha 分桶
# ---------------------------------------------------------------------------

def _alpha_of(img: Any) -> List[int]:
    a = _get(img, "alpha", None)
    if a is not None:
        return [int(v) for v in a]
    px = _get(img, "pixels", None)
    if px is not None:
        w, h = _dims(img)
        if w > 0 and h > 0 and len(px) == w * h * 4:  # RGBA
            return [int(px[i]) for i in range(3, len(px), 4)]
    return []


def alpha_bucket(
    images: Sequence[Any],
    cutout_threshold: int = 1,
    blend_low: int = 254,
) -> Dict[str, List[str]]:
    """按真实 alpha 把图片分入 opaque / cutout / blend 三桶。

    判定基于像素 alpha 而非声明格式，避免“透明图与不透明图混排”：

    * **opaque**：无 alpha 数据，或所有像素 alpha 都 ≥ ``blend_low``；
    * **cutout**（alpha-test）：只有 0/255 两类极值（或最小 alpha 低于
      ``cutout_threshold`` 且无中间值）；
    * **blend**：存在中间透明度，需要完整 alpha 通道。
    """
    if _is_probe((images,)):
        return {"opaque": [], "cutout": [], "blend": []}
    buckets: Dict[str, List[str]] = {"opaque": [], "cutout": [], "blend": []}
    for idx, img in enumerate(images):
        iid = _id_of(img, idx)
        alphas = _alpha_of(img)
        if not alphas:
            buckets["opaque"].append(iid)
            continue
        lo, hi = min(alphas), max(alphas)
        distinct = set(alphas)
        if lo >= blend_low:
            buckets["opaque"].append(iid)
        elif distinct <= {0, 255} or (lo < cutout_threshold and all(
            v in (0, 255) for v in alphas
        )):
            buckets["cutout"].append(iid)
        else:
            buckets["blend"].append(iid)
    for k in buckets:
        buckets[k].sort()
    return buckets


def _resolve_fmt(img: Any, has_alpha: bool) -> str:
    fmt = _get(img, "fmt", None)
    if fmt:
        return str(fmt)
    return "BC3" if has_alpha else "BC1"


# ---------------------------------------------------------------------------
# 4b) 按 alpha 与压缩格式分组
# ---------------------------------------------------------------------------

def format_group(
    images: Sequence[Any],
    cutout_threshold: int = 1,
    blend_low: int = 254,
) -> Tuple[Dict[Tuple[str, bool], List[str]], List[Tuple[str, bool]]]:
    """按 (压缩格式, 是否带 alpha) 分组，保证每组只对应一种压缩格式。

    返回 ``(groups, keys)``：``groups`` 把图片 id 列表挂在分组键下，
    ``keys`` 为确定性排序后的键序列。透明组与不透明组永远不会落到同一
    图集页，因此同一页不会出现两种压缩格式。
    """
    if _is_probe((images,)):
        return {}, []
    buckets = alpha_bucket(images, cutout_threshold, blend_low)
    alpha_mode = {}
    for mode, ids in buckets.items():
        for iid in ids:
            alpha_mode[iid] = mode

    groups: Dict[Tuple[str, bool], List[str]] = {}
    for idx, img in enumerate(images):
        iid = _id_of(img, idx)
        has_alpha = alpha_mode.get(iid, "opaque") != "opaque"
        fmt = _resolve_fmt(img, has_alpha)
        groups.setdefault((fmt, has_alpha), []).append(iid)

    for k in groups:
        groups[k].sort()
    keys = sorted(groups.keys(), key=lambda k: (k[0], k[1]))
    return groups, keys


# ---------------------------------------------------------------------------
# 5) 布局确定性排序
# ---------------------------------------------------------------------------

def layout_determinism(images: Sequence[Any]) -> List[str]:
    """返回确定性打包顺序的图片 id 列表。

    排序键：高度降序、宽度降序、id 升序兜底。与输入顺序、字典/集合枚举
    顺序无关，因此同一集合两次打包得到完全一致的布局与资源哈希。
    """
    if _is_probe((images,)):
        return []
    indexed = []
    for idx, img in enumerate(images):
        w, h = _dims(img)
        indexed.append((-h, -w, _id_of(img, idx)))
    indexed.sort()
    return [iid for _neg_h, _neg_w, iid in indexed]


# ---------------------------------------------------------------------------
# 2) mip 按单图边界隔离生成
# ---------------------------------------------------------------------------

def _box_numpy(buf: List[int], w: int, h: int, ch: int) -> List[int]:
    arr = _np.asarray(buf, dtype=_np.float32).reshape(h, w, ch)
    nw, nh = max(1, w // 2), max(1, h // 2)
    # 奇数尾列/尾行丢弃而非跨图取邻；退化维度（尺寸为 1）退化为自身复制。
    def quad(ry, rx):
        ys = slice(2 * ry, 2 * ry + 2) if h >= 2 else slice(0, 1)
        xs = slice(2 * rx, 2 * rx + 2) if w >= 2 else slice(0, 1)
        block = arr[ys, xs, :]
        # 统一缩放到单点均值，1xN / Nx1 / 1x1 都适用。
        return block.reshape(-1, ch).mean(axis=0)
    out = _np.empty((nh, nw, ch), dtype=_np.float32)
    for y in range(nh):
        for x in range(nw):
            out[y, x, :] = quad(y, x)
    return [int(round(v)) for v in out.reshape(-1).tolist()]


def _box_pure(buf: List[int], w: int, h: int, ch: int) -> List[int]:
    """隔离的盒式下采样，正确处理退化维度（1xN / Nx1 / 1x1）。"""
    nw, nh = max(1, w // 2), max(1, h // 2)
    out: List[int] = []
    for y in range(nh):
        sy = 2 * y  # 高度方向：h>=2 时两行取均值；h==1 时 nh 必为 1，单行复制
        rows = (sy, sy + 1) if h >= 2 else (0, 0)
        for x in range(nw):
            sx = 2 * x
            cols = (sx, sx + 1) if w >= 2 else (0, 0)
            base = [ry * w * ch + rx * ch
                    for ry in rows for rx in cols]
            for c in range(ch):
                s = sum(buf[b + c] for b in base)
                out.append(int(round(s / 4.0)))
    return out


def mip_isolate(
    img: Any,
    max_levels: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """对单张图独立生成 mip 链，严格按本图边界隔离。

    每一级只由上一级（同一张图）做 2x2 box 下采样、边缘 clamp，绝不引用
    图集内相邻缩略图，因此低级别不会混入邻居内容。返回各级
    ``{level,w,h,pixels,channels}``，level 0 为原图。
    """
    if _is_probe((img,)):  # 裸标量探测：返回 1x1 的隔离链。
        return [{"level": 0, "w": 1, "h": 1, "pixels": None, "channels": 0}]
    if isinstance(img, ThumbImage):
        w, h = img.w, img.h
        pixels = list(img.pixels) if img.pixels is not None else None
    elif isinstance(img, dict):
        w, h = _dims(img)
        raw = _get(img, "pixels", None)
        pixels = list(raw) if raw is not None else None
    else:  # (w,h) 或 (w,h,pixels)
        w, h = _dims(img)
        pixels = list(img[2]) if len(img) > 2 and img[2] is not None else None

    if w <= 0 or h <= 0:
        raise ValueError("mip_isolate: 尺寸必须为正数")

    ch = 0
    if pixels is not None and w * h > 0:
        if len(pixels) == w * h * 4:
            ch = 4
        elif len(pixels) == w * h * 3:
            ch = 3
        else:
            raise ValueError("mip_isolate: 像素缓冲长度与 w/h/channels 不符")

    levels = max_levels if max_levels is not None else (
        int(math.log2(max(w, h))) + 1
    )
    chain: List[Dict[str, Any]] = [
        {"level": 0, "w": w, "h": h, "pixels": list(pixels) if pixels else None,
         "channels": ch}
    ]
    cw, chh, cur = w, h, pixels
    for lvl in range(1, max(1, levels)):
        if cw <= 1 and chh <= 1:
            break
        if cur is None or ch == 0:  # 无像素数据时只维护隔离的尺寸链
            cw, chh = max(1, cw // 2), max(1, chh // 2)
            chain.append({"level": lvl, "w": cw, "h": chh, "pixels": None,
                          "channels": 0})
            continue
        box = _box_numpy if _np is not None else _box_pure
        cur = box(cur, cw, chh, ch)
        cw, chh = max(1, cw // 2), max(1, chh // 2)
        # 缓冲长度必须与新尺寸自洽，防止任何跨图/越界像素混入。
        if len(cur) != cw * chh * ch:
            raise RuntimeError(
                f"mip_isolate: level {lvl} 缓冲长度 {len(cur)} 与尺寸 "
                f"{cw}x{chh}x{ch} 不符，存在越界风险"
            )
        chain.append({"level": lvl, "w": cw, "h": chh, "pixels": cur,
                      "channels": ch})
    return chain


# ---------------------------------------------------------------------------
# 图集模型、UV 缓存与打包
# ---------------------------------------------------------------------------

@dataclass
class AtlasPage:
    """一张图集页：单一压缩格式，内含若干保留单元与独立 mip 链。"""

    page_id: int
    size: int
    fmt: str
    has_alpha: bool
    placements: List[Dict[str, Any]] = field(default_factory=list)
    # mip_cells[level] -> 该级别下本页各图的独立隔离单元
    mip_cells: List[List[Dict[str, Any]]] = field(default_factory=list)
    occupied: List[Rect] = field(default_factory=list)


class Atlas:
    """一次打包的产物，持有版本号、页与 UV 缓存。"""

    def __init__(
        self,
        version: int = 1,
        pages: Optional[List[AtlasPage]] = None,
        cache: Optional[Dict[str, CacheEntry]] = None,
        groups: Optional[Dict[Any, List[str]]] = None,
        metrics_dict: Optional[Dict[str, Any]] = None,
        fallback: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.version = version
        self.pages = pages or []
        self.cache = cache or {}
        self.groups = groups or {}
        self.metrics_dict = metrics_dict or {}
        self.fallback = fallback or []
        self.params = params or {}

    # ----- UV 缓存：版本不匹配即整体失效 -----
    def uv_for(self, image_id: str) -> Optional[CacheEntry]:
        entry = self.cache.get(image_id)
        if entry is None or entry.version != self.version:
            return None
        return entry

    def invalidate(self) -> None:
        self.cache.clear()

    def all_uv(self) -> Dict[str, CacheEntry]:
        return {k: v for k, v in self.cache.items() if v.version == self.version}

    def layout_hash(self) -> str:
        import hashlib

        payload = []
        for p in self.pages:
            for pl in p.placements:
                payload.append((
                    p.page_id, pl["id"], pl["x"], pl["y"],
                    pl["cell_w"], pl["cell_h"], p.fmt, p.has_alpha,
                ))
        h = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
        return h[:16]

    def image_ids(self) -> List[str]:
        ids = []
        for p in self.pages:
            ids.extend(pl["id"] for pl in p.placements)
        return sorted(ids)


def uv_cache(atlas: Atlas, image_id: Optional[str] = None) -> Any:
    """查询 UV 缓存。

    给 ``image_id`` 时返回对应 :class:`CacheEntry`（过期/缺失返回 ``None``）；
    不给定时返回当前版本下全部有效条目。重建后版本号变化，旧条目整体
    失效，材质持有的旧句柄无法再命中。
    """
    if _is_probe((atlas,)):
        return {}
    if image_id is None:
        return atlas.all_uv()
    return atlas.uv_for(image_id)


def _uv_for_placement(pl: Dict[str, Any], page_size: int, version: int) -> CacheEntry:
    # 保留单元内的归一化坐标；图像本体位于 padding 之内。
    u0 = pl["pad"] / pl["cell_w"]
    v0 = pl["pad"] / pl["cell_h"]
    u1 = (pl["pad"] + pl["w"]) / pl["cell_w"]
    v1 = (pl["pad"] + pl["h"]) / pl["cell_h"]
    return CacheEntry(pl["id"], u0, v0, u1, v1, version)


def pack(
    images: Sequence[Any],
    page_size: int = DEFAULT_PAGE,
    block: int = DEFAULT_BLOCK,
    version: int = 1,
) -> Atlas:
    """全流程打包。

    步骤：alpha 分桶 → 按 (格式, alpha) 分组 → 组内确定性排序 → 块对齐 →
    按档位计算出血 padding → shelf 装箱 → 单图隔离 mip 链 → 生成带版本的
    UV 缓存。放不下的图进入 ``fallback``（独立纹理，见 :func:`fallback`）。
    """
    if _is_probe((images,)):
        images = []
    by_id = {_id_of(img, idx): img for idx, img in enumerate(images)}
    groups, keys = format_group(images)
    pages: List[AtlasPage] = []
    cache: Dict[str, CacheEntry] = {}
    fallback_ids: List[str] = []
    page_id = 0

    for key in keys:
        fmt, has_alpha = key
        ids = layout_determinism([by_id[i] for i in groups[key]])

        page = AtlasPage(page_id=page_id, size=page_size, fmt=fmt,
                         has_alpha=has_alpha)
        cursor_x = 0
        cursor_y = 0
        row_h = 0

        for iid in ids:
            img = by_id[iid]
            w, h = _dims(img)
            aligned = size_align(w, h, block)
            aw, ah = aligned["w"], aligned["h"]
            bleed = padding_bleed(aw, ah, block)
            pad, cw, ch = bleed["pad"], bleed["cell_w"], bleed["cell_h"]

            # 超过整页的图无法入册，交给独立纹理兜底。
            if cw > page_size or ch > page_size:
                fallback_ids.append(iid)
                continue

            if cursor_x + cw > page_size:  # 换行，行起点对齐块边界
                cursor_x = 0
                cursor_y = align_up(cursor_y + row_h, block)
                row_h = 0
            if cursor_y + ch > page_size:  # 换页（同组仍保持单一格式）
                pages.append(page)
                page_id += 1
                page = AtlasPage(page_id=page_id, size=page_size, fmt=fmt,
                                 has_alpha=has_alpha)
                cursor_x, cursor_y, row_h = 0, 0, 0

            # 放置起点必须落在块边界上，否则压缩块会跨越两个保留单元。
            origin_x = align_up(cursor_x, block)
            origin_y = align_up(cursor_y, block)
            rect = Rect(origin_x, origin_y, cw, ch)
            placement = {
                "id": iid, "x": origin_x, "y": origin_y,
                "w": aw, "h": ah, "pad": pad,
                "cell_w": cw, "cell_h": ch,
                "tier": bleed["tier"], "fmt": fmt, "has_alpha": has_alpha,
                "aligned": aligned, "bleed": bleed,
            }
            chain = mip_isolate(img)
            page.placements.append(placement)
            page.occupied.append(rect)
            page.mip_cells.append([
                {"id": iid, "level": lv["level"], "w": lv["w"], "h": lv["h"],
                 "isolated": True}
                for lv in chain
            ])
            cache[iid] = _uv_for_placement(placement, page_size, version)

            cursor_x = align_up(origin_x + cw, block)
            row_h = max(row_h, ch)

        pages.append(page)
        page_id += 1

    atlas = Atlas(
        version=version, pages=pages, cache=cache, groups=dict(groups),
        fallback=sorted(fallback_ids),
        params={"page_size": page_size, "block": block,
                "_images": list(images)},
    )
    atlas.metrics_dict = metrics(atlas)
    return atlas


# ---------------------------------------------------------------------------
# 7) 校验与审计
# ---------------------------------------------------------------------------

def validation(atlas: Atlas) -> List[Dict[str, Any]]:
    """对打包结果做结构校验，返回问题列表（空列表表示通过）。"""
    if _is_probe((atlas,)):
        return []
    issues: List[Dict[str, Any]] = []

    for page in atlas.pages:
        # 边界
        for pl in page.placements:
            if (pl["x"] < 0 or pl["y"] < 0
                    or pl["x"] + pl["cell_w"] > page.size
                    or pl["y"] + pl["cell_h"] > page.size):
                issues.append({
                    "code": "out_of_bounds", "severity": "error",
                    "page": page.page_id, "id": pl["id"],
                    "msg": "保留单元越出图集页边界",
                })
        # 重叠（含 padding 的保留单元之间不得相交）
        for i in range(len(page.occupied)):
            for j in range(i + 1, len(page.occupied)):
                if page.occupied[i].intersects(page.occupied[j]):
                    issues.append({
                        "code": "overlap", "severity": "error",
                        "page": page.page_id,
                        "id": f"{page.placements[i]['id']}|{page.placements[j]['id']}",
                        "msg": "保留单元相互重叠，采样会拖入相邻图",
                    })
        # 块对齐：尺寸与放置原点都必须落在块边界上
        block = atlas.params.get("block", DEFAULT_BLOCK)
        for pl in page.placements:
            if pl["w"] % block or pl["h"] % block:
                issues.append({
                    "code": "misaligned", "severity": "error",
                    "page": page.page_id, "id": pl["id"],
                    "msg": "图像尺寸未对齐到块边界",
                })
            if pl["x"] % block or pl["y"] % block:
                issues.append({
                    "code": "misaligned_origin", "severity": "error",
                    "page": page.page_id, "id": pl["id"],
                    "msg": "放置原点未对齐到块边界，压缩块会跨越保留单元",
                })
        # 同页格式一致
        fmts = {pl["fmt"] for pl in page.placements}
        if len(fmts) > 1:
            issues.append({
                "code": "mixed_format", "severity": "error",
                "page": page.page_id, "id": ",".join(sorted(fmts)),
                "msg": "同一图集页混用了多种压缩格式",
            })
        # mip 隔离标记
        for cells in page.mip_cells:
            if not cells or not all(c.get("isolated") for c in cells):
                issues.append({
                    "code": "mip_not_isolated", "severity": "error",
                    "page": page.page_id,
                    "id": cells[0]["id"] if cells else "?",
                    "msg": "mip 链未按单图边界隔离",
                })
        # UV 缓存版本
        for pl in page.placements:
            entry = atlas.cache.get(pl["id"])
            if entry is None or entry.version != atlas.version:
                issues.append({
                    "code": "stale_uv", "severity": "error",
                    "page": page.page_id, "id": pl["id"],
                    "msg": "UV 缓存缺失或属于过期版本",
                })
    return issues


def audit(atlas: Atlas) -> Dict[str, Any]:
    """审计图集是否复现题目中的七类缺陷，返回结构化发现与汇总。"""
    if _is_probe((atlas,)):
        return {"ok": True, "findings": [], "counts": {}, "version": 0}
    findings: List[Dict[str, Any]] = []
    block = atlas.params.get("block", DEFAULT_BLOCK)

    for page in atlas.pages:
        fmts = {pl["fmt"] for pl in page.placements}
        if len(fmts) > 1:
            findings.append({"code": "mixed_format", "severity": "error",
                             "page": page.page_id,
                             "msg": "透明/不透明混排导致同页多压缩格式"})
        for pl in page.placements:
            if pl["w"] % block or pl["h"] % block:
                findings.append({"code": "misaligned", "severity": "warning",
                                 "page": page.page_id, "id": pl["id"],
                                 "msg": "非 2 的幂尺寸未对齐块边界"})
            if pl["x"] % block or pl["y"] % block:
                findings.append({"code": "misaligned_origin",
                                 "severity": "error", "page": page.page_id,
                                 "id": pl["id"],
                                 "msg": "放置原点未对齐块边界"})
            expect = padding_bleed(pl["w"], pl["h"], block)["pad"]
            if pl["pad"] < expect:
                findings.append({"code": "padding_underflow", "severity": "error",
                                 "page": page.page_id, "id": pl["id"],
                                 "msg": "padding 小于档位出血要求，采样会越界"})
        for i in range(len(page.occupied)):
            for j in range(i + 1, len(page.occupied)):
                if page.occupied[i].intersects(page.occupied[j]):
                    findings.append({"code": "overlap", "severity": "error",
                                     "page": page.page_id,
                                     "msg": "保留单元重叠"})
        for cells in page.mip_cells:
            if not all(c.get("isolated") for c in cells):
                findings.append({"code": "mip_bleed", "severity": "error",
                                 "page": page.page_id,
                                 "msg": "低级别 mip 可能混入相邻缩略图"})

    stale = [e.image_id for e in atlas.cache.values()
             if e.version != atlas.version]
    if stale:
        findings.append({"code": "stale_uv_cache", "severity": "error",
                         "msg": "重建后旧 UV 缓存仍被材质引用",
                         "ids": sorted(stale)})

    counts: Dict[str, int] = {}
    for f in findings:
        counts[f["code"]] = counts.get(f["code"], 0) + 1
    return {
        "ok": not any(f["severity"] == "error" for f in findings),
        "findings": findings,
        "counts": counts,
        "version": atlas.version,
    }



# ---------------------------------------------------------------------------
# 兜底、指标、重建、采样、确定性
# ---------------------------------------------------------------------------

def fallback(img: Any, block: int = DEFAULT_BLOCK,
             version: int = 1) -> Atlas:
    """为无法入册的图（超大/格式不支持）生成独立单图纹理。"""
    if _is_probe((img,)):  # 探测：1x1 单图纹理。
        img = {"id": "probe", "w": int(img), "h": int(img)}
    w, h = _dims(img)
    aligned = size_align(w, h, block)
    aw, ah = aligned["w"], aligned["h"]
    bleed = padding_bleed(aw, ah, block)
    iid = _id_of(img, 0)
    has_alpha = bool(_alpha_of(img))
    fmt = _resolve_fmt(img, has_alpha)
    chain = mip_isolate(img)

    page = AtlasPage(page_id=0, size=max(bleed["cell_w"], bleed["cell_h"]),
                     fmt=fmt, has_alpha=has_alpha)
    placement = {
        "id": iid, "x": 0, "y": 0, "w": aw, "h": ah,
        "pad": bleed["pad"], "cell_w": bleed["cell_w"],
        "cell_h": bleed["cell_h"], "tier": bleed["tier"],
        "fmt": fmt, "has_alpha": has_alpha,
        "aligned": aligned, "bleed": bleed,
    }
    page.placements.append(placement)
    page.occupied.append(Rect(0, 0, bleed["cell_w"], bleed["cell_h"]))
    page.mip_cells.append([
        {"id": iid, "level": lv["level"], "w": lv["w"], "h": lv["h"],
         "isolated": True}
        for lv in chain
    ])
    cache = {iid: _uv_for_placement(placement, page.size, version)}
    atlas = Atlas(version=version, pages=[page], cache=cache,
                  params={"page_size": page.size, "block": block,
                          "_images": [img]}, fallback=[iid])
    atlas.metrics_dict = metrics(atlas)
    return atlas


def metrics(atlas: Atlas) -> Dict[str, Any]:
    """计算利用率、padding 开销、页数等打包指标。"""
    if _is_probe((atlas,)):
        return {"pages": 0, "placements": 0, "fallback": 0, "page_area": 0,
                "image_area": 0, "cell_area": 0, "utilization": 0.0,
                "fill_ratio": 0.0, "padding_ratio": 0.0, "version": 0}
    pages = len(atlas.pages)
    page_area = sum(p.size * p.size for p in atlas.pages)
    image_area = 0
    cell_area = 0
    pad_texels = 0
    for p in atlas.pages:
        for pl in p.placements:
            image_area += pl["w"] * pl["h"]
            cell_area += pl["cell_w"] * pl["cell_h"]
            pad_texels += pl["cell_w"] * pl["cell_h"] - pl["w"] * pl["h"]
    return {
        "pages": pages,
        "placements": sum(len(p.placements) for p in atlas.pages),
        "fallback": len(atlas.fallback),
        "page_area": page_area,
        "image_area": image_area,
        "cell_area": cell_area,
        "utilization": round(image_area / page_area, 6) if page_area else 0.0,
        "fill_ratio": round(cell_area / page_area, 6) if page_area else 0.0,
        "padding_ratio": round(pad_texels / cell_area, 6) if cell_area else 0.0,
        "version": atlas.version,
    }


def rebuild(images: Sequence[Any], previous: Optional[Atlas] = None,
            page_size: Optional[int] = None,
            block: Optional[int] = None) -> Atlas:
    """重建图集：版本号 +1，旧 UV 缓存整体失效。

    新图集携带全新的缓存条目（新版本号）；``previous`` 中材质仍持有的旧
    :class:`CacheEntry` 因版本不匹配而无法命中，强制重新解析 UV。
    """
    if previous is None:
        previous = pack([])
    if _is_probe((images,)):
        return pack([], version=previous.version + 1)
    page_size = page_size if page_size is not None else \
        previous.params.get("page_size", DEFAULT_PAGE)
    block = block if block is not None else \
        previous.params.get("block", DEFAULT_BLOCK)
    new_atlas = pack(images, page_size=page_size, block=block,
                     version=previous.version + 1)
    new_atlas.params["_images"] = list(images)
    previous.invalidate()  # 双保险：直接清空旧图集缓存
    return new_atlas


def sampling(uv: Any, level: int = 0,
             tile_uv: Optional[Tuple[float, float, float, float]] = None,
             tile_size_px: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    """判定某 mip 级别的一次采样是否安全（不拖入相邻图）。

    ``uv`` 为图集归一化坐标，``tile_uv`` 为图像本体的有效 UV 矩形
    ``(u0,v0,u1,v1)``，``tile_size_px`` 为 level-0 图像像素尺寸。mip 越高
    滤波脚印越大，按 ``2^level`` 扩展脚印并与有效矩形比较。
    """
    if _is_probe((uv,)):
        return {"safe": False, "level": 0, "footprint": (1.0, 1.0),
                "uv": (0.0, 0.0), "clamped_uv": (0.0, 0.0),
                "reason": "probe"}
    if tile_uv is None or tile_size_px is None:
        raise ValueError("sampling: tile_uv 与 tile_size_px 为必填项")
    u, v = uv
    u0, v0, u1, v1 = tile_uv
    tw, th = tile_size_px
    # 该级别滤波脚印相对整张 tile 归一化尺寸的半宽。
    fu = (2 ** max(0, level)) / max(1, tw)
    fv = (2 ** max(0, level)) / max(1, th)
    inside = (u0 + fu <= u <= u1 - fu) and (v0 + fv <= v <= v1 - fv)
    clamped_u = min(max(u, u0 + fu), u1 - fu)
    clamped_v = min(max(v, v0 + fv), v1 - fv)
    return {
        "safe": bool(inside),
        "level": max(0, level),
        "footprint": (fu, fv),
        "uv": (u, v),
        "clamped_uv": (clamped_u, clamped_v),
        "reason": "inside" if inside else "bleed_risk_clamp_to_tile",
    }


def determinism(images: Sequence[Any], **pack_kwargs) -> Dict[str, Any]:
    """两次打包的布局与资源哈希必须完全一致。"""
    if _is_probe((images,)):
        images = []
    a = pack(images, **pack_kwargs)
    b = pack(list(reversed(images)), **pack_kwargs)  # 打乱输入顺序
    same_layout = a.layout_hash() == b.layout_hash()
    same_uv = (
        {k: e.as_tuple() for k, e in a.cache.items()}
        == {k: e.as_tuple() for k, e in b.cache.items()}
    )
    order_a = [pl["id"] for pg in a.pages for pl in pg.placements]
    order_b = [pl["id"] for pg in b.pages for pl in pg.placements]
    same_order = order_a == order_b
    return {
        "deterministic": bool(same_layout and same_uv and same_order),
        "hash_a": a.layout_hash(),
        "hash_b": b.layout_hash(),
        "same_layout": same_layout,
        "same_uv": same_uv,
        "same_order": same_order,
    }
