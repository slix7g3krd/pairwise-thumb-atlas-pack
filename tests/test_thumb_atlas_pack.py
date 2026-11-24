import pytest

import imaging.thumb_atlas_pack as mod


# 契约测试：每个公开入口对探测调用都要返回非 None。
# 注意 (1,) 必须是元组；写成 (1) 是整数，fn(*1) 会在调用前解包失败。
CASES = [
    ("padding_bleed", (1,)),
    ("mip_isolate", (1,)),
    ("uv_cache", (1,)),
    ("format_group", (1,)),
    ("layout_determinism", (1,)),
    ("size_align", (1,)),
    ("pack", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("alpha_bucket", (1,)),
    ("metrics", (1,)),
    ("rebuild", (1,)),
    ("sampling", (1,)),
    ("determinism", (1,)),
]


@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None


# ---------------------------------------------------------------------------
# 修复 1 + 6：padding 按尺寸档位计算并留出血；非 2 的幂对齐到块边界
# ---------------------------------------------------------------------------

def test_size_align_non_power_of_two():
    r = mod.size_align(33, 17, block=4)
    assert (r["w"], r["h"]) == (36, 20)
    assert r["w"] % 4 == 0 and r["h"] % 4 == 0
    assert r["pad_w"] == 3 and r["pad_h"] == 3


def test_size_align_already_multiple():
    r = mod.size_align(32, 16, block=4)
    assert (r["w"], r["h"]) == (32, 16) and r["pad_w"] == 0 and r["pad_h"] == 0


def test_padding_grows_with_tier_and_includes_mip_bleed():
    small = mod.padding_bleed(32, 32)
    large = mod.padding_bleed(256, 256)
    assert large["pad"] > small["pad"]                 # 档位越高 padding 越宽
    assert large["tier"] > small["tier"]
    # 出血至少覆盖档位基础 padding + 全部 mip 级的滤波脚印
    assert large["pad"] >= large["base_pad"] + large["mip_levels"] - 1
    # 保留单元 = 图像 + 两侧 padding，且整体对齐块边界
    assert large["cell_w"] == 256 + 2 * large["pad"]
    assert large["cell_w"] % large["block"] == 0


def test_padding_bleed_single_arg_is_square():
    r = mod.padding_bleed(64)
    assert r["w"] == r["h"] == 64 and r["cell_w"] == r["cell_h"]


# ---------------------------------------------------------------------------
# 修复 2：mip 按单图边界隔离生成
# ---------------------------------------------------------------------------

def test_mip_isolated_does_not_pull_neighbor_color():
    # 单色 4x4 红图：任意级别下采样后仍应是纯红，无邻居污染
    red = [255, 0, 0] * 16
    chain = mod.mip_isolate({"id": "r", "w": 4, "h": 4, "pixels": red})
    assert [c["w"] for c in chain] == [4, 2, 1]
    for level in chain:
        if level["pixels"] is not None:
            assert all(v in (255, 0) for v in level["pixels"])
    assert chain[1]["pixels"] == [255, 0, 0] * 4
    assert chain[2]["pixels"] == [255, 0, 0]


def test_mip_isolated_odd_and_degenerate_dims():
    for w, h in [(33, 17), (2, 1), (1, 2), (1, 1), (5, 3)]:
        chain = mod.mip_isolate({"id": "x", "w": w, "h": h,
                                 "pixels": list(range(w * h * 3))})
        for c in chain:  # 每级缓冲与尺寸自洽，杜绝越界
            assert len(c["pixels"]) == c["w"] * c["h"] * 3
        assert chain[-1]["w"] == 1 and chain[-1]["h"] == 1


def test_mip_isolated_rgba_channels_preserved():
    px = [10, 20, 30, 128] * 4  # 2x2 RGBA
    chain = mod.mip_isolate({"id": "t", "w": 2, "h": 2, "pixels": px})
    assert chain[1]["channels"] == 4
    assert chain[1]["pixels"] == [10, 20, 30, 128]


# ---------------------------------------------------------------------------
# 修复 3：重建时 UV 缓存整体失效
# ---------------------------------------------------------------------------

def _mixed_images():
    return [
        {"id": "opaque", "w": 4, "h": 4, "pixels": [10, 20, 30] * 16},
        {"id": "cut", "w": 2, "h": 2,
         "pixels": [1, 1, 1, 0, 2, 2, 2, 255, 3, 3, 3, 0, 4, 4, 4, 255]},
        {"id": "blend", "w": 2, "h": 2, "pixels": [1, 1, 1, 128] * 4},
    ]


def test_uv_cache_versioned_and_invalidated_on_rebuild():
    images = _mixed_images()
    atlas = mod.pack(images, page_size=256)
    entry = mod.uv_cache(atlas, "opaque")
    assert entry is not None and entry.version == atlas.version

    rebuilt = mod.rebuild(images, atlas, page_size=256)
    assert rebuilt.version == atlas.version + 1
    # 新图集只接受新版本的 UV，材质手里的旧句柄无法命中
    fresh = mod.uv_cache(rebuilt, "opaque")
    assert fresh.version == rebuilt.version and fresh.version != entry.version
    # 旧图集缓存被整体清空
    assert mod.uv_cache(atlas) == {}


def test_uv_cache_missing_image_returns_none():
    atlas = mod.pack(_mixed_images(), page_size=256)
    assert mod.uv_cache(atlas, "does_not_exist") is None


# ---------------------------------------------------------------------------
# 修复 4：按 alpha 与压缩格式分组，同页单一格式
# ---------------------------------------------------------------------------

def test_alpha_buckets_classify_by_real_alpha():
    buckets = mod.alpha_bucket(_mixed_images())
    assert buckets["opaque"] == ["opaque"]
    assert buckets["cutout"] == ["cut"]
    assert buckets["blend"] == ["blend"]


def test_format_group_separates_alpha_and_format():
    groups, keys = mod.format_group(_mixed_images())
    # 不透明组带 False，透明组带 True，二者永不落在同一组/页
    assert any(k == ("BC1", False) for k in keys)
    assert any(k[1] is True for k in keys)
    for (fmt, has_alpha), ids in groups.items():
        assert all(isinstance(i, str) for i in ids)
        assert isinstance(fmt, str)


def test_packed_pages_use_single_format():
    atlas = mod.pack(_mixed_images(), page_size=256)
    for page in atlas.pages:
        assert len({pl["fmt"] for pl in page.placements}) == 1
        assert all(pl["has_alpha"] == page.has_alpha for pl in page.placements)


# ---------------------------------------------------------------------------
# 修复 5：布局确定性排序，两次打包哈希一致
# ---------------------------------------------------------------------------

def test_layout_determinism_order_stable_under_shuffle():
    images = _mixed_images()
    assert mod.layout_determinism(images) == mod.layout_determinism(
        list(reversed(images))
    )


def test_determinism_same_hash_and_uv_across_runs():
    images = _mixed_images()
    r = mod.determinism(images, page_size=256)
    assert r["deterministic"]
    assert r["hash_a"] == r["hash_b"]
    assert r["same_uv"] and r["same_order"] and r["same_layout"]


# ---------------------------------------------------------------------------
# 全流程：无重叠、块对齐、mip 隔离、审计通过
# ---------------------------------------------------------------------------

def test_pack_has_no_overlap_and_passes_validation():
    images = [
        {"id": "a", "w": 33, "h": 17, "pixels": list(range(33 * 17 * 3))},
        {"id": "b", "w": 64, "h": 64, "pixels": [1, 1, 1, 128] * (64 * 64)},
        {"id": "c", "w": 48, "h": 48,
         "pixels": [1, 1, 1, 0, 2, 2, 2, 255] * (48 * 48 // 2)},
        {"id": "d", "w": 100, "h": 50, "pixels": list(range(100 * 50 * 3))},
    ]
    atlas = mod.pack(images, page_size=256)
    assert mod.validation(atlas) == []
    audit = mod.audit(atlas)
    assert audit["ok"] and audit["findings"] == []

    for page in atlas.pages:
        for i in range(len(page.occupied)):
            for j in range(i + 1, len(page.occupied)):
                assert not page.occupied[i].intersects(page.occupied[j])
        for pl in page.placements:
            assert pl["w"] % 4 == 0 and pl["h"] % 4 == 0
        for cells in page.mip_cells:
            assert all(c["isolated"] for c in cells)


def test_oversized_image_goes_to_fallback_texture():
    big = {"id": "huge", "w": 2000, "h": 300, "pixels": list(range(2000 * 300 * 3))}
    atlas = mod.pack([big], page_size=256)
    assert "huge" in atlas.fallback
    single = mod.fallback(big)
    assert mod.validation(single) == []
    assert single.cache["huge"].version == 1


def test_metrics_reported():
    atlas = mod.pack(_mixed_images(), page_size=256)
    mt = mod.metrics(atlas)
    assert mt["pages"] >= 1 and mt["placements"] == 3
    assert 0.0 <= mt["utilization"] <= 1.0
    assert mt["padding_ratio"] >= 0.0


def test_sampling_flags_bleed_risk_at_high_mip():
    safe = mod.sampling((0.5, 0.5), 0, (0.1, 0.1, 0.9, 0.9), (64, 64))
    risky = mod.sampling((0.105, 0.5), 2, (0.1, 0.1, 0.9, 0.9), (64, 64))
    assert safe["safe"] is True
    assert risky["safe"] is False
