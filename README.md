# pairwise-thumb-atlas-pack

## 领域

游戏、图形与多媒体

## 主题

缩略图图集打包

## 说明

padding 隔离、mip 独立与确定性布局。

初始状态下 `imaging/thumb_atlas_pack.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
