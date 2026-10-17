# 历史数据清单

更新：2026-10-17。本目录只跟踪本 README，历史数据载荷被 Git 忽略；干净克隆不包含下表中的数据。当前数据入口见 [data/README.md](../../data/README.md)，资产位置与可用性见 [DATA_ASSETS.md](../../DATA_ASSETS.md)。

| 原归档路径 | 内容 |
|---|---|
| `historical/dblp/` | 早期 DBLP 图变体 |
| `historical/synth_edge_control/` | 旧边属性对照数据 |
| `historical/synth_smoke/` | 仅用于 smoke 检查的合成数据 |
| `historical/urban_legacy/` | 早于当前 V3b 系列的 Urban 构建输出 |

这些是历史目录类别，不是下载地址或已验证的本机库存。按旧实验恢复前，应核对数据名、元数据、构建器和源数据是否齐全。

旧 `synth_mech_full_v1_*` / `synth_edgectrl_v1_*` 的节点特征泄露了类别结构，其高分不能说明边语义带来增益。当前机制检验使用弱特征 `synth_ctx_v1_*` / `synth_ctxxor_v1_*` 及对应无信号、置乱对照；见 [DATASETS.md](../../DATASETS.md)。
