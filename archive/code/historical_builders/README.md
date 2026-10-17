# 历史数据构建器

更新：2026-10-17。此处保留退出顶层 `tools/` 的转换器，用于数据回填与历史追溯。当前数据选择、格式和构建命令见 [data/README.md](../../../data/README.md)与 [DATASETS.md](../../../DATASETS.md)。

| 脚本 | 用途与边界 |
|---|---|
| [prepare_pyg_entities_datasets.py](prepare_pyg_entities_datasets.py) | PyG AIFB/MUTAG/BGS/AM 转换；仍是准备当前 `entities_aifb` 的入口，未知标签写为 `-1` |
| [prepare_fraud_datasets.py](prepare_fraud_datasets.py) | CARE-GNN Amazon/Yelp 原始 `.mat` 数据转换 |
| [prepare_bitcoin_wsn_datasets.py](prepare_bitcoin_wsn_datasets.py)、[make_wsn_topk_subgraph.py](make_wsn_topk_subgraph.py) | 有符号信任图转换及子图提取；数据没有可用的已知聚类标签 |
| [prepare_urban_plot_graph.py](prepare_urban_plot_graph.py)、[prepare_urban_plot_graph_v2.py](prepare_urban_plot_graph_v2.py) | 早期 Urban V1/V2；当前 Urban 系列入口在顶层 `tools/` |
| [prepare_edge_attr_control_datasets.py](prepare_edge_attr_control_datasets.py) | 旧边属性对照；依赖顶层 `tools/prepare_mechanism_synth_datasets.py`，且其旧数据不适合作为当前边语义证据 |

在仓库根目录、已配置的根项目环境中准备 AIFB：

```bash
.venv/bin/python archive/code/historical_builders/prepare_pyg_entities_datasets.py --root data --out_root data --datasets AIFB
```

其他构建器应先核对 `--help`、原始数据来源与输出名称。旧 edge-control 脚本的导入需要让 `tools/` 在 Python 搜索路径中；迁移后的脚本不保证无需调整即可复现原运行。不要将输出覆盖到当前数据名，除非已确认需要替换。

归档位置表示历史维护状态，不表示所有转换结果都已退出当前实验；例如 AIFB 仍用于关系图实验。反之，脚本或数据目录存在也不表示其适合作为聚类基准。资产可用性见 [DATA_ASSETS.md](../../../DATA_ASSETS.md)。
