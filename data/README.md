# 数据准备与文件格式

更新：2026-10-17。Git 只跟踪本目录的 README，**不包含预处理数据、下载缓存或 Urban 原始城市数据**；仓库也未配置完整数据的公开镜像。克隆后请按所需实验准备数据，不要把本机已有目录当成随仓库分发的资产。

- [DATASETS.md](../DATASETS.md)：数据含义、边语义、标签及实验适用范围。
- [DATA_ASSETS.md](../DATA_ASSETS.md)：大文件清单、本机 D: 数据位置、F: 备份及远端资产限制。
- [环境说明](../UV_ENVIRONMENTS.txt)：创建根项目 `.venv`；以下命令均从仓库根目录运行。
- [实验计划](../configs/breakthrough/README.md)和[已跟踪证据](../configs/breakthrough/evidence/README.md)：确认所需数据名与评价协议。

普通机器可直接使用本地 `data/` 目录，无需复刻原工作站的 Windows 盘符或绑定挂载。训练时可用 `--root_path /path/to/data` 指定其他根目录；各构建器的输入、输出参数名称见下方示例。

## 当前数据入口

| 数据名 | 获取方式 | 用途与边界 |
|---|---|---|
| `cora`、`citeseer`、`pubmed`、`computers`、`photo` | 训练加载器在缺失时通过 PyG 下载 | 标准拓扑/节点特征对照；边属性由图和特征派生 |
| `dblp_magnn_author_v2` | [DBLP 构建器](../tools/prepare_pyg_dblp_magnn_dataset.py) | 合作、主题、会议等原生多通道作者图 |
| `acm_multiplex_v2` | [多层图构建器](../tools/prepare_pyg_multiplex_dataset.py) | PAP/PSP/PcP/PTP 并图；已报告的 v2 使用 `psp_sample_k=20` |
| `entities_aifb` | [归档 Entities 转换器](../archive/code/historical_builders/prepare_pyg_entities_datasets.py) | 稀疏已知标签、关系属性；训练前须生成项目格式文件 |
| `synth_ctx_v1_*`、`synth_ctxxor_v1_*` | [上下文合成数据构建器](../tools/prepare_synth_ctx_datasets.py) | 弱节点特征，含无信号、直接边信号、上下文信号及置乱对照 |
| `urban_<city>_plot_v3bsjg`、hetero/v5 系列 | Urban 构建器与另行提供的 `urban_network_datasets/<city>/` | 城市数据构建与有效性研究；土地利用标签不适合作为社区恢复真值 |

```bash
# 原生关系图；下载需要网络
.venv/bin/python tools/prepare_pyg_dblp_magnn_dataset.py --root data/pyg_dblp --out_root data --name dblp_magnn_author_v2 --mode v2
.venv/bin/python tools/prepare_pyg_multiplex_dataset.py --dataset acm --root_path data --name acm_multiplex_v2 --psp_sample_k 20 --seed 0
.venv/bin/python archive/code/historical_builders/prepare_pyg_entities_datasets.py --root data --out_root data --datasets AIFB

# 明确使用报告中的 feature_sep=0.35；生成器默认值是 0.25
.venv/bin/python tools/prepare_synth_ctx_datasets.py --root_path data --feature_sep 0.35 --homophily 0.45,0.55 --data_seeds 0
.venv/bin/python tools/prepare_synth_ctx_datasets.py --root_path data --prefix synth_ctxxor_v1 --feature_sep 0.35 --homophily 0.45,0.55 --modes none,contextual --perm_modes contextual --role_mode xor2 --data_seeds 0
```

`ds00`、`ds01` 表示数据生成种子，与训练的 `--seed` 不同。需要 seed-1 图时使用 `--data_seeds 1`。生成后检查对应 `_meta.json` 与 manifest；同名数据已存在时，不应仅凭目录名认定参数一致，确认后再决定是否使用生成器的 `--overwrite`。

Urban 原始数据不是本仓库可自动恢复的资产。取得原始城市目录后，例如：

```bash
.venv/bin/python tools/prepare_urban_plot_graph_v3b.py --city beijing --urban_root data/urban_network_datasets --out_root data --variant v3bsjg
.venv/bin/python tools/prepare_urban_plot_hetero_v2.py --city beijing --urban_root data/urban_network_datasets --out_root data
```

更多 Urban 系列与验证入口见 [tools/README.md](../tools/README.md)，分析报告见 [reports/README.md](../reports/README.md)。异构导出使用独立 schema 和 runner，不能直接作为下方同构稀疏格式输入。

## 派生图与历史数据

`*_sepick`、`*_sepick_reg` 是搜索选定权重后写出的派生图，`*_cv80` 是边交叉验证训练图；它们都不是自动下载的数据集。按[实验计划索引](../configs/breakthrough/README.md)中的 pick、上下文维度和划分参数重建，并保留原图。部分派生图与对应检查点仅保存在远端工作副本，详情见 [DATA_ASSETS.md](../DATA_ASSETS.md)。

Fraud、Entities、Bitcoin 及早期 Urban 转换器保留在[历史构建器目录](../archive/code/historical_builders/README.md)。`synth_mech_full_v1_*` 和 `synth_edgectrl_v1_*` 的节点特征已足以揭示标签，其高分不构成当前边语义有效性的证据。

## 自定义同构图

在 `<root_path>/<name>/` 放置以下 NumPy 文件，文件名前缀须与数据名完全一致：

```text
<name>_edge_index.npy   # 必需：[2, E]，整数索引范围 [0, N)，E > 0
<name>_feat.npy         # 必需：[N, F]
<name>_label.npy        # 必需：[N]，未知标签用 -1
<name>_edge_weight.npy  # 可选：[E]，缺省为 1
<name>_edge_attr.npy    # 可选：[E, D]，每行与 edge_index 的对应列对齐
<name>_meta.json        # 可选：通道说明、标签映射、来源与构建参数
```

```bash
.venv/bin/python tools/run_preset.py --preset b31_dualscalar_assign --dataset my_graph --root_path data --max_nums 12 --seed 0
```

`--max_nums` 指层次树父节点预算；自定义数据名不在 runner 的自动预算表中，须显式指定。热启动伪分区默认使用已知类别数，可直接运行 `main.py --warm_start_k` 显式设置。加载器也支持 `<name>_adj.npy` 稠密邻接格式；同一目录同时有稠密和稀疏文件时优先读取稠密格式。

`main.py` 默认只在已知标签上评价；若源数据将未知标签保存为非负类别，应在元数据中声明，例如 `{"label_mapping": {"unknown": 8}}`。全无标签的图不能提供 NMI/ARI 等外部评价，也不能直接沿用依赖类别数的热启动设置。少标签训练/选型须使用专门的实验入口，并与无监督结果分别解读。
