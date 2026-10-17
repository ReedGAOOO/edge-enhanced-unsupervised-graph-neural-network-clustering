# 项目工具索引

更新：2026-10-17。以下命令从仓库根目录用 `.venv/bin/python` 执行。安装见
[环境声明](../UV_ENVIRONMENTS.txt)，数据准备见 [data/README.md](../data/README.md)，
E1–E22 的完整依赖顺序和命令见 [实验计划](../configs/breakthrough/README.md)。可执行脚本的参数以各自 `--help` 为准。

## 训练、搜索与报告

| 工具 | 用途与输入 |
|---|---|
| [run_preset.py](run_preset.py) | 单个预设运行；默认 B31、2000 epochs、冷启动；`--dry_run` 预览命令 |
| [run_breakthrough.py](run_breakthrough.py) | 展开 JSON 训练计划，在当前机器的一张 GPU 上并发；支持标签筛选和 `--num_shards/--shards` 分片 |
| [run_classic_baselines.py](run_classic_baselines.py) | KMeans(X)、spectral(A)、Louvain(A)、SGC(A,X)+KMeans 对照；也提供正则化谱聚类实现 |
| [run_se_guided_baseline.py](run_se_guided_baseline.py) | 显式边通道、图层、端点上下文和乘积搜索；用二层结构熵选权重；`--spectral reg`、`--order2`、`--beam B --max_order R` |
| [write_se_reweighted_dataset.py](write_se_reweighted_dataset.py) | 将搜索 pick 写成含 `<name>_edge_weight.npy` 的数据副本，供 edge-free DSE 训练 |
| [se_select_decoder.py](se_select_decoder.py) | 从已有运行、数据和检查点恢复分区，用同一加权图上的二层 SE 比较 DSE / spectral / SGC；`--spectral reg` |
| [report_e2_mechanism.py](report_e2_mechanism.py) | 合并合成图运行账本，计算架构、描述性边、直接边和上下文边效果 |
| [report_e3_real.py](report_e3_real.py) | 合并真实图账本与经典对照，按 baseline 的 SE 损失选 warm/cold；`--out_dir` 独立输出报告 |
| [build_results_index.py](build_results_index.py) | 索引本地结果文件 |

```bash
# 搜索 → 写出图；后续训练命令和全部 picks 见 configs/breakthrough/README.md
.venv/bin/python tools/run_se_guided_baseline.py --root_path data \
  --datasets acm_multiplex_v2 --seeds 0 --spectral reg --workers 8 \
  --out_dir results/breakthrough_v1/search_acm
.venv/bin/python tools/write_se_reweighted_dataset.py --root_path data \
  --dataset acm_multiplex_v2 --pick=layer+PAP --out_name acm_sepick
```

搜索的候选表为 `<dataset>__s<seed>_candidates.csv`，汇总为 `summary.csv`；NMI/oracle 列只用于评估，
pick 按 SE 选择。含 `+`、`-` 或 `*` 的 pick 建议写成 `--pick='<pick>'`，避免参数或 shell 解析歧义。
预设与批量运行器均不自动选择模型族；`max_nums` 是层次预算。

## 候选选择与少量标签（E17–E22）

| 工具 | 协议与输出 |
|---|---|
| [make_edge_cv_split.py](make_edge_cv_split.py) | 保留约 20% 无向边，不产生新孤立点；写训练图和 `<name>_heldout.npz` |
| [select_partition_rules.py](select_partition_rules.py) | 读取 decisions JSON，比较 SE、medoid、edge-CV、two-view、gated-CV；写 `instances.csv` / `summary.csv` |
| [run_selection_validation.py](run_selection_validation.py) | E19 可恢复驱动：新合成图、搜索、重加权图、CV 图、两阶段训练、规则评分；会重写 E19 plans/spec |
| [few_label_selection.py](few_label_selection.py) | E20：在冻结候选中按随机少量标签上的 ARI 选择；缓存 `partitions/*.npz`，输出各预算 regret |
| [few_label_edge_semantics.py](few_label_edge_semantics.py) | E21 methods 0–2：SGC 特征分类、SE top-10 + 标签选择、带/不带伪标签的边分类器；输出逐 draw CSV |
| [few_label_route_selection.py](few_label_route_selection.py) | E22：标签集内部交叉验证选择结构路线或特征分类路线；写 `draws.csv` / `summary.csv` |

E18/E19 的模型候选需要 spec 中的 `stdout.txt` 和匹配检查点；仅有 tracked CSV 不能恢复分区。
E20 更改候选、代码或图后应换 `--out_dir` 或清理对应分区缓存。`--draws 200` 在 E20 是抽样次数；
E21/E22 的 `--draws 0,1,2` 是标签随机种子列表。

E21/E22 的 CPU 工具固定读取仓库 `data/`，没有 `--root_path`；其 jobs CSV 中的候选路径相对当前工作目录。
必须先生成 E15 的真实图/合成图候选和 E21 的 direct 候选。少标签 DSE（method 3）由
[e21_few_label_dse.json](../configs/breakthrough/e21_few_label_dse.json) 交给训练运行器执行。
E20 只用标签选已有分区，并在全部已知标签节点上衡量 regret；E21/E22 则在排除抽取标签后的其余已知标签节点上评估。

## 边诊断

| 工具 | 输入与解释 |
|---|---|
| [screen_edge_headroom.py](screen_edge_headroom.py) | 已准备图；监督同类边分类与图重加权上限诊断 |
| [check_se_alignment.py](check_se_alignment.py) | 已准备图；比较真实标签分区与 spectral/SGC 分区的 SE，跳过 `-1` 标签 |
| [check_spectral_stability.py](check_spectral_stability.py) | 已准备图；改变 eigensolver 起始向量，比较 plain / regularized spectral NMI 与连通分量 |
| [inspect_edge_factors.py](inspect_edge_factors.py) | `stdout.txt`、对应数据和检查点；检查学习到的标量头 |
| [diagnose_b45_message_gate.py](diagnose_b45_message_gate.py) | 固定检查点干预、嵌入/分配图交叉，或短预算配对训练 |
| [prepare_edge_intervention_dataset.py](prepare_edge_intervention_dataset.py) | 从已准备图生成边属性干预对照 |

```bash
.venv/bin/python tools/check_se_alignment.py --root_path data \
  --datasets cora,photo,dblp_magnn_author_v2,acm_multiplex_v2 \
  --out_csv results/breakthrough_v1/e10_se_alignment/se_alignment.csv

# 已有 B45 Photo 检查点：交叉嵌入图与分配图
.venv/bin/python tools/diagnose_b45_message_gate.py \
  --run_dir results/breakthrough_v1/e3_real_dual/runs/b45_warm__photo__s0 \
  --graph_cross --out results/breakthrough_v1/b45_gate_diagnosis/photo_graph_cross.json

# 从共同初始化开始短预算配对训练；固定 SGC 伪分区和最终 epoch
.venv/bin/python tools/diagnose_b45_message_gate.py \
  --run_dir results/breakthrough_v1/e3_real_dual/runs/b45_warm__photo__s0 \
  --train --seed 0 --warm_epochs 200 --epochs 200 \
  --modes original off unit_gate no_renorm no_gate_dropout safe_isolates \
  --out results/breakthrough_v1/b45_gate_diagnosis/photo_paired_s0.json
```

门控工具无 `--train` / `--graph_cross` 时执行固定的检查点干预列表，必须有检查点。
`--modes` 只配置训练探针，`--cpu` 强制 CPU。重新训练需要源运行的 `stdout.txt` 配置，
不载入已训练权重；输出进度 JSON 和 `.pseudo.npy`。短预算结果用于机制诊断。
已记录的 `safe_isolates` 来自相同种子/初始化协议下的独立配对运行，其他干预不能视为完整基准。

## 数据构建

| 数据族 | 当前入口 |
|---|---|
| 合成机制基准 | [prepare_synth_ctx_datasets.py](prepare_synth_ctx_datasets.py)；当前参数为 `--feature_sep 0.35 --homophily 0.45,0.55`，XOR 加 `--prefix synth_ctxxor_v1 --role_mode xor2` |
| DBLP author | [prepare_pyg_dblp_magnn_dataset.py](prepare_pyg_dblp_magnn_dataset.py) `--mode v2` |
| ACM / IMDB meta-path union | [prepare_pyg_multiplex_dataset.py](prepare_pyg_multiplex_dataset.py)；ACM v2 使用 `--psp_sample_k 20` |
| Urban V3 / V3b | `prepare_urban_plot_graph_v3.py`、`prepare_urban_plot_graph_v3b.py` 和相应 `run_*_series.py` |
| Urban typed graphs | `prepare_urban_plot_hetero_v1.py`、`prepare_urban_plot_hetero_v2.py`、`validate_urban_plot_hetero_v1.py`、`smoke_test_urban_plot_hetero_v2.py` |
| Urban semantic / native | `prepare_urban_plot_graph_v4_semantic.py`、`prepare_urban_plot_graph_v5_native.py` |
| 历史合成复现 | [prepare_mechanism_synth_datasets.py](prepare_mechanism_synth_datasets.py)；不能作为当前边信息证据 |

旧 Entities/Fraud/Bitcoin 和 Urban V1/V2 构建器见
[archive/code/historical_builders/](../archive/code/historical_builders/README.md)。

## Urban 分析与制图

`run_urban_data_paper_pipeline.py` 与 `run_urban_*_experiments.py` 支持 Urban 数据研究；
`analyze_raw_urban_city.py`、`analyze_urban_raw_all_cities.py`、`run_raw_urban_analysis_all.py` 及城市/因子诊断检查原始数据和图构建。
`plot_*.R`、`render_*.R`、`visualize_*.py`、`export_urban_*.py` 与 `build_folium_cluster_map.py` 生成图表、报告输入和地图。
复用旧 Urban 模型比较前先读 [reports/README.md](../reports/README.md)。R 与地理空间工具可能需要根 Python lock 之外的依赖，
以具体脚本的 imports 和报告说明为准。

`run_urban_factor_experiment_packages.py` 仍硬编码已移除的 `gnn` conda Python 路径，并通过 `conda run -n r` 调用 R。
其 `--dry_run true` 不是全局预览：仍会写清单/账本并尝试其他子任务和 R。先用 `--help` 检查入口；
执行前需调整解释器配置，或用当前环境逐个运行所需子脚本，具体限制见报告索引。
