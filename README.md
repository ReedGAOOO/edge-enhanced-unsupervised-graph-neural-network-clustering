# Edge Enhanced Unsupervised Graph Neural Network Clustering

基于 DSE / LSEnet（Lorentz structural entropy）的图聚类研究项目，研究边属性在什么条件下有用，以及模型是否真正利用了边语义。

文档更新：**2026-10-17**。当前研究分支为 `breakthrough-2026-10`，实验记录覆盖 **E1–E22**。本次日期表示文档整理时间，结果来自已提交的实验记录。

## 当前结论

目前的研究路线是：**用结构熵（SE）搜索显式边权，生成候选划分；无标签时报告各方法结果，有少量标签时选择划分或选择结构/特征路线。**

- **无标签候选生成**：`run_se_guided_baseline.py --spectral reg` 搜索边通道、端点上下文及其乘积，再使用谱聚类、SGC+KMeans，或在固定重加权图上训练无边头 DSE。ACM 的 DSE NMI 从 0.464 提高到 0.538（E14/E16）；这不意味着每个数据集都会改善。
- **学习型边头保留为消融**：一阶上下文上，显式搜索可以匹配学习型边头；XOR 二阶上下文上，穷举搜索能恢复信号，现有边头与 beam 搜索缺少低阶信号（E12–E15）。B31 是便利脚本的默认预设，不能据此将它视为当前主方法。
- **无标签选模仍有局限**：在 E18/E19 共有 edge-CV 的 24 个决策上，边交叉验证的平均 NMI regret 为 0.026，直接按 SE 选择为 0.043。Photo / Computers 的可压缩图结构与类别标签并不一致，edge-CV 也无法消除这种偏差。
- **少标签选择有效，但属于半监督使用**：同一组 24 个决策中，20 / 50 个随机标签选择候选的 NMI regret 为 0.029 / 0.015（E20）。E22 再在标签内部做五折交叉验证，选择结构路线或特征分类路线；50 / 100 个标签时平均 NMI 为 0.696 / 0.730。
- **Urban 独立作为数据构建研究线**：已知 land-use 标签与图社区不一致；旧版短训练、把 `unknown` 当一类的分数不能作为当前聚类排名。

完整推理、正负结果与消融见 [E1–E22 实验记录](reading/BREAKTHROUGH_E1_E22.md)，对应数据见 [证据索引](configs/breakthrough/evidence/README.md)。

## 项目导航

| 路径 | 内容与入口 |
|---|---|
| [main.py](main.py)、[exp.py](exp.py)、[data.py](data.py) | 训练参数、实验生命周期、数据加载与边属性处理 |
| [modules/](modules/)、[manifold/](manifold/)、[utils/](utils/) | 模型、Lorentz 几何、warm start、解码与评估 |
| [configs/](configs/README.md) | 三个活动预设及参数边界 |
| [configs/breakthrough/](configs/breakthrough/README.md) | E1–E22 计划、先决条件和按依赖顺序排列的复现命令 |
| [tools/](tools/README.md) | 搜索、批量训练、选择规则、诊断、数据构建与报告工具 |
| [data/](data/README.md) | 数据获取、构建及自定义图格式；数据实体不随 Git 分发 |
| [DATASETS.md](DATASETS.md)、[DATA_ASSETS.md](DATA_ASSETS.md) | 数据集定义与用途、原工作站的本地资产和备份记录 |
| [reading/](reading/README.md)、[reference/](reference/README.md) | 实验详述、实现笔记、历史专家包及上游参考 |
| [reports/](reports/README.md) | Urban 数据构建、城市差异与有效性报告 |
| `results/`、`checkpoints/` | Git 忽略的本地运行输出和模型权重 |
| [archive/](archive/README.md) | 历史代码、配置、文档与工作区快照 |

## 环境安装

以下命令均在仓库根目录执行。已测试环境为 Linux / WSL、Python 3.11、PyTorch 2.8.0 + CUDA 12.9，安装方式见 [UV_ENVIRONMENTS.txt](UV_ENVIRONMENTS.txt)。

```bash
git clone https://github.com/ReedGAOOO/edge-enhanced-unsupervised-graph-neural-network-clustering.git
cd edge-enhanced-unsupervised-graph-neural-network-clustering
git switch breakthrough-2026-10

uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python --torch-backend cu129 requirements.uv.lock.txt

.venv/bin/python -c "import torch, torch_geometric, torch_scatter; print(torch.__version__, torch_geometric.__version__, torch_scatter.__version__); print('CUDA:', torch.cuda.is_available())"
```

[requirements.txt](requirements.txt) 是宽松依赖清单，[requirements.uv.lock.txt](requirements.uv.lock.txt) 固定已测试版本。锁文件使用与 torch 2.8 匹配的 `torch-scatter` CUDA wheel，不能只升级其中一个包；该锁面向 Linux/WSL CUDA 环境，不是通用 CPU 安装锁。`reference/urbanity` 有独立环境，根项目实验不需要先安装它。

## 快速开始

先预览参数，再运行 Cora 控制实验。训练加载器会在缺失时下载 Cora；需要能访问数据源。

```bash
.venv/bin/python tools/run_preset.py --list_presets
.venv/bin/python tools/run_preset.py --preset baseline_v1 --dataset cora --seed 0 --dry_run

# 无学习型边头的 DSE 控制组：2000 epochs，hid_dim 512
.venv/bin/python tools/run_preset.py \
  --preset baseline_v1 --dataset cora --seed 0 --gpu 0

# B31 消融；省略 --preset 时也会使用 B31
.venv/bin/python tools/run_preset.py \
  --preset b31_dualscalar_assign --dataset cora --seed 0 --gpu 0
```

`run_preset.py` 默认 cold start，不会自动训练 warm/cold 两组或自动选择模型。严格比较请使用 [实验计划](configs/breakthrough/README.md)：各组使用相同训练预算，并按相应协议选择起点。

`main.py` 的默认参数与预设脚本不同。CPU 执行可在兼容环境中使用 `main.py --no_gpu`；`--gpu 0` 表示选择第 0 张 GPU。便利脚本没有 `--no_gpu` 参数，可从 `--dry_run` 输出取得完整命令再改为 CPU 参数。

## 主线示例：搜索边权，再训练 DSE

先构建实验所用的 ACM v2（PSP 每个节点采样 20 个邻居），再进行 regularized spectral 搜索。此流程需要下载上游数据。

```bash
.venv/bin/python tools/prepare_pyg_multiplex_dataset.py \
  --root_path data --dataset acm --name acm_multiplex_v2 --psp_sample_k 20

.venv/bin/python tools/run_se_guided_baseline.py \
  --root_path data --datasets acm_multiplex_v2 --seeds 0 --spectral reg --workers 8 \
  --out_dir results/breakthrough_v1/search_acm
```

查看生成的 `summary.csv`，取 `ctx_pick` 的边权名称（去掉 `/spectral` 或 `/sgc` 后缀）。以下使用已记录 ACM 实验的 `layer+PAP`；新实验应使用自己搜索得到的选择。

```bash
.venv/bin/python tools/write_se_reweighted_dataset.py \
  --root_path data --dataset acm_multiplex_v2 \
  --pick=layer+PAP --out_name acm_sepick

.venv/bin/python tools/run_preset.py \
  --preset baseline_v1 --dataset acm_sepick --root_path data \
  --max_nums 6 --epochs 1000 --warm_start auto --seed 0 --gpu 0
```

新数据集的预设运行需要显式指定 `--max_nums`，它是层次结构的父节点预算，不保证最终簇数。上下文边权必须沿用搜索时的 `--ctx_dim`；XOR 示例需要二阶穷举，详见 [E14–E16 复现](configs/breakthrough/README.md)。上述单次训练用于说明流程，完整结论来自多种子、warm/cold 起点及解码比较。

搜索产生候选明细和 `summary.csv`；训练结果写入 `results/<version>/`，检查点写入 `checkpoints/<save_path>`。便利脚本的默认版本路径为 `mainline_evidence/raw_runs/manual/<preset>_<dataset>_s<seed>`；可用 `--version` 分开保存实验。批量计划写入 `results/breakthrough_v1/<plan>/`，包含 `runs.csv`、`summary.csv` 和各运行的 `stdout.txt`、metrics JSON。

## 数据准备

Git clone 包含代码、配置、文档和小型证据表，不包含完整数据、训练日志或模型检查点。原工作站的 D:/F: 位置只是本地资产记录，在其他机器上可直接创建普通 `data/` 目录。

| 数据族 | 获取方式与作用 |
|---|---|
| Cora / Citeseer / PubMed / Computers / Photo | 训练加载器自动下载；拓扑/特征控制组 |
| `dblp_magnn_author_v2` | `prepare_pyg_dblp_magnn_dataset.py --mode v2`；原生多通道 author 图 |
| `acm_multiplex_v2` | `prepare_pyg_multiplex_dataset.py --psp_sample_k 20`；meta-path union 图 |
| `entities_aifb` | 历史 Entities 转换器生成；仅在已知标签上评估 |
| `synth_ctx_v1_*` / `synth_ctxxor_v1_*` | 当前一阶 / XOR 二阶边语义机制基准 |
| Urban V3/V3b、hetero、v5 | 需要另外提供原始城市数据；用于数据构建研究 |

```bash
.venv/bin/python tools/prepare_synth_ctx_datasets.py \
  --root_path data --feature_sep 0.35 --homophily 0.45,0.55

.venv/bin/python tools/prepare_synth_ctx_datasets.py \
  --root_path data --prefix synth_ctxxor_v1 --role_mode xor2 \
  --feature_sep 0.35 --homophily 0.45,0.55 \
  --modes none,contextual --perm_modes contextual
```

完整命令和自定义 `[2, E] edge_index` / `[N, F] feat` / `[N] label` 稀疏图格式见 [data/README.md](data/README.md)。`synth_mech_full_v1_*` / `synth_edgectrl_v1_*` 的节点特征直接揭示类别，保留作历史检查，不作为当前边语义证据。

## 实验阶段与复现顺序

E1–E22 是研究阶段编号，部分阶段使用诊断脚本，并非每一阶段都有同名 JSON。详细命令与先决条件统一在 [configs/breakthrough/README.md](configs/breakthrough/README.md)。

| 阶段 | 回答的问题 | 主要入口 |
|---|---|---|
| E1–E4 | backbone 是否训练充分；边属性是否有效；B45 为何坍缩 | backbone、synthetic、real dual-start、collapse 计划 |
| E5–E10 | 真实数据边语义 headroom、上下文与固定 SE 目标 | headroom / alignment 工具，B31 与 fixed-ruler 计划 |
| E11–E13 | 安全 message gate；显式搜索与 XOR 上下文 | safe-gate / XOR 计划，SE-guided search |
| E14–E16 | 固定搜索边权后训练；谱聚类稳定性 | reweighted writer、search-then-train 计划、decoder 工具 |
| E17–E19 | 无标签选模及新决策上的验证 | edge-CV split / retrain、selection rules、validation driver |
| E20 | 少量标签选择已有候选 | `few_label_selection.py`，依赖 E18/E19 输出与检查点 |
| E21 | 少标签条件下的边语义和监督 DSE | edge methods、few-label DSE 计划；先生成 jobs CSV 引用的搜索结果 |
| E22 | 在结构与特征两条路线之间选择 | `few_label_route_selection.py`，复用 E21 搜索候选 |

复现时先完成数据构建，再完成搜索及派生图，再进行训练与候选选择。E20 需要 E19 完成；E21/E22 的 jobs CSV 引用了 E15 real/synthetic 搜索和 E21 direct 搜索，不能从干净 clone 直接跳到最终命令。E21 的 CPU 方法工具和 E22 路线工具固定读取仓库 `data/`，不支持 `--root_path`，候选 CSV 路径和数据集大小写也需与 jobs 文件一致。

批量运行前用 `run_breakthrough.py --dry_run` 检查命令。完成状态由 metrics 文件判断；修改代码或参数后使用新的计划名，或明确使用 `--force` 重跑。并发取决于显存与图规模，E19 的 PubMed / Computers 在 24 GB GPU 上使用 `--max_parallel 2`；不要把小图并发直接用于大图。

历史 EP1/EP2 计划依赖 `results/breakthrough_v1/_code/` 中未提交的专家工作区补丁副本，干净 clone 无法完整复现这部分。根目录 baseline / B31 / B45 代码、计划与小型结果证据已跟踪；记录了结果不等于分发了全部复现资产。

## 评估口径与可核查结果

默认 `known_only_eval` 排除 `-1` 标签及元数据声明的 `unknown`。未标注节点可以参与无监督图计算，但不参加已知类别指标。benchmark 的已知类别数用于若干经典聚类器和默认 warm-start 簇数；“label-free”表示未使用类别身份进行训练或候选选择，并不表示完全未知类别数。

| 设置 | 标签如何使用 | 评估节点 |
|---|---|---|
| 无监督训练 / SE 搜索 | 使用结构、特征和边属性；标签身份用于报告指标 | 所有已知标签节点 |
| E20 候选选择 | 随机抽取 m 个标签，以样本内 ARI 选择已有划分 | 所有已知标签节点，**包含这 m 个节点** |
| E21 少标签方法 | m 个标签用于选择候选、训练节点/边分类器或监督 DSE | 其余已知标签节点，排除 m 个监督节点 |
| E22 路线选择 | m 个标签内部五折 CV，选择路线后用全部 m 个标签重拟合 | 其余已知标签节点，排除 m 个监督节点 |

常规无监督训练按最小训练损失保存检查点；周期评估的 best-NMI / oracle / headroom 是诊断结果，不能当作无标签可实现的选择规则。E21 的监督 DSE 使用包含监督项的训练目标。历史阶段的起点选择规则不同，详见实验记录；E6 起比较组沿用 baseline 的起点选择。

E22 汇总如下，NMI 为每个数据集与标签抽样的均值，regret 相对同次抽样中更好的那条路线，**不是所有可能方法的最优值**。

| 标签预算 | 参与图数 / 抽样数 | 结构路线 | 特征路线 | CV 选择路线 | NMI regret |
|---|---|---|---|---|---|
| 10 | 11 / 55 | 0.6480 | 0.2420 | 0.6344 | 0.0187 |
| 20 | 11 / 55 | 0.6723 | 0.3219 | 0.6688 | 0.0165 |
| 50 | 11 / 55 | 0.6732 | 0.5019 | 0.6960 | 0.0037 |
| 100 | 11 / 55 | 0.7032 | 0.5508 | 0.7299 | 0.0022 |
| 200 | 10 / 50 | 0.7240 | 0.6169 | 0.7606 | 0.0024 |

数据来自 [route_selection_summary.csv](configs/breakthrough/evidence/route_selection_summary.csv) 和 [逐次抽样结果](configs/breakthrough/evidence/route_selection_draws.csv)。每图每预算五次抽样；AIFB 只有 176 个已知标签，200 预算被跳过，因此该行与前几行的图集合不同。10–20 标签时 CV 噪声较大，结构路线单独使用的均值略高。

E20 的 52 个候选决策、E21 各方法与 DSE 的对照、E14/E16 原图/重加权图结果及失败案例，均见 [详细实验记录](reading/BREAKTHROUGH_E1_E22.md) 和 [证据表说明](configs/breakthrough/evidence/README.md)。E21 CPU 方法原始表有三次标签抽样，详述中的 DSE 对照表统一取抽样 0–1；不要将不同抽样范围的均值混为同一结果。

## 研究边界与后续问题

- SE 优化目标与外部类别标签可能不一致；无标签结果应并列报告候选方法，少标签结果须说明监督预算和评估范围。
- 二阶穷举搜索会随上下文维数增长，候选池的改变也会改变 SE 选中的边权；beam 在 XOR 上的失败仍未解决。
- DSE 在 AIFB 等弱 backbone / 目标不对齐场景未优于经典方法；ACM 的边权收益不能推广为普遍优势。
- 随机、均匀抽取已知标签是现有少标签实验设定；非均匀标注、主动选择与极小预算下的鲁棒性仍待验证。
- 旧版 Urban 排名、短训练结果和专家包结论按历史背景保存，当前结论以本页及 E1–E22 记录为准。

## 参考与许可

上游实现：[DSE_clustering](https://github.com/RiemannGraph/DSE_clustering)。仓库包含 [DSE 论文副本](reference/2504.09970v2.pdf)、[LSEnet ICML 2024 论文副本](reference/sun24g.pdf) 和 [原实现说明](reading/EXPLAIN_DSE_original_en.md)；参考代码与来源边界见 [reference/README.md](reference/README.md)。

项目许可见 [LICENSE](LICENSE)。第三方参考代码、数据源及各自许可须分别遵循其原始声明。
