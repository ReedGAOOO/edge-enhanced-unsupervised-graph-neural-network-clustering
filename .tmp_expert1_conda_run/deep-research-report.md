# Edge Enhanced Unsupervised Graph Neural Network Clustering 的深度研究与论文式 README 重构草案

## 执行摘要

该项目围绕“无监督/自监督图聚类”构建了一条明确的工程—研究主线：在 **LSEnet / DSI（可微结构信息/结构熵）** 的框架上，引入 **ECHF（Edge-Calibrated Hierarchical Fusion）** 家族作为当前主线，重点解决“边信息（尤其是异质图/带属性边）在聚类中的可用性与稳定注入”问题，并通过一套分支对照脚本与阶段性汇总报告组织实验复现。仓库的 README 将 **B15_ECHF_main（分支主线）** 与 **g15_echf_main（默认可运行 preset）** 作为当前主推配置，强调三点增量：**V12 校准残差融合（assignment-score 级别）**、**Path‑B 层级边属性传播**、以及 **图级自适应门控 graph_alpha**（用于抑制异质图上“过强边注入”的风险）。这些设计在仓库内部报告中被解释为“在保持结构熵优化内核不变的同时，让 edge features 可用且不易失控”。（仓库 README 与 `exp/` 报告提供了该叙述与数值汇总；见后文链接）

从学术定位看，本项目最接近 **“结构信息/结构熵驱动的深度图聚类”** 这一新兴分支：其上游有 **LSEnet（ICML 2024）**，提出可微结构信息 DSI 并在 Lorentz 双曲空间中学习分割树；紧随其后有 **DeSE（KDD 2025）** 强调“深度结构熵”与结构学习层；而用户指定的 **arXiv:2504.09970** 当前版本标题为 **ASIL（Augmented Structural Information Learning）**，并显示“最后修订于 2026‑02‑02，IEEE TPAMI 接收”。citeturn7view0turn14view0turn14view1  
因此，本项目可以被理解为：在“结构熵/分割树范式”上，进一步把 **边的权重/属性/置信** 纳入 assignment 和分层传播路径，形成一条更接近真实异质网络（仓库称 urban graphs）的工程化研究支线。

但若以“论文可发表”为标准，当前仍存在显著缺口：  
关键缺口集中在 **（a）对真实 edge attributes 的系统验证不足**（因为许多常用基准并无边属性，仓库更多通过“从节点特征/度构造 generic edge features”来训练 V6/V7/V8/V12）；**（b）缺少与更广泛强基线（特别是近年的图聚类/对比学习聚类/结构学习聚类）在统一协议下的对照**；**（c）对 cluster-number 机制（未知 K vs 上界 K，空簇/小簇修复）缺少单独消融与理论讨论**；**（d）复杂度与可扩展性只在代码层面做了 edge‑KNN 近似，但缺少系统的效率实验与误差分析**。这些都需要通过“缺失实验清单 + 复现实验计划”补齐（后文给出可执行的补实验矩阵、默认超参建议与时间评估）。

## 仓库结构与实现细节概览

仓库以“可运行的实验平台”组织，核心入口与研究材料在根目录 + `modules/` + `tools/` + `configs/` + `exp/` 五部分完成闭环：

- **运行入口**：`main.py`（argparse 参数齐全，支持 edge 变体、gamma schedule、Path‑A/Path‑B 开关、known-only eval、BF16 AMP 等）。
- **训练编排**：`exp.py`（训练循环、早停、gamma 线性 schedule、以 *min train loss* 选模型、并可计算结构指标 modularity / conductance 等）。
- **数据与边构造**：`data.py`（PyG 内置数据集自动下载；自定义 `*_adj.npy/*_feat.npy/*_label.npy`；并实现多种 edge 预加权与 edge_attr 标准化/拼接、Path‑A 权重融合等）。
- **模型与损失**：`modules/dsi.py`（DSI/结构熵损失实现 + KNN augmentation；并在大图时提供 edge‑KNN 模式避免 O(N²)）；`modules/model.py`（LSENet 分层结构与 Path‑B 边属性上卷积/粗化）；`modules/layers.py`（LorentzAssignment：V5/V6/V7/V8/V12 的 edge‑fusion 发生在 assignment-score）。
- **实验工具链**：`tools/run_preset.py`（preset 驱动；根据数据集自动填 max_nums/epochs/hid_dim 等）；`tools/run_benchmark_branch_compare.py` 与 `tools/run_urban_branch_compare.py`（统一对照实验与汇总 CSV）。
- **实验中心**：`exp/`（版本命名、阶段结果快照、主线报告；README 指向 `MAINLINE_ECHF_B15_G15_2026-02-19.md` 等）。

依赖清单在 `requirements.txt`，包含 PyTorch、PyG、geoopt（Lorentz 流形）、以及评估所需 sklearn/networkx/munkres 等。

为便于审阅关键实现点，以下给出 **按提交 f31de462（2026‑02‑19）固定的文件锚点链接**（行号范围为“宽范围覆盖”，用于快速跳转与全文检索；如需精确定位可在页面内搜索函数名）：

```text
README（主线说明与证据快照）
https://github.com/ReedGAOOO/edge-enhanced-unsupervised-graph-neural-network-clustering/blob/f31de46273cdd0c7809779d01d4369a32c390a74/README.md#L1-L220

训练入口与参数（edge_variant / Path-A / Path-B / gamma schedule）
https://github.com/ReedGAOOO/edge-enhanced-unsupervised-graph-neural-network-clustering/blob/f31de46273cdd0c7809779d01d4369a32c390a74/main.py#L1-L220

训练循环（min train loss 选模型；结构指标 modularity/conductance；trial 稳定性）
https://github.com/ReedGAOOO/edge-enhanced-unsupervised-graph-neural-network-clustering/blob/f31de46273cdd0c7809779d01d4369a32c390a74/exp.py#L1-L420

数据加载与边构造（V1-V12 的 edge_weight 形成；edge_attr 标准化；Path-A 权重融合）
https://github.com/ReedGAOOO/edge-enhanced-unsupervised-graph-neural-network-clustering/blob/f31de46273cdd0c7809779d01d4369a32c390a74/data.py#L1-L320

DSI/结构熵损失（KNN augmentation；大图 edge-KNN 模式；SI loss 计算）
https://github.com/ReedGAOOO/edge-enhanced-unsupervised-graph-neural-network-clustering/blob/f31de46273cdd0c7809779d01d4369a32c390a74/modules/dsi.py#L1-L260

核心创新之一：LorentzAssignment 的边融合（V5/V6/V7/V8/V12）
https://github.com/ReedGAOOO/edge-enhanced-unsupervised-graph-neural-network-clustering/blob/f31de46273cdd0c7809779d01d4369a32c390a74/modules/layers.py#L1-L360

核心创新之二：LSENet 分层与 Path-B 层级 edge_attr 粗化传播
https://github.com/ReedGAOOO/edge-enhanced-unsupervised-graph-neural-network-clustering/blob/f31de46273cdd0c7809779d01d4369a32c390a74/modules/model.py#L1-L340

默认主线 preset（g15_echf_main）
https://github.com/ReedGAOOO/edge-enhanced-unsupervised-graph-neural-network-clustering/blob/f31de46273cdd0c7809779d01d4369a32c390a74/configs/presets/g15_echf_main.json#L1-L120

preset 运行器（数据集到默认超参映射；别名兼容）
https://github.com/ReedGAOOO/edge-enhanced-unsupervised-graph-neural-network-clustering/blob/f31de46273cdd0c7809779d01d4369a32c390a74/tools/run_preset.py#L1-L260
```

## 研究问题与方法定位

### 问题陈述

给定一个图 \(G=(V,E)\)，节点特征矩阵 \(X\in \mathbb{R}^{|V|\times F}\)，以及（可选的）边权/边属性，本项目目标是学习节点的聚类划分 \(y^{pred}\)（或更一般地：学习一棵层级划分树），在 **不使用真实标签监督** 的前提下，使划分与图的结构与语义一致，并在下游用 NMI/ARI/ACC 等指标对齐评估（标签仅用于评估，而非训练）。

在学术谱系上，本项目显式沿用 LSEnet/DSI 的核心动机：经典深度图聚类往往依赖 **预设聚类数 \(K\)** 或“先嵌入再 k-means”的两阶段流程；而结构信息/结构熵视角试图把“划分树/社区结构”直接作为可优化对象，从而弱化对固定 \(K\) 的依赖。LSEnet 的摘要明确提出：构建可微结构信息 DSI，最小化 DSI 可得到“最优分割树”，并在 Lorentz 双曲空间里实现神经网络化的树学习，从而支持“未知聚类数”的聚类目标。citeturn14view0turn13search1

本仓库进一步把“边的信息”前移到聚类决策路径中，试图解决两类现实痛点：  
其一，许多真实网络（仓库称 urban/hetero graphs）边的异质性较强，若直接把结构信号以固定强度注入，容易造成错误聚合；其二，若图存在边属性（关系类型、强度、时间、空间交互等），仅用邻接矩阵会浪费信息，但“盲目使用边属性”也容易引入噪声与偏差。因此仓库主线采用 **“校准（calibrated）+ 门控（adaptive）+ 层级传播（hierarchical）”** 的组合策略。

### 目标与关键假设（明确标注未给出的细节）

已在代码与 README 中明确/可推断的假设：

- **训练阶段不使用标签**：训练目标来自 `modules/dsi.py` 的 `se_loss`（结构熵/结构信息形式）。  
- **聚类数机制为“上界/层级容量”**：`main.py` 将 `max_nums` 作为一个列表输入（示例 `[50,10]`），决定LSENet层级各层的“最大父节点数”；最终评估时通过 `fix_cluster_results` 处理“过小簇”（小于 `epsInt`）把其节点重分配到“大簇父节点”上，从而允许出现空簇或减少有效簇数（但这不等同于严谨的“自动选择 K”）。  
- **边属性可缺省**：若数据集无 `edge_attr`，代码会构造 generic edge features（由节点特征相似度、度/度差等组成），并标准化用于 V6/V7/V8/V12（这意味着“edge-enhanced”在许多基准上依赖“派生边特征”而非真实边属性）。

仓库未明确给出的关键实验设定（需在论文写作中标注为 *未声明*，并给出默认建议）：

- 数据集的官方/通行 split、是否移除自环、多图训练还是单图训练：**未声明**（从实现看多数是单图全量训练）。  
- 是否进行特征归一化、adj 对称化等细节：**部分隐含在代码中，但未在 README 以可复现表格给出**。  
- 城市（urban）数据集的来源、规模、edge_attr 的真实语义：**未在公共 README 给出**（但工具脚本内列出了数据集名列表）。

## 模型与算法剖析

### 数据管线与边信息构造

本项目把“边的处理”分成三层：

第一层：**预加权（pre-weighting）**——在 `data.py` 中针对不同 `edge_variant` 形成用于消息传递/结构熵计算的稀疏邻接（至少两份：`adj_msg` 与 `adj_si`）。其核心思想是：在原始边上构造结构权重（度相关）、特征相似权重（余弦与温度 `edge_feat_temp`）、以及二者的混合（`edge_hybrid_alpha`），并可通过 `edge_input_prior_alpha` 混入数据集自带权重（若存在）。这一阶段对应 README 中的 V1–V4/V5“hybrid pre-weight”。（实现细节见 `data.py` 与 `main.py` 参数说明链接）

第二层：**边属性编码（edge_attr）**——若数据集提供 `edge_attr`，可选择 `append_generic_edge_attr` 把 generic edge features 追加到给定 edge_attr；若未提供，则直接构造 generic edge features。之后标准化，并与稀疏边索引对齐、去重 coalesce。该 `edge_attr` 主要用于 V6/V7/V8/V12 的 assignment-score 融合与 Path‑B 的层级传播。

第三层：**Path‑A vs Path‑B**——二者可同时启用但主线默认强调 Path‑B：  
- **Path‑A（edge_attr_weight_blend）**：把 edge_attr 映射成标量权重并混入结构熵图（默认 `apply_to=si_only`），直觉是“让结构熵看到边属性”。但 README 明确指出主线默认 `edge_attr_weight_blend=0`，认为把富边特征压缩为单标量会损失语义。  
- **Path‑B（edge_attr_hierarchical）**：把 edge_attr 随着层级聚合向上“粗化传播”，使更高层的聚合仍能感知边语义。这与“层级社区/树结构”更相容，因此成为 ECHF 主线。

### DSI/结构熵损失与训练图构造

训练损失在 `modules/dsi.py::se_loss`，核心由两部分组成：

一是 **KNN augmentation**：  
- 在小图或 `knn_mode=dense` 时，计算叶子嵌入 \(z\) 的两两相似（通过 Lorentz 内积/距离的变形），对每个节点取 top‑K 形成稀疏 `adj_aug`，并经过 softmax 得到概率图。  
- 在大图或 `knn_mode=edge/auto` 时，为避免 O(N²)，只在“已有边 + 自环”集合上打分，并对每个源节点做 top‑K 截断再 scatter_softmax，形成近似的 `adj_aug`。这是一种“候选边集合受限”的 KNN 增强策略，牺牲了全图近邻搜索的完整性以换取可扩展性。

二是 **结构熵图组合与 SI loss**：  
`adj_train = adj_base + alpha * adj_aug`（同时对 message 图与 SI 图各自组合），并把该训练图送入编码器得到层级 assignment 矩阵。随后通过反复 `assᵀ A ass` 得到各层邻接，再按层累加结构熵项，最终得到归一化的负结构熵损失。该范式与 LSEnet/DSI 的摘要描述一致：通过神经 assignment 构建分割树并以结构信息为目标。citeturn14view0turn13search1

### ECHF 的核心：assignment-score 级别的边校准融合

仓库 README 把主线概括为 “calibrated edge fusion + hierarchical edge propagation”，其中最关键的“edge fusion”发生在 `modules/layers.py::LorentzAssignment.forward`。其计算流程可概括为：

- 基础 attention-score：对每条边 \((u,v)\) 计算 \(score_{uv} = -d(q_u,k_v)\)，再对同一源节点的外连边做 softmax 得到注意力权重，并用其聚合 assignment logits。  
- 在此基础上叠加不同的边融合项（V5/V6/V7/V8/V12），改变 softmax 前的 logits，从而直接影响聚类树的 parent assignment。

ECHF 主线强调的 **V12** 可以视为“V5 的稳定结构项 + 边属性残差校准项”：

- **V5（结构项 trunk）**：将结构边权 \(w_{uv}\) 的对数 `log(w)` 作为先验信号注入，同时通过 `reliability`（基于权重分布的 sigmoid 标准化 + 可选分位数阈值）抑制低置信边；并可引入 **graph_alpha**（图级门控）控制全局注入强度，以适配异质图。  
- **V6/V7（边属性门控）**：使用一个小型 MLP 编码 edge_attr，输出 bias 与 gate，通过 gate 作为“边属性可靠性”，并把 bias（标准化后）作为注入信号。V7 增加与结构 log 的“对齐残差”项。  
- **V8（结构—属性校准混合）**：同时计算结构可靠性与属性可靠性，并用二者 agreement（标准化的 edge_log 与 attr_bias 的相关）来生成 mix_beta，实现结构项与属性项的可调混合。  
- **V12（主线）**：保留 V5 的结构 trunk，并把属性残差作为在 trunk 上的“校准补偿”，使模型在边属性噪声大时仍可退化为较稳健的结构注入，在边属性可信且与结构一致时再提升属性权重。

这一设计与“结构信息驱动 + 边属性增强 + 校准防过拟合”的研究直觉一致，也与用户指定的结构信息路线（LSEnet、DeSE、以及 2504.09970/ASIL）在问题动机上高度相容：这些工作共同强调“结构信息/结构熵在深图聚类里是关键但难以可微/难以融入属性”，并持续在可微化、复杂度与属性融合上推进。citeturn14view0turn14view1turn7view0

### Path‑B：层级 edge_attr 传播的意义

在树/层级聚类中，边属性的“语义”往往不仅作用于叶子层：当若干节点被合并为超点（cluster），原始边属性需要被汇聚到超边（cluster‑to‑cluster）的层级上，才能让更高层决策继续感知关系语义。`modules/model.py` 中实现了 edge_attr 的“hard coarsen”策略：在每次 pooling（assignment）后，根据 cluster‑to‑cluster 的 adjacency 权重，对落在同一对 cluster 的叶边属性进行加权汇总，从而形成下一层的 edge_attr，并继续用于更高层的 LorentzAssignment（当启用 `edge_attr_hierarchical`）。这与 ASIL/LSEnet 类型方法“学习分割树并逐层上卷积”的结构天然匹配：树的每一层都可被视为一个新的图，需要可继承的关系描述。citeturn14view0turn7view0

### 训练与评估协议

训练循环在 `exp.py`，最重要的两点是：

- **模型选择规则**：主默认以 *min train loss* 选择 checkpoint；对照脚本（如 benchmark compare）甚至把 `eval_freq=epochs` 以避免“用标签评估做中途挑 epoch”导致的隐性监督。此点对于无监督聚类论文非常关键，应在论文协议中显式写出并作为可复现性承诺。  
- **评估指标扩展**：除 NMI/ARI/ACC（标签对齐）外，还计算 modularity 与 conductance（结构质量）以及 trial 稳定性（多次 cluster trial 的 pairwise NMI）。这与 ASIL 摘要中强调的“conductance 上界/改进”叙述更容易形成呼应与理论对照。citeturn7view0

### 与至少六类相关方法的对照表

下表以“监督类型、边使用方式、典型数据集、指标”对齐本项目研究定位。注意：不同论文/代码对“是否用边属性”定义不一；这里按“是否显式建模 edge attributes / edge weights 并进入目标函数或消息传递”解释。

| 方法 | 年份 | 监督类型 | 边使用方式 | 主要创新点 | 常用数据集（示例） | 常报告指标 |
|---|---:|---|---|---|---|---|
| VGAE（Variational Graph Auto-Encoders）citeturn11view2 | 2016 | 无监督表示学习（常再接聚类） | 使用邻接矩阵重构；通常不含边属性 | 变分图自编码器，用重构学习节点表示 | Cora/Citeseer/Pubmed 等citeturn11view2 | AUC/AP（链路），有时 NMI/ARI/ACC（聚类） |
| DGI（Deep Graph Infomax）citeturn12view0 | 2018 | 无监督表示学习（常再接聚类） | 使用邻接做 GCN 编码；不建模边属性 | 最大化局部—全局互信息的节点表征学习citeturn12view0 | 引文网络、社交网络等citeturn12view0 | 下游分类/有时聚类 NMI/ARI |
| DAEGC（Attributed Graph Clustering: A Deep Attentional Embedding Approach）citeturn15search11 | 2019 | 端到端图聚类（无监督 + self-training） | 利用邻接/属性； attention 作用于邻居 | 目标导向聚类：图注意自编码 + 自训练聚类联合优化citeturn15search11 | 属性图：Cora/Citeseer/Pubmed 等 | NMI/ARI/ACC |
| SDCN（Structural Deep Clustering Network）citeturn11view0 | 2020 | 深聚类（无监督 + 双自监督） | 通常由特征构 KNN 图；不含真实边属性 | AE 与 GCN 通过 delivery operator 融合，并用双自监督机制优化citeturn11view0 | 多模态/多领域数据（含图） | NMI/ARI/ACC |
| DAGC（Deep Attention-guided Graph Clustering with Dual Self-supervision）citeturn16view0 | 2021/2023 | 深图聚类（无监督 + 双自监督） | 使用结构图；强调多尺度融合 | 异质融合+尺度融合+分布融合；软/硬双自监督（含 KL 变体）citeturn16view0 | 图基准数据集 | NMI/ARI/ACC |
| LSEnet（Lorentz Structural Entropy Neural Network）citeturn14view0turn13search1 | 2024 | 深图聚类（结构信息驱动） | 主要用结构（邻接）+ 节点属性；不强调边属性 | 提出可微结构信息 DSI 与 Lorentz 双曲空间分割树学习，实现“未知 K”导向citeturn14view0 | Cora/Citeseer/Pubmed/Amazon 等（论文常见）citeturn13search1 | NMI/ARI/ACC（聚类） |
| DeSE（Unsupervised Graph Clustering with Deep Structural Entropy）citeturn14view1 | 2025 | 无监督图聚类（结构熵 + 结构学习） | 强调结构学习层生成/增强图；边多作为结构对象 | 用软 assignment 计算结构熵；结构学习层缓解稀疏/噪声；ASS 层联合优化citeturn14view1 | 多个图基准citeturn14view1 | NMI/ARI/ACC |
| ASIL（arXiv:2504.09970，TPAMI 接收，v2 最后修订 2026‑02‑02）citeturn7view0 | 2026 | 无监督深图聚类（结构信息 + 增强/对比） | 用结构信息/结构熵并讨论树对比的效率界 | 提出可微结构信息框架与增广结构信息学习，并声称在线性复杂度下改进 conductance 等citeturn7view0 | 多基准（摘要提 Citeseer 等）citeturn7view0 | NMI 等（摘要给出提升）citeturn7view0 |

对照中可以看到：ECHF 的“创新位置”与传统 DAEGC/SDCN/DAGC 不同——它不以“自训练/伪标签/KL 目标分布”作为核心，而是继承 LSEnet/DSI/ASIL 的结构信息路线，把主要创新投入到 **边信号如何进入‘树’的学习**（尤其是 assignment-score）以及 **边属性如何沿层级传播**。

### 建议的模型流程图（Mermaid）

```mermaid
flowchart TD
  A[输入: 节点特征 X, 邻接 A, 可选 edge_attr] --> B[边预加权/构造: V1-V4<br/>得到 adj_msg / adj_si]
  B --> C[叶子嵌入: LorentzConv + LorentzBoost]
  C --> D[KNN 增强: adj_aug<br/>dense 或 edge-KNN 模式]
  D --> E[训练图: adj_train = adj_base + alpha * adj_aug]
  E --> F[LSENet 分层: 多层 LorentzAssignment + pooling]
  F --> G{edge_variant}
  G -->|V5| H1[结构 trunk: log(w) + reliability<br/>可选 graph_alpha]
  G -->|V6/V7| H2[边属性门控: attr_encoder -> gate/bias]
  G -->|V8| H3[结构-属性校准混合: mix_beta]
  G -->|V12| H4[trunk + 校准残差]
  H1 --> I[得到各层 assignment 矩阵 C^h]
  H2 --> I
  H3 --> I
  H4 --> I
  I --> J[层级邻接: ass^T A ass]
  J --> K[结构熵/结构信息损失: SI loss]
  K --> L[AdamW 更新]
  B -. Path-A .-> PA[edge_attr_weight_blend<br/>把 edge_attr 映射为 SI 权重]
  F -. Path-B .-> PB[edge_attr_hierarchical<br/>edge_attr 随层级粗化传播]
```

## 研究空白与待完成实验

本节以“可发表论文”的标准列出最关键的缺口，并给出必须补齐的实验与默认设定。凡仓库未声明的细节，均标注为“未指定”，并给出可复现默认值。

### 关键研究空白

**边属性有效性缺口**  
当前主线 V12/Path‑B 的叙述对“边属性有助于异质图聚类”非常关键，但公共基准（Cora/Citeseer/Pubmed/Amazon 等）往往没有真实 edge_attr，代码更多使用 generic edge features 替代。这会让论文评审质疑：提升来自“更复杂的边权工程”还是来自“真实关系语义”？必须用至少一个“真实边属性/边类型”数据集补强。

**边属性有效性问题（已回答，2026-02-20）**  
对“初始几个公共数据集是否带 edge_attr”的审计结论如下：
- 在原始 PyG 对象中，`cora/citeseer/pubmed/computers/photo` 均 **无** `edge_attr`。证据：`results/edge_attr_audit_v1/README.md`，明细：`results/edge_attr_audit_v1/edge_attr_presence.csv`。  
- 当前仓库训练时会在无 `edge_attr` 情况下自动构造 7 维 generic edge features（余弦相似、L1/L2 差、度相关统计等），并作为 `data.edge_attr` 使用。实现：`data.py:45`, `data.py:211`, `data.py:212`, `data.py:254`。  
- 结论：经典基准上“edge-enhanced”的改进应解释为“**派生边先验 + 融合机制**”而非“真实关系语义边属性”。论文中需把 claim 收敛为该表述，并额外补一个真实边属性/边类型数据集实验来闭环。

**未知 K 的论证缺口**  
LSEnet/ASIL/DeSE 路线的关键卖点之一是“无需预设 K 或弱化对 K 的依赖”。然而本仓库仍需 `max_nums` 作为层级容量，并在评估中对小簇做修复（epsInt）。这在实践上合理，但需要用实验说明：  
- `max_nums` 作为上界是否鲁棒？  
- “空簇/小簇修复”是否造成指标偏差？  
- 是否存在更一致的“自适应截断/自动停止层级”策略？

**未知 K 问题（已回答，2026-02-20）**  
已完成一轮针对 `max_nums/epsInt` 的系统消融：`results/benchmark_unknownk_ablation_v2_compact`（120/120 成功，见 `decision.json`）。

- 原版 DSE 是否也有这些参数：**是**。参考实现同样包含 `max_nums` 与 `epsInt`，并调用 `fix_cluster_results`。证据：  
`reference/DSE_clustering-main/main.py:35`, `reference/DSE_clustering-main/main.py:43`, `reference/DSE_clustering-main/modules/dsi.py:49`。  
- `max_nums` 上界鲁棒性：**不鲁棒，且高度敏感**。  
在 `k in {6,10,16}` 扫描中，`k16-k6` 的总体退化明显（例如 baseline ΔNMI=-0.1254，g15 ΔNMI=-0.1338）。证据：`results/benchmark_unknownk_ablation_v2_compact/analysis_summary.md`，`results/benchmark_unknownk_ablation_v2_compact/k16_minus_k6_overall.csv`。  
- 是否“自动发现 K”：当前证据显示 **没有**。  
`pred_n_clusters_mean` 与 `max_nums` 完全相等（逐数据集逐条件均为 True），说明当前更接近“容量上界=输出簇数”。证据：`results/benchmark_unknownk_ablation_v2_compact/max_sweep_predk_equals_max_check.csv`。  
- `epsInt` 修复是否引入指标偏差：**有影响但总体较小**。  
相对 no-fix（`epsInt=-1`）的总体变化在小量级（baseline: ΔNMI≈+0.0012@eps=8；g15: ΔNMI≈+0.0040@eps=8），且 `delta_pred_k_vs_nofix` 为 0。证据：`results/benchmark_unknownk_ablation_v2_compact/eps_sweep_delta_overall.csv`。  
- 能否取消或自适应：当前代码路径下 **不能直接取消 `max_nums`**（层级 assignment 维度由其定义），可做的是把 no-fix (`epsInt=-1`) 作为评估对照，或引入“自适应截断”策略（尚未实现）。

**自适应截断代理实验（已回答）**  
- 用更小上界 `k=6` 作为“保守容量代理”相对默认 k（每数据集默认值）整体提升明显（baseline ΔNMI=+0.0616，g15 ΔNMI=+0.0672）。证据：`results/benchmark_unknownk_ablation_v2_compact/k6_minus_default_overall.csv`。  
- 这证明“上界容量可调”有效，但本质仍是手动超参，不是自动停止/自动选 K。

**对比基线不足与协议统一问题**  
仓库内部对比主要围绕自家 V 变体与少量条件（例如 G15 vs G17）展开。论文需要引入更广泛基线，至少覆盖：  
- 结构熵路线：LSEnet、DeSE、ASIL（若能复现实验最好）。citeturn14view0turn14view1turn7view0  
- 深图聚类路线：SDCN、DAEGC、DAGC 等。citeturn11view0turn15search11turn16view0  
- 表示学习+聚类路线：VGAE、DGI + 聚类头。citeturn11view2turn12view0  

**效率—精度权衡缺口**  
代码提供 edge‑KNN 模式以支持大图，但尚缺“效率曲线/误差曲线”：当从 dense‑KNN 切换到 edge‑KNN 时，性能与时间如何变化？这类实验在系统论文中几乎必需。

**效率—精度权衡补实验（已完成第一轮，2026-02-20）**  
为补齐该缺口，已新增并执行统一脚本 `tools/run_knn_mode_tradeoff.py`，输出 runtime/accuracy 曲线与 dense 参照误差：

- **经典基准全量（5 数据集 × 2 seeds × k={4,8,16} × mode={dense,edge,auto}）**  
  - 命令与结果：`results/benchmark_knn_mode_tradeoff_v1/README.md`  
  - 规模：90/90 成功。  
  - 曲线：  
    - `results/benchmark_knn_mode_tradeoff_v1/figures/fig1_efficiency_runtime_vs_k.png`  
    - `results/benchmark_knn_mode_tradeoff_v1/figures/fig2_error_delta_nmi_vs_k.png`  
    - `results/benchmark_knn_mode_tradeoff_v1/figures/fig3_tradeoff_speedup_vs_delta_nmi.png`  
  - 关键结论（相对 dense）：  
    - `edge`：平均 **更慢**（speedup≈0.70x），但 NMI 有正增益（k=4/8/16 分别约 +0.0035/+0.0093/+0.0195）。  
    - `auto`：速度接近 dense（0.96x~0.99x），精度差异接近 0（ΔNMI 约 -0.0005~+0.0014）。  

- **城市规模梯度（boston/washingtondc/singapore/melbourne，k=8）**  
  - 命令与结果：`results/benchmark_knn_mode_tradeoff_urban_scale_v1/README.md`  
  - 规模：12/12 成功。  
  - 曲线：  
    - `results/benchmark_knn_mode_tradeoff_urban_scale_v1/figures/fig1_efficiency_runtime_vs_k.png`  
    - `results/benchmark_knn_mode_tradeoff_urban_scale_v1/figures/fig2_error_delta_nmi_vs_k.png`  
    - `results/benchmark_knn_mode_tradeoff_urban_scale_v1/figures/fig3_tradeoff_speedup_vs_delta_nmi.png`  
  - 关键结论：在这些中等规模城市图上，`auto` 与 `dense` 的精度几乎一致，运行时间略优（平均 speedup≈1.03x）；`edge` 精度近似持平但平均偏慢（≈0.92x）。  

- **大图压力（boston/tokyo/beijing，k=8）**  
  - 命令与结果：`results/benchmark_knn_mode_tradeoff_urbanstress_v1/README.md`  
  - 规模：7/9 成功。  
  - 关键结论：`dense` 在 `tokyo` 与 `beijing` 直接 OOM（见 `results/benchmark_knn_mode_tradeoff_urbanstress_v1_urban_tokyo_plot_s0_dense_k8/runner.log` 与 `results/benchmark_knn_mode_tradeoff_urbanstress_v1_urban_beijing_plot_s0_dense_k8/runner.log`），而 `edge/auto` 可完成，说明 edge‑KNN 的核心价值在于**可扩展性与可运行性边界**，不只是小图提速。

> 结论性建议：论文中应将 efficiency‑accuracy 结论拆成两层报告：  
> (1) 小/中图：dense 往往更快，edge 主要贡献在精度或结构指标；  
> (2) 大图：dense 进入 OOM 区间，edge/auto 提供可训练路径（这是系统价值主轴）。

### B30 系列新一代 edge-aware 路线与控制实验进展（已回答，2026-03-25）

在完成 ECHF/G15 主线整理后，仓库进一步从“**现有代码主干本身**”出发，而非继续沿论文叙事反推，提出了 `B30` 系列作为下一代 edge-aware 路线。其出发点是：当前系统真正决定聚类树生成的主干是 `encoder -> ass_dict -> adj_dict -> se_loss`，因此边信息若要成为有效信号，就必须进入 **assignment、层级上传、结构图权重** 这三条主路径，而不能只在叶层局部做 edge MLP。

#### B30 系列的结构分解

- `B30 / V30`：新增双标量边头，把 `edge_attr` 同时映射到 `msg` 图与 `si` 图的结构权重（dual-scalar route）。  
- `B31 / V31`：在 `B30` 基础上，把边属性进一步注入 assignment-score，测试“边是否应该先影响谁并到谁”。  
- `B32 / V32`：在 `B31` 基础上加入层级 edge-state 上传，测试边语义是否应跨层传播。  
- `B33 / V33`：在 `B32` 基础上再把边先验送入 augment graph，测试“边先验是否也应影响 KNN augmentation”。

这一路线并未重写 `_si_loss()` 本体，而是保持结构熵损失仍然只吃标量邻接，把多维边语义限制在“**树生成机制**”而非“**结构熵公式本体**”上。这是出于工程可控性考虑：先验证 edge-aware 是否真的改变了树，再决定是否需要改目标函数。

#### P0：运行期诊断先验证“模块是否活着”

在继续大规模实验前，已补充分支健康诊断，新增以下 runtime 指标：

- `diag_factor_live`
- `diag_dual_live`
- `diag_assign_live`
- `diag_hier_live`
- `diag_aug_live`
- `diag_dead_branch_count`

并同时记录：

- `hier_edge_levels_active_ratio`
- `hier_edge_nonzero_ratio`
- `hier_edge_mean_abs`
- `edge_aug_bias_mean`
- `edge_aug_bias_std`

结果文件：

- `results/diagnostic_b30_components_smoke_v2/branch_health_summary.csv`
- `results/diagnostic_b30_components_smoke_v2/stage_delta_summary.csv`

结论：

- `B20` 的结构边权学习是活的。  
- `B30` 的 dual `msg/si` scalar 头是活的，且两头数值已分化，不是同一条死支路。  
- `B31` 的 assignment fusion 是活的。  
- `B32/B33` 的 hierarchical edge pooling 通路是活的，并且确实向父层上传了非零 edge state。  
- `B33` 的 augment prior 也不是死分支；问题不是“没接通”，而是“接通了但当前有害”。

这一步非常关键，因为它排除了“代码接线错误导致的假消融”。

#### P1/P2：第一轮控制实验筛选（3 类机制数据）

本轮不是直接上公开 benchmark，而是先在 3 个代表性控制数据上做机制筛选：

- `synth_edgectrl_v1_mredu_h65_s90_ds00`：冗余 edge_attr（边信息主要重复节点/拓扑信息）  
- `synth_edgectrl_v1_mmisl_h65_s90_ds00`：误导 edge_attr（边信息与最终 fine label 错位）  
- `synth_edgectrl_v1_mhier_h65_s90_ds00`：层级 edge_attr（边信息更偏 coarse/hierarchical semantics）

结果文件：

- `results/diagnostic_b30_round1_screen_v1/summary_by_condition.csv`
- `results/diagnostic_b30_round1_screen_v1/summary_by_condition_dataset.csv`
- `results/diagnostic_b30_round1_screen_v1/stage_delta_summary.csv`
- `results/diagnostic_b30_round1_screen_v1/stage_delta_by_dataset.csv`
- `results/diagnostic_b30_round1_screen_v1/branch_health_summary.csv`

本轮配置：

- `seed=0`
- `epochs=20`
- 版本：`baseline_v1`, `g20_se_consistent_main`, `B30`, `B31`, `B32`, `B33`，以及 5 个修正版本 `B34/B35/B36/B37/B38`

#### 第一轮保留的 3 个版本

| 版本 | 保留理由 | 当前结论 |
|---|---|---|
| `B32` | hierarchy 打开后在 3 个控制机制上没有负增益，且整体分数最高一档 | 当前 hierarchy 主候选 |
| `B37` | 与 `B32` 打平，但更简单（hard hierarchy），且 SI loss 略低 | `B32` 的简化等价替代 |
| `B36` | 是 `B33` augment 修复线里唯一稳定回升的版本 | augment 线唯一值得继续保留的候选 |

#### 为什么保留 `B32`

`B32` 相比 `B31` 只多一个模块：**hierarchical edge-state pooling**。  
如果这条线真的有价值，那么最基本的要求是：它至少不应在多种 edge semantics 下引入稳定负增益。

而结果正是如此：

- `mhier`：`B31 -> B32` 的 `ΔNMI = 0`
- `mmisl`：`ΔNMI = 0`
- `mredu`：`ΔNMI = +2.14e-5`

也就是说，`hier_on` 在 3 个控制机制上都是**非负**的。结合 `diag_hier_live_mean = 1.0` 与 `diag_dead_branch_count_mean = 0`，可以判断：

> `B32` 的 hierarchy 分支不是“无效接线”，而是“当前已经接通，且至少稳健不伤害”的可保留主线。

#### 为什么保留 `B37`

`B37` 的改动很小，只是把 `B32` 的 soft top-k hierarchy 改为更硬的 `topk=1` pooling。  
这条分支保留的逻辑不是“它更强”，而是：

- 在 3 个控制数据上，`B37` 与 `B32` 的 `NMI/ARI` 完全打平；  
- 但 `SI loss` 在 3 个控制数据上都略低一些。

这说明当前 `B32` 的 soft hierarchy 还没有打出比 hard hierarchy 更强的增益。于是：

> `B37` 应作为 `B32` 的低复杂度对照保留，用于下一阶段确认“当前是否真的需要更复杂的层级 pooling”。

#### 为什么保留 `B36`

`B33` 的主要问题已经由控制实验确认：  
`B32 -> B33` 这一跳会在 3 个控制数据上全部掉点，平均 `ΔNMI = -0.002302`。  
因此第一轮的重点不是“继续加 augment”，而是判断 augment 线是否**可救**。

`B36` 的设计是：

- `positive-only augment prior`
- `small-scale augment prior`

其理论含义是：把 augment 从“强 signed perturbation”改为“弱正向支持项”。  
结果也支持这一点：

- 在 `mredu` 上，`B36 > B33`
- 在 `mmisl` 上，`B36 > B33`
- 在 `mhier` 上，`B36 = B33`

因此：

> `B36` 证明了 augment 线的问题更像是“注入方式不对/强度过大”，而不是“augment 这个想法完全错误”。  
> 它还不是当前最优版本，但足以进入下一阶段确认。

#### 当前证据的边界

需要明确指出：上述结论目前仍然是**“控制实验上的初筛结论”**，还不是“最终主线定版”。

当前证据覆盖：

- 3 类代表性控制机制
- `seed=0`
- `epochs=20`

尚未覆盖：

- 多 seed 稳定性
- 更长训练轮次
- 更完整的 `homophily / signal / noise` 网格

因此更准确的表述是：

> `B32/B37/B36` 是当前值得推进到下一阶段的 **promoted candidates**，  
> 不是已经完成全变量验证的最终主线。

#### 下一阶段实验顺序

下一步已确定为“中程确认轮”，而非立即扩展到公开 benchmark：

- 数据：仍用 `mredu/mmisl/mhier`
- seeds：`0,1,2`
- epochs：`60`
- 对照版本：`baseline_v1`, `g20_se_consistent_main`, `B31`, `B32`, `B36`, `B37`

判据：

1. 若 `B36` 仍明显落后于 `B32/B37`，则冻结 augment 线。  
2. 若 `B37` 持续打平 `B32`，则后续优先保留 `B37`。  
3. 只有在 `B36` 逼近或超过 `B32/B37` 时，才继续推进 augment + hierarchy 的组合线。

#### 中程确认结果（已完成，2026-03-25）

上述确认轮已经完成，结果目录：

- `results/diagnostic_b30_round2_confirm60_v2/summary_by_condition.csv`
- `results/diagnostic_b30_round2_confirm60_v2/summary_by_condition_dataset.csv`
- `results/diagnostic_b30_round2_confirm60_v2/stage_delta_summary.csv`
- `results/diagnostic_b30_round2_confirm60_v2/stage_delta_by_dataset.csv`
- `results/diagnostic_b30_round2_confirm60_v2/branch_health_summary.csv`

配置：

- 数据：`mredu/mmisl/mhier`
- seeds：`0,1,2`
- epochs：`60`
- 对照：`baseline_v1`, `g20_se_consistent_main`, `B31`, `B32`, `B36`, `B37`

总体排序（按 `NMI mean`）：

1. `G20`：`0.02922`
2. `B31`：`0.01635`
3. `B32`：`0.01553`
4. `B37`：`0.01547`
5. `B36`：`0.01450`
6. `baseline`：`0.00675`

这轮有两个关键修正：

**其一，`B31` 成为当前 `B30` 家族里最稳的版本。**  
首轮 `20 epoch / seed=0` 的结果里，`B32/B37` 看起来略优；但在 `60 epoch × 3 seeds` 后，`B31` 反而整体最好。说明先前的 hierarchy 优势并不稳固，更像是短程训练下的弱正偏差。

**其二，`hier_on: B31 -> B32` 在中程确认中转为总体负增益。**  
`stage_delta_summary.csv` 给出：

- `hier_on` 平均 `ΔNMI = -0.000821`
- 平均 `ΔARI = -0.000529`

逐数据集看：

- `mhier`：`+1.54e-5`（极小正增益）
- `mmisl`：`-0.001022`
- `mredu`：`-0.001458`

这意味着：

> hierarchy 分支仍然是活的，但它只在“确实具有层级 edge semantics”的控制数据上保持近乎中性/极小正增益；  
> 一旦 edge signal 是冗余或误导性的，层级上传会把这类边语义进一步扩散到更高层，反而伤害结果。

因此，`B32/B37` 不应再作为当前主推主线，只能作为“**层级 edge semantics 专项验证分支**”保留。

#### 对 `B36` 的最终判断

`B36` 在中程确认轮里仍然优于 baseline，但没有超过 `B31`，也没有成为稳定优于 hierarchy 线的版本：

- 总体 `NMI`：`B36 = 0.01450 < B31 = 0.01635`
- 在 `mredu` 上，`B36` 略高于 `B32/B37`
- 但在 `mmisl/mhier` 上都落后于 `B31/B32/B37`

因此更准确的结论是：

> `B36` 证明 augment 线仍可被“保守化”后继续研究，但它不足以进入当前主线；  
> 在下一阶段应将 augment 线降级为探索分支，而不是继续与主线并行扩展。

#### B30 系列当前阶段性结论

综合首轮筛选与中程确认：

- `B31`：当前 `B30` 家族的**最佳保留版本**  
- `B32/B37`：从“主候选”降级为“层级语义专项分支”  
- `B36`：从“augment 修复候选”降级为“保守 augment 探索分支”

与 `G20` 对比也很重要：在这 3 类控制数据和本轮设置下，`G20` 明显强于所有 `B30` 版本。  
这说明“**SE-consistent scalar route**”目前仍然是更稳的 edge-aware 主线，而 `B30` 家族的价值更多在于：

1. 验证哪些 edge-aware 结构是**真的活着**；  
2. 验证 assignment / hierarchy / augment 三条路径各自的边界条件；  
3. 为下一阶段更强的 edge-conditioned message passing 或更精细的层级聚合提供归因基础。

#### 扩展控制网格结果（已完成，2026-03-25）

为了避免把结论建立在单一控制点上，随后又把 edge-control 控制数据从 3 个扩展到 9 个：

- 模式：`mredu / mmisl / mhier`
- 同配性：`h45 / h65 / h85`
- 固定：`signal=0.90`
- seeds：`0,1,2`
- epochs：`60`

结果目录：

- `results/diagnostic_b30_round3_edgectrl9_v1/summary_by_condition.csv`
- `results/diagnostic_b30_round3_edgectrl9_v1/summary_by_condition_dataset.csv`
- `results/diagnostic_b30_round3_edgectrl9_v1/stage_delta_summary.csv`
- `results/diagnostic_b30_round3_edgectrl9_v1/stage_delta_by_dataset.csv`

总体上，`NMI` 排名为：

1. `B36`: `0.11473`
2. `G20`: `0.11412`
3. `B32`: `0.10699`
4. `B37`: `0.10688`
5. `B31`: `0.10591`
6. `baseline`: `0.07422`

但若看 `ARI`，则 `G20` 最高（`0.10542`），`B36` 仅为 `0.09213`。  
这说明：

> `B36` 与 `G20` 的优劣已经不是“单一主线 vs 失败备份”的关系，  
> 而是开始呈现**不同 edge-control 机制下的互补适用区间**。

##### 按控制模式拆解

- **`mhier` 模式**：`B36` 最好（`NMI≈0.11364`），略高于 `G20`
- **`mredu` 模式**：`B36` 最好（`NMI≈0.11499`），高于 `G20`
- **`mmisl` 模式**：`G20` 最好（`NMI≈0.12143`），明显高于 `B36`

这说明：

- 当 edge semantics 更偏层级/冗余但仍有稳定信息时，`B36` 的“保守 augment”更能放大有用边先验；  
- 当 edge semantics 明显带有错位/误导性时，`G20` 的 **SE-consistent scalar route** 仍然更稳。

##### 按同配性拆解

- **`h45`**：`G20` 最好，`B36` 最弱  
- **`h65`**：`G20` 仍最好，`B32/B36/B31/B37` 接近但明显落后  
- **`h85`**：`B36` 最好（`NMI≈0.32286`），超过 `G20`

这说明：

> `B36` 的优势主要出现在**高同配 + 强边信号**区间；  
> `G20` 则在低/中同配区间更稳，更像一个泛化主线。

##### 对 hierarchy 线的进一步修正

在 9 数据控制网格上，`hier_on: B31 -> B32` 的总体平均竟然转为正值：

- 平均 `ΔNMI = +0.001086`
- 平均 `ΔARI = +0.001499`

但拆开看后发现，这个正增益并不“普遍”：

- 按模式：
  - `mmisl`: 正增益
  - `mredu`: 负增益
  - `mhier`: 微负/近中性
- 按同配性：
  - `h45`: 近零
  - `h65`: 小正
  - `h85`: 较明显正

因此 hierarchy 的更准确结论应改写为：

> `B32/B37` 不是普遍劣于 `B31`，  
> 但其收益高度依赖于“高同配”以及部分特殊 edge-control 机制，  
> 暂时仍不适合作为统一主线，只适合作为**条件性分支**保留。

#### B30 家族在当前阶段的最终判断（截至 round3）

综合 `round1 -> round2 -> round3`，当前应采用**双主线、条件选择**的口径：

- **泛化主线**：`G20`
  - 适用于低/中同配，或边语义可能误导/错位的情形
- **高信号主线**：`B36`
  - 适用于高同配，且 edge semantics 更可能是层级/冗余但稳定有效的情形

同时：

- `B31`：保留为最干净的 `B30` 无 augment / 无 hierarchy 对照
- `B32/B37`：保留为 hierarchy 条件性分支，不再视作当前主线

这一结果也意味着，若继续推进 `B30` 家族，下一步更合理的目标不再是“继续调 hierarchy 的 top-k”，而是：

1. 让 `B36` 在低/中同配区间更稳；  
2. 或者直接进入更高层次的 **edge-conditioned message passing**，尝试弥补 `B36` 对错位边语义的脆弱性。

#### B40 系列：转向 edge-conditioned message passing / refined edge-state pooling（已完成首轮）

基于前述判断，下一阶段不再继续堆 `hierarchy/augment`，而是直接测试两条更接近“边信息进入树生成机制”的路线：

- `B40`
  - 以 `B31` 为底座
  - 新增 **edge-conditioned message passing**
  - 具体做法是在 leaf-level `LorentzAgg` 中用 `edge_attr -> gate factor` 调制消息图权重，并重新做 degree normalization
- `B41`
  - 以 `B32` 为底座
  - 不加 message gating
  - 只把层级 `edge-state pooling` 改为 **assignment-confidence weighted pooling**
- `B42`
  - 以 `B32` 为底座
  - 同时启用 **message gating + confidence-weighted pooling**

这轮首筛已经完成，结果目录：

- `results/diagnostic_b40_round1_confirm60_v1/summary_by_condition.csv`
- `results/diagnostic_b40_round1_confirm60_v1/summary_by_condition_dataset.csv`
- `results/diagnostic_b40_round1_confirm60_v1/branch_health_summary.csv`

实验设置：

- 数据：`mredu / mmisl / mhier`
- seeds：`0,1,2`
- epochs：`60`
- 对照：`baseline_v1`, `g20_se_consistent_main`, `B31`, `B36`, `B40`, `B41`, `B42`

总体 `NMI` 排名为：

1. `G20`: `0.02886`
2. `B40`: `0.01929`
3. `B42`: `0.01915`
4. `B41`: `0.01571`
5. `B36`: `0.01516`
6. `B31`: `0.01499`
7. `baseline`: `0.00678`

分机制结果更说明问题：

- `mhier`
  - `G20`: `0.03275`
  - `B40`: `0.02225`
  - `B42`: `0.02201`
  - `B41`: `0.01733`
- `mmisl`
  - `G20`: `0.01942`
  - `B41`: `0.01614`
  - `B31`: `0.01449`
  - `B40`: `0.01429`
- `mredu`
  - `G20`: `0.03443`
  - `B42`: `0.02152`
  - `B40`: `0.02135`
  - `B36`: `0.01474`

可以得到三点结论。

**第一，`edge-conditioned message passing` 确实比继续调 augment/hierarchy 更值得推进。**

`B40` 相对 `B31` 的平均增益是：

- `ΔNMI = +0.00430`
- `ΔARI = +0.00348`

并且收益主要来自：

- `mhier`: `ΔNMI = +0.00550`
- `mredu`: `ΔNMI = +0.00762`

这说明 message gating 至少在“层级边语义”和“冗余但稳定的边语义”两类场景下，能比纯 assignment residual 更有效地把边信息转成聚类增益。

**第二，只改 refined pooling（`B41`）的收益有限。**

`B41` 相对 `B31` 的平均增益只有：

- `ΔNMI = +0.00072`
- `ΔARI = +0.00042`

且主要只在 `mmisl/mhier` 上有轻微改善，在 `mredu` 上基本没有收益。  
这说明当前瓶颈不在 pooling 的 top-k 细节，而更在于“leaf message graph 是否真的被 edge semantics 调制”。

**第三，`B42` 没有再明显超过 `B40`。**

`B42` 相对 `B40` 的平均差值为：

- `ΔNMI = -0.00015`
- `ΔARI = -0.00010`

所以在当前实现下：

> `confidence-weighted pooling` 并没有给已经启用 message gating 的模型再带来稳定额外收益；  
> 这意味着下一步主线应优先围绕 `B40` 展开，而不是继续把 `B42` 当成默认升级版。

同时，`branch_health_summary.csv` 也确认了这次不是“死分支假提升”：

- `B40`: `diag_msg_live = 1.0`
- `B42`: `diag_msg_live = 1.0`, `diag_hier_live = 1.0`
- `B41`: `diag_hier_live = 1.0`
- 所有条件 `diag_dead_branch_count_mean = 0`

因此当前阶段的最准确判断应更新为：

- **`G20` 仍是更稳的泛化主线**
- **`B40` 是 `B30` 家族下一代最值得继续推进的新主候选**
- `B41` 只证明 refined pooling 可行，但不足以单独成为主线
- `B42` 说明“message gating + refined pooling”目前并非简单相加增益

所以下一步应转入：

1. 以 `B40 vs G20` 为核心，对更完整的控制变量网格做最终机制对决  
2. 若 `B40` 在更广控制网格上继续稳于 `B31/B36`，再围绕 `B40` 做更细的 gate-scale / matched-edge / reliability 设计

#### B40 扩展控制网格（9 datasets × 1 seed，已完成）

为快速判断 `B40` 的适用区间，又补跑了一轮更广但较轻的区域图筛选：

- 目录：`results/diagnostic_b40_round2_edgectrl9_s0_v1`
- 数据：`mhier/mmisl/mredu × h45/h65/h85`
- 条件：`baseline_v1`, `G20`, `B31`, `B36`, `B40`, `B42`
- 设置：`seed=0`, `epochs=60`

总体 `NMI` 排名：

1. `B40`: `0.17499`
2. `B42`: `0.17458`
3. `G20`: `0.15094`
4. `B36`: `0.14391`
5. `B31`: `0.13640`
6. `baseline`: `0.08793`

这轮的价值主要在“区域图”而不是统计显著性。

按同配性拆开：

- `h45`
  - `B36` 最好：`≈0.00971`
  - `B40/B42` 仅略高于 baseline
- `h65`
  - `G20` 最好：`≈0.06187`
  - `B40` 第二：`≈0.04096`
- `h85`
  - `B40` 最好：`≈0.47783`
  - `B42` 紧随其后：`≈0.47769`

按控制模式拆开：

- `mhier`
  - `B40` 最好：`≈0.18077`
- `mmisl`
  - `B40` 最好：`≈0.16416`
- `mredu`
  - `B40` 最好：`≈0.18003`

这轮与上一轮 `3 seeds × 3 representative controls` 的关系应这样理解：

- `round1_confirm60_v1` 给的是**更稳的 3-seed 判断**
- `round2_edgectrl9_s0_v1` 给的是**更广的区域图判断**

两轮并不矛盾：

- 在 `h65` 的代表性 3-seed 控制组上，`G20` 仍更稳
- 但在更广的 9-grid 区域图里，`B40` 在 `h85` 区间的优势非常明显，足以把整体均值抬到第一

因此当前最稳的口径应是：

> `B40` 已经成为比 `B31/B36` 更强的新一代 `B30` 家族候选；  
> 它的优势主要来自 **high-homophily / high-signal** 区间，且在单 seed 区域图上对三类 edge-control 都表现出竞争力；  
> 但若要把它升级成真正的统一主线，仍需对 `h65` 区间做更完整的多 seed 对决，并继续与 `G20` 正面比较。

#### B40 message 分支调参与 B43 结构变体（已完成）

为了把 “edge-conditioned message passing” 进一步拆解，又做了两件事：

1. **结构变体**：`B43 = B40 + matched-edge-only message gating`
2. **message gate 调参**：
   - `B44`: `edge_msg_gate_scale = 0.20`
   - `B45`: `edge_msg_gate_scale = 0.50`
   - `B46`: `B40 + confidence gate`

结果目录：

- `results/diagnostic_b40_tuning_repr3_v1`
- `results/diagnostic_b45_grid9_s0_v1`

其中：

- `diagnostic_b40_tuning_repr3_v1`
  - 数据：`mredu/mmisl/mhier @ h65`
  - seeds：`0,1,2`
  - 用来做稳健筛选
- `diagnostic_b45_grid9_s0_v1`
  - 数据：`9-grid`
  - seed：`0`
  - 用来做区域图确认

##### 调参筛选结果（3 controls × 3 seeds）

总体 `NMI`：

1. `B45`: `0.02984`
2. `G20`: `0.02818`
3. `B43`: `0.02003`
4. `B40`: `0.01911`
5. `B36`: `0.01661`
6. `B31`: `0.01523`
7. `B46`: `0.01461`
8. `B44`: `0.01228`

这说明三件事：

**第一，gate scale 是真正敏感的主旋钮。**

- 把 `edge_msg_gate_scale` 从 `0.35` 提到 `0.50`（`B45`）后，平均 `NMI` 从 `0.01911` 提到 `0.02984`
- 且在三个代表性控制组上都提升：
  - `mhier`: `0.02230 -> 0.03499`
  - `mmisl`: `0.01378 -> 0.02106`
  - `mredu`: `0.02126 -> 0.03347`

**第二，`matched-edge-only`（`B43`）是正收益，但幅度有限。**

- `B43` 相对 `B40` 有小幅改善
- 但提升远不如直接把 gate scale 调大

所以当前最值得保留的结构结论不是“B43 替代 B40”，而是：

> `matched-edge-only` 有价值，但暂时只是辅助修正；  
> 当前最强增益来源仍然是 **更充分的 message gating 强度**。

**第三，confidence gate（`B46`）目前无效。**

- `B46` 不仅没超过 `B40`，还低于 `B31/B36`
- 这说明当前这版 “低置信度向 1 收缩” 做法过于保守，把有用 message gate 也压掉了

##### `B45` 区域图确认（9-grid × 1 seed）

总体 `NMI`：

1. `B45`: `0.19981`
2. `B40`: `0.17447`
3. `B43`: `0.17431`
4. `G20`: `0.15097`

关键观察：

- `h45`
  - `B45` 没有优势，和 `B40/B43`、`G20` 一样都接近零区间
- `h65`
  - `B45` 已明显超过 `G20`
  - `mhier`: `0.08588 > 0.07517`
  - `mmisl`: `0.04341 > 0.03137`
  - `mredu`: `0.08444 > 0.07924`
- `h85`
  - `B45` 优势更大，显著高于 `B40/B43/G20`

因此当前 message 分支的最准确更新结论是：

- `B40` 证明了 **edge-conditioned message passing** 方向成立
- `B43` 证明了 **matched-edge-only** 是可用修正，但不是主增益来源
- `B45` 则进一步说明：  
  **真正需要继续深挖的，是 message gate 的强度与形状，而不是继续堆 hierarchy/augment**

截至目前，`B30` 家族里最值得继续推进的消息分支主候选，已经从 `B40` 更新为：

- **`B45 = B40 + stronger edge message gate`**

#### `B45` 全量确认轮（9-grid × 3 seeds，已完成）

为避免 `9-grid × 1 seed` 的区域图结论过早下判断，又补跑了完整确认轮：

- 目录：`results/diagnostic_b45_confirm_grid9_v1`
- 数据：`mhier/mmisl/mredu × h45/h65/h85`
- 条件：`G20`, `B40`, `B45`
- seeds：`0,1,2`
- epochs：`60`

总体 `NMI / ARI`：

1. `B45`: `0.16001 / 0.15256`
2. `B40`: `0.14575 / 0.13826`
3. `G20`: `0.11657 / 0.10914`

这轮的结论比之前更硬：

**第一，`B45` 不只是单 seed 偶然占优，而是在全量 `9-grid × 3 seeds` 上稳定领先。**

- 相对 `B40`
  - `ΔNMI = +0.01426`
  - `ΔARI = +0.01430`
- 相对 `G20`
  - `ΔNMI = +0.04343`
  - `ΔARI = +0.04342`

**第二，`B45` 的优势主要来自中高同配区间，而不是低同配区。**

按同配性汇总：

- `h45`
  - `G20`: `0.00715`
  - `B40`: `0.00676`
  - `B45`: `0.00683`
- `h65`
  - `G20`: `0.02951`
  - `B40`: `0.01914`
  - `B45`: `0.03040`
- `h85`
  - `G20`: `0.31306`
  - `B40`: `0.41135`
  - `B45`: `0.44279`

这说明：

> `B45` 不是“任何图都更强”，  
> 它真正的稳定收益区间是 **中高同配 + 边语义可被 message gate 利用** 的场景；  
> 在 `h45` 低同配区间，它并没有显著优于 `G20`。

**第三，`B45` 对三类 edge-control 都形成了整体优势。**

按控制模式汇总：

- `mhier`
  - `G20`: `0.11287`
  - `B40`: `0.14231`
  - `B45`: `0.15404`
- `mmisl`
  - `G20`: `0.12197`
  - `B40`: `0.14938`
  - `B45`: `0.16721`
- `mredu`
  - `G20`: `0.11488`
  - `B40`: `0.14555`
  - `B45`: `0.15877`

也就是说，在这批控制实验里，`B45` 已经不只是“对某一类机制有效”，而是在：

- 层级边语义
- 误导边语义
- 冗余边语义

三种模式下都能超过 `G20`。

同时，`branch_health_summary.csv` 也表明这不是死分支或统计幻觉：

- `B45`: `diag_factor_live = 1.0`
- `B45`: `diag_dual_live = 1.0`
- `B45`: `diag_msg_live = 1.0`
- `B45`: `diag_assign_live = 1.0`
- `diag_dead_branch_count = 0.0`

因此这轮之后，`B45` 的定位应更新为：

> **`B45` 已经成为当前控制实验体系下，`B30` 家族最强、且整体上超过 `G20` 的新主候选。**

#### `B47 / B48` 结构修正对照（已完成）

在确认 `B45` 为 message 主候选后，又补了两个更保守的结构修正：

- `B47 = B45 + matched-edge-only`
- `B48 = B45 + confidence gate`

结果目录：

- `results/diagnostic_b47b48_repr3_v1`

设置：

- 数据：`mredu/mmisl/mhier @ h65`
- seeds：`0,1,2`
- 对照：`G20`, `B40`, `B45`, `B47`, `B48`

总体 `NMI / ARI`：

1. `B47`: `0.03029 / 0.01637`
2. `B45`: `0.03000 / 0.01604`
3. `G20`: `0.02885 / 0.01587`
4. `B48`: `0.01961 / 0.00798`
5. `B40`: `0.01921 / 0.00773`

更细看三类机制：

- `mhier`
  - `B47`: `0.03452`
  - `B45`: `0.03446`
  - `G20`: `0.03337`
- `mmisl`
  - `B47`: `0.02272`
  - `B45`: `0.02191`
  - `G20`: `0.01887`
- `mredu`
  - `G20`: `0.03430`
  - `B45`: `0.03362`
  - `B47`: `0.03362`

因此这组对照给出两个清楚结论：

**第一，`matched-edge-only` 是一个可保留的小修正，但不是主增益来源。**

`B47` 的确略高于 `B45`，但幅度非常小：

- `ΔNMI(B47-B45) = +0.00029`
- `ΔARI(B47-B45) = +0.00033`

这说明：

> `matched-edge-only` 更像是在高强度 message gate 上做边界收缩，  
> 它可以作为稳健性修饰项，但不构成新的主方向。

**第二，confidence gate 当前版本应直接放弃。**

`B48` 基本退回到 `B40` 水平，远低于 `B45/B47`。  
原因也比较直接：当前这版 confidence gate 把 message factor 过度压向 `1`，使得强门控收益被抵消。

所以这一轮后，message 分支在这批 `repr3 @ h65` 对照上的排序应更新为：

1. `B45`
2. `B47`
3. `G20`
4. `B40`
5. `B48`

其中：

- **默认主候选仍是 `B45`**
- `B47` 保留为“可选的保守 matched-edge 修正版”
- `B48` 不再继续推进

### 必须补齐的实验清单（含默认值与预估算力）

下表给出一组“最低可发表”补实验矩阵。时间为经验估计（以单张 24GB GPU、PyTorch 2.1+、图全量训练为参考；实际需以你们的硬件与图规模校准）。

| 实验模块 | 数据集建议 | 指标 | 必要基线 | 关键超参（默认） | 必要消融 | 预估算力/时间 |
|---|---|---|---|---|---|---|
| 真实边属性验证 | 选择至少 1 个带 edge features/edge types 的图聚类设置（未指定；可用异质网络/知识图谱子图/交互网络） | NMI/ARI/ACC + modularity/conductance | 主线 G15_ECHF_main；V5；Path‑A only；Path‑B only | epochs=100（或 60/180 依规模）；hid_dim=64/256；knn=8；alpha=0.01 | edge_attr 置零/打乱；只用 generic vs 只用真实 edge_attr；edge_attr_hierarchical 开关 | 视规模：中等图（<50k边）每 run 5–20 分钟 |
| 未知 K 鲁棒性 | 现有经典集（Cora/Citeseer/Pubmed/Amazon） + 1 个异质图（Chameleon/Squirrel/Actor 等，需外部预处理） | 同上 + pred_n_clusters | LSEnet/DeSE（若可复现），或至少与 SDCN/DAEGC 比 | max_nums 扫描：如 [100,20],[50,10],[30,6]；epsInt 扫描：{2,5,8,12} | “不做 fix_cluster_results” vs 做；epsInt 对性能/结构指标影响 | 小图每配置 3 seeds × 180 epochs：总计数小时级 |
| ECHF 组件消融 | 经典图 + urban 图（仓库已有） | label 指标 + 结构指标 + 稳定性 | G15_ECHF_main 与 G17_V5_temp15 | edge_fusion_gamma schedule：start=0.2,end=1.2,sched_epochs=100 | 去掉 graph_alpha；去掉 mix_beta；V12→V5；Path‑B off | 每数据集 3 seeds：可与现有脚本复用 |
| Path‑A vs Path‑B 系统对照 | 同上 | 同上 | Path‑A only、Path‑B only、A+B | edge_attr_weight_blend ∈ {0,0.3,0.5,0.7}；apply_to ∈ {si_only,both} | edge_attr_weight_temp 扫描 {0.5,1.0,1.5} | 小图每点 1–3 分钟；点数多需做 DOE |
| 效率曲线 | 至少 1–2 个大图（>20k nodes） | wall‑clock、显存、NMI/ARI（若有标签） | dense‑KNN vs edge‑KNN | knn_mode ∈ {dense,edge,auto}；knn_auto_threshold 扫描 | k ∈ {4,8,16}；alpha ∈ {0.005,0.01,0.02} | 每 run 10–60 分钟；需记录资源 |
| 统一基线复现 | 经典图（最少 3 个） | NMI/ARI/ACC | SDCN、DAEGC、DAGC、VGAE、DGI | 统一 seeds=0/1/2；统一 early stop=off（patience=0）；统一 eval=final epoch | 同一指标计算实现（避免实现差异） | 取决于基线实现；至少 1–2 周工程量 |

> 默认复现协议建议：  
> seeds = {0,1,2}；每个设置跑 3 次；模型选择按仓库做法 *min train loss* 或在公平对比时固定“最后 epoch”；禁用早停（`patience=0`）以避免不同方法早停策略不一致导致的偏置；报告 mean±std。对比脚本中显式写出 `eval_freq=epochs` 的动机（避免“用标签挑 epoch”）。这一点与无监督论文写作一致性非常重要。

## 论文式 README 草案与实验路线图

下面给出一个“可直接替换 README.md、并可扩展为论文初稿”的结构化草案。为满足可复现性与审稿习惯，草案强调：贡献点可验证、假设透明、缺失实验明确列为 TODO。

### 标题

**ECHF：面向无监督图聚类的边校准层级融合网络——在结构熵分割树框架上的 edge‑enhanced 扩展**

### 摘要

我们研究无监督图聚类，目标是在不依赖人工标签监督的情况下发现图中的社区/簇结构。现有深图聚类方法常依赖预设聚类数 \(K\) 或两阶段“嵌入+聚类”流程；结构信息/结构熵路线通过学习层级分割树，提供了无需固定 \(K\) 的新范式。基于此，我们提出 ECHF（Edge‑Calibrated Hierarchical Fusion），在结构熵分割树网络之上引入边增强机制，使边权与边属性能够以“校准、门控、层级传播”的方式进入 parent assignment，从而在异质图上避免过强边注入导致的错误聚合。我们实现了 V5/V6/V8/V12 等一系列 edge‑fusion 变体，并提出主线 V12：以结构项为稳定 trunk，叠加与结构—属性一致性相关的校准残差；同时提出 Path‑B，把 edge_attr 沿层级 coarsen 传播至高层图。我们在多组基准上报告 NMI/ARI/ACC 与 modularity/conductance，并给出 cross‑trial 稳定性分析。结果表明：在城市异质图（known‑only）设置上，ECHF 相较结构项基线得到更一致的结构质量改善；在经典图基准上，ECHF 在鲁棒性排名上取得优势。我们进一步列出尚需补齐的真实边属性验证、unknown‑K 鲁棒性与效率曲线实验，以推动该方向形成可复现、可解释的 edge‑enhanced 结构熵聚类研究。

> 结构熵路线关键参考：LSEnet（ICML 2024）citeturn14view0turn13search1，DeSE（KDD 2025）citeturn14view1，ASIL（arXiv:2504.09970，v2 2026‑02‑02，TPAMI 接收）citeturn7view0。

### 引言

- 研究动机：真实图的边往往异质、噪声与稀疏并存；“仅用邻接”浪费关系语义，“盲目用边属性”又易过拟合。  
- 结构熵分割树范式的优势：把聚类结构作为可优化对象，与 conductance 等结构目标相关联，且可在不固定 \(K\) 的情况下学习层级划分。citeturn14view0turn7view0  
- 本文贡献：  
  1) 在 parent assignment 的 softmax logits 级别引入校准边融合（V12）；  
  2) 提出 graph_alpha 图级门控与 reliability 边级置信约束，缓解异质图过强注入；  
  3) 提出 Path‑B 层级 edge_attr 传播，使边语义跨层保持；  
  4) 给出可复现的对照脚本、阶段性汇总与结构指标评估；  
  5) 明确列出缺失实验与未来工作（真实 edge_attr、unknown‑K、效率曲线）。

### 相关工作

- 深图聚类（自训练/AE+GCN 融合）：DAEGCciteturn15search11、SDCNciteturn11view0、DAGCciteturn16view0。  
- 表示学习与聚类组合：VGAEciteturn11view2、DGIciteturn12view0 等。  
- 结构信息/结构熵路线：LSEnetciteturn14view0turn13search1、DeSEciteturn14view1、ASILciteturn7view0。  
- 边增强图建模的一般趋势：在更广义的图学习中，显式 edge channels/edge disentanglement 是近年持续方向（例如 Graph Transformer with edge channels 的思路）。citeturn12view3

### 方法

- **输入与图构造**：说明 `adj_msg/adj_si`、edge_variant 的预加权、edge_attr 的来源（真实 vs generic）、Path‑A/Path‑B 的区别。  
- **叶嵌入与 KNN augmentation**：说明 dense‑KNN 与 edge‑KNN 近似模式，给出复杂度差异，并声明默认 `knn=8, alpha=0.01`（若未指定）。  
- **LorentzAssignment 与 ECHF 融合**：形式化写出  
  \[
  score_{uv} = -d(q_u,k_v) + \gamma \cdot graph\_alpha \cdot \Psi(w_{uv}, edge\_attr_{uv})
  \]
  并分别解释 V5（结构 trunk）、V6/V7（属性门控）、V8（校准混合）、V12（trunk+残差）。  
- **层级 edge_attr 传播（Path‑B）**：说明从叶层到父层的超边属性聚合/加权方式。  
- **目标函数**：给出层级结构熵（SI loss）的计算形式，并解释为何以 *min train loss* 做模型选择能够避免标签泄漏（或规定对照时固定最后 epoch）。

### 实验

- **数据集**：  
  - 经典 PyG 基准：Cora/Citeseer/Pubmed/Amazon（Computers/Photo）等（仓库自动下载）。  
  - 城市异质图（urban\_*）：需在论文补充来源与预处理（当前未指定）。  
  - **必须新增**：至少 1 个“真实 edge_attr/edge type”数据集（TODO）。  
- **评估**：NMI/ARI/ACC（标签对齐）、modularity/conductance（结构）、trial‑stability（pairwise NMI），并注明 known‑only 评估策略（将未知标签映射为 -1）。  
- **基线**：V5 vs V12；Path‑A only vs Path‑B only；并补充外部基线（SDCN/DAEGC/DAGC/VGAE/DGI/LSEnet/DeSE/ASIL，尽可能统一协议）。  
- **实现细节**（默认建议，若未指定）：  
  - optimizer=AdamW(lr=1e‑3, wd=1e‑2)；  
  - epochs：经典集 180；urban 60；  
  - hid_dim：经典集 256；urban 64；  
  - early stop：对照实验关闭（patience=0）。  
- **消融**：graph_alpha、mix_beta、edge_confidence_quantile、edge_attr_hierarchical、edge_attr_weight_blend 等。

### 结果与讨论

- 报告 mean±std；同时报告结构指标是否与 label 指标一致改善（尤其在异质图上）。  
- 讨论何时 V12 会退化为 V5（当结构—属性 agreement 低、mix_beta 小）；何时属性残差起主要作用。  
- 讨论 edge‑KNN 近似模式的潜在偏差，并用效率曲线佐证。

### 结论

总结 ECHF 在结构熵分割树范式上引入边校准层级融合的价值，并强调未来必须补齐的真实边属性、unknown‑K 与效率实证。

### 推荐引用列表（优先一次文献）

- LSEnet（ICML 2024 / arXiv:2405.11801）citeturn13search1turn14view0  
- DeSE（KDD 2025 / arXiv:2505.14040）citeturn14view1  
- ASIL（arXiv:2504.09970，v2: 2026‑02‑02，TPAMI 接收）citeturn7view0  
- SDCN（arXiv:2002.01633，WWW 2020）citeturn11view0  
- DAEGC（IJCAI 2019）citeturn15search11  
- DAGC（arXiv:2111.05548，TCSVT 接收）citeturn16view0  
- DGI（arXiv:1809.10341，ICLR 2019）citeturn12view0  
- VGAE（arXiv:1611.07308）citeturn11view2  

### 优先级检查清单

高优先级（决定论文能否立得住）  
- 真实 edge_attr 数据集验证（至少 1 个）  
- unknown‑K/`max_nums`/`epsInt` 鲁棒性与机制解释  
- 외部强基线统一协议复现（至少覆盖 SDCN、DAEGC、DAGC、VGAE、DGI、LSEnet、DeSE；ASIL 若实现可用则加分）citeturn11view0turn15search11turn16view0turn11view2turn12view0turn14view0turn14view1turn7view0  
- dense‑KNN vs edge‑KNN 的效率—精度曲线

中优先级（增强说服力与可解释性）  
- Path‑A vs Path‑B 系统对照（含 edge_attr 打乱/置零控制实验）  
- graph_alpha 与 mix_beta 的可视化统计（跨数据集分布）  
- 结构指标与 label 指标一致性分析（相关性、反例）

低优先级（锦上添花）  
- 在更多异质图/噪声图上做鲁棒性 stress test  
- 进一步的理论补充（例如为何 V12 校准项应当与 conductance/结构熵目标一致）

### 实验时间线（Mermaid）

```mermaid
gantt
  title ECHF 论文补实验时间线（建议）
  dateFormat  YYYY-MM-DD
  axisFormat  %m/%d

  section 复现实验整理
  固化复现协议（seeds/epochs/selection）      :a1, 2026-02-20, 3d
  现有脚本跑通并生成统一汇总表               :a2, after a1, 4d

  section 关键缺口补齐
  真实 edge_attr 数据集接入与清洗（TODO）    :b1, after a2, 10d
  unknown-K 鲁棒性实验（max_nums/epsInt 扫描）:b2, after a2, 7d
  dense-KNN vs edge-KNN 效率曲线              :b3, after a2, 5d

  section 外部基线复现
  SDCN/DAEGC/DAGC 统一协议复现               :c1, after a2, 14d
  VGAE/DGI + 聚类头复现                       :c2, after a2, 7d
  LSEnet/DeSE/ASIL 对齐复现实验（若可行）      :c3, after a2, 14d

  section 论文产出
  方法与消融写作（含图表）                     :d1, after b2, 10d
  结果表与附录（超参/复杂度/统计显著性）        :d2, after c1, 10d
  定稿与开源复现脚本清理                        :d3, after d2, 5d
```
