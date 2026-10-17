# 紧凑证据快照

文档核对：2026-10-17。CSV 保留既有实验结果，本次只更新说明。首批快照于 2026-10-15 从本地
`results/breakthrough_v1/` 导出，后续扩展至 E22；最后结果提交为 `8b3f0d5`（2026-10-16）。
这些小表便于直接在 GitHub 核查结论；完整日志、检查点和原始数据仍是本地忽略资产，见
[DATA_ASSETS.md](../../../DATA_ASSETS.md)。

[根 README](../../../README.md) 给出主要结论，[研究详述](../../../reading/BREAKTHROUGH_E1_E22.md) 保留研究过程，
[计划索引](../README.md) 给出生成命令及 E14–E22 的依赖顺序。下表行数均不含表头。

## 文件与来源

| 文件 | 行数 | 来源与协议 |
|---|---:|---|
| [real_selected.csv](real_selected.csv) | 104 | `combined_real/dual_start_detail.csv` 的 chosen 行；合并 E3/E6/E8/E9/E11，五种真实图、九种模型、baseline-guided starts |
| [synth_mechanism.csv](synth_mechanism.csv) | 18 | `report_e2_mechanism.py` 合并 E2/E6/E7/E9/E11；`synth_ctx_v1` 架构/描述/直接/上下文效果 |
| [xor_context_learned.csv](xor_context_learned.csv) | 14 | E13 `summary.csv`；`synth_ctxxor_v1` 的学习边头与 edge-free baseline |
| [se_search_plain.csv](se_search_plain.csv) | 30 | E12/E13 plain-spectral 搜索汇总：真实 seed 0、AIFB 1–4、合成 0–2、XOR 一/二阶，以及 RTX 4090 Photo 重跑 |
| [se_search_reg_beam.csv](se_search_reg_beam.csv) | 19 | E15 正则化谱与 width-8 order-2 beam：真实图 seeds 0–2，合成 seed 0 且启用 `--order2` |
| [spectral_stability.csv](spectral_stability.csv) | 6 | E15 uniform 图上，plain / regularized spectral 在五个 eigensolver 起点下的 NMI |
| [search_then_train.csv](search_then_train.csv) | 84 | E14 plain 与 E16 reg 两种解码协议，各 42 行；保留 tag 和训练损失用于选启动 |
| [selection_rules_summary.csv](selection_rules_summary.csv) | 30 | E18 各规则/决策集汇总；`cv-subset` 为有 edge-CV 重训的 seed-0 子集 |
| [selection_rules_instances.csv](selection_rules_instances.csv) | 42 | E18 逐决策 picks/NMI、候选最优值、feature-gate 统计 |
| [selection_validation_summary.csv](selection_validation_summary.csv) | 30 | E19 十个新决策的规则汇总；规则冻结于 `876486b` |
| [selection_validation_instances.csv](selection_validation_instances.csv) | 10 | E19 逐决策 picks/NMI、候选最优值、feature-gate 统计 |
| [few_label_selection_summary.csv](few_label_selection_summary.csv) | 35 | E20 按预算和分组汇总；SE/edge-CV 列仅覆盖各自可评分的决策 |
| [few_label_selection_instances.csv](few_label_selection_instances.csv) | 260 | E20，52 决策 × 5 预算；每项 200 次标签抽样的 NMI/ARI regret 与尾部风险 |
| [few_label_edge_methods.csv](few_label_edge_methods.csv) | 162 | E21 methods 0–2，逐 dataset/budget/draw 的留出节点 NMI、标签边计数与回退标志 |
| [few_label_dse_runs.csv](few_label_dse_runs.csv) | 88 | E21 method 3，`basesup` / `b31ctxsup` 训练账本；tag 编码标签预算、draw、启动 |
| [route_selection_summary.csv](route_selection_summary.csv) | 15 | E22，5 预算 × all/synthetic/real；各路线、自动选择、per-draw 较优路线与 regret |
| [route_selection_draws.csv](route_selection_draws.csv) | 270 | E22 逐 dataset/budget/draw 的路线 NMI、CV accuracies、最终选择 |
| [machine_check.csv](machine_check.csv) | 2 | E9/E9x 匹配 seed-0 `b31fr_warm` 行；两数据集，单独比较，不能作为额外训练种子并入主表 |
| [se_alignment.csv](se_alignment.csv) | 82 | `e10_se_alignment/se_alignment.csv`；有限候选权重下使用标签的目标一致性诊断 |
| [b45_gate_interventions.csv](b45_gate_interventions.csv) | 6 | `b45_gate_diagnosis/summary.json` 最终 SE-epoch 行；Photo seed 0，200 warm + 200 SE epochs，最终解码 seed 999 |
| [b45_graph_cross.csv](b45_graph_cross.csv) | 12 | 同一诊断摘要中的固定检查点图交叉；叶层 argmax 分配，不同于完整解码干预 |
| [b45_graph_stats.csv](b45_graph_stats.csv) | 3 | 同一摘要中的图统计；归一化权重占比按门控层分别计算 |

## 读取时的协议边界

`real_selected.csv` 每行是一个训练种子，NMI 已在该次运行的 3 次解码内取均值。文中 `±` 是这些种子均值的总体标准差。
多数模型有 3 个训练种子；B31+ctx、B31+ctx msg_only、B31 msg_only 和 B45 在 Photo/AIFB 只有 1 个。
启动取匹配 edge-free baseline pair 较低的 `best_train_loss_mean`，没有用标签指标挑启动。
类别数和层次预算是基准输入。`oracle_*` 列是使用标签计算的参考上限，不能当成无监督选择结果。

`search_then_train.csv` 的 84 行不是 84 次独立训练：E14 完成 42 runs，E16 新训练 12 runs，
正则化解码协议另外复用了 E14 的 30 runs。应先按 `decoder_protocol` 分组，再按 dataset/seed 选较低
`best_train_loss` 的 warm/cold（合成图只有一种启动）。DSE 分数与 SE-selected decoder 分数应分列报告。

E18 有 42 个决策，只有 14 个具备 edge-CV；E19 的 10 个新决策全部具备 edge-CV。
R4 `gated_cv` 是看到 E18 结果后提出的，其 E18 得分不是独立验证，E19 才检验新图上的泛化。
E20 的 SE 对照覆盖 52 个决策，edge-CV 覆盖 24 个；比较 regret 必须使用相同决策子集。

E20 冻结模型与候选，在抽取标签节点上按 ARI 选择，并在全部已知标签节点上计算 NMI/ARI regret。
E21/E22 则将抽出的标签用于训练或选择，只在其余已知标签节点评估。它们都排除原始未知标签，
但不能直接将 E20 的 regret 当作 E21/E22 的留出泛化分数。

E21 methods 0–2 有 3 次 label draw，预算为 10/20/50/100/200；AIFB 只有 176 个已知标签，因此跳过 200。
method 3 只有 2 次 draw，预算为 20/200（AIFB 高预算为 150）。主文可比表使用共有预算和 draws 0/1，
不要把 CPU 与 DSE 的全部行分别平均。88 次训练已完成，5 次 OOM 后降到 2 并发重跑；`cached` 表示本地 runner 复用已有 metrics，
不是额外训练或缺失结果。

E22 的 10/20/50/100 标签汇总各为 11 图 × 5 draws；200 标签只有 10 图 × 5 draws，AIFB 缺席，
所以跨预算均值也包含图组成变化。`auto` 按标签集内部 CV accuracy 选路线，平局选结构；
`oracle` 是两路线每个 draw 中 NMI 较优者，只用于 regret。

图交叉探针每个数据集只有一个检查点，不是完整训练种子分布。Photo `safe_isolates` 为同一随机种子/初始化协议下的独立配对运行；
DBLP 配对训练因资源争用中断，不纳入干预表。门控重归一化修复后来以 `--no-edge_msg_gate_renorm` 在 E11 全预算验证，
其他 hooks 仍只用于诊断。E9 的 48 个合成 runs 来自 RTX 4060 Laptop；E14/E16 来自 RTX 4090。

## 更新证据

本次没有重导出或改写数值。结果变更时，先按 [计划索引](../README.md) 恢复源账本、数据和检查点，
用真实图报告的 `chosen=True` 行更新五图表；其他表从上列源 CSV/JSON 更新，保留阶段、解码和选择协议区别。
核对行数、数据集、种子、预算与标签边界后，再记录新导出日期和来源 revision。

历史导出链：初版为 `8d77f9e` 加 E9 `msg_only`；E9/E11–E16 后续快照更新于 2026-10-15；
E17/E18 为 `876486b`，E19 为 `9551bc8`，E20 为 `36245fd`，E21 为 `ea1cb9e`，E22 为 `8b3f0d5`。
