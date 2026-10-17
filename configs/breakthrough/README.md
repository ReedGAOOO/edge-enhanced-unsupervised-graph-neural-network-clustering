# E1–E22 实验计划与复现

更新：2026-10-17。项目结论见 [根 README](../../README.md)，完整研究过程见
[BREAKTHROUGH_E1_E22.md](../../reading/BREAKTHROUGH_E1_E22.md)，Git 内证据见 [evidence/README.md](evidence/README.md)。
以下完成数描述已记录的历史运行，本次文档更新未重新训练。完整日志、数据和检查点不纳入 Git。

## 计划与工具清单

| 文件 / 阶段 | 问题与范围 | 已记录状态 |
|---|---|---|
| [e1_backbone_gate.json](e1_backbone_gate.json) | 加长预算是否让 backbone 学到结构？含历史合成图 | gate 运行已记录 |
| [e2d_plateau_diag.json](e2d_plateau_diag.json) | 弱特征冷启动为何停滞？ | 诊断参数扫描 |
| [e2w_warm_smoke.json](e2w_warm_smoke.json) | 伪分区 warm start 能否脱离平台？ | 冒烟检查 |
| [e2_synth_ctx_warm.json](e2_synth_ctx_warm.json) | `synth_ctx_v1` 的边语义与架构效果 | 主要 E2 证据；EP2 崩塌后提前停止 |
| [e3_real_dual.json](e3_real_dual.json) | 五种真实图的 warm/cold starts | 原 E3 表按各模型自身 SE 损失选启动 |
| [e4_b45_collapse_ablation.json](e4_b45_collapse_ablation.json) | 哪个 B45 组件导致崩塌？ | Photo/AIFB 组件消融 |
| [e6_b31_robust.json](e6_b31_robust.json) | 去掉门控能否保留收益且避免崩塌？ | B31 合成/真实图，含 ACM |
| [e7_ctx_edge.json](e7_ctx_edge.json) | 端点乘积能否解读上下文边？ | 合成上下文对照 |
| [e8_ctx_real.json](e8_ctx_real.json) | B31 + context 是否改善真实图？ | 真实图跟进 |
| [e9_fixed_ruler.json](e9_fixed_ruler.json) | 只让学习标量作用于消息路径 | 92/92；其中 48 个合成任务在 RTX 4060 Laptop 完成 |
| [e9x_machine_check.json](e9x_machine_check.json) | 两个 seed-0 运行跨 GPU 漂移多少？ | 2/2；单独报告 |
| [e11_safe_gate_pathways.json](e11_safe_gate_pathways.json) | 去掉门控重归一化；分解 B31 路径 | 57/57 |
| [e13_xor_context.json](e13_xor_context.json) | 学习边头能否解读 XOR 二阶上下文？ | 42/42 |
| [e14_search_then_train.json](e14_search_then_train.json) | 在 plain-spectral 搜索 pick 上训练 edge-free DSE | 42/42，RTX 4090 |
| [e16_search_then_train_reg.json](e16_search_then_train_reg.json) | 正则化搜索改变的 Cora/AIFB picks 重训 | 12/12，其他图复用 E14 |
| [e17_edge_cv.json](e17_edge_cv.json) | 去掉 20% 边后重训选择基准的 seed-0 候选 | 29/29 |
| [selection_rules_spec.json](selection_rules_spec.json) | E18 的 42 个决策，交给 `select_partition_rules.py` | 已评分；14 个具备 edge-CV |
| [e19a_selection_validation.json](e19a_selection_validation.json)、[e19b_selection_validation.json](e19b_selection_validation.json) | E19 驱动生成的两阶段训练计划 | 20/20、24/24 |
| [selection_validation_spec.json](selection_validation_spec.json) | E19 的十个新决策，规则冻结于 `876486b` | 已评分 |
| E20：`few_label_selection.py` | E18/E19 冻结候选中的少标签选择 | 52 决策 × 5 预算，每项 200 次抽样 |
| [e21_few_label_dse.json](e21_few_label_dse.json) | method 3：监督 pair loss 的 baseline vs 另加监督边头的 B31+ctx | 88/88，RTX 4090；OOM 后 5 项降至 2 并发重跑 |
| [e21_few_label_edges_jobs.csv](e21_few_label_edges_jobs.csv) | E21 methods 0–2 和 E22 所用 dataset / 候选 CSV / ctx_dim | 11 图；CSV 清单，不是训练 plan |
| E22：`few_label_route_selection.py` | 少量标签内部 CV 选择结构或特征路线 | 270 个 dataset/budget/draw 行，15 个汇总行 |

E5 headroom、E10 SE alignment、B45 门控诊断、E12 搜索、E15 谱稳定性/正则化/beam 搜索均由独立工具产生，
不对应缺失的编号 JSON。E20/E22 同样由 CPU/评分工具驱动。

## 运行前提与缓存

先按 [环境声明](../../UV_ENVIRONMENTS.txt) 建立 `.venv`，按 [data/README.md](../../data/README.md) 准备所选任务的数据。
命令从仓库根目录执行。下文统一使用 `data/`，因为 E21/E22 的 CPU 工具固定读取该目录，没有 `--root_path`。

```bash
# 当前合成基准；参数与脚本默认值不同，不能省略
.venv/bin/python tools/prepare_synth_ctx_datasets.py --root_path data \
  --feature_sep 0.35 --homophily 0.45,0.55
.venv/bin/python tools/prepare_synth_ctx_datasets.py --root_path data \
  --prefix synth_ctxxor_v1 --role_mode xor2 --feature_sep 0.35 \
  --homophily 0.45,0.55 --modes none,contextual --perm_modes contextual

# 预览 E9 的 44 个真实图任务，不训练、不写结果
.venv/bin/python tools/run_breakthrough.py --plan configs/breakthrough/e9_fixed_ruler.json --root_path data \
  --only_tags b31fr_warm,b31fr_nowarm,b31frctx_warm,b31frctx_nowarm --dry_run
```

批量运行器按 preset → plan defaults → job overrides 合并，输出位于
`results/breakthrough_v1/<plan-name>/`：`plan.json`、`runs.csv`、`summary.csv` 以及各 run 的 `stdout.txt`、日志、metrics JSON。
`stdout.txt` 首行为精确命令；主代码的检查点位于 `checkpoints/breakthrough_v1/...`。

- 已有 `<dataset>_metrics.json` 就视为缓存，不校验代码/参数版本；修改任一项后使用新 plan `name` 或 `--force`。
- 筛选 tags 或 shards 的正常运行会重写该 plan 的 `runs.csv` / `summary.csv`，只包含本次选中任务；先保留完整账本。
  `--dry_run` 不写文件，也会显示缓存任务的命令。
- `--max_parallel` 是当前 GPU 的并发进程数；较大图先从 2–4 开始，E19/E21 建议 2。`--threads` 限制每进程 CPU 线程。
- 跨机器分片需在每台机器显式使用相同 `--num_shards N` 和各自 `--shards i,j`。
  分片按筛选后任务索引 `% N` 分配，所有机器应使用相同 tags。运行器不负责连接远程机器。
  汇回各 run 后，全量运行可从缓存重建账本。
- E2 的 EP1/EP2 需要 `results/breakthrough_v1/_code/` 下带 warm-start hook 的本地代码副本；克隆不含这些忽略文件，
  archive 中原始 workspace 也不是同一实现。只复现根代码可用 `--only_tags baseline,b45`。
- E1/E2d 引用历史合成图，需用 `tools/prepare_mechanism_synth_datasets.py` 另外重建；它们不是当前机制基准。

## 复现顺序

按下列顺序恢复依赖：当前数据 → E2/E3/E6–E13 训练与诊断 → E12/E15 搜索 → 写出 picks → E14/E16 →
E17/E18 → E19 → E20。E21/E22 的 CPU 路线在 E15 和 direct 搜索完成后即可运行，无需 E21 DSE 检查点。

### 1. 训练和机制对照（E1–E13）

准备好各 plan 命名的真实数据后，当前主线可分别启动：

```bash
# 根代码 E2；EP1/EP2 需先恢复上文所述本地副本，才可省略 --only_tags
.venv/bin/python tools/run_breakthrough.py --plan configs/breakthrough/e2_synth_ctx_warm.json \
  --root_path data --only_tags baseline,b45 --max_parallel 2
for plan in e3_real_dual e4_b45_collapse_ablation e6_b31_robust e7_ctx_edge e8_ctx_real \
            e9_fixed_ruler e11_safe_gate_pathways e13_xor_context; do
  .venv/bin/python tools/run_breakthrough.py --plan "configs/breakthrough/$plan.json" \
    --root_path data --max_parallel 2
done

# E5 / E10 是使用标签的诊断，不是公平的无监督模型选择
.venv/bin/python tools/screen_edge_headroom.py --root_path data \
  --datasets cora,photo,dblp_magnn_author_v2,acm_multiplex_v2 \
  --out_csv results/breakthrough_v1/e5_headroom/headroom_screen.csv
.venv/bin/python tools/check_se_alignment.py --root_path data \
  --datasets cora,photo,dblp_magnn_author_v2,acm_multiplex_v2 \
  --out_csv results/breakthrough_v1/e10_se_alignment/se_alignment.csv
```

历史 gate/smoke 或机器检查使用相同运行器，替换为 `e1_backbone_gate.json`、`e2d_plateau_diag.json`、
`e2w_warm_smoke.json` 或 `e9x_machine_check.json`。E9x 仅两项 warm runs，应与主表分开。
门控检查点干预和共同初始化短训练见 [tools/README.md](../../tools/README.md)。

### 2. 显式搜索（E12/E13/E15）

E15 候选同时供 E21/E22 使用，保留以下目录名和 `Cora,Photo` 大小写（候选 CSV 文件名区分大小写）：

```bash
.venv/bin/python tools/run_se_guided_baseline.py --root_path data \
  --datasets Cora,Photo,dblp_magnn_author_v2,entities_aifb,acm_multiplex_v2 \
  --seeds 0,1,2 --spectral reg --beam 8 --max_order 2 --workers 8 \
  --out_dir results/breakthrough_v1/e15_search_robustness/real_reg_beam
.venv/bin/python tools/run_se_guided_baseline.py --root_path data \
  --datasets synth_ctx_v1_h45_contextual_ds00,synth_ctx_v1_h55_contextual_ds00,synth_ctxxor_v1_h45_contextual_ds00,synth_ctxxor_v1_h55_contextual_ds00 \
  --seeds 0 --spectral reg --order2 --beam 8 --max_order 2 --workers 8 \
  --out_dir results/breakthrough_v1/e15_search_robustness/synth_reg_beam
.venv/bin/python tools/check_spectral_stability.py --root_path data \
  --datasets Cora,Photo,dblp_magnn_author_v2,entities_aifb,acm_multiplex_v2 \
  --out_csv results/breakthrough_v1/e15_search_robustness/spectral_stability.csv
```

复现 E12 时，以 `--spectral plain` 在真实图 seed 0、direct/contextual 合成图 seeds 0,1,2 上搜索；
AIFB 另跑 seeds 1,2,3,4。E13 XOR 比较一阶与 `--order2`（二阶记录为 seed 0）。
`--ctx_dim` 默认 32，`synth_ctx*` 使用 `--synth_ctx_dim 64`；写出图时必须保持相同宽度。
`ctx_pick` / `ctx2_pick` 中 `/method` 之前的部分是 weighting 名，按 SE 决定；oracle 列只是标签评估上限。

### 3. 写出 picks，再训练与比较解码器（E14/E16）

以下表固定了历史实验所用输入；新的搜索若得到不同 pick，应另命名实验以保留可比性。

| out_name | source dataset | pick | ctx_dim | 来源 |
|---|---|---|---|---|
| `cora_sepick` | `Cora` | `-logdeg_gap*ctx23` | 32 | E12 plain |
| `cora_sepick_reg` | `Cora` | `+logdeg_gap` | 32 | E15 reg |
| `photo_sepick` | `Photo` | `+logdeg_dst` | 32 | RTX 4090 重跑；本机 E12 为 `+l2*ctx1`；E15 同意重跑 pick |
| `dblp_sepick` | `dblp_magnn_author_v2` | `+apcpa_conference_cosine` | 32 | E12/E15 |
| `aifb_sepick` | `entities_aifb` | `+attr0*ctx5` | 32 | E12 plain |
| `aifb_sepick_reg` | `entities_aifb` | `+attr1*ctx2` | 32 | E15 reg 的 ctx pick；noctx pick `+attr1` 的 NMI 更高 |
| `acm_sepick` | `acm_multiplex_v2` | `layer+PAP` | 32 | E12/E15 |
| `synthctx_h{45,55}_sepick` | `synth_ctx_v1_h{45,55}_contextual_ds00` | `+attr0*ctx32` | 64 | E12/E15 |
| `synthxor_h{45,55}_sepick` | `synth_ctxxor_v1_h{45,55}_contextual_ds00` | `+attr0*ctx32*ctx33` | 64 | E13/E15 order2 |

用每行参数写出图（`{45,55}` 表示分别展开 h45/h55 两行），例如：

```bash
.venv/bin/python tools/write_se_reweighted_dataset.py --root_path data \
  --dataset Cora --pick='-logdeg_gap*ctx23' --ctx_dim 32 --out_name cora_sepick
.venv/bin/python tools/write_se_reweighted_dataset.py --root_path data \
  --dataset Cora --pick='+logdeg_gap' --ctx_dim 32 --out_name cora_sepick_reg
.venv/bin/python tools/write_se_reweighted_dataset.py --root_path data \
  --dataset synth_ctxxor_v1_h45_contextual_ds00 --pick='+attr0*ctx32*ctx33' --ctx_dim 64 --out_name synthxor_h45_sepick

# 完成表中全部数据副本后运行
.venv/bin/python tools/run_breakthrough.py --plan configs/breakthrough/e14_search_then_train.json --root_path data --max_parallel 2
.venv/bin/python tools/run_breakthrough.py --plan configs/breakthrough/e16_search_then_train_reg.json --root_path data --max_parallel 2
.venv/bin/python tools/se_select_decoder.py --spectral plain \
  --run_dirs results/breakthrough_v1/e14_search_then_train/runs/*/ \
  --out_csv results/breakthrough_v1/e14_search_then_train/decoder_select.csv
.venv/bin/python tools/se_select_decoder.py --spectral reg \
  --run_dirs results/breakthrough_v1/e14_search_then_train/runs/*/ results/breakthrough_v1/e16_search_then_train_reg/runs/*/ \
  --out_csv results/breakthrough_v1/e16_search_then_train_reg/decoder_select_reg.csv
```

E16 的汇总应选新训练的 Cora/AIFB，并复用 E14 的 Photo/DBLP/ACM/合成图；不能把旧 Cora/AIFB 重复算入。
`se_select_decoder.py` 从源 `stdout.txt` 读取数据路径，迁移机器后应先确认该路径仍有效；`--cpu` 可用 CPU 解码。
Cora/Photo 连通分量数超过簇数，plain spectral 的简并子空间随 eigensolver 起点改变，跨机器 pick 可能不同。

### 4. 无标签选择规则与新图验证（E17–E19）

```bash
.venv/bin/python tools/make_edge_cv_split.py --root_path data --frac 0.2 --seed 0 --suffix _cv80 \
  --datasets cora,photo,dblp_magnn_author_v2,entities_aifb,acm_multiplex_v2,cora_sepick_reg,photo_sepick,dblp_sepick,aifb_sepick_reg,acm_sepick,synthctx_h45_sepick,synthctx_h55_sepick,synthxor_h45_sepick,synthxor_h55_sepick
.venv/bin/python tools/run_breakthrough.py --plan configs/breakthrough/e17_edge_cv.json --root_path data --max_parallel 2
.venv/bin/python tools/select_partition_rules.py --root_path data \
  --spec configs/breakthrough/selection_rules_spec.json --out_dir results/breakthrough_v1/e18_selection_rules

# E19 依次生成 seed-1 合成图、搜索、重加权图、CV 图、两阶段训练和规则结果
.venv/bin/python tools/run_selection_validation.py --root_path data --max_parallel 2 --threads 4 --workers 8
```

E18 spec 引用前面各阶段的本地运行和检查点；E17 只补 CV 运行，不代替 full-graph 训练。
held-out 比例目标为 20%，为避免新孤立点可能略少。R0–R3 在评估前定义；R4 `gated_cv` 在 E18 后定义，
应以 E19 的新决策检验，不能把 E18 的分数当独立验证。
E19 驱动按文件存在跳过部分准备步骤，会重写 `e19a` / `e19b` plans 与 `selection_validation_spec.json`；
恢复旧输出时应核对其来源，修改实验后不要混用旧缓存。

### 5. 冻结候选中的少标签选择（E20）

```bash
.venv/bin/python tools/few_label_selection.py --root_path data \
  --specs configs/breakthrough/selection_rules_spec.json configs/breakthrough/selection_validation_spec.json \
  --reference results/breakthrough_v1/e18_selection_rules/instances.csv results/breakthrough_v1/e19_selection_validation/rules/instances.csv \
  --budgets 5,10,20,50,100 --draws 200 --out_dir results/breakthrough_v1/e20_few_label_selection
```

候选模型不使用这些标签训练；每次从已知标签节点均匀抽 m 个，用子集 ARI 选择已有分区，平局取候选均值。
NMI/ARI regret 的 oracle 与评估使用全部已知标签节点，属于选择诊断；不能等同于 E21/E22 的留出节点结果。
模型分区只恢复一次，缓存在输出目录 `partitions/`，修改图、候选或代码时需换目录或清理相应缓存。
SE 对照覆盖 52 个决策，edge-CV 只覆盖其中 24 个，比较时使用相同决策子集。

### 6. 少标签边语义与路线选择（E21/E22）

先完成第 2 步的两个 E15 搜索，再生成 jobs CSV 依赖的 direct 搜索：

```bash
.venv/bin/python tools/run_se_guided_baseline.py --root_path data \
  --datasets synth_ctx_v1_h45_direct_ds00,synth_ctx_v1_h55_direct_ds00 --seeds 0 \
  --spectral reg --workers 8 --out_dir results/breakthrough_v1/e21_few_label_edges/search_direct
.venv/bin/python tools/few_label_edge_semantics.py --jobs configs/breakthrough/e21_few_label_edges_jobs.csv \
  --budgets 10,20,50,100,200 --draws 0,1,2 --workers 6 \
  --out_csv results/breakthrough_v1/e21_few_label_edges/methods012.csv
.venv/bin/python tools/run_breakthrough.py --plan configs/breakthrough/e21_few_label_dse.json \
  --root_path data --max_parallel 2
.venv/bin/python tools/few_label_route_selection.py --jobs configs/breakthrough/e21_few_label_edges_jobs.csv \
  --budgets 10,20,50,100,200 --draws 0,1,2,3,4 --workers 6 \
  --out_dir results/breakthrough_v1/e22_route_selection
```

E21/E22 使用 `data.choose_labeled_nodes` 相同随机种子抽样，只在其余已知标签节点上评估 NMI；
原始未知标签节点不进入评估。methods 0–2 是半监督节点分类、SE top-10 + ARI 选择、监督边分类器。
method 3 训练带监督损失的 DSE：`basesup` 只有 pair loss，`b31ctxsup` 另有 context 和 edge-head loss，默认权重均为 1.0。
计划仅训练 budgets 20/200、draws 0/1，AIFB 高预算改为 150；CPU 表有 3 次 draw，不能将两者全量平均后直接比较。

E22 在抽取标签内部做最多 5 折 CV，以验证 accuracy 选择路线（平局选结构），再用全部 m 标签拟合选择后的路线；
结构路线的簇到类别映射只在各训练折拟合，未见标签的簇回退到训练折多数类。
`oracle_nmi` 为每个 draw 的两路线中较优者，仅用于 regret。AIFB 只有 176 个已知标签，CPU 工具跳过 budget 200：
因此 10/20/50/100 标签各为 11 图 × 5 draws，200 标签为 10 图 × 5 draws；不同预算均值的图组成不完全相同。

## 报告与解释口径

```bash
# 经典对照：报告脚本需要这些 CSV
.venv/bin/python tools/run_classic_baselines.py --root_path data \
  --datasets cora,photo,dblp_magnn_author_v2,entities_aifb,urban_paris_plot_v3bsjg \
  --out_csv results/breakthrough_v1/e3_real_dual/classic_baselines.csv
.venv/bin/python tools/run_classic_baselines.py --root_path data --datasets acm_multiplex_v2 \
  --out_csv results/breakthrough_v1/e6_b31_robust/classic_acm_v2.csv
.venv/bin/python tools/report_e3_real.py \
  --extra_dirs results/breakthrough_v1/e6_b31_robust,results/breakthrough_v1/e8_ctx_real,results/breakthrough_v1/e9_fixed_ruler,results/breakthrough_v1/e11_safe_gate_pathways \
  --extra_classic results/breakthrough_v1/e6_b31_robust/classic_acm_v2.csv \
  --datasets cora,photo,dblp_magnn_author_v2,entities_aifb,acm_multiplex_v2 \
  --out_dir results/breakthrough_v1/combined_real
.venv/bin/python tools/report_e2_mechanism.py \
  --extra_dirs results/breakthrough_v1/e6_b31_robust,results/breakthrough_v1/e7_ctx_edge,results/breakthrough_v1/e9_fixed_ruler,results/breakthrough_v1/e11_safe_gate_pathways
```

真实图经典对照的命令包含历史 Urban 图；若仅恢复五图报告且没有 Urban 数据，可从 `--datasets` 删除该项。
机制报告可另提供 `--classic_csv`，默认读取 E2 目录的 `classic_baselines.csv`；缺少该文件时经典/oracle 行为空。
`--out_dir` 只改变真实图报告的输出目录，不改变输入 CSV 位置。

- checkpoint 按训练损失选择；`oracle_best_nmi` 按标签选择评估 epoch，仅作参考，不能当无标签 checkpoint 分数。
- 合并真实图表用 edge-free baseline 较低的 `best_train_loss_mean` 选各 dataset/seed 的 warm/cold，然后应用到所有模型。
  baseline pair 不全时报告会回退到模型自身损失，需检查 `dual_start_detail.csv`。
- 类别数/层次预算是基准输入，`max_nums` 不保证最终簇数。E9 `msg_only` 跳过学习标量的 SE 加权，
  仍保留 augmentation、基础边权、分配残差和边正则；其 SE 图不是冻结原始邻接。
- E14/E16 应分别报告 DSE 列与 SE-selected decoder 列。搜索后训练、解码器选择、跨模型选择是不同决策。
  search-then-train 表按 `best_train_loss` 选同图同 seed 的启动；E16 正则化协议会复用部分 E14 训练。
- Git 内紧凑表足以核查既有结论，重新生成模型分区仍需要原数据、配置、检查点和各阶段账本。
