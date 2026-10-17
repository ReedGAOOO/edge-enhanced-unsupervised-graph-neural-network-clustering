# Breakthrough experiments E1–E22

文档整理日期：2026-10-17。本页保留原项目首页的英文实验叙述、结果表及限制，覆盖 2026 年 10 月的 E1–E22；整理文档不代表重新运行实验。

- 项目入口、环境和快速开始：[根 README](../README.md)。
- 按依赖顺序复现：[实验计划与命令](../configs/breakthrough/README.md)。
- 可核查结果：[已跟踪的证据 CSV](../configs/breakthrough/evidence/README.md)。

`results/...`、`checkpoints/...` 是本地生成路径，不随 Git 分发。下文的 historical / retired 表述记录当时的实验边界。
E20 在全部已知标签上评估（包含选择样本）；E21/E22 在未用于监督的已知标签节点上评估，二者不能直接横向比较。
下文提到的仓库路径和脚本命令均以仓库根目录为基准。

## Breakthrough pass (2026-10-13)

Everything before this pass was measured on a backbone that had not been trained. The pass fixed the backbone,
rebuilt the synthetic benchmark, re-ran the surviving models and retired the rest. Results live in
`results/breakthrough_v1/` (plans in `configs/breakthrough/`); the pre-pass README is archived at
`archive/docs/README_2026-04_snapshot.md` and its numbers should not be cited.

### What was wrong

1. **The backbone was never trained.** All earlier runs used 60-120 epochs and `hid_dim` 128-256 against a reference
   config of 2000 epochs / 512. Training loss moved <1%, and partitions were close to random (Urban conductance 0.98,
   modularity 0.005). B45's "wins" were mostly one lucky seed escaping a degenerate start.
2. **The synthetic suite could not test edges.** On `synth_mech_full_v1` / `synth_edgectrl_v1`, KMeans on node features
   alone gives NMI 1.000 on every graph, and edge channel 0 is a noisy copy of the pairwise same-label bit.
3. **Urban labels were scored with `unknown` as a class** (84% of plots in WashingtonDC/Boston). With known-only labels,
   every classic method scores NMI 0.02-0.07 on Urban while Louvain finds communities with modularity 0.90+: land-use labels
   are unrelated to the graph's community structure.
4. **No floors.** Plain spectral clustering reaches 0.728 on DBLP, where the old B45 scored 0.170.

### What changed

- **Converged budget**: 1000-2000 epochs, `hid_dim` 512, `w_decay` 0, `temperature` 1.0, 3 decode trials (now the
  `tools/run_preset.py` defaults).
- **Warm start + dual-start selection** (`--warm_start auto`, `utils/warm_start.py`). On weak-feature graphs the Lorentz
  soft assignment sits on a balanced, graph-agnostic plateau: its loss stays above the SE of a *random* hard partition while
  spectral clustering finds a far lower one (`e2d_plateau_diag`; higher lr, smaller `hid_dim`, wider `max_nums` and S²X input
  features all stay stuck). The warm start pre-trains the leaf assignment on spectral(A) or SGC(A,X)+KMeans, whichever has
  lower 2-level SE. Because a mediocre pseudo partition can also trap a backbone that would have escaped on its own (Cora),
  each configuration is trained with and without warm start and the run with the lower SE checkpoint loss is kept.
  E3 used each model's own loss; E6 onward use the baseline's choice for all models. Class identities are not used to fit
  or select the partition, but the benchmark class count supplies the default pseudo-partition size.
- **Known-only evaluation is the default** (`--no-known_only_eval` restores the old behaviour).
- **New synthetic benchmark** `synth_ctx_v1` (`tools/prepare_synth_ctx_datasets.py`): weak features (KMeans 0.22),
  homophily 0.45 (below the topology-only detection threshold) and 0.55, and edge channel 0 in three modes: `none`,
  `direct` (context-free signal), `contextual` (sign flips with a latent endpoint role, so it is useless without endpoint context).
- **Classic floors** for every dataset (`tools/run_classic_baselines.py`).

### Evidence

**E1, backbone gate** (`e1_backbone_gate`, seed 0). Cora baseline NMI 0.237 → 0.421 and modularity 0.24 → 0.735 once
trained; B45 0.450. The released DSE code reaches 0.39 on Cora (best-of-eval, 1500 epochs), so the paper's 0.640 is not
reproducible from public code and is no longer a target.

**E2, does a model use edges?** (`e2_synth_ctx_warm`, 3 seeds, warm start throughout because the cold baseline is stuck
on this benchmark). Effects are NMI differences between graphs that share topology, features and labels: *arch* =
permEA − baseline, *descr* = none − permEA (aligned nuisance channels such as endpoint feature cosine), *direct eff* =
direct − none, *context eff* = contextual − none. EP2 is absent: all six runs collapsed to a single cluster after warm
start (ACC 25%, NMI 0.00) and were stopped at epoch 600-1100.

| h | model | permEA | none | direct | contextual | arch | descr | direct eff | context eff |
|---|---|---|---|---|---|---|---|---|---|
| h45 | baseline (no edges) | | 0.084±0.002 | | | | | | |
| h45 | SGC2+KMeans / oracle | | 0.472±0.001 | 0.855±0.000 | 0.839±0.000 | | | | |
| h45 | b45 | 0.088±0.020 | 0.362±0.012 | 0.729±0.008 | 0.357±0.016 | +0.004 | +0.274 | +0.367 | -0.005 |
| h45 | ep1_ctx | 0.094±0.012 | 0.091±0.006 | 0.093±0.012 | 0.096±0.003 | +0.010 | -0.003 | +0.002 | +0.006 |
| h45 | ep1_noctx | — | — | 0.099±0.010 | 0.101±0.007 | — | — | — | — |
| h55 | baseline (no edges) | | 0.602±0.006 | | | | | | |
| h55 | SGC2+KMeans / oracle | | 0.727±0.001 | 0.944±0.000 | 0.957±0.000 | | | | |
| h55 | b45 | 0.503±0.013 | 0.637±0.009 | 0.914±0.013 | 0.631±0.007 | -0.099 | +0.133 | +0.278 | -0.006 |
| h55 | ep1_ctx | 0.619±0.005 | 0.626±0.012 | 0.695±0.005 | 0.628±0.004 | +0.017 | +0.006 | +0.070 | +0.003 |
| h55 | ep1_noctx | — | — | 0.710±0.014 | 0.614±0.007 | — | — | — | — |

- **B45 uses context-free edge semantics strongly** (+0.37 / +0.28, close to the oracle-weighted SGC) **and cannot use
  contextual semantics at all** (−0.005).
- **B45 is hurt by uninformative edges**: with shuffled edge attributes it loses 0.10 against the edge-free baseline at h55.
- **EP1's context conditioning does nothing**: contextual effect ≈ 0, and the context-off ablation is as good or better.
- In E2 no model uses contextual edges, although the oracle shows 0.84-0.96 NMI is available. E7 below resolves part of this gap.
- The DSE baseline trails SGC2+KMeans on these weak-feature graphs (0.60 vs 0.73 at h55, 0.08 vs 0.47 at h45).

**E3, real graphs** (`e3_real_dual`; 3 seeds for Cora/DBLP, 1 for the rest; selection = lower best-checkpoint SE loss between warm
and cold start; known-only labels).

| dataset | old B45 (60 ep) | kmeans_x | spectral_a | louvain_a | sgc2_ax | baseline (SE-selected) | B45 (SE-selected) | baseline warm / nowarm | B45 warm / nowarm |
|---|---|---|---|---|---|---|---|---|---|
| cora | 0.164 | 0.043 | 0.202 | 0.447 (k=102) | 0.414 | 0.430±0.020 (nowarm) | 0.424±0.019 (nowarm) | 0.373±0.055 / 0.430±0.020 | 0.353±0.045 / 0.424±0.019 |
| photo | 0.002 | 0.170 | 0.556 | 0.657 (k=151) | 0.416 | 0.637 (warm) | 0.004 (warm) | 0.637 / 0.215 | 0.004 / 0.002 |
| dblp_magnn_author_v2 | 0.170 | 0.067 | 0.728 | 0.724 (k=5) | 0.515 | 0.759±0.002 (warm) | 0.667±0.014 (nowarm) | 0.759±0.002 / 0.664±0.034 | 0.738±0.012 / 0.667±0.014 |
| entities_aifb | 0.082 | 0.126 | 0.129 | 0.546 (k=18) | 0.137 | 0.158 (warm) | 0.017 (warm) | 0.158 / 0.086 | 0.017 / 0.008 |
| urban_paris_plot_v3bsjg | n/a | 0.049 | 0.023 | 0.039 (k=32) | 0.055 | 0.035 (warm) | 0.025 (warm) | 0.035 / 0.039 | 0.025 / 0.031 |

- **The fixed backbone is the real gain**: 0.170 → 0.759 on DBLP (above spectral 0.728 and Louvain 0.724), 0.002 → 0.637
  on Photo, 0.164 → 0.430 on Cora. On real graphs SE training improves on the warm-start partition (DBLP 0.727 → 0.759,
  Photo 0.415 → 0.637, AIFB 0.130 → 0.158).
- **B45 adds nothing on real graphs and collapses on Photo and AIFB** (0.004 / 0.017 with either start).
- SE-based start selection picked the better start for the baseline wherever the starts differed clearly, but picked the
  worse one for B45 on DBLP (0.667 vs 0.738).
- In E3 no tested real dataset shows B45 helping clustering beyond topology and features; see the E6/E8 follow-up below.

### Round 2 (same day): robustness, real-data headroom, contextual edges

Plans `e4`-`e8` in `configs/breakthrough/`; results in `results/breakthrough_v1/`.

**E4, why B45 collapses.** One component off at a time on Photo / AIFB (warm start). Only removing the edge-conditioned
message gate rescues training (Photo 0.004 → 0.591, AIFB 0.017 → 0.128; baseline 0.637 / 0.158). Removing the assignment
residual, the SE edge weights or the adaptive alpha, or weakening the residual, leaves both collapsed. B45 without the gate
is exactly the `b31_dualscalar_assign` preset, which is now the `run_preset.py` default.

**E5, how much can edge attributes help at all?** `tools/screen_edge_headroom.py` fits P(same label | edge_attr) with
labels and reweights the graph with it, a label-informed headroom diagnostic for context-free reweighting
(not a mathematical upper bound or an unsupervised result):

| dataset | AUC edge_attr / endpoint cosine | SGC plain → reweighted | spectral plain → reweighted |
|---|---|---|---|
| synth h55 direct | 0.875 / 0.664 | 0.728 → 0.932 | 0.449 → 0.909 |
| synth h55 contextual | 0.664 / 0.664 | 0.728 → 0.753 | 0.449 → 0.562 |
| Cora (derived attrs) | 0.596 / 0.596 | 0.414 → 0.460 | — |
| Photo (derived attrs) | 0.720 / 0.612 | 0.416 → 0.431 | 0.420 → 0.422 |
| DBLP v2 (native) | 0.722 / 0.506 | 0.513 → 0.513 | 0.728 → 0.759 |
| Urban Paris / WashingtonDC | 0.575 / 0.552, 0.582 / 0.554 | 0.055 → 0.056, 0.064 → 0.069 | ≈ 0 |

The repo's real graphs leave little room for any edge-aware model. Multiplex meta-path graphs
(`tools/prepare_pyg_multiplex_dataset.py`) are the exception: on `acm_multiplex_v2` the layers differ in homophily (PcP 0.88,
PAP 0.79, PTP 0.69, PSP 0.67) and SGC+KMeans goes from 0.55 on the union to 0.71 on the citation layer alone. IMDB genres
are unrecoverable from any layer (≤ 0.05).

**SE-driven edge weighting learns community crispness, not labels.** `tools/inspect_edge_factors.py` on B31/ACM: the SE
factor is 1.80 (upper clip) for PSP, the least homophilous layer, and 0.52-0.53 for PAP/PcP/PTP (corr with same-label
−0.07). Same-subject papers form cliques, which minimise structural entropy. On `synth_ctx_v1` the SE-cheap edges and the
same-label edges coincide (factor 1.62 vs 0.75, corr +0.60), which is why the synthetic gains are large and the ACM gain small.

**E6/E7, B31 and endpoint context on the synthetic benchmark** (3 seeds, warm start). `--edge_ctx_dim` appends
z_i ⊙ z_j of standardized node features to `edge_attr`:

| h | model | permEA | none | direct | contextual | arch | descr | direct eff | context eff |
|---|---|---|---|---|---|---|---|---|---|
| h45 | baseline (no edges) | | 0.084±0.002 | | | | | | |
| h45 | SGC2+KMeans / oracle | | 0.472±0.001 | 0.855±0.000 | 0.839±0.000 | | | | |
| h45 | b31_ctx | 0.308±0.026 | 0.291±0.018 | 0.504±0.049 | 0.472±0.048 | +0.224 | -0.017 | +0.213 | +0.181 |
| h45 | b31 | 0.089±0.013 | 0.217±0.059 | 0.616±0.009 | 0.219±0.049 | +0.005 | +0.128 | +0.399 | +0.002 |
| h45 | b45 | 0.088±0.020 | 0.362±0.012 | 0.729±0.008 | 0.357±0.016 | +0.004 | +0.274 | +0.367 | -0.005 |
| h45 | b45_ctx | — | — | — | 0.708±0.004 | — | — | — | — |
| h55 | baseline (no edges) | | 0.602±0.006 | | | | | | |
| h55 | SGC2+KMeans / oracle | | 0.727±0.001 | 0.944±0.000 | 0.957±0.000 | | | | |
| h55 | b31_ctx | 0.714±0.011 | 0.703±0.006 | 0.890±0.006 | 0.883±0.006 | +0.112 | -0.011 | +0.187 | +0.180 |
| h55 | b31 | 0.540±0.014 | 0.654±0.015 | 0.897±0.005 | 0.654±0.003 | -0.063 | +0.114 | +0.243 | +0.000 |
| h55 | b45 | 0.503±0.013 | 0.637±0.009 | 0.914±0.013 | 0.631±0.007 | -0.099 | +0.133 | +0.278 | -0.006 |
| h55 | b45_ctx | — | — | — | 0.901±0.007 | — | — | — | — |

- B31 keeps the direct-semantics gain (+0.40 / +0.24) and, like B45, gets nothing from contextual edges.
- **With endpoint context, contextual edges become usable**: B31+ctx +0.18 at both homophily levels, and B45+ctx reaches
  0.708 / 0.901 on contextual edges, close to B45 on direct edges. Shuffling the native attributes (permEA, context kept)
  removes the gain, so it comes from the attr0 × context interaction. EP1's failure was its implementation, not the idea.
- In the B31+ctx rows, *arch* (permEA − baseline) is not "no edge information": the context features stay aligned.
- Context costs some direct-mode accuracy for B31 (0.504 vs 0.616 at h45).

**E6/E8, real graphs** (start chosen per dataset and seed by the edge-free baseline's SE loss, then applied to every model;
SE losses of edge-aware models are computed on their own reweighted graphs and picked the worse start on DBLP and ACM):

| dataset | old B45 (60 ep) | kmeans_x | spectral_a | louvain_a | sgc2_ax | baseline (selected) | b31 (selected) | b31ctx (selected) | b45 (selected) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cora | 0.164 | 0.043 | 0.202 | 0.447 (k=102) | 0.414 | 0.430±0.020 (nowarm) | 0.404±0.024 (nowarm) | 0.472±0.016 (nowarm) | 0.424±0.019 (nowarm) |
| photo | 0.002 | 0.170 | 0.556 | 0.657 (k=151) | 0.416 | 0.637 (warm) | 0.552 (warm) | 0.598 (warm) | 0.004 (warm) |
| dblp_magnn_author_v2 | 0.170 | 0.067 | 0.728 | 0.724 (k=5) | 0.515 | 0.759±0.002 (warm) | 0.749±0.007 (warm) | 0.765±0.010 (warm) | 0.738±0.012 (warm) |
| entities_aifb | 0.082 | 0.126 | 0.129 | 0.546 (k=18) | 0.137 | 0.158 (warm) | 0.131 (warm) | 0.126 (warm) | 0.017 (warm) |
| urban_paris_plot_v3bsjg | n/a | 0.049 | 0.023 | 0.039 (k=32) | 0.055 | 0.035 (warm) | 0.036 (warm) | — | 0.025 (warm) |
| acm_multiplex_v2 | n/a | 0.391 | 0.566 | 0.394 (k=8) | 0.549 | 0.464±0.002 (warm) | 0.491±0.006 (warm) | 0.519±0.001 (warm) | 0.494±0.004 (warm) |

- **B31+ctx is the strongest edge-aware setting in E6/E8**: under this selected-start protocol it avoids the B45 collapse and beats the baseline on Cora (+0.04, also above Louvain and
  SGC), DBLP (+0.006) and ACM (+0.055), but trails it on Photo (−0.04) and AIFB (−0.03).
- ACM's layer headroom is mostly not captured (0.519 vs spectral 0.566 vs 0.71 with the citation layer alone), for the
  reason above.
- Neither modularity nor SE loss reliably chooses *between* models without labels; the baseline is the safe default.

### E9: removing learned weights from the SE graph

`configs/breakthrough/e9_fixed_ruler.json` tests B31 with `--edge_weight_learn_apply_to msg_only` (`b31fr`),
with and without endpoint context (`b31frctx`). Learned scalar factors act on the message graph only. The SE graph
retains its base weights and learned top-k augmentation, so “fixed ruler” means **no learned scalar SE reweighting**,
not a frozen graph or a detached encoder. The assignment residual and edge-weight regularizer remain active.

The same baseline-guided start rule as E6/E8 gives the following NMI. Cora, DBLP and ACM use three training seeds;
Photo and AIFB use seed 0. `±` is the population standard deviation across seed-level means; each run averages
three stochastic decodes. The [tracked selected-run table](../configs/breakthrough/evidence/real_selected.csv)
records each seed and start.

| dataset | selected start | baseline | B31 | B31 msg_only | B31+ctx | B31+ctx msg_only |
|---|---|---|---|---|---|---|
| cora | nowarm | 0.430±0.020 | 0.404±0.024 | 0.437±0.036 | 0.472±0.016 | 0.472±0.009 |
| photo | warm | 0.637 | 0.552 | 0.550 | 0.598 | 0.591 |
| dblp_magnn_author_v2 | warm | 0.759±0.002 | 0.749±0.007 | 0.759±0.001 | 0.765±0.010 | 0.760±0.009 |
| entities_aifb | warm | 0.158 | 0.131 | 0.134 | 0.126 | 0.138 |
| acm_multiplex_v2 | warm | 0.464±0.002 | 0.491±0.006 | 0.493±0.002 | 0.519±0.001 | 0.479±0.013 |

This does not solve real-data edge selection: Photo/AIFB still trail the baseline, and ACM with context gets worse.
The 48 synthetic jobs ran afterwards on an RTX 4060 Laptop with the same plan. Removing learned SE reweighting roughly
halves the synthetic edge gains: the direct effect falls from +0.40 to +0.24 (h45) and from +0.24 to +0.13 (h55); the
context effect with endpoint context falls from +0.18 to +0.09 / +0.10. Where the SE-cheapest weighting is the
label-aligned one (synthetic), about half of the edge gain flows through the learned SE graph; on real graphs that
pathway is roughly neutral. Rows: [synth_mechanism.csv](../configs/breakthrough/evidence/synth_mechanism.csv); the
cross-device drift check below bounds the machine effect at about 0.02.

`e9x_machine_check` reruns two warm, seed-0 configurations on an RTX 4060 Laptop: Cora changes from 0.4344 to
0.4511 and DBLP from 0.7600 to 0.7607. Pseudo-partition NMI is unchanged. This is a two-run drift check, not proof of
cross-device determinism; see [machine_check.csv](../configs/breakthrough/evidence/machine_check.csv).

### E10: does the SE objective prefer the target partition?

`tools/check_se_alignment.py` compares the label partition's two-level SE with the lower SE of spectral and
SGC+KMeans partitions at the same class count, for a finite set of candidate edge weightings. Its
`label_gap = SE(label) − min(SE(spectral), SE(SGC))` is diagnostic: a positive gap favours a tested non-label
partition. It does not find the global minimum or establish a causal explanation of a trained model.

On ACM, the SE-preferred candidate is `layer+PAP` (NMI 0.5572, gap +0.0989), whereas `layer+PcP` has gap −0.0142.
On direct synthetic graphs, `+attr0` is preferred and reaches NMI 0.7242 / 0.9097 at h45 / h55. These examples support
an objective-alignment concern, but DBLP's SE-preferred venue weighting also reaches NMI 0.7662: the relationship is
dataset-dependent. The full small screen is tracked in [se_alignment.csv](../configs/breakthrough/evidence/se_alignment.csv).
Labels are used only in this screen; graphs containing `-1` labels are skipped.

### B45 follow-up: collapse mechanism

`tools/diagnose_b45_message_gate.py` adds checkpoint interventions, graph-cross probes and paired short training
without changing the production gate. On Photo, seed 0, a common initialization and fixed SGC pseudo partition
are used for 200 warm-start + 200 SE epochs. Full decoding at the final epoch uses seed 999; NMI never selects a checkpoint.

| intervention | final decode NMI |
|---|---|
| original B45 | 0.0070 |
| bypass message gate | 0.6416 |
| unit gate factor (normalization retained) | 0.0080 |
| remove post-gate degree normalization | 0.6419 |
| remove gate dropout | 0.0058 |
| preserve transformed features at isolated aggregation rows | 0.6418 |

Checkpoint graph-cross probes localize the mismatch to the leaf embeddings: Photo has NMI 0.6052 with the training's
augmented embedding graph and 0.0052 with the base graph used for decoding. Changing only the assignment graph
does not rescue base-graph embeddings. About 98.5% of new directed edges target originally isolated nodes;
post-gate normalization increases the new-edge weight share from 3.56% to roughly 31–41%.
These are different checkpoints/protocols from the paired short-training table and should not be combined into one score.

The AIFB checkpoint shows the same direction of mismatch; DBLP does not. The DBLP paired-training probe was interrupted
by resource contention and is excluded. The normalization fix is now a model switch (`--no-edge_msg_gate_renorm`;
`--keep_isolated_nodes` implements the isolated-row variant) and is validated at the full budget in E11 below.
See the [intervention](../configs/breakthrough/evidence/b45_gate_interventions.csv),
[graph-cross](../configs/breakthrough/evidence/b45_graph_cross.csv) and
[graph-statistics](../configs/breakthrough/evidence/b45_graph_stats.csv) tables.

### E11: safe message gate and B31 pathways

`e11_safe_gate_pathways.json` trains B45 + endpoint context with gate re-normalization off (`b45nrctx`), ablates B31's
pathways on Photo/AIFB, and adds seeds 1–2 to the Photo/AIFB references (all real rows now use three seeds, except
where marked). Selected start, NMI:

| dataset | baseline | B31 | B31+ctx | B45 | B45 safe gate + ctx |
|---|---|---|---|---|---|
| cora | 0.430±0.020 | 0.404±0.024 | 0.472±0.016 | 0.424±0.019 | **0.479±0.027** |
| photo | **0.638±0.001** | 0.583±0.036 | 0.598 (1 seed) | 0.004 (1 seed) | 0.634±0.023 |
| dblp_magnn_author_v2 | 0.759±0.002 | 0.749±0.007 | **0.765±0.010** | 0.738±0.012 | 0.740±0.014 |
| entities_aifb | **0.184±0.019** | 0.116±0.016 | 0.126 (1 seed) | 0.017 (1 seed) | 0.129±0.013 |
| acm_multiplex_v2 | 0.464±0.002 | 0.491±0.006 | **0.519±0.001** | 0.494±0.004 | 0.508±0.004 |

On `synth_ctx_v1` the safe gate keeps B45's contextual reading: contextual 0.720 / 0.902 (h45 / h55) against
permEA 0.357 / 0.728 and none 0.370 / 0.725; the original B45 + context reached 0.708 / 0.901. B31 pathway ablations
(warm start, three seeds unless marked) give Photo 0.55–0.60 for every variant, so no single pathway explains the Photo gap; on AIFB
the learned SE factors hurt most (SE-graph-only 0.084, message-only 0.134) and removing the assignment residual is worse
still (0.029). A graph-cross probe of the B31 Photo checkpoint shows no train/decode mismatch (0.51–0.56 in all cells).

### E12–E13: a training-free search, and contexts the learned heads cannot read

`tools/run_se_guided_baseline.py` scores explicit edge weightings without training: one edge channel or multiplex layer
(F1), one endpoint-context channel (F2), an edge channel times a context channel (F3), optionally exhaustive
second-order products (F4) or a beam (Br). Factors use B31's range [0.5, 2]. Each weighted graph is clustered by
spectral clustering and SGC+KMeans at the benchmark class count; the lowest two-level SE wins. Label identities
score the pick only; the benchmark class count is supplied.

| benchmark | search, no context | search, first-order context | search, second-order | B31+ctx | B45+ctx |
|---|---|---|---|---|---|
| synth_ctx_v1 contextual h45 / h55 | 0.531 / 0.733 | **0.724** / 0.822 | — | 0.472 / 0.883 | 0.708 / **0.901** |
| synth_ctxxor_v1 contextual h45 / h55 | 0.094 / 0.559 | 0.015 / 0.559 | **0.448 / 0.825** | 0.169 / 0.688 | 0.170 / 0.717 (safe gate) |

`synth_ctxxor_v1` (`--role_mode xor2`) exposes two latent bits per node and sets role = bit1 XOR bit2, so the useful
reading is `attr0 × ctx_a × ctx_b`. The learned heads' context effect is +0.00 / +0.01–0.03 there, while second-order
enumeration finds `+attr0*ctx32*ctx33` with 4,837 candidates (349 at first order). On first-order contexts the search
matches the learned heads at h45 and trails them by 0.06–0.08 at h55.

On real graphs (plain spectral, seed 0) the search reaches Cora 0.468, Photo 0.566, DBLP 0.766, AIFB 0.461 and ACM
0.557. E15 shows that the Cora/Photo numbers depend on plain spectral's unstable eigenvectors and that the AIFB number
mostly reflects plain spectral's weakness there. Rows: [se_search_plain.csv](../configs/breakthrough/evidence/se_search_plain.csv),
[xor_context_learned.csv](../configs/breakthrough/evidence/xor_context_learned.csv).

### E14–E16: search, then train

The search pick is written as fixed edge weights (`tools/write_se_reweighted_dataset.py`), the edge-free DSE is trained
on that graph (dual starts, baseline-guided), and `tools/se_select_decoder.py` optionally chooses between the DSE
assignment and spectral / SGC partitions of the same graph by structural entropy. E14 uses the plain-spectral search
and decoder choice; E16 repeats it with regularized spectral after E15. Three seeds, NMI:

| dataset | DSE, original graph | DSE on searched graph | E14: + SE decoder (plain) | E16: + SE decoder (regularized) |
|---|---|---|---|---|
| cora | 0.430 | 0.450 | 0.451 | 0.431 |
| photo | **0.638** | 0.629 | 0.629 | 0.453 |
| dblp_magnn_author_v2 | 0.759 | 0.761 | **0.766** | 0.761 |
| entities_aifb | 0.184 | 0.083 | **0.461** | 0.443 |
| acm_multiplex_v2 | 0.464 | **0.538** | 0.538 | 0.538 |
| mean | 0.495 | 0.492 | 0.569 | 0.525 |

The "DSE on searched graph" column uses E16's picks (Cora `+logdeg_gap`, AIFB `+attr1*ctx2`; Photo, DBLP and ACM as in
E14). On the synthetic picks the regularized decoder choice gives 0.725 / 0.913 (first-order contexts) and 0.477 / 0.811
(XOR, second-order pick). The search improves the graph DSE trains on where an edge channel carries the target
structure (ACM +0.074). Choosing the decoder by SE is the unreliable part: a stronger SE optimizer (regularized spectral)
wins the comparison on Photo with a less label-aligned partition. E14's higher mean partly reflects plain spectral
losing to DSE, and its Photo pick differed between two machines.

**E15, robustness and scale.** `tools/check_spectral_stability.py` runs spectral clustering from five eigensolver starts:

| dataset | components | plain spectral NMI (min–max) | regularized spectral NMI |
|---|---|---|---|
| Cora | 78 | 0.027–0.177 | 0.336 |
| Photo | 136 | 0.005–0.513 | 0.461 |
| DBLP | 1 | 0.728 | 0.751 |
| AIFB | 3 | 0.129 | 0.512 |
| ACM | 1 | 0.566 | 0.501 |

Regularized spectral clustering (`spectral_reg`, τ = mean degree) is start-independent; with it the search picks are
identical across seeds. On AIFB the regularized uniform graph alone gives 0.512 and the best edge-only pick 0.62, while
adding context candidates lets SE prefer a weaker pick (0.443), so AIFB's earlier gain was not a contextual effect.
A beam over context products (width 8, order 2; at most ~500 extra candidates) never selects a second-order term on the real graphs
and fails on XOR (0.019 / 0.698 against 0.477 / 0.812 for exhaustive search): without lower-order signal there is nothing
for the beam to extend. Rows: [spectral_stability.csv](../configs/breakthrough/evidence/spectral_stability.csv),
[se_search_reg_beam.csv](../configs/breakthrough/evidence/se_search_reg_beam.csv),
[search_then_train.csv](../configs/breakthrough/evidence/search_then_train.csv).

### E17–E18: choosing between methods without labels

E14/E16 exposed the selection problem: structural entropy prefers the most compressible partition, whichever method
produced it, and spectral clustering optimizes nearly the same criterion. `tools/select_partition_rules.py` compares
rules on 42 decisions: model choice on the five real graphs (baseline, B31, B31+ctx, B45 safe gate + ctx, regularized
spectral, SGC; three seeds) and decoder choice on the nine E14/E16 searched graphs (DSE vs regularized spectral / SGC at
the DSE and class cluster counts; three seeds). The instances are listed in
[selection_rules_spec.json](../configs/breakthrough/selection_rules_spec.json). Rules R0–R3 were fixed before evaluation:

- R0 `se`: lowest structural entropy on the input graph (E14/E16's rule);
- R1 `medoid`: one candidate per method, highest mean AMI to the others;
- R2 `edge_cv`: hold out 20% of undirected edges (`tools/make_edge_cv_split.py`), recompute every candidate on the
  rest (models retrained: `e17_edge_cv.json`, 29 runs, seed 0), pick the lowest structural entropy on the held-out edges;
- R3 `two_view`: rank sum of structural entropy on the input graph and on a 10-NN feature graph.

Seed-0 decisions (14, the subset with edge-CV runs); regret = best candidate's NMI minus the pick's:

| rule | mean NMI | mean regret | max regret | regret > 0.05 |
|---|---|---|---|---|
| R0 se | 0.598 | 0.050 | 0.225 | 5 |
| R1 medoid | 0.587 | 0.060 | 0.404 | 2 |
| R2 edge_cv | 0.625 | 0.022 | 0.176 | 1 |
| R3 two_view | 0.596 | 0.052 | 0.322 | 4 |
| R4 gated_cv (defined after E18) | 0.644 | 0.004 | 0.029 | 0 |
| best candidate | 0.647 | | | |

Over all 42 decisions R0, R1 and R3 have mean regret 0.048, 0.057 and 0.044 with 15, 6 and 11 misses. Edge
cross-validation fixes Cora and ACM model choice, the DBLP decoder and both XOR decoders (0.477 → 0.702, 0.811 → 0.880).
Its one miss, the Photo decoder, is not overfitting: regularized spectral's partition also compresses held-out edges
better, so it reflects real graph structure that the labels do not follow. The feature view picks DSE there but fails on
AIFB, whose features compress held-out edges worse than a random partition (gate r = −0.44 / −0.59; 0.11–0.78 on the
other datasets). R4 combines the two: rank by held-out and feature-graph structural entropy when a feature-only
partition beats random on held-out edges (r > 0), otherwise use R2. It was designed with these 14 decisions in view,
so its score is a hypothesis, not evidence; E19 below tests it. Rows: [selection_rules_summary.csv](../configs/breakthrough/evidence/selection_rules_summary.csv),
[selection_rules_instances.csv](../configs/breakthrough/evidence/selection_rules_instances.csv).

### E19: validating the selection rules on new decisions

`tools/run_selection_validation.py` builds ten decisions none of the rules saw, with the rule code frozen at `876486b`:
model choice on Citeseer, Amazon Computers and PubMed, decoder choice on their SE-searched graphs, and decoder choice on
synthetic graphs drawn with data seed 1 (first-order and XOR contexts). It runs the search, writes the reweighted and
edge-CV datasets, trains in two stages (`e19a_selection_validation.json`: baseline starts and synthetic runs;
`e19b_selection_validation.json`: the other models and all CV runs at the baseline-chosen start), writes
[selection_validation_spec.json](../configs/breakthrough/selection_validation_spec.json) and scores it. Seed 0, one split:

| rule | E18 mean regret (14) | E19 mean regret (10) | E19 regret > 0.05 | pooled 24 |
|---|---|---|---|---|
| R0 se | 0.050 | 0.034 | 2 | 0.043 |
| R1 medoid | 0.060 | 0.033 | 2 | 0.049 |
| R2 edge_cv | 0.022 | 0.031 | 2 | 0.026 |
| R3 two_view | 0.052 | 0.018 | 1 | 0.038 |
| R4 gated_cv | 0.004 (designed on these) | 0.025 | 2 | — |

All rules agree on five of the ten new decisions (the four synthetic graphs and the Citeseer decoder), so E19 separates
the rules less than E18, and ten decisions cannot resolve differences of a few hundredths. R4's advantage over R2 shrank
from 0.018 to 0.006 with the same number of misses, so it is not adopted. Amazon Computers reproduces the Photo failure:
regularized spectral clustering wins the decoder choice under every graph-based rule (0.388 against DSE 0.452) and the
model choice under R0/R2 (0.394 against B31 0.556); its feature gate is positive (r = 0.08), yet the rank sum still
prefers spectral. PubMed is weak for every method (0.22–0.27). Over the 24 decisions where R0–R3 were fixed in advance,
edge cross-validation has the lowest regret, but no rule removes label misalignment. Rows:
[selection_validation_summary.csv](../configs/breakthrough/evidence/selection_validation_summary.csv),
[selection_validation_instances.csv](../configs/breakthrough/evidence/selection_validation_instances.csv).
On a 24 GB GPU, PubMed- and Computers-sized graphs need `--max_parallel 2`: DSE's dense N×N score matrices take 8–10 GB
per job below `knn_auto_threshold` (20,000 nodes), and four parallel jobs ran out of memory.

### E20: few-label selection

Because label-free selection cannot tell which real structure the labels follow, `tools/few_label_selection.py` asks how
many labels the choice needs. For each of the 52 E18/E19 decisions it rebuilds the candidates once, then draws m labeled
nodes uniformly (200 draws per m), picks the candidate with the highest adjusted Rand index on those nodes (ties count as
their mean) and scores the pick on all labeled nodes. The labels are used only to select among at most six candidates.

| selection | NMI regret (24 decisions with edge CV) | ARI regret (same 24) | NMI regret, all 52 | P(NMI regret > 0.05), all 52 |
|---|---|---|---|---|
| SE, no labels | 0.043 | 0.060 | 0.045 | |
| edge CV, no labels | 0.026 | 0.049 | | |
| 5 labels | 0.075 | 0.085 | 0.077 | 0.48 |
| 10 labels | 0.052 | 0.057 | 0.052 | 0.28 |
| 20 labels | 0.029 | 0.031 | 0.029 | 0.16 |
| 50 labels | 0.015 | 0.012 | 0.014 | 0.08 |
| 100 labels | 0.009 | 0.006 | 0.009 | 0.04 |

Twenty labels match edge cross-validation in NMI and beat both label-free rules in ARI; fifty beat them clearly. The
misaligned decisions resolve: Photo decoder regret falls from 0.175 (any label-free rule) to 0.031 with 20 labels and
0.001 with 50; Computers model choice from 0.163 to 0.015 with 50 and 0.002 with 100. Residual NMI regret on the
Computers decoder and the XOR h55 decoder comes from candidates whose ARI and NMI rankings differ or that are nearly tied
(ARI regret at 100 labels: 0.011 and 0.014). Caveats: labeled nodes are drawn uniformly at random, which is optimistic
for real annotation; regret is scored on all labeled nodes including the m drawn (on AIFB, 100 of 176). Rows:
[few_label_selection_summary.csv](../configs/breakthrough/evidence/few_label_selection_summary.csv),
[few_label_selection_instances.csv](../configs/breakthrough/evidence/few_label_selection_instances.csv).

### E21: few-label edge semantics

Three ways to use m labeled nodes (drawn by `data.choose_labeled_nodes`, excluded from evaluation) are compared with
semi-supervised node classification (logistic regression on SGC features, `node_clf`):
1. `se_top_label`: the ten lowest-SE weightings of the E15 search (plus uniform), each clustered by regularized spectral
   clustering and SGC+KMeans; the labels pick one by ARI (`tools/few_label_edge_semantics.py`);
2. `edge_clf`: a same-label classifier on edges between labeled nodes (edge attributes + endpoint context; logistic
   regression, MLP, and an MLP with pseudo-labeled edges from `node_clf`) reweights the graph before clustering;
3. few-label DSE (`e21_few_label_dse.json`, `--label_budget m --label_seed r`): B31 + context with a balanced pairwise
   same-cluster loss on labeled nodes and a same-label loss on its edge-weight heads (`b31ctxsup`), against the edge-free
   baseline with the pairwise loss only (`basesup`). Both losses use weight 1.0, untuned.

NMI on held-out known-label nodes, mean of label draws 0–1, m = 20 / 200 (AIFB: m = 20; it has 176 known labels):

| dataset | node_clf | se_top_label | edge_clf (pseudo) | basesup | b31ctxsup |
|---|---|---|---|---|---|
| synth direct h45 | 0.15 / 0.44 | **0.72 / 0.72** | 0.47 / 0.65 | 0.07 / 0.10 | 0.39 / 0.46 |
| synth direct h55 | 0.33 / 0.70 | 0.83 / **0.91** | 0.60 / 0.88 | 0.49 / 0.52 | **0.87** / 0.85 |
| synth contextual h45 | 0.15 / 0.44 | **0.60 / 0.72** | 0.47 / 0.54 | 0.02 / 0.10 | 0.37 / 0.42 |
| synth contextual h55 | 0.33 / 0.70 | 0.83 / **0.91** | 0.60 / 0.89 | 0.49 / 0.55 | **0.86** / 0.83 |
| synth XOR h45 | 0.13 / 0.43 | **0.70 / 0.70** | 0.45 / 0.49 | 0.01 / 0.09 | 0.17 / 0.19 |
| synth XOR h55 | 0.37 / 0.67 | **0.81 / 0.88** | 0.58 / 0.72 | 0.45 / 0.50 | 0.61 / 0.62 |
| Cora | 0.20 / 0.57 | **0.57 / 0.58** | 0.32 / 0.37 | 0.36 / 0.43 | 0.42 / 0.45 |
| Photo | 0.53 / **0.81** | 0.44 / 0.52 | 0.44 / 0.56 | **0.67** / 0.65 | 0.66 / 0.65 |
| DBLP | 0.64 / 0.73 | **0.78** / 0.74 | 0.76 / **0.78** | 0.76 / 0.77 | 0.77 / **0.78** |
| AIFB (m = 20) | 0.09 | **0.53** | 0.31 | 0.01 | 0.05 |
| ACM | **0.57 / 0.67** | 0.57 / 0.57 | 0.55 / 0.65 | 0.53 / 0.59 | 0.46 / 0.65 |

Edges between labeled nodes are scarce (0–13 at m ≤ 50; about 250 at m = 200 on the synthetic graphs, 886 on ACM), so
learned edge semantics (edge classifiers, supervised edge heads) only start to work around 200 labels and only for
first-order readings; neither learns the XOR context. `se_top_label` reaches it because exhaustive second-order
enumeration put the right weighting among the SE top ten. Semi-supervised classification is the stronger choice when
node features are informative and labels are not scarce (Photo, ACM at ≥ 100 labels); structure- and edge-based few-label
clustering is stronger when features are weak or labels very few. The pairwise loss sometimes hurts DSE (AIFB, synthetic
h55 against the unsupervised baseline), and method 3 has two label draws only. Rows:
[few_label_edge_methods.csv](../configs/breakthrough/evidence/few_label_edge_methods.csv),
[few_label_dse_runs.csv](../configs/breakthrough/evidence/few_label_dse_runs.csv).

### E22: letting the labels choose the route

`tools/few_label_route_selection.py` uses the same labeled nodes to choose between the structure route (`se_top_label`)
and the feature route (`node_clf`): 5-fold cross-validation inside the labeled set, the structure route mapping each
cluster to its training-fold majority label; the more accurate route is refit on all m labels. Up to eleven graphs, five label draws per budget, NMI on known-label nodes excluded from the label draw.
The 10–100-label rows cover 11 graphs (55 draws); at 200 labels AIFB is skipped because it has only 176 known labels,
so that row covers 10 graphs (50 draws):

| labels | structure only | feature only | chosen route | better route per draw | regret | structure chosen |
|---|---|---|---|---|---|---|
| 10 | 0.648 | 0.242 | 0.634 | 0.653 | 0.019 | 85% |
| 20 | 0.672 | 0.322 | 0.669 | 0.685 | 0.017 | 84% |
| 50 | 0.673 | 0.502 | **0.696** | 0.700 | 0.004 | 80% |
| 100 | 0.703 | 0.551 | **0.730** | 0.732 | 0.002 | 73% |
| 200 | 0.724 | 0.617 | **0.761** | 0.763 | 0.002 | 66% |

The choice follows the data: the structure route on every synthetic graph and on AIFB (93–100% of draws), the feature
route on Photo from 20 labels and on ACM from 100, and a switch on Cora when node classification overtakes at 200
labels. DBLP, where the two routes are close, is the least stable (regret about 0.02). At 10–20 labels the
cross-validation is noisy and the structure route alone is slightly better. Rows:
[route_selection_summary.csv](../configs/breakthrough/evidence/route_selection_summary.csv),
[route_selection_draws.csv](../configs/breakthrough/evidence/route_selection_draws.csv).

### Retired in this pass

| Item | Why |
|---|---|
| All rankings from ≤120-epoch runs (B45 > B40 > G20, EP1/EP2 verdicts, Urban NMI) | Backbone untrained; see above |
| `synth_mech_full_v1`, `synth_edgectrl_v1` as evidence; `tools/run_mechanism_synth_suite.py` (archived) | Features alone give NMI 1.0 |
| EP1 (Expert1 contextual edge state) | E2: contextual effect ≈ 0, context-off ablation as good |
| EP2 (Expert2 relation state, B50) | E2: collapses to one cluster even with a warm start |
| Expert3 shared-tree multi-channel prototype | Never beat its own early-scalarised ablation (e.g. 0.39 vs 0.86 on old h85); dense C×N×N adjacency |
| Presets B40, B47, B48, G20 (moved to `archive/configs/historical_presets/`) | B40/B47 are minor B45 variants, B48 was harmful, G20's SE path is inside B45 |
| B45 as the default preset (kept, not retired) | Its message gate collapsed Photo/AIFB (E4); fixed by `--no-edge_msg_gate_renorm` (E11), still an ablation |
| Learned edge heads as the proposed method (kept as ablations) | Matched by a training-free search on first-order contexts; blind to second-order contexts (E12–E13) |
| IMDB multiplex | Genre labels unrecoverable from any layer |
| `Expert suggestion/` (moved to `archive/docs/expert_suggestion_2026-04/`) | Both directions refuted |
| Urban land-use NMI as clustering evidence | Labels unrelated to graph communities; Urban stays a data-paper track |
| Paper-reported LSEnet numbers as targets | Not reproducible from the released code |

### Open problems, in priority order

1. **Label-free choice between partition generators and models.** Structural entropy prefers the most compressible
   partition, whichever method produced it. Edge cross-validation (E18) removes the self-evaluation advantage but not
   label misalignment (Photo, Amazon Computers). The feature-gated rule R4 did not validate (E19). Without labels, report
   per-method results with edge-CV selection as a modest improvement (pooled regret 0.026 vs 0.043); with 20–50 labeled
   nodes, few-label selection resolves the choice (E20). Open: non-uniform or actively chosen labels, and combining the
   edge-CV ranking with labels at budgets below 20.
2. **Higher-order contexts.** Exhaustive enumeration works but grows combinatorially (349 → 4,837 candidates at order 2
   with 34 context dims); gradients and beams need lower-order signal. A prior on which node attributes can carry context
   (categorical node types or roles) is the practical lever to test.
3. **DSE where SE is misaligned or the backbone is weak.** AIFB: DSE 0.18 against regularized spectral 0.51;
   ACM: 0.46–0.54 against spectral 0.566; SE training can drift away from a good initial partition.
4. **More real graphs with edge-type headroom.** ACM remains the main case; AIFB's gain comes from a relation channel
   without context. Screen candidates with `tools/screen_edge_headroom.py` and the search before training.
5. **Search robustness beyond the eigensolver.** Picks are now start-independent, but the candidate pool still matters:
   adding context candidates changed AIFB's winner from 0.62 to 0.443.
