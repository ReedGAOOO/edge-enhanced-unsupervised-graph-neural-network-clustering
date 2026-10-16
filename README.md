# Edge Enhanced Unsupervised Graph Neural Network Clustering

Unsupervised graph clustering built on DSE/LSEnet (Lorentz structural entropy), with edge-aware variants. The
question the project asks: when do edge attributes help structural-entropy clustering, and which model actually uses them?

## Current status (2026-10-15, evening)

The active branch is `breakthrough-2026-10`. Plans E1–E16 are recorded. The current position:

- **Learned edge heads are not the method.** On first-order contextual edges a training-free, SE-guided search over
  explicit edge weightings matches them; on second-order (XOR-role) contexts the learned heads read nothing (context
  effect ≈ 0) while exhaustive second-order search recovers the signal (E12–E13). The SE gradient has no first-order
  signal to follow for parity-like interactions, and neither does a beam search (E15).
- **Mainline: SE-guided edge-weight search, then the edge-free DSE on the searched graph.** The search
  (`tools/run_se_guided_baseline.py --spectral reg`) uses no labels. It finds the useful weighting where one exists
  (ACM `layer+PAP`, the DBLP venue channel, the synthetic contexts including XOR at second order), and training DSE on
  the searched ACM graph lifts it from 0.464 to 0.538 (E14/E16).
- **Label-free choice between methods: edge cross-validation helps, misalignment remains.** Choosing by plain structural
  entropy picks a worse partition in 5 of 14 seed-0 decisions (Photo DSE 0.629 → spectral 0.453). Scoring candidates
  on 20% held-out edges after recomputing them on the rest (E17–E18) cuts the mean regret from 0.050 to 0.022, with one
  miss left (Photo again: real but label-misaligned graph structure). A feature-gated variant designed after seeing E18
  did not hold up on ten new decisions (E19: regret 0.025 against 0.031 for edge CV, two misses each); Amazon Computers
  repeats Photo's failure. No label-free rule resolves label misalignment; report per-method results.
- **A few labels resolve the choice.** Selecting among the same candidates with 20 random labeled nodes matches the
  best label-free rule, and 50 labels beat it clearly (E20: NMI regret 0.029 / 0.015 against 0.026 for edge CV and
  0.043 for SE on the same 24 decisions); 50 labels fix the Photo decoder completely. The defensible pipeline is
  unsupervised candidate generation (SE-guided search, DSE, classic clusterers) followed by few-label selection.
- **With a few labels, edge semantics still matter where features are weak; the learned heads still do not.** In E21 the
  best few-label method is "SE proposes, labels pick" (top-10 SE weightings, labels choose): best or near best on all
  synthetic graphs (XOR included, 0.70 / 0.88), AIFB, DBLP and Cora at small budgets. Semi-supervised node
  classification wins where node features are strong (Photo 0.81, ACM 0.67 at 200 labels). Supervising B31's edge heads
  and cluster assignments does not teach it XOR contexts (0.19 / 0.62) and helps only ACM (0.652 vs 0.592 without edges).
- **The labels can also choose the route.** Five-fold cross-validation inside the labeled set picks between the
  structure route ("SE proposes, labels pick") and the feature route (node classification); from 50 labels the choice is
  within 0.004 of the better route per draw and beats either route alone (E22: NMI 0.696 / 0.730 / 0.761 at 50 / 100 /
  200 labels over 11 graphs).
- **Plain spectral clustering is unstable on graphs with more components than clusters** (Cora 78, Photo 136): NMI over
  five eigensolver starts spans 0.03–0.18 on Cora and 0.005–0.51 on Photo. `spectral_reg` removes this (E15).
- **B45's message gate is fixed.** `--no-edge_msg_gate_renorm` removes the Photo collapse (0.004 → 0.634) and keeps
  the first-order contextual gains (0.720 / 0.902) at full budget (E11). B31/B45 remain as ablations.
- Training and start selection use SE rather than NMI. Label-based headroom, alignment and oracle columns are diagnostics.

Start with [Quick start](#quick-start), [Reproducing the breakthrough pass](#reproducing-the-breakthrough-pass),
or the [dataset guide](#dataset-guide-type-location-commands). Small evidence tables are tracked in
[configs/breakthrough/evidence/](configs/breakthrough/evidence/README.md); full logs, datasets and checkpoints are local assets.

## Project map

| Path | Purpose |
|---|---|
| `main.py`, `exp.py`, `data.py` | Training CLI, experiment lifecycle, dataset loading and edge preparation |
| `modules/`, `manifold/`, `utils/` | DSI/LSENet, Lorentz layers and geometry, warm start, decoding and evaluation |
| [configs/](configs/README.md) | Three active presets and the [experiment plan index](configs/breakthrough/README.md) |
| [tools/](tools/README.md) | Runners, dataset builders, result tables and diagnostic commands |
| [data/](data/README.md) | Git-ignored data; [DATASETS.md](DATASETS.md) describes datasets and [DATA_ASSETS.md](DATA_ASSETS.md) records local assets |
| [reading/](reading/README.md), [reference/](reference/README.md) | Implementation notes, historical expert bundles and upstream reference material |
| [reports/](reports/README.md) | Urban data-construction reports; separate from current clustering claims |
| `results/`, `checkpoints/` | Generated local outputs, excluded from Git |
| [archive/](archive/README.md) | Historical code, configs, reports and workspaces |

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
three stochastic decodes. The [tracked selected-run table](configs/breakthrough/evidence/real_selected.csv)
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
pathway is roughly neutral. Rows: [synth_mechanism.csv](configs/breakthrough/evidence/synth_mechanism.csv); the
cross-device drift check below bounds the machine effect at about 0.02.

`e9x_machine_check` reruns two warm, seed-0 configurations on an RTX 4060 Laptop: Cora changes from 0.4344 to
0.4511 and DBLP from 0.7600 to 0.7607. Pseudo-partition NMI is unchanged. This is a two-run drift check, not proof of
cross-device determinism; see [machine_check.csv](configs/breakthrough/evidence/machine_check.csv).

### E10: does the SE objective prefer the target partition?

`tools/check_se_alignment.py` compares the label partition's two-level SE with the lower SE of spectral and
SGC+KMeans partitions at the same class count, for a finite set of candidate edge weightings. Its
`label_gap = SE(label) − min(SE(spectral), SE(SGC))` is diagnostic: a positive gap favours a tested non-label
partition. It does not find the global minimum or establish a causal explanation of a trained model.

On ACM, the SE-preferred candidate is `layer+PAP` (NMI 0.5572, gap +0.0989), whereas `layer+PcP` has gap −0.0142.
On direct synthetic graphs, `+attr0` is preferred and reaches NMI 0.7242 / 0.9097 at h45 / h55. These examples support
an objective-alignment concern, but DBLP's SE-preferred venue weighting also reaches NMI 0.7662: the relationship is
dataset-dependent. The full small screen is tracked in [se_alignment.csv](configs/breakthrough/evidence/se_alignment.csv).
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
See the [intervention](configs/breakthrough/evidence/b45_gate_interventions.csv),
[graph-cross](configs/breakthrough/evidence/b45_graph_cross.csv) and
[graph-statistics](configs/breakthrough/evidence/b45_graph_stats.csv) tables.

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
spectral clustering and SGC+KMeans at the class count; the lowest two-level SE wins. Labels score the pick only.

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
mostly reflects plain spectral's weakness there. Rows: [se_search_plain.csv](configs/breakthrough/evidence/se_search_plain.csv),
[xor_context_learned.csv](configs/breakthrough/evidence/xor_context_learned.csv).

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
for the beam to extend. Rows: [spectral_stability.csv](configs/breakthrough/evidence/spectral_stability.csv),
[se_search_reg_beam.csv](configs/breakthrough/evidence/se_search_reg_beam.csv),
[search_then_train.csv](configs/breakthrough/evidence/search_then_train.csv).

### E17–E18: choosing between methods without labels

E14/E16 exposed the selection problem: structural entropy prefers the most compressible partition, whichever method
produced it, and spectral clustering optimizes nearly the same criterion. `tools/select_partition_rules.py` compares
rules on 42 decisions: model choice on the five real graphs (baseline, B31, B31+ctx, B45 safe gate + ctx, regularized
spectral, SGC; three seeds) and decoder choice on the nine E14/E16 searched graphs (DSE vs regularized spectral / SGC at
the DSE and class cluster counts; three seeds). The instances are listed in
[selection_rules_spec.json](configs/breakthrough/selection_rules_spec.json). Rules R0–R3 were fixed before evaluation:

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
so its score is a hypothesis, not evidence; E19 below tests it. Rows: [selection_rules_summary.csv](configs/breakthrough/evidence/selection_rules_summary.csv),
[selection_rules_instances.csv](configs/breakthrough/evidence/selection_rules_instances.csv).

### E19: validating the selection rules on new decisions

`tools/run_selection_validation.py` builds ten decisions none of the rules saw, with the rule code frozen at `876486b`:
model choice on Citeseer, Amazon Computers and PubMed, decoder choice on their SE-searched graphs, and decoder choice on
synthetic graphs drawn with data seed 1 (first-order and XOR contexts). It runs the search, writes the reweighted and
edge-CV datasets, trains in two stages (`e19a_selection_validation.json`: baseline starts and synthetic runs;
`e19b_selection_validation.json`: the other models and all CV runs at the baseline-chosen start), writes
[selection_validation_spec.json](configs/breakthrough/selection_validation_spec.json) and scores it. Seed 0, one split:

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
[selection_validation_summary.csv](configs/breakthrough/evidence/selection_validation_summary.csv),
[selection_validation_instances.csv](configs/breakthrough/evidence/selection_validation_instances.csv).
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
[few_label_selection_summary.csv](configs/breakthrough/evidence/few_label_selection_summary.csv),
[few_label_selection_instances.csv](configs/breakthrough/evidence/few_label_selection_instances.csv).

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

NMI on unlabeled nodes, mean of label draws 0–1, m = 20 / 200 (AIFB: m = 20; it has 176 labeled nodes):

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
[few_label_edge_methods.csv](configs/breakthrough/evidence/few_label_edge_methods.csv),
[few_label_dse_runs.csv](configs/breakthrough/evidence/few_label_dse_runs.csv).

### E22: letting the labels choose the route

`tools/few_label_route_selection.py` uses the same labeled nodes to choose between the structure route (`se_top_label`)
and the feature route (`node_clf`): 5-fold cross-validation inside the labeled set, the structure route mapping each
cluster to its training-fold majority label; the more accurate route is refit on all m labels. Eleven graphs, five label
draws per budget, NMI on unlabeled nodes:

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
[route_selection_summary.csv](configs/breakthrough/evidence/route_selection_summary.csv),
[route_selection_draws.csv](configs/breakthrough/evidence/route_selection_draws.csv).

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

## Quick start

```bash
uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python --torch-backend cu129 requirements.uv.lock.txt

# edge-free control, then the runner's default B31 (2000 epochs, hid 512)
.venv/bin/python tools/run_preset.py --preset baseline_v1 --dataset cora --seed 0 --gpu 0
.venv/bin/python tools/run_preset.py --dataset cora --seed 0 --gpu 0
.venv/bin/python tools/run_preset.py --list_presets   # baseline_v1, b31_dualscalar_assign, b45_v31_msgcond_gs050

# B31 + endpoint context; DBLP v2 must first be built (see the dataset guide)
.venv/bin/python tools/run_preset.py --dataset dblp_magnn_author_v2 --epochs 1000 --warm_start auto --edge_ctx_dim 32

# preview a command without training
.venv/bin/python tools/run_preset.py --preset baseline_v1 --dataset cora --dry_run

# mainline: label-free edge-weight search, then the edge-free DSE on the searched graph
.venv/bin/python tools/run_se_guided_baseline.py --datasets acm_multiplex_v2 --seeds 0 --spectral reg --workers 8 \
  --out_dir results/breakthrough_v1/search_acm
# take summary.csv's ctx_pick before the "/" (here layer+PAP); context picks also need the search's --ctx_dim
.venv/bin/python tools/write_se_reweighted_dataset.py --dataset acm_multiplex_v2 --pick=layer+PAP --out_name acm_sepick
.venv/bin/python tools/run_preset.py --preset baseline_v1 --dataset acm_sepick --max_nums 6 --epochs 1000 --warm_start auto
```

Run commands from the repository root. See [UV_ENVIRONMENTS.txt](UV_ENVIRONMENTS.txt) for the Python 3.11 /
PyTorch 2.8.0 CUDA 12.9 / matching torch-scatter environment. `requirements.txt` is the broad dependency list;
`requirements.uv.lock.txt` records the tested versions. This lock targets Linux/WSL with CUDA; it is not a generic CPU lock.
For CPU runs in a compatible environment use `main.py --no_gpu`; `--gpu 0` selects a GPU rather than disabling it.
The direct `main.py` training defaults differ from `run_preset.py`, so prefer a preset or a plan for comparisons.

`run_preset.py` starts cold by default. It does not automatically train both starts. For unknown dataset names, supply
`--max_nums`; use the JSON plans for paired starts and parameters such as `edge_weight_learn_apply_to` that the
convenience runner does not expose as command-line overrides.

## Reproducing the breakthrough pass

```bash
DATA=data
# synthetic benchmark (writes synth_ctx_v1_* into $DATA)
.venv/bin/python tools/prepare_synth_ctx_datasets.py --root_path "$DATA" --feature_sep 0.35 --homophily 0.45,0.55
# real multiplex benchmark (prepare before E6/E8/E9)
.venv/bin/python tools/prepare_pyg_multiplex_dataset.py --root_path "$DATA" --dataset acm --name acm_multiplex_v2 --psp_sample_k 20
# classic floors
.venv/bin/python tools/run_classic_baselines.py --root_path "$DATA" \
  --datasets cora,photo,dblp_magnn_author_v2,entities_aifb,urban_paris_plot_v3bsjg \
  --out_csv results/breakthrough_v1/e3_real_dual/classic_baselines.csv
.venv/bin/python tools/run_classic_baselines.py --root_path "$DATA" --datasets acm_multiplex_v2 \
  --out_csv results/breakthrough_v1/e6_b31_robust/classic_acm_v2.csv
# experiment plans (requires all named real datasets and the E2 expert code copies described below)
for plan in e1_backbone_gate e2d_plateau_diag e2_synth_ctx_warm e3_real_dual e4_b45_collapse_ablation \
            e6_b31_robust e7_ctx_edge e8_ctx_real; do
  .venv/bin/python tools/run_breakthrough.py --plan configs/breakthrough/$plan.json --root_path "$DATA" --max_parallel 4
done
# E9: real-graph subset; omit --only_tags to include the 48 synthetic jobs
.venv/bin/python tools/run_breakthrough.py --plan configs/breakthrough/e9_fixed_ruler.json --root_path "$DATA" \
  --only_tags b31fr_warm,b31fr_nowarm,b31frctx_warm,b31frctx_nowarm --max_parallel 4
.venv/bin/python tools/report_e2_mechanism.py --extra_dirs results/breakthrough_v1/e6_b31_robust,results/breakthrough_v1/e7_ctx_edge,results/breakthrough_v1/e9_fixed_ruler,results/breakthrough_v1/e11_safe_gate_pathways
# E11 safe gate / pathways; E13 learned heads on XOR contexts (build synth_ctxxor_v1 first)
.venv/bin/python tools/run_breakthrough.py --plan configs/breakthrough/e11_safe_gate_pathways.json --root_path "$DATA" --max_parallel 4
.venv/bin/python tools/prepare_synth_ctx_datasets.py --root_path "$DATA" --prefix synth_ctxxor_v1 --role_mode xor2 \
  --feature_sep 0.35 --homophily 0.45,0.55 --modes none,contextual --perm_modes contextual
.venv/bin/python tools/run_breakthrough.py --plan configs/breakthrough/e13_xor_context.json --root_path "$DATA" --max_parallel 4
# E12/E15 searches (CPU) and the spectral start-vector check
.venv/bin/python tools/run_se_guided_baseline.py --root_path "$DATA" --datasets Cora,Photo,dblp_magnn_author_v2,entities_aifb,acm_multiplex_v2 \
  --seeds 0,1,2 --spectral reg --beam 8 --max_order 2 --workers 16 --out_dir results/breakthrough_v1/e15_search_robustness/real_reg_beam
.venv/bin/python tools/check_spectral_stability.py --root_path "$DATA" --datasets Cora,Photo,dblp_magnn_author_v2,entities_aifb,acm_multiplex_v2 \
  --out_csv results/breakthrough_v1/e15_search_robustness/spectral_stability.csv
# E17/E18: edge-CV splits, retraining on the training graphs, selection-rule benchmark (the E14/E16 picks must exist)
.venv/bin/python tools/make_edge_cv_split.py --root_path "$DATA" --frac 0.2 --seed 0 --suffix _cv80 \
  --datasets cora,photo,dblp_magnn_author_v2,entities_aifb,acm_multiplex_v2,cora_sepick_reg,photo_sepick,dblp_sepick,aifb_sepick_reg,acm_sepick,synthctx_h45_sepick,synthctx_h55_sepick,synthxor_h45_sepick,synthxor_h55_sepick
.venv/bin/python tools/run_breakthrough.py --plan configs/breakthrough/e17_edge_cv.json --root_path "$DATA" --max_parallel 5
.venv/bin/python tools/select_partition_rules.py --root_path "$DATA" --spec configs/breakthrough/selection_rules_spec.json \
  --out_dir results/breakthrough_v1/e18_selection_rules
# E22: route choice by cross-validation inside the labeled set (CPU)
.venv/bin/python tools/few_label_route_selection.py --jobs configs/breakthrough/e21_few_label_edges_jobs.csv \
  --budgets 10,20,50,100,200 --draws 0,1,2,3,4 --workers 8 --out_dir results/breakthrough_v1/e22_route_selection
# E21: few-label edge semantics (methods 0-2 on CPU; method 3 trains few-label DSE runs)
.venv/bin/python tools/run_se_guided_baseline.py --datasets synth_ctx_v1_h45_direct_ds00,synth_ctx_v1_h55_direct_ds00 --seeds 0 \
  --spectral reg --workers 12 --out_dir results/breakthrough_v1/e21_few_label_edges/search_direct
.venv/bin/python tools/few_label_edge_semantics.py --jobs configs/breakthrough/e21_few_label_edges_jobs.csv \
  --budgets 10,20,50,100,200 --draws 0,1,2 --workers 6 --out_csv results/breakthrough_v1/e21_few_label_edges/methods012.csv
.venv/bin/python tools/run_breakthrough.py --plan configs/breakthrough/e21_few_label_dse.json --root_path "$DATA" --max_parallel 3
# E20: few-label selection over the E18/E19 decisions (reuses their checkpoints; caches partitions)
.venv/bin/python tools/few_label_selection.py --specs configs/breakthrough/selection_rules_spec.json configs/breakthrough/selection_validation_spec.json \
  --reference results/breakthrough_v1/e18_selection_rules/instances.csv results/breakthrough_v1/e19_selection_validation/rules/instances.csv \
  --budgets 5,10,20,50,100 --draws 200 --out_dir results/breakthrough_v1/e20_few_label_selection
# E19: frozen-rule validation on new decisions (search, datasets, two training stages, scoring); 2 jobs per 24 GB GPU
.venv/bin/python tools/run_selection_validation.py --max_parallel 2 --threads 4 --workers 16
# E14/E16: write each pick (table in configs/breakthrough/README.md), train, choose decoders
.venv/bin/python tools/run_breakthrough.py --plan configs/breakthrough/e16_search_then_train_reg.json --root_path "$DATA" --max_parallel 5
.venv/bin/python tools/se_select_decoder.py --spectral reg --run_dirs results/breakthrough_v1/e16_search_then_train_reg/runs/*/ \
  --out_csv results/breakthrough_v1/e16_search_then_train_reg/decoder_select_reg.csv
.venv/bin/python tools/report_e3_real.py --extra_dirs results/breakthrough_v1/e6_b31_robust,results/breakthrough_v1/e8_ctx_real,results/breakthrough_v1/e9_fixed_ruler \
  --extra_classic results/breakthrough_v1/e6_b31_robust/classic_acm_v2.csv \
  --datasets cora,photo,dblp_magnn_author_v2,entities_aifb,urban_paris_plot_v3bsjg,acm_multiplex_v2 \
  --out_dir results/breakthrough_v1/combined_real
# label-informed diagnostics (not training)
.venv/bin/python tools/screen_edge_headroom.py --datasets cora,photo,dblp_magnn_author_v2,acm_multiplex_v2 --out_csv results/breakthrough_v1/e5_headroom/headroom.csv
.venv/bin/python tools/check_se_alignment.py --datasets cora,photo,dblp_magnn_author_v2,acm_multiplex_v2 \
  --out_csv results/breakthrough_v1/e10_se_alignment/se_alignment.csv
.venv/bin/python tools/inspect_edge_factors.py --run_dir results/breakthrough_v1/e6_b31_robust/runs/b31_warm__acm_multiplex_v2__s0
```

`e2_synth_ctx_warm` runs EP1/EP2 from copies of the archived expert workspaces under `results/breakthrough_v1/_code/`
(the local copies add the warm start through an encoder hook and are not included in a clean Git clone). Restore those
copies to reproduce EP1/EP2; B31/B45 and baseline jobs use the root code. E1/E2d also reference deprecated synthetic
datasets, which can be rebuilt with `tools/prepare_mechanism_synth_datasets.py` for historical checks only.
DBLP, Entities and Urban prepared graphs are prerequisites; use the builders below and
[data/README.md](data/README.md). Urban raw city data must be supplied separately. Reports require their input
`runs.csv` and classic-baseline CSVs; a fresh clone has the compact evidence snapshots, not these raw outputs.

Use `--dry_run` to inspect a plan before launching it. Finished runs are cached by the presence of their metrics file;
use a new plan name or `--force` after changing code or parameters. Keep `--max_parallel` at 4 or lower for graphs above ~5k nodes: under WSL
an over-committed GPU pages to host memory instead of raising OOM and every job slows to a crawl.

Checkpoint diagnostics and the paired Photo training example are listed in [tools/README.md](tools/README.md).
The machine-check plan runs on the current host; it does not connect to another machine automatically.

## Dataset Guide (Type, Location, Commands)

Data is expected under `--root_path` (default `data/`). On the original workstation, `data/` is an fstab bind mount of a D: mirror folder; see
`DATA_ASSETS.md` for the physical location, the F: backup and verification.

For a concise inventory of what each current dataset means, see [DATASETS.md](DATASETS.md).

### 1) Auto-download datasets (no manual file placement)

These run directly and are downloaded by PyG loaders when missing:

- `cora`, `citeseer`, `pubmed`, `computers`, `photo`

```bash
.venv/bin/python tools/run_preset.py --dataset cora --seed 0 --gpu 0
```

### 2) Current active construction scripts

The active top-level `tools/` surface now keeps only the builders that still matter for the current mainline and current data-construction series.

#### DBLP (MAGNN-style author graph)
- raw: auto-downloaded by script
- output: `data/dblp_magnn_author` or `data/dblp_magnn_author_v2`

```bash
.venv/bin/python tools/prepare_pyg_dblp_magnn_dataset.py --root data/pyg_dblp --out_root data --name dblp_magnn_author_v2 --mode v2
```

#### Multiplex graphs and the current synthetic benchmark

```bash
.venv/bin/python tools/prepare_pyg_multiplex_dataset.py --dataset acm --name acm_multiplex_v2 --psp_sample_k 20
.venv/bin/python tools/prepare_synth_ctx_datasets.py --feature_sep 0.35 --homophily 0.45,0.55
```

The synthetic generator defaults to `feature_sep=0.25` and also includes h70; pass the explicit options above to
recreate the documented benchmark. ACM v2 samples PSP neighbours; use v2 for the reported E6–E9 comparisons.

#### Urban plot graphs
- raw expected: `data/urban_network_datasets/<city>/...`
- output: `data/urban_<city>_plot_v3*`

```bash
# semantics-first V3 graph builders
# v3s   : shared-street topology only
# v3sj  : shared-street + shared-junction topology
# v3sjg : v3sj + geometric fallback edges only for structurally under-connected plots
# design note:
# - feature similarity does not create edges in V3
# - shared-building overlap is kept only as an edge attribute
.venv/bin/python tools/prepare_urban_plot_graph_v3.py --city beijing --urban_root data/urban_network_datasets --out_root data --variant v3sjg

# batch-build all cities and all V3 variants
.venv/bin/python tools/run_urban_plot_graph_v3_series.py --force --summary_dir results/data_construction_series/urban_plot_v3_series

# refined V3b graph builders
# edge schema changes:
# - relation bits use street-backed / junction-only / geom-fallback / shared-building
# - endpoint context is symmetric via sum/gap features
# - geometry keeps orientation_diff
# - node feature relations (cosine, l2) stay descriptive only
.venv/bin/python tools/prepare_urban_plot_graph_v3b.py --city beijing --urban_root data/urban_network_datasets --out_root data --variant v3bsjg

# batch-build all cities and all V3b variants
.venv/bin/python tools/run_urban_plot_graph_v3b_series.py --force --summary_dir results/data_construction_series/urban_plot_v3b_series

# new plot-centric heterogeneous urban dataset
# node types:
# - plot, street, intersection, building
# typed relations:
# - plot-plot via shared streets
# - plot-street boundary relation
# - street-intersection incidence
# - plot-building containment
# - building-building kNN proximity
.venv/bin/python tools/prepare_urban_plot_hetero_v1.py --city beijing --urban_root data/urban_network_datasets --out_root data
.venv/bin/python tools/validate_urban_plot_hetero_v1.py --dataset_dir data/urban_beijing_plot_hetero_v1

# richer hetero v2 dataset
# keeps the same four node types, but upgrades each relation with typed edge_attr
# and adds separate spatial / semantic building-building relations
.venv/bin/python tools/prepare_urban_plot_hetero_v2.py --city paris --urban_root data/urban_network_datasets --out_root data
.venv/bin/python tools/validate_urban_plot_hetero_v1.py --dataset_dir data/urban_paris_plot_hetero_v2
.venv/bin/python tools/smoke_test_urban_plot_hetero_v2.py --dataset_dir data/urban_paris_plot_hetero_v2
```

Archived conversion scripts:

- older Entities / Fraud / Bitcoin WSN converters
- legacy Urban `V1/V2` builders
- older edge-control dataset helpers

These now live under `archive/code/historical_builders/` and should be used only for backfill or historical reproduction.

### 3) Custom dataset format (manual placement)

Create `data/<dataset_name>/` with sparse-format files:

```text
<name>_edge_index.npy   # [2, E], required
<name>_feat.npy         # [N, F], required
<name>_label.npy        # [N], required
<name>_edge_weight.npy  # [E], optional
<name>_edge_attr.npy    # [E, D], optional
<name>_meta.json        # optional
```

Recommended placement for your own datasets:

- Small/medium datasets: `data/<dataset_name>/`
- Multiple private datasets: `data/custom/<dataset_name>/` (and run with `--root_path data/custom`)
- Very large datasets on another disk: keep them outside repo and create a symlink into `data/`

Then run:

```bash
# If data is placed under data/<dataset_name>/
.venv/bin/python tools/run_preset.py --dataset <dataset_name> --max_nums 12 --seed 0 --gpu 0

# If data is placed under data/custom/<dataset_name>/
.venv/bin/python main.py --dataset <dataset_name> --root_path data/custom --max_nums 12 --gpu 0 --edge_variant V1 --epochs 2000 --hid_dim 512 --w_decay 0 --temperature 1.0 --warm_start auto

# Example (replace with your real name)
.venv/bin/python tools/run_preset.py --dataset my_graph_v1 --max_nums 12 --seed 0 --gpu 0
```

## Archive

- `archive/docs/README_2026-04_snapshot.md`: the pre-pass README (numbers not valid)
- `archive/docs/expert_suggestion_2026-04/`: Expert1/Expert2 guidance and code bundles (retired)
- `archive/configs/historical_presets/`: retired presets, still loadable by path
- `archive/code/`: retired runners, scripts and dataset builders
- `archive/workspaces/`: expert workspaces used for the April EP comparison
- `archive/results/historical/`: pre-B45 result families

## References

- Original DSE repository: https://github.com/RiemannGraph/DSE_clustering
- DSE/LSEnet paper: https://arxiv.org/abs/2504.09970 (local copy `reference/2504.09970v2.pdf`); LSEnet ICML 2024: `reference/sun24g.pdf`
- Original implementation explanation note: `reading/EXPLAIN_DSE_original_en.md`

## License

See `LICENSE`.
