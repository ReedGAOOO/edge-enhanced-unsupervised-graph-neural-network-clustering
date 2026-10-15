# Edge Enhanced Unsupervised Graph Neural Network Clustering

Unsupervised graph clustering built on DSE/LSEnet (Lorentz structural entropy), with edge-aware variants. The
question the project asks: when do edge attributes help structural-entropy clustering, and which model actually uses them?

## Current status (2026-10-15)

The active branch is `breakthrough-2026-10`. The October pass now includes E1–E9, an SE-alignment screen (E10),
and controlled B45 message-gate diagnostics. The practical findings are:

- Use `baseline_v1` as the edge-free control. The convenience runner still defaults to `b31_dualscalar_assign`;
  that default is an edge-aware starting point, not a claim that B31 beats the baseline on every dataset.
- B31 with endpoint context can use contextual synthetic edge signals and improves some real-graph results.
  It still trails the baseline on Photo/AIFB under the selected-start protocol.
- E9's `--edge_weight_learn_apply_to msg_only` removes learned scalar weights from the SE graph. It recovers B31's
  DBLP result to baseline level, but does not generally improve real graphs. **44 real-graph runs are complete;
  the 48 planned synthetic runs have no recorded results.**
- B45's Photo collapse is linked to train/decode embedding mismatch, amplified by gate normalization and isolated-node
  geometry. Short, single-seed interventions recover NMI ≈ 0.642; these are diagnostic hooks, not a production fix.
- Training and start selection use SE rather than NMI. The benchmark supplies a cluster budget; the default warm-start
  pseudo-partition size uses the dataset's class count. Label-based headroom and alignment screens are diagnostics only.

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
The plan also contains 48 synthetic runs, but the local result ledger contains only the 44 real-graph runs above.
There is no E9 synthetic conclusion yet.

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

### B45 follow-up: collapse mechanism, not a merged fix

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
by resource contention and is excluded. This is single-seed diagnostic evidence; the `safe_isolates` hook and normalization
interventions have **not** been adopted in `modules/layers.py` or validated at the full benchmark budget.
See the [intervention](configs/breakthrough/evidence/b45_gate_interventions.csv),
[graph-cross](configs/breakthrough/evidence/b45_graph_cross.csv) and
[graph-statistics](configs/breakthrough/evidence/b45_graph_stats.csv) tables.

### Retired in this pass

| Item | Why |
|---|---|
| All rankings from ≤120-epoch runs (B45 > B40 > G20, EP1/EP2 verdicts, Urban NMI) | Backbone untrained; see above |
| `synth_mech_full_v1`, `synth_edgectrl_v1` as evidence; `tools/run_mechanism_synth_suite.py` (archived) | Features alone give NMI 1.0 |
| EP1 (Expert1 contextual edge state) | E2: contextual effect ≈ 0, context-off ablation as good |
| EP2 (Expert2 relation state, B50) | E2: collapses to one cluster even with a warm start |
| Expert3 shared-tree multi-channel prototype | Never beat its own early-scalarised ablation (e.g. 0.39 vs 0.86 on old h85); dense C×N×N adjacency |
| Presets B40, B47, B48, G20 (moved to `archive/configs/historical_presets/`) | B40/B47 are minor B45 variants, B48 was harmful, G20's SE path is inside B45 |
| B45 as the default preset (kept, not retired) | Its message gate collapses training on Photo/AIFB (E4); B31 is the default |
| IMDB multiplex | Genre labels unrecoverable from any layer |
| `Expert suggestion/` (moved to `archive/docs/expert_suggestion_2026-04/`) | Both directions refuted |
| Urban land-use NMI as clustering evidence | Labels unrelated to graph communities; Urban stays a data-paper track |
| Paper-reported LSEnet numbers as targets | Not reproducible from the released code |

### Open problems, in priority order

1. **Aligning the objective and edge selection with useful structure**: E9 removes scalar reweighting from the SE graph
   without a general gain. E10 shows that SE-preferred candidates need not be the label-relevant ones. Feature consistency
   and per-type volume constraints remain hypotheses to test, especially on ACM.
2. **Fixing train/decode geometry before redesigning the gate**: validate isolated-row handling and graph normalization
   in a consistent training/inference path, then rerun B45 at full budget across seeds. The Photo hooks are promising
   diagnostics, not a replacement benchmark. Preserve direct/contextual synthetic edge gains when testing a fix.
3. **Model selection without labels**: the baseline-guided start rule works; choosing between baseline, B31+ctx and B45 does not
   (modularity and SE loss both prefer the wrong run on DBLP).
4. **Backbone vs simple baselines**: DSE still trails spectral clustering on ACM (0.46-0.52 vs 0.566) and SGC on weak-feature
   synthetic graphs, and SE training can move away from a good initial partition (ACM 0.566 → 0.46, synth h55 0.73 → 0.60).
5. **More real graphs with edge-type headroom**: ACM is the only one so far; other multiplex/heterogeneous graphs with
   recoverable labels are the next datasets to screen with `tools/screen_edge_headroom.py` and a layer-subset check.

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
# E9: run the real-graph subset with recorded evidence; omit --only_tags to include the uncompleted synthetic jobs
.venv/bin/python tools/run_breakthrough.py --plan configs/breakthrough/e9_fixed_ruler.json --root_path "$DATA" \
  --only_tags b31fr_warm,b31fr_nowarm,b31frctx_warm,b31frctx_nowarm --max_parallel 4
.venv/bin/python tools/report_e2_mechanism.py --extra_dirs results/breakthrough_v1/e6_b31_robust,results/breakthrough_v1/e7_ctx_edge
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
