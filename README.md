# Edge Enhanced Unsupervised Graph Neural Network Clustering

Unsupervised graph clustering built on DSE/LSEnet (Lorentz structural entropy), with edge-aware variants. The
question the project asks: when do edge attributes help structural-entropy clustering, and which model actually uses them?

## Status (2026-10-13): breakthrough pass

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
  each configuration is trained with and without warm start and the run with the lower final SE loss is kept. No step uses labels.
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
- No model uses contextual edges, although the oracle shows 0.84-0.96 NMI is available. This remains open.
- The DSE baseline trails SGC2+KMeans on these weak-feature graphs (0.60 vs 0.73 at h55, 0.08 vs 0.47 at h45).

**E3, real graphs** (`e3_real_dual`; 3 seeds for Cora/DBLP, 1 for the rest; selection = lower final SE loss between warm
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
- No real dataset in the repo yet shows edge attributes helping clustering beyond topology and features.

### Retired in this pass

| Item | Why |
|---|---|
| All rankings from ≤120-epoch runs (B45 > B40 > G20, EP1/EP2 verdicts, Urban NMI) | Backbone untrained; see above |
| `synth_mech_full_v1`, `synth_edgectrl_v1` as evidence; `tools/run_mechanism_synth_suite.py` (archived) | Features alone give NMI 1.0 |
| EP1 (Expert1 contextual edge state) | E2: contextual effect ≈ 0, context-off ablation as good |
| EP2 (Expert2 relation state, B50) | E2: collapses to one cluster even with a warm start |
| Expert3 shared-tree multi-channel prototype | Never beat its own early-scalarised ablation (e.g. 0.39 vs 0.86 on old h85); dense C×N×N adjacency |
| Presets B40, B47, B48, G20 (moved to `archive/configs/historical_presets/`) | B40/B47 are minor B45 variants, B48 was harmful, G20's SE path is inside B45 |
| `Expert suggestion/` (moved to `archive/docs/expert_suggestion_2026-04/`) | Both directions refuted |
| Urban land-use NMI as clustering evidence | Labels unrelated to graph communities; Urban stays a data-paper track |
| Paper-reported LSEnet numbers as targets | Not reproducible from the released code |

### Open problems, in priority order

1. **B45 robustness**: it collapses on Photo/AIFB and loses to the baseline when edges carry no signal. It needs a fallback
   to identity edge weights when the edge channel is uninformative, and a diagnosis of the collapse.
2. **Real-data evidence for edge attributes**: none yet. Either find graphs whose edge attributes carry information that
   topology and features do not, or position the work as a mechanism study (synthetic positives, real negatives).
3. **Contextual edge semantics**: a large oracle headroom that no current model reaches.
4. **Backbone vs simple baselines**: DSE trails SGC+KMeans on weak-feature graphs, and SE training can drift away from a good
   initial partition (h55: 0.73 → 0.60).
5. **Start selection for edge-aware models**: B45's training loss is computed on its own reweighted graph, so comparing it
   across starts (or against the baseline) needs a common criterion, e.g. 2-level SE of the final hard partition on the raw graph.

## Quick start

```bash
uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python --torch-backend cu129 requirements.uv.lock.txt

DATA="/mnt/f/WSL Projects Data/edge-enhanced-unsupervised-graph-neural-network-clustering/data"
# one run with the converged defaults (2000 epochs, hid 512); add --warm_start auto via main.py for weak-feature graphs
.venv/bin/python tools/run_preset.py --preset b45_v31_msgcond_gs050 --dataset cora --root_path "$DATA" --seed 0 --gpu 0
.venv/bin/python tools/run_preset.py --list_presets   # baseline_v1, b31_dualscalar_assign, b45_v31_msgcond_gs050
```

See `UV_ENVIRONMENTS.txt` for why torch is pinned to 2.8.0.

## Reproducing the breakthrough pass

```bash
DATA=/path/without/spaces/to/data   # symlink to the SN730 data folder
# synthetic benchmark (writes synth_ctx_v1_* into $DATA)
.venv/bin/python tools/prepare_synth_ctx_datasets.py --root_path "$DATA" --feature_sep 0.35 --homophily 0.45,0.55
# classic floors
.venv/bin/python tools/run_classic_baselines.py --root_path "$DATA" --datasets cora,photo,dblp_magnn_author_v2 \
  --out_csv results/breakthrough_v1/e3_real_dual/classic_baselines.csv
# experiment plans (parallel jobs on one GPU; finished runs are cached)
for plan in e1_backbone_gate e2d_plateau_diag e2_synth_ctx_warm e3_real_dual; do
  .venv/bin/python tools/run_breakthrough.py --plan configs/breakthrough/$plan.json --root_path "$DATA" --max_parallel 4
done
.venv/bin/python tools/report_e2_mechanism.py
.venv/bin/python tools/report_e3_real.py
```

`e2_synth_ctx_warm` runs EP1/EP2 from copies of the archived expert workspaces under `results/breakthrough_v1/_code/`
(the copies add the warm start through an encoder hook). Keep `--max_parallel` at 4 for graphs above ~5k nodes: under WSL
an over-committed GPU pages to host memory instead of raising OOM and every job slows to a crawl.

## Dataset Guide (Type, Location, Commands)

Data is expected under `--root_path` (default `data/`). The prepared datasets currently live on the SN730 drive at
`/mnt/f/WSL Projects Data/edge-enhanced-unsupervised-graph-neural-network-clustering/data`; the repo-local `data/`
mount points at an empty folder on D:, so pass `--root_path` explicitly (a space-free symlink to that folder is convenient).

For a concise inventory of what each current dataset means, see [DATASETS.md](DATASETS.md).

### 1) Auto-download datasets (no manual file placement)

These run directly and are downloaded by PyG loaders when missing:

- `cora`, `citeseer`, `pubmed`, `computers`, `photo`

```bash
python3 tools/run_preset.py --dataset cora --seed 0 --gpu 0
```

### 2) Current active construction scripts

The active top-level `tools/` surface now keeps only the builders that still matter for the current mainline and current data-construction series.

1. PyG DBLP (MAGNN-style author graph)
- raw: auto-downloaded by script
- output: `data/dblp_magnn_author` or `data/dblp_magnn_author_v2`

```bash
python3 tools/prepare_pyg_dblp_magnn_dataset.py --root data/pyg_dblp --out_root data --name dblp_magnn_author_v2 --mode v2
```

2. Urban plot graph
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
python3 tools/prepare_urban_plot_graph_v3.py --city beijing --urban_root data/urban_network_datasets --out_root data --variant v3sjg

# batch-build all cities and all V3 variants
python3 tools/run_urban_plot_graph_v3_series.py --force --summary_dir results/data_construction_series/urban_plot_v3_series

# refined V3b graph builders
# edge schema changes:
# - relation bits use street-backed / junction-only / geom-fallback / shared-building
# - endpoint context is symmetric via sum/gap features
# - geometry keeps orientation_diff
# - node feature relations (cosine, l2) stay descriptive only
python3 tools/prepare_urban_plot_graph_v3b.py --city beijing --urban_root data/urban_network_datasets --out_root data --variant v3bsjg

# batch-build all cities and all V3b variants
python3 tools/run_urban_plot_graph_v3b_series.py --force --summary_dir results/data_construction_series/urban_plot_v3b_series

# new plot-centric heterogeneous urban dataset
# node types:
# - plot, street, intersection, building
# typed relations:
# - plot-plot via shared streets
# - plot-street boundary relation
# - street-intersection incidence
# - plot-building containment
# - building-building kNN proximity
python3 tools/prepare_urban_plot_hetero_v1.py --city beijing --urban_root data/urban_network_datasets --out_root data
python3 tools/validate_urban_plot_hetero_v1.py --dataset_dir data/urban_beijing_plot_hetero_v1

# richer hetero v2 dataset
# keeps the same four node types, but upgrades each relation with typed edge_attr
# and adds separate spatial / semantic building-building relations
python3 tools/prepare_urban_plot_hetero_v2.py --city paris --urban_root data/urban_network_datasets --out_root data
python3 tools/validate_urban_plot_hetero_v1.py --dataset_dir data/urban_paris_plot_hetero_v2
python3 tools/smoke_test_urban_plot_hetero_v2.py --dataset_dir data/urban_paris_plot_hetero_v2
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
python3 tools/run_preset.py --dataset <dataset_name> --max_nums 12 --seed 0 --gpu 0

# If data is placed under data/custom/<dataset_name>/
python3 main.py --dataset <dataset_name> --root_path data/custom --max_nums 12 --gpu 0 --edge_variant V1 --epochs 2000 --hid_dim 512 --w_decay 0 --temperature 1.0 --warm_start auto

# Example (replace with your real name)
python3 tools/run_preset.py --dataset my_graph_v1 --max_nums 12 --seed 0 --gpu 0
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
