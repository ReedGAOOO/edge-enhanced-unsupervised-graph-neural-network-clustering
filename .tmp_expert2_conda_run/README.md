# Edge Enhanced Unsupervised Graph Neural Network Clustering

Unsupervised graph clustering built on DSE/LSEnet, with edge-aware variants centered on structural entropy (SE).

## Current Mainline

- Default mainline: `B45` (`b45_v31_msgcond_gs050`)
- Reference SE-consistent branch: `G20` (`g20_se_consistent_main`)
- Minimal edge-aware ablation: `B31` (`b31_dualscalar_assign`)
- Conservative message variant: `B47` (`b47_v31_msgcond_gs050_matchonly`)
- Baseline: `V1` (`baseline_v1`)

`tools/run_preset.py` now defaults to `b45_v31_msgcond_gs050`.

`B45` is the current mainline because it changes the baseline at the three highest-leverage points without rewriting the structural-entropy objective itself: it keeps the original scalar SE loss, adds dual scalar edge weights for `adj_msg/adj_si`, preserves the calibrated `V31` assignment residual, and further introduces edge-conditioned message passing at the leaf encoder so that `edge_attr` can directly reshape the message graph before the partition tree is grown. Compared with the baseline, this is the first version in the repo that uses edge information simultaneously in representation learning, assignment, and SE-consistent structure weighting while still staying numerically stable through bounded gates, graph re-normalization, and the unchanged `_si_loss()` trunk.

## Quick Start

```bash
# 1) clone
git clone https://github.com/ReedGAOOO/edge-enhanced-unsupervised-graph-neural-network-clustering.git
cd edge-enhanced-unsupervised-graph-neural-network-clustering

# 2) environment
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 3) default run (B45)
python3 tools/run_preset.py --dataset cora --seed 0 --gpu 0

# optional: explicit presets
python3 tools/run_preset.py --preset b45_v31_msgcond_gs050 --dataset cora --seed 0 --gpu 0
python3 tools/run_preset.py --preset g20_se_consistent_main --dataset cora --seed 0 --gpu 0
python3 tools/run_preset.py --preset b31_dualscalar_assign --dataset cora --seed 0 --gpu 0
python3 tools/run_preset.py --preset b47_v31_msgcond_gs050_matchonly --dataset cora --seed 0 --gpu 0
python3 tools/run_preset.py --preset baseline_v1 --dataset cora --seed 0 --gpu 0

# optional: show all preset names
python3 tools/run_preset.py --list_presets
```

## Structure Evolution: Baseline -> B31 -> B40 -> B45

### Current recommendation

| Role | Preset | Meaning |
|---|---|---|
| Mainline | `b45_v31_msgcond_gs050` | strongest current control-suite model |
| Reference branch | `g20_se_consistent_main` | SE-consistent scalar edge weighting |
| Minimal ablation | `b31_dualscalar_assign` | no message gate / no hierarchy / no augment |
| Conservative variant | `b47_v31_msgcond_gs050_matchonly` | `B45` + matched-edge-only gate |
| Baseline | `baseline_v1` | no explicit edge-aware path |

### Structural comparison

| Item | Baseline (`V1`) | G20 (`V20`) | B45 (`V31 + msg-cond`) |
|---|---|---|---|
| Edge feature source | ignored in score path | scalarized from `edge_attr` | scalarized + direct message gate from `edge_attr` |
| Main injection stage | node-only leaf encoding + trunk assignment | SE graph measure first (`adj_msg/adj_si` reweight) | leaf message graph + `V31` assignment residual + scalar SE weights |
| Relation to SE objective | indirect | direct | direct, but without rewriting `_si_loss()` |
| High-leverage edge path | none | structure reweight only | structure reweight + edge-conditioned message passing |
| Why it stays stable | simplest path | bounded log-ratio + `edge_reg` | bounded message gate + degree re-normalization + `V31` stable trunk |
| Internal interpretable stats | N/A | `edge_factor_mean/std`, `edge_reg` | `msg_gate_factor_mean/std`, `edge_factor_mean/std`, `graph_alpha`, `edge_reliability`, `edge_mix_beta` |
| Key files | `modules/layers.py`, `modules/dsi.py` | `modules/dsi.py` | `modules/layers.py`, `modules/model.py`, `modules/dsi.py` |

### Evolution diagram

```text
[Baseline / V1]
  node-only leaf encoding
  + trunk assignment
  + scalar SE loss on adjacency
        |
        | add edge-aware scalar structure path
        v
[B31 / V31]
  baseline
  + dual scalar edge weights for adj_msg / adj_si
  + V31 assignment residual
  + calibrated graph_alpha / reliability / mix_beta
        |
        | move edge usage earlier into representation learning
        v
[B40 / V31 + msg-cond(0.35)]
  B31
  + edge-conditioned message passing on leaf graph
  + edge_attr -> gate factor -> reweighted adj_msg -> LorentzConv
        |
        | strengthen message gate to effective range
        v
[B45 / V31 + msg-cond(0.50)]
  B40
  + stronger edge message gate
  + same stable V31 assignment + scalar SE path
        |
        `-> optional conservative variant:
            [B47 = B45 + matched-edge-only gate]

Parallel reference branch:

[G20 / V20]
  baseline
  + scalar SE-consistent edge weighting
  + no explicit message-conditioned gate
```

### Why B45 is the current mainline

- It uses edge information at the three highest-leverage points:
  - leaf message graph
  - assignment score
  - scalar SE graph
- It keeps `_si_loss()` unchanged and therefore avoids destabilizing the structural entropy objective.
- It wins over `G20` on the current full control suite (`9-grid × 3 seeds`) while keeping all required branches alive.

### Current evidence

Source:
- `results/diagnostic_b45_confirm_grid9_v1/summary_by_condition.csv`
- `results/diagnostic_b45_confirm_grid9_v1/summary_by_condition_dataset.csv`
- `results/diagnostic_b47b48_repr3_v1/summary_by_condition.csv`

`9-grid × 3 seeds` overall `NMI / ARI`:

| Condition | NMI mean | ARI mean |
|---|---:|---:|
| `b45_v31_msgcond_gs050` | `0.16001` | `0.15256` |
| `b40_v31_msgcond` | `0.14575` | `0.13826` |
| `g20_se_consistent_main` | `0.11657` | `0.10914` |

Homophily-wise:

- `h45`: `B45` is not clearly better than `G20`
- `h65`: `B45` becomes slightly better than `G20`
- `h85`: `B45` is clearly stronger than both `B40` and `G20`

Mode-wise:

- `mhier`: `B45 > B40 > G20`
- `mmisl`: `B45 > B40 > G20`
- `mredu`: `B45 > B40 > G20`

Small follow-up on `h65` (`B47/B48`):

- `B47 = B45 + matched-edge-only` gives only a very small gain over `B45`
- `B48 = B45 + confidence gate` is harmful

So the current message-branch ordering is:

1. `B45`
2. `B47`
3. `G20`
4. `B40`
5. `B48`

### Minimal dataflow summary

```text
Baseline / V1
  node features -> Lorentz leaf encoder -> trunk assignment -> scalar SE loss

G20 / V20
  edge_attr -> scalar edge factors -> reweighted adj_msg/adj_si -> trunk assignment -> scalar SE loss

B45 / V31 + msg-cond
  edge_attr -> message gate -> reweighted leaf message graph
          +-> dual scalar edge factors for adj_msg/adj_si
          +-> calibrated V31 assignment residual
  -> partition tree -> unchanged scalar SE loss
```

For the full experimental trail from `Baseline -> B31 -> B40 -> B45 -> B47/B48`, see `deep-research-report.md`.
## Dataset Guide (Type, Location, Commands)

All data is expected under repo-local `data/` (default `--root_path data`).

### 1) Auto-download datasets (no manual file placement)

These run directly and are downloaded by PyG loaders when missing:

- `cora`, `citeseer`, `pubmed`, `computers`, `photo`

```bash
python3 tools/run_preset.py --dataset cora --seed 0 --gpu 0
```

### 2) Raw-source datasets that need conversion first

Place raw files in the indicated folder, then run conversion:

1. PyG Entities (`AIFB/MUTAG/BGS/AM`)
- raw: auto-downloaded by script
- output: `data/entities_*`

```bash
python3 tools/prepare_pyg_entities_datasets.py --root data --out_root data --datasets AIFB,MUTAG,BGS,AM
```

2. PyG DBLP (MAGNN-style author graph)
- raw: auto-downloaded by script
- output: `data/dblp_magnn_author` or `data/dblp_magnn_author_v2`

```bash
python3 tools/prepare_pyg_dblp_magnn_dataset.py --root data/pyg_dblp --out_root data --name dblp_magnn_author_v2 --mode v2
```

3. Fraud Amazon/Yelp
- raw expected: `data/FraudAmazon/Amazon.mat`, `data/FraudYelp/YelpChi.mat`
- output: `data/fraud_amazon_union`, `data/fraud_yelp_homo`

```bash
python3 tools/prepare_fraud_datasets.py --out_root data --datasets amazon,yelp --base_mode auto
```

4. Bitcoin WSN
- raw expected: CSVs under `data/Bitcoin_WSN/data-wsn/`
- output: `data/bitcoin_wsn_*`

```bash
python3 tools/prepare_bitcoin_wsn_datasets.py --src_dir data/Bitcoin_WSN/data-wsn --out_root data --datasets otc,alpha,rfa,wikisigned,epinion
```

5. Urban plot graph
- raw expected: `data/urban_network_datasets/<city>/...`
- output: `data/urban_<city>_plot`, `data/urban_<city>_plot_v2`, or `data/urban_<city>_plot_v3*`

```bash
# legacy graph builder
python3 tools/prepare_urban_plot_graph.py --city beijing --urban_root data/urban_network_datasets --out_root data --dataset_name urban_beijing_plot --topk_per_node 32

# recommended v2 graph builder
# default behavior:
# - excludes land-use score columns from node features
# - adds street-junction and feature-KNN edge sources
# - keeps label files compatible with known-only evaluation
python3 tools/prepare_urban_plot_graph_v2.py --city beijing --urban_root data/urban_network_datasets --out_root data --dataset_name urban_beijing_plot_v2

# semantics-first V3 graph builders
# v3s   : shared-street topology only
# v3sj  : shared-street + shared-junction topology
# v3sjg : v3sj + geometric fallback edges only for structurally under-connected plots
# design note:
# - feature similarity does not create edges in V3
# - shared-building overlap is kept only as an edge attribute
python3 tools/prepare_urban_plot_graph_v3.py --city beijing --urban_root data/urban_network_datasets --out_root data --variant v3sjg

# batch-build all cities and all V3 variants
python3 tools/run_urban_plot_graph_v3_series.py --force --summary_dir results/urban_plot_v3_series

# refined V3b graph builders
# edge schema changes:
# - relation bits use street-backed / junction-only / geom-fallback / shared-building
# - endpoint context is symmetric via sum/gap features
# - geometry keeps orientation_diff
# - node feature relations (cosine, l2) stay descriptive only
python3 tools/prepare_urban_plot_graph_v3b.py --city beijing --urban_root data/urban_network_datasets --out_root data --variant v3bsjg

# batch-build all cities and all V3b variants
python3 tools/run_urban_plot_graph_v3b_series.py --force --summary_dir results/urban_plot_v3b_series
```

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
python3 main.py --dataset <dataset_name> --root_path data/custom --max_nums 12 --gpu 0 --edge_variant V20

# Example (replace with your real name)
python3 tools/run_preset.py --dataset my_graph_v1 --max_nums 12 --seed 0 --gpu 0
```

## Why custom datasets are necessary

Native edge-attribute graph clustering datasets are limited. Many common benchmarks provide topology and node features but no high-quality native `edge_attr`.  
To analyze model mechanism instead of leaderboard score, this repo uses controlled custom datasets for causal-style validation.

### Custom dataset principle

The synthetic mechanism datasets are intervention-oriented:

- Keep node-label generation explicit (known communities).
- Control graph homophily (`homophily_target`).
- Control edge-attribute semantic strength (`edge_signal_target`).
- Keep all outputs in normal training format (`edge_index`, `edge_weight`, `edge_attr`, `feat`, `label`, `meta`).

Generator script:

```bash
python3 tools/prepare_mechanism_synth_datasets.py \
  --prefix synth_mech_full_v1 \
  --num_nodes 1200 \
  --num_classes 4 \
  --feat_dim 64 \
  --avg_degree 16 \
  --homophily 0.85,0.65,0.45 \
  --edge_signal 0.9,0.6,0.3,0.0 \
  --data_seeds 0
```

## Historical notes

Earlier `G20/G15` control experiments are kept for reference only:

- `results/benchmark_mechanism_synth_full_v1/summary_by_condition.csv`
- `results/benchmark_mechanism_synth_full_v1/stat_tests_summary.json`
- `results/benchmark_mechanism_permEA_v1/permutation_effect_summary.csv`

They document how the project moved from:

- `G15`: assignment-side calibrated residual
- `G20`: SE-consistent scalar edge weighting

to the current `B45` message-aware mainline.

## Repro commands

### Full mechanism benchmark

```bash
python3 tools/run_mechanism_synth_suite.py \
  --tag benchmark_mechanism_synth_full_v1 \
  --prefix synth_mech_full_v1 \
  --conditions baseline_v1,g15_echf_main,g20_se_consistent_main \
  --baseline_condition baseline_v1 \
  --seeds 0,1,2 \
  --epochs 80 \
  --eval_freq 80 \
  --train_log_interval 20 \
  --hid_dim 256 \
  --max_nums 12 \
  --gpu 0 \
  --amp_bf16 \
  --resume
```

### Permutation auxiliary benchmark

```bash
python3 tools/run_mechanism_synth_suite.py \
  --tag benchmark_mechanism_permEA_v1 \
  --datasets synth_mech_full_v1_h85_s90_ds00,synth_mech_full_v1_h85_s90_ds00_permEA,synth_mech_full_v1_h65_s90_ds00,synth_mech_full_v1_h65_s90_ds00_permEA,synth_mech_full_v1_h45_s00_ds00,synth_mech_full_v1_h45_s00_ds00_permEA \
  --conditions g15_echf_main,g20_se_consistent_main \
  --baseline_condition g15_echf_main \
  --seeds 0,1 \
  --epochs 80 \
  --eval_freq 80 \
  --train_log_interval 20 \
  --hid_dim 256 \
  --max_nums 12 \
  --gpu 0 \
  --amp_bf16 \
  --resume
```

## References

- Original DSE repository: https://github.com/RiemannGraph/DSE_clustering
- DSE/LSEnet paper: https://arxiv.org/abs/2504.09970
- Local paper copy: `reference/2504.09970v2.pdf`
- Original implementation explanation note: `reading/EXPLAIN_DSE_original_en.md`

## License

See `LICENSE`.
