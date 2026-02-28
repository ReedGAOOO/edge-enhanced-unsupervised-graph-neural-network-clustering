# Edge Enhanced Unsupervised Graph Neural Network Clustering

Edge-feature-aware unsupervised graph clustering built on DSE/LSEnet, with a mainline architecture focused on **calibrated edge fusion + hierarchical edge propagation**.

## Mainline (Renamed)

Current project mainline is the **ECHF family**:
- **ECHF** = **E**dge-**C**alibrated **H**ierarchical **F**usion
- Branch mainline: `B15_ECHF_main`
- Preset mainline: `g15_echf_main` (default)

Backward-compatible aliases are still supported.

| New canonical name | Legacy name |
|---|---|
| `B15_ECHF_main` | `B15_PATHB_v12_hier` |
| `B15_ECHF_s60` | `B15_PATHB_v12_hier_sched60` |
| `G15_ECHF_main` | `G15_default_hetero` |
| `G15_ECHF_noadapt` | `G15_noadapt_hetero` |
| `G17_V5_temp15` | `G17_temp1p5_mainline` |

## What Changed vs Base DSE

Compared with the original DSE-style assignment path, the mainline adds:
1. **V12 calibrated residual fusion** at assignment-score stage.
2. **Path-B hierarchical edge-attribute propagation** across coarsened graph levels.
3. **Adaptive bounded global fusion strength** (`graph_alpha`) for heterogeneity control.

This keeps DSE's structural-entropy optimization core while making edge features usable without hard over-injection.

## Visual Comparison: Original LSENet vs ECHF

### Original LSENet (conceptual path)

```mermaid
flowchart TD
    X[Node features X] --> L0[Leaf embedding in Lorentz space]
    A0[Adjacency A] --> L0
    L0 --> H[Hierarchical layers h=H..1]
    A0 --> H
    H --> S[Assignment S_h]
    S --> XP[Parent embedding X_{h-1} = normalize(S_h^T X_h)]
    S --> AP[Parent graph A_{h-1} = S_h^T A_h S_h]
    XP --> H
    AP --> H
    H --> R[Root + cluster tree]
    R --> SI[Structural entropy loss]
```

### ECHF mainline (V12 + Path-B)

```mermaid
flowchart TD
    X[Node features X] --> D[Data loader edge prep]
    Araw[Raw graph] --> D
    Eraw[Raw/provided edge_attr] --> D
    D --> Amsg[Message graph adj_msg]
    D --> Asi[SI graph adj_si]
    D --> E0[Edge attributes edge_attr_0]

    Amsg --> L0[Leaf embedding in Lorentz space]
    X --> L0

    L0 --> H[Hierarchical layers h=H..1]
    Asi --> H
    E0 --> H

    H --> V12[V12 assignment scoring:
    struct trunk + calibrated edge residual]
    V12 --> S[Assignment S_h]
    S --> XP[Parent embedding X_{h-1}]
    S --> AP[Parent graph A_{h-1}]
    S --> EP[Path-B edge_attr coarsen to edge_attr_{h-1}]

    XP --> H
    AP --> H
    EP --> H

    H --> R[Root + cluster tree]
    R --> SI[Structural entropy loss]
```

### Layer-level message passing (intuitive)

1. Edge score on current graph edges:
   `score_ij = -d_L(q_i, k_j) + edge_fusion_term_ij`
2. Source-wise edge normalization:
   `att_ij = softmax_j(score_ij)`
3. Assignment propagation:
   `S = att @ S_init`, then gumbel-softmax hardening
4. Node coarsening:
   `X_parent = normalize(S^T X)`
5. Graph coarsening:
   `A_parent = S^T A S`
6. ECHF-only Path-B:
   `edge_attr_parent = pool(edge_attr_current, S, A_parent)`

### What is the key difference?

| Aspect | Original LSENet path | ECHF path |
|---|---|---|
| Edge usage in assign score | Mostly structure-only | V12: structure trunk + calibrated edge residual |
| Edge attribute usage across levels | No explicit hierarchical propagation | Yes (`edge_attr_hierarchical=true`) |
| Fusion strength control | Fixed or weaker control | `gamma` schedule + adaptive `graph_alpha` |
| Objective | Structural entropy | Structural entropy (same core), but with edge-aware assignment dynamics |

### Minimal code map for this comparison

- Encoder hierarchy: `modules/model.py`
- Assignment/message path: `modules/layers.py`
- Structural entropy objective: `modules/dsi.py`
- Edge construction and graph split (`adj_msg` / `adj_si`): `data.py`

## Quick Start

```bash
cd /home/aitx/workspace/projects/edge-enhanced-unsupervised-graph-neural-network-clustering
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch -r requirements.txt
```

### Run the default mainline

```bash
python3 tools/run_preset.py --preset g15_echf_main --dataset cora --seed 0 --gpu 0
```

### Run branch-mainline-equivalent preset

```bash
python3 tools/run_preset.py --preset b15_echf_branch --dataset cora --seed 0 --gpu 0
```

### Urban comparison (known-only)

```bash
python3 tools/run_urban_branch_compare.py \
  --conditions G15_ECHF_main,G17_V5_temp15 \
  --seeds 0,1,2 \
  --epochs 60 \
  --known_only_eval \
  --tag urban_known_struct_g15g17_e60_v1
```

## Dataset Availability and Preparation

Default data root is `data/`.

### Availability check in this workspace (2026-02-28)

| Family | Auto-download | Raw source expected | Converted dataset names used by training | Current local status |
|---|---|---|---|---|
| Classic PyG (Planetoid/Amazon) | Yes (PyG) | none (download on first run) | `cora`, `citeseer`, `pubmed`, `computers`, `photo` | available |
| PyG Entities | Yes (PyG) | none (download by script/PyG) | `entities_aifb`, `entities_mutag`, `entities_bgs`, `entities_am` | available |
| PyG DBLP (MAGNN split) | Yes (PyG) | none (download by script/PyG) | `dblp_magnn_author`, `dblp_magnn_author_v2` | available |
| CARE-GNN Fraud | No (manual) | `data/FraudAmazon/Amazon.mat`, `data/FraudYelp/YelpChi.mat` | `fraud_amazon_union`, `fraud_yelp_homo` | available |
| Bitcoin WSN | No (manual) | CSV files under `data/Bitcoin_WSN/data-wsn/` | `bitcoin_wsn_*` | available |
| Urban plot graph | No (manual mount/sync) | `data/urban_network_datasets/<city>/...` | `urban_<city>_plot` | available (symlinked to `/mnt/e/urban network datasets`) |

If you need a quick machine check, run:

```bash
python3 - << 'PY'
from pathlib import Path
root = Path("data")
print("dataset,ready")
for d in sorted([p for p in root.iterdir() if p.is_dir()]):
    n = d.name
    dense = (d/f"{n}_adj.npy").exists() and (d/f"{n}_feat.npy").exists() and (d/f"{n}_label.npy").exists()
    sparse = (d/f"{n}_edge_index.npy").exists() and (d/f"{n}_feat.npy").exists() and (d/f"{n}_label.npy").exists()
    if dense or sparse:
        print(f"{n},yes")
PY
```

### Preprocessing method by dataset family

1. Classic PyG (`cora/citeseer/pubmed/computers/photo`)
- Raw datasets usually have no native `edge_attr`.
- Loader builds generic edge features from node similarity + degree statistics and standardizes them.
- Edge weighting path is controlled by `--edge_variant` and related fusion args.

2. Entities (`AIFB/MUTAG/BGS/AM`)
- Source: `torch_geometric.datasets.Entities`.
- Converts `edge_type` into edge channels (`relation_id_norm`, `relation_freq`) and builds structural node features.
- Labels are merged into full-node vector with unknown as `-1`.

3. DBLP (MAGNN author graph)
- Source: `torch_geometric.datasets.DBLP`.
- `v2` construction uses semantically stronger channels:
  - APA PathSim
  - TERM TF-IDF cosine
  - CONF TF-IDF cosine
- Outputs weighted edges and 3-channel edge attributes.

4. Fraud (Amazon/Yelp)
- Source: local `.mat` files.
- Builds base graph (`union` for Amazon, `homo` for Yelp by default).
- Edge channels: `[homo, rel1, rel2, rel3]`.
- Node features are stabilized with signed log compression + z-score.

5. Bitcoin WSN
- Source: local signed weighted CSV edge list.
- Keeps signed semantics via edge attributes `[w_norm, |w|, sign(w)]`.
- Uses `|w|` as scalar edge weight; builds structural node features from in/out degree and signed balance.
- For very large sets, use `tools/make_wsn_topk_subgraph.py` to build top-degree induced subgraphs.

6. Urban plot graphs
- Source per city: `plot.parquet`, `street.parquet`, `plot_street_id.npy`, `plot_bid.npy`.
- Builds edges from shared street/building memberships.
- Edge attributes include shared counts, Jaccard terms, degree terms, and node-feature relation terms (12 dims total).
- Produces `urban_<city>_plot` sparse dataset files.

### One-command conversion (from raw source to trainable dataset)

1. Entities

```bash
python3 tools/prepare_pyg_entities_datasets.py \
  --root data \
  --out_root data \
  --datasets AIFB,MUTAG,BGS,AM
```

2. DBLP (recommended v2)

```bash
python3 tools/prepare_pyg_dblp_magnn_dataset.py \
  --root data/pyg_dblp \
  --out_root data \
  --name dblp_magnn_author_v2 \
  --mode v2 \
  --k_apa 32 --k_term 32 --k_conf 12 \
  --min_apa 0.05 --min_term 0.05 --min_conf 0.05 \
  --w_apa 0.45 --w_term 0.35 --w_conf 0.20
```

3. Fraud

```bash
python3 tools/prepare_fraud_datasets.py \
  --datasets amazon,yelp \
  --out_root data \
  --base_mode auto
```

4. Bitcoin WSN

```bash
python3 tools/prepare_bitcoin_wsn_datasets.py \
  --src_dir data/Bitcoin_WSN/data-wsn \
  --out_root data \
  --datasets otc,alpha,rfa,wikisigned,epinion
```

Optional large-graph subgraph:

```bash
python3 tools/make_wsn_topk_subgraph.py \
  --src_dataset bitcoin_wsn_epinion \
  --topk_nodes 5000 \
  --out_dataset bitcoin_wsn_epinion_top5k
```

5. Urban city graph

```bash
python3 tools/prepare_urban_plot_graph.py \
  --city beijing \
  --urban_root data/urban_network_datasets \
  --out_root data \
  --dataset_name urban_beijing_plot \
  --topk_per_node 32
```

### Training run commands by dataset family

1. Classic PyG / converted sparse datasets with known `max_nums`

```bash
python3 tools/run_preset.py --preset g15_echf_main --dataset cora --seed 0 --gpu 0
python3 tools/run_preset.py --preset g15_echf_main --dataset entities_aifb --seed 0 --gpu 0
python3 tools/run_preset.py --preset g15_echf_main --dataset dblp_magnn_author_v2 --seed 0 --gpu 0
python3 tools/run_preset.py --preset g15_echf_main --dataset fraud_amazon_union --seed 0 --gpu 0
```

2. Custom datasets not in `run_preset` auto-map (pass `--max_nums` explicitly)

```bash
python3 tools/run_preset.py --preset g15_echf_main --dataset bitcoin_wsn_otc --max_nums 8 --seed 0 --gpu 0
python3 tools/run_preset.py --preset g15_echf_main --dataset urban_beijing_plot --max_nums 64 --seed 0 --gpu 0
```

3. Urban full-structure benchmark (recommended for city sets)

```bash
python3 tools/run_urban_branch_compare.py \
  --conditions G15_ECHF_main,G17_V5_temp15 \
  --seeds 0,1,2 \
  --epochs 60 \
  --known_only_eval \
  --tag urban_known_struct_g15g17_e60_v1
```

### Custom dataset format requirements

The loader supports two styles under `data/<dataset_name>/`.

1. Dense style

```text
<name>_adj.npy      # [N, N]
<name>_feat.npy     # [N, F]
<name>_label.npy    # [N]
```

2. Sparse style (recommended)

```text
<name>_edge_index.npy   # [2, E]
<name>_edge_weight.npy  # [E] (optional)
<name>_edge_attr.npy    # [E, D] (optional)
<name>_feat.npy         # [N, F]
<name>_label.npy        # [N]
<name>_meta.json        # optional (e.g., unknown label mapping)
```

Run custom dataset:

```bash
python3 tools/run_preset.py --preset g15_echf_main --dataset your_dataset --max_nums 10 --gpu 0
```

## Unified Experiment Organization

All experiment-version notes and stage summaries are now under `exp/`:

```text
exp/
  README.md
  versions/                 # B/G branch structure notes
  stage_results/            # per-stage summary CSV snapshots + normalized tables
  reports/                  # consolidated narrative report
```

Start from:
- `exp/README.md`
- `exp/stage_results/README.md`
- `exp/reports/MAINLINE_ECHF_B15_G15_2026-02-19.md`

## Key Evidence Snapshot (Extracted)

### Stage-2 (Path-A / Path-B top4 confirm)
Source: `exp/stage_results/stage2_pathab_top4_confirm180/summary_by_condition_normalized.csv`

| Condition | Mean NMI | Mean ARI | Rank Score |
|---|---:|---:|---:|
| `G17_V5_temp15` | 0.2297 | 0.1871 | 0.3073 |
| `G15_ECHF_main` | 0.2196 | 0.1769 | 0.2859 |
| `G15_ECHF_noadapt` | 0.2053 | 0.1728 | 0.2449 |

### Stage-4 (fair schedule ablation: 60 vs 100)
Source: `exp/stage_results/stage4_sched60_vs100_fair/compare_100_minus_60_by_dataset.csv`

- Mean favors `sched=100` on classic set due larger gains on `photo`/`computers`.
- `sched=60` still wins on some datasets (`citeseer`, `cora`, `pubmed`).
- Conclusion: schedule is dataset-sensitive; `100` is default, `60` is fallback for speed/overfit control.

### Stage-5 (urban known-only, multi-seed)
Source: `exp/stage_results/stage5_urban_known_struct_e60/summary_by_condition_normalized.csv`

| Condition | Mean NMI | Mean ARI | Mean ACC | Mean Modularity | Mean Conductance |
|---|---:|---:|---:|---:|---:|
| `G15_ECHF_main` | 0.0218705 | 0.0005596 | 0.0330773 | 0.0055676 | 0.9793812 |
| `G17_V5_temp15` | 0.0213721 | 0.0004606 | 0.0325701 | 0.0049209 | 0.9799362 |

`G15_ECHF_main` shows better overall structural quality and label-aligned metrics on urban real graphs.

### Post-refactor quick checks
- Multi-seed smoke (`exp/stage_results/stage10_echf_smoke_v2_s012/summary_by_condition.csv`):
  `B15_ECHF_s60` and `B15_ECHF_main` both outperform baseline; `s60` is slightly higher in this scope.
- Urban quick sanity (`exp/stage_results/stage11_urban_echf_quick_e20_s0/compare_g15_minus_g17_overall.csv`):
  `G15_ECHF_main` vs `G17_V5_temp15` is near-tie with slight mean edge to `G15` (`ΔNMI ~ +3.53e-05`).
- Urban E60 semi-full spot-check (`exp/stage_results/stage12_urban_echf_recheck_e60_s01_partial/README.md`):
  historical interrupted run (`5/52`) with directionally consistent subset (`bangkok` s0/s1).
- Urban E60 full rerun (`exp/stage_results/stage13_urban_echf_recheck_e60_s01_full/README.md`):
  completed `52/52` runs (`13 cities x 2 seeds x 2 conditions`), with `G15_ECHF_main` winning `11/13` cities on both NMI and ARI; this is the current primary post-refactor urban evidence.

## Core Structure Pointers

- B15 mainline definition: `exp/versions/B15_ECHF_main.md`
- G15 mainline preset: `exp/versions/G15_ECHF_main.md`
- Naming map: `exp/versions/NAMING.md`
- Legacy reading notes remain in `reading/` (for compatibility), but `exp/versions/` is canonical.

## References

- DSE repository: https://github.com/RiemannGraph/DSE_clustering
- DSE/LSEnet paper (arXiv): https://arxiv.org/abs/2504.09970
- Local paper copy: `reference/2504.09970v2.pdf`

## License

See `LICENSE`.
