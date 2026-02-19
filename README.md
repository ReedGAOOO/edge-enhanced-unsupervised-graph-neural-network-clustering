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

## Data Placement

Default root is `data/`.

### Built-in datasets
Auto-downloaded when missing (PyG loaders):
- `cora`, `citeseer`, `pubmed`, `computers`, `photo`, etc.

### Custom dataset format

```text
data/
  your_dataset/
    your_dataset_adj.npy
    your_dataset_feat.npy
    your_dataset_label.npy
```

- `*_adj.npy`: `[N, N]`
- `*_feat.npy`: `[N, F]`
- `*_label.npy`: `[N]`

Run:

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
