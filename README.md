# edge enhanced unsupervised graph neural network clustering

A compact research-oriented implementation for unsupervised graph clustering with edge-enhanced learning.

## Data And Artifact Policy
- Large datasets and heavy experiment artifacts are **not tracked in Git**.
- Only lightweight manifests are kept:
  - `data/README.md`
  - `Exp/README.md`
- For full packages, keep cloud links in those manifest files.
- Runtime behavior: missing benchmark datasets are auto-downloaded by PyG loaders under `--root_path data`.

## Deployment
### 1) Create environment
```bash
cd /path/to/edge-enhanced-unsupervised-graph-neural-network-clustering
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch -r requirements.txt
```

### 2) Quick sanity check
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python tools/run_preset.py --preset a2_u2_no_adapt --dataset cora --seed 0 --gpu 0 --dry_run
```

## Dataset Placement
Default data root is `data` (`--root_path data` in `main.py`).

Built-in datasets (auto-downloaded when missing):
- `cora`, `citeseer`, `pubmed`
- `computers`, `photo`
- `coauthorcs` / `coauthorphysics`
- `KarateClub`, `FootBall`, `Cornell`, `Texas`, `Wisconsin`

## Custom Dataset Format (ATS-style)
You can use your own dataset name directly (no code edit), as long as files follow:

```text
data/
  <your_dataset_name>/
    <your_dataset_name>_adj.npy
    <your_dataset_name>_feat.npy
    <your_dataset_name>_label.npy
```

Required shapes:
- `*_adj.npy`: `[N, N]` adjacency matrix (non-zero entries are treated as edges).
- `*_feat.npy`: `[N, F]` node features (`float32` recommended).
- `*_label.npy`: `[N]` node labels (`int64` recommended, class ids from `0..K-1`).

Important notes:
- The three files must share the same node count `N`.
- For clustering metrics, labels are required even though training is unsupervised.
- Weighted adjacency is currently interpreted as edge existence (non-zero = connected).
- For undirected graphs, use symmetric adjacency.

Run on your custom dataset:
```bash
python main.py --dataset your_dataset_name --root_path data --gpu 0 --max_nums 10
```

Or use preset runner (set `--max_nums` explicitly for custom datasets):
```bash
python tools/run_preset.py --preset a2_u2_no_adapt --dataset your_dataset_name --max_nums 10 --gpu 0
```

Outputs are written to:
- Logs: `results/<version>/<dataset>.log`
- Metrics JSON: `results/<version>/<dataset>_metrics.json`

## Minimal End-to-End Example (Your Own Dataset)
1. Prepare files:
```text
data/mygraph/mygraph_adj.npy
data/mygraph/mygraph_feat.npy
data/mygraph/mygraph_label.npy
```

2. Run:
```bash
python tools/run_preset.py --preset a2_u2_no_adapt --dataset mygraph --max_nums 10 --gpu 0
```

3. Check result:
```bash
cat results/a2_u2_no_adapt_mygraph_s0/mygraph_metrics.json
```

## Gap Statement
Unsupervised graph clustering often fails in one of two ways:
- relying only on node features misses structural cues,
- injecting edge signals too aggressively causes negative transfer.

This project addresses that tension with a **V5 edge-fusion base structure** and a robust best-practice configuration **`DSE_A2_u2_no_adapt`**.
The goal is not dataset-specific trick tuning, but a practical and interpretable recipe for stable gains.

## Core Innovation (Concise)
1. **V5 as the base structure**
- Build edge priors by combining feature similarity and structure consistency.
- Inject edge information at assignment-score stage (not hard early fusion).

2. **Reliability-aware edge contribution**
- Edge contribution is modulated by per-edge reliability, reducing noisy-edge amplification.

3. **Scheduled edge fusion strength**
- Use gamma schedule (`0.2 -> 1.2`) to avoid unstable early over-injection.

4. **Best-practice setting (`A2`)**
- Keep V5 + schedule, disable graph-level adaptive alpha.
- Empirically reduces tail risk and improves robustness in multi-round runs.

## Model Structure
### V5 edge weight construction
For each edge `(i,j)`:
- structural weight: degree-profile compatibility,
- feature weight: cosine-similarity-based score,
- hybrid edge prior:

`w_ij = alpha * w_feat + (1 - alpha) * w_struct`

where `alpha = edge_hybrid_alpha`.

### Assignment-score fusion (V5)
Base score is Lorentz distance-derived attention score. V5 adds:

`score_ij += gamma * graph_alpha * reliability_ij * log(w_ij)`

- `gamma`: fusion strength (scheduled by epoch),
- `graph_alpha`: graph-level scaling (disabled in A2),
- `reliability_ij`: edge-level confidence gate.

### Why A2 works in practice
`DSE_A2_u2_no_adapt` keeps:
- V5 hybrid edges,
- gamma schedule (`0.2 -> 1.2`, first `100` epochs),
- reliability gate,
and removes graph-level adaptive alpha to improve stability.

## Recommended Usage Scenarios
This repo is suitable when:
- labels are unavailable but graph topology is informative,
- node features alone are weak/noisy,
- you need reproducible, robust clustering improvement rather than one-off tuning.

Typical domains:
- citation graphs,
- product/co-purchase networks,
- user-item interaction graph abstractions,
- scientific collaboration graphs.

## Experiments And Evidence (Archived)
The tables below summarize completed runs. Full logs/results are hosted externally
(see `Exp/README.md` for cloud links).

### Experiment process
- **Round A (same-batch ablation)**: `90/90` runs, 5 datasets, seeds `3,4,5`.
- **Round B (A2-focused expansion)**: `240/240` runs, larger A2 variant search.
- **Round C (causal stage analysis)**: `225/225` runs, phase-level verification.

Robust score used in ranking:

`robust_score = mean_delta_nmi + 0.25 * win_rate - 0.5 * tail_risk`

### Table 1. Same-batch ablation summary (vs baseline)
Source: archived experiment exports (cloud mirror; see `Exp/README.md`)

| Condition | Avg ΔNMI | Avg ΔARI | Win Rate | Tail Risk | Worst-Set ΔNMI | Robust Score | Avg Time Overhead % |
|---|---:|---:|---:|---:|---:|---:|---:|
| DSE_U1_unified | 0.0137 | 0.0056 | 0.8000 | 0.0667 | 0.0001 | 0.1803 | 1.0604 |
| DSE_U2_unified | 0.0751 | 0.0658 | 0.8667 | 0.0667 | -0.0046 | 0.2584 | 1.1395 |
| DSE_S0_struct_only | 0.1035 | 0.1020 | 0.9333 | 0.0667 | 0.0210 | 0.3035 | 1.1980 |
| DSE_A1_u2_no_sched | 0.0944 | 0.0893 | 0.8000 | 0.1333 | -0.0038 | 0.2278 | 1.2295 |
| **DSE_A2_u2_no_adapt** | **0.1150** | **0.1028** | **0.8667** | **0.0000** | **0.0032** | **0.3316** | **0.6292** |

### Table 2. Best condition per dataset (same-batch)
Source: archived experiment exports (cloud mirror; see `Exp/README.md`)

| Dataset | Best Condition | Best ΔNMI | Best ΔARI | Best NMI | Best ARI |
|---|---|---:|---:|---:|---:|
| citeseer | DSE_S0_struct_only | 0.1838 | 0.1631 | 0.2539 | 0.2490 |
| computers | DSE_A1_u2_no_sched | 0.0893 | 0.0804 | 0.2364 | 0.1892 |
| cora | DSE_S0_struct_only | 0.0411 | 0.0330 | 0.4484 | 0.3545 |
| photo | **DSE_A2_u2_no_adapt** | **0.2753** | **0.2742** | **0.5123** | **0.4343** |
| pubmed | DSE_S0_struct_only | 0.0210 | 0.0250 | 0.2381 | 0.1952 |

### Table 3. A2 expansion top results
Source: archived experiment exports (cloud mirror; see `Exp/README.md`)

| Condition | Mean ΔNMI | Win Rate | Tail Risk | Worst-Set ΔNMI | Robust Score |
|---|---:|---:|---:|---:|---:|
| **DSE_A2_end16_sched60** | **0.1477** | **1.0000** | **0.0000** | **0.0077** | **0.3977** |
| DSE_A2_rel13_sched60 | 0.1404 | 0.9333 | 0.0000 | 0.0034 | 0.3738 |
| DSE_A2_sched60 | 0.1162 | 0.9333 | 0.0000 | 0.0030 | 0.3496 |
| DSE_A2_end16 | 0.1160 | 0.9333 | 0.0000 | 0.0047 | 0.3493 |
| DSE_A2_end14_sched60 | 0.1322 | 0.8667 | 0.0000 | 0.0015 | 0.3489 |
| DSE_A2_sched140 | 0.1063 | 0.9333 | 0.0000 | 0.0050 | 0.3396 |

### Table 4. Causal stage check (DSE best vs baseline)
Source: archived experiment exports (cloud mirror; see `Exp/README.md`)

| Dataset | Baseline NMI | Best Condition | Best NMI | ΔNMI vs Baseline |
|---|---:|---|---:|---:|
| citeseer | 0.0850 | DSE_M4_mid_a07g10 | 0.2422 | 0.1572 |
| cora | 0.4454 | DSE_B0_baseline | 0.4454 | 0.0000 |
| pubmed | 0.2181 | DSE_P1_struct_pre | 0.2342 | 0.0161 |

## Best Practice vs Base Structure
- **Best practice (default recommendation):** `DSE_A2_u2_no_adapt`
- **Base structure reference:** `V5` (`v5_mid_adaptive_u2` preset)

## Two Main Run Commands
### 1) Best practice (`DSE_A2_u2_no_adapt`)
```bash
cd /home/aitx/workspace/projects/edge-enhanced-unsupervised-graph-neural-network-clustering
python3 tools/run_preset.py --preset a2_u2_no_adapt --dataset cora --seed 0 --gpu 0
```

### 2) V5 base structure (`v5_mid_adaptive_u2`)
```bash
cd /home/aitx/workspace/projects/edge-enhanced-unsupervised-graph-neural-network-clustering
python3 tools/run_preset.py --preset v5_mid_adaptive_u2 --dataset cora --seed 0 --gpu 0
```

Optional baseline-vs-A2 batch comparison:
```bash
python3 tools/compare_baseline_vs_a2.py --datasets cora,citeseer,pubmed --seeds 0,1,2 --gpu 0
```

## Repository Layout
```text
edge-enhanced-unsupervised-graph-neural-network-clustering/
  data/                 # dataset manifest only (actual data kept out of git)
    README.md
  Exp/                  # experiment artifact manifest only
    README.md
  configs/presets/
  tools/
  scripts/
  modules/
  manifold/
  utils/
```

## License
This repository keeps the upstream `LICENSE` from the source project.
