# B15/G15 Mainline Experiment Report (2026-02-19)

## 1) Goal
This report consolidates the recent edge-fusion experiments and justifies promoting **B15/G15** as the current mainline.

- **B15**: branch condition `B15_ECHF_main` in branch-comparison suites.
- **G15**: runnable preset `g15_echf_main` (and `b15_echf_branch`) for day-to-day training.

## 2) Evaluation Protocol
### 2.1 Label policy
- We use **known-only evaluation** where unknown labels are remapped to `-1` and ignored by clustering metrics.
- Code path: `main.py --known_only_eval` + `data.py` unknown remap logic.

### 2.2 Metrics
We report both label metrics and structure metrics:
- label-aligned: `NMI`, `ARI`, `ACC`
- structure quality: `SI loss`, `modularity`, `conductance` (mean/weighted)
- stability: cross-seed std (`nmi_std`, `ari_std`) and trial-pair NMI stability

### 2.3 Key result bundles used in this report
- `results/benchmark_pathAB_full_v1/summary_by_condition.csv`
- `results/benchmark_pathAB_full_v1/summary_by_condition_dataset.csv`
- `results/urban_known_struct_g15g17_e60_v1/summary_by_condition.csv`
- `results/urban_known_struct_g15g17_e60_v1/g15_vs_g17_dataset_deltas.csv`
- `results/urban_known_struct_g15g17_e20e60_compare_v1/summary_condition_e60_vs_e20.csv`

## 3) Structural Definition of B15/G15
B15/G15 is a **Path-B-first V12 design**:

1. **V12 fusion trunk**
- Keep V5 structural term as trunk and add calibrated edge-attribute residual.
- `score += gamma * fusion_scale * graph_alpha * attr_term`

2. **Calibration by structure-attribute agreement**
- Compute `mix_beta` from agreement between structural edge prior and edge-attribute signal.
- This prevents blind over-trust of edge attributes.

3. **Graph-adaptive global scaling**
- `graph_alpha` is adaptive in G15/B15 default.
- On heterogeneous graphs this avoids hard full-strength edge injection.

4. **Hierarchical edge-attribute propagation (Path-B)**
- Edge attributes are coarsened and passed up the partition hierarchy.
- This preserves edge semantics beyond the leaf graph.

5. **No Path-A SI-weight blend in the default mainline**
- `edge_attr_weight_blend=0.0` in B15/G15 defaults.
- This avoids coarse mean-compression of rich edge features into one scalar SI weight.

## 4) Standard Graph Benchmark (5 datasets) Summary
Source: `results/benchmark_pathAB_full_v1/summary_by_condition.csv`

| Condition | Mean NMI | Mean ARI | Win Rate vs Baseline (NMI/ARI) | Rank Score |
|---|---:|---:|---:|---:|
| **B15_ECHF_main** | 0.2154 | 0.1784 | **1.0 / 1.0** | **0.3125** |
| B5_V5_mid_no_adapt | 0.2295 | 0.1835 | 0.8 / 0.8 | 0.2773 |
| B12_V12_residual_calibrated | 0.2241 | 0.1834 | 0.8 / 1.0 | 0.2719 |
| B17_PATHA_on_V5 | 0.2202 | 0.1795 | 0.8 / 0.8 | 0.2675 |

Interpretation:
- B15 is selected by **overall robustness criterion** (top rank score, 100% win-rate against baseline on both NMI and ARI).
- B5/B12 may reach higher mean on some datasets, but B15 shows the most stable global profile.

## 5) Urban Real-Graph Benchmark (known-only, 60 epochs)
Source: `results/urban_known_struct_g15g17_e60_v1/summary_by_condition.csv`

| Condition | Mean NMI | Mean ARI | Mean ACC | Mean Modularity | Mean Conductance | Mean SI loss |
|---|---:|---:|---:|---:|---:|---:|
| **G15_ECHF_main** | **0.0218705** | **0.0005596** | **0.0330773** | **0.0055676** | **0.9793812** | 14.6179 |
| G17_V5_temp15 | 0.0213721 | 0.0004606 | 0.0325701 | 0.0049209 | 0.9799362 | **14.5589** |

Per-dataset win counts (G15 - G17, source `g15_vs_g17_dataset_deltas.csv`):
- NMI: **12/13** wins, mean `+0.0004984`
- ARI: **12/13** wins, mean `+0.0000990`
- ACC: **10/13** wins, mean `+0.0005072`
- Modularity: **13/13** wins, mean `+0.0006467`
- Conductance (lower is better): **13/13** better, mean `-0.0005550`

Interpretation:
- On real urban graphs, G15 provides a more coherent partition structure (higher modularity, lower conductance) and converts that to better label-aligned metrics in most cities.

## 6) Training Sufficiency Check (20 -> 60 epochs)
Source: `results/urban_known_struct_g15g17_e20e60_compare_v1/summary_condition_e60_vs_e20.csv`

### G15_ECHF_main (60 - 20)
- `ΔNMI = +0.0031938`
- `ΔARI = +0.0005781`
- `ΔACC = +0.0053588`
- `ΔModularity = +0.0052142`
- `ΔConductance = -0.0046549`
- `ΔSI loss = -0.0188359`

### G17_V5_temp15 (60 - 20)
- `ΔNMI = +0.0026880`
- `ΔARI = +0.0004778`
- `ΔACC = +0.0048446`
- `ΔModularity = +0.0046270`
- `ΔConductance = -0.0041530`
- `ΔSI loss = -0.0166202`

Interpretation:
- Both models were under-trained at 20 epochs.
- After sufficient training, G15 improves more and its advantage becomes clearer.

## 7) Why G15 beats G17 on urban graphs (mechanistic evidence)
From run logs in `results/urban_known_struct_g15g17_e60_v1/*/*.log`:

- G17 (`V5`) final behavior:
  - `graph_alpha ~= 1.0000`
  - `edge_mix ~= 0.0000`
- G15 (`V12 + Path-B`) final behavior:
  - `graph_alpha ~= 0.6526`
  - `edge_mix ~= 0.5160`

Interpretation:
- G17 applies near full-strength structural injection with no attribute residual mixing.
- G15 keeps a moderated global strength and uses calibrated structure-attribute mixing, which is safer under urban edge heterogeneity.

## 8) Mainline Decision
We promote:
- **Branch mainline**: `B15_ECHF_main`
- **Preset mainline**: `g15_echf_main` (new default in `tools/run_preset.py`)
- **Reproduction alias preset**: `b15_echf_branch` (legacy alias `b15_pathb_v12_hier`)

Fallback:
- For extremely heterophilic/sensitive cases, test `g15_echf_noadapt`.

## 9) Reproduction Commands
### 9.1 Single-run mainline (recommended)
```bash
python3 tools/run_preset.py --preset g15_echf_main --dataset cora --seed 0 --gpu 0
```

### 9.2 Exact B15-parameter preset
```bash
python3 tools/run_preset.py --preset b15_echf_branch --dataset cora --seed 0 --gpu 0
```

### 9.3 Urban G15 vs G17 known-only benchmark
```bash
python3 tools/run_urban_branch_compare.py \
  --conditions G15_ECHF_main,G17_V5_temp15 \
  --datasets urban_bangkok_plot,urban_beijing_plot,urban_boston_plot,urban_chicago_plot,urban_johannesburg_plot,urban_madrid_plot,urban_melbourne_plot,urban_paris_plot,urban_shanghai_plot,urban_singapore_plot,urban_sydney_plot,urban_tokyo_plot,urban_washingtondc_plot \
  --seeds 0,1,2 \
  --epochs 60 \
  --known_only_eval \
  --tag urban_known_struct_g15g17_e60_v1
```
