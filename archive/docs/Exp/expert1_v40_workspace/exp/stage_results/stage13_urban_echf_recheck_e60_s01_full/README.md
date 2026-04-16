# Stage 13: Urban E60 Recheck (Seeds 0/1) - Full 52 Runs

## Purpose
Complete recheck after interrupted partial stage (`stage12`) for `G15_ECHF_main` vs `G17_V5_temp15` on all urban datasets.

## Run config
- Script: `tools/run_urban_branch_compare.py`
- Conditions: `G15_ECHF_main,G17_V5_temp15`
- Datasets: all 13 urban plot datasets
- Seeds: `0,1`
- Epochs: `60`
- Known-only eval: enabled
- Tag: `urban_echf_recheck_e60_s01_v1`

## Completion status
- Planned runs: `52` (`2 conditions x 13 datasets x 2 seeds`)
- Completed runs in `runs.csv`: `52`
- Status breakdown: `ok=45`, `skip_exists=7` (resume cache hits)

## Key result
Source: `summary_by_condition.csv`

| Condition | Mean NMI | Mean ARI | Mean ACC | Mean SI loss | Mean Modularity | Mean Conductance | Mean Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| `G15_ECHF_main` | 0.0218139 | 0.0005017 | 0.0332346 | 14.6178 | 0.0052580 | 0.9796994 | 66.3097 |
| `G17_V5_temp15` | 0.0213681 | 0.0004100 | 0.0328087 | 14.5589 | 0.0046156 | 0.9802546 | 53.7421 |

Overall deltas (`G15 - G17`, from `compare_g15_minus_g17_overall.csv`):
- `Delta NMI`: `+0.0004458`
- `Delta ARI`: `+0.0000918`
- `Delta ACC`: `+0.0004259`
- `Delta Modularity`: `+0.0006425`
- `Delta Conductance`: `-0.0005552` (lower is better, so favorable)

Per-dataset wins (`compare_g15_minus_g17_by_dataset.csv`):
- NMI wins: `11 / 13`
- ARI wins: `11 / 13`

## Files
- `runs.csv`
- `summary_by_condition.csv`
- `summary_by_condition_dataset.csv`
- `best_condition_by_dataset.csv`
- `delta_vs_baseline.csv`
- `compare_g15_minus_g17_by_dataset.csv`
- `compare_g15_minus_g17_overall.csv`
- `summary.json`

## Interpretation
This full rerun confirms the earlier urban ordering: `G15_ECHF_main` remains slightly but consistently better than `G17_V5_temp15` on real urban graphs, at the cost of higher runtime.
