# Stage 11: Urban Quick Recheck (E20, Seed0)

## Purpose
Fast city-level sanity check after refactor for `G15_ECHF_main` vs `G17_V5_temp15`.

## Run config
- Script: `tools/run_urban_branch_compare.py`
- Datasets: `urban_beijing_plot,urban_chicago_plot,urban_paris_plot`
- Conditions: `G15_ECHF_main,G17_V5_temp15`
- Seed: `0`
- Epochs: `20`
- Trials: `1`
- Known-only eval: enabled
- Tag: `urban_echf_quick_recheck_e20_s0_v1`

## Key result
Source: `summary_by_condition.csv`

| Condition | Mean NMI | Mean ARI | Mean ACC | Mean SI loss | Mean Modularity | Mean Conductance |
|---|---:|---:|---:|---:|---:|---:|
| `G15_ECHF_main` | 0.012442 | 0.0000300 | 0.025854 | 14.7547 | 0.000192 | 0.984229 |
| `G17_V5_temp15` | 0.012407 | 0.0000115 | 0.025836 | 14.7026 | 0.000133 | 0.984301 |

## Per-city NMI winner
- `urban_beijing_plot`: `G17_V5_temp15`
- `urban_chicago_plot`: `G15_ECHF_main` (almost tie)
- `urban_paris_plot`: `G15_ECHF_main`

Interpretation: quick E20 check shows near-tie overall, with slight edge to `G15_ECHF_main` on 2/3 cities.

Additional comparison files:
- `compare_g15_minus_g17_by_dataset.csv`
- `compare_g15_minus_g17_overall.csv`

Overall quick-delta (G15 - G17):
- mean ΔNMI: `+3.53e-05`
- mean ΔARI: `+1.86e-05`
