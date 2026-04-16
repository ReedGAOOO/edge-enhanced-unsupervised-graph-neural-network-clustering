# Stage 9: ECHF Smoke Benchmark (Post-Refactor)

## Purpose
Quickly verify that the renamed ECHF pipeline is runnable end-to-end and still beats baseline.

## Run config
- Script: `tools/run_benchmark_branch_compare.py`
- Datasets: `cora,citeseer,photo`
- Conditions: `B0_V1_baseline,B15_ECHF_s60,B15_ECHF_main`
- Seeds: `0`
- Epochs: `60`
- Trials: `1`
- Tag: `benchmark_echf_smoke_v1`

## Key result
Source: `summary_by_condition.csv`

| Condition | Mean NMI | Mean ARI | Mean ΔNMI vs baseline | Mean ΔARI vs baseline | Rank Score |
|---|---:|---:|---:|---:|---:|
| `B15_ECHF_s60` | 0.1251 | 0.0904 | +0.00845 | +0.00906 | 0.2598 |
| `B15_ECHF_main` | 0.1242 | 0.0902 | +0.00750 | +0.00885 | 0.2588 |
| `B0_V1_baseline` | 0.1167 | 0.0813 | 0 | 0 | 0 |

## Notes
- Both ECHF variants improved mean NMI/ARI over baseline.
- In this small smoke set, `B15_ECHF_s60` is marginally ahead of `B15_ECHF_main`.
- This stage is for refactor validation only, not final model selection.
