# Stage 10: ECHF Smoke Benchmark (Seeds 0/1/2)

## Purpose
Validate post-refactor ECHF naming on multiple seeds (not only seed0).

## Run config
- Script: `tools/run_benchmark_branch_compare.py`
- Datasets: `cora,citeseer,photo`
- Conditions: `B0_V1_baseline,B15_ECHF_s60,B15_ECHF_main`
- Seeds: `0,1,2`
- Epochs: `60`
- Trials: `1`
- Tag: `benchmark_echf_smoke_v2_s012`

## Key result
Source: `summary_by_condition.csv`

| Condition | Mean NMI | Mean ARI | Mean ΔNMI vs baseline | Mean ΔARI vs baseline | WinRate(NMI/ARI) | Rank Score |
|---|---:|---:|---:|---:|---:|---:|
| `B15_ECHF_s60` | 0.09572 | 0.07649 | +0.00617 | +0.00628 | 1.0 / 1.0 | 0.2571 |
| `B15_ECHF_main` | 0.09470 | 0.07605 | +0.00515 | +0.00583 | 1.0 / 1.0 | 0.2560 |
| `B0_V1_baseline` | 0.08955 | 0.07022 | 0 | 0 | 0 / 0 | 0 |

## Per-dataset winner by NMI
- `citeseer`: `B15_ECHF_s60`
- `cora`: `B15_ECHF_s60`
- `photo`: `B15_ECHF_s60`

Conclusion: in this smoke scope, `s60` remains slightly stronger than `main`.
