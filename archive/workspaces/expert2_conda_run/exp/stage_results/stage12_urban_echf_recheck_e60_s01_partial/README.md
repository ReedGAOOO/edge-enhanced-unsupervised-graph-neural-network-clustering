# Stage 12: Urban E60 Recheck (Seeds 0/1) - Partial

## Context
This stage started as a semi-full recheck (`13 cities × 2 conditions × seeds 0/1`) after refactor,
but was intentionally interrupted due high runtime on large cities (especially `urban_beijing_plot`).

## Planned config
- Script: `tools/run_urban_branch_compare.py`
- Conditions: `G15_ECHF_main,G17_V5_temp15`
- Seeds: `0,1`
- Epochs: `60`
- Known-only eval: enabled
- Tag: `urban_echf_recheck_e60_s01_v1`

## Completed subset
- Completed runs: `5 / 52`
- Fully paired comparison available for:
  - `urban_bangkok_plot` seed `0`
  - `urban_bangkok_plot` seed `1`
- Additional single run completed:
  - `urban_beijing_plot` seed `0` (`G17` only)

## Files
- `runs_completed.csv`
- `summary_completed_by_condition.csv`
- `paired_g15_vs_g17_completed.csv`

## Partial conclusion
- On completed paired samples (`bangkok`, seeds `0/1`), `G15_ECHF_main` beats `G17_V5_temp15` on NMI/ARI/ACC.
- This ordering is consistent with prior full stage evidence:
  - `exp/stage_results/stage5_urban_known_struct_e60` (full multi-seed historical run).

## Recommendation
Use stage5 full run as the main urban evidence, and treat this stage as refactor-consistency spot-check.
