# Experiment Center (`exp/`)

`exp/` is the unified place for:
- model-version notes,
- staged result summaries,
- consolidated mainline reports.

## Layout

```text
exp/
  README.md
  versions/              # model/branch structure notes
  stage_results/         # copied summary CSVs from key experiment stages
  reports/               # consolidated narrative reports
```

## Mainline naming (ECHF)
- `B15_ECHF_main`: canonical branch mainline (Path-B + V12 + hierarchical edge attributes).
- `G15_ECHF_main`: canonical runnable preset (default in `tools/run_preset.py`).
- `G15_ECHF_noadapt`: no-adaptive-alpha fallback.
- `G17_V5_temp15`: V5 comparator.

See: `versions/NAMING.md`.

## Fast entry points
- Stage summary: `stage_results/README.md`
- Mainline report: `reports/MAINLINE_ECHF_B15_G15_2026-02-19.md`
- Core structure notes:
  - `versions/B15_ECHF_main.md`
  - `versions/G15_ECHF_main.md`

## Reproduction shortcuts

Mainline preset:
```bash
python3 tools/run_preset.py --preset g15_echf_main --dataset cora --seed 0 --gpu 0
```

Urban ECHF vs V5 comparator:
```bash
python3 tools/run_urban_branch_compare.py \
  --conditions G15_ECHF_main,G17_V5_temp15 \
  --seeds 0,1,2 \
  --epochs 60 \
  --known_only_eval \
  --tag urban_known_struct_g15g17_e60_v1
```
