# Expert Code Bundle

This bundle contains the minimum code needed to discuss the current project status with an external expert.

## Included

- `EXPERT_BRIEF_model_status_zh.md`
  - Chinese brief describing the task, current bottlenecks, and the main technical question.
- `current_code/main.py`
  - Current CLI / experiment argument entry.
- `current_code/exp.py`
  - Training loop, evaluation protocol, and branch diagnostics.
- `current_code/model.py`
  - Lorentz-tree encoder with edge-conditioned leaf message passing.
- `current_code/layers.py`
  - Lorentz graph convolution, message gating, and assignment modules.
- `current_code/dsi.py`
  - Structural-entropy model trunk, fused graph construction, and edge-weighted SE path.
- `current_code/run_preset.py`
  - Preset launcher used to run `baseline / G20 / B45`.
- `presets/b45_v31_msgcond_gs050.json`
  - Current mainline preset.
- `presets/g20_se_consistent_main.json`
  - Reference SE-consistent branch.
- `presets/baseline_v1.json`
  - No-edge baseline.
- `reference/REFERENCE_DSE_original_merged.py`
  - Merged original released reference code for comparison.
- `CURRENT_MAINLINE_CORE_MERGED.py`
  - A merged single-file view of the current mainline-related source files.

## Suggested reading order

1. `EXPERT_BRIEF_model_status_zh.md`
2. `presets/*.json`
3. `current_code/run_preset.py`
4. `CURRENT_MAINLINE_CORE_MERGED.py`
5. `reference/REFERENCE_DSE_original_merged.py`
