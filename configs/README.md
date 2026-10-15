# Active configurations

Updated 2026-10-15. Run commands from the repository root; paths in the runners are relative to it.

## Presets

| File in `presets/` | Model | Use |
|---|---|---|
| `baseline_v1.json` | V1, edge-free DSE/LSENet | Required control for edge-aware comparisons |
| `b31_dualscalar_assign.json` | V31, dual message/SE scalar weights and assignment residual | Default of `tools/run_preset.py` |
| `b45_v31_msgcond_gs050.json` | B31 plus leaf message gate, scale 0.50 | Diagnostic comparator; collapses on Photo/AIFB in E3/E4 |

```bash
.venv/bin/python tools/run_preset.py --list_presets
.venv/bin/python tools/run_preset.py --preset baseline_v1 --dataset cora --dry_run
.venv/bin/python tools/run_preset.py --preset b31_dualscalar_assign --dataset cora --edge_ctx_dim 32
```

Preset JSONs contain edge-model settings. `run_preset.py` supplies the training budget: 2000 epochs, hidden size 512,
weight decay 0, temperature 1.0, three decodes, no early stopping and no warm start by default. `main.py` has different
defaults. A single preset run does not perform dual-start selection.

`--edge_ctx_dim` appends standardized symmetric endpoint products to the edge attributes. Use plans for overrides
that are not exposed by the convenience CLI, including E9's `edge_weight_learn_apply_to=msg_only`.

## Experiment plans

See [breakthrough/README.md](breakthrough/README.md) for E1–E9, machine checks, completion status and reports.
Plans merge their `defaults` and job `overrides` over the preset, expand datasets × seeds, and write local outputs
under `results/breakthrough_v1/<plan-name>/`.

Retired presets live in [archive/configs/](../archive/configs/README.md). They remain loadable by an explicit path,
but pre-pass rankings are not evidence for the current models.
