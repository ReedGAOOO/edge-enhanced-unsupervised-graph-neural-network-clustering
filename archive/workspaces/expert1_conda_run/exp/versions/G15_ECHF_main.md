# G15_ECHF_main Preset

## Definition
- Canonical preset: `g15_echf_main`
- Legacy alias: `g15_default_hetero`
- File: `configs/presets/g15_echf_main.json`

## Core hyperparameters
```json
{
  "edge_variant": "V12",
  "edge_fusion_gamma_start": 0.2,
  "edge_fusion_gamma_end": 1.2,
  "edge_fusion_gamma_sched_epochs": 100,
  "edge_adaptive_alpha": true,
  "edge_adaptive_alpha_strength": 2.0,
  "edge_adaptive_alpha_bias": 0.0,
  "edge_attr_fusion_scale": 0.7,
  "edge_attr_weight_blend": 0.0,
  "edge_attr_hierarchical": true
}
```

## Usage
```bash
python3 tools/run_preset.py --preset g15_echf_main --dataset cora --seed 0 --gpu 0
```

## Optional fallback
- For highly sensitive heterophilic graphs, test `g15_echf_noadapt`.
