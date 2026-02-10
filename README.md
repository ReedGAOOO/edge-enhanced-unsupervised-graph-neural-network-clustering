# edge enhanced unsupervised graph neural network clustering

Standalone repository focused on:
- `V5` as the base edge-fusion structure
- `DSE_A2_u2_no_adapt` as the best-practice configuration

This repo is extracted and reorganized from the current `DSE_clustering` codebase to make A2/V5 experiments reproducible in isolation.

## What is included
- Core training/evaluation pipeline: `main.py`, `exp.py`, `data.py`
- Hyperbolic model components: `modules/`, `manifold/`
- Utility functions: `utils/`
- Preset configs:
  - `configs/presets/baseline_v1.json`
  - `configs/presets/v5_mid_adaptive_u2.json`
  - `configs/presets/a2_u2_no_adapt.json`
- Run helpers:
  - `tools/run_preset.py`
  - `tools/compare_baseline_vs_a2.py`
  - `scripts/run_a2_u2_no_adapt.sh`

## Best Practice And Base Structure
Best practice:
- `DSE_A2_u2_no_adapt`

Base structure:
- `V5` edge fusion (`v5_mid_adaptive_u2` preset is the adaptive V5 reference)

## A2 preset definition
`DSE_A2_u2_no_adapt` uses:
- `edge_variant=V5`
- `edge_hybrid_alpha=0.7`
- `edge_fusion_gamma` schedule: `0.2 -> 1.2` during first `100` epochs
- `edge_adaptive_alpha=False`
- `edge_reliability_temp=1.0`

## Environment
Recommended Python: `3.10+`

Install dependencies:
```bash
python3 -m pip install -r requirements.txt
```

Notes:
- `torch-scatter` and `torch-geometric` should match your installed PyTorch/CUDA version.
- If you already have a working GNN env, use that env directly.

## Quick start
Run best-practice A2 on one dataset:
```bash
python3 tools/run_preset.py --preset a2_u2_no_adapt --dataset cora --seed 0 --gpu 0
```

Run V5 base structure (adaptive U2 reference) on one dataset:
```bash
python3 tools/run_preset.py --preset v5_mid_adaptive_u2 --dataset cora --seed 0 --gpu 0
```

Run multi-seed baseline vs A2 comparison:
```bash
python3 tools/compare_baseline_vs_a2.py --datasets cora,citeseer,pubmed --seeds 0,1,2 --gpu 0
```

Or use one-click scripts:
```bash
bash scripts/run_a2_u2_no_adapt.sh cora 0 0
bash scripts/run_compare_baseline_vs_a2.sh cora,citeseer,pubmed 0,1,2 0
```

## Output
- Per-run metrics: `results/<version>/<dataset>_metrics.json`
- Batch comparison CSV: `results/compare_baseline_vs_a2.csv`

## Dataset behavior
Datasets are loaded via `torch_geometric` (Planetoid/Amazon/etc.) and cached under local `datasets/`.

## Project layout
```text
edge-enhanced-unsupervised-graph-neural-network-clustering/
  configs/presets/
  scripts/
  tools/
  manifold/
  modules/
  utils/
  main.py
  exp.py
  data.py
```

## License
This extracted repo keeps the upstream `LICENSE` from the source project.
