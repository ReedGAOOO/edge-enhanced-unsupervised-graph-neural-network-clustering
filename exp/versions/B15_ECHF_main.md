# B15_ECHF_main Core Structure

## Definition
- Canonical name: `B15_ECHF_main`
- Legacy alias: `B15_PATHB_v12_hier`
- Source condition: `tools/run_benchmark_branch_compare.py`

```python
{
    "condition": "B15_ECHF_main",
    "edge_variant": "V12",
    "edge_hybrid_alpha": 0.7,
    "edge_input_prior_alpha": 0.35,
    "edge_fusion_gamma_start": 0.2,
    "edge_fusion_gamma_end": 1.2,
    "edge_fusion_gamma_sched_epochs": 100,
    "edge_adaptive_alpha": True,
    "edge_adaptive_alpha_strength": 2.0,
    "edge_adaptive_alpha_bias": 0.0,
    "edge_attr_fusion_scale": 0.7,
    "edge_attr_weight_blend": 0.0,
    "edge_attr_hierarchical": True,
}
```

## Why it is the main branch design
- V12 adds calibrated residual on assignment score (keeps V5 trunk stability).
- Path-B propagates edge attributes across hierarchy instead of leaf-only usage.
- Adaptive `graph_alpha` gates global injection on heterogeneous graphs.

## Key code points
- V12 score fusion: `modules/layers.py`
- Hierarchical edge-attribute coarsening: `modules/model.py`
