# B17_PATHA_on_V5 Core Code

## Condition definition
Source: `tools/run_benchmark_branch_compare.py:346`

```python
{
    "condition": "B17_PATHA_on_V5",
    "edge_variant": "V5",
    "edge_hybrid_alpha": 0.7,
    "edge_feat_temp": 1.0,
    "edge_input_prior_alpha": 0.35,
    "edge_fusion_gamma": 1.0,
    "edge_fusion_gamma_start": 0.2,
    "edge_fusion_gamma_end": 1.2,
    "edge_fusion_gamma_sched_epochs": 100,
    "edge_adaptive_alpha": False,
    "edge_adaptive_alpha_strength": 1.0,
    "edge_adaptive_alpha_bias": 2.0,
    "edge_reliability_temp": 1.0,
    "edge_confidence_quantile": 0.0,
    "edge_attr_weight_blend": 0.5,
    "edge_attr_weight_temp": 1.0,
    "edge_attr_weight_apply_to": "si_only",
    "edge_attr_hierarchical": False,
}
```

## V5 assignment-score fusion trunk
Source: `modules/layers.py:183`

```python
if self.edge_variant == "V5":
    reliability, edge_log = self._struct_reliability_and_log(edge_value)
    if self.edge_adaptive_alpha:
        with torch.no_grad():
            mean_w = edge_value.detach().mean()
            cv_w = edge_value.detach().std(unbiased=False) / mean_w.abs().clamp_min(1e-6)
            raw = mean_w - cv_w
            graph_alpha = torch.sigmoid(
                self.edge_adaptive_alpha_strength * raw + self.edge_adaptive_alpha_bias
            ).clamp(0.05, 0.95)
    else:
        graph_alpha = edge_log.new_tensor(1.0)
    self.last_graph_alpha = float(graph_alpha.detach().cpu().item())
    self.last_reliability_mean = float(reliability.detach().mean().cpu().item())
    self.last_mix_beta = 0.0
    score = score + float(self.edge_fusion_gamma) * graph_alpha * reliability * edge_log
```

## Path-A weight mapping (SI-side)
Source: `data.py:222`

```python
if edge_attr_weight_blend > 0.0:
    edge_weight_attr = build_edge_weight_from_attr(
        edge_attr=edge_attr,
        ref_weight=edge_weight,
        temp=float(getattr(configs, "edge_attr_weight_temp", 1.0)),
    )
    if edge_weight_attr is not None:
        edge_weight_si = (1.0 - edge_attr_weight_blend) * edge_weight + edge_attr_weight_blend * edge_weight_attr
```
