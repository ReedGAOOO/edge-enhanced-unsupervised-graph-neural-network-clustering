# B16_PATHA_PATHB_v12_hier Core Code

## Condition definition
Source: `tools/run_benchmark_branch_compare.py:324`

```python
{
    "condition": "B16_PATHA_PATHB_v12_hier",
    "edge_variant": "V12",
    "edge_hybrid_alpha": 0.7,
    "edge_feat_temp": 1.0,
    "edge_input_prior_alpha": 0.35,
    "edge_fusion_gamma": 1.0,
    "edge_fusion_gamma_start": 0.2,
    "edge_fusion_gamma_end": 1.2,
    "edge_fusion_gamma_sched_epochs": 100,
    "edge_adaptive_alpha": True,
    "edge_adaptive_alpha_strength": 2.0,
    "edge_adaptive_alpha_bias": 0.0,
    "edge_reliability_temp": 1.0,
    "edge_confidence_quantile": 0.0,
    "edge_attr_hidden_dim": 64,
    "edge_attr_fusion_scale": 0.7,
    "edge_attr_weight_blend": 0.5,
    "edge_attr_weight_temp": 1.0,
    "edge_attr_weight_apply_to": "si_only",
    "edge_attr_hierarchical": True,
}
```

## Path-A part: edge_attr -> SI edge weights
Source: `data.py:222`

```python
edge_attr_weight_blend = float(getattr(configs, "edge_attr_weight_blend", 0.0))
edge_attr_weight_blend = max(0.0, min(1.0, edge_attr_weight_blend))
edge_weight_msg = edge_weight.clone()
edge_weight_si = edge_weight.clone()
if edge_attr_weight_blend > 0.0:
    edge_weight_attr = build_edge_weight_from_attr(
        edge_attr=edge_attr,
        ref_weight=edge_weight,
        temp=float(getattr(configs, "edge_attr_weight_temp", 1.0)),
    )
    if edge_weight_attr is not None:
        edge_weight_si = (1.0 - edge_attr_weight_blend) * edge_weight + edge_attr_weight_blend * edge_weight_attr
        if str(getattr(configs, "edge_attr_weight_apply_to", "si_only")).lower() == "both":
            edge_weight_msg = edge_weight_si.clone()
```

## Path-B part: V12 assigner + hierarchical propagation
Sources: `modules/layers.py:257`, `modules/model.py:171`

```python
# V12 calibrated residual fusion on assignment score
attr_term = struct_rel * edge_log + mix_beta * residual
score = score + float(self.edge_fusion_gamma) * self.edge_attr_fusion_scale * graph_alpha * attr_term
```

```python
if layer_use_edge_attr and self.edge_attr_hierarchical:
    current_edge_attr = self._coarsen_edge_attr_hard(current_adj, current_edge_attr, ass, adj_par)
```

## SI loss graph path split
Source: `modules/dsi.py:113`

```python
adj_base_msg = getattr(data, "adj_msg", data.adj).clone()
adj_base_si = getattr(data, "adj_si", adj_base_msg).clone()
adj_train_msg = (self.alpha * adj_aug + adj_base_msg).coalesce()
adj_train_si = (self.alpha * adj_aug + adj_base_si).coalesce()
use_edge_attr = self._use_edge_attr_variant()
edge_attr = None
if use_edge_attr:
    edge_attr_base = getattr(data, "edge_attr", None)
    edge_attr = self._align_edge_attr_to_adj(adj_base_msg, edge_attr_base, adj_train_msg)
_, ass_aug_dict, _ = self.encoder(
    data.x, adj_train_msg, edge_attr=edge_attr, use_edge_attr=use_edge_attr
)
adj_si_dict = self._build_hierarchy_adj_from_assign(adj_train_si, ass_aug_dict)
loss = self._si_loss(ass_aug_dict, adj_si_dict, eps)
```
