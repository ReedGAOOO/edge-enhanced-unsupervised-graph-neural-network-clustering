# B14_PATHA_only_v1 Core Code

## Condition definition
Source: `tools/run_benchmark_branch_compare.py:282`

```python
{
    "condition": "B14_PATHA_only_v1",
    "edge_variant": "V1",
    "edge_hybrid_alpha": 0.7,
    "edge_feat_temp": 1.0,
    "edge_input_prior_alpha": 0.35,
    "edge_fusion_gamma": 1.0,
    "edge_fusion_gamma_start": None,
    "edge_fusion_gamma_end": None,
    "edge_fusion_gamma_sched_epochs": 0,
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

## Path-A edge-weight mapping (objective-side)
Source: `data.py:62`, `data.py:222`

```python
def build_edge_weight_from_attr(edge_attr, ref_weight=None, temp=1.0):
    score = edge_attr.mean(dim=1)
    score = (score - score.mean()) / score.std(unbiased=False).clamp_min(1e-6)
    w = torch.sigmoid(score / temp).clamp(1e-4, 1.0)
    if ref_weight is not None and ref_weight.numel() == w.numel():
        ref_weight = ref_weight.float()
        w_mean = w.mean().clamp_min(1e-6)
        ref_mean = ref_weight.mean().clamp_min(1e-6)
        w = w / w_mean * ref_mean
        upper = (ref_weight.mean() + 3.0 * ref_weight.std(unbiased=False)).clamp_min(ref_mean)
        w = w.clamp(1e-6, float(upper.detach().item()))
    return w

edge_attr_weight_blend = float(getattr(configs, "edge_attr_weight_blend", 0.0))
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

## SI/message graph split
Source: `data.py:240`

```python
adj_msg_raw = torch.sparse_coo_tensor(indices=edge_index, values=edge_weight_msg, size=(N, N)).coalesce()
adj_si = torch.sparse_coo_tensor(indices=edge_index, values=edge_weight_si, size=(N, N)).coalesce()
adj_msg = normalize_adj(adj_msg_raw, sparse=True).coalesce()
```

