# B15_PATHB_v12_hier Core Code

## Condition definition
Source: `tools/run_benchmark_branch_compare.py:302`

```python
{
    "condition": "B15_PATHB_v12_hier",
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
    "edge_attr_weight_blend": 0.0,
    "edge_attr_weight_temp": 1.0,
    "edge_attr_weight_apply_to": "si_only",
    "edge_attr_hierarchical": True,
}
```

## V12 assignment-score fusion
Source: `modules/layers.py:257`

```python
# V12: keep V5 as stable trunk and add calibrated edge-attribute residual.
struct_rel, _ = self._struct_reliability_and_log(edge_value)
attr_rel = attr_gate
if self.edge_confidence_quantile > 0.0:
    threshold = torch.quantile(struct_rel.detach(), qv)
    keep = (struct_rel >= threshold).to(struct_rel.dtype)
    struct_rel = struct_rel * keep
    attr_rel = attr_rel * keep
graph_alpha = self._graph_alpha(struct_rel, fallback_dtype=score.dtype, fallback_device=score.device)
reliability = struct_rel

edge_z = (edge_log - edge_log.mean()) / edge_log.std(unbiased=False).clamp_min(1e-6)
attr_z = (attr_bias - attr_bias.mean()) / attr_bias.std(unbiased=False).clamp_min(1e-6)
with torch.no_grad():
    agreement = (edge_z.detach() * attr_z.detach()).mean().clamp(-1.0, 1.0)
    mix_beta = torch.sigmoid(
        self.edge_adaptive_alpha_strength * agreement + self.edge_adaptive_alpha_bias
    ).clamp(0.05, 0.95)

residual = attr_rel * attr_bias
residual = residual / residual.std(unbiased=False).clamp_min(1e-6)
attr_term = struct_rel * edge_log + mix_beta * residual
score = score + float(self.edge_fusion_gamma) * self.edge_attr_fusion_scale * graph_alpha * attr_term
```

## Hierarchical edge-attribute propagation
Source: `modules/model.py:92`, `modules/model.py:171`

```python
if layer_use_edge_attr and self.edge_attr_hierarchical:
    current_edge_attr = self._coarsen_edge_attr_hard(current_adj, current_edge_attr, ass, adj_par)
else:
    current_edge_attr = None
```

```python
def _coarsen_edge_attr_hard(cls, adj_curr, edge_attr_curr, ass, adj_par):
    if edge_attr_curr is None or edge_attr_curr.numel() == 0:
        return None
    adj_curr_sp = cls._to_sparse_coalesced(adj_curr)
    idx = adj_curr_sp.indices()
    val = adj_curr_sp.values()
    if edge_attr_curr.shape[0] != idx.shape[1]:
        return None
    parent = ass.argmax(dim=1)
    num_parent = int(ass.shape[1])
    src_p = parent[idx[0]]
    dst_p = parent[idx[1]]
    key = src_p.long() * num_parent + dst_p.long()
    sorted_key, order = torch.sort(key)

    attr = edge_attr_curr.float()
    weighted_attr = (attr * val.unsqueeze(1).to(attr.dtype))[order]
    val_sorted = val[order].to(attr.dtype)
    uniq, inv = torch.unique(sorted_key, sorted=True, return_inverse=True)

    agg_attr = torch.zeros((uniq.numel(), attr.shape[1]), dtype=attr.dtype, device=attr.device)
    agg_attr.index_add_(0, inv, weighted_attr)
    agg_denom = torch.zeros((uniq.numel(),), dtype=attr.dtype, device=attr.device)
    agg_denom.index_add_(0, inv, val_sorted)
    agg_attr = agg_attr / agg_denom.clamp_min(1e-6).unsqueeze(1)

    adj_par_sp = cls._to_sparse_coalesced(adj_par)
    idx_par = adj_par_sp.indices()
    target_key = idx_par[0].long() * num_parent + idx_par[1].long()
    pos = torch.searchsorted(uniq, target_key)
    valid = pos < uniq.shape[0]
    matched = torch.zeros_like(valid, dtype=torch.bool)
    matched[valid] = uniq[pos[valid]] == target_key[valid]

    out_attr = torch.zeros((idx_par.shape[1], attr.shape[1]), dtype=attr.dtype, device=attr.device)
    if matched.any():
        out_attr[matched] = agg_attr[pos[matched]]
    return out_attr
```
