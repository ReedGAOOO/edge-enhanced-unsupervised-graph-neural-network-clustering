# V12 Core Code

## Where V12 is selected
Source: `main.py:57`

```python
parser.add_argument(
    "--edge_variant",
    type=str,
    default="V1",
    choices=["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V12"],
)
```

## Core logic in assigner
Source: `modules/layers.py:199`, `modules/layers.py:257`

```python
elif self.edge_variant in {"V6", "V7", "V8", "V12"} and bool(use_edge_attr) and edge_attr is not None:
    edge_attr = torch.nan_to_num(
        edge_attr.to(dtype=score.dtype, device=score.device),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    attr_out = self.edge_attr_encoder(edge_attr)
    attr_bias = attr_out[:, 0]
    attr_gate = torch.sigmoid(attr_out[:, 1])
    attr_bias = (attr_bias - attr_bias.mean()) / attr_bias.std(unbiased=False).clamp_min(1e-6)
    edge_log = torch.log(edge_value.clamp_min(1e-8))
    # V12
    struct_rel, _ = self._struct_reliability_and_log(edge_value)
    attr_rel = attr_gate
    graph_alpha = self._graph_alpha(struct_rel, fallback_dtype=score.dtype, fallback_device=score.device)
    edge_z = (edge_log - edge_log.mean()) / edge_log.std(unbiased=False).clamp_min(1e-6)
    attr_z = (attr_bias - attr_bias.mean()) / attr_bias.std(unbiased=False).clamp_min(1e-6)
    with torch.no_grad():
        agreement = (edge_z.detach() * attr_z.detach()).mean().clamp(-1.0, 1.0)
        mix_beta = torch.sigmoid(self.edge_adaptive_alpha_strength * agreement + self.edge_adaptive_alpha_bias).clamp(0.05, 0.95)
    residual = attr_rel * attr_bias
    residual = residual / residual.std(unbiased=False).clamp_min(1e-6)
    attr_term = struct_rel * edge_log + mix_beta * residual
    score = score + float(self.edge_fusion_gamma) * self.edge_attr_fusion_scale * graph_alpha * attr_term
```

## Hierarchical edge-attribute propagation (Path-B support)
Source: `modules/model.py:92`, `modules/model.py:171`

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

if layer_use_edge_attr and self.edge_attr_hierarchical:
    current_edge_attr = self._coarsen_edge_attr_hard(current_adj, current_edge_attr, ass, adj_par)
```

## SI objective graph rebuilding from assignments
Source: `modules/dsi.py:129`, `modules/dsi.py:140`, `modules/dsi.py:156`

```python
adj_train_msg = (self.alpha * adj_aug + adj_base_msg).coalesce()
adj_train_si = (self.alpha * adj_aug + adj_base_si).coalesce()
use_edge_attr = self._use_edge_attr_variant()
edge_attr = None
if use_edge_attr:
    edge_attr_base = getattr(data, "edge_attr", None)
    edge_attr = self._align_edge_attr_to_adj(adj_base_msg, edge_attr_base, adj_train_msg)
_, ass_aug_dict, _ = self.encoder(data.x, adj_train_msg, edge_attr=edge_attr, use_edge_attr=use_edge_attr)
adj_si_dict = self._build_hierarchy_adj_from_assign(adj_train_si, ass_aug_dict)
loss = self._si_loss(ass_aug_dict, adj_si_dict, eps)
```
