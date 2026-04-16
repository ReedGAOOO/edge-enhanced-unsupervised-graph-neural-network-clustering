# V5 Core Code

## Where V5 is selected
Source: `main.py:57`

```python
parser.add_argument(
    "--edge_variant",
    type=str,
    default="V1",
    choices=["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V12"],
)
```

## V5 assignment fusion branch
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
    score = score + float(self.edge_fusion_gamma) * graph_alpha * reliability * edge_log
```

## Structural reliability helper used by V5
Source: `modules/layers.py:161`

```python
def _struct_reliability_and_log(self, edge_value):
    edge_log = torch.log(edge_value.clamp_min(1e-8))
    center = torch.median(edge_value.detach())
    spread = edge_value.detach().std(unbiased=False).clamp_min(1e-6) * self.edge_reliability_temp
    reliability = torch.sigmoid((edge_value - center) / spread)
    if self.edge_confidence_quantile > 0.0:
        threshold = torch.quantile(edge_value.detach(), qv)
        conf_mask = (edge_value >= threshold).to(edge_log.dtype)
        reliability = reliability * conf_mask
    return reliability, edge_log
```

## Gamma schedule wiring
Sources: `exp.py:157`, `exp.py:111`

```python
curr_gamma = self._edge_fusion_gamma_for_epoch(epoch)
if hasattr(model, "set_edge_fusion_gamma"):
    model.set_edge_fusion_gamma(curr_gamma)
```

```python
ratio = min(1.0, max(0.0, float(epoch - 1) / float(sched_epochs - 1)))
return start_v + ratio * (end_v - start_v)
```
