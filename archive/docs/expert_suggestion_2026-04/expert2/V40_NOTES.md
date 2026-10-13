# V40 relation-state drop-in

This drop-in implements the **latent relation channels + hierarchical persistence** idea on top of the current DSE/LSEnet mainline.

## What is included

- `modules/layers.py`
  - adds `EdgeRelationStateEncoder`
  - adds relation-aware assignment scoring in `LorentzAssignment`
- `modules/model.py`
  - upgrades `LSENet` to carry persistent relation state across levels
- `modules/dsi.py`
  - applies relation state to message graph and SI hierarchy while keeping `_si_loss()` unchanged
- `exp.py`, `main.py`, `tools/run_preset.py`
  - expose config knobs and preset entry
- `configs/presets/b50_v40_relation_state.json`
  - starter preset

## One compatibility fix added here

The hidden V40 patch in the bundle upgrades the leaf message gate to consume **relation state** dimensions.
When `edge_msg_conditioned=True`, the leaf `embed_leaf()` path must therefore receive **encoded relation state**, not raw `edge_attr`, during the KNN/augment-graph stage as well.

This drop-in includes that extra fix in `modules/dsi.py` through `_encode_relation_state_for_adj(...)` and by pre-encoding the leaf edge state before calling `embed_leaf(...)`.

## Main new hyperparameters

- `edge_relation_channels`
- `edge_relation_hidden_dim`
- `edge_relation_assign_scale`
- `edge_relation_reg_lambda`

## Starter preset

Use:

```bash
python tools/run_preset.py --preset b50_v40_relation_state --dataset <your_dataset>
```
