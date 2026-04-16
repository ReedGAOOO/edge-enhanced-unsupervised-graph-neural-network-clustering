# Expert Brief: Current Task, Bottlenecks, and Key Code

## One-Paragraph Task Description

We are working on **edge-enhanced unsupervised graph clustering** built on the DSE/LSEnet-style structural-entropy framework. The current goal is **not just to improve benchmark scores**, but to answer a stricter question: **can edge information be integrated into the Lorentz-tree / structural-entropy clustering pipeline in a way that is mechanistically meaningful, numerically stable, and genuinely useful beyond synthetic control settings?** The current mainline is `B45`, which extends the baseline at three points simultaneously: it uses edge information to reweight the **leaf message-passing graph**, to modulate the **assignment branch**, and to reweight the **SE-consistent structure graph**, while keeping the scalar `_si_loss()` trunk unchanged. In controlled synthetic suites, this branch clearly improves over both the no-edge baseline and the simpler SE-consistent branch (`G20`), but on real-world datasets the evidence is still mixed and task-dependent.

## Current Experimental Status

What is already established:

- On the current control suite, `B45` is stronger than `G20` and the baseline.
- On mechanism-oriented synthetic datasets, edge semantics are genuinely being used rather than ignored.
- The current edge-aware design is more coherent than earlier variants that only injected edge information at one stage.

What is **not** established yet:

- We do **not** yet have strong evidence that the current mainline generalizes robustly to real graph datasets with native, semantically meaningful edge information.
- We do **not** yet have a paper-faithful baseline for the original conference model; the released `reference` code is a hybrid implementation and should not be treated as a strict ICML 2024 baseline.

## Main Bottlenecks

### 1. Baseline Ambiguity

The released `reference` code is neither a clean ICML 2024 `LSEnet` implementation nor a complete ASIL/2504 implementation. This makes it hard to claim that improvements are over a paper-faithful baseline rather than over a hybrid released implementation.

### 2. Dataset Mismatch

Most standard public graph-clustering benchmarks (e.g. citation/Amazon graphs) do not have native `edge_attr`, so many earlier experiments relied on constructed edge features. Those are useful for mechanism analysis, but weak as evidence for real semantic edge modeling.

### 3. Real-World Transfer Is Not Stable

On real datasets with native or relation-derived edge semantics (DBLP author graph, Entities, Fraud, Bitcoin, urban graphs), the results are mixed:

- some datasets are too heterogeneous for the current homogeneous-tree pipeline,
- some have weak/partial labels,
- some are not naturally clustering tasks,
- some require graph conversions that may destroy the original relation semantics.

### 4. Current Edge Integration Is Still Mostly “Graph Reparameterization”

Even in the stronger variants, edge information is mostly used to produce:

- scalar edge weights,
- bounded gates,
- reweighted adjacency matrices.

This works well in controlled settings, but may still be too weak for real multi-relational or heterogeneous edge semantics.

### 5. Evaluation Is Still a Bottleneck

For several real datasets, external labels are weak, partial, or task-misaligned. This means NMI/ARI alone are not enough; structure-aware metrics and stability metrics matter more, but the field-standard evaluation story is still incomplete.

### 6. Scalability Limits

Large graphs still expose bottlenecks:

- some datasets OOM,
- some code paths still rely on dense operations,
- this limits the breadth of realistic validation.

## Hard Experimental Evidence

### A. Control-Suite Success

From `results/diagnostic_b45_confirm_grid9_v1/summary_by_condition.csv`:

- `b45_v31_msgcond_gs050`: `NMI = 0.16001`, `ARI = 0.15256`
- `g20_se_consistent_main`: `NMI = 0.11657`, `ARI = 0.10914`

This is the strongest current evidence that the **current edge-aware mainline is structurally useful**.

### B. Edge Semantics Are Actually Used

From `results/benchmark_mechanism_permEA_v1/permutation_effect_summary.csv`:

- `g20_se_consistent_main` drops by about `0.07875` NMI after permuting `edge_attr`
- `g15_echf_main` drops only about `0.00301`

This is the strongest current evidence that **the model is not merely adding parameters; it is actually using edge semantics**.

### C. Real-World Evidence Is Still Mixed

Representative real-data results:

- `results/dblp_magnn_integration_v1/summary.csv`
  - baseline `NMI = 0.2316`
  - edge-aware branches are lower
- `results/fraud_integration_v1/summary.csv`
  - all variants are effectively near-zero NMI
- `results/entities_integration_v1/summary.csv`
  - only a subset is workable; others are weak or OOM
- `results/urban_v1_vs_v2_baseline_g15_g20_e20_h32_knn0_s0/paired_v1_vs_v2_summary_by_condition.csv`
  - urban graph redesign helps `G20` slightly, but not broadly

So the current state is:

> **mechanism success is real, but real-world generalization success is not yet proven.**

## Key Current Code

### 1. Preset Entry

Default preset selection:

- `tools/run_preset.py:43`
- `tools/run_preset.py:80`

```python
"b45_v31_msgcond_gs050": "configs/presets/b45_v31_msgcond_gs050.json",
...
parser.add_argument("--preset", type=str, default="b45_v31_msgcond_gs050")
```

### 2. Mainline Preset Definition

`B45` preset:

- `configs/presets/b45_v31_msgcond_gs050.json`

```json
{
  "edge_variant": "V31",
  "edge_adaptive_alpha": true,
  "edge_msg_conditioned": true,
  "edge_msg_gate_scale": 0.50,
  "edge_weight_learn_apply_to": "both"
}
```

Reference branch and plain baseline:

- `configs/presets/g20_se_consistent_main.json`
- `configs/presets/baseline_v1.json`

### 3. Leaf Encoder: Edge-Conditioned Message Passing

- `modules/model.py:38`
- `modules/model.py:126`
- `modules/layers.py:13`
- `modules/layers.py:82`
- `modules/layers.py:131`

Core idea: `B45` lets `edge_attr` directly gate the leaf message graph **before** the partition tree is built.

```python
self.input_proj = LorentzGraphConvolution(
    ...,
    edge_conditioned=self.edge_msg_conditioned,
    edge_attr_dim=edge_attr_dim,
    edge_gate_scale=self.edge_msg_gate_scale,
)
...
x = self.input_proj(x, adj, edge_attr=edge_attr, edge_mask=edge_mask, use_edge_attr=msg_use_edge_attr)
```

and in the aggregator:

```python
gate_raw = self.edge_gate_mlp(edge_attr).squeeze(-1)
gate_raw = torch.tanh(gate_raw) * float(self.edge_gate_scale)
factor = torch.exp(gate_raw)
gated = torch.sparse_coo_tensor(
    adj_sp.indices(),
    adj_sp.values() * factor,
    size=adj_sp.size(),
).coalesce()
gated = self._normalize_sparse_by_degree(gated)
```

### 4. SE-Consistent Edge Reweighting and Assignment Residual

- `modules/dsi.py:197`
- `modules/dsi.py:239`
- `modules/dsi.py:243`
- `modules/dsi.py:296`

Core idea: `B45` keeps the original `_si_loss()` scalar trunk, but learns edge-weight factors for both the message graph and the SE graph, using `V31`.

```python
adj_train_msg = (self.alpha * adj_aug + adj_base_msg).coalesce()
adj_train_si = (self.alpha * adj_aug + adj_base_si).coalesce()

if self._use_learnable_edge_weight_variant():
    adj_train_si, reg_si = self._apply_learned_edge_weight_to_adj(...)
    adj_train_msg, reg_msg = self._apply_learned_edge_weight_to_adj(...)
```

Variant switch:

```python
def _use_edge_attr_variant(self) -> bool:
    return self.edge_variant in {'V6', 'V7', 'V8', 'V12', 'V13', 'V31', 'V32', 'V33'}

def _use_learnable_edge_weight_variant(self) -> bool:
    return self.edge_variant in {'V20', 'V30', 'V31', 'V32', 'V33'}
```

This is the main difference between:

- `baseline_v1`: no explicit edge-aware path
- `g20_se_consistent_main`: SE-consistent scalar edge weighting
- `b45_v31_msgcond_gs050`: SE-consistent weighting **plus** message-conditioned leaf propagation

### 5. Training Loop

- `exp.py:147`
- `exp.py:185`

```python
model = DSI(...)
optimizer = AdamW(model.parameters(), lr=self.configs.lr, weight_decay=self.configs.w_decay)
```

The model records extra diagnostics for branch health, edge factors, and stability.

## Short Expert Question

The precise question we want expert feedback on is:

> In a Lorentz-tree / structural-entropy clustering framework, is the current direction of edge integration (`B45`: message-conditioned leaf propagation + SE-consistent dual edge weighting + assignment residual) the right structural path for real semantic edge information, or is it still too close to “edge-driven graph reparameterization” and therefore unlikely to generalize on real heterogeneous graphs? If so, what is the most principled next step: stronger edge-state modeling, hetero-graph reformulation, or a more task-aligned decoding/evaluation protocol?

