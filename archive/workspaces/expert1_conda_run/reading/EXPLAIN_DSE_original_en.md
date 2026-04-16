# Original DSE (Reference) Explained

This note explains the original DSE implementation in:
`reference/DSE_clustering-main`

It focuses on:
1. Corrected dataflow graph
2. Training-stage mapping (how the model learns clustering behavior)
3. Inference-stage mapping (how final cluster labels are produced)

---

## Corrected Dataflow (with key concepts)

```text
[Raw Dataset]
  |- node features: X
  `- edges: edge_index
        |
        v
[Build Original Graph]
  A0 = normalize(edge_index)             (original adjacency / graph prior)
        |
        +--------------------+
        |                    |
        v                    v
[Leaf Embedding Pipeline]    [Original Topology Prior]
  X --(append 0 dim)--> [0, X]               A0
      --(expmap0)--> X_Lorentz               |
      --(LorentzGraphConv #1 with A0)-->     |
      --(LorentzGraphConv #2 with A0)--> z_leaf
        |                                     |
        v                                     |
[Learned Augmented Graph from z_leaf]         |
  similarity(z_leaf, z_leaf)                  |
      -> topK + softmax -> A_aug              |
        |                                     |
        +-------------------+-----------------+
                            |
                            v
[Mixed Training Graph]
  A_train = A0 + alpha * A_aug
  (who can influence whom + how strong)
                            |
                            v
[Assignment Core]
  ass0 = softmax(W * logmap0(z))
  att  = softmax_j(-dist(q_i, k_j)) on A_train edges
  ass1 = att @ ass0
  S    = gumbel_softmax(log(ass1))
                            |
                            v
[Hierarchical Coarsening]
  X_parent = S^T X_current
  A_parent = S^T A_current S
```

---

## Training-Stage Mapping (learns how to cluster)

| Simple step | What code does | Code reference |
|---|---|---|
| 1. Build leaf embedding in Lorentz space | Append one dim, `expmap0`, then two Lorentz graph conv layers | `reference/DSE_clustering-main/modules/model.py:36` |
| 2. Build learned augmented graph | Compute pair similarity on `z_leaf`, keep top-K edges as `adj_aug` | `reference/DSE_clustering-main/modules/dsi.py:69` |
| 3. Mix augmented and original graph | `A_train = data.adj + alpha * adj_aug` and feed encoder | `reference/DSE_clustering-main/modules/dsi.py:73` |
| 4. Per-layer assignment | Soft assignment -> edge attention propagation -> gumbel hardening | `reference/DSE_clustering-main/modules/layers.py:117` |
| 5. Hierarchical coarsening by assignment | `x_par = ass^T x`, `adj_par = ass^T adj ass` | `reference/DSE_clustering-main/modules/layers.py:142` |
| 6. Structural entropy objective | `_si_loss` accumulates `delta_vol * log(degree/parent_vol)` over layers | `reference/DSE_clustering-main/modules/dsi.py:77` |
| 7. Parameter update | Backprop + optimizer step using `se_loss` | `reference/DSE_clustering-main/exp.py:62` |

---

## Inference-Stage Mapping (produces final cluster labels)

| Simple step | What code does | Code reference |
|---|---|---|
| 1. Run encoder once to get hierarchy outputs | Returns `coord_dict`, `ass_dict` | `reference/DSE_clustering-main/modules/dsi.py:32` |
| 2. Multiply assignments across layers | `clu_mat[k] = clu_mat[k+1] @ ass_dict[k+1]` | `reference/DSE_clustering-main/modules/dsi.py:40` |
| 3. Convert to hard cluster matrix | Row-wise `argmax` then one-hot rewrite | `reference/DSE_clustering-main/modules/dsi.py:43` |
| 4. Small-cluster repair (optional) | Reassign nodes from tiny clusters by Lorentz similarity | `reference/DSE_clustering-main/modules/dsi.py:49` |
| 5. Evaluate labels | NMI/ARI (and ACC in reference script) | `reference/DSE_clustering-main/exp.py:76` |

---

## What to remember

1. Assignment matrix `S` is the direct clustering action.
2. `se_loss` is the training signal that shapes `S`.
3. Final labels come from chained assignment matrices across hierarchy levels.
