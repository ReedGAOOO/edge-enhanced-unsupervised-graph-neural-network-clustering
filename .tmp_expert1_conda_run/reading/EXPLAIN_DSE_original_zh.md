# 原版 DSE（Reference）机制说明

本文档解释原版 DSE 实现：
`reference/DSE_clustering-main`

重点包括：
1. 修正后的数据流图
2. 训练阶段映射（模型如何学会“怎么分簇”）
3. 推理阶段映射（如何输出最终簇标签）

---

## 修正数据流图（含核心概念）

```text
[原始数据]
  |- 节点特征: X
  `- 边: edge_index
        |
        v
[构建原图]
  A0 = normalize(edge_index)             (original adjacency / 原始拓扑先验)
        |
        +--------------------+
        |                    |
        v                    v
[叶子层编码流水线]            [原始拓扑先验分支]
  X --(补0维)--> [0, X]                  A0
      --(expmap0)--> X_Lorentz           |
      --(LorentzGraphConv #1 with A0)--> |
      --(LorentzGraphConv #2 with A0)--> z_leaf
        |                                 |
        v                                 |
[基于 z_leaf 学习增强图]                  |
  similarity(z_leaf, z_leaf)              |
      -> topK + softmax -> A_aug          |
        |                                 |
        +-------------------+-------------+
                            |
                            v
[训练混合图]
  A_train = A0 + alpha * A_aug
  (决定谁能影响谁、影响强度多大)
                            |
                            v
[Assignment 核心]
  ass0 = softmax(W * logmap0(z))
  att  = softmax_j(-dist(q_i, k_j)) on A_train edges
  ass1 = att @ ass0
  S    = gumbel_softmax(log(ass1))
                            |
                            v
[层级粗化]
  X_parent = S^T X_current
  A_parent = S^T A_current S
```

---

## 训练阶段映射（学到“怎么分簇”）

| 简化步骤 | 代码实际行为 | 代码位置 |
|---|---|---|
| 1. 叶子层双曲表示 | 补1维，`expmap0`，再过两层 Lorentz 图卷积 | `reference/DSE_clustering-main/modules/model.py:36` |
| 2. 构造学习增强图 | 在 `z_leaf` 上算相似度，取 top-K 得到 `adj_aug` | `reference/DSE_clustering-main/modules/dsi.py:69` |
| 3. 增强图与原图混合 | `A_train = data.adj + alpha * adj_aug` 并送入编码器 | `reference/DSE_clustering-main/modules/dsi.py:73` |
| 4. 每层 assignment | 先 soft assignment，再边注意力传播，再 gumbel 硬化 | `reference/DSE_clustering-main/modules/layers.py:117` |
| 5. 基于 assignment 粗化 | `x_par = ass^T x`，`adj_par = ass^T adj ass` | `reference/DSE_clustering-main/modules/layers.py:142` |
| 6. 结构熵目标 | `_si_loss` 按层累计 `delta_vol * log(degree/parent_vol)` | `reference/DSE_clustering-main/modules/dsi.py:77` |
| 7. 参数更新 | 用 `se_loss` 反传并更新参数 | `reference/DSE_clustering-main/exp.py:62` |

---

## 推理阶段映射（产出最终簇标签）

| 简化步骤 | 代码实际行为 | 代码位置 |
|---|---|---|
| 1. 跑一次编码器拿层级输出 | 得到 `coord_dict`、`ass_dict` | `reference/DSE_clustering-main/modules/dsi.py:32` |
| 2. 多层 assignment 连乘 | `clu_mat[k] = clu_mat[k+1] @ ass_dict[k+1]` | `reference/DSE_clustering-main/modules/dsi.py:40` |
| 3. 硬化成簇矩阵 | 每行 `argmax` 后改写为 one-hot | `reference/DSE_clustering-main/modules/dsi.py:43` |
| 4. 小簇修复（可选） | 对极小簇节点按双曲相似度重分配 | `reference/DSE_clustering-main/modules/dsi.py:49` |
| 5. 聚类评估 | 计算 NMI/ARI（参考脚本里也有 ACC） | `reference/DSE_clustering-main/exp.py:76` |

---

## 抓住三点就够

1. `S`（assignment）是“真正执行分簇动作”的矩阵。
2. `se_loss` 是训练时唯一驱动“分得好不好”的目标信号。
3. 最终标签来自“多层 assignment 连乘”后的结果。
