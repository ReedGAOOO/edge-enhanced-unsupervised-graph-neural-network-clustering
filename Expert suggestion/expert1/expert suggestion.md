我看完材料后的判断很明确：

**B45 值得保留，但它更应该被定位成“最后一个强 scalar-control 主线”，而不是最终可泛化答案。**
它已经证明了两件重要事情：第一，边信息在这个 Lorentz tree / structural entropy 框架里**确实能被有效利用**；第二，把边信息同时放进 leaf message passing、assignment、SE-consistent 图三处，明显比只在单一路径注入更合理。你们自己的控制套件里，`b45_v31_msgcond_gs050` 明显强于 `g20_se_consistent_main`；而且 edge_attr permutation 会让相关分支显著掉点，这说明模型不是在“白加参数”，而是在用边语义。与此同时，brief 也很清楚地说了：真实数据上的泛化证据还不稳，baseline 也还不够 paper-faithful。

从代码层面看，B45 的核心确实就是你概括的那条主线：
leaf 端用 `edge_attr` 做消息门控；assignment 端用边相关 residual 改写分配打分；SE 端继续做双路 edge weighting；而 `_si_loss()` 主干本身没有被改写，只是输入进去的图被改写了。当前实现里，leaf 门控是 `edge_gate_mlp(edge_attr) -> tanh -> exp(factor)` 再做 degree normalization；SE 侧则是 `edge_weight_mapper(edge_attr)` 产出有界 log-score，经 `exp` 变成 factor，分别施加到 message 图和 SI 图；assignment 侧的 `V31` 也是把结构项与 attr residual 混进 edge score。换句话说，**边信息几乎一直在以“修改边权 / 修改边分数”的方式发挥作用**。

这也是我认为它在真实图上容易撞墙的根本原因：
**现在的边建模，本质上仍然是“context-free scalar graph reparameterization”。**

问题不只在“是 scalar，不够强”，而在于它大体上假设每条边都能被压成一个一维“对聚类有多帮助”的量，而且这个量主要由 edge_attr 自身决定。当前 `edge_weight_mapper` 和 leaf gate 都是直接从 `edge_attr` 出发的；hierarchical pooling 也只是把 edge_attr 按 assignment 权重往上平均。这在机制合成数据上会很好用，因为那类数据里 edge_attr 往往就等于“同簇支持度”；但在真实图里，同一种关系往往是**上下文相关**的：同样的 relation type，在某些局部是 cohesion，在另一些局部是 bridge、role-complementarity，甚至是对 clustering 无关的 interaction。当前主线几乎没有显式表示这种“边角色随端点上下文而变”的能力。

所以我对“下一步最该走哪条线”的回答是：

## 先走 **context-conditioned edge-state modeling**，而不是继续堆新的 scalar 变体

不是马上跳去 heterograph reformulation；也不是只改评价协议。
**最值得做的下一步，是把边从“权重”升级成“状态”。**

我建议新分支直接换研究对象，不再学习
`edge -> scalar helpfulness`
而是学习
`edge + endpoint context -> edge role/state`。

最小够用版本甚至不需要很复杂，先做 3 态就行：

* `support`：这条边支持“同簇聚合”
* `boundary/bridge`：这条边更像“跨簇边界”或“角色连接”
* `neutral/noise`：这条边对当前聚类任务弱相关

形式上可以是：

```text
q_ij = softmax( g(e_ij, h_i, h_j, h_i ⊙ h_j, d_L(h_i,h_j)) )
```

其中 `q_ij ∈ Δ^2`，是 edge state posterior。
这里最关键的是：**输入必须带端点上下文**，不能再只看 `edge_attr` 本身。

然后把它接进现有框架时，不要一口气推翻 `_si_loss()`，而是这样接：

```text
A_support(i,j) = w_ij * q_ij[support]
A_boundary(i,j) = w_ij * q_ij[boundary]
```

* message passing 主要走 `A_support`
* `_si_loss()` 继续跑在 `A_support` 或其 fused graph 上
* `A_boundary` 不喂进 SE 主干当“正边”，而是进一个辅助项，惩罚它落在同簇内部

例如可以加一个很直接的无监督辅助项：

```text
L_boundary = Σ_(i,j) A_boundary(i,j) * <p_i, p_j>
L_support  = Σ_(i,j) A_support(i,j) * (1 - <p_i, p_j>)
```

其中 `p_i` 是顶层或某一层 soft assignment。
这一步的含义非常重要：**不是所有语义边都应该被映射成“更大的同簇权重”。**
真实图失败，很多时候恰恰是因为你把 boundary edge 也硬塞进了同一条单调标尺里。

## 为什么这条路比继续做 V31/V32/V33 更像“真正泛化”

因为它同时解决了你们当前最核心的两个结构性短板。

第一，它把“边语义”从**一维强弱**变成了**多角色判别**。
当前 B45 的 leaf gate、SE factor、assignment residual，虽然入口有三处，但语义上还是一件事：把边往“更强/更弱”方向推。可真实图里最常见的不是“这条边强不强”，而是“这条边**代表什么关系**，以及这个关系在当前端点上下文下究竟该不该支持同簇”。这一步只有 edge-state 才能表达。

第二，它仍然保留了 DSI / ASIL 这一套最宝贵的理论骨架。
2504 里真正有理论抓手的是：把原图和 representation-induced virtual graph 融合后，用 structural entropy / DSI 去做 conductance-aligned 的优化。也就是说，理论抓手落在“**什么图被送进 structural entropy**”上，而不是任意地方都加 scalar gate。你们现在保持 `_si_loss()` 主干不变这一点其实是对的；只是“送进去的图”应该从简单的重标定图，升级为**经过 edge-state 过滤后的 support graph**。这样一来，SE 主干仍然是 conductance-interpretable 的，boundary 信号则通过单独正则表达“应被切开”的证据。这个方向比继续堆 residual 更接近 ASIL 的原始逻辑。

## 为什么我不建议第一步就直接跳 heterograph reformulation

因为现在最缺的还不是“多关系图框架”，而是一个更干净的因果判断：

> 失败到底是因为“单树不行”，还是因为“你们把边语义压成了 context-free scalar，所以单树还没被真正检验过”？

如果现在直接跳 heterograph / multi-tree，会把 baseline、evaluation、可扩展性一起重新打散。你们目前已经有一个很强的控制结论：B45 这条三点注入路线在机制数据上有效。最自然的下一步，就是在**同一总体框架里**把边从 scalar 提升为 state，看真实图能不能因此明显变稳。只有当这一步仍然失败，而且出现“不同 edge role 对应的最优 partition 明显彼此冲突”时，才有充分理由说：**不是 edge modeling 不够，而是 homogeneous-tree clustering 本身与任务不匹配。** 这时再转 heterograph，证据链才干净。

## 具体到你们当前代码，最小可行实现其实很顺

现有主线里已经有几块很适合复用的结构：

* `edge_weight_mapper` 已经能输出多头分数；现在只是语义上拿来当 `msg/si/aug` 因子。可以把它改成输出 `support/boundary/neutral` 的 logits。
* `_align_edge_attr_to_adj_with_mask()` 已经能把 base edge_attr 对齐到 target adjacency。
* V32 的 hierarchical edge pooling 已经能把**向量型 edge_attr**往上层池化；把 raw edge_attr 换成 edge-state posterior 就行。
* `_si_loss()` 已经是 sparse-aware 的，不需要动主干，只要把 `adj_train_si` 替换成 `A_support` 路径即可。

所以我会把新分支定义成类似下面的顺序：

1. **V40 / edge-role simplex**
   先做 `support/boundary/neutral` 三态，不加 hetero，不改 `_si_loss()`。

2. **leaf message 只吃 `support`，assignment 同时看 `support + boundary`**
   这样 message 不会再被 boundary edge 错误增强；assignment 才负责解释“这条边更像 cut evidence”。

3. **hierarchy 上池化 edge-state posterior，而不是原始 edge_attr**
   这一步会比当前 V32 更有语义，因为 pooled 的不再是原始 feature 平均，而是“上一层学到的边角色分布”。

4. **candidate augment edge 也要有 edge-state prior**
   当前 V33 只是给 augment candidate 加 attr bias，但新边本身没有真正的 relation state。V40 应该让 candidate edge 也通过一个 pairwise predictor 产生 `q_ij`，否则 augment 路径仍然是 relation-blind。

## 我会把“是否真泛化”定义成这几条验证，而不是只看 NMI/ARI

你们 brief 已经说得很对：很多真实数据 weak-label、task-misaligned，单看 NMI/ARI 不够。当前 mainline 其实已经在输出不少更靠谱的结构指标和 branch health 诊断，包括 modularity、conductance、stability_pair_nmi，以及 factor / msg / assign / hier / aug 的 liveness 统计。新分支必须把这些指标纳入主结论，而不是只保留作 debug。

我会强制做 4 个关键对照：

* **role-collapse ablation**：把三态 edge-state 压回单 scalar，应该退化回 B45 水平。否则说明新分支只是加参。
* **endpoint-context ablation**：去掉 `h_i,h_j`，只保留 edge_attr。若真实图明显退步、机制图不退步，就直接证明“真实泛化差距来自上下文依赖的边语义”。
* **support vs boundary calibration**：看 `A_boundary(i,j)` 是否真的更容易落在预测 cut 上，而不是仍然变成“弱 support”。
* **native-edge-only protocol**：把没有原生 edge_attr 的 constructed-edge 数据集明确降格为机制验证，不再拿来支撑真实泛化结论。

## 还有两件并行但不能混在一起的事

第一，**baseline 必须拆干净**。
brief 已经指出 released `reference` 不是 clean ICML 2024 LSEnet，也不是完整 ASIL/2504；而 reference `se_loss()` 里确实已经有 `LorentzBoost + graph_top_K + alpha * adj_aug + data.adj` 这一套 augmentation。也就是说，不把 baseline 拆成“paper-faithful ICML24”和“released hybrid reference”两条锚线，你们在真实数据上的任何结论都会被 baseline 争议拖住。

第二，**可扩展性要守住“不额外引入新的 dense O(N²)”**。
当前代码已经在用 `knn_mode=edge/auto` 回避大图全对全 candidate，但小图路径里仍会出现 dense pairwise score；层间 coarsening 里也有 `ass^T adj ass` 的 dense 化。新 edge-state 分支不能再加一个新的全对全 edge-state matrix；必须坚持“状态只存在于 sparse edge 上，relation 数量常数级（3 或 4）”。否则你们会在还没回答科学问题前先被 OOM 吃掉。

最后把我的结论压缩成一句话：

**B45 这条路在“边能否被利用”这个问题上已经成功，但在“边语义能否泛化”这个问题上还停留在 context-free scalar reparameterization。下一步最 principled 的不是继续做新的 scalar gate，也不是立刻跳 heterograph，而是做“context-conditioned multi-role edge-state”，并继续把 `_si_loss()` 作为 support graph 上的主干。**

这条线一旦跑通，你们就能非常清楚地区分三种情况：
是 edge 信息本身没用；是 scalar 边建模不够；还是 homogeneous-tree clustering 本身就和任务不匹配。

如果你愿意，我下一条可以直接把这个 **V40 edge-state 分支** 写成更接近代码层的模块设计和 loss 草案。


好，我把 **V40** 直接写成一版“能嵌进当前骨架”的实现草案。

## 我会把 V40 定义成什么

不是“大改框架”，而是一个**最小侵入的新分支**：

* `_si_loss()` **完全不改**
* `adj_aug` / `ass^T A ass` / 层级树主干 **完全保留**
* 新增一个 **context-conditioned edge-state predictor**
* 把当前 raw `edge_attr` 先变成三态 posterior
  `q_ij = [support, boundary, neutral]`
* 再把 `q_ij` 同时用于：

  1. message graph 的 support 过滤
  2. assignment score 的 support-vs-boundary 残差
  3. hierarchy 中的 edge-state pooling
  4. 一个显式的 boundary auxiliary loss

这条线和你们现在的代码骨架是兼容的，因为当前主干里已经有：

* `DSI.se_loss()` 负责图构造与 `_si_loss()` 调用；
* `LSENet.embed_leaf()` 已经支持 leaf message 端吃 `edge_attr`；
* `LSENet.forward()` 已经支持层间 `edge_attr_hierarchical` pooling；
* assignment 分支已经有 edge-aware residual 注入口。

---

## 先说最关键的工程判断

### V40 先做成 **V40A：leaf-context state + hierarchy pooling**

不要第一步就做“每层重新推一次 edge-state”。

也就是：

* **leaf 层**先用端点上下文 + raw `edge_attr` 推出 `q_ij`
* 往上层时，不重新做 pairwise predictor
* 而是复用你们现成的 `edge_attr_hierarchical`，把 `q_ij` 当成新的 edge_attr 向量往上池化

这是最稳的第一版，因为当前代码已经能把向量型 edge_attr 按 parent-pair 权重对齐和聚合；只要把输入从 raw `edge_attr` 换成 state simplex 即可。

我会把“每层重新推 state”的版本留给 **V41**。
先把 V40A 跑通，因果会更干净。

---

## 现有代码里，V40 要解决的那个真问题是什么

当前 V20-V33 的 learnable edge weighting，核心仍然是：

* 先把 `base_edge_attr` 对齐到 `target_adj`
* 然后只把对齐后的 `edge_attr` 喂进 `edge_weight_mapper`
* 对未匹配边，直接把 factor 设回 1。

这意味着一件很具体的事：

> **augment 出来的新边，只要没有原始 edge_attr 对齐，就基本回到 identity / relation-blind。**

而 assignment 侧当前也是对 `edge_attr` 直接过 `edge_attr_encoder`，再产出 `attr_bias/attr_gate` 加到 score 上，没有显式用端点上下文 `h_i,h_j` 去判断“同一种 edge_attr 在这个局部到底是 support 还是 boundary”。

V40 要补的，正是这个洞：

> 让 **matched edge** 和 **augment edge** 都能得到
> `edge + endpoint context -> role posterior`

---

## 模块怎么放

我建议新增一个类：

```python
class EdgeStatePredictor(nn.Module):
    ...
```

放法有两个：

1. 最省事：直接放进 `modules/dsi.py`
2. 更干净：新建 `modules/edge_state.py`

我更偏向第 2 个，但为了少改文件，第 1 个也完全可以。

---

## 先改一个非常重要的接口：把 raw edge_attr dim 和 encoder edge_attr dim 分开

当前 `DSI.__init__()` 只有一个 `edge_attr_dim`，并且它直接传给 encoder。

这在 V40 会出问题，因为：

* predictor 吃的是 **raw edge_attr_dim**
* encoder 吃的应该是 **state_dim = 3**

所以第一步不是写 predictor，而是把 `DSI.__init__()` 改成两个维度：

```python
self.raw_edge_attr_dim = int(max(1, edge_attr_dim))
self.edge_state_num_roles = 3

encoder_edge_attr_dim = (
    self.edge_state_num_roles if self._use_edge_state_variant() else self.raw_edge_attr_dim
)
```

然后：

```python
self.encoder = LSENet(
    ...,
    edge_attr_dim=encoder_edge_attr_dim,
    ...
)
```

否则你后面把 `q_ij` 传给 encoder，会直接维度不匹配。

---

## V40 的 predictor 输入，我建议这样做

### 输入不是只看 `edge_attr`

而是每条目标边 `(i,j)` 都构造一个 pair feature：

```python
phi_ij = [
    u_i, u_j,
    abs(u_i - u_j),
    u_i * u_j,
    lorentz_score(i,j),
    lorentz_dist(i,j),
    log_w_ij,
    raw_edge_attr_aligned_ij,
    matched_ij
]
```

其中：

* `u_i = logmap0(z_i)[..., 1:]`
* `u_j = logmap0(z_j)[..., 1:]`
* `z_i,z_j` 用 leaf context embedding
* `raw_edge_attr_aligned_ij`：对齐到当前 `target_adj` 的 raw edge_attr
* `matched_ij`：是否来自原始匹配边
* `log_w_ij`：当前 target edge 的 log weight

输出：

```python
logits_ij = mlp(phi_ij)          # [E, 3]
q_ij = softmax(logits_ij / T)    # [E, 3]
# role order: [support, boundary, neutral]
```

这样做有两个好处：

第一，**matched edge** 能用 raw edge_attr。
第二，**augment edge** 即使 `edge_attr` 缺失，也还能靠端点上下文 + matched flag + edge weight 学到 role，不会再退回 factor=1 的盲区。这个点正好针对当前 V20-V33 的实现缺口。

---

## 我建议的 predictor 代码骨架

```python
class EdgeStatePredictor(nn.Module):
    def __init__(
        self,
        manifold,
        node_dim: int,
        raw_edge_attr_dim: int,
        hidden_dim: int = 128,
        num_roles: int = 3,
        dropout: float = 0.1,
        temp: float = 1.0,
    ):
        super().__init__()
        self.manifold = manifold
        self.temp = float(max(1e-3, temp))
        in_dim = 4 * (node_dim - 1) + 3 + int(max(1, raw_edge_attr_dim)) + 1
        # ui, uj, |ui-uj|, ui*uj, lor_score, lor_dist, log_w, edge_attr, matched
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_roles),
        )

    def forward(self, z_leaf, target_adj, edge_attr_aligned=None, matched=None):
        tgt = target_adj.coalesce()
        idx = tgt.indices()
        val = tgt.values().float()
        src, dst = idx[0].long(), idx[1].long()

        u = self.manifold.logmap0(z_leaf).float()[:, 1:]
        ui = u[src]
        uj = u[dst]

        lor_score = (2 + 2 * self.manifold.cinner(z_leaf[src], z_leaf[dst])).unsqueeze(1).float()
        lor_dist = self.manifold.dist(z_leaf[src], z_leaf[dst]).unsqueeze(1).float()
        log_w = torch.log(val.clamp_min(1e-8)).unsqueeze(1)

        feat = [ui, uj, torch.abs(ui - uj), ui * uj, lor_score, lor_dist, log_w]

        if edge_attr_aligned is None:
            edge_attr_aligned = torch.zeros(
                (src.shape[0], 1), dtype=ui.dtype, device=ui.device
            )
        elif edge_attr_aligned.dim() == 1:
            edge_attr_aligned = edge_attr_aligned.unsqueeze(1)
        feat.append(edge_attr_aligned.float().to(ui.device))

        if matched is None:
            matched_f = torch.ones((src.shape[0], 1), dtype=ui.dtype, device=ui.device)
        else:
            matched_f = matched.float().unsqueeze(1).to(ui.dtype).to(ui.device)
        feat.append(matched_f)

        pair_feat = torch.cat(feat, dim=1)
        logits = self.mlp(pair_feat) / self.temp
        q = torch.softmax(logits, dim=-1)
        return logits, q
```

---

## V40 在 `DSI.__init__()` 里新增什么

### 新开关

```python
def _use_edge_state_variant(self) -> bool:
    return self.edge_variant in {'V40'}
```

### 新参数

我建议先加这些：

```python
edge_state_hidden_dim=128,
edge_state_temp=1.0,
edge_state_support_floor=0.15,
edge_state_support_lambda=1.0,
edge_state_boundary_lambda=1.0,
edge_state_conf_lambda=5e-3,
edge_state_balance_lambda=1e-3,
edge_state_detach_context=True,
edge_state_use_log_edge_weight=True,
```

### 新模块

```python
if self._use_edge_state_variant():
    self.edge_state_predictor = EdgeStatePredictor(
        manifold=self.manifold,
        node_dim=hid_dim + 1,
        raw_edge_attr_dim=self.raw_edge_attr_dim,
        hidden_dim=int(edge_state_hidden_dim),
        num_roles=int(self.edge_state_num_roles),
        dropout=dropout,
        temp=float(edge_state_temp),
    )
else:
    self.edge_state_predictor = None
```

同时加 diagnostics 缓存：

```python
self.last_edge_state_support_mean = 0.0
self.last_edge_state_boundary_mean = 0.0
self.last_edge_state_neutral_mean = 0.0
self.last_edge_state_entropy_mean = 0.0
self.last_edge_state_support_same_mean = 0.0
self.last_edge_state_boundary_same_mean = 0.0
```

---

## `se_loss()` 里怎么接

当前 `se_loss()` 的基本顺序是：

1. 从 `adj_base_msg/adj_base_si` 出发
2. 用 `embed_leaf + lorentz_proj` 构造 `adj_aug`
3. 形成 `adj_train_msg = alpha * adj_aug + adj_base_msg`
4. 形成 `adj_train_si = alpha * adj_aug + adj_base_si`
5. 再进入 edge weighting / encoder / hierarchy / `_si_loss()`。

V40 我建议只在这中间插一段，不碰 trunk：

### 第 0 步：先拿一个 leaf context

而且第一版建议**不用 raw edge_attr**，避免 predictor 先被 message gate 反向污染：

```python
z_ctx = self.encoder.embed_leaf(
    data.x,
    adj_base_msg,
    edge_attr=None,
    edge_mask=None,
    use_edge_attr=False,
)
z_ctx = self.lorentz_proj(z_ctx)
if self.edge_state_detach_context:
    z_ctx = z_ctx.detach()
```

这里这么做是因为当前 `embed_leaf()` 只有在 `edge_msg_conditioned=True` 且 `use_edge_attr=True` 时才真正把 edge_attr 用进 leaf message。V40 第一版我建议先把这条链断开，避免“先用边控消息，再用这个 embedding 反过来预测边角色”的自证循环。

### 第 1 步：照旧建 `adj_aug`

完全复用现有逻辑。

### 第 2 步：得到 raw target graphs

```python
adj_train_msg_raw = (self.alpha * adj_aug + adj_base_msg).coalesce()
adj_train_si_raw  = (self.alpha * adj_aug + adj_base_si).coalesce()
```

### 第 3 步：分别在 target graph 上预测 q

```python
raw_attr_msg, matched_msg = self._align_edge_attr_to_adj_with_mask(
    adj_base_msg, base_edge_attr, adj_train_msg_raw
)
_, q_msg = self.edge_state_predictor(
    z_ctx, adj_train_msg_raw, edge_attr_aligned=raw_attr_msg, matched=matched_msg
)

raw_attr_si, matched_si = self._align_edge_attr_to_adj_with_mask(
    adj_base_si, base_edge_attr, adj_train_si_raw
)
_, q_si = self.edge_state_predictor(
    z_ctx, adj_train_si_raw, edge_attr_aligned=raw_attr_si, matched=matched_si
)
```

### 第 4 步：把 support posterior 变成 support graph

我不建议直接 `adj *= q_support`，太容易把图打穿。
建议保留一个 residual floor：

```python
support_floor = float(self.edge_state_support_floor)
factor_msg = support_floor + (1.0 - support_floor) * q_msg[:, 0]
factor_si  = support_floor + (1.0 - support_floor) * q_si[:, 0]
```

然后：

```python
adj_train_msg = self._apply_state_factor_to_adj(
    adj_train_msg_raw, factor_msg, normalize_for_message=True
)
adj_train_si = self._apply_state_factor_to_adj(
    adj_train_si_raw, factor_si, normalize_for_message=False
)
```

这里 residual floor 的作用是：

* 保证 `_si_loss()` 的 `vol_G` 不会塌
* 保证 V40 第一版更像“support-filtered trunk”而不是“全新图任务”

当前 `_si_loss()` 本来就是吃 leaf adjacency，然后按层 `ass^T A ass` 上卷；只要把 leaf `adj_train_si` 换成 support-filtered 版本，主干就成立。

### 第 5 步：把 `q_msg` 当成新的 encoder edge_attr

```python
edge_attr_for_encoder = q_msg
edge_mask_for_encoder = None
```

然后：

```python
_, ass_aug_dict, _ = self.encoder(
    data.x,
    adj_train_msg,
    edge_attr=edge_attr_for_encoder,
    edge_mask=edge_mask_for_encoder,
    use_edge_attr=True,
)
adj_si_dict = self._build_hierarchy_adj_from_assign(adj_train_si, ass_aug_dict)
loss = self._si_loss(ass_aug_dict, adj_si_dict, eps)
```

### 第 6 步：加一个 state auxiliary

后面我单独写公式。

---

## `DSI.se_loss()` 的伪代码长这样

```python
def se_loss(self, data, eps=1e-6):
    adj_base_msg = getattr(data, "adj_msg", data.adj).clone()
    adj_base_si = getattr(data, "adj_si", adj_base_msg).clone()
    base_edge_attr = getattr(data, "edge_attr", None)

    # 1) leaf context for edge-state prediction
    z_ctx = self.encoder.embed_leaf(
        data.x, adj_base_msg,
        edge_attr=None, edge_mask=None, use_edge_attr=False
    )
    z_ctx = self.lorentz_proj(z_ctx)
    if self.edge_state_detach_context:
        z_ctx = z_ctx.detach()

    # 2) same augment-graph path as current mainline
    if self._use_edge_knn_mode(data.x.shape[0]):
        with torch.no_grad():
            adj_aug = self._edge_candidate_adj(z_ctx, adj_base_msg, self.knn, edge_bias=None)
    else:
        neg_dist2 = 2 + 2 * self.manifold.cinner(z_ctx, z_ctx)
        adj_aug = graph_top_K(torch.softmax(neg_dist2 / self.tau, dim=-1), k=self.knn)

    adj_train_msg_raw = (self.alpha * adj_aug + adj_base_msg).coalesce()
    adj_train_si_raw = (self.alpha * adj_aug + adj_base_si).coalesce()

    if self._use_edge_state_variant():
        raw_attr_msg, matched_msg = self._align_edge_attr_to_adj_with_mask(
            adj_base_msg, base_edge_attr, adj_train_msg_raw
        )
        logits_msg, q_msg = self.edge_state_predictor(
            z_ctx, adj_train_msg_raw, raw_attr_msg, matched_msg
        )

        raw_attr_si, matched_si = self._align_edge_attr_to_adj_with_mask(
            adj_base_si, base_edge_attr, adj_train_si_raw
        )
        logits_si, q_si = self.edge_state_predictor(
            z_ctx, adj_train_si_raw, raw_attr_si, matched_si
        )

        adj_train_msg = self._apply_state_factor_to_adj(
            adj_train_msg_raw,
            self.edge_state_support_floor + (1 - self.edge_state_support_floor) * q_msg[:, 0],
            normalize_for_message=True,
        )
        adj_train_si = self._apply_state_factor_to_adj(
            adj_train_si_raw,
            self.edge_state_support_floor + (1 - self.edge_state_support_floor) * q_si[:, 0],
            normalize_for_message=False,
        )

        edge_attr_for_encoder = q_msg
        edge_mask_for_encoder = None
    else:
        ...

    _, ass_aug_dict, _ = self.encoder(
        data.x,
        adj_train_msg,
        edge_attr=edge_attr_for_encoder,
        edge_mask=edge_mask_for_encoder,
        use_edge_attr=True,
    )

    adj_si_dict = self._build_hierarchy_adj_from_assign(adj_train_si, ass_aug_dict)
    loss = self._si_loss(ass_aug_dict, adj_si_dict, eps)

    if self._use_edge_state_variant():
        loss = loss + self._edge_state_aux_loss(
            ass_aug_dict=ass_aug_dict,
            adj_leaf_raw=adj_train_msg_raw,
            q_leaf=q_msg,
        )

    return loss
```

---

## assignment 分支怎么改

当前 assignment 分支里，edge-aware 逻辑是：

```python
edge_attr -> edge_attr_encoder -> attr_bias, attr_gate
score += ...
```

也就是它把 edge_attr 当成一组一般性 feature。

V40 我建议不要继续这么黑盒。
直接加一个专门分支：

```python
elif self.edge_variant == 'V40' and bool(use_edge_attr) and edge_attr is not None:
    q_state = edge_attr
    if q_state.dim() == 1:
        q_state = q_state.unsqueeze(1)
    if q_state.shape[1] >= 3 and q_state.shape[0] == edge_value.shape[0]:
        q_state = q_state / q_state.sum(dim=1, keepdim=True).clamp_min(1e-6)
        support = q_state[:, 0]
        boundary = q_state[:, 1]
        neutral = q_state[:, 2]

        confidence = 1.0 - neutral
        attr_term = support - boundary
        attr_term = (attr_term - attr_term.mean()) / attr_term.std(unbiased=False).clamp_min(1e-6)

        graph_alpha = self._graph_alpha(
            confidence, fallback_dtype=score.dtype, fallback_device=score.device
        )

        score = score + float(self.edge_fusion_gamma) * self.edge_attr_fusion_scale * graph_alpha * confidence * attr_term

        self.last_graph_alpha = float(graph_alpha.detach().cpu().item())
        self.last_reliability_mean = float(confidence.detach().mean().cpu().item())
        self.last_mix_beta = float(attr_term.detach().abs().mean().cpu().item())
```

这比继续走 `edge_attr_encoder(q)` 更好，因为：

* `support` 被明确解释成“同簇证据”
* `boundary` 被明确解释成“切边证据”
* `neutral` 提供 confidence shrinkage

它仍然沿用你们当前 assignment residual 的插口，不需要重写整个 `LorentzAssignment`。

---

## leaf message 端怎么处理

### 第一版我建议：**先关掉 `edge_msg_conditioned`**

因为 message graph 本身已经被 support posterior 重加权了。

当前 `embed_leaf()` 里，只有 `edge_msg_conditioned=True` 时才会把 edge_attr 喂进 leaf convolution。

V40 第一版若同时做：

* support-filtered message graph
* 再让 leaf gate 吃 `q_msg`

会很容易双重计数。

所以我的建议是：

* **V40A 主 preset**：`edge_msg_conditioned=false`
* **V40A-msg 次 preset**：`edge_msg_conditioned=true`，但 `edge_msg_gate_scale` 降到 `0.10 ~ 0.15`

先把“support graph + assignment residual + boundary auxiliary”这条线验证干净。

---

## auxiliary loss 我建议这样写

关键对象是 leaf 到 root 的 soft cluster membership：

```python
P_leaf = ass_dict[self.height]
for k in range(self.height - 1, 1, -1):
    P_leaf = P_leaf @ ass_dict[k]
# or equivalent compose-to-root helper
```

然后在 leaf edges 上定义

```python
same_ij = <P_i, P_j> = (P_leaf[src] * P_leaf[dst]).sum(dim=1)
```

再用 leaf role posterior：

* `q_s = q_leaf[:, 0]`
* `q_b = q_leaf[:, 1]`
* `q_n = q_leaf[:, 2]`

### 1. support loss

support edge 不应该落在不同簇：

```python
L_support =
    sum( w_ij * q_s * (1 - same_ij) )
    / sum( w_ij * q_s )
```

### 2. boundary loss

boundary edge 不应该落在同簇：

```python
L_boundary =
    sum( w_ij * q_b * same_ij )
    / sum( w_ij * q_b )
```

### 3. confidence regularizer

想让每条边的 role posterior 尽量明确，不要全是 1/3,1/3,1/3：

```python
H_edge = -sum(q * log q, dim=1)
L_conf = mean(H_edge)
```

直接最小化 `L_conf`，就是鼓励 low-entropy edge state。

### 4. balance regularizer

想避免全图塌成全 neutral 或全 support：

```python
q_bar = mean(q, dim=0)
L_balance = sum(q_bar * log(q_bar))
```

最小化它，相当于最大化 marginal entropy，避免单角色塌缩。

### 总损失

```python
L_total =
    L_si
    + λ_sup * L_support
    + λ_bnd * L_boundary
    + λ_conf * L_conf
    + λ_bal * L_balance
```

我建议初始系数：

```python
λ_sup  = 1.0
λ_bnd  = 1.0
λ_conf = 5e-3
λ_bal  = 1e-3
support_floor = 0.15
```

### 还有一个小细节

辅助 loss 要**去掉 self-loop**，因为当前 edge candidate 路径会显式给每个节点加 self-loop candidate；这些边放进 boundary/support loss 会污染统计。

所以：

```python
mask = src != dst
```

先过滤再算。

---

## `_edge_state_aux_loss()` 草案

```python
def _compose_leaf_soft_assignment(self, ass_dict: dict) -> torch.Tensor:
    running = None
    for k in range(self.height - 1, 0, -1):
        ass = ass_dict[k + 1]
        running = ass if running is None else (running @ ass)
    return running  # [N, K_root]

def _edge_state_aux_loss(self, ass_aug_dict, adj_leaf_raw, q_leaf):
    adj = adj_leaf_raw.coalesce()
    idx = adj.indices()
    val = adj.values().float()
    src, dst = idx[0].long(), idx[1].long()

    keep = src != dst
    if not torch.any(keep):
        return val.new_tensor(0.0)

    src = src[keep]
    dst = dst[keep]
    w = val[keep]
    q = q_leaf[keep]

    P = self._compose_leaf_soft_assignment(ass_aug_dict)
    same = (P[src] * P[dst]).sum(dim=1).clamp(0.0, 1.0)

    q_s = q[:, 0]
    q_b = q[:, 1]

    sup_denom = (w * q_s).sum().clamp_min(1e-6)
    bnd_denom = (w * q_b).sum().clamp_min(1e-6)

    loss_support = (w * q_s * (1.0 - same)).sum() / sup_denom
    loss_boundary = (w * q_b * same).sum() / bnd_denom

    entropy = -(q.clamp_min(1e-8) * torch.log(q.clamp_min(1e-8))).sum(dim=1).mean()
    q_bar = q.mean(dim=0)
    loss_balance = (q_bar.clamp_min(1e-8) * torch.log(q_bar.clamp_min(1e-8))).sum()

    self.last_edge_state_support_same_mean = float(
        ((w * q_s * same).sum() / sup_denom).detach().item()
    )
    self.last_edge_state_boundary_same_mean = float(
        ((w * q_b * same).sum() / bnd_denom).detach().item()
    )
    self.last_edge_state_support_mean = float(q_s.detach().mean().item())
    self.last_edge_state_boundary_mean = float(q_b.detach().mean().item())
    self.last_edge_state_neutral_mean = float(q[:, 2].detach().mean().item())
    self.last_edge_state_entropy_mean = float(entropy.detach().item())

    return (
        self.edge_state_support_lambda * loss_support
        + self.edge_state_boundary_lambda * loss_boundary
        + self.edge_state_conf_lambda * entropy
        + self.edge_state_balance_lambda * loss_balance
    )
```

---

## `_apply_state_factor_to_adj()` 很简单

```python
def _apply_state_factor_to_adj(self, target_adj, factor, normalize_for_message: bool):
    tgt = target_adj.coalesce()
    factor = factor.to(dtype=tgt.values().dtype, device=tgt.values().device)
    new_val = tgt.values() * factor
    out = torch.sparse_coo_tensor(
        tgt.indices(), new_val, size=tgt.size(), device=tgt.device
    ).coalesce()
    if normalize_for_message:
        out = self._normalize_sparse_by_degree(out)
    return out
```

其实它就是 V20/V30 那个 `_apply_learned_edge_weight_to_adj()` 的 state 版。
只不过 factor 不再来自 `edge_weight_mapper(edge_attr)`，而来自 `q_support`。

---

## `forward()` / `get_cluster_results()` 也要一起改

这个点很重要：
**不要只改 `se_loss()`。**

否则会出现：

* 训练时：用 support-filtered graph
* 推理时：仍然用原始 base message graph

V40 的 eval 路径应该也走一个简化版 prepare：

1. `adj_base_msg`
2. `z_ctx = embed_leaf(x, adj_base_msg, use_edge_attr=False)`
3. `q_msg = edge_state_predictor(z_ctx, adj_base_msg, raw_attr_aligned, matched)`
4. `adj_eval_msg = apply_state_factor_to_adj(adj_base_msg, support_floor + ... * q_s)`
5. `encoder(x, adj_eval_msg, edge_attr=q_msg, use_edge_attr=True)`

这样 train/infer 语义才一致。

---

## parser / preset 要怎么补

当前 parser 的 `edge_variant` 只到 `V33`，而且已经有 `edge_attr_hierarchical`、`edge_attr_pool_topk`、`edge_msg_conditioned` 等参数位。V40 最好沿用这些现有插口，不再新造第二套 hierarchy 配置。

### 你要加的 parser 参数

```python
parser.add_argument('--edge_state_hidden_dim', type=int, default=128)
parser.add_argument('--edge_state_temp', type=float, default=1.0)
parser.add_argument('--edge_state_support_floor', type=float, default=0.15)
parser.add_argument('--edge_state_support_lambda', type=float, default=1.0)
parser.add_argument('--edge_state_boundary_lambda', type=float, default=1.0)
parser.add_argument('--edge_state_conf_lambda', type=float, default=5e-3)
parser.add_argument('--edge_state_balance_lambda', type=float, default=1e-3)
parser.add_argument('--edge_state_detach_context', action='store_true')
```

### 我建议的最小 preset

```json
{
  "edge_variant": "V40",
  "edge_fusion_gamma": 0.7,
  "edge_attr_fusion_scale": 0.7,

  "edge_attr_hierarchical": true,
  "edge_attr_pool_topk": 1,
  "edge_attr_pool_confidence": false,

  "edge_msg_conditioned": false,

  "edge_state_hidden_dim": 128,
  "edge_state_temp": 1.0,
  "edge_state_support_floor": 0.15,
  "edge_state_support_lambda": 1.0,
  "edge_state_boundary_lambda": 1.0,
  "edge_state_conf_lambda": 0.005,
  "edge_state_balance_lambda": 0.001,
  "edge_state_detach_context": true
}
```

### 为什么第一版不用 `edge_msg_conditioned`

因为当前 B45 已经证明 leaf gate 是有效入口，但 V40 第一版更重要的是证明：

> “multi-role state” 本身是不是比 scalar reparameterization 更稳。

B45 现在主 preset 本来就是 `V31 + edge_msg_conditioned + gate_scale=0.50`；V40 第一版正好和它形成干净对照。

---

## 我建议你们先跑的 4 个对照

### 1. `V40` vs `B45`

主问题：multi-role state 是否优于 current scalar route。

### 2. `V40_context_off`

predictor 只吃 raw edge_attr + matched，不吃 `z_i,z_j`。
这个对照最关键。
如果机制数据差不多、真实数据明显退步，就说明真实图上的难点确实来自**上下文相关的边语义**。

### 3. `V40_support_only`

去掉 boundary auxiliary，只保留 support-filtered graph。
这能测“多角色”到底值不值。

### 4. `V40_no_hier`

leaf 用 q，但层间不池化 q。
这能直接测试你们现成的 `edge_attr_hierarchical` 在 V40 下是不是仍然有贡献。当前主干已经有这一套 pooling 机制，所以这个对照很便宜。

---

## 我最推荐先监控的 diagnostics

当前 exp 里已经有 factor/msg/assign/hier/aug 一套 liveness 统计思路。V40 最好照着补，而不是只看 NMI/ARI。

我会新增这些：

```python
edge_state_support_mean
edge_state_boundary_mean
edge_state_neutral_mean
edge_state_entropy_mean
edge_state_support_same_mean
edge_state_boundary_same_mean
edge_state_confidence_mean = 1 - neutral_mean
```

其中最有解释力的是：

* `boundary_same_mean`：应该低
* `support_same_mean`：应该高
* `neutral_mean`：不能全图塌太高
* `entropy_mean`：不能全图塌成极低但单一角色

---

## 为什么这版 V40 是“最值得先做”的

因为它刚好踩在当前代码已经有的三个接口上：

1. **leaf graph preparation 在 DSI 里**
   所以 support-filtered `adj_train_msg/adj_train_si` 很好接。

2. **encoder 已经支持 vector edge_attr + hierarchy pooling**
   所以 `q_ij` 可以无缝当新的 edge_attr。

3. **assignment 已经有 edge-aware residual 插口**
   所以 `support - boundary` 可以直接进 score。

换句话说，V40 不是另起炉灶，而是把你们现有三条入口从“scalar helpfulness”升级成“role posterior”。

---

## 最后给一句执行顺序

真正落地时，我建议按这个顺序 patch：

1. `main.py` / preset：先加 `V40` 与新 flags
2. `DSI.__init__()`：拆开 `raw_edge_attr_dim` 和 `encoder_edge_attr_dim`
3. `EdgeStatePredictor`
4. `DSI.se_loss()` 的 V40 路径
5. `LorentzAssignment.forward()` 的 `V40` 分支
6. `forward()/get_cluster_results()` 的 eval 路径一致化
7. diagnostics

这条顺序最不容易改乱。

下一步最合适的，就是直接把上面这些改成**接近你们当前 `CURRENT_MAINLINE_CORE_MERGED.py` 风格的具体 patch 位点**。
