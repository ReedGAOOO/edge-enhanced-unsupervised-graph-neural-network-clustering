import torch
import torch.nn as nn
from utils.model_utils import gumbel_softmax, graph_top_K
from manifold.lorentz import Lorentz
from modules.layers import LorentzBoost
from modules.model import LSENet
from torch_geometric.utils import softmax as scatter_softmax

MIN_NORM = 1e-15
EPS = 1e-6


class DSI(nn.Module):
    def __init__(self, in_dim, hid_dim, num_nodes, max_nums, temperature=0.2,
                 dropout=0.5, nonlin_str='relu', tau=1.0, alpha=0.01, knn=8,
                 edge_variant='V1', edge_fusion_gamma=1.0, edge_confidence_quantile=0.0,
                 edge_adaptive_alpha=False, edge_adaptive_alpha_strength=2.0,
                 edge_adaptive_alpha_bias=0.0, edge_reliability_temp=1.0,
                 edge_attr_hidden_dim=64, edge_attr_fusion_scale=1.0,
                 edge_relation_channels=4, edge_relation_hidden_dim=64, edge_relation_assign_scale=1.0,
                 edge_relation_reg_lambda=1e-3,
                 edge_relation_use_assignment=True,
                 edge_relation_use_si_hierarchy=True,
                 edge_relation_use_msg_adj=True,
                 edge_attr_dim=1,
                 edge_attr_hierarchical=False,
                 edge_attr_pool_topk=1,
                 edge_msg_conditioned=False,
                 edge_msg_gate_scale=0.35,
                 edge_msg_matched_only=False,
                 edge_msg_confidence_gate=False,
                 edge_msg_confidence_temp=1.0,
                 edge_attr_pool_confidence=False,
                 edge_attr_pool_conf_power=1.0,
                 edge_weight_learn_reg_lambda=0.02,
                 edge_weight_learn_logclip=0.8,
                 edge_weight_learn_temp=1.0,
                 edge_weight_learn_apply_to='both',
                 edge_aug_prior_scale=0.0,
                 edge_aug_prior_mode='raw',
                 knn_mode='auto', knn_auto_threshold=20000):
        super(DSI, self).__init__()
        self.num_nodes = num_nodes
        self.height = len(max_nums) + 1
        self.manifold = Lorentz()
        self.encoder = LSENet(
            self.manifold,
            in_dim,
            hid_dim,
            max_nums,
            temperature,
            dropout,
            nonlin_str,
            edge_variant=edge_variant,
            edge_fusion_gamma=edge_fusion_gamma,
            edge_confidence_quantile=edge_confidence_quantile,
            edge_adaptive_alpha=edge_adaptive_alpha,
            edge_adaptive_alpha_strength=edge_adaptive_alpha_strength,
            edge_adaptive_alpha_bias=edge_adaptive_alpha_bias,
            edge_reliability_temp=edge_reliability_temp,
            edge_attr_hidden_dim=edge_attr_hidden_dim,
            edge_attr_fusion_scale=edge_attr_fusion_scale,
            edge_relation_channels=edge_relation_channels,
            edge_relation_hidden_dim=edge_relation_hidden_dim,
            edge_relation_assign_scale=(edge_relation_assign_scale if bool(edge_relation_use_assignment) else 0.0),
            edge_attr_dim=edge_attr_dim,
            edge_attr_hierarchical=bool(edge_attr_hierarchical),
            edge_attr_pool_topk=int(edge_attr_pool_topk),
            edge_msg_conditioned=bool(edge_msg_conditioned),
            edge_msg_gate_scale=float(edge_msg_gate_scale),
            edge_msg_matched_only=bool(edge_msg_matched_only),
            edge_msg_confidence_gate=bool(edge_msg_confidence_gate),
            edge_msg_confidence_temp=float(edge_msg_confidence_temp),
            edge_attr_pool_confidence=bool(edge_attr_pool_confidence),
            edge_attr_pool_conf_power=float(edge_attr_pool_conf_power),
        )
        self.lorentz_proj = LorentzBoost(hid_dim + 1)
        self.temperature = temperature
        self.tau = tau
        self.alpha = alpha
        self.knn = knn
        self.edge_variant = str(edge_variant).upper()
        self.knn_mode = str(knn_mode).lower()
        self.knn_auto_threshold = int(knn_auto_threshold)
        self.edge_relation_use_assignment = bool(edge_relation_use_assignment)
        self.edge_relation_use_si_hierarchy = bool(edge_relation_use_si_hierarchy)
        self.edge_relation_use_msg_adj = bool(edge_relation_use_msg_adj)
        self.edge_weight_learn_reg_lambda = float(max(0.0, edge_weight_learn_reg_lambda))
        self.edge_weight_learn_logclip = float(max(1e-3, edge_weight_learn_logclip))
        self.edge_weight_learn_temp = float(max(1e-3, edge_weight_learn_temp))
        self.edge_weight_learn_apply_to = str(edge_weight_learn_apply_to).lower()
        if self.edge_weight_learn_apply_to not in {'si_only', 'both'}:
            self.edge_weight_learn_apply_to = 'both'
        self.edge_aug_prior_scale = float(edge_aug_prior_scale)
        self.edge_relation_reg_lambda = float(max(0.0, edge_relation_reg_lambda))
        self.edge_aug_prior_mode = str(edge_aug_prior_mode).lower()
        if self.edge_aug_prior_mode not in {'raw', 'positive', 'tanh'}:
            self.edge_aug_prior_mode = 'raw'

        self.last_edge_factor_mean = 1.0
        self.last_edge_factor_std = 0.0
        self.last_edge_factor_msg_mean = 1.0
        self.last_edge_factor_msg_std = 0.0
        self.last_edge_factor_si_mean = 1.0
        self.last_edge_factor_si_std = 0.0
        self.last_edge_aug_bias_mean = 0.0
        self.last_edge_aug_bias_std = 0.0
        self.last_edge_reg = 0.0

        if self._use_learnable_edge_weight_variant():
            hidden = max(8, int(edge_attr_hidden_dim))
            out_dim = 1
            if self._use_dual_edge_weight_variant():
                out_dim = 3 if self._use_edge_aug_prior_variant() else 2
            self.edge_weight_mapper = nn.Sequential(
                nn.Linear(int(max(1, edge_attr_dim)), hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim),
            )
        else:
            self.edge_weight_mapper = None

    def set_edge_fusion_gamma(self, gamma: float):
        if hasattr(self.encoder, "set_edge_fusion_gamma"):
            self.encoder.set_edge_fusion_gamma(gamma)

    def get_edge_adaptive_stats(self):
        stats = {"graph_alpha_mean": 1.0, "edge_reliability_mean": 1.0, "edge_mix_beta_mean": 0.0}
        if hasattr(self.encoder, "get_edge_adaptive_stats"):
            stats = self.encoder.get_edge_adaptive_stats()
        stats["edge_factor_mean"] = float(self.last_edge_factor_mean)
        stats["edge_factor_std"] = float(self.last_edge_factor_std)
        stats["edge_factor_msg_mean"] = float(self.last_edge_factor_msg_mean)
        stats["edge_factor_msg_std"] = float(self.last_edge_factor_msg_std)
        stats["edge_factor_si_mean"] = float(self.last_edge_factor_si_mean)
        stats["edge_factor_si_std"] = float(self.last_edge_factor_si_std)
        stats["edge_aug_bias_mean"] = float(self.last_edge_aug_bias_mean)
        stats["edge_aug_bias_std"] = float(self.last_edge_aug_bias_std)
        stats["edge_reg"] = float(self.last_edge_reg)
        return stats

    def _encode_relation_state_for_adj(self, base_adj, base_edge_attr, target_adj):
        if (not self._use_edge_relation_variant()) or base_edge_attr is None:
            return None, None
        edge_attr, edge_mask = self._align_edge_attr_to_adj_with_mask(base_adj, base_edge_attr, target_adj)
        if edge_attr is None:
            return None, edge_mask
        edge_state = self.encoder.encode_edge_relation_state(edge_attr, matched=edge_mask)
        return edge_state, edge_mask

    def forward(self, data):
        features = data.x
        adj = getattr(data, "adj_msg", data.adj).clone()
        if self._use_learnable_edge_weight_variant():
            head = 'msg' if self._use_dual_edge_weight_variant() else 'shared'
            adj, _ = self._apply_learned_edge_weight_to_adj(
                base_adj=getattr(data, "adj_msg", data.adj),
                target_adj=adj,
                base_edge_attr=getattr(data, "edge_attr", None),
                normalize_for_message=True,
                head=head,
            )
        use_edge_attr = self._use_edge_attr_variant()
        edge_attr = getattr(data, "edge_attr", None) if use_edge_attr else None
        edge_mask = None
        if use_edge_attr and self._use_edge_relation_variant():
            edge_attr, edge_mask = self._encode_relation_state_for_adj(getattr(data, "adj_msg", data.adj), edge_attr, adj)
            if self.edge_relation_use_msg_adj:
                adj = self._apply_relation_state_to_adj(adj, edge_attr, normalize_for_message=True, head='msg')
        tree_coord_dict, ass_dict, adj_dict = self.encoder(features, adj, edge_attr=edge_attr, edge_mask=edge_mask, use_edge_attr=use_edge_attr)
        return tree_coord_dict, ass_dict, adj_dict

    def get_cluster_results(self, data):
        features = data.x
        adj = getattr(data, "adj_msg", data.adj).clone()
        if self._use_learnable_edge_weight_variant():
            head = 'msg' if self._use_dual_edge_weight_variant() else 'shared'
            adj, _ = self._apply_learned_edge_weight_to_adj(
                base_adj=getattr(data, "adj_msg", data.adj),
                target_adj=adj,
                base_edge_attr=getattr(data, "edge_attr", None),
                normalize_for_message=True,
                head=head,
            )
        use_edge_attr = self._use_edge_attr_variant()
        edge_attr = getattr(data, "edge_attr", None) if use_edge_attr else None
        edge_mask = None
        if use_edge_attr and self._use_edge_relation_variant():
            edge_attr, edge_mask = self._encode_relation_state_for_adj(getattr(data, "adj_msg", data.adj), edge_attr, adj)
            if self.edge_relation_use_msg_adj:
                adj = self._apply_relation_state_to_adj(adj, edge_attr, normalize_for_message=True, head='msg')
        coord_dict, ass_dict, _ = self.encoder(features, adj, edge_attr=edge_attr, edge_mask=edge_mask, use_edge_attr=use_edge_attr)
        embed_dict = {}
        for height, x in coord_dict.items():
            embed_dict[height] = x.detach()
        # Avoid constructing an [N, N] identity for large graphs.
        clu_mat_dict = {}
        running = None
        for k in range(self.height - 1, 0, -1):
            ass = ass_dict[k + 1]
            running = ass if running is None else (running @ ass)
            idx = running.max(1)[1]
            t = torch.zeros_like(running)
            t[torch.arange(t.shape[0], device=t.device), idx] = 1.
            clu_mat_dict[k] = t
        return embed_dict, clu_mat_dict

    def fix_cluster_results(self, clu_res_mat, embed_dict, epsInt: int = 7):
        clu_nums = clu_res_mat.sum(0)
        clu_res = clu_res_mat.argmax(1)
        corr_idx = clu_nums > epsInt
        if torch.all(corr_idx):
            return clu_res
        idx = torch.arange(clu_res_mat.shape[1]).to(clu_res.device)
        idx = idx[corr_idx]
        err_idx = torch.where(clu_res_mat[:, clu_nums <= epsInt] == 1.)[0]
        node = embed_dict[self.height]
        parent = embed_dict[1]
        error_node = node[err_idx]
        fixed_parent = parent[corr_idx]
        score = torch.log_softmax(2 + 2 * self.manifold.cinner(error_node, fixed_parent), dim=-1)
        fixed_res = gumbel_softmax(score, self.temperature)
        fixed_res = idx[fixed_res.argmax(1)]
        clu_res[err_idx] = fixed_res
        return clu_res

    def se_loss(self, data, eps=1e-6):
        self.last_edge_reg = 0.0
        adj_base_msg = getattr(data, "adj_msg", data.adj).clone()
        adj_base_si = getattr(data, "adj_si", adj_base_msg).clone()
        base_edge_attr = getattr(data, "edge_attr", None)
        aug_prior_bias = None
        if self._use_edge_aug_prior_variant():
            aug_prior_bias = self._edge_prior_bias_for_base_adj(adj_base_msg, base_edge_attr)

        if self._use_edge_knn_mode(data.x.shape[0]):
            # For large-graph edge mode, stop gradient through adjacency construction.
            with torch.no_grad():
                leaf_edge_attr = base_edge_attr if self._use_edge_attr_variant() else None
                leaf_edge_mask = None
                if self._use_edge_relation_variant():
                    leaf_edge_attr, leaf_edge_mask = self._encode_relation_state_for_adj(adj_base_msg, base_edge_attr, adj_base_msg)
                z_leaf = self.encoder.embed_leaf(
                    data.x,
                    adj_base_msg,
                    edge_attr=leaf_edge_attr,
                    edge_mask=leaf_edge_mask,
                    use_edge_attr=self._use_edge_attr_variant(),
                )
                z_leaf = self.lorentz_proj(z_leaf)
                adj_aug = self._edge_candidate_adj(z_leaf, adj_base_msg, self.knn, edge_bias=aug_prior_bias)
        else:
            leaf_edge_attr = base_edge_attr if self._use_edge_attr_variant() else None
            leaf_edge_mask = None
            if self._use_edge_relation_variant():
                leaf_edge_attr, leaf_edge_mask = self._encode_relation_state_for_adj(adj_base_msg, base_edge_attr, adj_base_msg)
            z_leaf = self.encoder.embed_leaf(
                data.x,
                adj_base_msg,
                edge_attr=leaf_edge_attr,
                edge_mask=leaf_edge_mask,
                use_edge_attr=self._use_edge_attr_variant(),
            )
            z_leaf = self.lorentz_proj(z_leaf)
            neg_dist2 = 2 + 2 * self.manifold.cinner(z_leaf, z_leaf)
            score_dense = neg_dist2 / self.tau
            if aug_prior_bias is not None and float(self.edge_aug_prior_scale) > 0.0:
                prior_dense = torch.sparse_coo_tensor(
                    adj_base_msg.coalesce().indices(),
                    aug_prior_bias.to(dtype=score_dense.dtype, device=score_dense.device),
                    size=adj_base_msg.size(),
                    device=score_dense.device,
                ).to_dense()
                score_dense = score_dense + float(self.edge_aug_prior_scale) * prior_dense
            adj_aug = graph_top_K(torch.softmax(score_dense, dim=-1), k=self.knn)

        adj_train_msg = (self.alpha * adj_aug + adj_base_msg).coalesce()
        adj_train_si = (self.alpha * adj_aug + adj_base_si).coalesce()

        edge_reg_raw = adj_train_msg.values().new_tensor(0.0)
        if self._use_learnable_edge_weight_variant():
            if self._use_dual_edge_weight_variant():
                adj_train_si, reg_si = self._apply_learned_edge_weight_to_adj(
                    base_adj=adj_base_si,
                    target_adj=adj_train_si,
                    base_edge_attr=base_edge_attr,
                    normalize_for_message=False,
                    head='si',
                )
                adj_train_msg, reg_msg = self._apply_learned_edge_weight_to_adj(
                    base_adj=adj_base_msg,
                    target_adj=adj_train_msg,
                    base_edge_attr=base_edge_attr,
                    normalize_for_message=True,
                    head='msg',
                )
                edge_reg_raw = 0.5 * (reg_si + reg_msg)
            else:
                adj_train_si, reg_si = self._apply_learned_edge_weight_to_adj(
                    base_adj=adj_base_si,
                    target_adj=adj_train_si,
                    base_edge_attr=base_edge_attr,
                    normalize_for_message=False,
                    head='shared',
                )
                edge_reg_raw = reg_si
                if self.edge_weight_learn_apply_to == 'both':
                    adj_train_msg, reg_msg = self._apply_learned_edge_weight_to_adj(
                        base_adj=adj_base_msg,
                        target_adj=adj_train_msg,
                        base_edge_attr=base_edge_attr,
                        normalize_for_message=True,
                        head='shared',
                    )
                    edge_reg_raw = 0.5 * (reg_si + reg_msg)

        use_edge_attr = self._use_edge_attr_variant()
        edge_attr = None
        edge_mask = None
        edge_state_si = None
        edge_mask_si = None
        if use_edge_attr:
            edge_attr_base = getattr(data, "edge_attr", None)
            if self._use_edge_relation_variant():
                edge_attr, edge_mask = self._encode_relation_state_for_adj(adj_base_msg, edge_attr_base, adj_train_msg)
                if self.edge_relation_use_msg_adj:
                    adj_train_msg = self._apply_relation_state_to_adj(adj_train_msg, edge_attr, normalize_for_message=True, head='msg')
                if self.edge_relation_use_si_hierarchy:
                    edge_state_si, edge_mask_si = self._encode_relation_state_for_adj(adj_base_si, edge_attr_base, adj_train_si)
            else:
                edge_attr, edge_mask = self._align_edge_attr_to_adj_with_mask(adj_base_msg, edge_attr_base, adj_train_msg)
        _, ass_aug_dict, _ = self.encoder(
            data.x, adj_train_msg, edge_attr=edge_attr, edge_mask=edge_mask, use_edge_attr=use_edge_attr
        )
        if self._use_edge_relation_variant() and self.edge_relation_use_si_hierarchy and edge_state_si is not None:
            adj_si_dict = self._build_relation_hierarchy_adj_from_assign(adj_train_si, edge_state_si, ass_aug_dict)
        else:
            adj_si_dict = self._build_hierarchy_adj_from_assign(adj_train_si, ass_aug_dict)
        loss = self._si_loss(ass_aug_dict, adj_si_dict, eps)
        if self._use_learnable_edge_weight_variant() and self.edge_weight_learn_reg_lambda > 0.0:
            edge_reg = self.edge_weight_learn_reg_lambda * edge_reg_raw
            self.last_edge_reg = float(edge_reg.detach().item())
            loss = loss + edge_reg
        if self._use_edge_relation_variant() and self.edge_relation_reg_lambda > 0.0 and edge_attr is not None:
            rel_reg = self.edge_relation_reg_lambda * self.encoder.edge_relation_regularization(edge_attr, matched=edge_mask)
            self.last_edge_reg = float(self.last_edge_reg + rel_reg.detach().item())
            loss = loss + rel_reg
        return loss

    def _use_edge_attr_variant(self) -> bool:
        return self.edge_variant in {'V6', 'V7', 'V8', 'V12', 'V13', 'V31', 'V32', 'V33', 'V40'}

    def _use_learnable_edge_weight_variant(self) -> bool:
        return self.edge_variant in {'V20', 'V30', 'V31', 'V32', 'V33'}

    def _use_dual_edge_weight_variant(self) -> bool:
        return self.edge_variant in {'V30', 'V31', 'V32', 'V33'}

    def _use_edge_aug_prior_variant(self) -> bool:
        return self.edge_variant in {'V33'}

    def _use_edge_relation_variant(self) -> bool:
        return self.edge_variant in {'V40'}

    def _update_relation_factor_stats(self, factor: torch.Tensor | None, head: str) -> None:
        if factor is None or factor.numel() == 0:
            return
        mean = float(factor.detach().mean().item())
        std = float(factor.detach().std(unbiased=False).item())
        if str(head).lower() == 'msg':
            self.last_edge_factor_msg_mean = mean
            self.last_edge_factor_msg_std = std
        elif str(head).lower() == 'si':
            self.last_edge_factor_si_mean = mean
            self.last_edge_factor_si_std = std
        vals = [self.last_edge_factor_msg_mean, self.last_edge_factor_si_mean]
        self.last_edge_factor_mean = float(sum(vals) / len(vals))
        vals_std = [self.last_edge_factor_msg_std, self.last_edge_factor_si_std]
        self.last_edge_factor_std = float(sum(vals_std) / len(vals_std))

    def _apply_relation_state_to_adj(self, target_adj: torch.Tensor, edge_state: torch.Tensor | None, normalize_for_message: bool, head: str = 'msg') -> torch.Tensor:
        tgt = target_adj.coalesce() if target_adj.is_sparse else target_adj.to_sparse().coalesce()
        if (not self._use_edge_relation_variant()) or edge_state is None or edge_state.numel() == 0:
            return self._normalize_sparse_by_degree(tgt) if normalize_for_message else tgt
        factor = self.encoder.edge_relation_factor_from_state(
            edge_state.to(dtype=tgt.values().dtype, device=tgt.values().device),
            head=head,
        )
        if factor is None or factor.numel() == 0:
            return self._normalize_sparse_by_degree(tgt) if normalize_for_message else tgt
        self._update_relation_factor_stats(factor, head=head)
        new_val = tgt.values() * factor.to(dtype=tgt.values().dtype, device=tgt.values().device)
        out = torch.sparse_coo_tensor(tgt.indices(), new_val, size=tgt.size(), device=tgt.device).coalesce()
        if normalize_for_message:
            out = self._normalize_sparse_by_degree(out)
        return out

    @staticmethod
    def _ass_adj_ass(ass: torch.Tensor, adj: torch.Tensor):
        if adj.is_sparse and torch.is_autocast_enabled():
            out_dtype = ass.dtype
            with torch.autocast(device_type="cuda", enabled=False):
                out = ass.float().t() @ adj.float() @ ass.float()
            return out.to(dtype=out_dtype)
        return ass.t() @ adj @ ass

    def _build_hierarchy_adj_from_assign(self, adj_leaf: torch.Tensor, ass_dict: dict) -> dict:
        adj_dict = {self.height: adj_leaf}
        curr_adj = adj_leaf
        for k in range(self.height, 1, -1):
            ass = ass_dict[k]
            next_adj = self._ass_adj_ass(ass, curr_adj)
            if isinstance(next_adj, torch.Tensor) and next_adj.is_sparse:
                next_adj = next_adj.coalesce()
            adj_dict[k - 1] = next_adj
            curr_adj = next_adj
        return adj_dict

    def _build_relation_hierarchy_adj_from_assign(self, adj_leaf: torch.Tensor, edge_state_leaf: torch.Tensor | None, ass_dict: dict) -> dict:
        if (not self._use_edge_relation_variant()) or edge_state_leaf is None or edge_state_leaf.numel() == 0:
            return self._build_hierarchy_adj_from_assign(adj_leaf, ass_dict)
        adj_raw = adj_leaf.coalesce() if adj_leaf.is_sparse else adj_leaf.to_sparse().coalesce()
        curr_state = edge_state_leaf
        adj_dict = {self.height: self._apply_relation_state_to_adj(adj_raw, curr_state, normalize_for_message=False, head='si')}
        for k in range(self.height, 1, -1):
            ass = ass_dict[k]
            next_adj_raw = self._ass_adj_ass(ass, adj_raw)
            if isinstance(next_adj_raw, torch.Tensor) and next_adj_raw.is_sparse:
                next_adj_raw = next_adj_raw.coalesce()
            curr_state = self.encoder._coarsen_edge_attr_soft_topk(
                adj_raw,
                curr_state,
                ass,
                next_adj_raw,
                topk=getattr(self.encoder, 'edge_attr_pool_topk', 1),
                use_confidence=getattr(self.encoder, 'edge_attr_pool_confidence', False),
                conf_power=getattr(self.encoder, 'edge_attr_pool_conf_power', 1.0),
            )
            adj_dict[k - 1] = self._apply_relation_state_to_adj(next_adj_raw, curr_state, normalize_for_message=False, head='si')
            adj_raw = next_adj_raw
        return adj_dict

    def _use_edge_knn_mode(self, num_nodes: int) -> bool:
        mode = self.knn_mode
        if mode == 'edge':
            return True
        if mode == 'dense':
            return False
        return int(num_nodes) > int(self.knn_auto_threshold)

    def _edge_candidate_adj(self, z_leaf, adj, k: int, edge_bias: torch.Tensor | None = None):
        adj_coo = adj.coalesce()
        edge_index = adj_coo.indices()
        num_nodes = z_leaf.shape[0]

        # Ensure every node has at least one candidate edge.
        self_idx = torch.arange(num_nodes, device=z_leaf.device)
        edge_index = torch.cat([edge_index, torch.stack([self_idx, self_idx], dim=0)], dim=1)
        src, dst = edge_index[0], edge_index[1]

        # Edge-wise score; do not materialize an [E, E] matrix.
        score = 2 + 2 * self.manifold.inner(z_leaf[src], z_leaf[dst], keepdim=False)
        score = score / float(self.tau)
        if edge_bias is not None and edge_bias.numel() > 0 and float(self.edge_aug_prior_scale) > 0.0:
            edge_bias = edge_bias.to(dtype=score.dtype, device=score.device)
            if edge_bias.shape[0] == adj_coo.values().shape[0]:
                edge_bias = torch.cat([edge_bias, torch.zeros(num_nodes, dtype=edge_bias.dtype, device=edge_bias.device)], dim=0)
            score = score + float(self.edge_aug_prior_scale) * edge_bias
        edge_index, score = self._topk_edges_per_src(edge_index, score, num_nodes=num_nodes, k=int(k))
        src = edge_index[0]
        prob = scatter_softmax(score, src, dim=0)
        adj_aug = torch.sparse_coo_tensor(edge_index, prob, size=(num_nodes, num_nodes), device=z_leaf.device).coalesce()
        return adj_aug

    @staticmethod
    def _topk_edges_per_src(edge_index, score, num_nodes: int, k: int):
        if k <= 0:
            return edge_index, score
        src = edge_index[0]
        dst = edge_index[1]
        # Sorting by source lets us run a linear pass over edge segments.
        order = torch.argsort(src)
        src_s = src[order]
        dst_s = dst[order]
        score_s = score[order]
        counts = torch.bincount(src_s, minlength=num_nodes).cpu().tolist()

        keep_parts = []
        st = 0
        for c in counts:
            ed = st + int(c)
            if ed <= st:
                st = ed
                continue
            seg_len = ed - st
            if seg_len <= k:
                keep_parts.append(torch.arange(st, ed, device=score.device))
            else:
                top_local = torch.topk(score_s[st:ed], k=k, sorted=False).indices + st
                keep_parts.append(top_local)
            st = ed

        if len(keep_parts) == 0:
            return edge_index[:, :0], score[:0]
        keep = torch.cat(keep_parts, dim=0)
        edge_new = torch.stack([src_s[keep], dst_s[keep]], dim=0)
        score_new = score_s[keep]
        return edge_new, score_new

    @staticmethod
    def _align_edge_attr_to_adj(base_adj, base_edge_attr, target_adj):
        target_attr, _ = DSI._align_edge_attr_to_adj_with_mask(base_adj, base_edge_attr, target_adj)
        return target_attr

    @staticmethod
    def _align_edge_attr_to_adj_with_mask(base_adj, base_edge_attr, target_adj):
        if base_edge_attr is None:
            return None, None
        if base_edge_attr.numel() == 0:
            return base_edge_attr, None
        base = base_adj.coalesce()
        target = target_adj.coalesce()
        base_idx = base.indices()
        target_idx = target.indices()
        num_nodes = int(base.size(0))
        base_attr = base_edge_attr.float()
        if base_attr.shape[0] != base_idx.shape[1]:
            return None, None

        base_key = base_idx[0].long() * num_nodes + base_idx[1].long()
        target_key = target_idx[0].long() * num_nodes + target_idx[1].long()

        sorted_key, order = torch.sort(base_key)
        pos = torch.searchsorted(sorted_key, target_key)
        valid = pos < sorted_key.shape[0]
        matched = torch.zeros_like(valid, dtype=torch.bool)
        matched[valid] = sorted_key[pos[valid]] == target_key[valid]

        target_attr = torch.zeros(
            (target_idx.shape[1], base_attr.shape[1]),
            dtype=base_attr.dtype,
            device=base_attr.device,
        )
        if matched.any():
            mapped = order[pos[matched]]
            target_attr[matched] = base_attr[mapped]
        return target_attr, matched

    def _edge_scores_and_factors_from_attr(self, edge_attr: torch.Tensor, matched: torch.Tensor | None):
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)
        edge_attr = edge_attr.float()
        if self.edge_weight_mapper is None:
            ones = torch.ones((edge_attr.shape[0], 1), dtype=edge_attr.dtype, device=edge_attr.device)
            zeros = torch.zeros_like(ones)
            zero = torch.zeros((), dtype=edge_attr.dtype, device=edge_attr.device)
            return zeros, ones, zero

        score = self.edge_weight_mapper(edge_attr)
        score = score / float(self.edge_weight_learn_temp)
        score = torch.tanh(score) * float(self.edge_weight_learn_logclip)

        if matched is not None and matched.any():
            score = score - score[matched].mean(dim=0, keepdim=True)
        else:
            score = score - score.mean(dim=0, keepdim=True)

        score = score.clamp(-float(self.edge_weight_learn_logclip), float(self.edge_weight_learn_logclip))
        factor = torch.exp(score)
        if matched is not None:
            factor = torch.where(matched.unsqueeze(1), factor, torch.ones_like(factor))
            eff = matched
        else:
            eff = torch.ones(score.shape[0], dtype=torch.bool, device=score.device)

        if eff.any():
            reg_anchor = torch.mean(score[eff] ** 2)
            reg_scale = torch.mean((factor[eff] - 1.0) ** 2)
            reg = reg_anchor + 0.1 * reg_scale
            factor_eff = factor[eff]
            self.last_edge_factor_mean = float(factor_eff.detach().mean().item())
            self.last_edge_factor_std = float(factor_eff.detach().std(unbiased=False).item())
            self.last_edge_factor_msg_mean = float(factor_eff[:, 0].detach().mean().item())
            self.last_edge_factor_msg_std = float(factor_eff[:, 0].detach().std(unbiased=False).item())
            if factor_eff.shape[1] >= 2:
                self.last_edge_factor_si_mean = float(factor_eff[:, 1].detach().mean().item())
                self.last_edge_factor_si_std = float(factor_eff[:, 1].detach().std(unbiased=False).item())
            else:
                self.last_edge_factor_si_mean = self.last_edge_factor_msg_mean
                self.last_edge_factor_si_std = self.last_edge_factor_msg_std
            if score.shape[1] >= 3:
                self.last_edge_aug_bias_mean = float(score[eff, 2].detach().mean().item())
                self.last_edge_aug_bias_std = float(score[eff, 2].detach().std(unbiased=False).item())
            else:
                self.last_edge_aug_bias_mean = 0.0
                self.last_edge_aug_bias_std = 0.0
        else:
            reg = factor.new_tensor(0.0)
            self.last_edge_factor_mean = 1.0
            self.last_edge_factor_std = 0.0
            self.last_edge_factor_msg_mean = 1.0
            self.last_edge_factor_msg_std = 0.0
            self.last_edge_factor_si_mean = 1.0
            self.last_edge_factor_si_std = 0.0
            self.last_edge_aug_bias_mean = 0.0
            self.last_edge_aug_bias_std = 0.0
        return score, factor, reg

    @staticmethod
    def _edge_head_index(head: str) -> int:
        if head == 'si':
            return 1
        if head == 'aug':
            return 2
        return 0

    def _edge_prior_bias_for_base_adj(self, base_adj, base_edge_attr):
        if (not self._use_edge_aug_prior_variant()) or self.edge_weight_mapper is None:
            return None
        if base_edge_attr is None or base_edge_attr.numel() == 0:
            return None
        base = base_adj.coalesce()
        edge_attr = base_edge_attr.float().to(base.values().device)
        if edge_attr.shape[0] != base.indices().shape[1]:
            return None
        raw_score, _, _ = self._edge_scores_and_factors_from_attr(edge_attr, matched=None)
        if raw_score.shape[1] <= 2:
            return None
        bias = raw_score[:, 2]
        if self.edge_aug_prior_mode == 'positive':
            bias = torch.relu(bias)
        elif self.edge_aug_prior_mode == 'tanh':
            bias = torch.tanh(bias)
        return bias

    def _apply_learned_edge_weight_to_adj(self, base_adj, target_adj, base_edge_attr, normalize_for_message: bool, head: str = 'shared'):
        edge_attr, matched = self._align_edge_attr_to_adj_with_mask(base_adj, base_edge_attr, target_adj)
        if edge_attr is None or edge_attr.numel() == 0:
            self.last_edge_factor_mean = 1.0
            self.last_edge_factor_std = 0.0
            self.last_edge_factor_msg_mean = 1.0
            self.last_edge_factor_msg_std = 0.0
            self.last_edge_factor_si_mean = 1.0
            self.last_edge_factor_si_std = 0.0
            self.last_edge_aug_bias_mean = 0.0
            self.last_edge_aug_bias_std = 0.0
            return target_adj.coalesce(), target_adj.values().new_tensor(0.0)

        tgt = target_adj.coalesce()
        matched_dev = matched.to(tgt.values().device) if matched is not None else None
        _, factor_all, reg = self._edge_scores_and_factors_from_attr(edge_attr.to(tgt.values().device), matched_dev)
        head_idx = 0 if head == 'shared' else self._edge_head_index(head)
        head_idx = min(int(head_idx), int(factor_all.shape[1] - 1))
        factor = factor_all[:, head_idx]
        new_val = tgt.values() * factor.to(dtype=tgt.values().dtype, device=tgt.values().device)
        out = torch.sparse_coo_tensor(tgt.indices(), new_val, size=tgt.size(), device=tgt.device).coalesce()
        if normalize_for_message:
            out = self._normalize_sparse_by_degree(out)
        return out, reg

    @staticmethod
    def _sparse_degree_and_diag(adj_k):
        if adj_k.is_sparse:
            coo = adj_k.coalesce()
            idx = coo.indices()
            val = coo.values()
            degree = torch.sparse.sum(coo, dim=1).to_dense()
            diag = torch.zeros(coo.size(0), dtype=val.dtype, device=val.device)
            mask = idx[0] == idx[1]
            if mask.any():
                diag.index_add_(0, idx[0][mask], val[mask])
            return degree, diag
        degree = adj_k.sum(dim=1)
        diag = torch.diagonal(adj_k, dim1=0, dim2=1)
        return degree, diag

    @staticmethod
    def _normalize_sparse_by_degree(adj_k):
        coo = adj_k.coalesce()
        idx = coo.indices()
        val = coo.values()
        row = idx[0].long()
        col = idx[1].long()
        num_nodes = int(coo.size(0))
        deg = torch.zeros(num_nodes, dtype=val.dtype, device=val.device)
        deg.index_add_(0, row, val)
        deg_inv_sqrt = deg.clamp_min(1e-12).pow(-0.5)
        norm_val = deg_inv_sqrt[row] * val * deg_inv_sqrt[col]
        return torch.sparse_coo_tensor(idx, norm_val, size=coo.size(), device=coo.device).coalesce()

    def _si_loss(self, ass_dict: dict, adj_dict: dict, eps: float = 1e-6):
        se_loss = 0
        vol_G = adj_dict[self.height].sum()

        for k in range(self.height, 0, -1):
            degree, diag = self._sparse_degree_and_diag(adj_dict[k])
            if k == 1:
                vol_parent = vol_G
            else:
                if adj_dict[k - 1].is_sparse:
                    vol_parent = torch.sparse.sum(adj_dict[k - 1], dim=1).to_dense()
                else:
                    vol_parent = adj_dict[k - 1].sum(dim=-1)
                vol_parent = torch.einsum('ij, j->i', ass_dict[k], vol_parent)
            delta_vol = degree - diag
            log_vol_ratio_k = torch.log2((degree + eps) / (vol_parent + eps))
            se_loss += torch.sum(delta_vol * log_vol_ratio_k)
        se_loss = -1 / vol_G * se_loss
        return se_loss
