import torch
import torch.nn as nn
from modules.layers import LSENetLayer, LorentzGraphConvolution
from utils.model_utils import select_activation


class LSENet(nn.Module):
    def __init__(self, manifold, in_dim, hid_dim, max_nums,
                 temperature=0.2, dropout=0.5, nonlin_str='relu',
                 edge_variant='V1', edge_fusion_gamma=1.0, edge_confidence_quantile=0.0,
                 edge_adaptive_alpha=False, edge_adaptive_alpha_strength=2.0,
                 edge_adaptive_alpha_bias=0.0, edge_reliability_temp=1.0,
                 edge_attr_hidden_dim=64, edge_attr_fusion_scale=1.0,
                 edge_relation_channels=4, edge_relation_hidden_dim=64, edge_relation_assign_scale=1.0,
                 edge_attr_dim=1, edge_attr_hierarchical=False, edge_attr_pool_topk=1,
                 edge_msg_conditioned=False, edge_msg_gate_scale=0.35,
                 edge_msg_matched_only=False, edge_msg_confidence_gate=False, edge_msg_confidence_temp=1.0,
                 edge_attr_pool_confidence=False, edge_attr_pool_conf_power=1.0):
        super(LSENet, self).__init__()
        assert max_nums is not None
        self.manifold = manifold
        self.max_nums = max_nums  # [N_{H-1}, ..., N_1]
        self.height = len(max_nums) + 1
        self.edge_variant = str(edge_variant).upper()
        self.edge_relation_enabled = self.edge_variant in {'V40'}
        self.edge_attr_hierarchical = bool(edge_attr_hierarchical)
        self.edge_attr_pool_topk = max(1, int(edge_attr_pool_topk))
        self.edge_msg_conditioned = bool(edge_msg_conditioned)
        self.edge_msg_gate_scale = float(edge_msg_gate_scale)
        self.edge_msg_matched_only = bool(edge_msg_matched_only)
        self.edge_msg_confidence_gate = bool(edge_msg_confidence_gate)
        self.edge_msg_confidence_temp = float(max(1e-3, edge_msg_confidence_temp))
        self.edge_attr_pool_confidence = bool(edge_attr_pool_confidence)
        self.edge_attr_pool_conf_power = float(max(0.0, edge_attr_pool_conf_power))
        self.last_hier_edge_levels_active_ratio = 0.0
        self.last_hier_edge_nonzero_ratio = 0.0
        self.last_hier_edge_mean_abs = 0.0
        self.last_msg_gate_factor_mean = 1.0
        self.last_msg_gate_factor_std = 0.0
        if self.edge_relation_enabled:
            self.edge_relation_encoder = EdgeRelationStateEncoder(
                edge_attr_dim=int(max(1, edge_attr_dim)),
                num_relations=int(edge_relation_channels),
                hidden_dim=int(edge_relation_hidden_dim),
                dropout=dropout,
            )
            edge_model_attr_dim = int(self.edge_relation_encoder.out_dim)
        else:
            self.edge_relation_encoder = None
            edge_model_attr_dim = int(max(1, edge_attr_dim))
        self.edge_relation_assign_scale = float(edge_relation_assign_scale)

        # Project input to Lorentz space (d+1)
        self.input_proj = LorentzGraphConvolution(manifold, in_dim + 1, hid_dim + 1,
                                                  True, dropout, False,
                                                  select_activation(nonlin_str),
                                                  edge_conditioned=self.edge_msg_conditioned,
                                                  edge_attr_dim=edge_model_attr_dim,
                                                  edge_attr_hidden_dim=edge_attr_hidden_dim,
                                                  edge_gate_scale=self.edge_msg_gate_scale,
                                                  edge_matched_only=self.edge_msg_matched_only,
                                                  edge_confidence_gate=self.edge_msg_confidence_gate,
                                                  edge_confidence_temp=self.edge_msg_confidence_temp)
        self.input_proj2 = LorentzGraphConvolution(manifold, hid_dim + 1, hid_dim + 1,
                                                  True, dropout, False,
                                                  select_activation(nonlin_str),
                                                  edge_conditioned=self.edge_msg_conditioned,
                                                  edge_attr_dim=edge_model_attr_dim,
                                                  edge_attr_hidden_dim=edge_attr_hidden_dim,
                                                  edge_gate_scale=self.edge_msg_gate_scale,
                                                  edge_matched_only=self.edge_msg_matched_only,
                                                  edge_confidence_gate=self.edge_msg_confidence_gate,
                                                  edge_confidence_temp=self.edge_msg_confidence_temp)
        self.dropout = nn.Dropout(dropout)

        # Build layers bottom-up: layer[0] = leaf → level H-1; layer[-1] = level 1 → root
        self.layers = nn.ModuleList()
        curr_dim = hid_dim + 1  # +1 for time-like coordinate
        for i in range(self.height - 1):
            self.layers.append(LSENetLayer(
                manifold, curr_dim, hid_dim + 1, max_nums[i],
                dropout=dropout, temperature=temperature,
                nonlin=select_activation(nonlin_str),
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
                edge_relation_assign_scale=edge_relation_assign_scale,
                edge_attr_dim=edge_model_attr_dim,
            ))
            curr_dim = hid_dim + 1  # parent embedding dim

    def set_edge_fusion_gamma(self, gamma: float):
        for layer in self.layers:
            if hasattr(layer, "assigner") and hasattr(layer.assigner, "set_edge_fusion_gamma"):
                layer.assigner.set_edge_fusion_gamma(gamma)

    def get_edge_adaptive_stats(self):
        alphas = []
        rels = []
        mixes = []
        for layer in self.layers:
            assigner = getattr(layer, "assigner", None)
            if assigner is None:
                continue
            if hasattr(assigner, "last_graph_alpha"):
                alphas.append(float(assigner.last_graph_alpha))
            if hasattr(assigner, "last_reliability_mean"):
                rels.append(float(assigner.last_reliability_mean))
            if hasattr(assigner, "last_mix_beta"):
                mixes.append(float(assigner.last_mix_beta))
        msg_means = []
        msg_stds = []
        for proj in (self.input_proj, self.input_proj2):
            agg = getattr(proj, "agg", None)
            if agg is None:
                continue
            msg_means.append(float(getattr(agg, "last_msg_gate_factor_mean", 1.0)))
            msg_stds.append(float(getattr(agg, "last_msg_gate_factor_std", 0.0)))
        if msg_means:
            self.last_msg_gate_factor_mean = float(sum(msg_means) / len(msg_means))
            self.last_msg_gate_factor_std = float(sum(msg_stds) / len(msg_stds))
        else:
            self.last_msg_gate_factor_mean = 1.0
            self.last_msg_gate_factor_std = 0.0
        return {
            "graph_alpha_mean": float(sum(alphas) / len(alphas)) if len(alphas) > 0 else 1.0,
            "edge_reliability_mean": float(sum(rels) / len(rels)) if len(rels) > 0 else 1.0,
            "edge_mix_beta_mean": float(sum(mixes) / len(mixes)) if len(mixes) > 0 else 0.0,
            "msg_gate_factor_mean": float(self.last_msg_gate_factor_mean),
            "msg_gate_factor_std": float(self.last_msg_gate_factor_std),
            "hier_edge_levels_active_ratio": float(self.last_hier_edge_levels_active_ratio),
            "hier_edge_nonzero_ratio": float(self.last_hier_edge_nonzero_ratio),
            "hier_edge_mean_abs": float(self.last_hier_edge_mean_abs),
        }

    def use_edge_relation_state(self) -> bool:
        return bool(self.edge_relation_enabled and self.edge_relation_encoder is not None)

    def edge_relation_state_dim(self) -> int:
        if self.edge_relation_encoder is None:
            return 0
        return int(self.edge_relation_encoder.out_dim)

    def encode_edge_relation_state(self, edge_attr: torch.Tensor | None, matched: torch.Tensor | None = None) -> torch.Tensor | None:
        if not self.use_edge_relation_state():
            return edge_attr
        return self.edge_relation_encoder.encode(edge_attr, matched=matched)

    def edge_relation_factor_from_state(self, edge_state: torch.Tensor | None, head: str = 'msg') -> torch.Tensor | None:
        if not self.use_edge_relation_state():
            return None
        return self.edge_relation_encoder.factor_from_state(edge_state, head=head)

    def edge_relation_regularization(self, edge_state: torch.Tensor | None, matched: torch.Tensor | None = None) -> torch.Tensor:
        if not self.use_edge_relation_state():
            return self.input_proj.linear.weight.weight.new_zeros(())
        return self.edge_relation_encoder.regularization(edge_state, matched=matched)

    def embed_leaf(self, x, adj, edge_attr=None, edge_mask=None, use_edge_attr=False):
        # Map raw features to Lorentz leaf embedding
        o = torch.zeros_like(x[:, :1])                # (N, 1)
        x = torch.cat([o, x], dim=1)           # (N, d+1)
        x = self.manifold.expmap0(x)                  # project to Lorentz
        msg_use_edge_attr = bool(use_edge_attr) and self.edge_msg_conditioned
        x = self.input_proj(x, adj, edge_attr=edge_attr, edge_mask=edge_mask, use_edge_attr=msg_use_edge_attr)  # (N, d + 1)
        x = self.input_proj2(x, adj, edge_attr=edge_attr, edge_mask=edge_mask, use_edge_attr=msg_use_edge_attr)
        return x

    @staticmethod
    def _to_sparse_coalesced(adj: torch.Tensor) -> torch.Tensor:
        if adj.is_sparse:
            return adj.coalesce()
        return adj.to_sparse().coalesce()

    @classmethod
    def _coarsen_edge_attr_hard(
        cls,
        adj_curr: torch.Tensor,
        edge_attr_curr: torch.Tensor,
        ass: torch.Tensor,
        adj_par: torch.Tensor,
    ) -> torch.Tensor | None:
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

    @classmethod
    def _align_parent_edge_attr(
        cls,
        num_parent: int,
        key: torch.Tensor,
        pair_weight: torch.Tensor,
        edge_attr: torch.Tensor,
        adj_par: torch.Tensor,
    ) -> torch.Tensor | None:
        if key.numel() == 0 or pair_weight.numel() == 0:
            return None
        order = torch.argsort(key)
        key = key[order]
        pair_weight = pair_weight[order]
        weighted_attr = (edge_attr * pair_weight.unsqueeze(1))[order]
        uniq, inv = torch.unique(key, sorted=True, return_inverse=True)

        agg_attr = torch.zeros((uniq.numel(), edge_attr.shape[1]), dtype=edge_attr.dtype, device=edge_attr.device)
        agg_attr.index_add_(0, inv, weighted_attr)
        agg_denom = torch.zeros((uniq.numel(),), dtype=edge_attr.dtype, device=edge_attr.device)
        agg_denom.index_add_(0, inv, pair_weight)
        agg_attr = agg_attr / agg_denom.clamp_min(1e-6).unsqueeze(1)

        adj_par_sp = cls._to_sparse_coalesced(adj_par)
        idx_par = adj_par_sp.indices()
        target_key = idx_par[0].long() * num_parent + idx_par[1].long()
        pos = torch.searchsorted(uniq, target_key)
        valid = pos < uniq.shape[0]
        matched = torch.zeros_like(valid, dtype=torch.bool)
        matched[valid] = uniq[pos[valid]] == target_key[valid]

        out_attr = torch.zeros((idx_par.shape[1], edge_attr.shape[1]), dtype=edge_attr.dtype, device=edge_attr.device)
        if matched.any():
            out_attr[matched] = agg_attr[pos[matched]]
        return out_attr

    @classmethod
    def _coarsen_edge_attr_soft_topk(
        cls,
        adj_curr: torch.Tensor,
        edge_attr_curr: torch.Tensor,
        ass: torch.Tensor,
        adj_par: torch.Tensor,
        topk: int,
        use_confidence: bool = False,
        conf_power: float = 1.0,
    ) -> torch.Tensor | None:
        if edge_attr_curr is None or edge_attr_curr.numel() == 0:
            return None
        if topk <= 1:
            return cls._coarsen_edge_attr_hard(adj_curr, edge_attr_curr, ass, adj_par)

        adj_curr_sp = cls._to_sparse_coalesced(adj_curr)
        idx = adj_curr_sp.indices()
        val = adj_curr_sp.values().float()
        if edge_attr_curr.shape[0] != idx.shape[1]:
            return None

        attr = edge_attr_curr.float()
        num_parent = int(ass.shape[1])
        k = min(int(topk), num_parent)
        prob_topk, idx_topk = torch.topk(ass, k=k, dim=1)
        prob_topk = prob_topk / prob_topk.sum(dim=1, keepdim=True).clamp_min(1e-6)
        conf = None
        if bool(use_confidence):
            if k <= 1:
                conf = torch.ones((ass.shape[0],), dtype=prob_topk.dtype, device=prob_topk.device)
            else:
                entropy = -(prob_topk * torch.log(prob_topk.clamp_min(1e-8))).sum(dim=1)
                norm = torch.log(torch.tensor(float(k), dtype=prob_topk.dtype, device=prob_topk.device)).clamp_min(1e-6)
                conf = (1.0 - entropy / norm).clamp(0.0, 1.0)
                if float(conf_power) != 1.0:
                    conf = conf.pow(float(conf_power))

        src = idx[0].long()
        dst = idx[1].long()
        src_prob = prob_topk[src]
        dst_prob = prob_topk[dst]
        src_parent = idx_topk[src]
        dst_parent = idx_topk[dst]

        key_parts = []
        weight_parts = []
        attr_parts = []
        for a in range(k):
            for b in range(k):
                pw = (src_prob[:, a] * dst_prob[:, b] * val).to(attr.dtype)
                if conf is not None:
                    pw = pw * conf[src].to(attr.dtype) * conf[dst].to(attr.dtype)
                keep = pw > 1e-8
                if not torch.any(keep):
                    continue
                key_parts.append(src_parent[:, a][keep].long() * num_parent + dst_parent[:, b][keep].long())
                weight_parts.append(pw[keep])
                attr_parts.append(attr[keep])

        if len(key_parts) == 0:
            return cls._coarsen_edge_attr_hard(adj_curr, edge_attr_curr, ass, adj_par)

        key = torch.cat(key_parts, dim=0)
        pair_weight = torch.cat(weight_parts, dim=0)
        attr_cat = torch.cat(attr_parts, dim=0)
        return cls._align_parent_edge_attr(num_parent, key, pair_weight, attr_cat, adj_par)

    def forward(self, x, adj, edge_attr=None, edge_mask=None, use_edge_attr=False):
        """
        Args:
            x: raw node features (N, D)
            adj: sparse adjacency
        """
        z = self.embed_leaf(x, adj, edge_attr=edge_attr, edge_mask=edge_mask, use_edge_attr=use_edge_attr)
        tree_coord_dict = {self.height: z}
        ass_dict = {}
        adj_dict = {self.height: adj}

        current_z = z
        current_adj = adj
        current_edge_attr = edge_attr if bool(use_edge_attr) and edge_attr is not None else None
        hier_level_active = []
        hier_nonzero = []
        hier_mean_abs = []

        for i, layer in enumerate(self.layers):
            layer_use_edge_attr = bool(use_edge_attr) and current_edge_attr is not None
            layer_edge_attr = current_edge_attr if layer_use_edge_attr else None
            z_par, adj_par, ass, z_curr = layer(
                current_z,
                current_adj,
                edge_attr=layer_edge_attr,
                use_edge_attr=layer_use_edge_attr,
            )

            level_curr = self.height - i
            level_par = self.height - i - 1
            tree_coord_dict[level_par] = z_par
            ass_dict[level_curr] = ass
            adj_dict[level_par] = adj_par

            if layer_use_edge_attr and self.edge_attr_hierarchical:
                next_edge_attr = self._coarsen_edge_attr_soft_topk(
                    current_adj,
                    current_edge_attr,
                    ass,
                    adj_par,
                    topk=self.edge_attr_pool_topk,
                    use_confidence=self.edge_attr_pool_confidence,
                    conf_power=self.edge_attr_pool_conf_power,
                )
                current_edge_attr = next_edge_attr
                if next_edge_attr is None or next_edge_attr.numel() == 0:
                    hier_level_active.append(0.0)
                    hier_nonzero.append(0.0)
                    hier_mean_abs.append(0.0)
                else:
                    hier_level_active.append(1.0)
                    row_strength = next_edge_attr.abs().sum(dim=1)
                    hier_nonzero.append(float((row_strength > 1e-8).float().mean().item()))
                    hier_mean_abs.append(float(next_edge_attr.abs().mean().item()))
            else:
                current_edge_attr = None
            current_z = z_par
            current_adj = adj_par

        self.last_hier_edge_levels_active_ratio = float(sum(hier_level_active) / len(hier_level_active)) if len(hier_level_active) > 0 else 0.0
        self.last_hier_edge_nonzero_ratio = float(sum(hier_nonzero) / len(hier_nonzero)) if len(hier_nonzero) > 0 else 0.0
        self.last_hier_edge_mean_abs = float(sum(hier_mean_abs) / len(hier_mean_abs)) if len(hier_mean_abs) > 0 else 0.0

        # Root (level 0) is Frechet mean of level 1
        root = self.manifold.frechet_mean(current_z)
        tree_coord_dict[0] = root
        ass_dict[1] = torch.ones(current_z.size(0), 1, device=x.device)

        return tree_coord_dict, ass_dict, adj_dict


