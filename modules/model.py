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
                 edge_attr_hidden_dim=64, edge_attr_fusion_scale=1.0, edge_attr_dim=1,
                 edge_attr_hierarchical=False):
        super(LSENet, self).__init__()
        assert max_nums is not None
        self.manifold = manifold
        self.max_nums = max_nums  # [N_{H-1}, ..., N_1]
        self.height = len(max_nums) + 1
        self.edge_attr_hierarchical = bool(edge_attr_hierarchical)

        # Project input to Lorentz space (d+1)
        self.input_proj = LorentzGraphConvolution(manifold, in_dim + 1, hid_dim + 1,
                                                  True, dropout, False,
                                                  select_activation(nonlin_str))
        self.input_proj2 = LorentzGraphConvolution(manifold, hid_dim + 1, hid_dim + 1,
                                                  True, dropout, False,
                                                  select_activation(nonlin_str))
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
                edge_attr_dim=edge_attr_dim,
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
        return {
            "graph_alpha_mean": float(sum(alphas) / len(alphas)) if len(alphas) > 0 else 1.0,
            "edge_reliability_mean": float(sum(rels) / len(rels)) if len(rels) > 0 else 1.0,
            "edge_mix_beta_mean": float(sum(mixes) / len(mixes)) if len(mixes) > 0 else 0.0,
        }

    def embed_leaf(self, x, adj):
        # Map raw features to Lorentz leaf embedding
        o = torch.zeros_like(x[:, :1])                # (N, 1)
        x = torch.cat([o, x], dim=1)           # (N, d+1)
        x = self.manifold.expmap0(x)                  # project to Lorentz
        x = self.input_proj(x, adj)  # (N, d + 1)
        x = self.input_proj2(x, adj)
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

    def forward(self, x, adj, edge_attr=None, use_edge_attr=False):
        """
        Args:
            x: raw node features (N, D)
            adj: sparse adjacency
        """
        z = self.embed_leaf(x, adj)
        tree_coord_dict = {self.height: z}
        ass_dict = {}
        adj_dict = {self.height: adj}

        current_z = z
        current_adj = adj
        current_edge_attr = edge_attr if bool(use_edge_attr) and edge_attr is not None else None

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
                current_edge_attr = self._coarsen_edge_attr_hard(current_adj, current_edge_attr, ass, adj_par)
            else:
                current_edge_attr = None
            current_z = z_par
            current_adj = adj_par

        # Root (level 0) is Frechet mean of level 1
        root = self.manifold.frechet_mean(current_z)
        tree_coord_dict[0] = root
        ass_dict[1] = torch.ones(current_z.size(0), 1, device=x.device)

        return tree_coord_dict, ass_dict, adj_dict
