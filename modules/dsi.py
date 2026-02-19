import torch
import torch.nn as nn
from utils.model_utils import gumbel_softmax, graph_top_K
from manifold.lorentz import Lorentz
from modules.layers import LorentzBoost
from modules.model import LSENet
from torch_scatter import scatter_softmax

MIN_NORM = 1e-15
EPS = 1e-6


class DSI(nn.Module):
    def __init__(self, in_dim, hid_dim, num_nodes, max_nums, temperature=0.2,
                 dropout=0.5, nonlin_str='relu', tau=1.0, alpha=0.01, knn=8,
                 edge_variant='V1', edge_fusion_gamma=1.0, edge_confidence_quantile=0.0,
                 edge_adaptive_alpha=False, edge_adaptive_alpha_strength=2.0,
                 edge_adaptive_alpha_bias=0.0, edge_reliability_temp=1.0,
                 edge_attr_hidden_dim=64, edge_attr_fusion_scale=1.0,
                 edge_attr_dim=1,
                 edge_attr_hierarchical=False,
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
            edge_attr_dim=edge_attr_dim,
            edge_attr_hierarchical=bool(edge_attr_hierarchical),
        )
        self.lorentz_proj = LorentzBoost(hid_dim + 1)
        self.temperature = temperature
        self.tau = tau
        self.alpha = alpha
        self.knn = knn
        self.edge_variant = str(edge_variant).upper()
        self.knn_mode = str(knn_mode).lower()
        self.knn_auto_threshold = int(knn_auto_threshold)

    def set_edge_fusion_gamma(self, gamma: float):
        if hasattr(self.encoder, "set_edge_fusion_gamma"):
            self.encoder.set_edge_fusion_gamma(gamma)

    def get_edge_adaptive_stats(self):
        if hasattr(self.encoder, "get_edge_adaptive_stats"):
            return self.encoder.get_edge_adaptive_stats()
        return {"graph_alpha_mean": 1.0, "edge_reliability_mean": 1.0, "edge_mix_beta_mean": 0.0}

    def forward(self, data):
        features = data.x
        adj = getattr(data, "adj_msg", data.adj).clone()
        use_edge_attr = self._use_edge_attr_variant()
        edge_attr = getattr(data, "edge_attr", None) if use_edge_attr else None
        tree_coord_dict, ass_dict, adj_dict = self.encoder(features, adj, edge_attr=edge_attr, use_edge_attr=use_edge_attr)
        return tree_coord_dict, ass_dict, adj_dict

    def get_cluster_results(self, data):
        features = data.x
        adj = getattr(data, "adj_msg", data.adj).clone()
        use_edge_attr = self._use_edge_attr_variant()
        edge_attr = getattr(data, "edge_attr", None) if use_edge_attr else None
        coord_dict, ass_dict, _ = self.encoder(features, adj, edge_attr=edge_attr, use_edge_attr=use_edge_attr)
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
        adj_base_msg = getattr(data, "adj_msg", data.adj).clone()
        adj_base_si = getattr(data, "adj_si", adj_base_msg).clone()

        if self._use_edge_knn_mode(data.x.shape[0]):
            # For large-graph edge mode, stop gradient through adjacency construction.
            with torch.no_grad():
                z_leaf = self.encoder.embed_leaf(data.x, adj_base_msg)
                z_leaf = self.lorentz_proj(z_leaf)
                adj_aug = self._edge_candidate_adj(z_leaf, adj_base_msg, self.knn)
        else:
            z_leaf = self.encoder.embed_leaf(data.x, adj_base_msg)
            z_leaf = self.lorentz_proj(z_leaf)
            neg_dist2 = 2 + 2 * self.manifold.cinner(z_leaf, z_leaf)
            adj_aug = graph_top_K(torch.softmax(neg_dist2 / self.tau, dim=-1), k=self.knn)

        adj_train_msg = (self.alpha * adj_aug + adj_base_msg).coalesce()
        adj_train_si = (self.alpha * adj_aug + adj_base_si).coalesce()

        use_edge_attr = self._use_edge_attr_variant()
        edge_attr = None
        if use_edge_attr:
            edge_attr_base = getattr(data, "edge_attr", None)
            edge_attr = self._align_edge_attr_to_adj(adj_base_msg, edge_attr_base, adj_train_msg)
        _, ass_aug_dict, _ = self.encoder(
            data.x, adj_train_msg, edge_attr=edge_attr, use_edge_attr=use_edge_attr
        )
        adj_si_dict = self._build_hierarchy_adj_from_assign(adj_train_si, ass_aug_dict)
        loss = self._si_loss(ass_aug_dict, adj_si_dict, eps)
        return loss

    def _use_edge_attr_variant(self) -> bool:
        return self.edge_variant in {'V6', 'V7', 'V8', 'V12'}

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

    def _use_edge_knn_mode(self, num_nodes: int) -> bool:
        mode = self.knn_mode
        if mode == 'edge':
            return True
        if mode == 'dense':
            return False
        return int(num_nodes) > int(self.knn_auto_threshold)

    def _edge_candidate_adj(self, z_leaf, adj, k: int):
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
        if base_edge_attr is None:
            return None
        if base_edge_attr.numel() == 0:
            return base_edge_attr
        base = base_adj.coalesce()
        target = target_adj.coalesce()
        base_idx = base.indices()
        target_idx = target.indices()
        num_nodes = int(base.size(0))
        base_attr = base_edge_attr.float()
        if base_attr.shape[0] != base_idx.shape[1]:
            return None

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
        return target_attr

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
