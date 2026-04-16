import torch
import torch.nn as nn
from torch_scatter import scatter_softmax
import math
from utils.model_utils import gumbel_softmax, graph_top_K, normalize_adj, givens_rot_mat


class LorentzGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_dim, out_dim, use_bias, dropout, use_att, nonlin=None,
                 edge_conditioned=False, edge_attr_dim=1, edge_attr_hidden_dim=64, edge_gate_scale=0.35,
                 edge_matched_only=False, edge_confidence_gate=False, edge_confidence_temp=1.0):
        super(LorentzGraphConvolution, self).__init__()
        self.linear = LorentzLinear(manifold, in_dim, out_dim, use_bias, dropout, nonlin=nonlin)
        self.agg = LorentzAgg(
            manifold,
            out_dim,
            dropout,
            use_att,
            edge_conditioned=edge_conditioned,
            edge_attr_dim=edge_attr_dim,
            edge_attr_hidden_dim=edge_attr_hidden_dim,
            edge_gate_scale=edge_gate_scale,
            edge_matched_only=edge_matched_only,
            edge_confidence_gate=edge_confidence_gate,
            edge_confidence_temp=edge_confidence_temp,
        )

    def forward(self, x, adj, edge_attr=None, edge_mask=None, use_edge_attr=False):
        h = self.linear(x)
        h = self.agg(h, adj, edge_attr=edge_attr, edge_mask=edge_mask, use_edge_attr=use_edge_attr)
        return h


class LorentzLinear(nn.Module):
    def __init__(self,
                 manifold,
                 in_dim,
                 out_dim,
                 bias=True,
                 dropout=0.1,
                 scale=10,
                 fixscale=False,
                 nonlin=None):
        super().__init__()
        self.manifold = manifold
        self.nonlin = nonlin
        self.in_features = in_dim
        self.out_features = out_dim
        self.bias = bias
        self.weight = nn.Linear(
            self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * math.log(scale), requires_grad=not fixscale)

    def forward(self, x):
        if self.nonlin is not None:
            x = self.nonlin(x)
        x = self.weight(self.dropout(x))
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1
        scale = (time * time - 1) / \
            (x_narrow * x_narrow).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        step = self.in_features
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.weight.weight[:, idx] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)


class LorentzAgg(nn.Module):
    """
    Lorentz aggregation layer.
    """

    def __init__(self, manifold, in_dim, dropout, use_att,
                 edge_conditioned=False, edge_attr_dim=1, edge_attr_hidden_dim=64, edge_gate_scale=0.35,
                 edge_matched_only=False, edge_confidence_gate=False, edge_confidence_temp=1.0):
        super(LorentzAgg, self).__init__()
        self.manifold = manifold

        self.in_features = in_dim
        self.dropout = dropout
        self.use_att = use_att
        self.edge_conditioned = bool(edge_conditioned)
        self.edge_gate_scale = float(edge_gate_scale)
        self.edge_matched_only = bool(edge_matched_only)
        self.edge_confidence_gate = bool(edge_confidence_gate)
        self.edge_confidence_temp = float(max(1e-3, edge_confidence_temp))
        self.last_msg_gate_factor_mean = 1.0
        self.last_msg_gate_factor_std = 0.0
        if self.use_att:
            self.key_linear = LorentzLinear(manifold, in_dim, in_dim)
            self.query_linear = LorentzLinear(manifold, in_dim, in_dim)
            self.bias = nn.Parameter(torch.zeros(()) + 20)
            self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(in_dim))
        if self.edge_conditioned:
            hidden = max(8, int(edge_attr_hidden_dim))
            self.edge_gate_mlp = nn.Sequential(
                nn.Linear(int(max(1, edge_attr_dim)), hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

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

    def _apply_edge_gate_to_adj(self, adj, edge_attr, edge_mask, use_edge_attr: bool):
        self.last_msg_gate_factor_mean = 1.0
        self.last_msg_gate_factor_std = 0.0
        if (not self.edge_conditioned) or (not bool(use_edge_attr)) or edge_attr is None:
            return adj

        adj_sp = adj.coalesce() if adj.is_sparse else adj.to_sparse().coalesce()
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)
        if edge_attr.shape[0] != adj_sp.indices().shape[1]:
            return adj

        gate_raw = self.edge_gate_mlp(edge_attr.to(dtype=adj_sp.values().dtype, device=adj_sp.values().device)).squeeze(-1)
        gate_raw = gate_raw - gate_raw.mean()
        gate_raw = gate_raw / gate_raw.std(unbiased=False).clamp_min(1e-6)
        if self.edge_confidence_gate:
            conf = torch.sigmoid((gate_raw.abs() - 1.0) / self.edge_confidence_temp)
            gate_raw = conf * gate_raw
        gate_raw = torch.tanh(gate_raw) * float(self.edge_gate_scale)
        factor = torch.exp(gate_raw)
        eff_mask = None
        if edge_mask is not None:
            eff_mask = edge_mask.to(dtype=torch.bool, device=factor.device)
        if self.edge_matched_only and eff_mask is not None:
            factor = torch.where(eff_mask, factor, torch.ones_like(factor))
        gated = torch.sparse_coo_tensor(
            adj_sp.indices(),
            adj_sp.values() * factor.to(dtype=adj_sp.values().dtype, device=adj_sp.values().device),
            size=adj_sp.size(),
            device=adj_sp.device,
        ).coalesce()
        gated = self._normalize_sparse_by_degree(gated)
        if eff_mask is not None and torch.any(eff_mask):
            stat_factor = factor[eff_mask]
        else:
            stat_factor = factor
        self.last_msg_gate_factor_mean = float(stat_factor.detach().mean().item())
        self.last_msg_gate_factor_std = float(stat_factor.detach().std(unbiased=False).item())
        return gated

    def forward(self, x, adj, edge_attr=None, edge_mask=None, use_edge_attr=False):
        adj_eff = self._apply_edge_gate_to_adj(adj, edge_attr=edge_attr, edge_mask=edge_mask, use_edge_attr=use_edge_attr)
        if self.use_att:
            query = self.query_linear(x)
            key = self.key_linear(x)
            att_adj = 2 + 2 * self.manifold.cinner(query, key)
            att_adj = att_adj / self.scale + self.bias
            att_adj = torch.sigmoid(att_adj)
            att_adj = torch.mul(adj_eff.to_dense(), att_adj)
            support_t = torch.matmul(att_adj, x)
        else:
            # CUDA sparse addmm currently does not support bfloat16.
            # Keep sparse aggregation in fp32 and cast back to preserve AMP flow.
            if adj_eff.is_sparse and torch.is_autocast_enabled():
                out_dtype = x.dtype
                with torch.autocast(device_type="cuda", enabled=False):
                    support_t = torch.matmul(adj_eff.float(), x.float())
                support_t = support_t.to(dtype=out_dtype)
            else:
                support_t = torch.matmul(adj_eff, x)

        denorm = (-self.manifold.inner(None, support_t, keepdim=True))
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        output = support_t / denorm
        return output


class LorentzAssignment(nn.Module):
    def __init__(self, manifold, in_dim, hid_dim, num_assign, dropout,
                 bias=False, temperature=0.2, edge_variant='V1',
                 edge_fusion_gamma=1.0, edge_confidence_quantile=0.0,
                 edge_adaptive_alpha=False, edge_adaptive_alpha_strength=2.0,
                 edge_adaptive_alpha_bias=0.0, edge_reliability_temp=1.0,
                 edge_attr_hidden_dim=64, edge_attr_fusion_scale=1.0, edge_attr_dim=1):
        super(LorentzAssignment, self).__init__()
        self.manifold = manifold
        self.num_assign = num_assign
        self.edge_variant = edge_variant
        self.edge_fusion_gamma = edge_fusion_gamma
        self.edge_confidence_quantile = float(max(0.0, min(1.0, edge_confidence_quantile)))
        self.edge_adaptive_alpha = bool(edge_adaptive_alpha)
        self.edge_adaptive_alpha_strength = float(edge_adaptive_alpha_strength)
        self.edge_adaptive_alpha_bias = float(edge_adaptive_alpha_bias)
        self.edge_reliability_temp = float(max(1e-3, edge_reliability_temp))
        self.edge_attr_fusion_scale = float(edge_attr_fusion_scale)
        self.last_graph_alpha = 1.0
        self.last_reliability_mean = 1.0
        self.last_mix_beta = 0.0
        self.assign_linear = nn.Linear(in_dim, num_assign, bias=bias)
        nn.init.xavier_normal_(self.assign_linear.weight)
        self.temperature = temperature
        self.key_linear = LorentzLinear(manifold, in_dim, hid_dim, bias=False)
        self.query_linear = LorentzLinear(manifold, in_dim, hid_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.edge_attr_encoder = nn.Sequential(
            nn.Linear(int(max(1, edge_attr_dim)), max(8, int(edge_attr_hidden_dim))),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, int(edge_attr_hidden_dim)), 2),
        )
        self.edge_attr_lorentz_encoder = nn.Sequential(
            nn.Linear(int(max(1, edge_attr_dim)), max(8, int(edge_attr_hidden_dim))),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, int(edge_attr_hidden_dim)), max(1, int(hid_dim) - 1)),
        )

    def set_edge_fusion_gamma(self, gamma: float):
        self.edge_fusion_gamma = float(gamma)

    def _graph_alpha(self, signal: torch.Tensor, fallback_dtype: torch.dtype, fallback_device: torch.device):
        if not self.edge_adaptive_alpha:
            return torch.tensor(1.0, dtype=fallback_dtype, device=fallback_device)
        with torch.no_grad():
            s_mean = signal.detach().mean()
            s_std = signal.detach().std(unbiased=False)
            raw = s_mean - s_std
            alpha = torch.sigmoid(
                self.edge_adaptive_alpha_strength * raw + self.edge_adaptive_alpha_bias
            ).clamp(0.05, 0.95)
        return alpha

    def _struct_reliability_and_log(self, edge_value: torch.Tensor):
        edge_log = torch.log(edge_value.clamp_min(1e-8))
        center = torch.median(edge_value.detach())
        spread = edge_value.detach().std(unbiased=False).clamp_min(1e-6) * self.edge_reliability_temp
        reliability = torch.sigmoid((edge_value - center) / spread)
        if self.edge_confidence_quantile > 0.0:
            qv = float(min(0.999, max(0.0, self.edge_confidence_quantile)))
            threshold = torch.quantile(edge_value.detach(), qv)
            conf_mask = (edge_value >= threshold).to(edge_log.dtype)
            reliability = reliability * conf_mask
        return reliability, edge_log

    def forward(self, x, adj, edge_attr=None, use_edge_attr=False):
        ass = self.assign_linear(self.manifold.logmap0(x)).softmax(-1)
        q = self.query_linear(x)
        k = self.key_linear(x)
        adj_coo = adj.coalesce() if adj.is_sparse else adj.to_sparse().coalesce()
        edge_index = adj_coo.indices()
        edge_value = adj_coo.values()
        src, dst = edge_index[0], edge_index[1]
        score = self.manifold.dist(q[src], k[dst])
        score = -score
        if self.edge_variant == 'V5':
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
            self.last_graph_alpha = float(graph_alpha.detach().cpu().item())
            self.last_reliability_mean = float(reliability.detach().mean().cpu().item())
            self.last_mix_beta = 0.0
            score = score + float(self.edge_fusion_gamma) * graph_alpha * reliability * edge_log
        elif self.edge_variant == 'V40' and bool(use_edge_attr) and edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(1)
            if edge_attr.shape[0] == edge_value.shape[0] and edge_attr.shape[1] >= 3:
                state = torch.nan_to_num(
                    edge_attr[:, :3].to(dtype=score.dtype, device=score.device),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).clamp_min(0.0)
                state = state / state.sum(dim=1, keepdim=True).clamp_min(1e-6)
                support = state[:, 0]
                boundary = state[:, 1]
                neutral = state[:, 2]
                reliability = (1.0 - neutral).clamp(0.0, 1.0)
                graph_alpha = self._graph_alpha(reliability, fallback_dtype=score.dtype, fallback_device=score.device)
                attr_term = reliability * (support - boundary)
                attr_term = (attr_term - attr_term.mean()) / attr_term.std(unbiased=False).clamp_min(1e-6)
                score = score + float(self.edge_fusion_gamma) * self.edge_attr_fusion_scale * graph_alpha * attr_term
                self.last_graph_alpha = float(graph_alpha.detach().cpu().item())
                self.last_reliability_mean = float(reliability.detach().mean().cpu().item())
                self.last_mix_beta = float(boundary.detach().mean().cpu().item())
            else:
                self.last_graph_alpha = 1.0
                self.last_reliability_mean = 1.0
                self.last_mix_beta = 0.0
        elif self.edge_variant in {'V6', 'V7', 'V8', 'V12', 'V13', 'V31', 'V32', 'V33'} and bool(use_edge_attr) and edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(1)
            if edge_attr.shape[0] == edge_value.shape[0]:
                edge_attr = torch.nan_to_num(
                    edge_attr.to(dtype=score.dtype, device=score.device),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                attr_out = self.edge_attr_encoder(edge_attr)
                attr_bias = attr_out[:, 0]
                attr_gate = torch.sigmoid(attr_out[:, 1])
                attr_bias = (attr_bias - attr_bias.mean()) / attr_bias.std(unbiased=False).clamp_min(1e-6)
                edge_log = torch.log(edge_value.clamp_min(1e-8))
                if self.edge_variant == 'V6':
                    reliability = attr_gate
                    if self.edge_confidence_quantile > 0.0:
                        qv = float(min(0.999, max(0.0, self.edge_confidence_quantile)))
                        threshold = torch.quantile(reliability.detach(), qv)
                        reliability = reliability * (reliability >= threshold).to(reliability.dtype)
                    graph_alpha = self._graph_alpha(reliability, fallback_dtype=score.dtype, fallback_device=score.device)
                    attr_term = reliability * attr_bias
                    self.last_mix_beta = 1.0
                elif self.edge_variant == 'V7':
                    reliability = attr_gate
                    if self.edge_confidence_quantile > 0.0:
                        qv = float(min(0.999, max(0.0, self.edge_confidence_quantile)))
                        threshold = torch.quantile(reliability.detach(), qv)
                        reliability = reliability * (reliability >= threshold).to(reliability.dtype)
                    graph_alpha = self._graph_alpha(reliability, fallback_dtype=score.dtype, fallback_device=score.device)
                    align = torch.tanh(attr_bias * torch.tanh(edge_log))
                    attr_term = 0.7 * reliability * attr_bias + 0.3 * align
                    self.last_mix_beta = 1.0
                elif self.edge_variant == 'V8':
                    # V8: calibrate attribute fusion by agreement with structural prior.
                    struct_rel, _ = self._struct_reliability_and_log(edge_value)
                    attr_rel = attr_gate
                    reliability = 0.5 * (struct_rel + attr_rel)
                    if self.edge_confidence_quantile > 0.0:
                        qv = float(min(0.999, max(0.0, self.edge_confidence_quantile)))
                        threshold = torch.quantile(reliability.detach(), qv)
                        reliability = reliability * (reliability >= threshold).to(reliability.dtype)
                    graph_alpha = self._graph_alpha(reliability, fallback_dtype=score.dtype, fallback_device=score.device)

                    edge_z = (edge_log - edge_log.mean()) / edge_log.std(unbiased=False).clamp_min(1e-6)
                    attr_z = (attr_bias - attr_bias.mean()) / attr_bias.std(unbiased=False).clamp_min(1e-6)
                    with torch.no_grad():
                        agreement = (edge_z.detach() * attr_z.detach()).mean().clamp(-1.0, 1.0)
                        mix_beta = torch.sigmoid(
                            self.edge_adaptive_alpha_strength * agreement + self.edge_adaptive_alpha_bias
                        ).clamp(0.05, 0.95)
                    self.last_mix_beta = float(mix_beta.detach().cpu().item())

                    struct_term = struct_rel * torch.tanh(edge_log)
                    attr_term_raw = attr_rel * torch.tanh(attr_bias)
                    attr_term = (1.0 - mix_beta) * struct_term + mix_beta * attr_term_raw
                elif self.edge_variant in {'V12', 'V31', 'V32', 'V33'}:
                    # V12: keep V5 as stable trunk and add calibrated edge-attribute residual.
                    struct_rel, _ = self._struct_reliability_and_log(edge_value)
                    attr_rel = attr_gate
                    if self.edge_confidence_quantile > 0.0:
                        qv = float(min(0.999, max(0.0, self.edge_confidence_quantile)))
                        threshold = torch.quantile(struct_rel.detach(), qv)
                        keep = (struct_rel >= threshold).to(struct_rel.dtype)
                        struct_rel = struct_rel * keep
                        attr_rel = attr_rel * keep
                    graph_alpha = self._graph_alpha(struct_rel, fallback_dtype=score.dtype, fallback_device=score.device)
                    reliability = struct_rel

                    edge_z = (edge_log - edge_log.mean()) / edge_log.std(unbiased=False).clamp_min(1e-6)
                    attr_z = (attr_bias - attr_bias.mean()) / attr_bias.std(unbiased=False).clamp_min(1e-6)
                    with torch.no_grad():
                        agreement = (edge_z.detach() * attr_z.detach()).mean().clamp(-1.0, 1.0)
                        mix_beta = torch.sigmoid(
                            self.edge_adaptive_alpha_strength * agreement + self.edge_adaptive_alpha_bias
                        ).clamp(0.05, 0.95)
                    self.last_mix_beta = float(mix_beta.detach().cpu().item())

                    residual = attr_rel * attr_bias
                    residual = residual / residual.std(unbiased=False).clamp_min(1e-6)
                    attr_term = struct_rel * edge_log + mix_beta * residual
                else:
                    # V13: map edge attributes into Lorentz space and fuse as geometric residual.
                    reliability = attr_gate
                    if self.edge_confidence_quantile > 0.0:
                        qv = float(min(0.999, max(0.0, self.edge_confidence_quantile)))
                        threshold = torch.quantile(reliability.detach(), qv)
                        reliability = reliability * (reliability >= threshold).to(reliability.dtype)
                    graph_alpha = self._graph_alpha(reliability, fallback_dtype=score.dtype, fallback_device=score.device)
                    self.last_mix_beta = 1.0

                    # Build edge points in Lorentz model from edge_attr tangent vectors.
                    edge_spatial = self.edge_attr_lorentz_encoder(edge_attr)
                    edge_tangent = torch.cat(
                        [torch.zeros((edge_spatial.shape[0], 1), dtype=edge_spatial.dtype, device=edge_spatial.device), edge_spatial],
                        dim=1,
                    )
                    edge_lorentz = self.manifold.expmap0(edge_tangent)

                    # Pair node endpoints into a Lorentz midpoint surrogate.
                    node_pair = q[src] + k[dst]
                    node_denorm = (-self.manifold.inner(None, node_pair, keepdim=True)).abs().clamp_min(1e-8).sqrt()
                    node_pair = node_pair / node_denorm

                    edge_geo = 2 + 2 * self.manifold.inner(edge_lorentz, node_pair, keepdim=False)
                    edge_geo = (edge_geo - edge_geo.mean()) / edge_geo.std(unbiased=False).clamp_min(1e-6)
                    attr_term = reliability * (0.7 * torch.tanh(edge_geo) + 0.3 * torch.tanh(attr_bias))
                score = score + float(self.edge_fusion_gamma) * self.edge_attr_fusion_scale * graph_alpha * attr_term
                self.last_graph_alpha = float(graph_alpha.detach().cpu().item())
                self.last_reliability_mean = float(reliability.detach().mean().cpu().item())
            else:
                self.last_graph_alpha = 1.0
                self.last_reliability_mean = 1.0
                self.last_mix_beta = 0.0
        else:
            self.last_graph_alpha = 1.0
            self.last_reliability_mean = 1.0
            self.last_mix_beta = 0.0
        score = scatter_softmax(score, src, dim=-1)
        att = torch.sparse_coo_tensor(edge_index, score, size=(x.shape[0], x.shape[0])).to(x.device)
        if att.is_sparse and torch.is_autocast_enabled():
            out_dtype = ass.dtype
            with torch.autocast(device_type="cuda", enabled=False):
                ass = torch.matmul(att.float(), ass.float())
            ass = ass.to(dtype=out_dtype)
        else:
            ass = torch.matmul(att, ass)   # (N_k, N_{k-1})
        ass = gumbel_softmax(torch.log(ass + 1e-6), temperature=self.temperature)
        return ass


class LSENetLayer(nn.Module):
    def __init__(self, manifold, in_dim, hid_dim, num_assign, dropout,
                 bias=False, use_att=False, nonlin=None, temperature=0.2,
                 edge_variant='V1', edge_fusion_gamma=1.0, edge_confidence_quantile=0.0,
                 edge_adaptive_alpha=False, edge_adaptive_alpha_strength=2.0,
                 edge_adaptive_alpha_bias=0.0, edge_reliability_temp=1.0,
                 edge_attr_hidden_dim=64, edge_attr_fusion_scale=1.0, edge_attr_dim=1):
        super(LSENetLayer, self).__init__()
        self.manifold = manifold
        # self.embeder = LorentzGraphConvolution(manifold, in_dim, hid_dim,
        #                                        True, dropout, use_att, nonlin)
        self.assigner = LorentzAssignment(manifold, hid_dim,
                                          hid_dim, num_assign,
                                          dropout, bias, temperature,
                                          edge_variant=edge_variant,
                                          edge_fusion_gamma=edge_fusion_gamma,
                                          edge_confidence_quantile=edge_confidence_quantile,
                                          edge_adaptive_alpha=edge_adaptive_alpha,
                                          edge_adaptive_alpha_strength=edge_adaptive_alpha_strength,
                                          edge_adaptive_alpha_bias=edge_adaptive_alpha_bias,
                                          edge_reliability_temp=edge_reliability_temp,
                                          edge_attr_hidden_dim=edge_attr_hidden_dim,
                                          edge_attr_fusion_scale=edge_attr_fusion_scale,
                                          edge_attr_dim=edge_attr_dim)

    def forward(self, x, adj, edge_attr=None, use_edge_attr=False):
        # x = self.embeder(x, adj)
        ass = self.assigner(x, adj, edge_attr=edge_attr, use_edge_attr=use_edge_attr)
        support_t = ass.t() @ x
        denorm = (-self.manifold.inner(None, support_t, keepdim=True))
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        x_par = support_t / denorm

        if adj.is_sparse and torch.is_autocast_enabled():
            out_dtype = ass.dtype
            with torch.autocast(device_type="cuda", enabled=False):
                adj_par = ass.float().t() @ adj.float() @ ass.float()
            adj_par = adj_par.to(dtype=out_dtype)
        else:
            adj_par = ass.t() @ adj @ ass
        idx = adj_par.nonzero().t()
        adj_par = torch.sparse_coo_tensor(idx, adj_par[idx[0], idx[1]], size=adj_par.shape)
        return x_par, adj_par, ass, x


class LorentzBoost(nn.Module):
    """
    Implements a learnable Lorentz boost transformation without eigendecomposition.
    Input: x in Lorentz model L^{d} (shape: [..., d+1])
    Output: L(x) in L^{d} (same shape)
    """
    def __init__(self, in_dim):  # in_dim = d+1
        super().__init__()
        self.in_dim = in_dim
        # Parameterize boost velocity beta in R^d (spatial part)
        self.beta = nn.Parameter(torch.randn(in_dim - 1) * 0.01)  # small init

    def forward(self, x):
        """
        x: [..., d+1], assumed to be in Lorentz model (x0 > 0, <x,x>_L = -1)
        Returns L(x): [..., d+1] in Lorentz model
        """
        d = self.in_dim - 1
        # Re-parameterize to keep ||beta|| < 1 and avoid invalid Lorentz factor.
        beta = torch.tanh(self.beta)
        beta_norm = torch.norm(beta, p=2)
        max_norm = 1.0 - 1e-4
        if beta_norm > max_norm:
            beta = beta * (max_norm / (beta_norm + 1e-12))
        beta_norm_sq = (beta ** 2).sum().clamp(max=1.0 - 1e-8)
        gamma = 1.0 / torch.sqrt(1.0 - beta_norm_sq + 1e-8)  # Lorentz factor

        # Construct boost matrix L (d+1, d+1)
        L = torch.eye(self.in_dim, device=x.device)
        L[0, 0] = gamma
        L[0, 1:] = -gamma * beta
        L[1:, 0] = -gamma * beta
        L[1:, 1:] += (gamma - 1) * torch.outer(beta, beta) / (beta_norm_sq + 1e-8)

        # Apply transformation
        Lx = torch.einsum('ij,...j->...i', L, x)
        return Lx
