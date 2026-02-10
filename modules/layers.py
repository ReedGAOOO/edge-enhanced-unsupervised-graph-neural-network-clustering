import torch
import torch.nn as nn
from torch_scatter import scatter_softmax
import math
from utils.model_utils import gumbel_softmax, graph_top_K, normalize_adj, givens_rot_mat


class LorentzGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_dim, out_dim, use_bias, dropout, use_att, nonlin=None):
        super(LorentzGraphConvolution, self).__init__()
        self.linear = LorentzLinear(manifold, in_dim, out_dim, use_bias, dropout, nonlin=nonlin)
        self.agg = LorentzAgg(manifold, out_dim, dropout, use_att)

    def forward(self, x, adj):
        h = self.linear(x)
        h = self.agg(h, adj)
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

    def __init__(self, manifold, in_dim, dropout, use_att):
        super(LorentzAgg, self).__init__()
        self.manifold = manifold

        self.in_features = in_dim
        self.dropout = dropout
        self.use_att = use_att
        if self.use_att:
            self.key_linear = LorentzLinear(manifold, in_dim, in_dim)
            self.query_linear = LorentzLinear(manifold, in_dim, in_dim)
            self.bias = nn.Parameter(torch.zeros(()) + 20)
            self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(in_dim))

    def forward(self, x, adj):
        if self.use_att:
            query = self.query_linear(x)
            key = self.key_linear(x)
            att_adj = 2 + 2 * self.manifold.cinner(query, key)
            att_adj = att_adj / self.scale + self.bias
            att_adj = torch.sigmoid(att_adj)
            att_adj = torch.mul(adj.to_dense(), att_adj)
            support_t = torch.matmul(att_adj, x)
        else:
            support_t = torch.matmul(adj, x)

        denorm = (-self.manifold.inner(None, support_t, keepdim=True))
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        output = support_t / denorm
        return output


class LorentzAssignment(nn.Module):
    def __init__(self, manifold, in_dim, hid_dim, num_assign, dropout,
                 bias=False, temperature=0.2, edge_variant='V1',
                 edge_fusion_gamma=1.0, edge_confidence_quantile=0.0,
                 edge_adaptive_alpha=False, edge_adaptive_alpha_strength=2.0,
                 edge_adaptive_alpha_bias=0.0, edge_reliability_temp=1.0):
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
        self.last_graph_alpha = 1.0
        self.last_reliability_mean = 1.0
        self.assign_linear = nn.Linear(in_dim, num_assign, bias=bias)
        nn.init.xavier_normal_(self.assign_linear.weight)
        self.temperature = temperature
        self.key_linear = LorentzLinear(manifold, in_dim, hid_dim, bias=False)
        self.query_linear = LorentzLinear(manifold, in_dim, hid_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def set_edge_fusion_gamma(self, gamma: float):
        self.edge_fusion_gamma = float(gamma)

    def forward(self, x, adj):
        ass = self.assign_linear(self.manifold.logmap0(x)).softmax(-1)
        q = self.query_linear(x)
        k = self.key_linear(x)
        adj_coo = adj.coalesce()
        edge_index = adj_coo.indices()
        edge_value = adj_coo.values()
        src, dst = edge_index[0], edge_index[1]
        score = self.manifold.dist(q[src], k[dst])
        score = -score
        if self.edge_variant == 'V5':
            edge_log = torch.log(edge_value.clamp_min(1e-8))
            center = torch.median(edge_value.detach())
            spread = edge_value.detach().std(unbiased=False).clamp_min(1e-6) * self.edge_reliability_temp
            reliability = torch.sigmoid((edge_value - center) / spread)
            if self.edge_confidence_quantile > 0.0:
                qv = float(min(0.999, max(0.0, self.edge_confidence_quantile)))
                threshold = torch.quantile(edge_value.detach(), qv)
                conf_mask = (edge_value >= threshold).to(edge_log.dtype)
                reliability = reliability * conf_mask
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
            score = score + float(self.edge_fusion_gamma) * graph_alpha * reliability * edge_log
        else:
            self.last_graph_alpha = 1.0
            self.last_reliability_mean = 1.0
        score = scatter_softmax(score, src, dim=-1)
        att = torch.sparse_coo_tensor(edge_index, score, size=(x.shape[0], x.shape[0])).to(x.device)
        ass = torch.matmul(att, ass)   # (N_k, N_{k-1})
        ass = gumbel_softmax(torch.log(ass + 1e-6), temperature=self.temperature)
        return ass


class LSENetLayer(nn.Module):
    def __init__(self, manifold, in_dim, hid_dim, num_assign, dropout,
                 bias=False, use_att=False, nonlin=None, temperature=0.2,
                 edge_variant='V1', edge_fusion_gamma=1.0, edge_confidence_quantile=0.0,
                 edge_adaptive_alpha=False, edge_adaptive_alpha_strength=2.0,
                 edge_adaptive_alpha_bias=0.0, edge_reliability_temp=1.0):
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
                                          edge_reliability_temp=edge_reliability_temp)

    def forward(self, x, adj):
        # x = self.embeder(x, adj)
        ass = self.assigner(x, adj)
        support_t = ass.t() @ x
        denorm = (-self.manifold.inner(None, support_t, keepdim=True))
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        x_par = support_t / denorm

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
        beta = self.beta  # ensure |beta| < 1
        beta_norm_sq = (beta ** 2).sum()
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
