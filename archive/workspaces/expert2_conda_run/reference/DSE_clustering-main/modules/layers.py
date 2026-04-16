import torch
import torch.nn as nn
from torch_scatter import scatter_softmax
import math
from utils.model_utils import gumbel_softmax, graph_top_K, normalize_adj, givens_rot_mat


class LorentzGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_dim, out_dim, use_bias, dropout, use_att, nonlin=None, att_mode="legacy"):
        super(LorentzGraphConvolution, self).__init__()
        self.linear = LorentzLinear(manifold, in_dim, out_dim, use_bias, dropout, nonlin=nonlin)
        self.agg = LorentzAgg(manifold, out_dim, dropout, use_att, att_mode=att_mode)

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

    def __init__(self, manifold, in_dim, dropout, use_att, att_mode="legacy"):
        super(LorentzAgg, self).__init__()
        self.manifold = manifold

        self.in_features = in_dim
        self.dropout = dropout
        self.use_att = use_att
        self.att_mode = att_mode
        if self.use_att:
            self.key_linear = LorentzLinear(manifold, in_dim, in_dim)
            self.query_linear = LorentzLinear(manifold, in_dim, in_dim)
            if self.att_mode == "paper":
                self.att_proj = nn.Linear(2 * in_dim, 1, bias=False)
                self.leaky_relu = nn.LeakyReLU(0.2)
            else:
                self.bias = nn.Parameter(torch.zeros(()) + 20)
                self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(in_dim))

    def forward(self, x, adj):
        if self.use_att:
            query = self.query_linear(x)
            key = self.key_linear(x)
            if self.att_mode == "paper":
                edge_index = adj.coalesce().indices()
                src, dst = edge_index[0], edge_index[1]
                score = self.leaky_relu(self.att_proj(torch.cat([query[src], key[dst]], dim=-1))).squeeze(-1)
                score = scatter_softmax(score, src, dim=0)
                att_adj = torch.sparse_coo_tensor(edge_index, score, size=adj.shape).to(x.device)
                support_t = torch.sparse.mm(att_adj, x)
            else:
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
                 bias=False, temperature=0.2, att_mode="legacy", gumbel_assign=True):
        super(LorentzAssignment, self).__init__()
        self.manifold = manifold
        self.num_assign = num_assign
        self.assign_linear = nn.Linear(in_dim, num_assign, bias=bias)
        nn.init.xavier_normal_(self.assign_linear.weight)
        self.temperature = temperature
        self.key_linear = LorentzLinear(manifold, in_dim, hid_dim, bias=False)
        self.query_linear = LorentzLinear(manifold, in_dim, hid_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.att_mode = att_mode
        self.gumbel_assign = gumbel_assign
        if self.att_mode == "paper":
            self.att_proj = nn.Linear(2 * hid_dim, 1, bias=False)
            self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        ass = self.assign_linear(self.manifold.logmap0(x)).softmax(-1)
        q = self.query_linear(x)
        k = self.key_linear(x)
        edge_index = adj.coalesce().indices()
        src, dst = edge_index[0], edge_index[1]
        if self.att_mode == "paper":
            score = self.leaky_relu(self.att_proj(torch.cat([q[src], k[dst]], dim=-1))).squeeze(-1)
            score = scatter_softmax(score, src, dim=0)
        else:
            score = self.manifold.dist(q[src], k[dst])
            score = scatter_softmax(-score, src, dim=0)
        att = torch.sparse_coo_tensor(edge_index, score, size=(x.shape[0], x.shape[0])).to(x.device)
        ass = torch.sparse.mm(att, ass)   # (N_k, N_{k-1})
        if self.gumbel_assign:
            ass = gumbel_softmax(torch.log(ass + 1e-6), temperature=self.temperature)
        return ass


class LSENetLayer(nn.Module):
    def __init__(self, manifold, in_dim, hid_dim, num_assign, dropout,
                 bias=False, use_att=False, nonlin=None, temperature=0.2,
                 assign_att_mode="legacy", assign_gumbel=True):
        super(LSENetLayer, self).__init__()
        self.manifold = manifold
        # self.embeder = LorentzGraphConvolution(manifold, in_dim, hid_dim,
        #                                        True, dropout, use_att, nonlin)
        self.assigner = LorentzAssignment(manifold, hid_dim,
                                          hid_dim, num_assign,
                                          dropout, bias, temperature,
                                          att_mode=assign_att_mode,
                                          gumbel_assign=assign_gumbel)

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
