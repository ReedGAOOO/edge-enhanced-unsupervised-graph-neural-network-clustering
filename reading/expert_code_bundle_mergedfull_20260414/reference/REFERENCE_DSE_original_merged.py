"""
Merged original reference code from git HEAD.

Source repo section: reference/DSE_clustering-main
Generated for review/reading only. This file is not intended to be executed directly.
It preserves the original committed reference code before local faithful-baseline patches.

Included files:
  - reference/DSE_clustering-main/main.py
  - reference/DSE_clustering-main/exp.py
  - reference/DSE_clustering-main/data.py
  - reference/DSE_clustering-main/logger.py
  - reference/DSE_clustering-main/manifold/lorentz.py
  - reference/DSE_clustering-main/manifold/poincare.py
  - reference/DSE_clustering-main/modules/model.py
  - reference/DSE_clustering-main/modules/layers.py
  - reference/DSE_clustering-main/modules/dsi.py
  - reference/DSE_clustering-main/utils/decode.py
  - reference/DSE_clustering-main/utils/eval_utils.py
  - reference/DSE_clustering-main/utils/model_utils.py
  - reference/DSE_clustering-main/utils/plot_utils.py
  - reference/DSE_clustering-main/utils/train_utils.py
"""


# ========================================================================================
# FILE: reference/DSE_clustering-main/main.py
# ========================================================================================

import torch
import numpy as np
import os
import random
import argparse
from exp import Exp
from logger import create_logger
import json
from utils.train_utils import DotDict


seed = 3047
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser(description='Lorentz Structural Entropy')

# Experiment settings
parser.add_argument('--dataset', type=str, default='KarateClub')
parser.add_argument('--task', type=str, default='Clustering',
                    choices=['Clustering'])
parser.add_argument('--root_path', type=str, default='datasets')
parser.add_argument('--eval_freq', type=int, default=10)
parser.add_argument('--exp_iters', type=int, default=5)
parser.add_argument('--version', type=str, default="run")
parser.add_argument('--log_path', type=str, default="./results/FootBall.log")

parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--w_decay', type=float, default=1e-2)
parser.add_argument('--max_nums', type=int, nargs='+', default=[4], help="such as [50, 10]")
parser.add_argument('--hid_dim', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--nonlin', type=str, default="leaky_relu")
parser.add_argument('--temperature', type=float, default=0.9)
parser.add_argument('--n_cluster_trials', type=int, default=3)
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--knn', type=int, default=8)
parser.add_argument("--epsInt", type=int, default=8)

parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--save_path', type=str, default='model.pt')

# GPU
parser.add_argument('--use_gpu', action='store_false', help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--devices', type=str, default='0,1',
                    help='device ids of multiple gpus')


configs = parser.parse_args()
# with open(f'./configs/{configs.dataset}.json', 'wt') as f:
#     json.dump(vars(configs), f, indent=4)

# configs_dict = vars(configs)
# with open(f'./configs/{configs.dataset}.json', 'rt') as f:
#     configs_dict.update(json.load(f))
# configs = DotDict(configs_dict)
# f.close()

log_path = f"./results/{configs.version}/{configs.dataset}.log"
configs.log_path = log_path
if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')
if not os.path.exists(f"./results"):
    os.mkdir("./results")
if not os.path.exists(f"./results/{configs.dataset}"):
    os.mkdir(f"./results/{configs.dataset}")
if not os.path.exists(f"./results/{configs.version}"):
    os.mkdir(f"./results/{configs.version}")
print(f"Log path: {configs.log_path}")
logger = create_logger(configs.log_path)
logger.info(configs)

exp = Exp(configs)
exp.train()
torch.cuda.empty_cache()


# ========================================================================================
# FILE: reference/DSE_clustering-main/exp.py
# ========================================================================================

import torch
import numpy as np
from modules.dsi import DSI
from geoopt.optim import RiemannianAdam
from utils.eval_utils import cluster_metrics
from data import load_data
from logger import create_logger
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.optim import AdamW
import math


class Exp:
    def __init__(self, configs):
        self.configs = configs
        if self.configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

    def train(self):
        logger = create_logger(self.configs.log_path)
        device = self.device
        data = load_data(self.configs).to(device)

        total_nmi = []
        total_ari = []
        for exp_iter in range(self.configs.exp_iters):
            logger.info(f"\ntrain iters {exp_iter}")
            model = DSI(in_dim=data.x.shape[1],
                        hid_dim=self.configs.hid_dim,
                        num_nodes=data.x.shape[0],
                        temperature=self.configs.temperature,
                        dropout=self.configs.dropout,
                        nonlin_str=self.configs.nonlin,
                        max_nums=self.configs.max_nums,
                        alpha=self.configs.alpha,
                        knn=self.configs.knn).to(device)
            optimizer = AdamW(model.parameters(), lr=self.configs.lr, weight_decay=self.configs.w_decay)
            if self.configs.task == 'Clustering':
                nmi, ari = self.train_clu(data, model, optimizer, logger)
                total_nmi.append(nmi)
                total_ari.append(ari)

        if self.configs.task == 'Clustering':
            logger.info(f"NMI: {np.mean(total_nmi)}+-{np.std(total_nmi)}, "
                        f"ARI: {np.mean(total_ari)}+-{np.std(total_ari)}")

    def train_clu(self, data, model, optimizer, logger):
        best_cluster_result = {}
        best_cluster = {'nmi': 0, 'ari': 0}

        logger.info("--------------------------Training Start-------------------------")
        n_cluster_trials = self.configs.n_cluster_trials
        epoch_acc = []
        epoch_nmi = []
        epoch_ari = []

        for epoch in range(1, self.configs.epochs + 1):
            model.train()

            loss = model.se_loss(data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.info(f"[Stage2] Epoch {epoch}: loss={loss.item():.4f}")

            if epoch % self.configs.eval_freq == 0:
                logger.info("-----------------------Evaluation Start---------------------")
                model.eval()
                embed_dict, clu_mat_dict = model.get_cluster_results(data)
                predicts = model.fix_cluster_results(clu_mat_dict[1], embed_dict, self.configs.epsInt).cpu().numpy()
                trues = data.y.cpu().numpy()
                acc, nmi, ari = [], [], []
                for step in range(n_cluster_trials):
                    metrics = cluster_metrics(trues, predicts)
                    acc_, nmi_, ari_ = metrics.evaluateFromLabel(use_acc=True)
                    acc.append(acc_)
                    nmi.append(nmi_)
                    ari.append(ari_)
                acc, nmi, ari = np.mean(acc), np.mean(nmi), np.mean(ari)

                epoch_acc.append(acc)
                epoch_nmi.append(nmi)
                epoch_ari.append(ari)

                if nmi > best_cluster['nmi']:
                    best_cluster['nmi'] = nmi
                    best_cluster_result['nmi'] = [nmi, ari]
                    logger.info('------------------Saving best model-------------------')
                    torch.save(model.state_dict(), f"./checkpoints/{self.configs.save_path}")
                logger.info(
                    f"Epoch {epoch}: ACC: {acc * 100: .2f}, NMI: {nmi * 100: .2f}, ARI: {ari * 100: .2f}")
                logger.info(
                    "-------------------------------------------------------------------------")

        for k, result in best_cluster_result.items():
            nmi, ari = result
            logger.info(
                f"Best Results according to {k}: ACC: {acc * 100: .2f}, NMI: {nmi * 100: .2f}, ARI: {ari * 100: .2f} \n")
        return best_cluster['nmi'], best_cluster["ari"]

# ========================================================================================
# FILE: reference/DSE_clustering-main/data.py
# ========================================================================================

import torch
import networkx as nx
from torch_geometric.data import Dataset, Data
from torch_geometric.data.data import BaseData
from torch_geometric.datasets import (Amazon, KarateClub, Planetoid, WebKB)
from torch_geometric.utils import from_networkx
import urllib.request
import io
import zipfile
import numpy as np
from utils.model_utils import normalize_adj, adjacency2index


def load_data(configs):
    dataset = None
    if configs.dataset in ["computers", "Photo"]:
        dataset = Amazon(configs.root_path, name=configs.dataset)
    elif configs.dataset in ['Cora', 'Citeseer', 'PubMed']:
        dataset = Planetoid(configs.root_path, name=configs.dataset)
    elif configs.dataset == 'KarateClub':
        dataset = KarateClub()
    elif configs.dataset == 'FootBall':
        dataset = Football()
    elif configs.dataset in ['eat', 'bat', 'uat']:
        dataset = ATsDataset(root=configs.root_path, name=configs.dataset)
    elif configs.dataset in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(root=configs.root_path, name=configs.dataset)
    data = dataset[0].clone()
    N = data.x.shape[0]
    data.adj = torch.sparse_coo_tensor(indices=data.edge_index,
                                       values=torch.ones(data.edge_index.shape[1]),
                                       size=(N, N))
    data.adj = normalize_adj(data.adj, sparse=True)
    data.num_classes = data.y.max().item()
    return data


class Football(Dataset):
    """
    Refer to https://networkx.org/documentation/stable/auto_examples/graph/plot_football.html
    """
    def __init__(self):
        super().__init__()
        url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"

        sock = urllib.request.urlopen(url)  # open URL
        s = io.BytesIO(sock.read())  # read into BytesIO "file"
        sock.close()

        zf = zipfile.ZipFile(s)  # zipfile object
        txt = zf.read("football.txt").decode()  # read info file
        gml = zf.read("football.gml").decode()  # read gml data
        # throw away bogus first line with # from mejn files
        gml = gml.split("\n")[1:]
        graph = nx.parse_gml(gml)  # parse gml data

        data = from_networkx(graph)
        data.x = torch.eye(data.num_nodes)
        data.y = torch.tensor(data.value.tolist()).long()
        self.data = data

    def len(self) -> int:
        return 1

    def get(self, idx: int) -> BaseData:
        return self.data

    @property
    def num_node_features(self) -> int:
        return self.data.num_nodes

    @property
    def num_features(self) -> int:
        return self.data.num_nodes

    @property
    def num_classes(self) -> int:
        return len(np.unique(self.data.y))


class ATsDataset(Dataset):
    def __init__(self, root, name='eat'):
        super().__init__(root)
        adj = np.load(f'{root}/{name}/{name}_adj.npy')
        feat = np.load(f'{root}/{name}/{name}_feat.npy')
        label = np.load(f'{root}/{name}/{name}_label.npy')

        self.num_nodes = feat.shape[0]
        x = torch.tensor(feat).float()
        y = list(label)
        edge_index = adjacency2index(torch.tensor(adj))
        data = Data(x=x, edge_index=edge_index, y=y)
        self.data = data

    def len(self) -> int:
        return 1

    def get(self, idx: int) -> BaseData:
        return self.data

    @property
    def num_node_features(self) -> int:
        return self.data.x.shape[1]

    @property
    def num_features(self) -> int:
        return self.data.x.shape[1]

    @property
    def num_classes(self) -> int:
        return len(np.unique(self.data.y))


# ========================================================================================
# FILE: reference/DSE_clustering-main/logger.py
# ========================================================================================

import logging
import time
from datetime import timedelta


def create_logger(filepath, colored=False, debug=False):
    log_formatter = LogFormatter(colored=colored)

    # create file handler and set level
    if filepath is not None:
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.INFO)
        if debug:
            file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger


class LogFormatter:
    def __init__(self, colored=False):
        self.colored = colored
        self.start_time = time.time()

    def format(self, record):
        BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
        RESET_SEQ = "\033[0m"
        COLOR_SEQ = "\033[1;%dm"

        COLORS = {
            'WARNING': GREEN,
            'INFO': WHITE,
            'DEBUG': BLUE,
            'CRITICAL': YELLOW,
            'ERROR': RED
        }
        elapsed_seconds = round(record.created - self.start_time)
        levelname = record.levelname
        if self.colored:
            levelname = COLOR_SEQ % (
                30 + COLORS[record.levelname]) + record.levelname + RESET_SEQ

        prefix = "%s - %s - %s" % (
            levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''

# ========================================================================================
# FILE: reference/DSE_clustering-main/manifold/lorentz.py
# ========================================================================================

import geoopt
import torch


class Lorentz(geoopt.Lorentz):
    def __init__(self, k=1.0, learnable=False):
        super(Lorentz, self).__init__(k, learnable)

    def cinner(self, x, y):
        x = x.clone()
        x.narrow(-1, 0, 1).mul_(-1)
        return x @ y.transpose(-1, -2)

    def to_poincare(self, x, dim=-1):
        dn = x.size(dim) - 1
        return x.narrow(dim, 1, dn) / (x.narrow(dim, 0, 1) + torch.sqrt(self.k))

    def from_poincare(self, x, dim=-1, eps=1e-6):
        x_norm_square = torch.sum(x * x, dim=dim, keepdim=True)
        res = (
                torch.sqrt(self.k)
                * torch.cat((1 + x_norm_square, 2 * x), dim=dim)
                / (1.0 - x_norm_square + eps)
        )
        return res

    def frechet_mean(self, x, weights=None, keepdim=False):
        if weights is None:
            z = torch.sum(x, dim=0, keepdim=True)
        else:
            z = torch.sum(x * weights, dim=0, keepdim=keepdim)
        denorm = self.inner(None, z, keepdim=keepdim)
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        z = z / denorm
        return z

# ========================================================================================
# FILE: reference/DSE_clustering-main/manifold/poincare.py
# ========================================================================================

import geoopt
import torch
import geoopt.manifolds.lorentz.math as lmath


class Poincare(geoopt.PoincareBall):
    def __init__(self, c=0.8, learnable=False):
        super(Poincare, self).__init__(c=c, learnable=learnable)

    def from_lorentz(self, x, dim=-1):
        x = x.to(self.c.device)
        dn = x.size(dim) - 1
        return x.narrow(dim, 1, dn) / (x.narrow(dim, 0, 1) + torch.sqrt(self.c))

    def to_lorentz(self, x, dim=-1, eps=1e-6):
        x = x.to(self.c.device)
        x_norm_square = torch.sum(x * x, dim=dim, keepdim=True)
        res = (
                torch.sqrt(self.c)
                * torch.cat((1 + x_norm_square, 2 * x), dim=dim)
                / (1.0 - x_norm_square + eps)
        )
        return res

    def frechet_mean(self, embeddings, weights=None, keepdim=False):
        z = self.to_lorentz(embeddings)
        if weights is None:
            z = torch.sum(z, dim=0, keepdim=True)
        else:
            z = torch.sum(z * weights, dim=0, keepdim=keepdim)
        denorm = lmath.inner(z, z, keepdim=keepdim)
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        z = z / denorm
        z = self.from_lorentz(z).to(embeddings.device)
        return z

# ========================================================================================
# FILE: reference/DSE_clustering-main/modules/model.py
# ========================================================================================

import torch
import torch.nn as nn
from modules.layers import LSENetLayer, LorentzGraphConvolution
from utils.model_utils import select_activation


class LSENet(nn.Module):
    def __init__(self, manifold, in_dim, hid_dim, max_nums,
                 temperature=0.2, dropout=0.5, nonlin_str='relu'):
        super(LSENet, self).__init__()
        assert max_nums is not None
        self.manifold = manifold
        self.max_nums = max_nums  # [N_{H-1}, ..., N_1]
        self.height = len(max_nums) + 1

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
                nonlin=select_activation(nonlin_str)
            ))
            curr_dim = hid_dim + 1  # parent embedding dim

    def embed_leaf(self, x, adj):
        # Map raw features to Lorentz leaf embedding
        o = torch.zeros_like(x[:, :1])                # (N, 1)
        x = torch.cat([o, x], dim=1)           # (N, d+1)
        x = self.manifold.expmap0(x)                  # project to Lorentz
        x = self.input_proj(x, adj)  # (N, d + 1)
        x = self.input_proj2(x, adj)
        return x

    def forward(self, x, adj):
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

        for i, layer in enumerate(self.layers):
            z_par, adj_par, ass, z_curr = layer(current_z, current_adj)

            level_curr = self.height - i
            level_par = self.height - i - 1
            tree_coord_dict[level_par] = z_par
            ass_dict[level_curr] = ass
            adj_dict[level_par] = adj_par

            current_z = z_par
            current_adj = adj_par

        # Root (level 0) is Frechet mean of level 1
        root = self.manifold.frechet_mean(current_z)
        tree_coord_dict[0] = root
        ass_dict[1] = torch.ones(current_z.size(0), 1, device=x.device)

        return tree_coord_dict, ass_dict, adj_dict

# ========================================================================================
# FILE: reference/DSE_clustering-main/modules/layers.py
# ========================================================================================

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
                 bias=False, temperature=0.2):
        super(LorentzAssignment, self).__init__()
        self.manifold = manifold
        self.num_assign = num_assign
        self.assign_linear = nn.Linear(in_dim, num_assign, bias=bias)
        nn.init.xavier_normal_(self.assign_linear.weight)
        self.temperature = temperature
        self.key_linear = LorentzLinear(manifold, in_dim, hid_dim, bias=False)
        self.query_linear = LorentzLinear(manifold, in_dim, hid_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        ass = self.assign_linear(self.manifold.logmap0(x)).softmax(-1)
        q = self.query_linear(x)
        k = self.key_linear(x)
        edge_index = adj.coalesce().indices()
        src, dst = edge_index[0], edge_index[1]
        score = self.manifold.dist(q[src], k[dst])
        score = scatter_softmax(-score, src, dim=-1)
        att = torch.sparse_coo_tensor(edge_index, score, size=(x.shape[0], x.shape[0])).to(x.device)
        ass = torch.matmul(att, ass)   # (N_k, N_{k-1})
        ass = gumbel_softmax(torch.log(ass + 1e-6), temperature=self.temperature)
        return ass


class LSENetLayer(nn.Module):
    def __init__(self, manifold, in_dim, hid_dim, num_assign, dropout,
                 bias=False, use_att=False, nonlin=None, temperature=0.2):
        super(LSENetLayer, self).__init__()
        self.manifold = manifold
        # self.embeder = LorentzGraphConvolution(manifold, in_dim, hid_dim,
        #                                        True, dropout, use_att, nonlin)
        self.assigner = LorentzAssignment(manifold, hid_dim,
                                          hid_dim, num_assign,
                                          dropout, bias, temperature)

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

# ========================================================================================
# FILE: reference/DSE_clustering-main/modules/dsi.py
# ========================================================================================

import torch
import torch.nn as nn
from utils.model_utils import gumbel_softmax, graph_top_K
from manifold.lorentz import Lorentz
from modules.layers import LorentzBoost
from modules.model import LSENet

MIN_NORM = 1e-15
EPS = 1e-6


class DSI(nn.Module):
    def __init__(self, in_dim, hid_dim, num_nodes, max_nums, temperature=0.2,
                 dropout=0.5, nonlin_str='relu', tau=1.0, alpha=0.01, knn=8):
        super(DSI, self).__init__()
        self.num_nodes = num_nodes
        self.height = len(max_nums) + 1
        self.manifold = Lorentz()
        self.encoder = LSENet(self.manifold, in_dim, hid_dim, max_nums, temperature, dropout, nonlin_str)
        self.lorentz_proj = LorentzBoost(hid_dim + 1)
        self.temperature = temperature
        self.tau = tau
        self.alpha = alpha
        self.knn = knn

    def forward(self, data):
        features = data.x
        adj = data.adj.clone()
        tree_coord_dict, ass_dict, adj_dict = self.encoder(features, adj)
        return tree_coord_dict, ass_dict, adj_dict

    def get_cluster_results(self, data):
        features = data.x
        adj = data.adj.clone()
        coord_dict, ass_dict, _ = self.encoder(features, adj)
        embed_dict = {}
        for height, x in coord_dict.items():
            embed_dict[height] = x.detach()
        clu_mat_dict = {self.height: torch.eye(self.num_nodes).to(data.x.device)}
        for k in range(self.height - 1, 0, -1):
            clu_mat_dict[k] = clu_mat_dict[k + 1] @ ass_dict[k + 1]
        for k, v in clu_mat_dict.items():
            idx = v.max(1)[1]
            t = torch.zeros_like(v)
            t[torch.arange(t.shape[0]), idx] = 1.
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
        z_leaf = self.encoder.embed_leaf(data.x, data.adj)
        z_leaf = self.lorentz_proj(z_leaf)
        neg_dist2 = 2 + 2 * self.manifold.cinner(z_leaf, z_leaf)
        adj_aug = graph_top_K(torch.softmax(neg_dist2/ self.tau, dim=-1), k=self.knn)
        tree_coord_aug_dict, ass_aug_dict, adj_aug_dict = self.encoder(data.x, self.alpha * adj_aug + data.adj)
        loss = self._si_loss(ass_aug_dict, adj_aug_dict, eps)
        return loss

    def _si_loss(self, ass_dict: dict, adj_dict: dict, eps: float = 1e-6):
        se_loss = 0
        vol_G = adj_dict[self.height].sum()

        for k in range(self.height, 0, -1):
            adj_dense = adj_dict[k].to_dense()
            degree = adj_dense.sum(dim=1)
            diag = adj_dense.diag()
            if k == 1:
                vol_parent = vol_G
            else:
                vol_parent = adj_dict[k - 1].to_dense().sum(dim=-1)
                vol_parent = torch.einsum('ij, j->i', ass_dict[k], vol_parent)
            delta_vol = degree - diag
            log_vol_ratio_k = torch.log2((degree + eps) / (vol_parent + eps))
            se_loss += torch.sum(delta_vol * log_vol_ratio_k)
        se_loss = -1 / vol_G * se_loss
        return se_loss

# ========================================================================================
# FILE: reference/DSE_clustering-main/utils/decode.py
# ========================================================================================

import torch
import networkx as nx
from queue import Queue


class Node:
    def __init__(self, index: list, embeddings: torch.Tensor, coords=None,
                 tree_index=None, is_leaf=False, height: int = None):
        self.index = index  # T_alpha
        self.embeddings = embeddings  # coordinates of nodes in T_alpha
        self.children = []
        self.coords = coords  # node coordinates
        self.tree_index = tree_index
        self.is_leaf = is_leaf
        self.height = height


def construct_tree(nodes_list: torch.LongTensor, coords_list: dict,
                   ass_list: dict, height, num_nodes):
    nodes_count = num_nodes
    que = Queue()
    root = Node(nodes_list, coords_list[height][nodes_list].cpu(),
                coords=coords_list[0].cpu(), tree_index=nodes_count, height=0)
    que.put(root)

    while not que.empty():
        node = que.get()
        L_nodes = node.index
        k = node.height + 1
        if k == height:
            for i in L_nodes:
                node.children.append(Node(i.reshape(-1), coords_list[height][i].cpu(), coords=coords_list[k][i].cpu(),
                                          tree_index=i.item(), is_leaf=True, height=k))
        else:
            temp_ass = ass_list[k][L_nodes].cpu()
            for j in range(temp_ass.shape[-1]):
                temp_child = L_nodes[temp_ass[:, j].nonzero().flatten()]
                if len(temp_child) > 0:
                    nodes_count += 1
                    child_node = Node(temp_child, coords_list[height][temp_child].cpu(),
                                      coords=coords_list[k][j].cpu(),
                                      tree_index=nodes_count, height=k)
                    node.children.append(child_node)
                    que.put(child_node)
    return root


def to_networkx_tree(root: Node, manifold, height):
    edges_list = []
    nodes_list = []
    que = Queue()
    que.put(root)
    nodes_list.append(
        (
            root.tree_index,
            {'coords': root.coords.reshape(-1),
             'is_leaf': root.is_leaf,
             'children': root.index,
             'height': root.height}
        )
    )

    while not que.empty():
        cur_node = que.get()
        if cur_node.height == height:
            break
        for node in cur_node.children:
            nodes_list.append(
                (
                    node.tree_index,
                    {'coords': node.coords.reshape(-1),
                     'is_leaf': node.is_leaf,
                     'children': node.index,
                     'height': node.height}
                )
            )
            edges_list.append(
                (
                    cur_node.tree_index,
                    node.tree_index,
                    {'weight': torch.sigmoid(1. - manifold.dist(cur_node.coords, node.coords)).item()}
                )
            )
            que.put(node)

    graph = nx.Graph()
    graph.add_nodes_from(nodes_list)
    graph.add_edges_from(edges_list)
    return graph


# ========================================================================================
# FILE: reference/DSE_clustering-main/utils/eval_utils.py
# ========================================================================================

import numpy as np
import torch
from sklearn import metrics
from munkres import Munkres
import networkx as nx


def decoding_cluster_from_tree(manifold, tree: nx.Graph, num_clusters, num_nodes, height):
    root = tree.nodes[num_nodes]
    root_coords = root['coords']
    dist_dict = {}  # for every height of tree
    for u in tree.nodes():
        if u != num_nodes:  # u is not root
            h = tree.nodes[u]['height']
            dist_dict[h] = dist_dict.get(h, {})
            dist_dict[h].update({u: manifold.dist(root_coords, tree.nodes[u]['coords']).numpy()})

    h = 1
    sorted_dist_list = sorted(dist_dict[h].items(), reverse=False, key=lambda x: x[1])
    count = len(sorted_dist_list)
    group_list = [([u], dist) for u, dist in sorted_dist_list]  # [ ([u], dist_u) ]
    while len(group_list) <= 1:
        h = h + 1
        sorted_dist_list = sorted(dist_dict[h].items(), reverse=False, key=lambda x: x[1])
        count = len(sorted_dist_list)
        group_list = [([u], dist) for u, dist in sorted_dist_list]

    while count > num_clusters:
        group_list, count = merge_nodes_once(manifold, root_coords, tree, group_list, count)

    while count < num_clusters and h <= height:
        h = h + 1   # search next level
        pos = 0
        while pos < len(group_list):
            v1, d1 = group_list[pos]  # node to split
            sub_level_set = []
            v1_coord = tree.nodes[v1[0]]['coords']
            for u, v in tree.edges(v1[0]):
                if tree.nodes[v]['height'] == h:
                    v_coords = tree.nodes[v]['coords']
                    dist = manifold.dist(v_coords, v1_coord).cpu().numpy()
                    sub_level_set.append(([v], dist))    # [ ([v], dist_v) ]
            if len(sub_level_set) <= 1:
                pos += 1
                continue
            sub_level_set = sorted(sub_level_set, reverse=False, key=lambda x: x[1])
            count += len(sub_level_set) - 1
            if count > num_clusters:
                while count > num_clusters:
                    sub_level_set, count = merge_nodes_once(manifold, v1_coord, tree, sub_level_set, count)
                del group_list[pos]  # del the position node which will be split
                group_list += sub_level_set    # Now count == num_clusters
                break
            elif count == num_clusters:
                del group_list[pos]  # del the position node which will be split
                group_list += sub_level_set
                break
            else:
                del group_list[pos]
                group_list += sub_level_set
                pos += 1

    cluster_dist = {}
    for i in range(len(group_list)):
        u_list, _ = group_list[i]
        group = []
        for u in u_list:
            index = tree.nodes[u]['children'].tolist()
            group += index
        cluster_dist.update({k: i for k in group})
    results = sorted(cluster_dist.items(), key=lambda x: x[0])
    results = np.array([x[1] for x in results])
    return results


def merge_nodes_once(manifold, root_coords, tree, group_list, count):
    # group_list should be ordered ascend
    v1, v2 = group_list[-1], group_list[-2]
    merged_node = v1[0] + v2[0]
    merged_coords = torch.stack([tree.nodes[v]['coords'] for v in merged_node], dim=0)
    merged_point = manifold.frechet_mean(merged_coords)
    merged_dist = manifold.dist(merged_point, root_coords).cpu().numpy()
    merged_item = (merged_node, merged_dist)
    del group_list[-2:]
    group_list.append(merged_item)
    group_list = sorted(group_list, reverse=False, key=lambda x: x[1])
    count -= 1
    return group_list, count


class cluster_metrics:
    def __init__(self, trues, predicts):
        self.true_label = trues
        self.pred_label = predicts

    def clusterAcc(self):
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        from sklearn import metrics

        true_label = np.array(self.true_label)
        pred_label = np.array(self.pred_label)

        l1 = list(set(true_label))
        l2 = list(set(pred_label))
        numclass1 = len(l1)
        numclass2 = len(l2)

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            for j, c2 in enumerate(l2):
                cost[i, j] = np.sum((true_label == c1) & (pred_label == c2))

        row_ind, col_ind = linear_sum_assignment(-cost)

        pred_to_true = {}
        for r, c in zip(row_ind, col_ind):
            pred_to_true[l2[c]] = l1[r]

        if len(pred_to_true) < len(l2):
            fallback_class = true_label[np.bincount(true_label).argmax()]
            for c2 in l2:
                if c2 not in pred_to_true:
                    pred_to_true[c2] = fallback_class

        new_predict = np.array([pred_to_true[pred] for pred in pred_label])

        acc = metrics.accuracy_score(true_label, new_predict)

        return acc

    def evaluateFromLabel(self, use_acc=False):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        if use_acc:
            acc = self.clusterAcc()
            return acc, nmi, adjscore
        else:
            return nmi, adjscore


def cal_AUC_AP(scores, trues):
    auc = metrics.roc_auc_score(trues, scores)
    ap = metrics.average_precision_score(trues, scores)
    return auc, ap

# ========================================================================================
# FILE: reference/DSE_clustering-main/utils/model_utils.py
# ========================================================================================

import torch
import torch.nn.functional as F


def select_activation(activation):
    if activation == 'elu':
        return F.elu
    elif activation == 'relu':
        return F.relu
    elif activation == 'sigmoid':
        return F.sigmoid
    elif activation == 'tanh':
        return F.tanh
    elif activation == 'leaky_relu':
        return F.leaky_relu
    elif activation == "gelu":
        return F.gelu
    elif activation is None:
        return None
    else:
        raise NotImplementedError('the non_linear_function is not implemented')


def frechet_mean_poincare(manifold, embeddings, weights=None, keepdim=False):
    z = manifold.from_poincare(embeddings)
    if weights is None:
        z = torch.sum(z, dim=0, keepdim=True)
    else:
        z = torch.sum(z * weights, dim=0, keepdim=keepdim)
    denorm = manifold.inner(None, z, keepdim=keepdim)
    denorm = denorm.abs().clamp_min(1e-8).sqrt()
    z = z / denorm
    z = manifold.to_poincare(z).to(embeddings.device)
    return z


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size()).to(logits.device)
    return torch.nn.functional.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=0.2, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


def gumbel_sigmoid(logits, tau: float = 1, hard: bool = False, threshold: float = 0.5):
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0, 1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight through.
        indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def graph_top_K(dense_adj, k):
    assert k < dense_adj.shape[-1]
    _, indices = dense_adj.topk(k=k + 1, dim=-1)
    mask = torch.zeros(dense_adj.shape).bool().to(dense_adj.device)
    mask[torch.arange(dense_adj.shape[0])[:, None], indices] = True
    mask[torch.arange(dense_adj.shape[0]), torch.arange(dense_adj.shape[0])] = False
    sparse_adj = torch.masked_fill(dense_adj, ~mask, value=0.).to_sparse_coo()
    return sparse_adj


def adjacency2index(adjacency, weight=False, topk=False, k=10):
    """_summary_

    Args:
        adjacency (torch.tensor): [N, N] matrix
    return:
        edge_index: [2, E]
        edge_weight: optional
    """
    if topk and k:
        adj = graph_top_K(adjacency, k)
    else:
        adj = adjacency
    edge_index = torch.nonzero(adj).t().contiguous()
    if weight:
        weight = adjacency[edge_index[0], edge_index[1]].reshape(-1)
        return edge_index, weight

    else:
        return edge_index


def index2adjacency(N, edge_index, weight=None, is_sparse=True):
    adjacency = torch.zeros(N, N).to(edge_index.device)
    m = edge_index.shape[1]
    if weight is None:
        adjacency[edge_index[0], edge_index[1]] = 1
    else:
        adjacency[edge_index[0], edge_index[1]] = weight.reshape(-1)
    adjacency = normalize_adj(adjacency)
    if is_sparse:
        weight = adjacency[edge_index[0], edge_index[1]]
        adjacency = torch.sparse_coo_tensor(indices=edge_index, values=weight, size=(N, N))
    return adjacency


def normalize_adj(adj, sparse=True):
    if sparse:
        adj = adj.coalesce()
        row_sum = adj.sum(dim=1).to_dense()
        deg_inv_sqrt = row_sum.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm_adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    else:
        row_sum = adj.sum(dim=1)
        deg_inv_sqrt = row_sum.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm_adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

    return norm_adj


def givens_rot_mat(i, j, theta: torch.Tensor, n):
    assert 0 <= i <= n - 1 and 0 <= j <= n - 1, "Invalid rotation axis"
    if isinstance(theta, float):
        theta = torch.tensor([theta])
    G = torch.eye(n).to(theta.device)
    c = torch.cos(theta)
    s = torch.sin(theta)
    G[i, i] = c
    G[j, j] = c
    G[i, j] = -s
    G[j, i] = s
    return G

# ========================================================================================
# FILE: reference/DSE_clustering-main/utils/plot_utils.py
# ========================================================================================

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE

def mobius_add(x, y):
    """Mobius addition in numpy."""
    xy = np.sum(x * y, 1, keepdims=True)
    x2 = np.sum(x * x, 1, keepdims=True)
    y2 = np.sum(y * y, 1, keepdims=True)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    den = 1 + 2 * xy + x2 * y2
    return num / den


def mobius_mul(x, t):
    """Mobius multiplication in numpy."""
    normx = np.sqrt(np.sum(x * x, 1, keepdims=True))
    return np.tanh(t * np.arctanh(normx)) * x / normx


def geodesic_fn(x, y, nb_points=100):
    """Get coordinates of points on the geodesic between x and y."""
    t = np.linspace(0, 1, nb_points)
    x_rep = np.repeat(x.reshape((1, -1)), len(t), 0)
    y_rep = np.repeat(y.reshape((1, -1)), len(t), 0)
    t1 = mobius_add(-x_rep, y_rep)
    t2 = mobius_mul(t1, t.reshape((-1, 1)))
    return mobius_add(x_rep, t2)


def plot_geodesic(x, y, ax):
    """Plots geodesic between x and y."""
    points = geodesic_fn(x, y)
    ax.plot(points[:, 0], points[:, 1], color='black', linewidth=0.3, alpha=1.)


def plot_leaves(tree, manifold, embeddings, labels, height, save_path=None, colors_dict=None):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    circle = plt.Circle((0, 0), 1.0, color='y', alpha=0.1)
    ax.add_artist(circle)
    for k in range(1, height + 1):
        circle_k = plt.Circle((0, 0), k / (height + 1), color='b', alpha=0.05)
        ax.add_artist(circle_k)
    n = embeddings.shape[0]
    colors_dict = get_colors(labels, color_seed=1234) if colors_dict is None else colors_dict
    colors = [colors_dict[k] for k in labels]
    embeddings = manifold.to_poincare(embeddings).numpy()
    scatter = ax.scatter(embeddings[:n, 0], embeddings[:n, 1], c=colors, s=80, alpha=1.0)
    # legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    # ax.add_artist(legend)
    # ax.scatter(np.array([0]), np.array([0]), c='black')
    for u, v in tree.edges():
        x = manifold.to_poincare(tree.nodes[u]['coords']).numpy()
        y = manifold.to_poincare(tree.nodes[v]['coords']).numpy()
        if tree.nodes[u]['is_leaf'] is False:
            c = 'black' if tree.nodes[u]['height'] == 0 else 'red'
            m = '*' if tree.nodes[u]['height'] == 0 else 's'
            ax.scatter(x[0], x[1], c=c, s=30, marker=m)
        if tree.nodes[v]['is_leaf'] is False:
            c = 'black' if tree.nodes[v]['height'] == 0 else 'red'
            m = '*' if tree.nodes[u]['height'] == 0 else 's'
            ax.scatter(y[0], y[1], c=c, s=30, marker=m)
        plot_geodesic(y, x, ax)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")
    plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=500)
    plt.show()
    return ax, colors_dict


def get_colors(y, color_seed=1234):
    """random color assignment for label classes."""
    np.random.seed(color_seed)
    colors = {}
    for k in np.unique(y):
        r = np.random.random()
        b = np.random.random()
        g = np.random.random()
        colors[k] = (r, g, b)
    return colors


def plot_nx_graph(G: nx.Graph, root, save_path=None):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    pos = graphviz_layout(G, 'twopi')
    nx.draw(G, pos, ax=ax, with_labels=True)
    plt.savefig(save_path)
    plt.show()


def map_labels(true_labels, predicted_labels):
    D = np.zeros((np.unique(true_labels).size, np.unique(predicted_labels).size))
    for i, tl in enumerate(np.unique(true_labels)):
        for j, pl in enumerate(np.unique(predicted_labels)):
            D[i, j] = np.sum((true_labels == tl) & (predicted_labels == pl))

    row_ind, col_ind = linear_sum_assignment(-D)

    label_map = {}
    for i, j in zip(row_ind, col_ind):
        label_map[np.unique(true_labels)[i]] = np.unique(predicted_labels)[j]

    new_true_labels = np.vectorize(label_map.get)(true_labels)
    new_predicted_labels = predicted_labels

    return new_true_labels, new_predicted_labels


def plot_tsne(z, true_labels, predicted_labels, sample_num=7000, output_path="tsne_visualization.pdf"):
    """
    Visualize embedding by t-SNE with true and predicted labels using circles, and save the figures as PDF.
    """
    new_true_labels, new_predicted_labels = map_labels(true_labels, predicted_labels)

    sample_embeds = z
    sample_true_label = new_true_labels
    sample_predict_label = new_predicted_labels

    ts = TSNE(n_components=2, init='pca', random_state=0)
    ts_embeds = ts.fit_transform(sample_embeds)

    x_min, x_max = np.min(ts_embeds, 0), np.max(ts_embeds, 0)
    norm_ts_embeds = (ts_embeds - x_min) / (x_max - x_min)

    custom_colors = ['#ffff4e', '#329932', '#2a2aff', '#ff0f0f', '#8a148a', '#ffae1c', '#bf80bf', '#9a9a9a']

    all_labels = np.unique(np.concatenate([sample_true_label, sample_predict_label]))
    label_color_map = {label: custom_colors[i % len(custom_colors)] for i, label in enumerate(all_labels)}

    fig_true, ax_true = plt.subplots(figsize=(8, 8))
    for label in all_labels:
        class_mask = sample_true_label == label
        ax_true.scatter(norm_ts_embeds[class_mask, 0], norm_ts_embeds[class_mask, 1],
                        color=label_color_map[label], s=10, label=f'Class {label}')
    ax_true.set_title('t-SNE with True Labels', fontsize=14)
    ax_true.set_xticks([])
    ax_true.set_yticks([])
    ax_true.axis('off')

    fig_pred, ax_pred = plt.subplots(figsize=(8, 8))
    for label in all_labels:
        class_mask = sample_predict_label == label
        ax_pred.scatter(norm_ts_embeds[class_mask, 0], norm_ts_embeds[class_mask, 1],
                        color=label_color_map[label], s=10, label=f'Class {label}')
    ax_pred.set_title('t-SNE with Predicted Labels', fontsize=14)
    ax_pred.set_xticks([])
    ax_pred.set_yticks([])
    ax_pred.axis('off')

    fig_true.savefig(output_path.replace(".pdf", "_true.pdf"), bbox_inches='tight')
    fig_pred.savefig(output_path.replace(".pdf", "_pred.pdf"), bbox_inches='tight')
    plt.show()

    return fig_true, fig_pred



# ========================================================================================
# FILE: reference/DSE_clustering-main/utils/train_utils.py
# ========================================================================================

import os.path
import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')
        torch.save(model.state_dict(), f"./checkpoints/{path}")
        self.val_loss_min = val_loss


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        super().__init__()
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

