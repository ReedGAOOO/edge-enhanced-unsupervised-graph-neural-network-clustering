import torch
import networkx as nx
from torch_geometric.data import Dataset, Data
from torch_geometric.data.data import BaseData
from torch_geometric.datasets import (Amazon, KarateClub, Planetoid, WebKB, Coauthor)
from torch_geometric.utils import from_networkx
import urllib.request
import io
import zipfile
import numpy as np
import torch.nn.functional as F
from utils.model_utils import normalize_adj, adjacency2index


def build_structural_edge_weight(edge_index, num_nodes):
    src, dst = edge_index[0], edge_index[1]
    deg = torch.bincount(src, minlength=num_nodes).float()
    log_deg_src = torch.log1p(deg[src])
    log_deg_dst = torch.log1p(deg[dst])
    return torch.exp(-torch.abs(log_deg_src - log_deg_dst))


def build_feature_edge_weight(x, edge_index, temp=1.0):
    src, dst = edge_index[0], edge_index[1]
    temp = max(1e-6, float(temp))
    cos = F.cosine_similarity(x[src], x[dst], dim=1, eps=1e-8)
    return torch.sigmoid(cos / temp)


def load_data(configs):
    dataset = None
    name = configs.dataset
    name_l = name.lower()
    if name_l in ["computers", "photo"]:
        dataset = Amazon(configs.root_path, name="Computers" if name_l == "computers" else "Photo")
    elif name_l in ['cora', 'citeseer', 'pubmed']:
        planetoid_name = {'cora': 'Cora', 'citeseer': 'Citeseer', 'pubmed': 'PubMed'}[name_l]
        dataset = Planetoid(configs.root_path, name=planetoid_name)
    elif name_l in ['coauthorcs', 'coauthor_cs', 'cs']:
        dataset = Coauthor(root=f'{configs.root_path}/Coauthor', name='CS')
    elif name_l in ['coauthorphysics', 'coauthor_physics', 'physics']:
        dataset = Coauthor(root=f'{configs.root_path}/Coauthor', name='Physics')
    elif name == 'KarateClub':
        dataset = KarateClub()
    elif name == 'FootBall':
        dataset = Football()
    elif name in ['eat', 'bat', 'uat']:
        dataset = ATsDataset(root=configs.root_path, name=name)
    elif name in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(root=configs.root_path, name=name)
    data = dataset[0].clone()
    N = data.x.shape[0]
    variant = str(getattr(configs, 'edge_variant', 'V1')).upper()
    edge_index = data.edge_index
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)

    if variant in {'V2', 'V4', 'V5'}:
        w_struct = build_structural_edge_weight(edge_index, N)
    if variant in {'V3', 'V4', 'V5'}:
        w_feat = build_feature_edge_weight(data.x.float(), edge_index, temp=getattr(configs, 'edge_feat_temp', 1.0))

    if variant == 'V2':
        edge_weight = w_struct
    elif variant == 'V3':
        edge_weight = w_feat
    elif variant in {'V4', 'V5'}:
        alpha = float(getattr(configs, 'edge_hybrid_alpha', 0.5))
        alpha = max(0.0, min(1.0, alpha))
        edge_weight = alpha * w_feat + (1.0 - alpha) * w_struct

    edge_weight = edge_weight.clamp_min(1e-6)
    data.edge_weight = edge_weight
    data.adj = torch.sparse_coo_tensor(indices=edge_index,
                                       values=edge_weight,
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
