import torch
import networkx as nx
from pathlib import Path
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


def has_ats_style_files(root, name):
    base = Path(root) / name
    return (
        (base / f"{name}_adj.npy").exists()
        and (base / f"{name}_feat.npy").exists()
        and (base / f"{name}_label.npy").exists()
    )


def has_sparse_style_files(root, name):
    base = Path(root) / name
    return (
        (base / f"{name}_edge_index.npy").exists()
        and (base / f"{name}_feat.npy").exists()
        and (base / f"{name}_label.npy").exists()
    )


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
    elif name in ['eat', 'bat', 'uat'] or has_ats_style_files(configs.root_path, name) or has_sparse_style_files(configs.root_path, name):
        dataset = ATsDataset(root=configs.root_path, name=name)
    elif name in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(root=configs.root_path, name=name)
    if dataset is None:
        raise ValueError(
            f"Unsupported dataset '{name}'. Built-ins: "
            "cora/citeseer/pubmed/computers/photo/coauthorcs/coauthorphysics/KarateClub/FootBall/"
            "Cornell/Texas/Wisconsin, or custom ATS-style files under "
            f"'{configs.root_path}/{name}/' with dense files "
            f"({name}_adj.npy, {name}_feat.npy, {name}_label.npy) or sparse files "
            f"({name}_edge_index.npy, optional {name}_edge_weight.npy/{name}_edge_attr.npy, "
            f"{name}_feat.npy, {name}_label.npy)."
        )
    data = dataset[0].clone()
    N = data.x.shape[0]
    variant = str(getattr(configs, 'edge_variant', 'V1')).upper()
    edge_index = data.edge_index
    if hasattr(data, "edge_weight") and data.edge_weight is not None and data.edge_weight.numel() == edge_index.shape[1]:
        input_prior_weight = data.edge_weight.float()
    else:
        input_prior_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    edge_weight = input_prior_weight

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

    prior_alpha = float(getattr(configs, 'edge_input_prior_alpha', 0.0))
    prior_alpha = max(0.0, min(1.0, prior_alpha))
    if prior_alpha > 0.0 and variant in {'V2', 'V3', 'V4', 'V5'}:
        edge_weight = (1.0 - prior_alpha) * edge_weight + prior_alpha * input_prior_weight

    edge_weight = edge_weight.clamp_min(1e-6)
    data.edge_weight = edge_weight
    data.adj = torch.sparse_coo_tensor(indices=edge_index,
                                       values=edge_weight,
                                       size=(N, N))
    data.adj = normalize_adj(data.adj, sparse=True)
    valid_y = data.y[data.y >= 0]
    data.num_classes = int(valid_y.max().item() + 1) if valid_y.numel() > 0 else 1
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
        base = Path(root) / name
        feat_path = base / f"{name}_feat.npy"
        label_path = base / f"{name}_label.npy"
        dense_adj_path = base / f"{name}_adj.npy"
        sparse_edge_index_path = base / f"{name}_edge_index.npy"
        edge_weight_path = base / f"{name}_edge_weight.npy"
        edge_attr_path = base / f"{name}_edge_attr.npy"

        feat = np.load(feat_path)
        label = np.load(label_path)
        if feat.ndim != 2:
            raise ValueError(f"{name}_feat.npy must be [N, F], got {feat.shape}")
        if label.ndim != 1:
            raise ValueError(f"{name}_label.npy must be [N], got {label.shape}")
        if feat.shape[0] != label.shape[0]:
            raise ValueError(
                f"Node count mismatch in custom dataset '{name}': "
                f"feat={feat.shape[0]}, label={label.shape[0]}"
            )

        self.num_nodes = feat.shape[0]
        x = torch.tensor(feat).float()
        y = torch.tensor(label).long()

        if dense_adj_path.exists():
            adj = np.load(dense_adj_path)
            if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
                raise ValueError(f"{name}_adj.npy must be a square [N, N] matrix, got {adj.shape}")
            if adj.shape[0] != feat.shape[0]:
                raise ValueError(
                    f"Node count mismatch in dense custom dataset '{name}': "
                    f"adj={adj.shape[0]}, feat={feat.shape[0]}"
                )
            edge_index = adjacency2index(torch.tensor(adj))
            data = Data(x=x, edge_index=edge_index, y=y)
        elif sparse_edge_index_path.exists():
            edge_index = np.load(sparse_edge_index_path)
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError(f"{name}_edge_index.npy must be [2, E], got {edge_index.shape}")
            if edge_index.shape[1] == 0:
                raise ValueError(f"{name}_edge_index.npy has no edges.")
            if int(edge_index.max()) >= self.num_nodes or int(edge_index.min()) < 0:
                raise ValueError(
                    f"{name}_edge_index.npy contains invalid node index. "
                    f"num_nodes={self.num_nodes}, min={int(edge_index.min())}, max={int(edge_index.max())}"
                )

            edge_index_t = torch.tensor(edge_index, dtype=torch.long)
            if edge_weight_path.exists():
                edge_weight = np.load(edge_weight_path)
                if edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[1]:
                    raise ValueError(
                        f"{name}_edge_weight.npy must be [E], got {edge_weight.shape}, E={edge_index.shape[1]}"
                    )
                edge_weight_t = torch.tensor(edge_weight, dtype=torch.float32)
            else:
                edge_weight_t = torch.ones(edge_index.shape[1], dtype=torch.float32)

            data = Data(x=x, edge_index=edge_index_t, edge_weight=edge_weight_t, y=y)

            if edge_attr_path.exists():
                edge_attr = np.load(edge_attr_path)
                if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
                    raise ValueError(
                        f"{name}_edge_attr.npy must be [E, D], got {edge_attr.shape}, E={edge_index.shape[1]}"
                    )
                data.edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        else:
            raise FileNotFoundError(
                f"Custom dataset '{name}' not found. Need either dense style "
                f"({dense_adj_path.name}, {feat_path.name}, {label_path.name}) or sparse style "
                f"({sparse_edge_index_path.name}, optional {edge_weight_path.name}/{edge_attr_path.name}, "
                f"{feat_path.name}, {label_path.name})."
            )

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
