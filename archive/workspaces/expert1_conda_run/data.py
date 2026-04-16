import io
import json
import urllib.request
import zipfile
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.datasets import Amazon, Coauthor, KarateClub, Planetoid, WebKB
from torch_geometric.utils import from_networkx

from utils.model_utils import adjacency2index, normalize_adj


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


def standardize_edge_attr(edge_attr: torch.Tensor) -> torch.Tensor:
    if edge_attr is None or edge_attr.numel() == 0:
        return edge_attr
    edge_attr = edge_attr.float()
    mean = edge_attr.mean(dim=0, keepdim=True)
    std = edge_attr.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    edge_attr = (edge_attr - mean) / std
    edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)
    return edge_attr


def build_generic_edge_attr(x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
    src, dst = edge_index[0], edge_index[1]
    x = x.float()
    xi = x[src]
    xj = x[dst]
    cos = F.cosine_similarity(xi, xj, dim=1, eps=1e-8)
    l2 = torch.norm(xi - xj, p=2, dim=1)
    l1 = torch.mean(torch.abs(xi - xj), dim=1)

    deg = torch.bincount(src, minlength=x.shape[0]).float()
    log_deg_src = torch.log1p(deg[src])
    log_deg_dst = torch.log1p(deg[dst])
    log_deg_gap = torch.abs(log_deg_src - log_deg_dst)

    attrs = [cos, l2, l1, log_deg_src, log_deg_dst, log_deg_gap]
    if edge_weight is not None:
        attrs.append(edge_weight.float())
    return standardize_edge_attr(torch.stack(attrs, dim=1))


def build_edge_weight_from_attr(
    edge_attr: torch.Tensor,
    ref_weight: torch.Tensor | None = None,
    temp: float = 1.0,
) -> torch.Tensor | None:
    if edge_attr is None or edge_attr.numel() == 0:
        return None
    if edge_attr.dim() == 1:
        edge_attr = edge_attr.unsqueeze(1)
    edge_attr = edge_attr.float()
    temp = max(1e-6, float(temp))

    # Fixed, bounded mapping: robust and non-negative.
    score = edge_attr.mean(dim=1)
    score = (score - score.mean()) / score.std(unbiased=False).clamp_min(1e-6)
    w = torch.sigmoid(score / temp).clamp(1e-4, 1.0)

    # Keep scale close to reference weights to avoid objective-level collapse.
    if ref_weight is not None and ref_weight.numel() == w.numel():
        ref_weight = ref_weight.float()
        w_mean = w.mean().clamp_min(1e-6)
        ref_mean = ref_weight.mean().clamp_min(1e-6)
        w = w / w_mean * ref_mean
        upper = (ref_weight.mean() + 3.0 * ref_weight.std(unbiased=False)).clamp_min(ref_mean)
        w = w.clamp(1e-6, float(upper.detach().item()))
    return w


def coalesce_edge_data(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
    edge_attr: torch.Tensor | None = None,
):
    src = edge_index[0].long()
    dst = edge_index[1].long()
    key = src * int(num_nodes) + dst
    uniq, inv = torch.unique(key, sorted=True, return_inverse=True)

    new_src = torch.div(uniq, int(num_nodes), rounding_mode='floor')
    new_dst = uniq.remainder(int(num_nodes))
    edge_index_new = torch.stack([new_src, new_dst], dim=0)

    edge_weight_new = torch.zeros(uniq.numel(), dtype=edge_weight.dtype, device=edge_weight.device)
    edge_weight_new.index_add_(0, inv, edge_weight)

    edge_attr_new = None
    if edge_attr is not None:
        edge_attr = edge_attr.float()
        edge_attr_new = torch.zeros((uniq.numel(), edge_attr.shape[1]), dtype=edge_attr.dtype, device=edge_attr.device)
        edge_attr_new.index_add_(0, inv, edge_attr)
        counts = torch.bincount(inv, minlength=uniq.numel()).to(edge_attr.dtype).clamp_min(1.0)
        edge_attr_new = edge_attr_new / counts.unsqueeze(1)

    return edge_index_new, edge_weight_new, edge_attr_new


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
    if isinstance(dataset, ATsDataset) and getattr(dataset, "unknown_label", None) is not None:
        data.unknown_label = int(dataset.unknown_label)
    data.x = data.x.float()
    N = data.x.shape[0]
    variant = str(getattr(configs, 'edge_variant', 'V1')).upper()
    edge_index = data.edge_index.long()
    edge_attr_in = None
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        if data.edge_attr.dim() == 1:
            edge_attr_in = data.edge_attr.float().unsqueeze(1)
        elif data.edge_attr.dim() == 2:
            edge_attr_in = data.edge_attr.float()

    if hasattr(data, "edge_weight") and data.edge_weight is not None and data.edge_weight.numel() == edge_index.shape[1]:
        input_prior_weight = data.edge_weight.float()
    else:
        input_prior_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    edge_weight = input_prior_weight

    if variant in {'V2', 'V4', 'V5', 'V6', 'V7', 'V8', 'V12', 'V13', 'V20'}:
        w_struct = build_structural_edge_weight(edge_index, N)
    if variant in {'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V12', 'V13', 'V20'}:
        w_feat = build_feature_edge_weight(data.x.float(), edge_index, temp=getattr(configs, 'edge_feat_temp', 1.0))

    if variant == 'V2':
        edge_weight = w_struct
    elif variant == 'V3':
        edge_weight = w_feat
    elif variant in {'V4', 'V5', 'V6', 'V7', 'V8', 'V12', 'V13', 'V20'}:
        alpha = float(getattr(configs, 'edge_hybrid_alpha', 0.5))
        alpha = max(0.0, min(1.0, alpha))
        edge_weight = alpha * w_feat + (1.0 - alpha) * w_struct

    prior_alpha = float(getattr(configs, 'edge_input_prior_alpha', 0.0))
    prior_alpha = max(0.0, min(1.0, prior_alpha))
    if prior_alpha > 0.0 and variant in {'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V12', 'V13', 'V20'}:
        edge_weight = (1.0 - prior_alpha) * edge_weight + prior_alpha * input_prior_weight

    edge_weight = edge_weight.clamp_min(1e-6)
    if edge_attr_in is None:
        edge_attr = build_generic_edge_attr(data.x, edge_index, edge_weight=edge_weight)
    else:
        edge_attr = standardize_edge_attr(edge_attr_in)
        if bool(getattr(configs, "append_generic_edge_attr", False)):
            edge_attr_generic = build_generic_edge_attr(data.x, edge_index, edge_weight=edge_weight)
            edge_attr = torch.cat([edge_attr, edge_attr_generic], dim=1)

    edge_index, edge_weight, edge_attr = coalesce_edge_data(
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=N,
        edge_attr=edge_attr,
    )
    edge_attr = standardize_edge_attr(edge_attr)

    edge_attr_weight_blend = float(getattr(configs, 'edge_attr_weight_blend', 0.0))
    edge_attr_weight_blend = max(0.0, min(1.0, edge_attr_weight_blend))
    edge_weight_msg = edge_weight.clone()
    edge_weight_si = edge_weight.clone()
    if edge_attr_weight_blend > 0.0:
        edge_weight_attr = build_edge_weight_from_attr(
            edge_attr=edge_attr,
            ref_weight=edge_weight,
            temp=float(getattr(configs, 'edge_attr_weight_temp', 1.0)),
        )
        if edge_weight_attr is not None:
            edge_weight_si = (1.0 - edge_attr_weight_blend) * edge_weight + edge_attr_weight_blend * edge_weight_attr
            if str(getattr(configs, 'edge_attr_weight_apply_to', 'si_only')).lower() == 'both':
                edge_weight_msg = edge_weight_si.clone()

    edge_weight_msg = edge_weight_msg.clamp_min(1e-6)
    edge_weight_si = edge_weight_si.clamp_min(1e-6)

    adj_msg_raw = torch.sparse_coo_tensor(indices=edge_index, values=edge_weight_msg, size=(N, N)).coalesce()
    adj_si = torch.sparse_coo_tensor(indices=edge_index, values=edge_weight_si, size=(N, N)).coalesce()
    adj_msg = normalize_adj(adj_msg_raw, sparse=True).coalesce()

    data.edge_index = edge_index
    data.edge_weight_raw = edge_weight
    data.edge_weight_msg = edge_weight_msg
    data.edge_weight_si = edge_weight_si
    data.edge_weight = edge_weight_msg
    data.edge_attr = edge_attr
    data.adj_msg_raw = adj_msg_raw
    data.adj_msg = adj_msg
    data.adj_si = adj_si
    data.adj = adj_msg

    if bool(getattr(configs, "known_only_eval", False)):
        unknown_label = getattr(data, "unknown_label", None)
        if unknown_label is not None:
            y = data.y.clone()
            y[y == int(unknown_label)] = -1
            data.y = y

    valid_y = data.y[data.y >= 0]
    data.num_classes = int(valid_y.max().item() + 1) if valid_y.numel() > 0 else 1
    data.known_label_ratio = float((data.y >= 0).float().mean().item()) if data.y.numel() > 0 else 0.0
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

        self.unknown_label = None
        meta_path = base / f"{name}_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                mapping = meta.get("label_mapping", {})
                if isinstance(mapping, dict) and ("unknown" in mapping):
                    self.unknown_label = int(mapping["unknown"])
            except Exception:
                self.unknown_label = None

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
