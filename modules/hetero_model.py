from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


def _etype_key(edge_type: Tuple[str, str, str]) -> str:
    return "__".join(edge_type)


class EdgeAwareSAGEConv(MessagePassing):
    """
    Minimal edge-aware GraphSAGE-style relation block.

    This block does not add self-features on its own; the hetero layer adds a
    per-node-type self path once, which avoids over-counting self terms when a
    destination node receives multiple relation types.
    """

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__(aggr="mean")
        self.hidden_dim = int(hidden_dim)
        self.src_lin = nn.Linear(hidden_dim, hidden_dim)
        self.edge_dim = int(edge_dim)
        if self.edge_dim > 0:
            self.edge_mlp = nn.Sequential(
                nn.Linear(self.edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.edge_mlp = None

    def forward(
        self,
        x_src: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        num_dst: int,
    ) -> torch.Tensor:
        size = (int(x_src.shape[0]), int(num_dst))
        return self.propagate(edge_index=edge_index, x=x_src, edge_attr=edge_attr, size=size)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor | None) -> torch.Tensor:
        msg = self.src_lin(x_j)
        if self.edge_mlp is None or edge_attr is None:
            return msg
        edge_vec = self.edge_mlp(edge_attr)
        gate = torch.sigmoid(edge_vec)
        return gate * msg + edge_vec


class EdgeAwareHeteroSAGELayer(nn.Module):
    def __init__(
        self,
        node_types: Iterable[str],
        edge_types: Iterable[Tuple[str, str, str]],
        hidden_dim: int,
        edge_dims: Dict[Tuple[str, str, str], int],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

        self.self_lin = nn.ModuleDict({node_type: nn.Linear(hidden_dim, hidden_dim) for node_type in self.node_types})
        self.rel_convs = nn.ModuleDict(
            {
                _etype_key(edge_type): EdgeAwareSAGEConv(hidden_dim=hidden_dim, edge_dim=int(edge_dims.get(edge_type, 0)))
                for edge_type in self.edge_types
            }
        )
        self.out_norm = nn.ModuleDict({node_type: nn.LayerNorm(hidden_dim) for node_type in self.node_types})

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor | None],
    ) -> Dict[str, torch.Tensor]:
        out = {node_type: self.self_lin[node_type](x_dict[node_type]) for node_type in self.node_types}
        rel_count = {node_type: 1 for node_type in self.node_types}

        for edge_type in self.edge_types:
            src_type, _, dst_type = edge_type
            conv = self.rel_convs[_etype_key(edge_type)]
            edge_index = edge_index_dict[edge_type]
            edge_attr = edge_attr_dict.get(edge_type)
            rel_msg = conv(
                x_src=x_dict[src_type],
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_dst=int(x_dict[dst_type].shape[0]),
            )
            out[dst_type] = out[dst_type] + rel_msg
            rel_count[dst_type] += 1

        for node_type in self.node_types:
            out[node_type] = out[node_type] / float(rel_count[node_type])
            out[node_type] = self.out_norm[node_type](out[node_type])
            out[node_type] = F.relu(out[node_type])
            out[node_type] = F.dropout(out[node_type], p=self.dropout, training=self.training)
        return out


class EdgeAwareHeteroGraphSAGE(nn.Module):
    def __init__(
        self,
        in_dims: Dict[str, int],
        edge_dims: Dict[Tuple[str, str, str], int],
        metadata: Tuple[Iterable[str], Iterable[Tuple[str, str, str]]],
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        target_node_type: str = "plot",
        dropout: float = 0.0,
    ):
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.target_node_type = str(target_node_type)
        self.hidden_dim = int(hidden_dim)

        self.input_lin = nn.ModuleDict(
            {node_type: nn.Linear(int(in_dims[node_type]), hidden_dim) for node_type in self.node_types}
        )
        self.layers = nn.ModuleList(
            [
                EdgeAwareHeteroSAGELayer(
                    node_types=self.node_types,
                    edge_types=self.edge_types,
                    hidden_dim=hidden_dim,
                    edge_dims=edge_dims,
                    dropout=dropout,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.head = nn.Linear(hidden_dim, int(out_dim))

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor | None],
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        x_dict = {node_type: F.relu(self.input_lin[node_type](x)) for node_type, x in x_dict.items()}
        for layer in self.layers:
            x_dict = layer(x_dict=x_dict, edge_index_dict=edge_index_dict, edge_attr_dict=edge_attr_dict)
        logits = self.head(x_dict[self.target_node_type])
        return x_dict, logits
