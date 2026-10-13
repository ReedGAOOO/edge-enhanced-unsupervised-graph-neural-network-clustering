#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics as sk_metrics


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import load_data  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def make_data_configs(dataset: str, root_path: str, append_generic_edge_attr: bool = False):
    return SimpleNamespace(
        dataset=dataset,
        root_path=root_path,
        edge_variant="V1",
        append_generic_edge_attr=append_generic_edge_attr,
        known_only_eval=False,
        edge_feat_temp=1.0,
        edge_hybrid_alpha=0.5,
        edge_input_prior_alpha=0.0,
        edge_attr_weight_blend=0.0,
        edge_attr_weight_temp=1.0,
        edge_attr_weight_apply_to="si_only",
    )


def build_dense_adjs_from_edge_attr(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    add_self_loops: bool = False,
    symmetrize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    assert edge_index.dim() == 2 and edge_index.size(0) == 2
    assert edge_attr.dim() == 2
    device = edge_attr.device
    src, dst = edge_index[0].long(), edge_index[1].long()
    ea = edge_attr.float()
    ea_min = ea.min(dim=0, keepdim=True).values
    ea_max = ea.max(dim=0, keepdim=True).values
    ea = (ea - ea_min) / (ea_max - ea_min + eps)
    ea = torch.nan_to_num(ea, nan=0.0, posinf=0.0, neginf=0.0)

    num_channels = ea.shape[1]
    A = torch.zeros(num_channels, num_nodes, num_nodes, device=device, dtype=ea.dtype)
    A[:, src, dst] = ea.T
    if symmetrize:
        A[:, dst, src] = torch.maximum(A[:, dst, src], ea.T)

    if add_self_loops:
        eye = torch.eye(num_nodes, device=device, dtype=A.dtype)
        A = A + eye.unsqueeze(0)

    return A


def row_normalize(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return M / (M.sum(dim=-1, keepdim=True) + eps)


def build_scalar_graph(
    A0_channels: torch.Tensor,
    mode: str = "mean",
    logits: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    if mode == "mean":
        return A0_channels.mean(dim=0), None
    if mode == "learned":
        if logits is None:
            raise ValueError("build_scalar_graph(mode='learned') requires logits")
        alpha = torch.softmax(logits, dim=0)
        A_scalar = torch.einsum("c,cij->ij", alpha, A0_channels)
        return A_scalar, alpha
    raise ValueError(f"Unknown scalar graph mode: {mode}")


def coarsen_single(A_h: torch.Tensor, C_h: torch.Tensor) -> torch.Tensor:
    A_next = C_h.T @ A_h @ C_h
    return 0.5 * (A_next + A_next.T)


def coarsen_channels(A_channels_h: torch.Tensor, C_h: torch.Tensor) -> torch.Tensor:
    outs = []
    for c in range(A_channels_h.size(0)):
        A_next = C_h.T @ A_channels_h[c] @ C_h
        A_next = 0.5 * (A_next + A_next.T)
        outs.append(A_next)
    return torch.stack(outs, dim=0)


class LorentzOps:
    def __init__(self, kappa: float = -1.0, eps: float = 1e-8):
        assert kappa < 0
        self.kappa = float(kappa)
        self.eps = float(eps)

    def minkowski_dot(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -x[..., :1] * y[..., :1] + (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)

    def lorentz_norm_sq(self, x: torch.Tensor) -> torch.Tensor:
        return self.minkowski_dot(x, x)

    def project_to_lorentz(self, spatial: torch.Tensor) -> torch.Tensor:
        x_sq = (spatial * spatial).sum(dim=-1, keepdim=True)
        time = torch.sqrt(torch.clamp(x_sq - 1.0 / self.kappa, min=self.eps))
        return torch.cat([time, spatial], dim=-1)

    def lift_euclidean(self, x: torch.Tensor) -> torch.Tensor:
        return self.project_to_lorentz(x)

    def pairwise_distance_sq(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        q0, qs = q[:, :1], q[:, 1:]
        k0, ks = k[:, :1], k[:, 1:]
        md = -(q0 @ k0.T) + (qs @ ks.T)
        arg = torch.clamp(self.kappa * md, min=1.0 + self.eps)
        d = torch.acosh(arg)
        return d * d

    def weighted_centroid(self, weights_out_in: torch.Tensor, X_in: torch.Tensor) -> torch.Tensor:
        wsum = weights_out_in @ X_in
        lnorm = torch.sqrt(torch.clamp(-self.lorentz_norm_sq(wsum), min=self.eps))
        mu = (1.0 / math.sqrt(-self.kappa)) * (wsum / lnorm)
        return mu


class LorentzLinear(nn.Module):
    def __init__(self, in_dim_spatial: int, out_dim_spatial: int, kappa: float = -1.0):
        super().__init__()
        self.lin = nn.Linear(in_dim_spatial, out_dim_spatial)
        self.ops = LorentzOps(kappa=kappa)

    def forward(self, x_lorentz: torch.Tensor) -> torch.Tensor:
        spatial = x_lorentz[:, 1:]
        out_spatial = self.lin(spatial)
        return self.ops.project_to_lorentz(out_spatial)


class LorentzChannelConv(nn.Module):
    def __init__(self, in_dim_spatial: int, hidden_dim_spatial: int, kappa: float = -1.0):
        super().__init__()
        self.ops = LorentzOps(kappa=kappa)
        self.q_lin = LorentzLinear(in_dim_spatial, hidden_dim_spatial, kappa)
        self.k_lin = LorentzLinear(in_dim_spatial, hidden_dim_spatial, kappa)
        self.v_lin = LorentzLinear(in_dim_spatial, hidden_dim_spatial, kappa)

    def forward(self, x_lorentz: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        Q = self.q_lin(x_lorentz)
        K = self.k_lin(x_lorentz)
        V = self.v_lin(x_lorentz)
        d2 = self.ops.pairwise_distance_sq(Q, K)
        scale = math.sqrt(1.0 / max(1, Q.size(0)))
        att = torch.softmax(-scale * d2, dim=-1)
        masked = row_normalize(att * A)
        return self.ops.weighted_centroid(masked, V)


class ChannelAssigner(nn.Module):
    def __init__(self, hidden_dim_spatial: int, parent_size: int, kappa: float = -1.0):
        super().__init__()
        self.ops = LorentzOps(kappa=kappa)
        self.q_lin = LorentzLinear(hidden_dim_spatial, hidden_dim_spatial, kappa)
        self.k_lin = LorentzLinear(hidden_dim_spatial, hidden_dim_spatial, kappa)
        self.logit_mlp = nn.Sequential(
            nn.Linear(hidden_dim_spatial, hidden_dim_spatial),
            nn.ReLU(),
            nn.Linear(hidden_dim_spatial, parent_size),
        )

    def forward(self, Z_h: torch.Tensor, A_h: torch.Tensor) -> torch.Tensor:
        Q = self.q_lin(Z_h)
        K = self.k_lin(Z_h)
        d2 = self.ops.pairwise_distance_sq(Q, K)
        scale = math.sqrt(1.0 / max(1, Q.size(0)))
        att = torch.softmax(-scale * d2, dim=-1)
        masked = row_normalize(att * A_h)
        logits = self.logit_mlp(Z_h[:, 1:])
        return masked @ logits


class LevelGate(nn.Module):
    def __init__(self, hidden_dim_spatial: int, num_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim_spatial + num_channels, hidden_dim_spatial),
            nn.ReLU(),
            nn.Linear(hidden_dim_spatial, num_channels),
        )

    def forward(self, Z_h: torch.Tensor, A_channels_h: torch.Tensor) -> torch.Tensor:
        pooled = Z_h[:, 1:].mean(dim=0)
        edge_mass = A_channels_h.mean(dim=(1, 2))
        gate_in = torch.cat([pooled, edge_mass], dim=0)
        return torch.softmax(self.net(gate_in), dim=0)


def assignment_temperature(epoch: int, total_epochs: int, use_temp_anneal: bool) -> float:
    if not use_temp_anneal:
        return 1.0
    r = float(epoch) / max(1.0, float(total_epochs))
    if r <= 0.30:
        frac = r / 0.30
        return 2.0 + frac * (1.0 - 2.0)
    if r <= 0.70:
        return 1.0
    frac = (r - 0.70) / 0.30
    return 1.0 + frac * (0.5 - 1.0)


def decode_nearest_k(C_list: List[torch.Tensor], target_k: int) -> np.ndarray:
    assign_k = cumulative_assignment_for_k(C_list, target_k=target_k)
    return assign_k.argmax(dim=-1).detach().cpu().numpy()


def topdown_decode(C_list: List[torch.Tensor], target_k: int) -> np.ndarray:
    num_nodes = int(C_list[0].size(0))
    hard_levels: List[np.ndarray] = [C.argmax(dim=-1).detach().cpu().numpy() for C in C_list]

    leaf_to_level: List[np.ndarray] = []
    cur = np.arange(num_nodes, dtype=np.int64)
    for parent_of_child in hard_levels:
        cur = parent_of_child[cur]
        unique_nodes = np.unique(cur)
        remap = {int(node): idx for idx, node in enumerate(unique_nodes.tolist())}
        leaf_to_level.append(np.array([remap[int(node)] for node in cur], dtype=np.int64))

    pruned_partitions: List[np.ndarray] = []
    for part in leaf_to_level:
        if not pruned_partitions or not np.array_equal(pruned_partitions[-1], part):
            pruned_partitions.append(part)

    if not pruned_partitions:
        return np.zeros(num_nodes, dtype=np.int64)

    best = pruned_partitions[0]
    best_gap = abs(np.unique(best).size - target_k)
    for part in pruned_partitions[1:]:
        gap = abs(np.unique(part).size - target_k)
        if gap < best_gap:
            best = part
            best_gap = gap
    return best


class MultiChannelSharedTreeLSEnet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_channels: int,
        parent_sizes: List[int],
        kappa: float = -1.0,
        dsi_eps: float = 1e-8,
        usage_reg_lambda: float = 0.0,
        dsi_mode: str = "global",
        routing_mode: str = "learned_level",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.parent_sizes = parent_sizes
        self.ops = LorentzOps(kappa=kappa, eps=dsi_eps)
        self.dsi_eps = float(dsi_eps)
        self.usage_reg_lambda = float(usage_reg_lambda)
        self.dsi_mode = str(dsi_mode)
        self.routing_mode = str(routing_mode)

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.leaf_convs = nn.ModuleList(
            [LorentzChannelConv(hidden_dim, hidden_dim, kappa=kappa) for _ in range(num_channels)]
        )
        self.leaf_fuse_logits = nn.Parameter(torch.zeros(num_channels))

        self.assigners = nn.ModuleList()
        self.level_gates = nn.ModuleList()
        for parent_size in parent_sizes:
            self.assigners.append(
                nn.ModuleList(
                    [ChannelAssigner(hidden_dim, parent_size, kappa=kappa) for _ in range(num_channels)]
                )
            )
            if self.routing_mode == "learned_level":
                self.level_gates.append(LevelGate(hidden_dim, num_channels))

        if self.routing_mode == "global_shared":
            self.global_pi_logits = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("global_pi_logits", None)

        self.dsi_channel_logits = nn.Parameter(torch.zeros(num_channels))

    def lift_input(self, x: torch.Tensor) -> torch.Tensor:
        x_e = self.input_proj(x)
        return self.ops.lift_euclidean(x_e)

    def fuse_leaf(self, Z_channels: List[torch.Tensor]) -> torch.Tensor:
        alpha = torch.softmax(self.leaf_fuse_logits, dim=0)
        Z = sum(alpha[c] * Z_channels[c] for c in range(self.num_channels))
        return self.ops.project_to_lorentz(Z[:, 1:])

    def build_fused_graph_for_dsi(self, A0_channels: torch.Tensor) -> torch.Tensor:
        omega = torch.softmax(self.dsi_channel_logits, dim=0)
        return torch.einsum("c,cij->ij", omega, A0_channels)

    def routing_weights(
        self,
        level_idx: int,
        Z_h: torch.Tensor,
        A_channels_h: torch.Tensor,
    ) -> torch.Tensor:
        if self.num_channels == 1:
            return torch.ones(1, device=Z_h.device, dtype=Z_h.dtype)
        if self.routing_mode == "uniform":
            return torch.full(
                (self.num_channels,),
                1.0 / float(self.num_channels),
                device=Z_h.device,
                dtype=Z_h.dtype,
            )
        if self.routing_mode == "global_shared":
            return torch.softmax(self.global_pi_logits, dim=0)
        if self.routing_mode == "learned_level":
            return self.level_gates[level_idx](Z_h, A_channels_h)
        raise ValueError(f"Unknown routing_mode: {self.routing_mode}")

    def compute_level_assignment(
        self,
        level_idx: int,
        Z_h: torch.Tensor,
        A_channels_h: torch.Tensor,
        assignment_temp: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pi_h = self.routing_weights(level_idx, Z_h, A_channels_h)
        U = 0.0
        for c in range(self.num_channels):
            U_c = self.assigners[level_idx][c](Z_h, A_channels_h[c])
            U = U + pi_h[c] * U_c
        C_h = torch.softmax(U / max(float(assignment_temp), self.dsi_eps), dim=-1)
        return C_h, pi_h

    def parent_update(self, C_h: torch.Tensor, Z_h: torch.Tensor) -> torch.Tensor:
        return self.ops.weighted_centroid(C_h.T, Z_h)

    def coarsen_channels(self, C_h: torch.Tensor, A_channels_h: torch.Tensor) -> torch.Tensor:
        C_t = C_h.T
        out = []
        for c in range(self.num_channels):
            A_next = C_t @ A_channels_h[c] @ C_h
            A_next = 0.5 * (A_next + A_next.T)
            out.append(A_next)
        return torch.stack(out, dim=0)

    def fuse_levels_global(self, A_levels: List[torch.Tensor]) -> List[torch.Tensor]:
        omega = torch.softmax(self.dsi_channel_logits, dim=0)
        return [torch.einsum("c,cij->ij", omega, A_h) for A_h in A_levels]

    def fuse_levels_level_aware(
        self,
        A_levels: List[torch.Tensor],
        pi_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        fused: List[torch.Tensor] = []
        for idx, A_h in enumerate(A_levels):
            if idx < len(pi_list):
                weights = pi_list[idx]
            elif pi_list:
                weights = pi_list[-1]
            else:
                weights = torch.softmax(self.dsi_channel_logits, dim=0)
            fused.append(torch.einsum("c,cij->ij", weights, A_h))
        return fused

    def dsi_loss_from_levels(self, fused_levels: List[torch.Tensor], C_list: List[torch.Tensor]) -> torch.Tensor:
        vol_G = fused_levels[0].sum() + self.dsi_eps
        se_loss = fused_levels[0].new_tensor(0.0)
        for level_idx, C_h in enumerate(C_list):
            adj_child = fused_levels[level_idx]
            adj_parent = fused_levels[level_idx + 1]
            degree = adj_child.sum(dim=-1)
            diag = torch.diag(adj_child)
            vol_parent = adj_parent.sum(dim=-1)
            vol_parent_for_each_k = C_h @ vol_parent
            delta_vol = degree - diag
            log_ratio = torch.log2((degree + self.dsi_eps) / (vol_parent_for_each_k + self.dsi_eps))
            se_loss = se_loss + torch.sum(delta_vol * log_ratio)
        return -se_loss / vol_G

    def usage_entropy(self, C_list: List[torch.Tensor]) -> torch.Tensor:
        total = C_list[0].new_tensor(0.0)
        for C_h in C_list:
            p_h = C_h.mean(dim=0)
            total = total - torch.sum(p_h * torch.log(p_h + self.dsi_eps))
        return total

    def forward(
        self,
        x: torch.Tensor,
        A0_channels: torch.Tensor,
        assignment_temp: float = 1.0,
    ) -> Dict[str, object]:
        assert A0_channels.dim() == 3 and A0_channels.size(0) == self.num_channels
        X0 = self.lift_input(x)
        Z_leaf_channels = [self.leaf_convs[c](X0, A0_channels[c]) for c in range(self.num_channels)]
        Z_h = self.fuse_leaf(Z_leaf_channels)

        Z_levels = [Z_h]
        A_levels = [A0_channels]
        C_list = []
        pi_list = []
        A_channels_h = A0_channels

        for level_idx, _ in enumerate(self.parent_sizes):
            C_h, pi_h = self.compute_level_assignment(
                level_idx,
                Z_h,
                A_channels_h,
                assignment_temp=assignment_temp,
            )
            Z_parent = self.parent_update(C_h, Z_h)
            A_channels_parent = self.coarsen_channels(C_h, A_channels_h)

            C_list.append(C_h)
            pi_list.append(pi_h)
            Z_levels.append(Z_parent)
            A_levels.append(A_channels_parent)
            Z_h = Z_parent
            A_channels_h = A_channels_parent

        global_levels = self.fuse_levels_global(A_levels)
        if self.dsi_mode == "level_aware":
            fused_levels = self.fuse_levels_level_aware(A_levels, pi_list)
        elif self.dsi_mode == "global":
            fused_levels = global_levels
        else:
            raise ValueError(f"Unknown dsi_mode: {self.dsi_mode}")

        dsi_loss = self.dsi_loss_from_levels(fused_levels, C_list)
        usage_entropy = self.usage_entropy(C_list)
        total_loss = dsi_loss - self.usage_reg_lambda * usage_entropy
        return {
            "Z_levels": Z_levels,
            "A_levels": A_levels,
            "C_list": C_list,
            "pi_list": pi_list,
            "A_fused": global_levels[0],
            "fused_levels": fused_levels,
            "dsi_loss": dsi_loss,
            "usage_entropy": usage_entropy,
            "total_loss": total_loss,
            "leaf_alpha": torch.softmax(self.leaf_fuse_logits, dim=0),
            "dsi_alpha": torch.softmax(self.dsi_channel_logits, dim=0),
            "global_pi_alpha": (
                torch.softmax(self.global_pi_logits, dim=0)
                if self.global_pi_logits is not None
                else None
            ),
        }


class EarlyScalarizedSharedTreeLSEnet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_channels: int,
        parent_sizes: List[int],
        kappa: float = -1.0,
        dsi_eps: float = 1e-8,
        usage_reg_lambda: float = 0.0,
        dsi_mode: str = "global",
        routing_mode: str = "learned_level",
    ):
        super().__init__()
        self.scalar_logits = nn.Parameter(torch.zeros(num_channels))
        self.base = MultiChannelSharedTreeLSEnet(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_channels=1,
            parent_sizes=parent_sizes,
            kappa=kappa,
            dsi_eps=dsi_eps,
            usage_reg_lambda=usage_reg_lambda,
            dsi_mode=dsi_mode,
            routing_mode=routing_mode,
        )

    def forward(
        self,
        x: torch.Tensor,
        A0_channels: torch.Tensor,
        assignment_temp: float = 1.0,
    ) -> Dict[str, object]:
        omega = torch.softmax(self.scalar_logits, dim=0)
        A_scalar = torch.einsum("c,cij->ij", omega, A0_channels)
        out = self.base(x, A_scalar.unsqueeze(0), assignment_temp=assignment_temp)
        out["scalar_alpha"] = omega
        return out


class SingleChannelMeanSharedTreeLSEnet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        parent_sizes: List[int],
        kappa: float = -1.0,
        dsi_eps: float = 1e-8,
        usage_reg_lambda: float = 0.0,
        dsi_mode: str = "global",
        routing_mode: str = "learned_level",
    ):
        super().__init__()
        self.base = MultiChannelSharedTreeLSEnet(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_channels=1,
            parent_sizes=parent_sizes,
            kappa=kappa,
            dsi_eps=dsi_eps,
            usage_reg_lambda=usage_reg_lambda,
            dsi_mode=dsi_mode,
            routing_mode=routing_mode,
        )

    def forward(
        self,
        x: torch.Tensor,
        A0_channels: torch.Tensor,
        assignment_temp: float = 1.0,
    ) -> Dict[str, object]:
        A_scalar = A0_channels.mean(dim=0, keepdim=True)
        return self.base(x, A_scalar, assignment_temp=assignment_temp)


@dataclass
class VariantSpec:
    leaf_mode: Literal["mean_single", "scalar_single", "multi_branch"]
    assign_mode: Literal["single_graph", "multi_graph_shared_assigner"]
    scalar_source: Literal["mean", "learned"]
    assign_pi_mode: Literal["none", "global_shared"]
    use_channel_coarsen: bool
    dsi_source: Literal["mean", "learned_scalar"]
    assign_residual: bool = False
    assign_residual_init: float = 0.0


class SpecDrivenSharedTreeLSEnet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_channels: int,
        parent_sizes: List[int],
        spec: VariantSpec,
        kappa: float = -1.0,
        dsi_eps: float = 1e-8,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.parent_sizes = parent_sizes
        self.spec = spec
        self.ops = LorentzOps(kappa=kappa, eps=dsi_eps)
        self.dsi_eps = float(dsi_eps)

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        if spec.leaf_mode in {"mean_single", "scalar_single"}:
            self.leaf_conv_single = LorentzChannelConv(hidden_dim, hidden_dim, kappa=kappa)
        else:
            self.leaf_convs = nn.ModuleList(
                [LorentzChannelConv(hidden_dim, hidden_dim, kappa=kappa) for _ in range(num_channels)]
            )
            self.leaf_fuse_logits = nn.Parameter(torch.zeros(num_channels))

        if spec.assign_mode == "single_graph":
            self.assigners_single = nn.ModuleList(
                [ChannelAssigner(hidden_dim, parent_size, kappa=kappa) for parent_size in parent_sizes]
            )
        elif spec.assign_mode == "multi_graph_shared_assigner":
            self.assigners_shared = nn.ModuleList(
                [ChannelAssigner(hidden_dim, parent_size, kappa=kappa) for parent_size in parent_sizes]
            )
        else:
            raise ValueError(f"Unknown assign_mode: {spec.assign_mode}")

        needs_scalar_logits = spec.scalar_source == "learned" or spec.dsi_source == "learned_scalar"
        if needs_scalar_logits:
            self.scalar_alpha_logits = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("scalar_alpha_logits", None)

        if spec.assign_pi_mode == "global_shared":
            self.assign_pi_logits = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("assign_pi_logits", None)

        if spec.assign_residual:
            self.assign_residual_scale = nn.Parameter(
                torch.tensor(float(spec.assign_residual_init), dtype=torch.float32)
            )
        else:
            self.register_parameter("assign_residual_scale", None)

    def lift_input(self, x: torch.Tensor) -> torch.Tensor:
        x_e = self.input_proj(x)
        return self.ops.lift_euclidean(x_e)

    def fuse_lorentz_embeddings(self, Z_channels: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        leaf_alpha = torch.softmax(self.leaf_fuse_logits, dim=0)
        Z = sum(leaf_alpha[c] * Z_channels[c] for c in range(self.num_channels))
        return self.ops.project_to_lorentz(Z[:, 1:]), leaf_alpha

    def parent_update(self, C_h: torch.Tensor, Z_h: torch.Tensor) -> torch.Tensor:
        return self.ops.weighted_centroid(C_h.T, Z_h)

    def dsi_loss_from_levels(self, A_levels: List[torch.Tensor], C_list: List[torch.Tensor]) -> torch.Tensor:
        vol_G = A_levels[0].sum() + self.dsi_eps
        se_loss = A_levels[0].new_tensor(0.0)
        for level_idx, C_h in enumerate(C_list):
            adj_child = A_levels[level_idx]
            adj_parent = A_levels[level_idx + 1]
            degree = adj_child.sum(dim=-1)
            diag = torch.diag(adj_child)
            vol_parent = adj_parent.sum(dim=-1)
            vol_parent_for_each_k = C_h @ vol_parent
            delta_vol = degree - diag
            log_ratio = torch.log2((degree + self.dsi_eps) / (vol_parent_for_each_k + self.dsi_eps))
            se_loss = se_loss + torch.sum(delta_vol * log_ratio)
        return -se_loss / vol_G

    def dsi_loss_from_single_graph(self, A_root: torch.Tensor, C_list: List[torch.Tensor]) -> torch.Tensor:
        A_levels = [A_root]
        A_h = A_root
        for C_h in C_list:
            A_h = coarsen_single(A_h, C_h)
            A_levels.append(A_h)
        return self.dsi_loss_from_levels(A_levels, C_list)

    def col_usage_entropy_list(self, C_list: List[torch.Tensor]) -> List[torch.Tensor]:
        out = []
        for C_h in C_list:
            p_h = C_h.mean(dim=0)
            out.append(-torch.sum(p_h * torch.log(p_h + self.dsi_eps)))
        return out

    def forward(
        self,
        x: torch.Tensor,
        A0_channels: torch.Tensor,
        assignment_temp: float = 1.0,
    ) -> Dict[str, object]:
        assert A0_channels.dim() == 3 and A0_channels.size(0) == self.num_channels
        X0 = self.lift_input(x)

        A_mean, _ = build_scalar_graph(A0_channels, mode="mean")

        scalar_alpha = None
        if self.spec.scalar_source == "learned" or self.spec.dsi_source == "learned_scalar":
            if self.scalar_alpha_logits is None:
                raise ValueError("spec requires learned scalar graph but scalar_alpha_logits is missing")
            A_learned_scalar, scalar_alpha = build_scalar_graph(
                A0_channels,
                mode="learned",
                logits=self.scalar_alpha_logits,
            )
        else:
            A_learned_scalar = None

        if self.spec.leaf_mode == "mean_single":
            Z_h = self.leaf_conv_single(X0, A_mean)
            leaf_alpha = None
        elif self.spec.leaf_mode == "scalar_single":
            if A_learned_scalar is None:
                raise ValueError("scalar_single leaf mode requires learned scalar graph")
            Z_h = self.leaf_conv_single(X0, A_learned_scalar)
            leaf_alpha = None
        elif self.spec.leaf_mode == "multi_branch":
            Z_leaf_channels = [self.leaf_convs[c](X0, A0_channels[c]) for c in range(self.num_channels)]
            Z_h, leaf_alpha = self.fuse_lorentz_embeddings(Z_leaf_channels)
        else:
            raise ValueError(f"Unknown leaf_mode: {self.spec.leaf_mode}")

        C_list: List[torch.Tensor] = []
        pi_list: List[torch.Tensor] = []
        assign_pi = None
        residual_norm_ratio_list: List[torch.Tensor] = []
        channel_argmax_agreement_list: List[torch.Tensor] = []

        if self.spec.assign_mode == "single_graph":
            if self.spec.scalar_source == "mean":
                A_h = A_mean
            else:
                if A_learned_scalar is None:
                    raise ValueError("single_graph assignment needs scalar graph")
                A_h = A_learned_scalar
            for assigner in self.assigners_single:
                U_h = assigner(Z_h, A_h)
                C_h = torch.softmax(U_h / max(float(assignment_temp), self.dsi_eps), dim=-1)
                C_list.append(C_h)
                Z_h = self.parent_update(C_h, Z_h)
                A_h = coarsen_single(A_h, C_h)
        elif self.spec.assign_mode == "multi_graph_shared_assigner":
            if self.spec.assign_pi_mode != "global_shared" or self.assign_pi_logits is None:
                raise ValueError("multi_graph_shared_assigner requires global_shared assign_pi_mode")
            assign_pi = torch.softmax(self.assign_pi_logits, dim=0)
            if self.spec.scalar_source == "mean":
                A_scalar_h = A_mean
            else:
                if A_learned_scalar is None:
                    raise ValueError("multi_graph_shared_assigner needs scalar graph")
                A_scalar_h = A_learned_scalar
            A_channels_h = A0_channels
            for assigner in self.assigners_shared:
                U_base = assigner(Z_h, A_scalar_h)
                base_argmax = U_base.argmax(dim=-1)
                U_h = 0.0
                U_res = torch.zeros_like(U_base)
                agreement_per_channel = []
                for c in range(self.num_channels):
                    U_c = assigner(Z_h, A_channels_h[c])
                    agreement_per_channel.append(
                        (U_c.argmax(dim=-1) == base_argmax).float().mean()
                    )
                    if self.spec.assign_residual:
                        U_res = U_res + assign_pi[c] * (U_c - U_base)
                    else:
                        U_h = U_h + assign_pi[c] * U_c
                if self.spec.assign_residual:
                    if self.assign_residual_scale is None:
                        raise ValueError("assign_residual is enabled but assign_residual_scale is missing")
                    U_h = U_base + self.assign_residual_scale * U_res
                    residual_norm_ratio_list.append(
                        U_res.norm() / (U_base.norm() + self.dsi_eps)
                    )
                channel_argmax_agreement_list.append(torch.stack(agreement_per_channel))
                C_h = torch.softmax(U_h / max(float(assignment_temp), self.dsi_eps), dim=-1)
                C_list.append(C_h)
                pi_list.append(assign_pi)
                Z_h = self.parent_update(C_h, Z_h)
                A_scalar_h = coarsen_single(A_scalar_h, C_h)
                if self.spec.use_channel_coarsen:
                    A_channels_h = coarsen_channels(A_channels_h, C_h)
        else:
            raise ValueError(f"Unknown assign_mode: {self.spec.assign_mode}")

        if self.spec.dsi_source == "mean":
            A_dsi_root = A_mean
        elif self.spec.dsi_source == "learned_scalar":
            if A_learned_scalar is None:
                raise ValueError("learned_scalar DSI source requires learned scalar graph")
            A_dsi_root = A_learned_scalar
        else:
            raise ValueError(f"Unknown dsi_source: {self.spec.dsi_source}")

        dsi_loss = self.dsi_loss_from_single_graph(A_dsi_root, C_list)
        col_usage_entropy_list = self.col_usage_entropy_list(C_list)

        return {
            "C_list": C_list,
            "dsi_loss": dsi_loss,
            "total_loss": dsi_loss,
            "leaf_alpha": leaf_alpha,
            "scalar_alpha": scalar_alpha,
            "pi_assign": assign_pi,
            "pi_list": pi_list,
            "assign_residual_scale": self.assign_residual_scale,
            "residual_norm_ratio_list": [v.detach() for v in residual_norm_ratio_list],
            "channel_argmax_agreement_list": [v.detach() for v in channel_argmax_agreement_list],
            "col_usage_entropy_list": [v.detach() for v in col_usage_entropy_list],
        }


def default_parent_sizes(num_nodes: int, num_classes: int) -> List[int]:
    if num_classes <= 1:
        return [1]
    mid = min(max(4 * num_classes, 8), max(num_classes + 1, num_nodes // 4))
    if mid <= num_classes:
        return [num_classes, 1]
    return [mid, num_classes, 1]


def cumulative_assignment_for_k(C_list: List[torch.Tensor], target_k: int) -> torch.Tensor:
    cur = C_list[0]
    best = cur
    best_gap = abs(cur.size(1) - target_k)
    if cur.size(1) == target_k:
        return cur
    for C_h in C_list[1:]:
        cur = cur @ C_h
        gap = abs(cur.size(1) - target_k)
        if gap < best_gap:
            best = cur
            best_gap = gap
        if cur.size(1) == target_k:
            return cur
    return best


def evaluate_labels(true_y: np.ndarray, pred_y: np.ndarray) -> Dict[str, float]:
    mask = true_y >= 0
    true_valid = true_y[mask]
    pred_valid = pred_y[mask]
    if true_valid.size == 0:
        return {"nmi": float("nan"), "ari": float("nan")}
    return {
        "nmi": float(sk_metrics.normalized_mutual_info_score(true_valid, pred_valid)),
        "ari": float(sk_metrics.adjusted_rand_score(true_valid, pred_valid)),
    }


VARIANT_SPECS = {
    "single_channel_mean": VariantSpec(
        leaf_mode="mean_single",
        assign_mode="single_graph",
        scalar_source="mean",
        assign_pi_mode="none",
        use_channel_coarsen=False,
        dsi_source="mean",
    ),
    "early_scalarized": VariantSpec(
        leaf_mode="scalar_single",
        assign_mode="single_graph",
        scalar_source="learned",
        assign_pi_mode="none",
        use_channel_coarsen=False,
        dsi_source="learned_scalar",
    ),
    "mc_leaf_scalar_assign": VariantSpec(
        leaf_mode="multi_branch",
        assign_mode="single_graph",
        scalar_source="learned",
        assign_pi_mode="none",
        use_channel_coarsen=False,
        dsi_source="learned_scalar",
    ),
    "scalar_leaf_mc_assign_shared": VariantSpec(
        leaf_mode="scalar_single",
        assign_mode="multi_graph_shared_assigner",
        scalar_source="learned",
        assign_pi_mode="global_shared",
        use_channel_coarsen=True,
        dsi_source="learned_scalar",
    ),
    "scalar_leaf_mc_assign_residual": VariantSpec(
        leaf_mode="scalar_single",
        assign_mode="multi_graph_shared_assigner",
        scalar_source="learned",
        assign_pi_mode="global_shared",
        use_channel_coarsen=True,
        dsi_source="learned_scalar",
        assign_residual=True,
        assign_residual_init=0.0,
    ),
}


def run_one(
    dataset: str,
    variant: str,
    seed: int,
    root_path: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    device: str,
) -> Dict[str, object]:
    set_seed(seed)
    configs = make_data_configs(dataset=dataset, root_path=root_path)
    data = load_data(configs)

    num_nodes = int(data.x.shape[0])
    num_classes = int(data.num_classes)
    parent_sizes = default_parent_sizes(num_nodes=num_nodes, num_classes=num_classes)

    device_t = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    x = data.x.float().to(device_t)
    y = data.y.cpu().numpy()
    edge_index = data.edge_index.to(device_t)
    edge_attr = data.edge_attr.float().to(device_t)
    A0_channels = build_dense_adjs_from_edge_attr(
        num_nodes=num_nodes,
        edge_index=edge_index,
        edge_attr=edge_attr,
        add_self_loops=True,
        symmetrize=True,
    )

    if variant not in VARIANT_SPECS:
        raise ValueError(f"Unknown variant: {variant}")
    spec = VARIANT_SPECS[variant]
    model = SpecDrivenSharedTreeLSEnet(
        in_dim=int(x.shape[1]),
        hidden_dim=hidden_dim,
        num_channels=int(A0_channels.size(0)),
        parent_sizes=parent_sizes,
        spec=spec,
    )

    model = model.to(device_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best: Dict[str, object] | None = None
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        temp = 1.0
        out = model(x, A0_channels, assignment_temp=temp)
        loss = out["total_loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        with torch.no_grad():
            pred_nearest = decode_nearest_k(out["C_list"], target_k=num_classes)
            pred_topdown = topdown_decode(out["C_list"], target_k=num_classes)
            metrics_nearest = evaluate_labels(y, pred_nearest)
            metrics_topdown = evaluate_labels(y, pred_topdown)
            total_loss_value = float(loss.detach().cpu().item())
            dsi_loss_value = float(out["dsi_loss"].detach().cpu().item())
            col_usage_entropy_list = [float(v.cpu().item()) for v in out["col_usage_entropy_list"]]
            usage_entropy_value = float(sum(col_usage_entropy_list))
            if best is None or total_loss_value < float(best["best_total_loss"]):
                best = {
                    "best_epoch": epoch,
                    "best_total_loss": total_loss_value,
                    "best_dsi_loss": dsi_loss_value,
                    "best_usage_entropy": usage_entropy_value,
                    "nmi_nearest": metrics_nearest["nmi"],
                    "ari_nearest": metrics_nearest["ari"],
                    "num_pred_clusters_nearest": int(np.unique(pred_nearest).size),
                    "nmi_topdown": metrics_topdown["nmi"],
                    "ari_topdown": metrics_topdown["ari"],
                    "num_pred_clusters_topdown": int(np.unique(pred_topdown).size),
                    "pred_labels_nearest": pred_nearest.tolist(),
                    "pred_labels_topdown": pred_topdown.tolist(),
                    "parent_sizes": list(parent_sizes),
                    "leaf_alpha": (
                        out.get("leaf_alpha").detach().cpu().tolist()
                        if out.get("leaf_alpha") is not None
                        else None
                    ),
                    "pi_list": [p.detach().cpu().tolist() for p in out.get("pi_list", [])],
                    "scalar_alpha": (
                        out.get("scalar_alpha").detach().cpu().tolist()
                        if out.get("scalar_alpha") is not None
                        else None
                    ),
                    "pi_assign": (
                        out.get("pi_assign").detach().cpu().tolist()
                        if out.get("pi_assign") is not None
                        else None
                    ),
                    "assign_residual_scale": (
                        float(out.get("assign_residual_scale").detach().cpu().item())
                        if out.get("assign_residual_scale") is not None
                        else None
                    ),
                    "residual_norm_ratio_list": [
                        float(v.cpu().item()) for v in out.get("residual_norm_ratio_list", [])
                    ],
                    "channel_argmax_agreement_list": [
                        [float(x) for x in v.cpu().tolist()]
                        for v in out.get("channel_argmax_agreement_list", [])
                    ],
                    "col_usage_entropy_list": col_usage_entropy_list,
                    "assignment_temp": temp,
                }

    assert best is not None
    best["dataset"] = dataset
    best["variant"] = variant
    best["seed"] = seed
    best["num_nodes"] = num_nodes
    best["num_edges"] = int(data.edge_index.shape[1])
    best["num_channels"] = int(edge_attr.shape[1])
    best["num_classes"] = num_classes
    best["runtime_sec"] = time.time() - t0
    return best


DEFAULT_MATRIX = [
    {"dataset": "synth_mech_full_v1_h85_s90_ds00", "group": "mechanism_signal"},
    {"dataset": "synth_mech_full_v1_h85_s90_ds00_permEA", "group": "mechanism_perm"},
    {"dataset": "synth_mech_full_v1_h65_s60_ds00", "group": "mechanism_mid"},
    {"dataset": "dblp_magnn_author_v2", "group": "native_multi"},
    {"dataset": "citeseer", "group": "sanity_negative"},
]


def aggregate_results(out_dir: Path, rows: List[Dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "runs.csv", index=False)

    if df.empty:
        return

    summary = (
        df.groupby(["dataset", "group", "variant"], as_index=False)
        .agg(
            runs=("seed", "count"),
            best_total_loss_mean=("best_total_loss", "mean"),
            best_dsi_loss_mean=("best_dsi_loss", "mean"),
            best_usage_entropy_mean=("best_usage_entropy", "mean"),
            assign_residual_scale_mean=("assign_residual_scale", "mean"),
            nmi_nearest_mean=("nmi_nearest", "mean"),
            nmi_nearest_std=("nmi_nearest", "std"),
            ari_nearest_mean=("ari_nearest", "mean"),
            ari_nearest_std=("ari_nearest", "std"),
            num_pred_clusters_nearest_mean=("num_pred_clusters_nearest", "mean"),
            nmi_topdown_mean=("nmi_topdown", "mean"),
            nmi_topdown_std=("nmi_topdown", "std"),
            ari_topdown_mean=("ari_topdown", "mean"),
            ari_topdown_std=("ari_topdown", "std"),
            num_pred_clusters_topdown_mean=("num_pred_clusters_topdown", "mean"),
            runtime_sec_mean=("runtime_sec", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
        )
        .sort_values(["dataset", "variant"])
    )
    summary.to_csv(out_dir / "summary_by_dataset_variant.csv", index=False)

    group_summary = (
        df.groupby(["group", "variant"], as_index=False)
        .agg(
            datasets=("dataset", "nunique"),
            runs=("seed", "count"),
            nmi_nearest_mean=("nmi_nearest", "mean"),
            ari_nearest_mean=("ari_nearest", "mean"),
            nmi_topdown_mean=("nmi_topdown", "mean"),
            ari_topdown_mean=("ari_topdown", "mean"),
            best_total_loss_mean=("best_total_loss", "mean"),
            best_dsi_loss_mean=("best_dsi_loss", "mean"),
            best_usage_entropy_mean=("best_usage_entropy", "mean"),
            assign_residual_scale_mean=("assign_residual_scale", "mean"),
        )
        .sort_values(["group", "variant"])
    )
    group_summary.to_csv(out_dir / "summary_by_group_variant.csv", index=False)

    wide_nearest = summary.pivot(index="dataset", columns="variant", values="nmi_nearest_mean")
    wide_topdown = summary.pivot(index="dataset", columns="variant", values="nmi_topdown_mean")
    if "shared_tree_multi_channel" in wide_nearest.columns and "early_scalarized" in wide_nearest.columns:
        delta_df = pd.DataFrame(
            {
                "dataset": wide_nearest.index,
                "delta_nmi_nearest": (
                    wide_nearest["shared_tree_multi_channel"] - wide_nearest["early_scalarized"]
                ).values,
                "delta_nmi_topdown": (
                    wide_topdown["shared_tree_multi_channel"] - wide_topdown["early_scalarized"]
                ).values,
            }
        )
        delta_df.to_csv(out_dir / "delta_shared_minus_scalar.csv", index=False)

    readme_lines = [
        "# Expert3 Dense Shared-Tree Prototype",
        "",
        "This directory contains standalone prototype runs for the expert3 factorized variant comparison.",
        "",
        "Variants:",
        "- `single_channel_mean`: fixed one-channel sanity baseline using the mean over edge-feature channels.",
        "- `early_scalarized`: collapse edge-feature channels first, then run a one-channel shared-tree model.",
        "- `mc_leaf_scalar_assign`: keep multi-channel branches only at the leaf encoder, then revert assignment/coarsening/DSI to a single learned scalar graph.",
        "- `scalar_leaf_mc_assign_shared`: keep a single scalar leaf graph, but do multi-channel assignment with one shared assigner per level and a global shared channel mixture.",
        "- `scalar_leaf_mc_assign_residual`: keep the early-scalarized trunk and add only a small residual assignment correction from the multi-channel shared-assigner path.",
        "",
        "Files:",
        "- `runs.csv`: per-run metrics and diagnostics.",
        "- `summary_by_dataset_variant.csv`: averaged metrics by dataset and variant, with both nearest-K and top-down decode metrics.",
        "- `summary_by_group_variant.csv`: grouped averages.",
        "- `delta_shared_minus_scalar.csv`: NMI advantage of the legacy shared multi-channel branch over early scalarization when both are present.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the expert3 dense shared-tree prototype.")
    parser.add_argument("--datasets", type=str, default="", help="Comma-separated dataset list.")
    parser.add_argument(
        "--variants",
        type=str,
        default="single_channel_mean,early_scalarized,mc_leaf_scalar_assign,scalar_leaf_mc_assign_shared,scalar_leaf_mc_assign_residual",
    )
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--root_path", type=str, default="data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="archive/workspaces/expert3_conda_run/results/expert3_factorized_variants_v1",
    )
    args = parser.parse_args()

    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.datasets.strip():
        dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
        matrix = [{"dataset": d, "group": "custom"} for d in dataset_list]
    else:
        matrix = DEFAULT_MATRIX

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    rows: List[Dict[str, object]] = []
    total = len(matrix) * len(variants) * len(seeds)
    done = 0
    for item in matrix:
        for variant in variants:
            for seed in seeds:
                done += 1
                print(f"[{done}/{total}] dataset={item['dataset']} variant={variant} seed={seed}", flush=True)
                row = run_one(
                    dataset=item["dataset"],
                    variant=variant,
                    seed=seed,
                    root_path=args.root_path,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    hidden_dim=args.hidden_dim,
                    device=args.device,
                )
                row["group"] = item["group"]
                rows.append(row)
                with open(out_dir / "latest_run.json", "w", encoding="utf-8") as f:
                    json.dump(row, f, indent=2)

    aggregate_results(out_dir=out_dir, rows=rows)
    print(f"[ok] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
