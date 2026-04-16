#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

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


class MultiChannelSharedTreeLSEnet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_channels: int,
        parent_sizes: List[int],
        kappa: float = -1.0,
        dsi_eps: float = 1e-8,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.parent_sizes = parent_sizes
        self.ops = LorentzOps(kappa=kappa, eps=dsi_eps)
        self.dsi_eps = float(dsi_eps)

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
            self.level_gates.append(LevelGate(hidden_dim, num_channels))

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

    def compute_level_assignment(
        self,
        level_idx: int,
        Z_h: torch.Tensor,
        A_channels_h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pi_h = self.level_gates[level_idx](Z_h, A_channels_h)
        U = 0.0
        for c in range(self.num_channels):
            U_c = self.assigners[level_idx][c](Z_h, A_channels_h[c])
            U = U + pi_h[c] * U_c
        C_h = torch.softmax(U, dim=-1)
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

    def build_S_matrices(self, C_list: List[torch.Tensor], num_nodes: int) -> Dict[int, torch.Tensor]:
        S = {}
        cur = torch.eye(num_nodes, device=C_list[0].device, dtype=C_list[0].dtype)
        S[0] = cur
        for t, C_h in enumerate(C_list):
            cur = cur @ C_h
            S[t + 1] = cur
        return S

    def dsi_loss(self, A_fused: torch.Tensor, C_list: List[torch.Tensor]) -> torch.Tensor:
        N = A_fused.size(0)
        d = A_fused.sum(dim=-1)
        V_total = d.sum() + self.dsi_eps
        S_dict = self.build_S_matrices(C_list, num_nodes=N)

        total_loss = A_fused.new_tensor(0.0)
        for t, C_h in enumerate(C_list):
            S_h = S_dict[t]
            S_parent = S_dict[t + 1]
            V_h = S_h.T @ d
            V_parent = S_parent.T @ d
            V_parent_for_each_k = C_h @ V_parent
            retained = torch.diag(S_h.T @ A_fused @ S_h)
            cut_like = V_h - retained
            ratio = torch.clamp(V_h / (V_parent_for_each_k + self.dsi_eps), min=self.dsi_eps)
            H_h = -(cut_like * torch.log2(ratio)).sum() / V_total
            total_loss = total_loss + H_h
        return total_loss

    def forward(self, x: torch.Tensor, A0_channels: torch.Tensor) -> Dict[str, object]:
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
            C_h, pi_h = self.compute_level_assignment(level_idx, Z_h, A_channels_h)
            Z_parent = self.parent_update(C_h, Z_h)
            A_channels_parent = self.coarsen_channels(C_h, A_channels_h)

            C_list.append(C_h)
            pi_list.append(pi_h)
            Z_levels.append(Z_parent)
            A_levels.append(A_channels_parent)
            Z_h = Z_parent
            A_channels_h = A_channels_parent

        A_fused = self.build_fused_graph_for_dsi(A0_channels)
        loss = self.dsi_loss(A_fused, C_list)
        return {
            "Z_levels": Z_levels,
            "A_levels": A_levels,
            "C_list": C_list,
            "pi_list": pi_list,
            "A_fused": A_fused,
            "dsi_loss": loss,
            "leaf_alpha": torch.softmax(self.leaf_fuse_logits, dim=0),
            "dsi_alpha": torch.softmax(self.dsi_channel_logits, dim=0),
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
        )

    def forward(self, x: torch.Tensor, A0_channels: torch.Tensor) -> Dict[str, object]:
        omega = torch.softmax(self.scalar_logits, dim=0)
        A_scalar = torch.einsum("c,cij->ij", omega, A0_channels)
        out = self.base(x, A_scalar.unsqueeze(0))
        out["scalar_alpha"] = omega
        return out


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

    if variant == "shared_tree_multi_channel":
        model = MultiChannelSharedTreeLSEnet(
            in_dim=int(x.shape[1]),
            hidden_dim=hidden_dim,
            num_channels=int(A0_channels.size(0)),
            parent_sizes=parent_sizes,
        )
    elif variant == "early_scalarized":
        model = EarlyScalarizedSharedTreeLSEnet(
            in_dim=int(x.shape[1]),
            hidden_dim=hidden_dim,
            num_channels=int(A0_channels.size(0)),
            parent_sizes=parent_sizes,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    model = model.to(device_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best: Dict[str, object] | None = None
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, A0_channels)
        loss = out["dsi_loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        with torch.no_grad():
            assign_k = cumulative_assignment_for_k(out["C_list"], target_k=num_classes)
            pred = assign_k.argmax(dim=-1).detach().cpu().numpy()
            metrics = evaluate_labels(y, pred)
            loss_value = float(loss.detach().cpu().item())
            if best is None or loss_value < float(best["best_loss"]):
                best = {
                    "best_epoch": epoch,
                    "best_loss": loss_value,
                    "nmi": metrics["nmi"],
                    "ari": metrics["ari"],
                    "num_pred_clusters": int(np.unique(pred).size),
                    "pred_labels": pred.tolist(),
                    "parent_sizes": list(parent_sizes),
                    "leaf_alpha": (
                        out.get("leaf_alpha").detach().cpu().tolist()
                        if out.get("leaf_alpha") is not None
                        else None
                    ),
                    "dsi_alpha": (
                        out.get("dsi_alpha").detach().cpu().tolist()
                        if out.get("dsi_alpha") is not None
                        else None
                    ),
                    "pi_list": [p.detach().cpu().tolist() for p in out["pi_list"]],
                    "scalar_alpha": (
                        out.get("scalar_alpha").detach().cpu().tolist()
                        if out.get("scalar_alpha") is not None
                        else None
                    ),
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
    {"dataset": "cora", "group": "derived_real"},
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
            best_loss_mean=("best_loss", "mean"),
            nmi_mean=("nmi", "mean"),
            nmi_std=("nmi", "std"),
            ari_mean=("ari", "mean"),
            ari_std=("ari", "std"),
            runtime_sec_mean=("runtime_sec", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
            num_pred_clusters_mean=("num_pred_clusters", "mean"),
        )
        .sort_values(["dataset", "variant"])
    )
    summary.to_csv(out_dir / "summary_by_dataset_variant.csv", index=False)

    group_summary = (
        df.groupby(["group", "variant"], as_index=False)
        .agg(
            datasets=("dataset", "nunique"),
            runs=("seed", "count"),
            nmi_mean=("nmi", "mean"),
            ari_mean=("ari", "mean"),
            best_loss_mean=("best_loss", "mean"),
        )
        .sort_values(["group", "variant"])
    )
    group_summary.to_csv(out_dir / "summary_by_group_variant.csv", index=False)

    wide = summary.pivot(index="dataset", columns="variant", values="nmi_mean")
    if "shared_tree_multi_channel" in wide.columns and "early_scalarized" in wide.columns:
        delta = (wide["shared_tree_multi_channel"] - wide["early_scalarized"]).rename("delta_nmi")
        delta_df = delta.reset_index()
        delta_df.to_csv(out_dir / "delta_shared_minus_scalar.csv", index=False)

    readme_lines = [
        "# Expert3 Dense Shared-Tree Prototype",
        "",
        "This directory contains standalone prototype runs for the expert3 shared-tree multi-channel proposal.",
        "",
        "Variants:",
        "- `early_scalarized`: collapse edge-feature channels first, then run a one-channel shared-tree model.",
        "- `shared_tree_multi_channel`: keep channels separate through leaf conv, assignment, and coarsening; share one tree.",
        "",
        "Files:",
        "- `runs.csv`: per-run metrics and diagnostics.",
        "- `summary_by_dataset_variant.csv`: averaged metrics by dataset and variant.",
        "- `summary_by_group_variant.csv`: grouped averages.",
        "- `delta_shared_minus_scalar.csv`: NMI advantage of multi-channel over early scalarization when both are present.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the expert3 dense shared-tree prototype.")
    parser.add_argument("--datasets", type=str, default="", help="Comma-separated dataset list.")
    parser.add_argument("--variants", type=str, default="early_scalarized,shared_tree_multi_channel")
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
        default="archive/workspaces/expert3_conda_run/results/expert3_shared_tree_proto_v1",
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
