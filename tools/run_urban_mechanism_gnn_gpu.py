#!/usr/bin/env python3
"""GPU GNN mechanism probes for selected Urban cities.

The goal is not to run the main clustering model. This script asks a stricter
diagnostic question: after topology, geometry, distance-bin permutation, and
relation interventions are controlled, does edge information still improve a
supervised GraphSAGE-style probe on known land-use labels?
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split


RELATION_BITS = [
    "is_street_backed",
    "is_junction_only",
    "is_geom_fallback",
    "is_shared_building",
]

SEMANTIC_RELATION_BITS = [
    "is_street_backed",
    "is_junction_only",
    "is_shared_building",
]

SPATIAL_FEATURES = [
    "is_geom_fallback",
    "centroid_dist_m",
    "log_area_gap",
    "orientation_diff_deg",
]

ENDPOINT_SIMILARITY_FEATURES = [
    "node_feat_cosine",
    "node_feat_l2",
]

SEMANTIC_FEATURES = [
    "is_street_backed",
    "is_junction_only",
    "is_shared_building",
    "shared_street_cnt",
    "shared_junction_cnt",
    "jaccard_street",
    "jaccard_junction",
    "shared_street_len_mean",
    "street_count_sum",
    "street_count_gap",
    "junction_count_sum",
    "junction_count_gap",
    "building_count_sum",
    "building_count_gap",
    "is_building_neighbor",
    "building_neighbor_cnt",
    "building_dist_mean",
    "building_dist_min",
    "building_count_gap",
    "building_area_mean_gap",
    "building_height_mean_gap",
]

BASE_GRAPH_ROOT = "/mnt/e/wsl-project-data/edge-enhanced-unsupervised-graph-neural-network-clustering/data"
DERIVED_GRAPH_ROOT = "/mnt/e/wsl-project-data/edge-enhanced-unsupervised-graph-neural-network-clustering/data/urban_factor_experiments"


@dataclass
class GraphSpec:
    graph_family: str
    graph_condition: str
    root: Path
    dataset: str
    attr_conditions: list[str]


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(False)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def unknown_label(meta: dict[str, Any]) -> int | None:
    mapping = meta.get("label_mapping", {})
    if isinstance(mapping, dict) and "unknown" in mapping:
        try:
            return int(mapping["unknown"])
        except Exception:
            return None
    return None


def standardize_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    x[~np.isfinite(x)] = 0.0
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    out = (x - mu) / sd
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32, copy=False)


def feature_indices(feature_names: list[str], names: list[str]) -> list[int]:
    return [feature_names.index(name) for name in names if name in feature_names]


def make_splits(labels: np.ndarray, train_ratio: float, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    known_idx = np.where(labels >= 0)[0]
    y_known = labels[known_idx]
    temp_ratio = 1.0 - float(train_ratio)
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        known_idx,
        y_known,
        test_size=temp_ratio,
        random_state=int(seed),
        stratify=y_known,
    )
    val_share = float(val_ratio) / max(temp_ratio, 1e-9)
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx,
        y_temp,
        test_size=1.0 - val_share,
        random_state=int(seed) + 13,
        stratify=y_temp,
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def macro_class_weights(labels: torch.Tensor, train_idx: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels[train_idx], minlength=int(num_classes)).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    return weights / weights.mean().clamp_min(1e-6)


class MeanSAGEConv(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int = 0, edge_gated: bool = False):
        super().__init__()
        self.self_lin = nn.Linear(hidden_dim, hidden_dim)
        self.neigh_lin = nn.Linear(hidden_dim, hidden_dim)
        self.edge_gated = bool(edge_gated) and int(edge_dim) > 0
        self.edge_mlp = None
        if self.edge_gated:
            self.edge_mlp = nn.Sequential(
                nn.Linear(int(edge_dim), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        src = edge_index[0].long()
        dst = edge_index[1].long()
        msg = self.neigh_lin(x[src])
        if self.edge_mlp is not None and edge_attr is not None:
            edge_vec = self.edge_mlp(edge_attr)
            msg = torch.sigmoid(edge_vec) * msg + edge_vec
        out = torch.zeros_like(x)
        out.index_add_(0, dst, msg)
        deg = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        deg.index_add_(0, dst, torch.ones((dst.shape[0], 1), dtype=x.dtype, device=x.device))
        return self.self_lin(x) + out / deg.clamp_min(1.0)


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None, edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(x)


class SAGEProbe(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, edge_dim: int, edge_gated: bool, dropout: float, layers: int):
        super().__init__()
        self.input_lin = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [MeanSAGEConv(hidden_dim=hidden_dim, edge_dim=edge_dim, edge_gated=edge_gated) for _ in range(max(1, int(layers)))]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in self.convs])
        self.dropout = float(dropout)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        h = F.relu(self.input_lin(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        for conv, norm in zip(self.convs, self.norms):
            residual = h
            h = conv(h, edge_index=edge_index, edge_attr=edge_attr)
            h = norm(h)
            h = F.relu(h + residual)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h)


def evaluate(logits: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor) -> dict[str, float]:
    y_true = labels[idx].detach().cpu().numpy()
    y_pred = logits[idx].argmax(dim=1).detach().cpu().numpy()
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def load_graph(root: Path, dataset: str) -> dict[str, Any]:
    base = root / dataset
    meta = load_json(base / f"{dataset}_meta.json")
    x_np = np.load(base / f"{dataset}_feat.npy", allow_pickle=False)
    y_raw = np.load(base / f"{dataset}_label.npy", allow_pickle=False)
    edge_index_np = np.load(base / f"{dataset}_edge_index.npy", allow_pickle=False)
    edge_attr_np = np.load(base / f"{dataset}_edge_attr.npy", allow_pickle=False)
    feature_names = list(meta.get("edge_feature_names", []))
    if len(feature_names) != edge_attr_np.shape[1]:
        feature_names = [str(i) for i in range(edge_attr_np.shape[1])]
    unk = unknown_label(meta)
    known = y_raw >= 0 if unk is None else ((y_raw >= 0) & (y_raw != int(unk)))
    known_classes = sorted(int(x) for x in np.unique(y_raw[known]))
    class_map = {old: new for new, old in enumerate(known_classes)}
    labels_np = np.full_like(y_raw, -1, dtype=np.int64)
    for old, new in class_map.items():
        labels_np[y_raw == old] = int(new)
    return {
        "meta": meta,
        "x_np": x_np,
        "labels_np": labels_np,
        "edge_index_np": edge_index_np,
        "edge_attr_np": edge_attr_np,
        "feature_names": feature_names,
        "num_classes": len(known_classes),
    }


def distance_bins(edge_attr_np: np.ndarray, feature_names: list[str], n_bins: int) -> np.ndarray:
    idx = feature_names.index("centroid_dist_m") if "centroid_dist_m" in feature_names else None
    if idx is None:
        return np.zeros((edge_attr_np.shape[0],), dtype=np.int16)
    vals = edge_attr_np[:, idx].astype(np.float64, copy=False)
    finite = np.isfinite(vals)
    out = np.zeros((edge_attr_np.shape[0],), dtype=np.int16)
    if finite.sum() < max(10, int(n_bins)):
        return out
    try:
        out[finite] = pd.qcut(vals[finite], q=min(int(n_bins), np.unique(vals[finite]).size), labels=False, duplicates="drop").astype(np.int16)
    except Exception:
        return np.zeros((edge_attr_np.shape[0],), dtype=np.int16)
    return out


def permute_columns_within_bins(
    edge_attr: np.ndarray,
    columns: list[int],
    bins: np.ndarray,
    seed: int,
) -> np.ndarray:
    out = edge_attr.copy()
    if not columns:
        return out
    rng = np.random.default_rng(int(seed) + 929)
    for b in np.unique(bins):
        idx = np.flatnonzero(bins == b)
        if idx.size <= 1:
            continue
        perm = rng.permutation(idx)
        out[np.ix_(idx, columns)] = out[np.ix_(perm, columns)]
    return out


def build_edge_attr(condition: str, edge_attr_np: np.ndarray, feature_names: list[str], seed: int, distance_n_bins: int) -> np.ndarray | None:
    cond = condition.strip()
    if cond in {"mlp", "sage_topology"}:
        return None
    rng = np.random.default_rng(int(seed) + 202)
    full = edge_attr_np.astype(np.float32, copy=True)
    if cond == "edge_full":
        return full
    if cond == "edge_full_perm_global":
        return full[rng.permutation(full.shape[0])]
    spatial_idx = feature_indices(feature_names, SPATIAL_FEATURES)
    no_spatial_idx = [i for i in range(full.shape[1]) if i not in set(spatial_idx)]
    if cond == "edge_nonspatial_perm_distance_bin":
        bins = distance_bins(full, feature_names, n_bins=distance_n_bins)
        return permute_columns_within_bins(full, no_spatial_idx, bins, seed=seed)
    if cond == "edge_no_spatial":
        out = full.copy()
        out[:, spatial_idx] = 0.0
        return out
    if cond == "edge_no_spatial_or_similarity":
        out = full.copy()
        idx = sorted(set(spatial_idx + feature_indices(feature_names, ENDPOINT_SIMILARITY_FEATURES)))
        out[:, idx] = 0.0
        return out
    if cond == "edge_spatial_only":
        idx = spatial_idx
        if not idx:
            raise ValueError("No spatial edge features found.")
        return full[:, idx]
    if cond == "edge_relation_bits":
        idx = feature_indices(feature_names, RELATION_BITS)
        if not idx:
            raise ValueError("No relation-bit edge features found.")
        return full[:, idx]
    if cond == "edge_semantic_relation_bits":
        idx = feature_indices(feature_names, SEMANTIC_RELATION_BITS)
        if not idx:
            raise ValueError("No semantic relation-bit edge features found.")
        return full[:, idx]
    if cond == "edge_semantic_features_no_spatial":
        idx = feature_indices(feature_names, SEMANTIC_FEATURES)
        idx = [i for i in idx if i not in set(spatial_idx)]
        if not idx:
            raise ValueError("No semantic non-spatial edge features found.")
        return full[:, idx]
    raise ValueError(f"Unknown attr condition: {condition}")


def graph_specs_for_city(city: str, base_root: Path, derived_root: Path, groups: set[str]) -> list[GraphSpec]:
    base = f"urban_{city}_plot_v3bsjg"
    specs: list[GraphSpec] = []
    native_conditions = [
        "mlp",
        "sage_topology",
        "edge_full",
        "edge_full_perm_global",
        "edge_nonspatial_perm_distance_bin",
        "edge_no_spatial",
        "edge_no_spatial_or_similarity",
        "edge_spatial_only",
        "edge_relation_bits",
        "edge_semantic_relation_bits",
        "edge_semantic_features_no_spatial",
    ]
    if "native_attr" in groups:
        specs.append(GraphSpec("native_attr", "v3b_original", base_root, base, native_conditions))
    if "relation_intervention" in groups:
        rel_names = [
            ("v3b_original", base_root, base),
            ("drop_street", derived_root, f"{base}_drop_street"),
            ("drop_junction", derived_root, f"{base}_drop_junction"),
            ("drop_geom", derived_root, f"{base}_drop_geom"),
            ("drop_building", derived_root, f"{base}_drop_building"),
            ("zero_relation_bits", derived_root, f"{base}_zero_relation_bits"),
        ]
        for cond, root, dataset in rel_names:
            specs.append(GraphSpec("relation_intervention", cond, root, dataset, ["sage_topology", "edge_full"]))
    if "v4_construction" in groups:
        for cond in ["v4sl", "v4slj", "v4sljb", "v4g"]:
            dataset = f"urban_{city}_plot_v4factor_{cond}"
            specs.append(GraphSpec("v4_construction", cond, derived_root, dataset, ["sage_topology", "edge_full"]))
    return specs


def run_one(
    *,
    graph: dict[str, Any],
    dataset: str,
    city: str,
    graph_family: str,
    graph_condition: str,
    attr_condition: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(seed)
    train_idx_np, val_idx_np, test_idx_np = make_splits(graph["labels_np"], args.train_ratio, args.val_ratio, seed)
    x = torch.from_numpy(standardize_np(graph["x_np"])).float().to(device)
    labels = torch.from_numpy(graph["labels_np"]).long().to(device)
    edge_index = torch.from_numpy(graph["edge_index_np"]).long().to(device)
    train_idx = torch.from_numpy(train_idx_np).long().to(device)
    val_idx = torch.from_numpy(val_idx_np).long().to(device)
    test_idx = torch.from_numpy(test_idx_np).long().to(device)

    edge_attr_selected = build_edge_attr(
        attr_condition,
        graph["edge_attr_np"],
        graph["feature_names"],
        seed=seed,
        distance_n_bins=int(args.distance_bins),
    )
    edge_attr = None
    if edge_attr_selected is not None:
        edge_attr = torch.from_numpy(standardize_np(edge_attr_selected)).float().to(device)

    num_classes = int(graph["num_classes"])
    if attr_condition == "mlp":
        model: nn.Module = MLPProbe(x.shape[1], args.hidden_dim, num_classes, args.dropout).to(device)
    elif attr_condition == "sage_topology":
        model = SAGEProbe(x.shape[1], args.hidden_dim, num_classes, 0, False, args.dropout, args.layers).to(device)
    else:
        if edge_attr is None:
            raise ValueError(f"{attr_condition} requires edge_attr.")
        model = SAGEProbe(x.shape[1], args.hidden_dim, num_classes, edge_attr.shape[1], True, args.dropout, args.layers).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = macro_class_weights(labels, train_idx, num_classes)
    best_state = None
    best_val = -1.0
    best_epoch = 0
    no_improve = 0
    start = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, edge_index=edge_index, edge_attr=edge_attr)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx], weight=weights)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % int(args.eval_freq) == 0 or epoch == int(args.epochs):
            model.eval()
            with torch.no_grad():
                val_logits = model(x, edge_index=edge_index, edge_attr=edge_attr)
                val_metrics = evaluate(val_logits, labels, val_idx)
            score = val_metrics["macro_f1"]
            if score > best_val + 1e-8:
                best_val = score
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += int(args.eval_freq)
            if int(args.patience) > 0 and no_improve >= int(args.patience):
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index=edge_index, edge_attr=edge_attr)
        train_metrics = evaluate(logits, labels, train_idx)
        val_metrics = evaluate(logits, labels, val_idx)
        test_metrics = evaluate(logits, labels, test_idx)

    return {
        "city": city,
        "dataset": dataset,
        "graph_family": graph_family,
        "graph_condition": graph_condition,
        "attr_condition": attr_condition,
        "seed": int(seed),
        "device": str(device),
        "n_nodes": int(x.shape[0]),
        "n_known": int((graph["labels_np"] >= 0).sum()),
        "n_edges_directed": int(edge_index.shape[1]),
        "n_classes": int(num_classes),
        "edge_attr_dim_used": int(edge_attr.shape[1]) if edge_attr is not None else 0,
        "best_epoch": int(best_epoch),
        "seconds": float(time.time() - start),
        "train_acc": train_metrics["acc"],
        "train_macro_f1": train_metrics["macro_f1"],
        "val_acc": val_metrics["acc"],
        "val_macro_f1": val_metrics["macro_f1"],
        "test_acc": test_metrics["acc"],
        "test_balanced_acc": test_metrics["balanced_acc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
    }


def aggregate(out_dir: Path, rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(out_dir / "gpu_gnn_mechanism_runs.csv", index=False)
    failure_cols = ["city", "dataset", "graph_family", "graph_condition", "attr_condition", "seed", "error"]
    pd.DataFrame(failures, columns=failure_cols).to_csv(out_dir / "failures.csv", index=False)
    if runs.empty:
        return
    summary = (
        runs.groupby(["city", "graph_family", "graph_condition", "attr_condition"], as_index=False)
        .agg(
            runs=("seed", "count"),
            test_macro_f1_mean=("test_macro_f1", "mean"),
            test_macro_f1_std=("test_macro_f1", "std"),
            test_balanced_acc_mean=("test_balanced_acc", "mean"),
            test_acc_mean=("test_acc", "mean"),
            val_macro_f1_mean=("val_macro_f1", "mean"),
            seconds_mean=("seconds", "mean"),
            edge_attr_dim_used=("edge_attr_dim_used", "max"),
        )
        .sort_values(["city", "graph_family", "graph_condition", "test_macro_f1_mean"], ascending=[True, True, True, False])
    )
    summary.to_csv(out_dir / "gpu_gnn_mechanism_summary.csv", index=False)

    city_rows = []
    for city, cdf in summary.groupby("city"):
        def m(family: str, graph_condition: str, attr_condition: str) -> float:
            cur = cdf[
                (cdf["graph_family"] == family)
                & (cdf["graph_condition"] == graph_condition)
                & (cdf["attr_condition"] == attr_condition)
            ]
            if cur.empty:
                return float("nan")
            return float(cur["test_macro_f1_mean"].iloc[0])

        native_topo = m("native_attr", "v3b_original", "sage_topology")
        native_full = m("native_attr", "v3b_original", "edge_full")
        native_perm = m("native_attr", "v3b_original", "edge_full_perm_global")
        dist_perm = m("native_attr", "v3b_original", "edge_nonspatial_perm_distance_bin")
        no_spatial = m("native_attr", "v3b_original", "edge_no_spatial")
        spatial_only = m("native_attr", "v3b_original", "edge_spatial_only")
        semantic_no_spatial = m("native_attr", "v3b_original", "edge_semantic_features_no_spatial")
        semantic_bits = m("native_attr", "v3b_original", "edge_semantic_relation_bits")

        relation_deltas = {}
        for condition in ["drop_street", "drop_junction", "drop_geom", "drop_building", "zero_relation_bits"]:
            val = m("relation_intervention", condition, "edge_full")
            relation_deltas[f"{condition}_delta_vs_original"] = val - native_full if np.isfinite(val) and np.isfinite(native_full) else float("nan")
        relation_delta_vals = [abs(v) for v in relation_deltas.values() if np.isfinite(v)]

        v4g = m("v4_construction", "v4g", "edge_full")
        semantic_v4_vals = [m("v4_construction", cond, "edge_full") for cond in ["v4sl", "v4slj", "v4sljb"]]
        semantic_v4_vals = [x for x in semantic_v4_vals if np.isfinite(x)]
        best_semantic_v4 = max(semantic_v4_vals) if semantic_v4_vals else float("nan")

        full_minus_topology = native_full - native_topo if np.isfinite(native_full) and np.isfinite(native_topo) else float("nan")
        full_minus_global_perm = native_full - native_perm if np.isfinite(native_full) and np.isfinite(native_perm) else float("nan")
        full_minus_distance_perm = native_full - dist_perm if np.isfinite(native_full) and np.isfinite(dist_perm) else float("nan")
        no_spatial_minus_topology = no_spatial - native_topo if np.isfinite(no_spatial) and np.isfinite(native_topo) else float("nan")
        spatial_only_minus_topology = spatial_only - native_topo if np.isfinite(spatial_only) and np.isfinite(native_topo) else float("nan")
        semantic_minus_spatial = semantic_no_spatial - spatial_only if np.isfinite(semantic_no_spatial) and np.isfinite(spatial_only) else float("nan")
        semantic_bits_minus_spatial = semantic_bits - spatial_only if np.isfinite(semantic_bits) and np.isfinite(spatial_only) else float("nan")
        v4_semantic_minus_geom = best_semantic_v4 - v4g if np.isfinite(best_semantic_v4) and np.isfinite(v4g) else float("nan")

        city_rows.append(
            {
                "city": city,
                "native_topology_macro_f1": native_topo,
                "native_edge_full_macro_f1": native_full,
                "native_edge_perm_global_macro_f1": native_perm,
                "native_edge_distance_bin_perm_macro_f1": dist_perm,
                "native_edge_no_spatial_macro_f1": no_spatial,
                "native_edge_spatial_only_macro_f1": spatial_only,
                "native_edge_semantic_no_spatial_macro_f1": semantic_no_spatial,
                "native_edge_semantic_bits_macro_f1": semantic_bits,
                "full_minus_topology_macro_f1": full_minus_topology,
                "full_minus_global_perm_macro_f1": full_minus_global_perm,
                "full_minus_distance_perm_macro_f1": full_minus_distance_perm,
                "no_spatial_minus_topology_macro_f1": no_spatial_minus_topology,
                "spatial_only_minus_topology_macro_f1": spatial_only_minus_topology,
                "semantic_no_spatial_minus_spatial_only_macro_f1": semantic_minus_spatial,
                "semantic_bits_minus_spatial_only_macro_f1": semantic_bits_minus_spatial,
                "v4g_edge_full_macro_f1": v4g,
                "best_semantic_v4_edge_full_macro_f1": best_semantic_v4,
                "best_semantic_v4_minus_v4g_macro_f1": v4_semantic_minus_geom,
                "max_abs_relation_intervention_delta_macro_f1": max(relation_delta_vals) if relation_delta_vals else float("nan"),
                **relation_deltas,
                "gpu_edge_attr_supported": bool(
                    np.isfinite(full_minus_topology)
                    and np.isfinite(full_minus_global_perm)
                    and full_minus_topology > 0.01
                    and full_minus_global_perm > 0.01
                ),
                "relation_semantics_supported": bool(
                    (np.isfinite(full_minus_distance_perm) and full_minus_distance_perm > 0.01)
                    or (np.isfinite(semantic_minus_spatial) and semantic_minus_spatial > 0.01)
                    or (np.isfinite(semantic_bits_minus_spatial) and semantic_bits_minus_spatial > 0.01)
                ),
                "geometry_confound_persists": bool(
                    (np.isfinite(v4_semantic_minus_geom) and v4_semantic_minus_geom <= 0.005)
                    or (np.isfinite(spatial_only_minus_topology) and np.isfinite(no_spatial_minus_topology) and spatial_only_minus_topology >= no_spatial_minus_topology - 0.005)
                ),
                "relation_rule_sensitive": bool(relation_delta_vals and max(relation_delta_vals) > 0.02),
            }
        )
    pd.DataFrame(city_rows).to_csv(out_dir / "gpu_gnn_mechanism_city_summary.csv", index=False)


def write_readme(out_dir: Path, args: argparse.Namespace, device: torch.device) -> None:
    text = f"""# Urban Mechanism GNN GPU Probe

Scope: `{args.cities}`

Device: `{device}`

This package runs supervised GPU GNN diagnostics on selected Urban cities. It is
not the main clustering model. It tests whether edge attributes and graph
projection choices still matter after geometry and relation controls.

Key controls:

- `edge_full` vs `sage_topology`: whether edge attributes help beyond topology.
- `edge_full` vs `edge_full_perm_global`: whether edge attributes are used.
- `edge_full` vs `edge_nonspatial_perm_distance_bin`: whether non-spatial edge
  semantics survive a distance-bin controlled permutation.
- `edge_spatial_only` vs `edge_semantic_features_no_spatial`: whether relation
  semantics beat geometry-only features.
- `drop_street/drop_junction/drop_geom/drop_building/zero_relation_bits`: whether
  graph projection rules dominate the result.
- `v4sl/v4slj/v4sljb` vs `v4g`: whether semantic construction beats geometric
  kNN construction.

Important files:

- `gpu_gnn_mechanism_runs.csv`
- `gpu_gnn_mechanism_summary.csv`
- `gpu_gnn_mechanism_city_summary.csv`
- `failures.csv`
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_root = Path(args.base_graph_root)
    derived_root = Path(args.derived_graph_root)
    cities = parse_csv(args.cities)
    seeds = parse_int_csv(args.seeds)
    groups = set(parse_csv(args.groups))
    if "all" in groups:
        groups = {"native_attr", "relation_intervention", "v4_construction"}

    requested = args.device
    if requested == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(requested)
    if device.type == "cuda":
        torch.cuda.set_device(int(args.cuda_index))
        torch.backends.cuda.matmul.allow_tf32 = True

    manifest = {
        "cities": cities,
        "seeds": seeds,
        "groups": sorted(groups),
        "base_graph_root": str(base_root),
        "derived_graph_root": str(derived_root),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "args": vars(args),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(out_dir, args, device)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    graph_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for city in cities:
        specs = graph_specs_for_city(city, base_root, derived_root, groups)
        for spec in specs:
            key = (str(spec.root), spec.dataset)
            try:
                if key not in graph_cache:
                    graph_cache[key] = load_graph(spec.root, spec.dataset)
                graph = graph_cache[key]
            except Exception as exc:
                failures.append(
                    {
                        "city": city,
                        "dataset": spec.dataset,
                        "graph_family": spec.graph_family,
                        "graph_condition": spec.graph_condition,
                        "attr_condition": "",
                        "seed": "",
                        "error": str(exc),
                    }
                )
                aggregate(out_dir, rows, failures)
                continue
            for attr_condition in spec.attr_conditions:
                for seed in seeds:
                    try:
                        print(
                            f"[run] city={city} graph={spec.graph_family}/{spec.graph_condition} "
                            f"attr={attr_condition} seed={seed} device={device}",
                            flush=True,
                        )
                        row = run_one(
                            graph=graph,
                            dataset=spec.dataset,
                            city=city,
                            graph_family=spec.graph_family,
                            graph_condition=spec.graph_condition,
                            attr_condition=attr_condition,
                            seed=seed,
                            args=args,
                            device=device,
                        )
                        rows.append(row)
                        aggregate(out_dir, rows, failures)
                        print(
                            f"[done] macro_f1={row['test_macro_f1']:.4f} "
                            f"bal_acc={row['test_balanced_acc']:.4f} sec={row['seconds']:.1f}",
                            flush=True,
                        )
                    except RuntimeError as exc:
                        if "out of memory" in str(exc).lower() and device.type == "cuda":
                            torch.cuda.empty_cache()
                        failures.append(
                            {
                                "city": city,
                                "dataset": spec.dataset,
                                "graph_family": spec.graph_family,
                                "graph_condition": spec.graph_condition,
                                "attr_condition": attr_condition,
                                "seed": seed,
                                "error": str(exc),
                            }
                        )
                        aggregate(out_dir, rows, failures)
                    except Exception as exc:
                        failures.append(
                            {
                                "city": city,
                                "dataset": spec.dataset,
                                "graph_family": spec.graph_family,
                                "graph_condition": spec.graph_condition,
                                "attr_condition": attr_condition,
                                "seed": seed,
                                "error": str(exc),
                            }
                        )
                        aggregate(out_dir, rows, failures)
    aggregate(out_dir, rows, failures)
    return {
        "out_dir": str(out_dir),
        "runs": len(rows),
        "failures": len(failures),
        "device": str(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPU GNN mechanism probes for selected Urban cities.")
    parser.add_argument("--cities", type=str, default="singapore,washingtondc,shanghai")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--groups", type=str, default="all")
    parser.add_argument("--base_graph_root", type=str, default=BASE_GRAPH_ROOT)
    parser.add_argument("--derived_graph_root", type=str, default=DERIVED_GRAPH_ROOT)
    parser.add_argument("--out_dir", type=str, default="results/urban_mechanism_gpu/tri_city")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cuda_index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--distance_bins", type=int, default=8)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
