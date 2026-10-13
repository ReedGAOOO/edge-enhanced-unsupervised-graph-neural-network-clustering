#!/usr/bin/env python3
"""
Independent CPU GNN probe for Urban edge semantics.

This is deliberately separate from the main unsupervised DSE model. It asks a
supervised, diagnostic question: given the same Urban graph and node features,
does an edge-conditioned GraphSAGE model perform better than a topology-only
GraphSAGE model, and does that advantage disappear when edge_attr is permuted?
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split


RELATION_FEATURES = [
    "is_street_backed",
    "is_junction_only",
    "is_geom_fallback",
    "is_shared_building",
]


def _parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(False)


def _discover(root: Path, pattern: str) -> list[str]:
    return sorted(p.name for p in root.glob(pattern) if p.is_dir())


def _load_meta(base: Path, dataset: str) -> dict[str, Any]:
    path = base / f"{dataset}_meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _unknown_label(meta: dict[str, Any]) -> int | None:
    mapping = meta.get("label_mapping", {})
    if isinstance(mapping, dict) and "unknown" in mapping:
        try:
            return int(mapping["unknown"])
        except Exception:
            return None
    return None


def _standardize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x = (x - mean) / std
    x[~np.isfinite(x)] = 0.0
    return x.astype(np.float32, copy=False)


def _relation_bits(edge_attr: np.ndarray, feature_names: list[str]) -> np.ndarray:
    idx = [feature_names.index(name) for name in RELATION_FEATURES if name in feature_names]
    if not idx:
        return np.zeros((edge_attr.shape[0], 0), dtype=np.float32)
    return edge_attr[:, idx].astype(np.float32, copy=False)


def _make_splits(labels: np.ndarray, train_ratio: float, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    known_idx = np.where(labels >= 0)[0]
    y_known = labels[known_idx]
    temp_ratio = 1.0 - float(train_ratio)
    if temp_ratio <= 0.0:
        raise ValueError("train_ratio must be < 1")

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


def _macro_class_weights(labels: torch.Tensor, train_idx: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels[train_idx], minlength=int(num_classes)).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    weights = weights / weights.mean().clamp_min(1e-6)
    return weights


class MeanSAGEConv(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int = 0, edge_gated: bool = False):
        super().__init__()
        self.self_lin = nn.Linear(hidden_dim, hidden_dim)
        self.neigh_lin = nn.Linear(hidden_dim, hidden_dim)
        self.edge_gated = bool(edge_gated) and int(edge_dim) > 0
        if self.edge_gated:
            self.edge_mlp = nn.Sequential(
                nn.Linear(int(edge_dim), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.edge_mlp = None

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
        out = out / deg.clamp_min(1.0)
        return self.self_lin(x) + out


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


def _build_condition_edge_attr(
    condition: str,
    edge_attr_np: np.ndarray,
    feature_names: list[str],
    seed: int,
) -> np.ndarray | None:
    if condition in {"mlp", "sage"}:
        return None
    rng = np.random.default_rng(int(seed) + 202)
    if condition == "edge_sage":
        return edge_attr_np
    if condition == "edge_sage_perm":
        return edge_attr_np[rng.permutation(edge_attr_np.shape[0])]
    if condition == "edge_sage_relation_bits":
        rel = _relation_bits(edge_attr_np, feature_names)
        if rel.shape[1] == 0:
            raise ValueError("No relation-bit features found.")
        return rel
    if condition == "edge_sage_relation_bits_perm":
        rel = _relation_bits(edge_attr_np, feature_names)
        if rel.shape[1] == 0:
            raise ValueError("No relation-bit features found.")
        return rel[rng.permutation(rel.shape[0])]
    raise ValueError(f"Unknown condition: {condition}")


def _evaluate(logits: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor) -> dict[str, float]:
    y_true = labels[idx].detach().cpu().numpy()
    y_pred = logits[idx].argmax(dim=1).detach().cpu().numpy()
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _run_one(dataset: str, condition: str, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    _set_seed(seed)
    root = Path(args.root_path)
    base = root / dataset
    meta = _load_meta(base, dataset)
    x_np = np.load(base / f"{dataset}_feat.npy", allow_pickle=False)
    y_raw = np.load(base / f"{dataset}_label.npy", allow_pickle=False)
    edge_index_np = np.load(base / f"{dataset}_edge_index.npy", allow_pickle=False)
    edge_attr_np = np.load(base / f"{dataset}_edge_attr.npy", allow_pickle=False)
    feature_names = list(meta.get("edge_feature_names", []))
    if len(feature_names) != edge_attr_np.shape[1]:
        feature_names = [str(i) for i in range(edge_attr_np.shape[1])]

    unknown = _unknown_label(meta)
    known = y_raw >= 0 if unknown is None else ((y_raw >= 0) & (y_raw != int(unknown)))
    known_classes = sorted(int(x) for x in np.unique(y_raw[known]))
    class_map = {old: new for new, old in enumerate(known_classes)}
    labels_np = np.full_like(y_raw, -1, dtype=np.int64)
    for old, new in class_map.items():
        labels_np[y_raw == old] = int(new)

    train_idx_np, val_idx_np, test_idx_np = _make_splits(labels_np, args.train_ratio, args.val_ratio, seed)
    x = torch.from_numpy(_standardize(x_np)).float()
    edge_index = torch.from_numpy(edge_index_np).long()
    labels = torch.from_numpy(labels_np).long()
    train_idx = torch.from_numpy(train_idx_np).long()
    val_idx = torch.from_numpy(val_idx_np).long()
    test_idx = torch.from_numpy(test_idx_np).long()

    edge_attr_selected = _build_condition_edge_attr(condition, _standardize(edge_attr_np), feature_names, seed)
    edge_attr = torch.from_numpy(_standardize(edge_attr_selected)).float() if edge_attr_selected is not None else None

    num_classes = len(known_classes)
    if condition == "mlp":
        model: nn.Module = MLPProbe(
            in_dim=x.shape[1],
            hidden_dim=args.hidden_dim,
            out_dim=num_classes,
            dropout=args.dropout,
        )
    elif condition == "sage":
        model = SAGEProbe(
            in_dim=x.shape[1],
            hidden_dim=args.hidden_dim,
            out_dim=num_classes,
            edge_dim=0,
            edge_gated=False,
            dropout=args.dropout,
            layers=args.layers,
        )
    else:
        if edge_attr is None:
            raise ValueError(f"{condition} needs edge attributes.")
        model = SAGEProbe(
            in_dim=x.shape[1],
            hidden_dim=args.hidden_dim,
            out_dim=num_classes,
            edge_dim=edge_attr.shape[1],
            edge_gated=True,
            dropout=args.dropout,
            layers=args.layers,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = _macro_class_weights(labels, train_idx, num_classes=num_classes)
    best_state = None
    best_val = -1.0
    best_epoch = 0
    no_improve = 0
    start = time.time()

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index=edge_index, edge_attr=edge_attr)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx], weight=weights)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % int(args.eval_freq) == 0 or epoch == int(args.epochs):
            model.eval()
            with torch.no_grad():
                logits_eval = model(x, edge_index=edge_index, edge_attr=edge_attr)
                val = _evaluate(logits_eval, labels, val_idx)
            score = val["macro_f1"]
            if score > best_val + 1e-8:
                best_val = score
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += int(args.eval_freq)
            if args.patience > 0 and no_improve >= int(args.patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index=edge_index, edge_attr=edge_attr)
        train_metrics = _evaluate(logits, labels, train_idx)
        val_metrics = _evaluate(logits, labels, val_idx)
        test_metrics = _evaluate(logits, labels, test_idx)

    row: dict[str, Any] = {
        "dataset": dataset,
        "city": str(meta.get("city", dataset)),
        "condition": condition,
        "seed": int(seed),
        "n_nodes": int(x.shape[0]),
        "n_known": int((labels_np >= 0).sum()),
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
    return row


def _aggregate(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "edge_gnn_probe_runs.csv", index=False)
    if df.empty:
        return

    summary = (
        df.groupby(["dataset", "condition"], as_index=False)
        .agg(
            runs=("seed", "count"),
            test_acc_mean=("test_acc", "mean"),
            test_acc_std=("test_acc", "std"),
            test_macro_f1_mean=("test_macro_f1", "mean"),
            test_macro_f1_std=("test_macro_f1", "std"),
            test_balanced_acc_mean=("test_balanced_acc", "mean"),
            test_weighted_f1_mean=("test_weighted_f1", "mean"),
            val_macro_f1_mean=("val_macro_f1", "mean"),
            seconds_mean=("seconds", "mean"),
        )
        .sort_values(["dataset", "test_macro_f1_mean"], ascending=[True, False])
    )
    summary.to_csv(out_dir / "edge_gnn_probe_summary.csv", index=False)

    if {"sage", "edge_sage", "edge_sage_perm"}.issubset(set(df["condition"].unique())):
        base = df[df["condition"] == "sage"][["dataset", "seed", "test_macro_f1", "test_acc"]].rename(
            columns={"test_macro_f1": "sage_macro_f1", "test_acc": "sage_acc"}
        )
        perm = df[df["condition"] == "edge_sage_perm"][["dataset", "seed", "test_macro_f1", "test_acc"]].rename(
            columns={"test_macro_f1": "edge_perm_macro_f1", "test_acc": "edge_perm_acc"}
        )
        delta = df.merge(base, on=["dataset", "seed"], how="left").merge(perm, on=["dataset", "seed"], how="left")
        delta["delta_macro_f1_vs_sage"] = delta["test_macro_f1"] - delta["sage_macro_f1"]
        delta["delta_macro_f1_vs_perm"] = delta["test_macro_f1"] - delta["edge_perm_macro_f1"]
        delta["delta_acc_vs_sage"] = delta["test_acc"] - delta["sage_acc"]
        delta["delta_acc_vs_perm"] = delta["test_acc"] - delta["edge_perm_acc"]
        delta.to_csv(out_dir / "edge_gnn_probe_deltas.csv", index=False)

    readme = [
        "# Urban Edge GNN Probe",
        "",
        "Independent supervised CPU probe. It does not use the main DSE model.",
        "",
        "Useful evidence for edge semantics:",
        "- `edge_sage` improves over `sage`.",
        "- `edge_sage` improves over `edge_sage_perm`.",
        "- The improvement is consistent across seeds and at least two cities.",
        "",
        "If these checks fail, Urban edge_attr is currently too weak or too misaligned with labels to support a main edge-modeling claim.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent Urban edge-aware GNN probe on CPU.")
    parser.add_argument("--root_path", type=str, default="data")
    parser.add_argument("--datasets", type=str, default="urban_paris_plot_v3bsjg,urban_boston_plot_v3bsjg,urban_washingtondc_plot_v3bsjg")
    parser.add_argument("--pattern", type=str, default="urban_*_plot_v3bsjg")
    parser.add_argument("--conditions", type=str, default="mlp,sage,edge_sage,edge_sage_perm")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--omp_threads", type=int, default=8)
    parser.add_argument("--out_dir", type=str, default="results/edge_probe/urban_edge_gnn_v1")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(max(1, int(args.omp_threads)))
    os.environ["MKL_NUM_THREADS"] = str(max(1, int(args.omp_threads)))

    root = Path(args.root_path)
    if str(args.datasets).strip().lower() == "all":
        datasets = _discover(root, args.pattern)
    else:
        datasets = _parse_csv(args.datasets)
    conditions = _parse_csv(args.conditions)
    seeds = _parse_int_csv(args.seeds)
    if not datasets or not conditions or not seeds:
        raise ValueError("datasets, conditions, and seeds must be non-empty.")

    out_dir = Path(args.out_dir)
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for condition in conditions:
            for seed in seeds:
                print(f"[run] dataset={dataset} condition={condition} seed={seed}", flush=True)
                row = _run_one(dataset=dataset, condition=condition, seed=seed, args=args)
                rows.append(row)
                _aggregate(out_dir, rows)
                print(
                    f"[done] test_macro_f1={row['test_macro_f1']:.4f} "
                    f"test_acc={row['test_acc']:.4f} seconds={row['seconds']:.1f}",
                    flush=True,
                )

    _aggregate(out_dir, rows)
    print(f"[ok] runs={out_dir / 'edge_gnn_probe_runs.csv'}")
    print(f"[ok] summary={out_dir / 'edge_gnn_probe_summary.csv'}")


if __name__ == "__main__":
    main()
