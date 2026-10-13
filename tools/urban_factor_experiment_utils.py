#!/usr/bin/env python3
"""Shared helpers for Urban factor experiment packages."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CITY_ORDER = [
    "bangkok",
    "beijing",
    "boston",
    "chicago",
    "johannesburg",
    "madrid",
    "melbourne",
    "paris",
    "shanghai",
    "singapore",
    "sydney",
    "tokyo",
    "washingtondc",
]

CITY_DISPLAY = {
    "washingtondc": "WashingtonDC",
    "tokyo": "Tokyo",
    "paris": "Paris",
    "boston": "Boston",
    "madrid": "Madrid",
    "beijing": "Beijing",
    "shanghai": "Shanghai",
    "singapore": "Singapore",
    "melbourne": "Melbourne",
    "sydney": "Sydney",
    "bangkok": "Bangkok",
    "chicago": "Chicago",
    "johannesburg": "Johannesburg",
}

RELATION_FEATURES = [
    "is_street_backed",
    "is_junction_only",
    "is_geom_fallback",
    "is_shared_building",
]

RELATION_ALIASES = {
    "street": "is_street_backed",
    "junction": "is_junction_only",
    "geom": "is_geom_fallback",
    "geometry": "is_geom_fallback",
    "building": "is_shared_building",
    "shared_building": "is_shared_building",
}

COARSE_LABEL_GROUPS = {
    "commercial_like": {"Commercial", "Entertainment"},
    "civic_institutional": {"Civic", "Institutional"},
    "recreation_social": {"Recreational", "Social"},
    "food_healthcare": {"Food", "Healthcare"},
}


@dataclass
class SparseGraph:
    dataset: str
    city: str
    root: Path
    x: np.ndarray
    labels: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    edge_weight: np.ndarray | None
    meta: dict[str, Any]
    feature_names: list[str]


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_float_csv(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_cities(text: str, graph_root: Path | None = None, pattern: str = "urban_*_plot_v3bsjg") -> list[str]:
    if str(text).strip().lower() == "all":
        if graph_root is None:
            return list(CITY_ORDER)
        found = []
        for p in sorted(graph_root.glob(pattern)):
            if p.is_dir():
                city = dataset_to_city(p.name)
                if city:
                    found.append(city)
        return found or list(CITY_ORDER)
    return parse_csv(text)


def dataset_to_city(dataset: str) -> str:
    name = dataset
    if name.startswith("urban_"):
        name = name[len("urban_") :]
    if "_plot_" in name:
        name = name.split("_plot_", 1)[0]
    return name


def city_to_dataset(city: str, variant: str = "v3bsjg") -> str:
    return f"urban_{city}_plot_{variant}"


def city_label(city: str) -> str:
    return CITY_DISPLAY.get(city, city.title())


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_sparse_graph(root: Path, dataset: str) -> SparseGraph:
    base = root / dataset
    if not base.exists():
        raise FileNotFoundError(f"Dataset directory not found: {base}")
    meta = load_json(base / f"{dataset}_meta.json")
    x = np.load(base / f"{dataset}_feat.npy", allow_pickle=False).astype(np.float32, copy=False)
    labels = np.load(base / f"{dataset}_label.npy", allow_pickle=False)
    edge_index = np.load(base / f"{dataset}_edge_index.npy", allow_pickle=False)
    edge_attr = np.load(base / f"{dataset}_edge_attr.npy", allow_pickle=False).astype(np.float32, copy=False)
    ew_path = base / f"{dataset}_edge_weight.npy"
    edge_weight = np.load(ew_path, allow_pickle=False).astype(np.float32, copy=False) if ew_path.exists() else None
    feature_names = list(meta.get("edge_feature_names", []))
    if len(feature_names) != edge_attr.shape[1]:
        feature_names = [str(i) for i in range(edge_attr.shape[1])]
    city = str(meta.get("city", dataset_to_city(dataset)))
    return SparseGraph(
        dataset=dataset,
        city=city,
        root=root,
        x=x,
        labels=labels,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_weight=edge_weight,
        meta=meta,
        feature_names=feature_names,
    )


def unknown_label(meta: dict[str, Any]) -> int | None:
    mapping = meta.get("label_mapping", {})
    if isinstance(mapping, dict) and "unknown" in mapping:
        try:
            return int(mapping["unknown"])
        except Exception:
            return None
    return None


def label_id_to_name(meta: dict[str, Any]) -> dict[int, str]:
    out = {}
    for name, idx in meta.get("label_mapping", {}).items():
        try:
            out[int(idx)] = str(name)
        except Exception:
            continue
    return out


def coarse_labels(labels: np.ndarray, meta: dict[str, Any]) -> tuple[np.ndarray, dict[str, int]]:
    inv = label_id_to_name(meta)
    coarse_mapping = {
        "commercial_like": 0,
        "civic_institutional": 1,
        "recreation_social": 2,
        "food_healthcare": 3,
        "other": 4,
        "unknown": 5,
    }
    out = np.full(labels.shape, coarse_mapping["unknown"], dtype=np.int64)
    unk = unknown_label(meta)
    for i, val in enumerate(labels):
        if unk is not None and int(val) == int(unk):
            out[i] = coarse_mapping["unknown"]
            continue
        name = inv.get(int(val), "unknown")
        assigned = False
        for group_name, members in COARSE_LABEL_GROUPS.items():
            if name in members:
                out[i] = coarse_mapping[group_name]
                assigned = True
                break
        if not assigned:
            out[i] = coarse_mapping["other"]
    return out, coarse_mapping


def known_mask(labels: np.ndarray, meta: dict[str, Any], unknown: int | None = None) -> np.ndarray:
    unk = unknown_label(meta) if unknown is None else unknown
    if unk is None:
        return labels >= 0
    return (labels >= 0) & (labels != int(unk))


def undirected_unique(
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    edge_weight: np.ndarray | None,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    src = edge_index[0].astype(np.int64, copy=False)
    dst = edge_index[1].astype(np.int64, copy=False)
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    key = lo * int(num_nodes) + hi
    _, keep = np.unique(key, return_index=True)
    keep = np.sort(keep)
    edge_index_u = np.stack([lo[keep], hi[keep]], axis=0)
    edge_attr_u = edge_attr[keep]
    edge_weight_u = edge_weight[keep] if edge_weight is not None else None
    return edge_index_u, edge_attr_u, edge_weight_u


def endpoint_features(x: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    xi = x[src].astype(np.float32, copy=False)
    xj = x[dst].astype(np.float32, copy=False)
    return np.concatenate([0.5 * (xi + xj), np.abs(xi - xj), xi * xj], axis=1).astype(np.float32, copy=False)


def relation_feature_indices(feature_names: list[str]) -> list[int]:
    return [feature_names.index(name) for name in RELATION_FEATURES if name in feature_names]


def feature_index(feature_names: list[str], name: str) -> int | None:
    key = RELATION_ALIASES.get(name, name)
    if key in feature_names:
        return int(feature_names.index(key))
    return None


def relation_bits(edge_attr: np.ndarray, feature_names: list[str]) -> np.ndarray:
    idx = relation_feature_indices(feature_names)
    if not idx:
        return np.zeros((edge_attr.shape[0], 0), dtype=np.float32)
    return edge_attr[:, idx].astype(np.float32, copy=False)


def fit_eval(X: np.ndarray, y: np.ndarray, seed: int, test_size: float = 0.3) -> dict[str, float]:
    if X.shape[0] < 20 or X.shape[1] == 0 or len(np.unique(y)) < 2:
        return {"auc": np.nan, "ap": np.nan, "acc": np.nan, "log_loss": np.nan}
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=float(test_size),
            random_state=int(seed),
            stratify=y,
        )
    except ValueError:
        return {"auc": np.nan, "ap": np.nan, "acc": np.nan, "log_loss": np.nan}
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, tol=1e-3, class_weight="balanced", solver="lbfgs"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(np.int64)
    return {
        "auc": float(roc_auc_score(y_test, prob)),
        "ap": float(average_precision_score(y_test, prob)),
        "acc": float(accuracy_score(y_test, pred)),
        "log_loss": float(log_loss(y_test, prob, labels=[0, 1])),
    }


def prepare_known_edges(
    graph: SparseGraph,
    labels: np.ndarray | None = None,
    known: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_node = graph.labels if labels is None else labels
    kmask = known_mask(y_node, graph.meta) if known is None else known
    edge_index, edge_attr, _ = undirected_unique(
        graph.edge_index,
        graph.edge_attr,
        graph.edge_weight,
        num_nodes=graph.x.shape[0],
    )
    src = edge_index[0]
    dst = edge_index[1]
    keep = kmask[src] & kmask[dst]
    src = src[keep]
    dst = dst[keep]
    edge_attr = edge_attr[keep]
    target = (y_node[src] == y_node[dst]).astype(np.int64)
    return src, dst, edge_attr, target


def sample_edge_indices(
    target: np.ndarray,
    seed: int,
    max_edges: int,
    mode: str = "random",
    labels_i: np.ndarray | None = None,
    labels_j: np.ndarray | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    n = int(target.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    all_idx = np.arange(n, dtype=np.int64)
    if mode == "random":
        if n <= int(max_edges):
            return all_idx
        return np.sort(rng.choice(all_idx, size=int(max_edges), replace=False))
    if mode == "target_balanced":
        pos = all_idx[target == 1]
        neg = all_idx[target == 0]
        m = min(len(pos), len(neg), max(1, int(max_edges) // 2))
        if m == 0:
            return sample_edge_indices(target, seed, max_edges, mode="random")
        out = np.concatenate([rng.choice(pos, size=m, replace=False), rng.choice(neg, size=m, replace=False)])
        return np.sort(out)
    if mode == "label_pair_balanced":
        if labels_i is None or labels_j is None:
            return sample_edge_indices(target, seed, max_edges, mode="target_balanced")
        pair_keys = []
        for a, b in zip(labels_i, labels_j):
            aa = int(a)
            bb = int(b)
            pair_keys.append((aa, bb) if aa <= bb else (bb, aa))
        groups: dict[tuple[int, int], list[int]] = {}
        for idx, key in enumerate(pair_keys):
            groups.setdefault(key, []).append(idx)
        if not groups:
            return sample_edge_indices(target, seed, max_edges, mode="target_balanced")
        cap = max(1, int(max_edges) // max(len(groups), 1))
        chosen = []
        for vals in groups.values():
            vals_arr = np.asarray(vals, dtype=np.int64)
            m = min(len(vals_arr), cap)
            chosen.extend(rng.choice(vals_arr, size=m, replace=False).tolist())
        chosen = np.asarray(chosen, dtype=np.int64)
        if chosen.shape[0] > int(max_edges):
            chosen = rng.choice(chosen, size=int(max_edges), replace=False)
        return np.sort(chosen)
    raise ValueError(f"Unknown sample mode: {mode}")


def run_edge_probe_arrays(
    *,
    graph: SparseGraph,
    src: np.ndarray,
    dst: np.ndarray,
    edge_attr: np.ndarray,
    target: np.ndarray,
    seed: int,
    feature_set_prefix: str = "",
    feature_sets_mode: str | None = None,
    test_size: float = 0.3,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed) + 17)
    endpoint = endpoint_features(graph.x, src, dst)
    edge_attr_perm = edge_attr[rng.permutation(edge_attr.shape[0])] if edge_attr.shape[0] else edge_attr
    rel = relation_bits(edge_attr, graph.feature_names)
    rel_perm = rel[rng.permutation(rel.shape[0])] if rel.shape[0] else rel
    feature_sets: dict[str, np.ndarray] = {
        "endpoint": endpoint,
        "endpoint_edge_attr": np.concatenate([endpoint, edge_attr], axis=1),
        "endpoint_edge_attr_perm": np.concatenate([endpoint, edge_attr_perm], axis=1),
    }
    mode = (feature_sets_mode or os.environ.get("URBAN_FACTOR_PROBE_FEATURES", "core")).strip().lower()
    if mode in {"all", "full"}:
        feature_sets["edge_attr"] = edge_attr
    if mode in {"all", "full"} and rel.shape[1] > 0:
        feature_sets["relation_bits"] = rel
        feature_sets["endpoint_relation_bits"] = np.concatenate([endpoint, rel], axis=1)
        feature_sets["endpoint_relation_bits_perm"] = np.concatenate([endpoint, rel_perm], axis=1)
    rows = []
    prefix = f"{feature_set_prefix}:" if feature_set_prefix else ""
    for name, X in feature_sets.items():
        metrics = fit_eval(X, target, seed=seed, test_size=test_size)
        rows.append(
            {
                "feature_set": f"{prefix}{name}",
                "seed": int(seed),
                "n_edges_known_undirected": int(target.shape[0]),
                "positive_same_label_rate": float(np.mean(target)) if target.shape[0] else np.nan,
                "n_features": int(X.shape[1]),
                **metrics,
            }
        )
    return rows


def add_probe_deltas(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    key = group_cols + ["seed"]
    endpoint = out[out["feature_set"] == "endpoint"][key + ["auc"]].rename(columns={"auc": "endpoint_auc"})
    perm = out[out["feature_set"] == "endpoint_edge_attr_perm"][key + ["auc"]].rename(
        columns={"auc": "endpoint_edge_attr_perm_auc"}
    )
    out = out.merge(endpoint, on=key, how="left")
    out = out.merge(perm, on=key, how="left")
    out["auc_gain_vs_endpoint"] = out["auc"] - out["endpoint_auc"]
    out["auc_drop_vs_perm"] = out["auc"] - out["endpoint_edge_attr_perm_auc"]
    return out


def summarize_probe(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.groupby(group_cols + ["feature_set"], as_index=False)
        .agg(
            runs=("seed", "count"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            ap_mean=("ap", "mean"),
            acc_mean=("acc", "mean"),
            auc_gain_vs_endpoint_mean=("auc_gain_vs_endpoint", "mean"),
            auc_gain_vs_endpoint_std=("auc_gain_vs_endpoint", "std"),
            auc_drop_vs_perm_mean=("auc_drop_vs_perm", "mean"),
            n_edges_known_undirected_mean=("n_edges_known_undirected", "mean"),
            positive_same_label_rate_mean=("positive_same_label_rate", "mean"),
        )
        .sort_values(group_cols + ["feature_set"])
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_readme(path: Path, title: str, lines: Iterable[str]) -> None:
    path.write_text("# " + title + "\n\n" + "\n".join(lines).strip() + "\n", encoding="utf-8")


def run_command(cmd: list[str], cwd: Path, dry_run: bool = False, env: dict[str, str] | None = None) -> dict[str, Any]:
    started = time.time()
    if dry_run:
        return {
            "cmd": " ".join(cmd),
            "rc": 0,
            "elapsed_sec": 0.0,
            "stdout": "",
            "stderr": "",
            "dry_run": True,
        }
    run_env = os.environ.copy()
    run_env["CUDA_VISIBLE_DEVICES"] = ""
    if env:
        run_env.update(env)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=run_env,
    )
    return {
        "cmd": " ".join(cmd),
        "rc": int(proc.returncode),
        "elapsed_sec": float(time.time() - started),
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "dry_run": False,
    }


def bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def entropy_norm(counts: np.ndarray) -> float:
    arr = np.asarray(counts, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size <= 1:
        return 0.0
    p = arr / arr.sum()
    return float(-(p * np.log(p)).sum() / math.log(len(p)))
