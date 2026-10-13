#!/usr/bin/env python3
"""
Independent CPU edge-semantics probe for Urban graphs.

This does not use the main GNN architecture. It tests whether edge attributes
carry task-aligned signal by predicting whether two connected plots share the
same known label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RELATION_FEATURES = [
    "is_street_backed",
    "is_junction_only",
    "is_geom_fallback",
    "is_shared_building",
]


def _parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


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


def _undirected_unique(edge_index: np.ndarray, edge_attr: np.ndarray, edge_weight: np.ndarray | None, num_nodes: int):
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


def _endpoint_features(x: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    xi = x[src].astype(np.float32, copy=False)
    xj = x[dst].astype(np.float32, copy=False)
    return np.concatenate(
        [
            0.5 * (xi + xj),
            np.abs(xi - xj),
            xi * xj,
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def _relation_bits(edge_attr: np.ndarray, feature_names: list[str]) -> np.ndarray:
    idx = [feature_names.index(name) for name in RELATION_FEATURES if name in feature_names]
    if not idx:
        return np.zeros((edge_attr.shape[0], 0), dtype=np.float32)
    return edge_attr[:, idx].astype(np.float32, copy=False)


def _fit_eval(X: np.ndarray, y: np.ndarray, seed: int, test_size: float) -> dict[str, float]:
    if X.shape[0] < 20 or len(np.unique(y)) < 2:
        return {
            "auc": float("nan"),
            "ap": float("nan"),
            "acc": float("nan"),
            "log_loss": float("nan"),
        }
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(seed),
        stratify=y,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        ),
    )
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(np.int64)
    return {
        "auc": float(roc_auc_score(y_test, prob)),
        "ap": float(average_precision_score(y_test, prob)),
        "acc": float(accuracy_score(y_test, pred)),
        "log_loss": float(log_loss(y_test, prob, labels=[0, 1])),
    }


def _probe_one(root: Path, dataset: str, seed: int, max_edges: int, test_size: float) -> list[dict[str, Any]]:
    base = root / dataset
    meta = _load_meta(base, dataset)
    x = np.load(base / f"{dataset}_feat.npy", allow_pickle=False).astype(np.float32, copy=False)
    y_node = np.load(base / f"{dataset}_label.npy", allow_pickle=False)
    edge_index = np.load(base / f"{dataset}_edge_index.npy", allow_pickle=False)
    edge_attr = np.load(base / f"{dataset}_edge_attr.npy", allow_pickle=False).astype(np.float32, copy=False)
    ew_path = base / f"{dataset}_edge_weight.npy"
    edge_weight = np.load(ew_path, allow_pickle=False).astype(np.float32, copy=False) if ew_path.exists() else None

    edge_index, edge_attr, edge_weight = _undirected_unique(edge_index, edge_attr, edge_weight, num_nodes=x.shape[0])
    src = edge_index[0]
    dst = edge_index[1]
    unknown = _unknown_label(meta)
    known = y_node >= 0 if unknown is None else ((y_node >= 0) & (y_node != int(unknown)))
    keep = known[src] & known[dst]
    src = src[keep]
    dst = dst[keep]
    edge_attr = edge_attr[keep]
    if edge_weight is not None:
        edge_weight = edge_weight[keep]

    target = (y_node[src] == y_node[dst]).astype(np.int64)
    if target.shape[0] > int(max_edges):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(target.shape[0], size=int(max_edges), replace=False)
        idx = np.sort(idx)
        src = src[idx]
        dst = dst[idx]
        edge_attr = edge_attr[idx]
        target = target[idx]

    feature_names = list(meta.get("edge_feature_names", []))
    if len(feature_names) != edge_attr.shape[1]:
        feature_names = [str(i) for i in range(edge_attr.shape[1])]

    rng = np.random.default_rng(int(seed) + 17)
    edge_attr_perm = edge_attr[rng.permutation(edge_attr.shape[0])] if edge_attr.shape[0] else edge_attr

    endpoint = _endpoint_features(x, src, dst)
    rel_bits = _relation_bits(edge_attr, feature_names)
    rel_bits_perm = rel_bits[rng.permutation(rel_bits.shape[0])] if rel_bits.shape[0] else rel_bits

    feature_sets: dict[str, np.ndarray] = {
        "endpoint": endpoint,
        "edge_attr": edge_attr,
        "relation_bits": rel_bits,
        "endpoint_edge_attr": np.concatenate([endpoint, edge_attr], axis=1),
        "endpoint_edge_attr_perm": np.concatenate([endpoint, edge_attr_perm], axis=1),
    }
    if rel_bits.shape[1] > 0:
        feature_sets["endpoint_relation_bits"] = np.concatenate([endpoint, rel_bits], axis=1)
        feature_sets["endpoint_relation_bits_perm"] = np.concatenate([endpoint, rel_bits_perm], axis=1)

    rows: list[dict[str, Any]] = []
    pos_rate = float(target.mean()) if target.shape[0] else float("nan")
    for feature_set, X in feature_sets.items():
        if X.shape[1] == 0:
            metrics = {"auc": float("nan"), "ap": float("nan"), "acc": float("nan"), "log_loss": float("nan")}
        else:
            metrics = _fit_eval(X, target, seed=seed, test_size=test_size)
        rows.append(
            {
                "dataset": dataset,
                "city": str(meta.get("city", dataset)),
                "seed": int(seed),
                "feature_set": feature_set,
                "n_edges_known_undirected": int(target.shape[0]),
                "positive_same_label_rate": pos_rate,
                "n_features": int(X.shape[1]),
                **metrics,
            }
        )
    return rows


def _add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    key = ["dataset", "seed"]
    endpoint = out[out["feature_set"] == "endpoint"][key + ["auc"]].rename(columns={"auc": "endpoint_auc"})
    concat_perm = out[out["feature_set"] == "endpoint_edge_attr_perm"][key + ["auc"]].rename(
        columns={"auc": "endpoint_edge_attr_perm_auc"}
    )
    out = out.merge(endpoint, on=key, how="left")
    out = out.merge(concat_perm, on=key, how="left")
    out["auc_gain_vs_endpoint"] = out["auc"] - out["endpoint_auc"]
    out["auc_drop_vs_perm"] = out["auc"] - out["endpoint_edge_attr_perm_auc"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent Urban edge-semantics probe.")
    parser.add_argument("--root_path", type=str, default="data")
    parser.add_argument("--datasets", type=str, default="urban_paris_plot_v3bsjg,urban_boston_plot_v3bsjg,urban_washingtondc_plot_v3bsjg")
    parser.add_argument("--pattern", type=str, default="urban_*_plot_v3bsjg")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--max_edges", type=int, default=50000)
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--out_dir", type=str, default="results/edge_probe/urban_v3bsjg_same_label")
    args = parser.parse_args()

    root = Path(args.root_path)
    if str(args.datasets).strip().lower() == "all":
        datasets = _discover(root, args.pattern)
    else:
        datasets = _parse_csv(args.datasets)
    seeds = [int(x) for x in _parse_csv(args.seeds)]
    if not datasets or not seeds:
        raise ValueError("datasets and seeds must be non-empty.")

    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            print(f"[probe] {dataset} seed={seed}", flush=True)
            rows.extend(_probe_one(root, dataset, seed=seed, max_edges=args.max_edges, test_size=args.test_size))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _add_deltas(pd.DataFrame(rows))
    df.to_csv(out_dir / "edge_probe_runs.csv", index=False)
    summary = (
        df.groupby(["dataset", "feature_set"], as_index=False)
        .agg(
            runs=("seed", "count"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            ap_mean=("ap", "mean"),
            acc_mean=("acc", "mean"),
            auc_gain_vs_endpoint_mean=("auc_gain_vs_endpoint", "mean"),
            auc_drop_vs_perm_mean=("auc_drop_vs_perm", "mean"),
            n_edges_known_undirected_mean=("n_edges_known_undirected", "mean"),
            positive_same_label_rate_mean=("positive_same_label_rate", "mean"),
        )
        .sort_values(["dataset", "auc_mean"], ascending=[True, False])
    )
    summary.to_csv(out_dir / "edge_probe_summary.csv", index=False)

    readme = [
        "# Urban Edge Probe",
        "",
        "Independent CPU probe: predict whether connected plots with known labels share the same label.",
        "",
        "Feature sets:",
        "- `endpoint`: node-feature pair features only.",
        "- `edge_attr`: edge attributes only.",
        "- `relation_bits`: four relation bits only.",
        "- `endpoint_edge_attr`: endpoint features plus edge attributes.",
        "- `*_perm`: edge attributes or relation bits permuted across edges.",
        "",
        "Evidence that edge information is meaningful:",
        "- `edge_attr` AUC is materially above 0.5.",
        "- `endpoint_edge_attr` improves over `endpoint`.",
        "- `endpoint_edge_attr` beats `endpoint_edge_attr_perm`.",
        "",
        "This probe is not a replacement for clustering experiments; it tests whether the edge channel contains task-aligned signal at all.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print(f"[ok] runs={out_dir / 'edge_probe_runs.csv'}")
    print(f"[ok] summary={out_dir / 'edge_probe_summary.csv'}")


if __name__ == "__main__":
    main()
