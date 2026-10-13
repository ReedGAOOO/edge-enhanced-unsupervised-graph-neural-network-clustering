#!/usr/bin/env python3
"""Second-order diagnostics for why Urban graph effects differ by city.

This extends the first driver audit with factors that are easy to miss:

- label-task mismatch: meaningful urban relations may be heterophilous rather
  than same-label;
- class/degree baselines: raw same-label rates can be inflated by dominant
  labels or degree-skewed known nodes;
- spatial and node-feature confounding: distance and endpoint similarity may
  explain the apparent edge signal;
- edge-attribute redundancy: edge_attr may be predictable from endpoint node
  features, so a probe gain is not necessarily independent edge semantics;
- relation granularity: relation mix entropy and relation-specific lift.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RELATION_FEATURES = {
    "street": "is_street_backed",
    "junction": "is_junction_only",
    "geometry": "is_geom_fallback",
    "building": "is_shared_building",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-path",
        type=Path,
        default=Path("/mnt/e/wsl-project-data/edge-enhanced-unsupervised-graph-neural-network-clustering/data"),
    )
    parser.add_argument("--pattern", type=str, default="urban_*_plot_v3bsjg")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/data_construction_series/urban_all_city_exploration"),
    )
    parser.add_argument("--max-r2-edges", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def city_display(city: str) -> str:
    return {
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
    }.get(city, city.title())


def unknown_label(meta: dict[str, Any]) -> int | None:
    mapping = meta.get("label_mapping", {})
    if "unknown" not in mapping:
        return None
    return int(mapping["unknown"])


def label_names(meta: dict[str, Any]) -> dict[int, str]:
    return {int(v): str(k) for k, v in meta.get("label_mapping", {}).items()}


def entropy_norm(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[np.isfinite(counts) & (counts > 0)]
    if counts.size <= 1:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / math.log(len(p)))


def undirected_unique(
    edge_index: np.ndarray, edge_attr: np.ndarray, num_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    src = edge_index[0].astype(np.int64, copy=False)
    dst = edge_index[1].astype(np.int64, copy=False)
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    key = lo * int(num_nodes) + hi
    _, keep = np.unique(key, return_index=True)
    return np.stack([lo[keep], hi[keep]], axis=0), edge_attr[keep]


def endpoint_features(x: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    xi = x[src].astype(np.float32, copy=False)
    xj = x[dst].astype(np.float32, copy=False)
    return np.concatenate([0.5 * (xi + xj), np.abs(xi - xj), xi * xj], axis=1).astype(np.float32)


def safe_auc(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target)
    score = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(score)
    target = target[mask]
    score = score[mask]
    if target.size < 20 or np.unique(target).size < 2:
        return float("nan")
    return float(roc_auc_score(target, score))


def normalized_label_mi(labels_i: np.ndarray, labels_j: np.ndarray, known_labels: np.ndarray) -> float:
    classes = np.asarray(sorted(np.unique(known_labels)), dtype=np.int64)
    if classes.size <= 1 or labels_i.size == 0:
        return float("nan")
    idx = {int(c): k for k, c in enumerate(classes)}
    mat = np.zeros((classes.size, classes.size), dtype=np.float64)
    for a, b in zip(labels_i, labels_j):
        ia, ib = idx[int(a)], idx[int(b)]
        mat[ia, ib] += 1.0
        mat[ib, ia] += 1.0
    total = mat.sum()
    if total <= 0:
        return float("nan")
    pxy = mat / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    denom = px @ py
    mask = (pxy > 0) & (denom > 0)
    mi = float((pxy[mask] * np.log(pxy[mask] / denom[mask])).sum())
    h = float(-(px[px > 0] * np.log(px[px > 0])).sum())
    return mi / h if h > 0 else float("nan")


def top_cross_pair_share(labels_i: np.ndarray, labels_j: np.ndarray, names: dict[int, str]) -> tuple[float, str]:
    diff = labels_i != labels_j
    if not np.any(diff):
        return float("nan"), ""
    counts: dict[tuple[int, int], int] = {}
    for a, b in zip(labels_i[diff], labels_j[diff]):
        key = (int(a), int(b)) if int(a) <= int(b) else (int(b), int(a))
        counts[key] = counts.get(key, 0) + 1
    key, val = max(counts.items(), key=lambda kv: kv[1])
    pair_name = f"{names.get(key[0], str(key[0]))}-{names.get(key[1], str(key[1]))}"
    return float(val / max(int(np.sum(diff)), 1)), pair_name


def edge_attr_r2(
    x: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    feature_names: list[str],
    max_edges: int,
    seed: int,
) -> tuple[float, float]:
    n_edges = edge_index.shape[1]
    if n_edges < 50:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = np.arange(n_edges)
    if n_edges > max_edges:
        idx = rng.choice(idx, size=max_edges, replace=False)
    src = edge_index[0, idx]
    dst = edge_index[1, idx]
    X = endpoint_features(x, src, dst)
    Y = edge_attr[idx].astype(np.float32, copy=False)
    finite_cols = np.isfinite(Y).all(axis=0) & (np.nanstd(Y, axis=0) > 1e-8)
    if finite_cols.sum() == 0:
        return float("nan"), float("nan")

    rel_names = set(RELATION_FEATURES.values())
    nonrelation_cols = np.asarray([name not in rel_names for name in feature_names], dtype=bool)

    train_idx, test_idx = train_test_split(np.arange(X.shape[0]), test_size=0.3, random_state=seed)

    def fit_score(cols: np.ndarray) -> float:
        cols = cols & finite_cols
        if cols.sum() == 0:
            return float("nan")
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(X[train_idx], Y[train_idx][:, cols])
        pred = model.predict(X[test_idx])
        score = r2_score(Y[test_idx][:, cols], pred, multioutput="variance_weighted")
        return float(score)

    return fit_score(finite_cols), fit_score(nonrelation_cols)


def relation_masks(edge_attr: np.ndarray, feature_names: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for rel, name in RELATION_FEATURES.items():
        if name in feature_names:
            out[rel] = np.asarray(edge_attr[:, feature_names.index(name)] > 0.5)
        else:
            out[rel] = np.zeros(edge_attr.shape[0], dtype=bool)
    return out


def analyze_one(dataset_dir: Path, max_r2_edges: int, seed: int) -> dict[str, Any]:
    dataset = dataset_dir.name
    base = dataset_dir / dataset
    meta = load_json(base.with_name(f"{dataset}_meta.json"))
    feature_names = list(meta.get("edge_feature_names", []))

    x = np.load(base.with_name(f"{dataset}_feat.npy"), mmap_mode="r")
    labels = np.asarray(np.load(base.with_name(f"{dataset}_label.npy"), mmap_mode="r"))
    edge_index = np.load(base.with_name(f"{dataset}_edge_index.npy"), mmap_mode="r")
    edge_attr = np.load(base.with_name(f"{dataset}_edge_attr.npy"), mmap_mode="r")
    edge_index_u, edge_attr_u = undirected_unique(np.asarray(edge_index), np.asarray(edge_attr), num_nodes=x.shape[0])

    city = str(meta.get("city", dataset.replace("urban_", "").replace("_plot_v3bsjg", "")))
    unknown = unknown_label(meta)
    known_node = labels >= 0 if unknown is None else ((labels >= 0) & (labels != unknown))
    known_labels = labels[known_node]
    known_classes, known_counts = np.unique(known_labels, return_counts=True)
    node_baseline = float(np.square(known_counts / known_counts.sum()).sum()) if known_counts.sum() else float("nan")

    src, dst = edge_index_u
    both_known = known_node[src] & known_node[dst]
    li = labels[src[both_known]]
    lj = labels[dst[both_known]]
    same = li == lj
    same_ratio = float(np.mean(same)) if same.size else float("nan")

    endpoint_labels = np.concatenate([li, lj]) if li.size else np.asarray([], dtype=np.int64)
    endpoint_classes, endpoint_counts = np.unique(endpoint_labels, return_counts=True) if endpoint_labels.size else ([], [])
    edge_endpoint_baseline = (
        float(np.square(endpoint_counts / endpoint_counts.sum()).sum()) if len(endpoint_counts) else float("nan")
    )

    rel_masks = relation_masks(edge_attr_u, feature_names)
    rel_fracs = {rel: float(mask.mean()) for rel, mask in rel_masks.items()}
    rel_entropy = entropy_norm(np.asarray(list(rel_fracs.values()), dtype=np.float64))
    rel_dominance = float(max(rel_fracs.values())) if rel_fracs else float("nan")

    row: dict[str, Any] = {
        "city": city,
        "city_label": city_display(city),
        "dataset": dataset,
        "n_nodes": int(x.shape[0]),
        "n_edges_undirected": int(edge_index_u.shape[1]),
        "known_label_ratio": float(known_node.mean()),
        "n_known_edges": int(both_known.sum()),
        "node_label_random_same_baseline": node_baseline,
        "edge_endpoint_label_baseline": edge_endpoint_baseline,
        "observed_edge_same_label_ratio": same_ratio,
        "same_label_lift_vs_node_baseline": same_ratio - node_baseline,
        "same_label_lift_vs_edge_endpoint_baseline": same_ratio - edge_endpoint_baseline,
        "edge_label_mi_norm": normalized_label_mi(li, lj, known_labels),
        "heterophily_ratio": float(1.0 - same_ratio) if np.isfinite(same_ratio) else float("nan"),
        "relation_mix_entropy_norm": rel_entropy,
        "relation_dominance": rel_dominance,
        **{f"{rel}_edge_frac": frac for rel, frac in rel_fracs.items()},
    }

    top_share, top_pair = top_cross_pair_share(li, lj, label_names(meta))
    row["top_cross_label_pair_share"] = top_share
    row["top_cross_label_pair"] = top_pair

    rel_lifts = []
    for rel, mask in rel_masks.items():
        rel_known = mask & both_known
        if np.any(rel_known):
            r_same = float(np.mean(labels[src[rel_known]] == labels[dst[rel_known]]))
            lift = r_same - node_baseline
            row[f"{rel}_same_label_ratio"] = r_same
            row[f"{rel}_same_label_lift"] = lift
            rel_lifts.append(lift)
        else:
            row[f"{rel}_same_label_ratio"] = float("nan")
            row[f"{rel}_same_label_lift"] = float("nan")
    row["relation_lift_range"] = float(np.nanmax(rel_lifts) - np.nanmin(rel_lifts)) if rel_lifts else float("nan")

    degree = np.bincount(np.concatenate([src, dst]), minlength=x.shape[0]).astype(np.float64)
    known_degree = np.log1p(degree[known_node])
    if known_degree.size and np.unique(known_labels).size > 1:
        grand = float(np.mean(known_degree))
        ss_total = float(np.sum((known_degree - grand) ** 2))
        ss_between = 0.0
        for cls in np.unique(known_labels):
            vals = known_degree[known_labels == cls]
            ss_between += float(vals.size * (np.mean(vals) - grand) ** 2)
        row["degree_label_eta2"] = ss_between / ss_total if ss_total > 0 else float("nan")
    else:
        row["degree_label_eta2"] = float("nan")

    if "centroid_dist_m" in feature_names:
        dist = edge_attr_u[both_known, feature_names.index("centroid_dist_m")]
        row["distance_only_same_label_auc"] = safe_auc(same.astype(np.int64), -dist)
        row["known_edge_distance_p50"] = float(np.nanmedian(dist)) if dist.size else float("nan")
        row["known_edge_distance_p90"] = float(np.nanquantile(dist, 0.90)) if dist.size else float("nan")
    else:
        row["distance_only_same_label_auc"] = float("nan")
        row["known_edge_distance_p50"] = float("nan")
        row["known_edge_distance_p90"] = float("nan")

    if "node_feat_cosine" in feature_names:
        score = edge_attr_u[both_known, feature_names.index("node_feat_cosine")]
        row["node_similarity_same_label_auc"] = safe_auc(same.astype(np.int64), score)
    else:
        row["node_similarity_same_label_auc"] = float("nan")
    if "node_feat_l2" in feature_names:
        score = edge_attr_u[both_known, feature_names.index("node_feat_l2")]
        row["node_l2_same_label_auc"] = safe_auc(same.astype(np.int64), -score)
    else:
        row["node_l2_same_label_auc"] = float("nan")

    r2_all, r2_nonrelation = edge_attr_r2(
        np.asarray(x), edge_index_u, edge_attr_u, feature_names, max_edges=max_r2_edges, seed=seed
    )
    row["edge_attr_endpoint_r2_all"] = r2_all
    row["edge_attr_endpoint_r2_nonrelation"] = r2_nonrelation
    return row


def write_report(df: pd.DataFrame, out_path: Path) -> None:
    top_lift = df.sort_values("same_label_lift_vs_node_baseline", ascending=False).head(5)
    weak_lift = df.sort_values("same_label_lift_vs_node_baseline", ascending=True).head(5)
    high_mi = df.sort_values("edge_label_mi_norm", ascending=False).head(5)
    redundancy = df.sort_values("edge_attr_endpoint_r2_nonrelation", ascending=False).head(5)

    lines = [
        "# Urban Additional Difference Factors",
        "",
        "This audit looks beyond data source/label quality, projection-tail risk, and coarse urban-form proxies.",
        "",
        "## Additional Factors Found",
        "",
        "1. **Task-label mismatch**: same-label is only one kind of edge semantics. A city can have meaningful heterophilous label-pair structure even when same-label lift is weak.",
        "2. **Class and degree baselines**: raw same-label ratios must be compared with node-label and edge-endpoint baselines; otherwise dominant classes and degree-skewed labels inflate homophily.",
        "3. **Spatial confounding**: if distance alone predicts same-label edges, edge_attr may be learning spatial autocorrelation rather than relation semantics.",
        "4. **Endpoint-feature redundancy**: high edge_attr-from-endpoint R2 means the edge attributes are partly re-encoding node morphology/similarity.",
        "5. **Relation granularity**: relation mix entropy and relation-specific lift decide whether relation bits contain useful variation.",
        "6. **Topology-degree confounding**: labels can differ in graph degree, making edge probes sensitive to network position rather than semantic relation.",
        "",
        "## Highest Same-Label Lift vs Node Baseline",
        "",
        "| city | observed same | node baseline | lift | edge MI |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in top_lift.itertuples(index=False):
        lines.append(
            f"| {row.city_label} | {row.observed_edge_same_label_ratio:.3f} | {row.node_label_random_same_baseline:.3f} | {row.same_label_lift_vs_node_baseline:.3f} | {row.edge_label_mi_norm:.3f} |"
        )
    lines.extend(["", "## Weakest Same-Label Lift vs Node Baseline", "", "| city | observed same | node baseline | lift | top cross-label pair |", "|---|---:|---:|---:|---|"])
    for row in weak_lift.itertuples(index=False):
        lines.append(
            f"| {row.city_label} | {row.observed_edge_same_label_ratio:.3f} | {row.node_label_random_same_baseline:.3f} | {row.same_label_lift_vs_node_baseline:.3f} | {row.top_cross_label_pair} |"
        )
    lines.extend(["", "## Highest Label-Pair Mutual Information", "", "| city | edge label MI | heterophily ratio | top cross-label pair |", "|---|---:|---:|---|"])
    for row in high_mi.itertuples(index=False):
        lines.append(
            f"| {row.city_label} | {row.edge_label_mi_norm:.3f} | {row.heterophily_ratio:.3f} | {row.top_cross_label_pair} |"
        )
    lines.extend(["", "## Highest Endpoint-Feature Redundancy", "", "| city | edge_attr R2 nonrelation | distance AUC | node-sim AUC |", "|---|---:|---:|---:|"])
    for row in redundancy.itertuples(index=False):
        lines.append(
            f"| {row.city_label} | {row.edge_attr_endpoint_r2_nonrelation:.3f} | {row.distance_only_same_label_auc:.3f} | {row.node_similarity_same_label_auc:.3f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset_dir in sorted(args.root_path.glob(args.pattern)):
        if not dataset_dir.is_dir():
            continue
        print(f"[additional] {dataset_dir.name}", flush=True)
        rows.append(analyze_one(dataset_dir, max_r2_edges=args.max_r2_edges, seed=args.seed))
    df = pd.DataFrame(rows).sort_values("city").reset_index(drop=True)
    out_csv = args.out_dir / "additional_city_difference_factors.csv"
    df.to_csv(out_csv, index=False)
    report_path = args.out_dir / "urban_additional_difference_factors.md"
    write_report(df, report_path)
    print(f"Wrote {out_csv}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
