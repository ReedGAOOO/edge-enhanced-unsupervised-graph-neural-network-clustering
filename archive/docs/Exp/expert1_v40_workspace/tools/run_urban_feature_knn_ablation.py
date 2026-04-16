#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DEFAULT_CITIES = [
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


def parse_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def dataset_name(city: str, feature_knn_k: int) -> str:
    return f"urban_{city}_plot_v2fk{int(feature_knn_k)}"


def run_cmd(cmd: List[str], cwd: Path) -> None:
    subprocess.check_call(cmd, cwd=str(cwd))


def prepare_graphs(repo: Path, cities: List[str], k_values: List[int], args) -> List[str]:
    datasets: List[str] = []
    for city in cities:
        for k in k_values:
            name = dataset_name(city, k)
            datasets.append(name)
            cmd = [
                sys.executable,
                "tools/prepare_urban_plot_graph_v2.py",
                "--city",
                city,
                "--urban_root",
                args.urban_root,
                "--out_root",
                args.out_root,
                "--dataset_name",
                name,
                "--feature_knn_k",
                str(k),
                "--feature_knn_temp",
                str(args.feature_knn_temp),
                "--feature_edge_blend",
                str(args.feature_edge_blend),
                "--topk_per_node",
                str(args.topk_per_node),
                "--max_plots_per_street",
                str(args.max_plots_per_street),
                "--max_plots_per_building",
                str(args.max_plots_per_building),
                "--max_plots_per_junction",
                str(args.max_plots_per_junction),
                "--feature_profile",
                args.feature_profile,
                "--force",
            ]
            if args.include_sparse_city_specific:
                cmd.append("--include_sparse_city_specific")
            if args.include_landuse_features:
                cmd.append("--include_landuse_features")
            run_cmd(cmd, cwd=repo)
    return datasets


def run_benchmark(repo: Path, datasets: List[str], args) -> Path:
    tag = args.tag
    cmd = [
        sys.executable,
        "tools/run_urban_branch_compare.py",
        "--datasets",
        ",".join(datasets),
        "--conditions",
        args.conditions,
        "--seeds",
        args.seeds,
        "--gpu",
        str(args.gpu),
        "--epochs",
        str(args.epochs),
        "--eval_freq",
        str(args.eval_freq),
        "--train_log_interval",
        str(args.train_log_interval),
        "--hid_dim",
        str(args.hid_dim),
        "--n_cluster_trials",
        str(args.n_cluster_trials),
        "--max_nums",
        str(args.max_nums),
        "--knn",
        str(args.knn),
        "--tag",
        tag,
        "--root_path",
        args.out_root,
    ]
    if args.amp_bf16:
        cmd.append("--amp_bf16")
    if args.known_only_eval:
        cmd.append("--known_only_eval")
    if args.force_rerun:
        cmd.append("--force_rerun")
    run_cmd(cmd, cwd=repo)
    return repo / "results" / tag


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def graph_diagnostics(repo: Path, datasets: List[str], out_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    data_root = repo / "data"
    for name in datasets:
        city = name.replace("urban_", "").replace("_plot", "")
        city = city.rsplit("_v2fk", 1)[0]
        feature_knn_k = int(name.rsplit("_v2fk", 1)[1])

        base = data_root / name / name
        meta = json.loads(base.with_name(f"{name}_meta.json").read_text(encoding="utf-8"))
        y = np.load(base.with_name(f"{name}_label.npy"))
        edge_index = np.load(base.with_name(f"{name}_edge_index.npy"))
        edge_weight = np.load(base.with_name(f"{name}_edge_weight.npy"))
        edge_attr = np.load(base.with_name(f"{name}_edge_attr.npy"))

        src = edge_index[0]
        dst = edge_index[1]
        keep = src < dst
        src = src[keep]
        dst = dst[keep]
        w = edge_weight[keep]
        attr = edge_attr[keep]

        feature_names = meta.get("edge_feature_names", [])
        idx = {k: i for i, k in enumerate(feature_names)}
        feat_mask = attr[:, idx["feat_knn_sim"]] > 0 if "feat_knn_sim" in idx else np.zeros(src.shape[0], dtype=bool)
        street_mask = attr[:, idx["shared_street_cnt"]] > 0 if "shared_street_cnt" in idx else np.zeros(src.shape[0], dtype=bool)
        junction_mask = attr[:, idx["shared_junction_cnt"]] > 0 if "shared_junction_cnt" in idx else np.zeros(src.shape[0], dtype=bool)
        building_mask = attr[:, idx["shared_building_cnt"]] > 0 if "shared_building_cnt" in idx else np.zeros(src.shape[0], dtype=bool)

        unknown = (meta.get("label_mapping") or {}).get("unknown", None)
        known = np.ones_like(y, dtype=bool) if unknown is None else (y != int(unknown))
        known_edge = known[src] & known[dst]

        same = np.zeros(src.shape[0], dtype=bool)
        same[known_edge] = y[src[known_edge]] == y[dst[known_edge]]

        rows.append(
            {
                "dataset": name,
                "city": city,
                "feature_knn_k": feature_knn_k,
                "nodes": int(meta["n_nodes"]),
                "edges_undirected": int(src.shape[0]),
                "avg_deg": float(2.0 * src.shape[0] / max(int(meta["n_nodes"]), 1)),
                "known_ratio": float(known.mean()),
                "known_edge_ratio": float(known_edge.mean()),
                "known_homophily_unweighted": float(same[known_edge].mean()) if known_edge.any() else np.nan,
                "known_homophily_weighted": float(w[known_edge][same[known_edge]].sum() / w[known_edge].sum())
                if known_edge.any() and w[known_edge].sum() > 0
                else np.nan,
                "feature_knn_edge_share": float(feat_mask.mean()),
                "street_edge_share": float(street_mask.mean()),
                "junction_edge_share": float(junction_mask.mean()),
                "building_edge_share": float(building_mask.mean()),
                "feature_knn_known_homophily": float(same[feat_mask & known_edge].mean())
                if (feat_mask & known_edge).any()
                else np.nan,
            }
        )

    df = pd.DataFrame(rows).sort_values(["city", "feature_knn_k"])
    df.to_csv(out_dir / "graph_diagnostics_by_city_k.csv", index=False)
    return df


def aggregate_results(out_dir: Path) -> None:
    runs = pd.read_csv(out_dir / "runs.csv")
    runs["city"] = runs["dataset"].map(lambda x: str(x).replace("urban_", "").replace("_plot", "").rsplit("_v2fk", 1)[0])
    runs["feature_knn_k"] = runs["dataset"].map(lambda x: int(str(x).rsplit("_v2fk", 1)[1]))
    ok = runs[runs["status"].isin(["ok", "skip_exists"])].copy()

    summary_by_ck = (
        ok.groupby(["condition", "feature_knn_k"], as_index=False)
        .agg(
            datasets_ok=("dataset", "count"),
            mean_nmi=("nmi", "mean"),
            mean_ari=("ari", "mean"),
            mean_acc=("acc", "mean"),
            mean_seconds=("seconds", "mean"),
        )
        .sort_values(["condition", "feature_knn_k"])
    )
    summary_by_ck.to_csv(out_dir / "summary_by_condition_k.csv", index=False)

    best_k = (
        ok.sort_values(["condition", "city", "nmi"], ascending=[True, True, False])
        .groupby(["condition", "city"], as_index=False)
        .head(1)
        .loc[:, ["condition", "city", "feature_knn_k", "nmi", "ari", "acc", "seconds"]]
        .rename(columns={"feature_knn_k": "best_feature_knn_k"})
    )
    best_k.to_csv(out_dir / "best_k_by_city_condition.csv", index=False)

    best_k_counts = (
        best_k.groupby(["condition", "best_feature_knn_k"], as_index=False)
        .agg(city_count=("city", "count"))
        .sort_values(["condition", "best_feature_knn_k"])
    )
    best_k_counts.to_csv(out_dir / "best_k_counts_by_condition.csv", index=False)

    ref8 = ok[ok["feature_knn_k"] == 8][["city", "condition", "nmi", "ari", "acc", "seconds", "status"]].rename(
        columns={
            "nmi": "nmi_k8",
            "ari": "ari_k8",
            "acc": "acc_k8",
            "seconds": "seconds_k8",
            "status": "status_k8",
        }
    )
    paired8 = runs.merge(ref8, on=["city", "condition"], how="left")
    paired8["delta_nmi_vs_k8"] = paired8["nmi"] - paired8["nmi_k8"]
    paired8["delta_ari_vs_k8"] = paired8["ari"] - paired8["ari_k8"]
    paired8["delta_acc_vs_k8"] = paired8["acc"] - paired8["acc_k8"]
    paired8.to_csv(out_dir / "paired_vs_k8.csv", index=False)

    summary_vs_k8 = (
        paired8[paired8["status"].isin(["ok", "skip_exists"]) & paired8["status_k8"].isin(["ok", "skip_exists"])]
        .groupby(["condition", "feature_knn_k"], as_index=False)
        .agg(
            cities_compared=("city", "count"),
            mean_delta_nmi_vs_k8=("delta_nmi_vs_k8", "mean"),
            median_delta_nmi_vs_k8=("delta_nmi_vs_k8", "median"),
            win_count_nmi_vs_k8=("delta_nmi_vs_k8", lambda s: int((s > 0).sum())),
            loss_count_nmi_vs_k8=("delta_nmi_vs_k8", lambda s: int((s < 0).sum())),
            mean_delta_ari_vs_k8=("delta_ari_vs_k8", "mean"),
        )
        .sort_values(["condition", "feature_knn_k"])
    )
    summary_vs_k8.to_csv(out_dir / "summary_vs_k8_by_condition_k.csv", index=False)

    ref0 = ok[ok["feature_knn_k"] == 0][["city", "condition", "nmi", "ari", "acc", "seconds", "status"]].rename(
        columns={
            "nmi": "nmi_k0",
            "ari": "ari_k0",
            "acc": "acc_k0",
            "seconds": "seconds_k0",
            "status": "status_k0",
        }
    )
    paired0 = runs.merge(ref0, on=["city", "condition"], how="left")
    paired0["delta_nmi_vs_k0"] = paired0["nmi"] - paired0["nmi_k0"]
    paired0["delta_ari_vs_k0"] = paired0["ari"] - paired0["ari_k0"]
    paired0["delta_acc_vs_k0"] = paired0["acc"] - paired0["acc_k0"]
    paired0.to_csv(out_dir / "paired_vs_k0.csv", index=False)

    summary_vs_k0 = (
        paired0[paired0["status"].isin(["ok", "skip_exists"]) & paired0["status_k0"].isin(["ok", "skip_exists"])]
        .groupby(["condition", "feature_knn_k"], as_index=False)
        .agg(
            cities_compared=("city", "count"),
            mean_delta_nmi_vs_k0=("delta_nmi_vs_k0", "mean"),
            median_delta_nmi_vs_k0=("delta_nmi_vs_k0", "median"),
            win_count_nmi_vs_k0=("delta_nmi_vs_k0", lambda s: int((s > 0).sum())),
            loss_count_nmi_vs_k0=("delta_nmi_vs_k0", lambda s: int((s < 0).sum())),
            mean_delta_ari_vs_k0=("delta_ari_vs_k0", "mean"),
        )
        .sort_values(["condition", "feature_knn_k"])
    )
    summary_vs_k0.to_csv(out_dir / "summary_vs_k0_by_condition_k.csv", index=False)

    fail_summary = (
        runs.groupby(["condition", "feature_knn_k"], as_index=False)
        .agg(
            runs_total=("dataset", "count"),
            runs_ok=("status", lambda s: int(np.sum(s.isin(["ok", "skip_exists"])))),
            runs_fail=("status", lambda s: int(np.sum(~s.isin(["ok", "skip_exists"])))),
        )
        .sort_values(["condition", "feature_knn_k"])
    )
    fail_summary.to_csv(out_dir / "failure_summary_by_condition_k.csv", index=False)

    report = {
        "summary_by_condition_k_csv": str(out_dir / "summary_by_condition_k.csv"),
        "summary_vs_k8_csv": str(out_dir / "summary_vs_k8_by_condition_k.csv"),
        "summary_vs_k0_csv": str(out_dir / "summary_vs_k0_by_condition_k.csv"),
        "best_k_counts_csv": str(out_dir / "best_k_counts_by_condition.csv"),
        "failure_summary_csv": str(out_dir / "failure_summary_by_condition_k.csv"),
        "graph_diagnostics_csv": str(out_dir / "graph_diagnostics_by_city_k.csv"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Ablation on feature-KNN proportion in urban v2 graph construction.")
    parser.add_argument("--cities", type=str, default=",".join(DEFAULT_CITIES))
    parser.add_argument("--k_values", type=str, default="0,2,4,8")
    parser.add_argument("--conditions", type=str, default="B0_V1_baseline,G15_ECHF_main,G20_SE_main")
    parser.add_argument("--urban_root", type=str, default="data/urban_network_datasets")
    parser.add_argument("--out_root", type=str, default="data")
    parser.add_argument("--tag", type=str, default="urban_feature_knn_ablation_e20_h32_s0")
    parser.add_argument("--feature_profile", type=str, default="morph_env")
    parser.add_argument("--feature_knn_temp", type=float, default=3.0)
    parser.add_argument("--feature_edge_blend", type=float, default=0.18)
    parser.add_argument("--topk_per_node", type=int, default=24)
    parser.add_argument("--max_plots_per_street", type=int, default=60)
    parser.add_argument("--max_plots_per_building", type=int, default=40)
    parser.add_argument("--max_plots_per_junction", type=int, default=48)
    parser.add_argument("--include_sparse_city_specific", action="store_true")
    parser.add_argument("--include_landuse_features", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--train_log_interval", type=int, default=10)
    parser.add_argument("--hid_dim", type=int, default=32)
    parser.add_argument("--n_cluster_trials", type=int, default=1)
    parser.add_argument("--max_nums", type=int, default=64)
    parser.add_argument("--knn", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--amp_bf16", action="store_true")
    parser.add_argument("--known_only_eval", action="store_true")
    parser.add_argument("--force_rerun", action="store_true")
    args = parser.parse_args()

    cities = parse_list(args.cities)
    k_values = parse_int_list(args.k_values)

    datasets = prepare_graphs(repo=repo, cities=cities, k_values=k_values, args=args)
    out_dir = run_benchmark(repo=repo, datasets=datasets, args=args)
    graph_diagnostics(repo=repo, datasets=datasets, out_dir=out_dir)
    aggregate_results(out_dir=out_dir)
    print(f"[ok] urban feature-KNN ablation results: {out_dir}")


if __name__ == "__main__":
    main()
