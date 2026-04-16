#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean


REPO_ROOT = Path("/home/aitx/workspace/projects/edge-enhanced-unsupervised-graph-neural-network-clustering").resolve()
RESULT_ROOT = REPO_ROOT / "archive" / "workspaces" / "ep_compare_run" / "results"

MODELS = [
    {
        "model": "B45",
        "workspace": REPO_ROOT,
        "preset": "b45_v31_msgcond_gs050",
        "family": "baseline_mainline",
    },
    {
        "model": "EP1",
        "workspace": REPO_ROOT / "archive" / "workspaces" / "expert1_conda_run",
        "preset": "ep1_main",
        "family": "expert1_contextual_edge_state",
    },
    {
        "model": "EP2",
        "workspace": REPO_ROOT / "archive" / "workspaces" / "expert2_conda_run",
        "preset": "ep2_main",
        "family": "expert2_relation_state",
    },
]

DATASETS = [
    {
        "dataset": "synth_mech_full_v1_h85_s90_ds00",
        "group": "synthetic_control",
        "credibility": "highest",
        "edge_semantic_type": "synthetic controlled",
        "rationale": "Edge attributes are generated with explicit homophily and edge-signal targets.",
        "seeds": [0, 1],
        "epochs": 80,
        "max_nums": 12,
        "n_cluster_trials": 2,
    },
    {
        "dataset": "synth_mech_full_v1_h85_s90_ds00_permEA",
        "group": "synthetic_control_perm",
        "credibility": "highest",
        "edge_semantic_type": "synthetic permutation stress",
        "rationale": "Same synthetic graph as h85 but edge attributes are permuted to break semantics.",
        "seeds": [0, 1],
        "epochs": 80,
        "max_nums": 12,
        "n_cluster_trials": 2,
    },
    {
        "dataset": "synth_mech_full_v1_h65_s90_ds00",
        "group": "synthetic_control",
        "credibility": "highest",
        "edge_semantic_type": "synthetic controlled",
        "rationale": "Controlled mid-homophily synthetic graph with known edge-signal target.",
        "seeds": [0, 1],
        "epochs": 80,
        "max_nums": 12,
        "n_cluster_trials": 2,
    },
    {
        "dataset": "synth_mech_full_v1_h65_s90_ds00_permEA",
        "group": "synthetic_control_perm",
        "credibility": "highest",
        "edge_semantic_type": "synthetic permutation stress",
        "rationale": "Permutation control for the h65 synthetic graph.",
        "seeds": [0, 1],
        "epochs": 80,
        "max_nums": 12,
        "n_cluster_trials": 2,
    },
    {
        "dataset": "synth_mech_full_v1_h45_s00_ds00",
        "group": "synthetic_control",
        "credibility": "highest",
        "edge_semantic_type": "synthetic controlled",
        "rationale": "Low-homophily synthetic graph; useful for checking failure modes under weak semantics.",
        "seeds": [0, 1],
        "epochs": 80,
        "max_nums": 12,
        "n_cluster_trials": 2,
    },
    {
        "dataset": "synth_mech_full_v1_h45_s00_ds00_permEA",
        "group": "synthetic_control_perm",
        "credibility": "highest",
        "edge_semantic_type": "synthetic permutation stress",
        "rationale": "Permutation control for the h45 synthetic graph.",
        "seeds": [0, 1],
        "epochs": 80,
        "max_nums": 12,
        "n_cluster_trials": 2,
    },
    {
        "dataset": "urban_paris_plot_v3bsjg",
        "group": "urban_v3b_native",
        "credibility": "high",
        "edge_semantic_type": "native modeled urban relations",
        "rationale": "Later urban V3b graph: street-boundary plus junction edges, with geometric fallback only for under-connected plots.",
        "seeds": [0],
        "epochs": 60,
        "max_nums": 64,
        "n_cluster_trials": 2,
    },
    {
        "dataset": "urban_boston_plot_v3bsjg",
        "group": "urban_v3b_native",
        "credibility": "high",
        "edge_semantic_type": "native modeled urban relations",
        "rationale": "Later urban V3b graph with explicit street and junction semantics and limited geometric fallback.",
        "seeds": [0],
        "epochs": 60,
        "max_nums": 64,
        "n_cluster_trials": 2,
    },
    {
        "dataset": "urban_bangkok_plot_v3bsjg",
        "group": "urban_v3b_native",
        "credibility": "high",
        "edge_semantic_type": "native modeled urban relations",
        "rationale": "Larger later-stage urban V3b graph; useful to test whether EP1/EP2 survive on higher-scale native edge semantics.",
        "seeds": [0],
        "epochs": 60,
        "max_nums": 64,
        "n_cluster_trials": 2,
    },
    {
        "dataset": "dblp_magnn_author_v2",
        "group": "native_multirelation",
        "credibility": "high",
        "edge_semantic_type": "native bibliographic meta-relations",
        "rationale": "Edge channels are explicit APA/TERM/CONF author relations rather than constructed generic edge features.",
        "seeds": [0],
        "epochs": 60,
        "max_nums": 12,
        "n_cluster_trials": 2,
    },
    {
        "dataset": "entities_aifb",
        "group": "native_multirelation_partial_label",
        "credibility": "medium",
        "edge_semantic_type": "native KG relations",
        "rationale": "Relation types are real, but supervision coverage is partial and evaluation can be noisy for clustering.",
        "seeds": [0],
        "epochs": 60,
        "max_nums": 16,
        "n_cluster_trials": 2,
    },
    {
        "dataset": "fraud_yelp_homo",
        "group": "real_semantic_task_mismatch",
        "credibility": "medium_low",
        "edge_semantic_type": "real relation channels with task mismatch",
        "rationale": "Edge channels are real interaction networks, but the label semantics are fraud vs non-fraud rather than community structure.",
        "seeds": [0],
        "epochs": 60,
        "max_nums": 2,
        "n_cluster_trials": 2,
    },
    {
        "dataset": "cora",
        "group": "derived_generic",
        "credibility": "low",
        "edge_semantic_type": "derived generic edge features",
        "rationale": "No native edge_attr; the pipeline synthesizes generic edge features from nodes and degrees.",
        "seeds": [0],
        "epochs": 60,
        "max_nums": 10,
        "n_cluster_trials": 2,
    },
    {
        "dataset": "photo",
        "group": "derived_generic",
        "credibility": "low",
        "edge_semantic_type": "derived generic edge features",
        "rationale": "No native edge_attr; used only as a sanity check for stability, not for semantic-edge claims.",
        "seeds": [0],
        "epochs": 60,
        "max_nums": 10,
        "n_cluster_trials": 2,
    },
]


def parse_csv_list(s: str) -> set[str]:
    return {x.strip() for x in s.split(",") if x.strip()}


def version_name(prefix: str, model: str, dataset: str, seed: int) -> str:
    safe_dataset = dataset.replace("/", "_")
    return f"{prefix}_{model.lower()}_{safe_dataset}_s{seed}"


def metrics_path(workspace: Path, version: str, dataset: str) -> Path:
    return workspace / "results" / version / f"{dataset}_metrics.json"


def run_one(model_cfg: dict, data_cfg: dict, seed: int, gpu: int, prefix: str, force: bool) -> dict:
    workspace = Path(model_cfg["workspace"]).resolve()
    version = version_name(prefix, model_cfg["model"], data_cfg["dataset"], seed)
    mpath = metrics_path(workspace, version, data_cfg["dataset"])
    status = "cached"
    start = time.time()

    if force or (not mpath.exists()):
        status = "ran"
        cmd = [
            "conda",
            "run",
            "-n",
            "gnn",
            "python",
            "tools/run_preset.py",
            "--preset",
            str(model_cfg["preset"]),
            "--dataset",
            str(data_cfg["dataset"]),
            "--seed",
            str(seed),
            "--gpu",
            str(gpu),
            "--epochs",
            str(data_cfg["epochs"]),
            "--eval_freq",
            str(data_cfg["epochs"]),
            "--train_log_interval",
            str(max(20, int(data_cfg["epochs"]) // 4)),
            "--n_cluster_trials",
            str(data_cfg["n_cluster_trials"]),
            "--max_nums",
            str(data_cfg["max_nums"]),
            "--version",
            version,
        ]
        print(f"[run] model={model_cfg['model']} dataset={data_cfg['dataset']} seed={seed}")
        proc = subprocess.run(cmd, cwd=workspace)
        status = "ok" if proc.returncode == 0 else f"failed:{proc.returncode}"

    elapsed = time.time() - start
    row = {
        "model": model_cfg["model"],
        "preset": model_cfg["preset"],
        "workspace": str(workspace),
        "family": model_cfg["family"],
        "dataset": data_cfg["dataset"],
        "group": data_cfg["group"],
        "credibility": data_cfg["credibility"],
        "edge_semantic_type": data_cfg["edge_semantic_type"],
        "rationale": data_cfg["rationale"],
        "seed": seed,
        "epochs": data_cfg["epochs"],
        "max_nums": data_cfg["max_nums"],
        "version": version,
        "status": status,
        "elapsed_sec": elapsed,
        "metrics_path": str(mpath),
    }
    if mpath.exists():
        obj = json.load(open(mpath, "r", encoding="utf-8"))
        metric_key_map = {
            "final_nmi": ["final_nmi", "nmi_mean"],
            "final_ari": ["final_ari", "ari_mean"],
            "si_loss": ["si_loss", "si_loss_mean"],
            "modularity": ["modularity", "modularity_mean"],
            "conductance_mean": ["conductance_mean", "conductance_mean"],
            "conductance_weighted": ["conductance_weighted", "conductance_weighted_mean"],
            "best_epoch": ["best_epoch", "best_epoch_mean"],
            "best_train_loss": ["best_train_loss", "best_train_loss_mean"],
            "final_edge_reliability": ["final_edge_reliability", "final_edge_reliability_mean"],
            "final_edge_factor_mean": ["final_edge_factor_mean", "final_edge_factor_mean_mean"],
            "final_msg_gate_factor_mean": ["final_msg_gate_factor_mean", "final_msg_gate_factor_mean"],
            "final_hier_edge_nonzero_ratio": ["final_hier_edge_nonzero_ratio", "final_hier_edge_nonzero_ratio_mean"],
            "diag_all_required_live": ["diag_all_required_live", "diag_all_required_live_mean"],
            "diag_dead_branch_count": ["diag_dead_branch_count", "diag_dead_branch_count_mean"],
        }
        for out_key, candidate_keys in metric_key_map.items():
            row[out_key] = None
            for candidate in candidate_keys:
                if candidate in obj:
                    row[out_key] = obj.get(candidate)
                    break
    return row


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_or_none(vals: list[float]) -> float | None:
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return None
    return mean(vals)


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    ok_rows = [r for r in rows if str(r.get("status", "")).startswith(("ok", "cached", "ran")) and r.get("final_nmi") is not None]

    by_dataset_model: list[dict] = []
    for dataset in sorted({r["dataset"] for r in ok_rows}):
        for model in sorted({r["model"] for r in ok_rows}):
            subset = [r for r in ok_rows if r["dataset"] == dataset and r["model"] == model]
            if not subset:
                continue
            sample = subset[0]
            by_dataset_model.append(
                {
                    "dataset": dataset,
                    "group": sample["group"],
                    "credibility": sample["credibility"],
                    "edge_semantic_type": sample["edge_semantic_type"],
                    "model": model,
                    "runs": len(subset),
                    "nmi_mean": mean_or_none([r.get("final_nmi") for r in subset]),
                    "ari_mean": mean_or_none([r.get("final_ari") for r in subset]),
                    "si_loss_mean": mean_or_none([r.get("si_loss") for r in subset]),
                    "modularity_mean": mean_or_none([r.get("modularity") for r in subset]),
                    "conductance_mean_mean": mean_or_none([r.get("conductance_mean") for r in subset]),
                    "best_epoch_mean": mean_or_none([r.get("best_epoch") for r in subset]),
                }
            )

    by_group_model: list[dict] = []
    for group in sorted({r["group"] for r in ok_rows}):
        for model in sorted({r["model"] for r in ok_rows}):
            subset = [r for r in ok_rows if r["group"] == group and r["model"] == model]
            if not subset:
                continue
            sample = subset[0]
            by_group_model.append(
                {
                    "group": group,
                    "credibility": sample["credibility"],
                    "edge_semantic_type": sample["edge_semantic_type"],
                    "model": model,
                    "runs": len(subset),
                    "nmi_mean": mean_or_none([r.get("final_nmi") for r in subset]),
                    "ari_mean": mean_or_none([r.get("final_ari") for r in subset]),
                    "si_loss_mean": mean_or_none([r.get("si_loss") for r in subset]),
                    "modularity_mean": mean_or_none([r.get("modularity") for r in subset]),
                    "conductance_mean_mean": mean_or_none([r.get("conductance_mean") for r in subset]),
                }
            )

    leaderboard: list[dict] = []
    for dataset in sorted({r["dataset"] for r in by_dataset_model}):
        ordered = sorted(
            [r for r in by_dataset_model if r["dataset"] == dataset],
            key=lambda x: (
                -(x["nmi_mean"] if x["nmi_mean"] is not None else -1e9),
                -(x["ari_mean"] if x["ari_mean"] is not None else -1e9),
                (x["si_loss_mean"] if x["si_loss_mean"] is not None else 1e9),
            ),
        )
        for rank, row in enumerate(ordered, start=1):
            leaderboard.append(
                {
                    "dataset": dataset,
                    "group": row["group"],
                    "credibility": row["credibility"],
                    "rank": rank,
                    "model": row["model"],
                    "nmi_mean": row["nmi_mean"],
                    "ari_mean": row["ari_mean"],
                    "si_loss_mean": row["si_loss_mean"],
                }
            )
    return by_dataset_model, by_group_model, leaderboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B45 vs EP1 vs EP2 across credibility-layered datasets.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--version_prefix", type=str, default="ep_compare_v1")
    parser.add_argument("--only_models", type=str, default="")
    parser.add_argument("--only_datasets", type=str, default="")
    args = parser.parse_args()

    keep_models = parse_csv_list(args.only_models) if args.only_models else None
    keep_datasets = parse_csv_list(args.only_datasets) if args.only_datasets else None

    selected_models = [m for m in MODELS if keep_models is None or m["model"] in keep_models]
    selected_datasets = [d for d in DATASETS if keep_datasets is None or d["dataset"] in keep_datasets]

    rows: list[dict] = []
    catalog_rows = []
    for d in selected_datasets:
        catalog_rows.append(
            {
                "dataset": d["dataset"],
                "group": d["group"],
                "credibility": d["credibility"],
                "edge_semantic_type": d["edge_semantic_type"],
                "rationale": d["rationale"],
                "seeds": ",".join(str(s) for s in d["seeds"]),
                "epochs": d["epochs"],
                "max_nums": d["max_nums"],
            }
        )
        for seed in d["seeds"]:
            for model in selected_models:
                rows.append(run_one(model, d, seed, args.gpu, args.version_prefix, args.force))

    out_dir = RESULT_ROOT / args.version_prefix
    write_csv(out_dir / "runs.csv", rows)
    write_csv(out_dir / "dataset_catalog.csv", catalog_rows)
    by_dataset_model, by_group_model, leaderboard = summarize(rows)
    write_csv(out_dir / "summary_by_dataset_model.csv", by_dataset_model)
    write_csv(out_dir / "summary_by_group_model.csv", by_group_model)
    write_csv(out_dir / "leaderboard_by_dataset.csv", leaderboard)

    print(f"Wrote results to {out_dir}")
    for row in by_group_model:
        print(
            row["group"],
            row["model"],
            f"nmi={row['nmi_mean']}",
            f"ari={row['ari_mean']}",
            f"si={row['si_loss_mean']}",
        )


if __name__ == "__main__":
    main()
