#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None


PRESET_MAP = {
    "baseline_v1": "configs/presets/baseline_v1.json",
    "g20_se_consistent_main": "configs/presets/g20_se_consistent_main.json",
    "b30_dualscalar": "configs/presets/b30_dualscalar.json",
    "b31_dualscalar_assign": "configs/presets/b31_dualscalar_assign.json",
    "b32_dualscalar_assign_hier": "configs/presets/b32_dualscalar_assign_hier.json",
    "b33_dualscalar_assign_hier_aug": "configs/presets/b33_dualscalar_assign_hier_aug.json",
    "b34_v33_augsmall": "configs/presets/b34_v33_augsmall.json",
    "b35_v33_augpositive": "configs/presets/b35_v33_augpositive.json",
    "b36_v33_augpositive_small": "configs/presets/b36_v33_augpositive_small.json",
    "b37_v32_hardhier": "configs/presets/b37_v32_hardhier.json",
    "b38_v32_topk3": "configs/presets/b38_v32_topk3.json",
    "b40_v31_msgcond": "configs/presets/b40_v31_msgcond.json",
    "b41_v32_confpool": "configs/presets/b41_v32_confpool.json",
    "b42_v32_msgcond_confpool": "configs/presets/b42_v32_msgcond_confpool.json",
    "b43_v31_msgcond_matchonly": "configs/presets/b43_v31_msgcond_matchonly.json",
    "b44_v31_msgcond_gs020": "configs/presets/b44_v31_msgcond_gs020.json",
    "b45_v31_msgcond_gs050": "configs/presets/b45_v31_msgcond_gs050.json",
    "b46_v31_msgcond_confgate": "configs/presets/b46_v31_msgcond_confgate.json",
    "b47_v31_msgcond_gs050_matchonly": "configs/presets/b47_v31_msgcond_gs050_matchonly.json",
    "b48_v31_msgcond_gs050_confgate": "configs/presets/b48_v31_msgcond_gs050_confgate.json",
    "v40_edge_state_ctx": "configs/presets/v40_edge_state_ctx.json",
    "v40_edge_state_noctx": "configs/presets/v40_edge_state_noctx.json",
}

STAGE_PAIRS = [
    ("b31_dualscalar_assign", "b30_dualscalar", "assign_on"),
    ("b32_dualscalar_assign_hier", "b31_dualscalar_assign", "hier_on"),
    ("b33_dualscalar_assign_hier_aug", "b32_dualscalar_assign_hier", "aug_on"),
]


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str] | None = None) -> None:
    if fieldnames is None:
        keys = set()
        for row in rows:
            keys.update(row.keys())
        fieldnames = list(keys)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean_group(rows: List[Dict[str, object]], group_keys: List[str], mean_fields: List[str], count_name: str | None = None) -> List[Dict[str, object]]:
    buckets: Dict[tuple, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in group_keys)].append(row)
    out = []
    for key, items in buckets.items():
        agg = {k: v for k, v in zip(group_keys, key)}
        if count_name is not None:
            agg[count_name] = len(items)
        for field in mean_fields:
            vals = [float(it[field]) for it in items if it.get(field) is not None]
            agg[field] = sum(vals) / len(vals) if vals else float("nan")
        out.append(agg)
    return out


def sort_rows(rows: List[Dict[str, object]], keys: List[str], reverse: bool = False) -> List[Dict[str, object]]:
    return sorted(rows, key=lambda r: tuple(r.get(k) for k in keys), reverse=reverse)


def parse_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def load_preset(repo_root: Path, name_or_path: str) -> Dict[str, object]:
    p = Path(name_or_path)
    if p.exists():
        target = p
    else:
        if name_or_path not in PRESET_MAP:
            raise FileNotFoundError(f"Unknown preset '{name_or_path}'.")
        target = repo_root / PRESET_MAP[name_or_path]
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


def build_cmd(repo_root: Path, dataset: str, seed: int, version: str, preset: Dict[str, object], args) -> List[str]:
    cmd = [
        sys.executable,
        "main.py",
        "--dataset", dataset,
        "--root_path", args.root_path,
        "--version", version,
        "--seed", str(seed),
        "--gpu", str(args.gpu),
        "--epochs", str(args.epochs),
        "--eval_freq", str(args.eval_freq),
        "--train_log_interval", str(args.train_log_interval),
        "--hid_dim", str(args.hid_dim),
        "--n_cluster_trials", str(args.n_cluster_trials),
        "--exp_iters", "1",
        "--patience", "0",
        "--max_nums", str(args.max_nums),
        "--edge_variant", str(preset.get("edge_variant", "V1")),
        "--edge_hybrid_alpha", str(preset.get("edge_hybrid_alpha", 0.5)),
        "--edge_feat_temp", str(preset.get("edge_feat_temp", 1.0)),
        "--edge_input_prior_alpha", str(preset.get("edge_input_prior_alpha", 0.0)),
        "--edge_fusion_gamma", str(preset.get("edge_fusion_gamma", 1.0)),
        "--edge_fusion_gamma_sched_epochs", str(preset.get("edge_fusion_gamma_sched_epochs", 0)),
        "--edge_confidence_quantile", str(preset.get("edge_confidence_quantile", 0.0)),
        "--edge_adaptive_alpha_strength", str(preset.get("edge_adaptive_alpha_strength", 2.0)),
        "--edge_adaptive_alpha_bias", str(preset.get("edge_adaptive_alpha_bias", 0.0)),
        "--edge_reliability_temp", str(preset.get("edge_reliability_temp", 1.0)),
        "--edge_attr_hidden_dim", str(preset.get("edge_attr_hidden_dim", 64)),
        "--edge_attr_fusion_scale", str(preset.get("edge_attr_fusion_scale", 1.0)),
        "--edge_attr_pool_topk", str(preset.get("edge_attr_pool_topk", 1)),
        "--edge_msg_gate_scale", str(preset.get("edge_msg_gate_scale", 0.35)),
        "--edge_msg_confidence_temp", str(preset.get("edge_msg_confidence_temp", 1.0)),
        "--edge_attr_pool_conf_power", str(preset.get("edge_attr_pool_conf_power", 1.0)),
        "--edge_weight_learn_reg_lambda", str(preset.get("edge_weight_learn_reg_lambda", 0.02)),
        "--edge_weight_learn_logclip", str(preset.get("edge_weight_learn_logclip", 0.8)),
        "--edge_weight_learn_temp", str(preset.get("edge_weight_learn_temp", 1.0)),
        "--edge_weight_learn_apply_to", str(preset.get("edge_weight_learn_apply_to", "both")),
        "--edge_aug_prior_scale", str(preset.get("edge_aug_prior_scale", 0.0)),
        "--edge_aug_prior_mode", str(preset.get("edge_aug_prior_mode", "raw")),
        "--edge_attr_weight_blend", str(preset.get("edge_attr_weight_blend", 0.0)),
        "--edge_attr_weight_temp", str(preset.get("edge_attr_weight_temp", 1.0)),
        "--edge_attr_weight_apply_to", str(preset.get("edge_attr_weight_apply_to", "si_only")),
        "--edge_state_temp", str(preset.get("edge_state_temp", 1.0)),
        "--edge_state_lambda_boundary", str(preset.get("edge_state_lambda_boundary", 0.1)),
        "--edge_state_lambda_support", str(preset.get("edge_state_lambda_support", 0.1)),
    ]
    if bool(preset.get("edge_adaptive_alpha", False)):
        cmd.append("--edge_adaptive_alpha")
    if bool(preset.get("edge_attr_hierarchical", False)):
        cmd.append("--edge_attr_hierarchical")
    if bool(preset.get("edge_msg_conditioned", False)):
        cmd.append("--edge_msg_conditioned")
    if bool(preset.get("edge_msg_matched_only", False)):
        cmd.append("--edge_msg_matched_only")
    if bool(preset.get("edge_msg_confidence_gate", False)):
        cmd.append("--edge_msg_confidence_gate")
    if bool(preset.get("edge_attr_pool_confidence", False)):
        cmd.append("--edge_attr_pool_confidence")
    if bool(preset.get("edge_state_use_context", False)):
        cmd.append("--edge_state_use_context")
    if preset.get("edge_fusion_gamma_start", None) is not None:
        cmd += ["--edge_fusion_gamma_start", str(preset["edge_fusion_gamma_start"])]
    if preset.get("edge_fusion_gamma_end", None) is not None:
        cmd += ["--edge_fusion_gamma_end", str(preset["edge_fusion_gamma_end"])]
    return cmd


def run_one(cmd: List[str], cwd: Path, log_path: Path) -> int:
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def metric_float(metrics: Dict[str, object], key: str) -> float:
    try:
        return float(metrics.get(key, float("nan")))
    except Exception:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Component-level diagnostics for B30 family")
    parser.add_argument("--tag", type=str, default="diagnostic_b30_components_v1")
    parser.add_argument(
        "--datasets",
        type=str,
        default="synth_edgectrl_v1_mredu_h65_s90_ds00,synth_edgectrl_v1_mmisl_h65_s90_ds00,synth_edgectrl_v1_mhier_h65_s90_ds00",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="baseline_v1,g20_se_consistent_main,b30_dualscalar,b31_dualscalar_assign,b32_dualscalar_assign_hier,b33_dualscalar_assign_hier_aug",
    )
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--root_path", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--eval_freq", type=int, default=60)
    parser.add_argument("--train_log_interval", type=int, default=20)
    parser.add_argument("--hid_dim", type=int, default=128)
    parser.add_argument("--max_nums", type=int, default=12)
    parser.add_argument("--n_cluster_trials", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "results" / args.tag
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    datasets = parse_list(args.datasets)
    conditions = parse_list(args.conditions)
    seeds = parse_int_list(args.seeds)
    presets = {cond: load_preset(repo_root, cond) for cond in conditions}

    rows = []
    total = len(datasets) * len(conditions) * len(seeds)
    done = 0
    for dataset in datasets:
        for seed in seeds:
            for cond in conditions:
                done += 1
                version = f"{args.tag}_{cond}_{dataset}_s{seed}"
                log_path = logs_dir / f"{version}.log"
                metrics_path = repo_root / "results" / version / f"{dataset}_metrics.json"
                cmd = build_cmd(repo_root, dataset, seed, version, presets[cond], args)
                print(f"[{done}/{total}] {dataset} | {cond} | seed={seed}")
                t0 = time.time()
                if args.dry_run:
                    print("CMD:", " ".join(cmd))
                    rc = 0
                elif args.resume and metrics_path.exists():
                    rc = 0
                else:
                    rc = run_one(cmd, repo_root, log_path)
                sec = time.time() - t0

                metrics = {}
                if metrics_path.exists():
                    try:
                        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    except Exception:
                        metrics = {}

                rows.append({
                    "dataset": dataset,
                    "seed": seed,
                    "condition": cond,
                    "rc": rc,
                    "runtime_sec": sec,
                    "nmi_mean": metric_float(metrics, "nmi_mean"),
                    "ari_mean": metric_float(metrics, "ari_mean"),
                    "acc_mean": metric_float(metrics, "acc_mean"),
                    "si_loss_mean": metric_float(metrics, "si_loss_mean"),
                    "modularity_mean": metric_float(metrics, "modularity_mean"),
                    "conductance_weighted_mean": metric_float(metrics, "conductance_weighted_mean"),
                    "graph_alpha_mean": metric_float(metrics, "final_graph_alpha_mean"),
                    "edge_rel_mean": metric_float(metrics, "final_edge_reliability_mean"),
                    "edge_mix_mean": metric_float(metrics, "final_edge_mix_beta_mean"),
                    "edge_factor_mean": metric_float(metrics, "final_edge_factor_mean_mean"),
                    "edge_factor_msg_mean": metric_float(metrics, "final_edge_factor_msg_mean"),
                    "edge_factor_si_mean": metric_float(metrics, "final_edge_factor_si_mean"),
                    "msg_gate_factor_mean": metric_float(metrics, "final_msg_gate_factor_mean"),
                    "msg_gate_factor_std": metric_float(metrics, "final_msg_gate_factor_std"),
                    "edge_aug_bias_mean": metric_float(metrics, "final_edge_aug_bias_mean"),
                    "edge_aug_bias_std": metric_float(metrics, "final_edge_aug_bias_std"),
                    "edge_reg_mean": metric_float(metrics, "final_edge_reg_mean"),
                    "edge_state_support_mean": metric_float(metrics, "final_edge_state_support_mean"),
                    "edge_state_boundary_mean": metric_float(metrics, "final_edge_state_boundary_mean"),
                    "edge_state_neutral_mean": metric_float(metrics, "final_edge_state_neutral_mean"),
                    "edge_state_entropy_mean": metric_float(metrics, "final_edge_state_entropy_mean"),
                    "edge_state_support_loss_mean": metric_float(metrics, "final_edge_state_support_loss_mean"),
                    "edge_state_boundary_loss_mean": metric_float(metrics, "final_edge_state_boundary_loss_mean"),
                    "hier_levels_active_ratio_mean": metric_float(metrics, "final_hier_edge_levels_active_ratio_mean"),
                    "hier_nonzero_ratio_mean": metric_float(metrics, "final_hier_edge_nonzero_ratio_mean"),
                    "hier_mean_abs_mean": metric_float(metrics, "final_hier_edge_mean_abs_mean"),
                    "diag_factor_live_mean": metric_float(metrics, "diag_factor_live_mean"),
                    "diag_dual_live_mean": metric_float(metrics, "diag_dual_live_mean"),
                    "diag_msg_live_mean": metric_float(metrics, "diag_msg_live_mean"),
                    "diag_assign_live_mean": metric_float(metrics, "diag_assign_live_mean"),
                    "diag_hier_live_mean": metric_float(metrics, "diag_hier_live_mean"),
                    "diag_aug_live_mean": metric_float(metrics, "diag_aug_live_mean"),
                    "diag_role_live_mean": metric_float(metrics, "diag_role_live_mean"),
                    "diag_dead_branch_count_mean": metric_float(metrics, "diag_dead_branch_count_mean"),
                    "diag_all_required_live_mean": metric_float(metrics, "diag_all_required_live_mean"),
                    "metrics_path": str(metrics_path),
                    "log_path": str(log_path),
                })

    write_csv(out_dir / "runs.csv", rows, fieldnames=list(rows[0].keys()) if rows else None)
    ok = [row for row in rows if int(row["rc"]) == 0]
    if not ok:
        return

    summary_fields = [
        "nmi_mean",
        "ari_mean",
        "si_loss_mean",
        "graph_alpha_mean",
        "edge_rel_mean",
        "edge_mix_mean",
        "edge_factor_mean",
        "edge_factor_msg_mean",
        "edge_factor_si_mean",
        "msg_gate_factor_mean",
        "msg_gate_factor_std",
        "edge_aug_bias_mean",
        "edge_aug_bias_std",
        "edge_reg_mean",
        "edge_state_support_mean",
        "edge_state_boundary_mean",
        "edge_state_neutral_mean",
        "edge_state_entropy_mean",
        "edge_state_support_loss_mean",
        "edge_state_boundary_loss_mean",
        "hier_levels_active_ratio_mean",
        "hier_nonzero_ratio_mean",
        "diag_factor_live_mean",
        "diag_dual_live_mean",
        "diag_msg_live_mean",
        "diag_assign_live_mean",
        "diag_hier_live_mean",
        "diag_aug_live_mean",
        "diag_role_live_mean",
        "diag_dead_branch_count_mean",
        "diag_all_required_live_mean",
    ]
    summary = mean_group(ok, ["condition"], summary_fields, count_name="runs")
    summary = sort_rows(summary, ["nmi_mean", "ari_mean"], reverse=True)
    write_csv(out_dir / "summary_by_condition.csv", summary)

    health_fields = [
        "condition",
        "diag_factor_live_mean",
        "diag_dual_live_mean",
        "diag_msg_live_mean",
        "diag_assign_live_mean",
        "diag_hier_live_mean",
        "diag_aug_live_mean",
        "diag_role_live_mean",
        "diag_dead_branch_count_mean",
        "diag_all_required_live_mean",
        "hier_levels_active_ratio_mean",
        "hier_nonzero_ratio_mean",
        "edge_aug_bias_mean",
        "edge_aug_bias_std",
        "edge_state_support_mean",
        "edge_state_boundary_mean",
        "edge_state_entropy_mean",
    ]
    health = [{k: row.get(k) for k in health_fields} for row in summary]
    write_csv(out_dir / "branch_health_summary.csv", health, fieldnames=health_fields)

    by_cd_fields = [
        "nmi_mean",
        "ari_mean",
        "si_loss_mean",
        "edge_factor_mean",
        "msg_gate_factor_mean",
        "diag_dead_branch_count_mean",
        "diag_all_required_live_mean",
    ]
    by_cd = mean_group(ok, ["dataset", "condition"], by_cd_fields)
    by_cd = sort_rows(by_cd, ["dataset", "condition"])
    write_csv(out_dir / "summary_by_condition_dataset.csv", by_cd)

    stage_delta_rows: List[Dict[str, object]] = []
    by_key = {(row["dataset"], row["seed"], row["condition"]): row for row in ok}
    for curr, prev, stage_name in STAGE_PAIRS:
        for row in ok:
            if row["condition"] != curr:
                continue
            prev_row = by_key.get((row["dataset"], row["seed"], prev))
            if prev_row is None:
                continue
            stage_delta_rows.append({
                "dataset": row["dataset"],
                "seed": row["seed"],
                "stage_name": stage_name,
                "condition_curr": curr,
                "condition_prev": prev,
                "delta_nmi": float(row["nmi_mean"]) - float(prev_row["nmi_mean"]),
                "delta_ari": float(row["ari_mean"]) - float(prev_row["ari_mean"]),
                "delta_si_loss": float(row["si_loss_mean"]) - float(prev_row["si_loss_mean"]),
                "delta_dead_branch_count": float(row["diag_dead_branch_count_mean"]) - float(prev_row["diag_dead_branch_count_mean"]),
                "delta_all_required_live": float(row["diag_all_required_live_mean"]) - float(prev_row["diag_all_required_live_mean"]),
            })

    if stage_delta_rows:
        write_csv(out_dir / "stage_delta_runs.csv", stage_delta_rows)
        stage_summary_fields = [
            "delta_nmi",
            "delta_ari",
            "delta_si_loss",
            "delta_dead_branch_count",
            "delta_all_required_live",
        ]
        stage_delta_summary = mean_group(
            stage_delta_rows,
            ["stage_name", "condition_prev", "condition_curr"],
            stage_summary_fields,
        )
        for row in stage_delta_summary:
            row["delta_nmi_mean"] = row.pop("delta_nmi")
            row["delta_ari_mean"] = row.pop("delta_ari")
            row["delta_si_loss_mean"] = row.pop("delta_si_loss")
            row["delta_dead_branch_count_mean"] = row.pop("delta_dead_branch_count")
            row["delta_all_required_live_mean"] = row.pop("delta_all_required_live")
        stage_delta_summary = sort_rows(stage_delta_summary, ["stage_name"])
        write_csv(out_dir / "stage_delta_summary.csv", stage_delta_summary)

        stage_delta_by_dataset = mean_group(
            stage_delta_rows,
            ["dataset", "stage_name", "condition_prev", "condition_curr"],
            stage_summary_fields,
        )
        for row in stage_delta_by_dataset:
            row["delta_nmi_mean"] = row.pop("delta_nmi")
            row["delta_ari_mean"] = row.pop("delta_ari")
            row["delta_si_loss_mean"] = row.pop("delta_si_loss")
            row["delta_dead_branch_count_mean"] = row.pop("delta_dead_branch_count")
            row["delta_all_required_live_mean"] = row.pop("delta_all_required_live")
        stage_delta_by_dataset = sort_rows(stage_delta_by_dataset, ["dataset", "stage_name"])
        write_csv(out_dir / "stage_delta_by_dataset.csv", stage_delta_by_dataset)

    print(f"[ok] wrote {out_dir / 'summary_by_condition.csv'}")


if __name__ == "__main__":
    main()
