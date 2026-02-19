#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_DATASETS = [
    "urban_bangkok_plot",
    "urban_beijing_plot",
    "urban_boston_plot",
    "urban_chicago_plot",
    "urban_johannesburg_plot",
    "urban_madrid_plot",
    "urban_melbourne_plot",
    "urban_paris_plot",
    "urban_shanghai_plot",
    "urban_singapore_plot",
    "urban_sydney_plot",
    "urban_tokyo_plot",
    "urban_washingtondc_plot",
]


STAGE2_RE = re.compile(
    r"\[Stage2\]\s+Epoch\s+(\d+):\s+loss=([0-9eE+\-.]+),\s+edge_fusion_gamma=([0-9eE+\-.]+),\s+graph_alpha=([0-9eE+\-.]+),\s+edge_rel=([0-9eE+\-.]+)"
)
EVAL_RE = re.compile(
    r"Epoch\s+(\d+):\s+ACC:\s*([0-9eE+\-.]+),\s+NMI:\s*([0-9eE+\-.]+),\s+ARI:\s*([0-9eE+\-.]+)"
)


def parse_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def mean(vals: List[float]) -> float:
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def std(vals: List[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))


def run_one(repo: Path, dataset: str, seed: int, version: str, args) -> Dict[str, object]:
    out_dir = repo / "results" / version
    metrics_path = out_dir / f"{dataset}_metrics.json"
    if metrics_path.exists() and not args.force_rerun:
        return {
            "dataset": dataset,
            "seed": seed,
            "version": version,
            "status": "skip_exists",
            "seconds": 0.0,
            "metrics_path": str(metrics_path),
            "log_path": str(out_dir / f"{dataset}.log"),
            "stderr_tail": "",
        }

    cmd = [
        sys.executable,
        "main.py",
        "--dataset",
        dataset,
        "--version",
        version,
        "--epochs",
        str(args.epochs),
        "--eval_freq",
        str(args.eval_freq),
        "--train_log_interval",
        str(args.train_log_interval),
        "--seed",
        str(seed),
        "--gpu",
        str(args.gpu),
        "--hid_dim",
        str(args.hid_dim),
        "--max_nums",
        str(args.max_nums),
        "--exp_iters",
        "1",
        "--n_cluster_trials",
        "1",
        "--knn_mode",
        "edge",
        "--knn_auto_threshold",
        "20000",
        "--edge_variant",
        "V5",
        "--edge_fusion_gamma",
        "1.0",
        "--edge_fusion_gamma_start",
        "0.2",
        "--edge_fusion_gamma_end",
        "1.2",
        "--edge_fusion_gamma_sched_epochs",
        "60",
        "--edge_adaptive_alpha_strength",
        "1.0",
        "--edge_adaptive_alpha_bias",
        "2.0",
    ]

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    sec = time.time() - t0
    stderr_tail = (proc.stderr or "").strip().splitlines()[-1:] if proc.stderr else []
    return {
        "dataset": dataset,
        "seed": seed,
        "version": version,
        "status": "ok" if proc.returncode == 0 else f"fail_{proc.returncode}",
        "seconds": sec,
        "metrics_path": str(metrics_path),
        "log_path": str(out_dir / f"{dataset}.log"),
        "stderr_tail": (stderr_tail[0] if stderr_tail else ""),
    }


def parse_run_log(log_path: Path) -> Dict[str, object]:
    epochs: List[int] = []
    losses: List[float] = []
    gammas: List[float] = []
    graph_alphas: List[float] = []
    edge_rels: List[float] = []
    eval_epochs: List[int] = []
    eval_nmi: List[float] = []
    eval_ari: List[float] = []
    if not log_path.exists():
        return {
            "n_points": 0,
            "n_eval_points": 0,
            "losses": losses,
            "epochs": epochs,
            "gammas": gammas,
            "graph_alphas": graph_alphas,
            "edge_rels": edge_rels,
            "eval_epochs": eval_epochs,
            "eval_nmi": eval_nmi,
            "eval_ari": eval_ari,
        }

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            ms = STAGE2_RE.search(line)
            if ms:
                epochs.append(int(ms.group(1)))
                losses.append(float(ms.group(2)))
                gammas.append(float(ms.group(3)))
                graph_alphas.append(float(ms.group(4)))
                edge_rels.append(float(ms.group(5)))
                continue
            me = EVAL_RE.search(line)
            if me:
                eval_epochs.append(int(me.group(1)))
                eval_nmi.append(float(me.group(3)) / 100.0)
                eval_ari.append(float(me.group(4)) / 100.0)
    return {
        "n_points": len(losses),
        "n_eval_points": len(eval_nmi),
        "losses": losses,
        "epochs": epochs,
        "gammas": gammas,
        "graph_alphas": graph_alphas,
        "edge_rels": edge_rels,
        "eval_epochs": eval_epochs,
        "eval_nmi": eval_nmi,
        "eval_ari": eval_ari,
    }


def run_curve_metrics(losses: List[float]) -> Dict[str, float]:
    eps = 1e-12
    if not losses:
        return {
            "loss_start": float("nan"),
            "loss_end": float("nan"),
            "loss_rel_change_pct": float("nan"),
            "tail_drift_pct": float("nan"),
            "tail_band_pct": float("nan"),
            "tail_slope_per_epoch_pct": float("nan"),
            "sign_flip_ratio": float("nan"),
        }

    base = abs(losses[0]) + eps
    diffs = [losses[i + 1] - losses[i] for i in range(len(losses) - 1)]
    flip_cnt = 0
    for i in range(len(diffs) - 1):
        if diffs[i] * diffs[i + 1] < 0:
            flip_cnt += 1
    sign_flip_ratio = (flip_cnt / max(1, len(diffs) - 1)) if len(diffs) > 1 else 0.0

    k = min(len(losses), max(5, int(math.ceil(len(losses) * 0.25))))
    tail = losses[-k:]
    tail_drift_pct = 100.0 * abs(tail[-1] - tail[0]) / base if len(tail) > 1 else 0.0
    tail_band_pct = 100.0 * (max(tail) - min(tail)) / base
    tail_slope_per_epoch_pct = (
        100.0 * (tail[-1] - tail[0]) / (max(1, len(tail) - 1) * base)
        if len(tail) > 1
        else 0.0
    )

    return {
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "loss_rel_change_pct": 100.0 * (losses[-1] - losses[0]) / base,
        "tail_drift_pct": tail_drift_pct,
        "tail_band_pct": tail_band_pct,
        "tail_slope_per_epoch_pct": tail_slope_per_epoch_pct,
        "sign_flip_ratio": sign_flip_ratio,
    }


def classify_dataset(row: Dict[str, float]) -> str:
    # Conservative rule: judge by tail stability + cross-seed consistency.
    if int(row["n_runs_ok"]) < int(row["n_runs"]):
        return "incomplete"
    hard = 0
    soft = 0
    if row["tail_drift_pct_mean"] > 0.35:
        hard += 1
    elif row["tail_drift_pct_mean"] > 0.20:
        soft += 1
    if row["tail_band_pct_mean"] > 0.50:
        hard += 1
    elif row["tail_band_pct_mean"] > 0.30:
        soft += 1
    if row["final_loss_cv_pct"] > 0.30:
        hard += 1
    elif row["final_loss_cv_pct"] > 0.20:
        soft += 1
    if row["sign_flip_ratio_mean"] > 0.70:
        hard += 1
    elif row["sign_flip_ratio_mean"] > 0.55:
        soft += 1

    if hard >= 1:
        return "unstable"
    if soft >= 2:
        return "mostly_stable"
    return "stable"


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run and analyze convergence stability for urban datasets.")
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--train_log_interval", type=int, default=1)
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--max_nums", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tag", type=str, default="convergence_stability_v1")
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--skip_run", action="store_true")
    args = parser.parse_args()

    datasets = parse_list(args.datasets)
    seeds = parse_int_list(args.seeds)
    out_dir = repo / "results" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    run_rows: List[Dict[str, object]] = []
    if not args.skip_run:
        for d in datasets:
            for s in seeds:
                ver = f"{args.tag}_{d}_s{s}"
                rr = run_one(repo, d, s, ver, args)
                run_rows.append(rr)
                print(f"[run] dataset={d} seed={s} status={rr['status']} sec={rr['seconds']:.2f}")
    else:
        for d in datasets:
            for s in seeds:
                ver = f"{args.tag}_{d}_s{s}"
                rr = {
                    "dataset": d,
                    "seed": s,
                    "version": ver,
                    "status": "skip_run",
                    "seconds": 0.0,
                    "metrics_path": str(repo / "results" / ver / f"{d}_metrics.json"),
                    "log_path": str(repo / "results" / ver / f"{d}.log"),
                    "stderr_tail": "",
                }
                run_rows.append(rr)

    run_csv = out_dir / "runs.csv"
    write_csv(
        run_csv,
        run_rows,
        ["dataset", "seed", "version", "status", "seconds", "metrics_path", "log_path", "stderr_tail"],
    )

    per_run_rows: List[Dict[str, object]] = []
    for rr in run_rows:
        log_path = Path(str(rr["log_path"]))
        parsed = parse_run_log(log_path)
        loss_metrics = run_curve_metrics(parsed["losses"])

        nmi_final = float("nan")
        ari_final = float("nan")
        mp = Path(str(rr["metrics_path"]))
        if mp.exists():
            try:
                with open(mp, "r", encoding="utf-8") as f:
                    mj = json.load(f)
                nmi_final = float(mj.get("nmi_mean", float("nan")))
                ari_final = float(mj.get("ari_mean", float("nan")))
            except Exception:
                pass

        per_run_rows.append(
            {
                "dataset": rr["dataset"],
                "seed": rr["seed"],
                "version": rr["version"],
                "status": rr["status"],
                "seconds": rr["seconds"],
                "n_points": parsed["n_points"],
                "n_eval_points": parsed["n_eval_points"],
                "loss_start": loss_metrics["loss_start"],
                "loss_end": loss_metrics["loss_end"],
                "loss_rel_change_pct": loss_metrics["loss_rel_change_pct"],
                "tail_drift_pct": loss_metrics["tail_drift_pct"],
                "tail_band_pct": loss_metrics["tail_band_pct"],
                "tail_slope_per_epoch_pct": loss_metrics["tail_slope_per_epoch_pct"],
                "sign_flip_ratio": loss_metrics["sign_flip_ratio"],
                "graph_alpha_mean": mean(parsed["graph_alphas"]),
                "edge_rel_mean": mean(parsed["edge_rels"]),
                "eval_nmi_last": parsed["eval_nmi"][-1] if parsed["eval_nmi"] else float("nan"),
                "eval_ari_last": parsed["eval_ari"][-1] if parsed["eval_ari"] else float("nan"),
                "nmi_final": nmi_final,
                "ari_final": ari_final,
            }
        )

    per_run_csv = out_dir / "per_run_curve_metrics.csv"
    write_csv(
        per_run_csv,
        per_run_rows,
        [
            "dataset",
            "seed",
            "version",
            "status",
            "seconds",
            "n_points",
            "n_eval_points",
            "loss_start",
            "loss_end",
            "loss_rel_change_pct",
            "tail_drift_pct",
            "tail_band_pct",
            "tail_slope_per_epoch_pct",
            "sign_flip_ratio",
            "graph_alpha_mean",
            "edge_rel_mean",
            "eval_nmi_last",
            "eval_ari_last",
            "nmi_final",
            "ari_final",
        ],
    )

    grouped: Dict[str, List[Dict[str, object]]] = {}
    for r in per_run_rows:
        grouped.setdefault(str(r["dataset"]), []).append(r)

    dataset_rows: List[Dict[str, object]] = []
    for d in datasets:
        rows = grouped.get(d, [])
        ok_rows = [r for r in rows if str(r["status"]).startswith("ok") or str(r["status"]) == "skip_exists" or str(r["status"]) == "skip_run"]
        final_losses = [float(r["loss_end"]) for r in ok_rows if not math.isnan(float(r["loss_end"]))]
        item = {
            "dataset": d,
            "n_runs": len(rows),
            "n_runs_ok": len(ok_rows),
            "tail_drift_pct_mean": mean([float(r["tail_drift_pct"]) for r in ok_rows if not math.isnan(float(r["tail_drift_pct"]))]),
            "tail_drift_pct_std": std([float(r["tail_drift_pct"]) for r in ok_rows if not math.isnan(float(r["tail_drift_pct"]))]),
            "tail_band_pct_mean": mean([float(r["tail_band_pct"]) for r in ok_rows if not math.isnan(float(r["tail_band_pct"]))]),
            "tail_band_pct_std": std([float(r["tail_band_pct"]) for r in ok_rows if not math.isnan(float(r["tail_band_pct"]))]),
            "tail_slope_per_epoch_pct_mean": mean([abs(float(r["tail_slope_per_epoch_pct"])) for r in ok_rows if not math.isnan(float(r["tail_slope_per_epoch_pct"]))]),
            "sign_flip_ratio_mean": mean([float(r["sign_flip_ratio"]) for r in ok_rows if not math.isnan(float(r["sign_flip_ratio"]))]),
            "final_loss_mean": mean(final_losses),
            "final_loss_std": std(final_losses),
            "final_loss_cv_pct": 100.0 * std(final_losses) / (abs(mean(final_losses)) + 1e-12) if final_losses else float("nan"),
            "nmi_final_mean": mean([float(r["nmi_final"]) for r in ok_rows if not math.isnan(float(r["nmi_final"]))]),
            "nmi_final_std": std([float(r["nmi_final"]) for r in ok_rows if not math.isnan(float(r["nmi_final"]))]),
            "ari_final_mean": mean([float(r["ari_final"]) for r in ok_rows if not math.isnan(float(r["ari_final"]))]),
            "ari_final_std": std([float(r["ari_final"]) for r in ok_rows if not math.isnan(float(r["ari_final"]))]),
            "seconds_mean": mean([float(r["seconds"]) for r in ok_rows if not math.isnan(float(r["seconds"]))]),
        }
        item["stability"] = classify_dataset(item)
        dataset_rows.append(item)

    dataset_csv = out_dir / "per_dataset_stability.csv"
    write_csv(
        dataset_csv,
        dataset_rows,
        [
            "dataset",
            "n_runs",
            "n_runs_ok",
            "tail_drift_pct_mean",
            "tail_drift_pct_std",
            "tail_band_pct_mean",
            "tail_band_pct_std",
            "tail_slope_per_epoch_pct_mean",
            "sign_flip_ratio_mean",
            "final_loss_mean",
            "final_loss_std",
            "final_loss_cv_pct",
            "nmi_final_mean",
            "nmi_final_std",
            "ari_final_mean",
            "ari_final_std",
            "seconds_mean",
            "stability",
        ],
    )

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "tag": args.tag,
                "datasets": datasets,
                "seeds": seeds,
                "epochs": args.epochs,
                "eval_freq": args.eval_freq,
                "train_log_interval": args.train_log_interval,
                "run_csv": str(run_csv),
                "per_run_csv": str(per_run_csv),
                "per_dataset_csv": str(dataset_csv),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[ok] wrote {run_csv}")
    print(f"[ok] wrote {per_run_csv}")
    print(f"[ok] wrote {dataset_csv}")


if __name__ == "__main__":
    main()
