#!/usr/bin/env python3
"""Batch runner for the breakthrough experiment plans in configs/breakthrough/.

Each plan expands (tag x dataset x seed) into main.py runs, executes them in parallel on one GPU,
and writes runs.csv / summary.csv under results/breakthrough_v1/<plan>/.

Besides the honest metrics (model selected by min train loss), the runner parses the periodic
evaluation lines from the log and reports `oracle_best_nmi`: the best NMI over evaluation epochs.
That number uses labels for model selection and is only kept to compare with the reference
DSE/LSEnet code, which reports it as the headline metric.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))
from run_preset import DATASET_MAX_NUMS, load_preset  # noqa: E402

EVAL_RE = re.compile(r"Epoch (\d+): ACC:\s*([-\d.]+), NMI:\s*([-\d.]+), ARI:\s*([-\d.]+)")
METRIC_KEYS = [
    "nmi_mean", "ari_mean", "acc_mean", "modularity_mean", "conductance_weighted_mean",
    "pred_n_clusters_mean", "best_epoch_mean", "best_train_loss_mean", "si_loss_mean",
    "warm_chosen", "warm_pseudo_nmi",
]


def preset_to_cli(preset: dict) -> list:
    cli = []
    for key, val in preset.items():
        if isinstance(val, bool):
            if val:
                cli.append(f"--{key}")
        elif val is not None:
            cli += [f"--{key}", str(val)]
    return cli


def expand_jobs(plan: dict) -> list:
    jobs = []
    for spec in plan["jobs"]:
        for dataset in spec["datasets"]:
            for seed in spec.get("seeds", plan.get("seeds", [0])):
                jobs.append({
                    "tag": spec["tag"],
                    "preset": spec.get("preset", "baseline_v1"),
                    "workspace": spec.get("workspace", ""),
                    "dataset": dataset,
                    "seed": int(seed),
                    "overrides": {**plan.get("defaults", {}), **spec.get("overrides", {})},
                })
    return jobs


def max_nums_for(dataset: str, plan: dict) -> int:
    per_ds = plan.get("max_nums", {})
    if dataset in per_ds:
        return int(per_ds[dataset])
    key = dataset.lower()
    if key in DATASET_MAX_NUMS:
        return DATASET_MAX_NUMS[key]
    return int(plan.get("max_nums_default", 12))


def resolve_preset(name_or_path: str) -> dict:
    candidate = REPO_ROOT / name_or_path
    return load_preset(REPO_ROOT, str(candidate) if candidate.exists() else name_or_path)


def build_cmd(job: dict, plan: dict, args, version: str) -> list:
    preset = resolve_preset(job["preset"])
    overrides = dict(job["overrides"])
    overrides.setdefault("max_nums", max_nums_for(job["dataset"], plan))
    cmd = [
        sys.executable, "main.py",
        "--dataset", job["dataset"],
        "--root_path", args.root_path,
        "--seed", str(job["seed"]),
        "--gpu", str(args.gpu),
        "--version", version,
        "--save_path", f"{version}.pt",
    ]
    # Later flags win in argparse, so plan overrides take precedence over the preset.
    cmd += preset_to_cli(preset)
    cmd += preset_to_cli(overrides)
    return cmd


def parse_run(results_dir: Path, dataset: str) -> dict:
    out = {}
    metrics_path = results_dir / f"{dataset}_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        for k in METRIC_KEYS:
            out[k] = metrics.get(k, float("nan"))
    log_path = results_dir / f"{dataset}.log"
    if log_path.exists():
        evals = [(int(m.group(1)), float(m.group(3)) / 100.0) for m in EVAL_RE.finditer(log_path.read_text())]
        if evals:
            best_epoch, best_nmi = max(evals, key=lambda t: t[1])
            out["oracle_best_nmi"] = best_nmi
            out["oracle_best_epoch"] = best_epoch
            out["last_eval_nmi"] = evals[-1][1]
            out["eval_trajectory"] = ";".join(f"{e}:{n:.3f}" for e, n in evals)
    return out


def run_job(job: dict, plan: dict, args, out_root: Path) -> dict:
    run_id = f"{job['tag']}__{job['dataset']}__s{job['seed']}"
    if job["workspace"]:
        # Expert code copies only accept a flat version name under their own ./results.
        cwd = REPO_ROOT / job["workspace"]
        version = f"bt_{plan['name']}__{run_id}"
    else:
        cwd = REPO_ROOT
        version = f"breakthrough_v1/{plan['name']}/runs/{run_id}"
    results_dir = cwd / "results" / version
    row = {"run_id": run_id, "tag": job["tag"], "preset": job["preset"], "dataset": job["dataset"], "seed": job["seed"],
           "results_dir": str(results_dir.relative_to(REPO_ROOT))}
    if (results_dir / f"{job['dataset']}_metrics.json").exists() and not args.force:
        row.update(parse_run(results_dir, job["dataset"]))
        row["status"] = "cached"
        return row
    cmd = build_cmd(job, plan, args, version)
    if args.dry_run:
        print(" ".join(cmd))
        row["status"] = "dry_run"
        return row
    results_dir.mkdir(parents=True, exist_ok=True)
    # The logger appends, so a log left by an interrupted attempt would mix two trajectories.
    for stale in results_dir.glob("*.log"):
        stale.unlink()
    # Without a cap every job spawns one OpenMP/BLAS thread per core and parallel jobs thrash the CPU.
    env = {**os.environ, **{k: str(args.threads) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    # Under WSL an over-committed GPU pages to host memory instead of raising OOM, so keep fragmentation low.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    t0 = time.time()
    with open(results_dir / "stdout.txt", "w") as fh:
        fh.write("CMD: " + " ".join(cmd) + "\n")
        fh.flush()
        rc = subprocess.run(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    row["elapsed_sec"] = round(time.time() - t0, 1)
    row["status"] = "ok" if rc == 0 else f"failed:{rc}"
    row.update(parse_run(results_dir, job["dataset"]))
    return row


def summarize(rows: list, out_root: Path) -> None:
    groups = {}
    for r in rows:
        if r.get("status") in ("ok", "cached") and "nmi_mean" in r:
            groups.setdefault((r["tag"], r["dataset"]), []).append(r)
    keys = ["nmi_mean", "ari_mean", "modularity_mean", "conductance_weighted_mean", "oracle_best_nmi",
            "best_epoch_mean", "pred_n_clusters_mean", "warm_pseudo_nmi"]
    with open(out_root / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "dataset", "n"] + [f"{k}_{s}" for k in keys for s in ("mean", "std")])
        for (tag, ds), rs in sorted(groups.items()):
            vals = []
            for k in keys:
                arr = np.array([float(r.get(k, np.nan)) for r in rs], dtype=float)
                vals += [f"{np.nanmean(arr):.4f}", f"{np.nanstd(arr):.4f}"]
            w.writerow([tag, ds, len(rs)] + vals)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", required=True, help="Path to a plan JSON under configs/breakthrough/.")
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_parallel", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2, help="CPU threads per job.")
    parser.add_argument("--only_tags", type=str, default="", help="Comma-separated subset of tags to run.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text())
    jobs = expand_jobs(plan)
    if args.only_tags:
        keep = set(args.only_tags.split(","))
        jobs = [j for j in jobs if j["tag"] in keep]
    out_root = REPO_ROOT / "results" / "breakthrough_v1" / plan["name"]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "plan.json").write_text(json.dumps(plan, indent=2))
    print(f"[plan] {plan['name']}: {len(jobs)} jobs -> {out_root}", flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as pool:
        futures = {pool.submit(run_job, j, plan, args, out_root): j for j in jobs}
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            print(f"[{len(rows)}/{len(jobs)}] {row['run_id']} {row['status']} "
                  f"nmi={row.get('nmi_mean', float('nan')):.4f} oracle={row.get('oracle_best_nmi', float('nan')):.4f} "
                  f"mod={row.get('modularity_mean', float('nan')):.4f} t={row.get('elapsed_sec', 0)}s", flush=True)

    if args.dry_run:
        return
    fields = ["run_id", "tag", "preset", "dataset", "seed", "status", "elapsed_sec"] + METRIC_KEYS + [
        "oracle_best_nmi", "oracle_best_epoch", "last_eval_nmi", "eval_trajectory", "results_dir"]
    rows.sort(key=lambda r: r["run_id"])
    with open(out_root / "runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    summarize(rows, out_root)
    print(f"[done] {out_root / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
