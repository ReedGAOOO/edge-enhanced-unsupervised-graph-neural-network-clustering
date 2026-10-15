#!/usr/bin/env python3
"""E19: validate the label-free selection rules (tools/select_partition_rules.py, frozen at commit 876486b) on decisions
that were not used to design them.

New decisions: model choice on Citeseer, Amazon Computers and PubMed (baseline, B31, B31+ctx, B45 safe gate + ctx,
regularized spectral, SGC), decoder choice on their SE-searched graphs, and decoder choice on synthetic graphs drawn
with data seed 1 (first-order and XOR contexts). Seed 0, one 20% edge hold-out per graph. Steps, each resumable:
synthetic data, SE-guided search, reweighted datasets, edge-CV splits, stage 1 (baseline warm/cold starts, synthetic
runs), stage 2 (other models and CV runs at the baseline-chosen start), decision spec, rule scoring.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / ".venv/bin/python")
OUT = REPO / "results/breakthrough_v1/e19_selection_validation"
REAL = {"citeseer": 9, "computers": 12, "pubmed": 5}
SYNTH = {"synthctx_h45_ds01": "synth_ctx_v1_h45_contextual_ds01", "synthctx_h55_ds01": "synth_ctx_v1_h55_contextual_ds01",
         "synthxor_h45_ds01": "synth_ctxxor_v1_h45_contextual_ds01", "synthxor_h55_ds01": "synth_ctxxor_v1_h55_contextual_ds01"}
DEFAULTS = {"epochs": 1000, "hid_dim": 512, "w_decay": 0.0, "temperature": 1.0, "dropout": 0.1, "eval_freq": 100,
            "train_log_interval": 200, "n_cluster_trials": 3, "patience": 0, "exp_iters": 1}
WARM = {"warm_start": "auto", "warm_start_epochs": 200}
COLD = {"warm_start": "none"}
MODELS = {"baseline": ("baseline_v1", {}), "b31": ("b31_dualscalar_assign", {}),
          "b31ctx": ("b31_dualscalar_assign", {"edge_ctx_dim": 32}),
          "b45nrctx": ("b45_v31_msgcond_gs050", {"edge_msg_gate_renorm": False, "edge_ctx_dim": 32})}


def run(cmd, cwd=REPO, env_threads=None):
    env = dict(os.environ)
    if env_threads:
        env.update({k: str(env_threads) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def summary_pick(path, dataset, scope):
    for r in csv.DictReader(open(path)):
        if r["dataset"] == dataset:
            return r[f"{scope}_pick"].split("/")[0]
    raise KeyError(dataset)


def start_of(runs_csv, dataset):
    rows = {r["tag"].rsplit("_", 1)[1]: r for r in csv.DictReader(open(runs_csv)) if r["dataset"] == dataset}
    return min(rows, key=lambda s: float(rows[s]["best_train_loss_mean"]))


def write_plan(name, description, jobs, max_nums):
    path = REPO / "configs/breakthrough" / f"{name}.json"
    path.write_text(json.dumps({"name": name, "description": description, "seeds": [0], "max_nums_default": 6,
                                "max_nums": max_nums, "defaults": DEFAULTS, "jobs": jobs}, indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root_path", default="data")
    parser.add_argument("--max_parallel", type=int, default=4)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    root = REPO / args.root_path
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. synthetic graphs with data seed 1
    for prefix, extra in (("synth_ctx_v1", []), ("synth_ctxxor_v1", ["--role_mode", "xor2"])):
        if not (root / f"{prefix}_h55_contextual_ds01").exists():
            run([PY, "prepare_synth_ctx_datasets.py", "--root_path", str(root), "--prefix", prefix, *extra,
                 "--feature_sep", "0.35", "--homophily", "0.45,0.55", "--modes", "contextual", "--perm_modes", "",
                 "--data_seeds", "1"], cwd=REPO / "tools")

    # 2. SE-guided search (regularized spectral; exhaustive second order for the XOR graphs)
    searches = {"real": (list(REAL), []), "synth": ([SYNTH[k] for k in SYNTH if k.startswith("synthctx")], []),
                "xor": ([SYNTH[k] for k in SYNTH if k.startswith("synthxor")], ["--order2"])}
    for tag, (dss, extra) in searches.items():
        if not (OUT / f"search_{tag}" / "summary.csv").exists():
            run([PY, "tools/run_se_guided_baseline.py", "--root_path", args.root_path, "--datasets", ",".join(dss),
                 "--seeds", "0", "--spectral", "reg", "--workers", str(args.workers), *extra,
                 "--out_dir", str(OUT / f"search_{tag}")])

    # 3. reweighted datasets from the picks
    picks = {}
    for ds in REAL:
        picks[f"{ds}_sepick"] = (ds, summary_pick(OUT / "search_real/summary.csv", ds, "ctx"), 32)
    for short, src in SYNTH.items():
        tag, scope = ("xor", "ctx2") if short.startswith("synthxor") else ("synth", "ctx")
        picks[f"{short}_sepick"] = (src, summary_pick(OUT / f"search_{tag}/summary.csv", src, scope), 64)
    for out_name, (src, pick, ctx_dim) in picks.items():
        if not (root / out_name).exists():
            run([PY, "tools/write_se_reweighted_dataset.py", "--root_path", args.root_path, "--dataset", src,
                 f"--pick={pick}", "--ctx_dim", str(ctx_dim), "--out_name", out_name])
    (OUT / "picks.json").write_text(json.dumps(picks, indent=2))

    # 4. edge-CV splits
    split = [d for d in list(REAL) + list(picks) if not (root / f"{d}_cv80").exists()]
    if split:
        run([PY, "tools/make_edge_cv_split.py", "--root_path", args.root_path, "--frac", "0.2", "--seed", "0",
             "--suffix", "_cv80", "--datasets", ",".join(split)], env_threads=8)

    max_nums = {}
    for ds, k in REAL.items():
        for suffix in ("", "_cv80", "_sepick", "_sepick_cv80"):
            max_nums[f"{ds}{suffix}"] = k
    runner = [PY, "tools/run_breakthrough.py", "--root_path", args.root_path, "--max_parallel", str(args.max_parallel),
              "--threads", str(args.threads)]

    # 5. stage 1: baseline starts on real graphs and searched graphs; synthetic searched graphs (warm) and their CV runs
    real_and_sep = list(REAL) + [f"{d}_sepick" for d in REAL]
    synth_sep = [f"{s}_sepick" for s in SYNTH]
    jobs = [{"tag": "baseline_warm", "preset": "baseline_v1", "datasets": real_and_sep, "overrides": WARM},
            {"tag": "baseline_nowarm", "preset": "baseline_v1", "datasets": real_and_sep, "overrides": COLD},
            {"tag": "baseline_synth", "preset": "baseline_v1", "datasets": synth_sep + [f"{d}_cv80" for d in synth_sep],
             "overrides": {**WARM, "epochs": 1500}}]
    p1 = write_plan("e19a_selection_validation", "E19 stage 1: edge-free baseline warm/cold starts on Citeseer, Computers, "
                    "PubMed and their SE-searched graphs; synthetic seed-1 searched graphs and their edge-CV graphs (warm).",
                    jobs, max_nums)
    run([*runner, "--plan", str(p1)])

    # 6. stage 2: other models and CV runs at the baseline-chosen start
    r1 = REPO / "results/breakthrough_v1/e19a_selection_validation/runs.csv"
    starts = {d: start_of(r1, d) for d in real_and_sep}
    (OUT / "starts.json").write_text(json.dumps(starts, indent=2))
    jobs = []
    for ds in REAL:
        st = starts[ds]
        for model, (preset, ov) in MODELS.items():
            dsets = [f"{ds}_cv80"] if model == "baseline" else [ds, f"{ds}_cv80"]
            jobs.append({"tag": f"{model}_{st}", "preset": preset, "datasets": dsets, "overrides": {**ov, **(WARM if st == "warm" else COLD)}})
        sst = starts[f"{ds}_sepick"]
        jobs.append({"tag": f"baseline_{sst}", "preset": "baseline_v1", "datasets": [f"{ds}_sepick_cv80"],
                     "overrides": WARM if sst == "warm" else COLD})
    p2 = write_plan("e19b_selection_validation", "E19 stage 2: B31, B31+ctx, B45 safe gate + ctx on the real graphs and all "
                    "edge-CV runs, each at the start the edge-free baseline chose in stage 1.", jobs, max_nums)
    run([*runner, "--plan", str(p2)])

    # 7. decisions and rule scoring
    r2 = "results/breakthrough_v1/e19b_selection_validation/runs"
    r1d = "results/breakthrough_v1/e19a_selection_validation/runs"
    insts = []
    for ds in REAL:
        st = starts[ds]
        cands = []
        for model in MODELS:
            full = f"{r1d}/baseline_{st}__{ds}__s0" if model == "baseline" else f"{r2}/{model}_{st}__{ds}__s0"
            cands.append({"name": model, "family": model, "kind": "model", "run": full, "cv_run": f"{r2}/{model}_{st}__{ds}_cv80__s0"})
        cands += [{"name": "spectral@kcls", "family": "spectral", "kind": "spectral", "k": "kcls"},
                  {"name": "sgc@kcls", "family": "sgc", "kind": "sgc", "k": "kcls"}]
        insts.append({"id": f"model/{ds}/s0", "set": "model", "dataset": ds, "seed": 0, "cv_dataset": f"{ds}_cv80", "candidates": cands})
    decoder_tail = [{"name": "spectral@kdse", "family": "spectral", "kind": "spectral", "k": "kdse"},
                    {"name": "sgc@kdse", "family": "sgc", "kind": "sgc", "k": "kdse"},
                    {"name": "spectral@kcls", "family": "spectral", "kind": "spectral", "k": "kcls"},
                    {"name": "sgc@kcls", "family": "sgc", "kind": "sgc", "k": "kcls"}]
    for ds in REAL:
        sep, sst = f"{ds}_sepick", starts[f"{ds}_sepick"]
        dse = {"name": "dse", "family": "dse", "kind": "model", "run": f"{r1d}/baseline_{sst}__{sep}__s0",
               "cv_run": f"{r2}/baseline_{sst}__{sep}_cv80__s0"}
        insts.append({"id": f"decoder/{sep}/s0", "set": "decoder", "dataset": sep, "seed": 0, "cv_dataset": f"{sep}_cv80",
                      "candidates": [dse] + decoder_tail})
    for sep in synth_sep:
        dse = {"name": "dse", "family": "dse", "kind": "model", "run": f"{r1d}/baseline_synth__{sep}__s0",
               "cv_run": f"{r1d}/baseline_synth__{sep}_cv80__s0"}
        insts.append({"id": f"decoder/{sep}/s0", "set": "decoder", "dataset": sep, "seed": 0, "cv_dataset": f"{sep}_cv80",
                      "candidates": [dse] + decoder_tail})
    spec = REPO / "configs/breakthrough/selection_validation_spec.json"
    spec.write_text(json.dumps({"description": "E19 validation decisions, unseen when the rules were designed.",
                                "instances": insts}, indent=1) + "\n")
    run([PY, "tools/select_partition_rules.py", "--root_path", args.root_path, "--spec", str(spec),
         "--out_dir", str(OUT / "rules")], env_threads=8)
    print("e19 done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
