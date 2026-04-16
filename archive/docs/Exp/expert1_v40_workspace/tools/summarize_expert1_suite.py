#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


SUMMARY_FIELDS = [
    "acc_mean",
    "nmi_mean",
    "ari_mean",
    "si_loss_mean",
    "modularity_mean",
    "conductance_mean",
    "conductance_weighted_mean",
    "pred_n_clusters_mean",
    "pred_cluster_size_cv_mean",
    "best_epoch_mean",
    "elapsed_sec",
]


def mean(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def stdev(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if len(vals) < 2:
        return 0.0 if vals else None
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def load_rows(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            clean = {}
            for k, v in row.items():
                if v in ("", "None", "null"):
                    clean[k] = None
                    continue
                if k in {"preset", "dataset", "version", "status", "metrics_path"}:
                    clean[k] = v
                    continue
                try:
                    clean[k] = int(v)
                    continue
                except ValueError:
                    pass
                try:
                    clean[k] = float(v)
                    continue
                except ValueError:
                    clean[k] = v
            rows.append(clean)
        return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize expert1 suite results")
    parser.add_argument("--runs_csv", required=True, type=str)
    args = parser.parse_args()

    runs_csv = Path(args.runs_csv).resolve()
    out_dir = runs_csv.parent
    rows = load_rows(runs_csv)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["preset"])].append(row)

    summary_rows = []
    for (dataset, preset), group in sorted(grouped.items()):
        out = {
            "dataset": dataset,
            "preset": preset,
            "n_runs": len(group),
        }
        for field in SUMMARY_FIELDS:
            vals = [row.get(field) for row in group]
            out[f"{field}_avg"] = mean(vals)
            out[f"{field}_std"] = stdev(vals)
        summary_rows.append(out)

    csv_path = out_dir / "suite_summary.csv"
    json_path = out_dir / "suite_summary.json"
    md_path = out_dir / "suite_summary.md"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    lines = [
        "# Expert1 Suite Summary",
        "",
        "| Dataset | Preset | Runs | ACC | NMI | ARI | SI-loss | Modularity | Conductance | Weighted Conductance | Time(s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {dataset} | {preset} | {n_runs} | {acc:.4f} | {nmi:.4f} | {ari:.4f} | {si:.4f} | {mod:.4f} | {cond:.4f} | {wcond:.4f} | {time:.1f} |".format(
                dataset=row["dataset"],
                preset=row["preset"],
                n_runs=row["n_runs"],
                acc=row["acc_mean_avg"] or float("nan"),
                nmi=row["nmi_mean_avg"] or float("nan"),
                ari=row["ari_mean_avg"] or float("nan"),
                si=row["si_loss_mean_avg"] or float("nan"),
                mod=row["modularity_mean_avg"] or float("nan"),
                cond=row["conductance_mean_avg"] or float("nan"),
                wcond=row["conductance_weighted_mean_avg"] or float("nan"),
                time=row["elapsed_sec_avg"] or float("nan"),
            )
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {json_path}")
    print(f"[done] wrote {md_path}")


if __name__ == "__main__":
    main()
