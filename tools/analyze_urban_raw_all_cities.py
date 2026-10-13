#!/usr/bin/env python3
"""Raw all-city diagnostics for Urban graph-construction exploration."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_urban_plot_graph_v3 as v3  # noqa: E402


LAND_USE = v3.LAND_USE_CANDIDATES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose raw Urban layers across cities.")
    parser.add_argument("--urban-root", type=Path, default=Path("/mnt/e/urban network datasets"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/data_construction_series/urban_all_city_exploration"),
    )
    parser.add_argument("--cities", type=str, default="all")
    return parser.parse_args()


def parse_city_list(text: str, urban_root: Path) -> list[str]:
    if str(text).strip().lower() == "all":
        return sorted(p.name for p in urban_root.iterdir() if p.is_dir())
    return [x.strip() for x in text.split(",") if x.strip()]


def safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def q(values: np.ndarray, prob: float) -> float:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values.astype(np.float64), prob))


def describe_numeric(prefix: str, values: Any) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_p99": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_p50": q(arr, 0.50),
        f"{prefix}_p90": q(arr, 0.90),
        f"{prefix}_p95": q(arr, 0.95),
        f"{prefix}_p99": q(arr, 0.99),
        f"{prefix}_max": float(np.max(arr)),
    }


def entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]
    if counts.size <= 1:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / math.log(len(p)))


def class_rows(city: str, y: np.ndarray, mapping: dict[str, int]) -> list[dict[str, Any]]:
    inv = {int(v): k for k, v in mapping.items()}
    total = len(y)
    counts = Counter(int(v) for v in y)
    rows = []
    for label_id, count in sorted(counts.items()):
        name = inv.get(int(label_id), str(label_id))
        rows.append(
            {
                "city": city,
                "label": name if name != "unknown" else "Unknown",
                "label_id": int(label_id),
                "count": int(count),
                "fraction_all": float(count / max(total, 1)),
            }
        )
    return rows


def membership_lengths(path: Path, n: int, zero_scalar_empty: bool = False) -> tuple[list[list[int]], np.ndarray]:
    if not path.exists() or n <= 0:
        return [[] for _ in range(n)], np.zeros(n, dtype=np.int32)
    memberships = v3._load_membership_array(path, n=n, zero_scalar_empty=zero_scalar_empty)
    lengths = np.asarray([len(x) for x in memberships], dtype=np.int32)
    return memberships, lengths


def inverse_counts(memberships: list[list[int]]) -> np.ndarray:
    counts: dict[int, int] = defaultdict(int)
    for ids in memberships:
        for x in ids:
            counts[int(x)] += 1
    return np.asarray(list(counts.values()), dtype=np.int32)


def junction_memberships(plot_street: list[list[int]], street_df: pd.DataFrame) -> list[list[int]]:
    endpoint_map = v3._street_endpoint_map(street_df)
    out = []
    for ids in plot_street:
        endpoints: set[int] = set()
        for sid in ids:
            endpoints.update(endpoint_map.get(int(sid), ()))
        out.append(sorted(endpoints))
    return out


def label_summary(plot_df: pd.DataFrame) -> tuple[np.ndarray, dict[str, int], dict[str, float]]:
    y, mapping = v3._build_labels(plot_df, "landuse")
    if mapping is None:
        mapping = {"unknown": 0}
    unknown = int(mapping.get("unknown", -1))
    known_mask = y != unknown
    known = y[known_mask]
    counts = np.unique(known, return_counts=True)[1] if known.size else np.asarray([], dtype=np.int64)
    summary = {
        "known_label_ratio": float(known_mask.mean()) if len(y) else float("nan"),
        "n_known_classes": int(np.unique(known).size) if known.size else 0,
        "known_label_entropy_norm": entropy(counts),
        "largest_known_class_ratio": float(counts.max() / counts.sum()) if counts.size and counts.sum() else float("nan"),
    }
    return y, mapping, summary


def analyze_city(city: str, urban_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    city_dir = urban_root / city
    plot_df = safe_read_parquet(city_dir / "plot.parquet")
    street_df = safe_read_parquet(city_dir / "street.parquet")
    building_df = safe_read_parquet(city_dir / "building.parquet")
    intersection_df = safe_read_parquet(city_dir / "intersection.parquet")

    n_plots = int(len(plot_df))
    plot_street, street_member_len = membership_lengths(city_dir / "plot_street_id.npy", n_plots)
    plot_building, building_member_len = membership_lengths(city_dir / "plot_bid.npy", n_plots, zero_scalar_empty=True)
    plot_junction = junction_memberships(plot_street, street_df)
    junction_member_len = np.asarray([len(x) for x in plot_junction], dtype=np.int32)

    y, mapping, label_stats = label_summary(plot_df)

    street_entity_load = inverse_counts(plot_street)
    junction_entity_load = inverse_counts(plot_junction)
    building_entity_load = inverse_counts(plot_building)

    row: dict[str, Any] = {
        "city": city,
        "n_plots": n_plots,
        "n_streets": int(len(street_df)),
        "n_buildings": int(len(building_df)),
        "n_intersections": int(len(intersection_df)),
        **label_stats,
        "plot_without_street_frac": float(np.mean(street_member_len == 0)) if n_plots else float("nan"),
        "plot_without_building_frac": float(np.mean(building_member_len == 0)) if n_plots else float("nan"),
    }

    row.update(describe_numeric("plot_street_membership", street_member_len))
    row.update(describe_numeric("plot_junction_membership", junction_member_len))
    row.update(describe_numeric("plot_building_membership", building_member_len))
    row.update(describe_numeric("street_entity_plot_load", street_entity_load))
    row.update(describe_numeric("junction_entity_plot_load", junction_entity_load))
    row.update(describe_numeric("building_entity_plot_load", building_entity_load))

    if "length" in street_df.columns:
        row.update(describe_numeric("street_length_m", pd.to_numeric(street_df["length"], errors="coerce").to_numpy()))
    if "plot_area" in plot_df.columns:
        row.update(describe_numeric("plot_area", pd.to_numeric(plot_df["plot_area"], errors="coerce").to_numpy()))
    if "bid_area" in building_df.columns:
        row.update(describe_numeric("building_area", pd.to_numeric(building_df["bid_area"], errors="coerce").to_numpy()))
    if "bid_height" in building_df.columns:
        h = pd.to_numeric(building_df["bid_height"], errors="coerce").to_numpy()
        row.update(describe_numeric("building_height", h))
        row["building_height_known_frac"] = float(np.mean(np.isfinite(h) & (h > 0))) if h.size else float("nan")
    if "degree" in intersection_df.columns:
        row.update(describe_numeric("intersection_degree", pd.to_numeric(intersection_df["degree"], errors="coerce").to_numpy()))
    if "plot_orientation" in plot_df.columns:
        orientation = pd.to_numeric(plot_df["plot_orientation"], errors="coerce").to_numpy()
        finite = orientation[np.isfinite(orientation)]
        if finite.size:
            hist, _ = np.histogram(np.mod(finite, 180.0), bins=np.linspace(0, 180, 19))
            row["plot_orientation_entropy_norm"] = entropy(hist)
        else:
            row["plot_orientation_entropy_norm"] = float("nan")

    label_rows = class_rows(city, y, mapping)
    tail_rows = []
    for metric, values in [
        ("plot_street_membership", street_member_len),
        ("plot_junction_membership", junction_member_len),
        ("plot_building_membership", building_member_len),
        ("street_entity_plot_load", street_entity_load),
        ("junction_entity_plot_load", junction_entity_load),
        ("building_entity_plot_load", building_entity_load),
    ]:
        stats = describe_numeric(metric, values)
        tail_rows.append(
            {
                "city": city,
                "metric": metric,
                "mean": stats[f"{metric}_mean"],
                "p95": stats[f"{metric}_p95"],
                "p99": stats[f"{metric}_p99"],
                "max": stats[f"{metric}_max"],
            }
        )

    return row, label_rows, tail_rows


def main() -> None:
    args = parse_args()
    cities = parse_city_list(args.cities, args.urban_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    labels = []
    tails = []
    for city in cities:
        print(f"[raw] {city}", flush=True)
        row, label_rows, tail_rows = analyze_city(city, args.urban_root)
        rows.append(row)
        labels.extend(label_rows)
        tails.extend(tail_rows)

    city_df = pd.DataFrame(rows).sort_values("city")
    label_df = pd.DataFrame(labels).sort_values(["city", "label"])
    tail_df = pd.DataFrame(tails).sort_values(["metric", "city"])

    city_df.to_csv(args.out_dir / "raw_city_diagnostics.csv", index=False)
    label_df.to_csv(args.out_dir / "raw_label_distribution.csv", index=False)
    tail_df.to_csv(args.out_dir / "raw_membership_tail_summary.csv", index=False)

    (args.out_dir / "raw_city_diagnostics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    readme = [
        "# Urban All-City Raw Data Exploration",
        "",
        "This folder diagnoses the raw Urban layers before graph construction.",
        "",
        "Files:",
        "- `raw_city_diagnostics.csv`: per-city raw layer size, label coverage, label entropy, morphology and membership-tail statistics.",
        "- `raw_label_distribution.csv`: per-city land-use label distribution derived from raw plot POI score columns.",
        "- `raw_membership_tail_summary.csv`: compact long-format table for plot/entity membership tails.",
        "",
        "Use this before deciding which cities should receive expensive V4 graph construction and GNN ablation.",
    ]
    (args.out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"[ok] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
