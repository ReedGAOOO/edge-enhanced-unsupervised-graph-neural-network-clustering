#!/usr/bin/env python3
"""
Batch-build urban plot hetero v2 datasets and aggregate validation summaries.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _discover_cities(urban_root: Path) -> list[str]:
    return sorted([p.name for p in urban_root.iterdir() if p.is_dir()])


def _parse_csv_arg(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _run_checked(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _building_count(city_dir: Path) -> int:
    return int(pd.read_parquet(city_dir / "building.parquet", columns=["bid"]).shape[0])


def _adaptive_relation_config(city_dir: Path) -> dict:
    n_building = _building_count(city_dir)
    if n_building > 300_000:
        return {
            "building_spatial_knn_k": 1,
            "building_semantic_knn_k": 0,
            "disable_building_semantic": True,
            "proxy_building_centroid_from_plot": True,
            "mode": "ultra_large",
            "n_building": n_building,
        }
    if n_building > 150_000:
        return {
            "building_spatial_knn_k": 1,
            "building_semantic_knn_k": 1,
            "disable_building_semantic": False,
            "proxy_building_centroid_from_plot": True,
            "mode": "large",
            "n_building": n_building,
        }
    if n_building > 100_000:
        return {
            "building_spatial_knn_k": 2,
            "building_semantic_knn_k": 2,
            "disable_building_semantic": False,
            "proxy_building_centroid_from_plot": True,
            "mode": "medium",
            "n_building": n_building,
        }
    if n_building > 50_000:
        return {
            "building_spatial_knn_k": 3,
            "building_semantic_knn_k": 3,
            "disable_building_semantic": False,
            "proxy_building_centroid_from_plot": False,
            "mode": "small_medium",
            "n_building": n_building,
        }
    return {
        "building_spatial_knn_k": 5,
        "building_semantic_knn_k": 5,
        "disable_building_semantic": False,
        "proxy_building_centroid_from_plot": False,
        "mode": "default",
        "n_building": n_building,
    }


def _build_one(
    builder: Path,
    city: str,
    city_dir: Path,
    urban_root: Path,
    out_root: Path,
    force: bool,
    standardize_node_features: bool,
) -> tuple[str, dict]:
    rel_cfg = _adaptive_relation_config(city_dir)
    cmd = [
        sys.executable,
        str(builder),
        "--city",
        city,
        "--urban_root",
        str(urban_root),
        "--out_root",
        str(out_root),
        "--building_spatial_knn_k",
        str(rel_cfg["building_spatial_knn_k"]),
        "--building_semantic_knn_k",
        str(rel_cfg["building_semantic_knn_k"]),
    ]
    if force:
        cmd.append("--force")
    if bool(rel_cfg["disable_building_semantic"]):
        cmd.append("--disable_building_semantic")
    if bool(rel_cfg["proxy_building_centroid_from_plot"]):
        cmd.append("--proxy_building_centroid_from_plot")
    if standardize_node_features:
        cmd.append("--standardize_node_features")
    _run_checked(cmd)
    return f"urban_{city}_plot_hetero_v2", rel_cfg


def _validate_one(validator: Path, dataset_dir: Path, out_json: Path) -> dict:
    cmd = [
        sys.executable,
        str(validator),
        "--dataset_dir",
        str(dataset_dir),
        "--out_json",
        str(out_json),
    ]
    _run_checked(cmd)
    return json.loads(out_json.read_text(encoding="utf-8"))


def _meta_row(meta_path: Path, validate: dict, rel_cfg: dict) -> dict:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rel_stats = meta.get("relation_stats", {})
    build_street = rel_stats.get("building_street_selection", {})
    build_sem = rel_stats.get("building_semantic", {})
    plot_build = rel_stats.get("plot_building_reference_mode", {})
    return {
        "city": meta.get("city", ""),
        "dataset_name": meta.get("dataset_name", ""),
        "dataset_family": meta.get("dataset_family", ""),
        "n_plot": validate.get("node_counts", {}).get("plot", 0),
        "n_street": validate.get("node_counts", {}).get("street", 0),
        "n_intersection": validate.get("node_counts", {}).get("intersection", 0),
        "n_building": validate.get("node_counts", {}).get("building", 0),
        "plot_plot_edges": validate.get("edge_counts", {}).get("plot__shares_street__plot", 0),
        "plot_street_edges": validate.get("edge_counts", {}).get("plot__bounded_by__street", 0),
        "street_intersection_edges": validate.get("edge_counts", {}).get("street__connects__intersection", 0),
        "plot_building_edges": validate.get("edge_counts", {}).get("plot__contains__building", 0),
        "building_street_edges": validate.get("edge_counts", {}).get("building__proximal_to__street", 0),
        "building_spatial_edges": validate.get("edge_counts", {}).get("building__near__building", 0),
        "building_semantic_edges": validate.get("edge_counts", {}).get("building__semantic__building", 0),
        "largest_plot_component_ratio": validate.get("plot_component_stats", {}).get("largest_component_ratio", 0.0),
        "plot_building_ref_mode": plot_build.get("mode", ""),
        "building_street_selected_ratio": build_street.get("selected_ratio", 0.0),
        "building_semantic_enabled": build_sem.get("enabled", False),
        "building_semantic_sigma": build_sem.get("sigma", 0.0),
        "adaptive_mode": rel_cfg.get("mode", ""),
        "adaptive_n_building": rel_cfg.get("n_building", 0),
        "adaptive_spatial_k": rel_cfg.get("building_spatial_knn_k", 0),
        "adaptive_semantic_k": rel_cfg.get("building_semantic_knn_k", 0),
        "adaptive_semantic_disabled": rel_cfg.get("disable_building_semantic", False),
        "adaptive_building_centroid_proxy": rel_cfg.get("proxy_building_centroid_from_plot", False),
        "status": validate.get("status", ""),
    }


def _write_summary(summary_dir: Path, rows: list[dict]) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["city"]).reset_index(drop=True)
    df.to_csv(summary_dir / "urban_plot_hetero_v2_summary.csv", index=False)
    (summary_dir / "urban_plot_hetero_v2_summary.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )

    cities = sorted(df["city"].unique().tolist()) if not df.empty else []
    readme = "\n".join(
        [
            "# Urban Plot Hetero V2 Series",
            "",
            f"- cities: {len(cities)}",
            f"- datasets: {len(df)}",
            "",
            "Files:",
            "- `urban_plot_hetero_v2_summary.csv`: flat manifest across all generated city datasets.",
            "- `urban_plot_hetero_v2_summary.json`: JSON version of the same manifest.",
            "- `validate_<city>.json`: per-city structural validation summary.",
            "",
            "Design note:",
            "- `hetero_v2` preserves `plot/street/intersection/building` nodes.",
            "- It upgrades each relation with typed edge attributes.",
            "- It keeps spatial and semantic building-building relations separate.",
        ]
    )
    (summary_dir / "README.md").write_text(readme + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-build urban plot hetero v2 datasets.")
    parser.add_argument("--urban_root", type=str, default="data/urban_network_datasets")
    parser.add_argument("--out_root", type=str, default="data")
    parser.add_argument("--summary_dir", type=str, default="results/data_construction_series/urban_plot_hetero_v2_series")
    parser.add_argument("--cities", type=str, default="all", help='Comma-separated list, or "all"')
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--standardize_node_features", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    urban_root = (repo_root / args.urban_root).resolve()
    out_root = (repo_root / args.out_root).resolve()
    summary_dir = (repo_root / args.summary_dir).resolve()
    builder = (repo_root / "tools" / "prepare_urban_plot_hetero_v2.py").resolve()
    validator = (repo_root / "tools" / "validate_urban_plot_hetero_v1.py").resolve()

    if not urban_root.exists():
        raise FileNotFoundError(f"Urban root not found: {urban_root}")
    if not builder.exists():
        raise FileNotFoundError(f"Builder script not found: {builder}")
    if not validator.exists():
        raise FileNotFoundError(f"Validator script not found: {validator}")

    cities = _discover_cities(urban_root) if str(args.cities).strip().lower() == "all" else _parse_csv_arg(args.cities)
    if not cities:
        raise ValueError("No cities selected.")

    summary_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for idx, city in enumerate(cities, start=1):
        print(f"[{idx}/{len(cities)}] building {city}", flush=True)
        city_dir = urban_root / city
        dataset_name, rel_cfg = _build_one(
            builder=builder,
            city=city,
            city_dir=city_dir,
            urban_root=urban_root,
            out_root=out_root,
            force=bool(args.force),
            standardize_node_features=bool(args.standardize_node_features),
        )
        dataset_dir = out_root / dataset_name
        meta_path = dataset_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Expected meta file not found: {meta_path}")
        validate_json = summary_dir / f"validate_{city}.json"
        validate = _validate_one(validator=validator, dataset_dir=dataset_dir, out_json=validate_json)
        rows.append(_meta_row(meta_path=meta_path, validate=validate, rel_cfg=rel_cfg))

    _write_summary(summary_dir=summary_dir, rows=rows)
    print(f"[ok] generated {len(rows)} hetero_v2 datasets", flush=True)
    print(f" summary={summary_dir}", flush=True)


if __name__ == "__main__":
    main()
