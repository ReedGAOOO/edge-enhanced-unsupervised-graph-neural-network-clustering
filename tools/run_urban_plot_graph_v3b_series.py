#!/usr/bin/env python3
"""
Batch-build urban plot graph V3b series and aggregate metadata summaries.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_VARIANTS = ["v3bs", "v3bsj", "v3bsjg"]


def _discover_cities(urban_root: Path) -> list[str]:
    return sorted([p.name for p in urban_root.iterdir() if p.is_dir()])


def _parse_csv_arg(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _build_one(
    builder: Path,
    city: str,
    variant: str,
    urban_root: Path,
    out_root: Path,
    force: bool,
) -> str:
    cmd = [
        sys.executable,
        str(builder),
        "--city",
        city,
        "--urban_root",
        str(urban_root),
        "--out_root",
        str(out_root),
        "--variant",
        variant,
    ]
    if force:
        cmd.append("--force")
    subprocess.run(cmd, check=True)
    return f"urban_{city}_plot_{variant}"


def _meta_row(meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rel = meta.get("relation_counts_undirected", {})
    caps = meta.get("adaptive_caps", {})
    geom = meta.get("geom_fallback", {})
    schema = meta.get("schema_mapping", {})
    return {
        "city": meta.get("city", ""),
        "dataset_name": meta.get("dataset_name", ""),
        "variant": meta.get("variant", ""),
        "edge_schema": meta.get("edge_schema", ""),
        "n_nodes": meta.get("n_nodes", 0),
        "n_features": meta.get("n_features", 0),
        "n_edges_undirected": meta.get("n_edges_undirected", 0),
        "n_edges_directed": meta.get("n_edges_directed", 0),
        "edge_feature_dim": meta.get("edge_feature_dim", 0),
        "street_edges": rel.get("street", 0),
        "junction_edges": rel.get("junction", 0),
        "geom_edges": rel.get("geom", 0),
        "street_entity_cap": caps.get("street_entity_cap", 0),
        "junction_entity_cap": caps.get("junction_entity_cap", 0),
        "building_attr_entity_cap": caps.get("building_attr_entity_cap", 0),
        "geom_radius_m": geom.get("geom_radius_m", 0.0),
        "geom_k": geom.get("geom_k", 0),
        "geom_max_struct_deg": geom.get("geom_max_struct_deg", 0),
        "population_schema": schema.get("population", ""),
        "built_schema": schema.get("built", ""),
        "canopy_mean_schema": schema.get("canopy_mean", ""),
        "canopy_std_schema": schema.get("canopy_std", ""),
    }


def _write_summary(summary_dir: Path, rows: list[dict]) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["city", "variant"]).reset_index(drop=True)
    df.to_csv(summary_dir / "urban_plot_v3b_summary.csv", index=False)
    (summary_dir / "urban_plot_v3b_summary.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )

    variants = sorted(df["variant"].unique().tolist()) if not df.empty else []
    cities = sorted(df["city"].unique().tolist()) if not df.empty else []
    readme = "\n".join(
        [
            "# Urban Plot Graph V3b Series",
            "",
            f"- cities: {len(cities)}",
            f"- variants: {', '.join(variants) if variants else '(none)'}",
            f"- datasets: {len(df)}",
            "",
            "Files:",
            "- `urban_plot_v3b_summary.csv`: flat manifest across all generated city/variant datasets.",
            "- `urban_plot_v3b_summary.json`: JSON version of the same manifest.",
            "",
            "Variant semantics:",
            "- `v3bs`: shared-street topology only.",
            "- `v3bsj`: shared-street + shared-junction topology with relation-aware pruning.",
            "- `v3bsjg`: `v3bsj` plus geometric fallback edges only for structurally under-connected plots.",
            "",
            "Edge schema note:",
            "- `V3b` uses symmetric endpoint context, keeps `orientation_diff`, and retains `node_feat_cosine/node_feat_l2` as descriptive edge attributes only.",
        ]
    )
    (summary_dir / "README.md").write_text(readme + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-build urban plot graph V3b series.")
    parser.add_argument("--urban_root", type=str, default="data/urban_network_datasets")
    parser.add_argument("--out_root", type=str, default="data")
    parser.add_argument("--summary_dir", type=str, default="results/data_construction_series/urban_plot_v3b_series")
    parser.add_argument("--cities", type=str, default="all", help='Comma-separated list, or "all"')
    parser.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    urban_root = (repo_root / args.urban_root).resolve()
    out_root = (repo_root / args.out_root).resolve()
    summary_dir = (repo_root / args.summary_dir).resolve()
    builder = (repo_root / "tools" / "prepare_urban_plot_graph_v3b.py").resolve()

    if not urban_root.exists():
        raise FileNotFoundError(f"Urban root not found: {urban_root}")
    if not builder.exists():
        raise FileNotFoundError(f"Builder script not found: {builder}")

    cities = _discover_cities(urban_root) if str(args.cities).strip().lower() == "all" else _parse_csv_arg(args.cities)
    variants = _parse_csv_arg(args.variants)
    if not cities:
        raise ValueError("No cities selected.")
    if not variants:
        raise ValueError("No variants selected.")

    rows: list[dict] = []
    for city in cities:
        for variant in variants:
            dataset_name = _build_one(
                builder=builder,
                city=city,
                variant=variant,
                urban_root=urban_root,
                out_root=out_root,
                force=bool(args.force),
            )
            meta_path = out_root / dataset_name / f"{dataset_name}_meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"Expected meta file not found: {meta_path}")
            rows.append(_meta_row(meta_path))

    _write_summary(summary_dir=summary_dir, rows=rows)
    print(f"[ok] generated {len(rows)} V3b datasets")
    print(f" summary={summary_dir}")


if __name__ == "__main__":
    main()
