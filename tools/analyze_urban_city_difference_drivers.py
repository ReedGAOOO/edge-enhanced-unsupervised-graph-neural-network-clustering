#!/usr/bin/env python3
"""Audit likely drivers of cross-city Urban graph differences.

This script does not claim causal identification.  It combines the raw Urban
diagnostics, existing v3b graph diagnostics, and the CPU edge-probe screen to
separate three plausible explanation layers:

1. data/label measurement differences,
2. graph-projection artifacts introduced by our construction rules,
3. observable urban-form proxies available inside the current Urban files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


CITY_DISPLAY = {
    "washingtondc": "WashingtonDC",
    "tokyo": "Tokyo",
    "paris": "Paris",
    "boston": "Boston",
    "madrid": "Madrid",
    "beijing": "Beijing",
    "shanghai": "Shanghai",
    "singapore": "Singapore",
    "melbourne": "Melbourne",
    "sydney": "Sydney",
    "bangkok": "Bangkok",
    "chicago": "Chicago",
    "johannesburg": "Johannesburg",
}


DRIVERS = {
    "known_label_ratio": ("data_label", "Observed label coverage"),
    "known_label_entropy_norm": ("data_label", "Known-label class balance"),
    "largest_known_class_ratio": ("data_label", "Dominant known-label share"),
    "label_quality_proxy": ("data_label", "Coverage x entropy x non-dominance"),
    "building_height_known_frac": ("data_source", "Building-height completeness"),
    "plot_without_building_frac": ("data_source", "Plots without linked buildings"),
    "projection_tail_score": ("graph_projection", "Combined high-load projection tail"),
    "street_entity_plot_load_p99": ("graph_projection", "Street segment to plot p99 load"),
    "junction_entity_plot_load_p99": ("graph_projection", "Junction to plot p99 load"),
    "plot_building_membership_p99": ("graph_projection", "Plot to building p99 membership"),
    "avg_undirected_degree": ("graph_projection", "Constructed graph mean degree"),
    "street_edge_frac": ("graph_projection", "Constructed street-edge share"),
    "junction_edge_frac": ("graph_projection", "Constructed junction-edge share"),
    "geom_edge_frac": ("graph_projection", "Constructed geometry-edge share"),
    "shared_building_bit_frac": ("graph_projection", "Constructed building-edge share"),
    "building_per_plot": ("urban_form_proxy", "Raw buildings per plot"),
    "street_per_plot": ("urban_form_proxy", "Raw street segments per plot"),
    "intersection_per_plot": ("urban_form_proxy", "Raw intersections per plot"),
    "plot_area_p50": ("urban_form_proxy", "Median plot area"),
    "plot_area_tail": ("urban_form_proxy", "Plot area p99/p50 tail"),
    "building_area_p50": ("urban_form_proxy", "Median building footprint area"),
    "building_area_tail": ("urban_form_proxy", "Building area p99/p50 tail"),
    "building_height_mean": ("urban_form_proxy", "Mean building height"),
    "plot_orientation_entropy_norm": ("urban_form_proxy", "Plot orientation entropy"),
    "compact_intensity_proxy": ("urban_form_proxy", "Buildings per plot adjusted by plot area"),
}


TARGETS = {
    "edge_probe_gain": "edge_probe_gain",
    "edge_probe_perm_drop": "edge_probe_perm_drop",
    "overall_same_label_ratio": "overall_same_label_ratio",
    "exploration_priority": "exploration_priority",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(skipna=True)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean(skipna=True)) / std


def build_driver_frame(args: argparse.Namespace) -> pd.DataFrame:
    base_dir = Path(args.base_dir)
    raw = _read_csv(base_dir / "raw_city_diagnostics.csv")
    priority = _read_csv(base_dir / "all_city_exploration_priority.csv")
    v3b = _read_csv(Path(args.v3b_diag_path))
    probe = _read_csv(Path(args.probe_summary_path))

    probe = probe.loc[probe["feature_set"] == "endpoint_edge_attr"].copy()
    probe["city"] = probe["dataset"].str.extract(r"urban_([^_]+)_")
    probe = probe.rename(
        columns={
            "auc_gain_vs_endpoint_mean": "edge_probe_gain",
            "auc_drop_vs_perm_mean": "edge_probe_perm_drop",
        }
    )

    keep_v3b = [
        "city",
        "overall_same_label_ratio",
        "street_edge_frac",
        "junction_edge_frac",
        "geom_edge_frac",
        "shared_building_bit_frac",
        "avg_undirected_degree",
        "largest_component_ratio",
    ]
    df = (
        raw.merge(v3b[keep_v3b], on="city", how="left")
        .merge(probe[["city", "edge_probe_gain", "edge_probe_perm_drop"]], on="city", how="left")
        .merge(priority[["city", "exploration_priority", "priority_group"]], on="city", how="left")
    )

    df["city_label"] = df["city"].map(CITY_DISPLAY).fillna(df["city"].str.title())
    df["building_per_plot"] = df["n_buildings"] / df["n_plots"]
    df["street_per_plot"] = df["n_streets"] / df["n_plots"]
    df["intersection_per_plot"] = df["n_intersections"] / df["n_plots"]
    df["label_quality_proxy"] = (
        df["known_label_ratio"] * df["known_label_entropy_norm"] * (1 - df["largest_known_class_ratio"])
    )
    df["projection_tail_score"] = (
        np.log1p(df["street_entity_plot_load_p99"])
        + np.log1p(df["junction_entity_plot_load_p99"])
        + np.log1p(df["plot_building_membership_p99"])
    )
    df["plot_area_tail"] = df["plot_area_p99"] / df["plot_area_p50"].replace(0, np.nan)
    df["building_area_tail"] = df["building_area_p99"] / df["building_area_p50"].replace(0, np.nan)
    df["compact_intensity_proxy"] = np.log1p(df["building_per_plot"]) / np.log1p(df["plot_area_p50"])

    df["label_z"] = _zscore(df["label_quality_proxy"])
    df["projection_tail_z"] = _zscore(df["projection_tail_score"])
    df["edge_signal_z"] = _zscore(df["edge_probe_gain"].fillna(0)) + _zscore(
        df["edge_probe_perm_drop"].fillna(0)
    )
    df["urban_density_z"] = _zscore(df["building_per_plot"]) + _zscore(df["street_per_plot"]) - _zscore(
        np.log1p(df["plot_area_p50"])
    )
    return df


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for driver, (group, label) in DRIVERS.items():
        if driver not in df.columns:
            continue
        for target_name, target_col in TARGETS.items():
            pair = df[[driver, target_col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < 4:
                continue
            if pair[driver].nunique() <= 1 or pair[target_col].nunique() <= 1:
                continue
            rho, p_value = spearmanr(pair[driver], pair[target_col])
            if not np.isfinite(rho):
                continue
            rows.append(
                {
                    "driver": driver,
                    "driver_label": label,
                    "driver_group": group,
                    "target": target_name,
                    "spearman_rho": rho,
                    "abs_spearman_rho": abs(rho),
                    "p_value": p_value,
                    "n": len(pair),
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["target", "abs_spearman_rho"], ascending=[True, False])


def classify_cities(df: pd.DataFrame) -> pd.DataFrame:
    labels = []
    for row in df.itertuples(index=False):
        reasons = []
        if row.edge_probe_gain > 0.006 and row.edge_probe_perm_drop > 0.006:
            reasons.append("strong edge_attr signal")
        elif row.edge_probe_gain > 0.001 and row.edge_probe_perm_drop > 0.001:
            reasons.append("weak positive edge_attr signal")
        elif row.edge_probe_gain <= 0 or row.edge_probe_perm_drop <= 0:
            reasons.append("edge_attr not robust")
        else:
            reasons.append("near-zero edge_attr signal")

        if row.known_label_ratio < 0.12:
            reasons.append("low label coverage")
        elif row.known_label_ratio >= 0.25:
            reasons.append("usable/high label coverage")

        if row.projection_tail_z > 0.8:
            reasons.append("high projection-tail risk")
        if row.urban_density_z > 0.8:
            reasons.append("dense/fine-grained urban-form proxy")
        if row.urban_density_z < -0.8:
            reasons.append("large-plot/sparse-form proxy")

        if row.edge_probe_gain > 0.006 and row.known_label_ratio >= 0.12:
            typology = "anchor candidate"
        elif row.edge_probe_gain > 0.006 and row.known_label_ratio < 0.12:
            typology = "edge-rich but label-limited"
        elif row.known_label_ratio >= 0.25 and row.edge_probe_gain <= 0.001:
            typology = "label-rich but edge-weak"
        elif row.projection_tail_z > 0.8:
            typology = "projection-risk diagnostic"
        elif row.edge_probe_gain > 0.001:
            typology = "secondary diagnostic"
        else:
            typology = "low-priority or negative control"

        labels.append(
            {
                "city": row.city,
                "city_label": row.city_label,
                "typology": typology,
                "known_label_ratio": row.known_label_ratio,
                "edge_probe_gain": row.edge_probe_gain,
                "edge_probe_perm_drop": row.edge_probe_perm_drop,
                "projection_tail_score": row.projection_tail_score,
                "projection_tail_z": row.projection_tail_z,
                "urban_density_z": row.urban_density_z,
                "main_reasons": "; ".join(reasons),
            }
        )
    return pd.DataFrame(labels).sort_values(
        ["typology", "edge_probe_gain", "known_label_ratio"], ascending=[True, False, False]
    )


def write_report(df: pd.DataFrame, corr: pd.DataFrame, city_types: pd.DataFrame, out_path: Path) -> None:
    top_edge = (
        corr.loc[corr["target"].isin(["edge_probe_gain", "edge_probe_perm_drop"])]
        .assign(abs_rho=lambda x: x["spearman_rho"].abs())
        .sort_values("abs_rho", ascending=False)
        .head(8)
    )
    top_same = (
        corr.loc[corr["target"] == "overall_same_label_ratio"]
        .assign(abs_rho=lambda x: x["spearman_rho"].abs())
        .sort_values("abs_rho", ascending=False)
        .head(5)
    )

    lines = [
        "# Urban Cross-City Difference Driver Audit",
        "",
        "This is a diagnostic audit, not causal identification. It separates three plausible sources of cross-city differences: data/label measurement, graph-projection artifacts, and observable urban-form proxies available in the current Urban files.",
        "",
        "## Main Reading",
        "",
        "The current evidence supports a mixed explanation. Data/label measurement and graph-projection choices explain a large part of the observed variation, while real urban morphology likely modulates whether those constructed edges align with land-use labels. External causes such as rivers, terrain, planning history, and local economy remain plausible, but they are not identified by the current tables without adding external covariates.",
        "",
        "## Strongest Edge-Probe Associations",
        "",
        "| target | driver group | driver | rho | p |",
        "|---|---:|---|---:|---:|",
    ]
    for row in top_edge.itertuples(index=False):
        lines.append(
            f"| {row.target} | {row.driver_group} | {row.driver_label} | {row.spearman_rho:.3f} | {row.p_value:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Strongest Same-Label Associations",
            "",
            "| target | driver group | driver | rho | p |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for row in top_same.itertuples(index=False):
        lines.append(
            f"| {row.target} | {row.driver_group} | {row.driver_label} | {row.spearman_rho:.3f} | {row.p_value:.3f} |"
        )

    lines.extend(
        [
            "",
            "## City Typology",
            "",
            "| city | typology | main reasons |",
            "|---|---|---|",
        ]
    )
    for row in city_types.itertuples(index=False):
        lines.append(f"| {row.city_label} | {row.typology} | {row.main_reasons} |")

    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- If label coverage or label entropy dominates, the city is mainly a measurement/target-quality issue.",
            "- If projection-tail metrics dominate, the city is testing our graph-construction rule rather than only urban semantics.",
            "- If urban-form proxies dominate after controlling the first two layers, then external urban mechanisms become a stronger explanation.",
            "- To claim rivers, terrain, or economic structure, add external covariates and run matched/counterfactual graph construction.",
        ]
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        default="results/data_construction_series/urban_all_city_exploration",
        help="Directory containing all-city raw diagnostics and priority table.",
    )
    parser.add_argument(
        "--v3b-diag-path",
        default="results/data_diagnostics/urban_v3bsjg/urban_v3bsjg_diagnostics.csv",
    )
    parser.add_argument(
        "--probe-summary-path",
        default="results/edge_probe/urban_v3bsjg_same_label_allcity_cpu_v1/edge_probe_summary.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="results/data_construction_series/urban_all_city_exploration",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_driver_frame(args)
    corr = compute_correlations(df)
    city_types = classify_cities(df)

    df.to_csv(out_dir / "city_difference_driver_frame.csv", index=False)
    corr.to_csv(out_dir / "city_difference_driver_correlations.csv", index=False)
    city_types.to_csv(out_dir / "city_difference_city_typology.csv", index=False)
    write_report(df, corr, city_types, out_dir / "urban_city_difference_driver_audit.md")

    print(f"Wrote {out_dir / 'city_difference_driver_frame.csv'}")
    print(f"Wrote {out_dir / 'city_difference_driver_correlations.csv'}")
    print(f"Wrote {out_dir / 'city_difference_city_typology.csv'}")
    print(f"Wrote {out_dir / 'urban_city_difference_driver_audit.md'}")


if __name__ == "__main__":
    main()
