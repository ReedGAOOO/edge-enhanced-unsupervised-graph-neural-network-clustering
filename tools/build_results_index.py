#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
STAGE_DIR = REPO_ROOT / "archive" / "docs" / "exp" / "stage_results"
INDEX_DIR = RESULTS_DIR / "_index"

AGG_KEY_FILES = {
    "README.md",
    "PLAN.md",
    "runs.csv",
    "summary.csv",
    "summary.json",
    "report.json",
    "analysis.md",
    "decision.json",
}

ENTRY_PREFERENCE = [
    "README.md",
    "PLAN.md",
    "summary.json",
    "summary_by_condition.csv",
    "summary_by_condition_dataset.csv",
    "summary.csv",
    "summary_by_dataset_mode_k.csv",
    "summary_by_mode_k.csv",
    "summary_by_alpha.csv",
    "summary_overall_4way.csv",
    "summary_overall_3way.csv",
    "summary_overall_by_preset.csv",
    "best_condition_by_dataset.csv",
    "delta_vs_baseline.csv",
    "report.json",
    "analysis.md",
    "decision.json",
    "runs.csv",
]

IMPORTANT_RESULTS = [
    ("Current B45 control-suite summary", Path("results/mainline_evidence/diagnostic_b45_confirm_grid9_v1/summary_by_condition.csv")),
    ("Current B45 per-dataset breakdown", Path("results/mainline_evidence/diagnostic_b45_confirm_grid9_v1/summary_by_condition_dataset.csv")),
    ("B47/B48 follow-up", Path("results/mainline_evidence/diagnostic_b47b48_repr3_v1/summary_by_condition.csv")),
    ("EP1/EP2 comparison by group", Path("results/mainline_evidence/ep_compare_v1/summary_by_group_model.csv")),
    ("Current Urban V3b series", Path("results/data_construction_series/urban_plot_v3b_series/urban_plot_v3b_summary.csv")),
    ("Archived legacy stage hub", Path("archive/docs/exp/stage_results/README.md")),
]

CURATED_AGGREGATES = [
    ("Current B45 control-suite summary", "results/mainline_evidence/diagnostic_b45_confirm_grid9_v1/summary_by_condition.csv"),
    ("Current B45 per-dataset breakdown", "results/mainline_evidence/diagnostic_b45_confirm_grid9_v1/summary_by_condition_dataset.csv"),
    ("B47/B48 follow-up", "results/mainline_evidence/diagnostic_b47b48_repr3_v1/summary_by_condition.csv"),
    ("Current EP1/EP2 comparison", "results/mainline_evidence/ep_compare_v1/summary_by_group_model.csv"),
    ("Current Urban V3b construction series", "results/data_construction_series/urban_plot_v3b_series/urban_plot_v3b_summary.csv"),
    ("Current Urban V3 vs V3b comparison", "results/data_construction_series/urban_v3_v3b_compare/summary.csv"),
    ("Current DBLP construction comparison", "results/data_construction_series/dblp_magnn_integration_v2/v1_vs_v2_model_compare.csv"),
    ("Archived mechanism-control legacy summary", "archive/results/historical/benchmark_mechanism_synth_full_v1/summary_by_condition.csv"),
    ("Archived permutation legacy summary", "archive/results/historical/benchmark_mechanism_permEA_v1/permutation_effect_summary.csv"),
]


@dataclass
class DirRecord:
    name: str
    rel_path: str
    bucket: str
    dir_type: str
    n_files: int
    n_subdirs: int
    updated_at: str
    entry_file: str
    key_files: str
    notes: str


def _iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def _rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _bucket_for_name(name: str) -> str:
    if name == "raw_maps" or "map" in name:
        return "maps"
    if name.startswith("urban_"):
        return "urban"
    if name.startswith("benchmark_"):
        return "benchmark"
    if name.startswith("diagnostic_"):
        return "diagnostic"
    if name.startswith("convergence_"):
        return "stability"
    if "integration" in name:
        return "integration"
    if name.startswith(("phase2_", "prior_", "edge_attr_", "ablation_", "ablate_")):
        return "analysis"
    if name.startswith(("bf16_", "tmp_")):
        return "scratch"
    return "misc"


def _is_aggregate_dir(path: Path, files: Sequence[str]) -> bool:
    if any(name in AGG_KEY_FILES for name in files):
        return True
    return any(
        name.startswith("summary")
        or name.startswith("best_condition_by_dataset")
        or name.startswith("delta_vs_baseline")
        or name.startswith("compare_")
        or name.endswith(".md")
        for name in files
    )


def _choose_entry(files: Sequence[str]) -> str:
    file_set = set(files)
    for name in ENTRY_PREFERENCE:
        if name in file_set:
            return name
    summary_like = sorted([name for name in files if name.startswith("summary")])
    if summary_like:
        return summary_like[0]
    return files[0] if files else ""


def _key_files(files: Sequence[str], limit: int = 8) -> List[str]:
    scored = sorted(files, key=lambda x: (0 if x == _choose_entry(files) else 1, x))
    return scored[:limit]


def _parse_stage_hub_table(stage_hub: Path) -> Dict[str, Dict[str, str]]:
    if not stage_hub.exists():
        return {}
    rows: Dict[str, Dict[str, str]] = {}
    pattern = re.compile(
        r"^\|\s*`(?P<stage>[^`]+)`\s*\|\s*(?P<focus>.*?)\s*\|\s*`(?P<source>[^`]+)`\s*\|\s*(?P<top>.*?)\s*\|\s*(?P<takeaway>.*?)\s*\|$"
    )
    for raw_line in stage_hub.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        m = pattern.match(line)
        if not m:
            continue
        stage = m.group("stage")
        rows[stage] = {
            "focus": m.group("focus"),
            "source_tag": m.group("source"),
            "top_line": m.group("top"),
            "key_takeaway": m.group("takeaway"),
        }
    return rows


def _collect_results_records() -> Dict[str, List[DirRecord]]:
    aggregate: List[DirRecord] = []
    raw_runs: List[DirRecord] = []
    other: List[DirRecord] = []

    for path in sorted([p for p in RESULTS_DIR.iterdir() if p.is_dir()]):
        if path.name == "_index":
            continue
        files = sorted([p.name for p in path.iterdir() if p.is_file()])
        subdirs = sorted([p.name for p in path.iterdir() if p.is_dir()])

        entry = _choose_entry(files)
        key_files = _key_files(files)
        record = DirRecord(
            name=path.name,
            rel_path=_rel(path),
            bucket=_bucket_for_name(path.name),
            dir_type="aggregate" if _is_aggregate_dir(path, files) else "other",
            n_files=len(files),
            n_subdirs=len(subdirs),
            updated_at=_iso_mtime(path),
            entry_file=f"{_rel(path / entry)}" if entry else "",
            key_files=" | ".join(key_files),
            notes="",
        )
        if files and all(name.endswith(".log") or name.endswith("_metrics.json") for name in files):
            record.dir_type = "raw_run"
            raw_runs.append(record)
        elif record.dir_type == "aggregate":
            if path.name == "raw_maps":
                record.notes = "Rendered figures and JSON summaries."
            aggregate.append(record)
        else:
            if not files and not subdirs:
                record.notes = "Empty placeholder directory."
            elif path.name == "raw_maps":
                record.notes = "Rendered figures and JSON summaries."
            else:
                record.notes = "Non-standard directory; inspect manually if needed."
            other.append(record)

    return {"aggregate": aggregate, "raw_run": raw_runs, "other": other}


def _assign_raw_run_families(raw_runs: Sequence[DirRecord], aggregate_names: Sequence[str]) -> List[Dict[str, str]]:
    agg_sorted = sorted(aggregate_names, key=len, reverse=True)
    family_counts: Counter[str] = Counter()
    family_examples: Dict[str, str] = {}

    for record in raw_runs:
        family = ""
        for agg_name in agg_sorted:
            if record.name.startswith(f"{agg_name}_"):
                family = agg_name
                break
        if not family:
            family = re.sub(r"_s\d+$", "", record.name)
            family = re.sub(r"_(cora|citeseer|pubmed|computers|photo|karateclub|football)$", "", family, flags=re.IGNORECASE)
        family_counts[family] += 1
        family_examples.setdefault(family, record.rel_path)

    rows: List[Dict[str, str]] = []
    for family, count in sorted(family_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        rows.append(
            {
                "family": family,
                "bucket": _bucket_for_name(family),
                "raw_run_count": str(count),
                "example_run_dir": family_examples[family],
            }
        )
    return rows


def _collect_stage_records(stage_meta: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    stage_paths = sorted(
        [p for p in STAGE_DIR.iterdir() if p.is_dir() and p.name.startswith("stage")],
        key=lambda p: _stage_sort_key(p.name),
    )
    for path in stage_paths:
        files = sorted([p.name for p in path.iterdir() if p.is_file()])
        entry = _choose_entry(files)
        meta = stage_meta.get(path.name, {})
        rows.append(
            {
                "stage": path.name,
                "rel_path": _rel(path),
                "updated_at": _iso_mtime(path),
                "entry_file": _rel(path / entry) if entry else "",
                "focus": meta.get("focus", ""),
                "source_tag": meta.get("source_tag", ""),
                "top_line": meta.get("top_line", ""),
                "key_takeaway": meta.get("key_takeaway", ""),
                "key_files": " | ".join(_key_files(files)),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _md_link(rel_path: str) -> str:
    return f"[{rel_path}]({rel_path})"


def _results_readme_target(repo_rel_path: str) -> str:
    full = (REPO_ROOT / repo_rel_path).resolve()
    return Path(os.path.relpath(full, RESULTS_DIR.resolve())).as_posix()


def _results_readme_link(repo_rel_path: str) -> str:
    target = _results_readme_target(repo_rel_path)
    return f"[{repo_rel_path}]({target})"


def _stage_sort_key(stage_name: str) -> tuple[int, str]:
    m = re.match(r"stage(\d+)", stage_name)
    if m:
        return (int(m.group(1)), stage_name)
    return (10**9, stage_name)


def _render_results_readme(
    aggregate: Sequence[DirRecord],
    other: Sequence[DirRecord],
    stage_rows: Sequence[Dict[str, str]],
    raw_family_rows: Sequence[Dict[str, str]],
) -> str:
    lines: List[str] = []
    lines.append("# Results Hub")
    lines.append("")
    lines.append("This file is the entrypoint for result lookup. It is generated by `tools/build_results_index.py`.")
    lines.append("")
    lines.append("## Start Here")
    for label, rel_path in IMPORTANT_RESULTS:
        full = REPO_ROOT / rel_path
        if full.exists():
            lines.append(f"- {label}: {_results_readme_link(rel_path.as_posix())}")
    lines.append("")
    lines.append("## What Lives Where")
    lines.append(f"- `results/`: active category hubs only. Top-level active dirs: `{len(aggregate)}`.")
    lines.append("- `results/mainline_evidence/raw_runs/`: nested raw runs produced by current active experiment entrypoints.")
    lines.append(f"- `results/_index/`: machine-readable manifests generated for lookup.")
    lines.append(f"- `archive/docs/exp/stage_results/`: archived stage-level summaries from earlier branches. Stage dirs: `{len(stage_rows)}`.")
    other_count = len(other)
    if other_count:
        lines.append(f"- Non-standard or asset dirs under `results/`: `{other_count}`.")
    lines.append("")
    lines.append("## Primary Current Summaries")
    for label, rel_path in CURATED_AGGREGATES:
        full = REPO_ROOT / rel_path
        if full.exists():
            lines.append(f"- {label}: {_results_readme_link(rel_path)}")
    lines.append("")
    lines.append("## Archived Stage Summaries")
    lines.append(f"- Stage hub: {_results_readme_link('archive/docs/exp/stage_results/README.md')}")
    latest_stage_rows = sorted(stage_rows, key=lambda row: _stage_sort_key(row["stage"]))[-4:]
    for row in latest_stage_rows:
        focus = f" - {row['focus']}" if row["focus"] else ""
        lines.append(f"- {row['stage']}: {_results_readme_link(row['entry_file'])}{focus}")
    lines.append("")
    lines.append("## Full Catalog Files")
    lines.append(f"- Aggregate result dirs: {_results_readme_link('results/_index/aggregate_results.csv')}")
    lines.append(f"- Stage result dirs: {_results_readme_link('results/_index/stage_results.csv')}")
    lines.append(f"- Raw run family counts: {_results_readme_link('results/_index/raw_run_families.csv')}")
    lines.append(f"- Scan overview JSON: {_results_readme_link('results/_index/overview.json')}")
    lines.append("")
    if raw_family_rows:
        lines.append("## Top Raw Run Families")
        for row in raw_family_rows[:15]:
            lines.append(
                f"- `{row['family']}`: `{row['raw_run_count']}` raw runs. Example: {_results_readme_link(row['example_run_dir'])}"
            )
        lines.append("")
    lines.append("## Notes")
    lines.append("- Early branch-era results were moved into `archive/results/historical/` to keep the repo root focused on the current mainline.")
    lines.append("- If you add new runs later, rebuild this hub with `python3 tools/build_results_index.py`.")
    return "\n".join(lines) + "\n"


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    result_records = _collect_results_records()
    aggregate = sorted(result_records["aggregate"], key=lambda r: (r.bucket, r.name))
    raw_runs = sorted(result_records["raw_run"], key=lambda r: r.name)
    other = sorted(result_records["other"], key=lambda r: r.name)

    aggregate_rows = [
        {
            "name": r.name,
            "rel_path": r.rel_path,
            "bucket": r.bucket,
            "dir_type": r.dir_type,
            "n_files": str(r.n_files),
            "n_subdirs": str(r.n_subdirs),
            "updated_at": r.updated_at,
            "entry_file": r.entry_file,
            "key_files": r.key_files,
            "notes": r.notes,
        }
        for r in aggregate
    ]
    _write_csv(
        INDEX_DIR / "aggregate_results.csv",
        aggregate_rows,
        ["name", "rel_path", "bucket", "dir_type", "n_files", "n_subdirs", "updated_at", "entry_file", "key_files", "notes"],
    )

    other_rows = [
        {
            "name": r.name,
            "rel_path": r.rel_path,
            "bucket": r.bucket,
            "dir_type": r.dir_type,
            "n_files": str(r.n_files),
            "n_subdirs": str(r.n_subdirs),
            "updated_at": r.updated_at,
            "entry_file": r.entry_file,
            "key_files": r.key_files,
            "notes": r.notes,
        }
        for r in other
    ]
    _write_csv(
        INDEX_DIR / "other_dirs.csv",
        other_rows,
        ["name", "rel_path", "bucket", "dir_type", "n_files", "n_subdirs", "updated_at", "entry_file", "key_files", "notes"],
    )

    raw_family_rows = _assign_raw_run_families(raw_runs, [r.name for r in aggregate])
    _write_csv(
        INDEX_DIR / "raw_run_families.csv",
        raw_family_rows,
        ["family", "bucket", "raw_run_count", "example_run_dir"],
    )

    stage_meta = _parse_stage_hub_table(STAGE_DIR / "README.md")
    stage_rows = _collect_stage_records(stage_meta)
    _write_csv(
        INDEX_DIR / "stage_results.csv",
        stage_rows,
        ["stage", "rel_path", "updated_at", "entry_file", "focus", "source_tag", "top_line", "key_takeaway", "key_files"],
    )

    overview = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "aggregate_result_dir_count": len(aggregate),
        "raw_run_dir_count": len(raw_runs),
        "other_dir_count": len(other),
        "stage_dir_count": len(stage_rows),
        "top_raw_run_families": raw_family_rows[:20],
        "buckets": {bucket: sum(1 for r in aggregate if r.bucket == bucket) for bucket in sorted({r.bucket for r in aggregate})},
    }
    (INDEX_DIR / "overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")

    results_readme = _render_results_readme(aggregate, other, stage_rows, raw_family_rows)
    (RESULTS_DIR / "README.md").write_text(results_readme, encoding="utf-8")

    print("[ok] results index built")
    print(f" aggregate_result_dirs={len(aggregate)}")
    print(f" raw_run_dirs={len(raw_runs)}")
    print(f" other_dirs={len(other)}")
    print(f" stage_dirs={len(stage_rows)}")
    print(f" readme={RESULTS_DIR / 'README.md'}")
    print(f" index_dir={INDEX_DIR}")


if __name__ == "__main__":
    main()
