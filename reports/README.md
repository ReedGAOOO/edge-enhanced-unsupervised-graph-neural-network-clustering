# Urban data reports

Updated 2026-10-15. These are Urban data-construction and validity reports, separate from the October clustering
benchmark in the [root README](../README.md).

| Source | Rendered artifacts | Scope |
|---|---|---|
| `urban_graph_factor_report.Rmd` | `.md`, `.pdf`, `urban_graph_factor_report_files/` | Cross-city graph differences and the 3+6 factor diagnosis |
| `urban_factor_experiment_packages.Rmd` | `.md`, `.pdf`, associated figures | Follow-up factor experiment packages |

Sources use outputs of the Urban analysis tools and locally prepared/raw datasets. Start with
`tools/run_urban_data_paper_pipeline.py` or the specific factor runner, and inspect `tools/render_urban_factor_report.R`
and `tools/render_urban_factor_experiment_packages.R` for rendering inputs and R dependencies. A Git clone alone
does not include the ignored source experiment outputs.

These reports preserve their original analysis context. Older model comparisons used a short training budget and,
in some cases, treated `unknown` as a class; do not cite them as current clustering rankings. Current known-only
Urban NMI does not support using land-use labels as a community-recovery benchmark. Source quality, projection
rules and city differences remain relevant to the Urban data-paper track.

Some generated Markdown figure paths are absolute paths from the original workstation. The checked-in PDFs provide
a portable rendered view; rerender from the Rmd source when regenerating figures in another environment.
