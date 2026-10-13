# Archive

This folder keeps branch-era material that is no longer part of the active repo surface.

## Layout

- `docs/`
  - archived reports
  - the older `exp/` / `Exp/` documentation hubs
- `configs/historical_presets/`
  - preset JSONs that are no longer part of the active top-level preset surface
- `code/`
  - historical helper scripts and branch-era experiment drivers
  - legacy dataset builders that are no longer part of the active top-level tool surface
- `data/historical/`
  - older dataset variants moved out of the active `data/` surface
- `checkpoints/root_historical/`
  - root-level checkpoint files from older campaigns
- `workspaces/`
  - temporary or branch-specific experiment workspaces
  - archived branch-specific copies and smoke artifacts
- `results/historical/`
  - older raw result families moved out of the repo root

## Policy

- Active training code, active presets, current mainline evidence, and current data-construction summaries stay at the repo root.
- Expert guidance that is still actively used would stay at the repo root under `Expert suggestion/`; as of 2026-10-13 none is (the April guidance is in `docs/expert_suggestion_2026-04/`).
- `docs/README_2026-04_snapshot.md` is the pre-breakthrough README; its model rankings came from untrained 60-120 epoch runs.
- Historical branch exploration, obsolete reports, older raw campaign outputs, and older data variants are moved here.
- Historical preset JSONs and legacy dataset builders are archived here as well.
- Some archived CSVs may still preserve the original execution paths used when the runs were created.
