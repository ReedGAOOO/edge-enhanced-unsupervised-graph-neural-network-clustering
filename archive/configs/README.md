# Archived Configs

This folder keeps preset JSONs that are no longer part of the active top-level preset surface.

- `historical_presets/`
  - earlier branch families (`G13/G15/G17`, `B15`, `B30`-`B38`, interim `B41/B42/B43/B44/B46`, legacy `V40`, and older `A2/V5` presets)
  - retired in the 2026-10 breakthrough pass: `B40` (B45 with a weaker message gate), `B47` (B45 + matched-only gate, no clear gain),
    `B48` (confidence gate, harmful), `G20` (scalar SE weighting, subsumed by B45's SE path). Their rankings came from
    60-epoch runs that did not train the backbone; see `results/breakthrough_v1/` and the README "Breakthrough pass" section.

Use these files by explicit path if you need to reproduce a historical branch exactly.
