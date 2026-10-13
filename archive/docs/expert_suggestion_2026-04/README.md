> **Retired 2026-10-13.** Both directions were re-tested with a trained backbone on `synth_ctx_v1`, a benchmark built to
> reward context-dependent edges. Expert1 (contextual edge state, EP1): no contextual effect (+0.003/+0.006 NMI) and
> the context-off ablation is just as good. Expert2 (relation state, EP2/B50): every run collapses to a single
> cluster after warm start. See `results/breakthrough_v1/e2_synth_ctx_warm/mechanism_table.md` and the README.

# Expert Suggestion

This folder keeps the currently active expert guidance used for ongoing model iterations.

## Current experts

- `expert1/`
  - contextual edge-state direction
  - primary note: `expert suggestion.md`
- `expert2/`
  - relation-state / `B50` direction
  - primary note: `expert suggestion.md`
  - supporting implementation bundle: `V40_NOTES.md`, `b50_v40_relation_state.json`, `relation_state_v40_local.patch`, `v40_relation_state_dropin/`

## Usage

- Keep currently active expert guidance here.
- If a direction becomes historical only, move it back into `archive/`.
- Add future experts as sibling folders such as `expert3/`.
