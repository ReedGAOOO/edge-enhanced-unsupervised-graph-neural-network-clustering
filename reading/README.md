# Reading notes and historical bundles

Updated 2026-10-15. Current results and model choices are documented in the [root README](../README.md),
[experiment index](../configs/breakthrough/README.md) and [tracked evidence](../configs/breakthrough/evidence/README.md).

- [Original DSE explanation (English)](EXPLAIN_DSE_original_en.md) and
  [Chinese version](EXPLAIN_DSE_original_zh.md) explain the reference implementation's training/decoding flow.
  They are reference-code notes rather than a specification of every current edge-aware variant.
- `EXPERT_BRIEF_model_status_zh.md` is an earlier expert briefing; its rankings predate the October backbone audit.
- `expert_code_bundle*_20260414/` preserves April code bundles and their original README files. EP1/EP2 are retired
  as current directions; their original documentation is retained for provenance.
- Historical version notes live under `archive/docs/exp/versions/`. Empty branch-note stubs here are compatibility
  placeholders, not evidence or supported implementations.

Full E2 reproduction also needs local warm-start-patched expert copies under `results/breakthrough_v1/_code/`;
those ignored copies are distinct from the original bundles in this folder.
