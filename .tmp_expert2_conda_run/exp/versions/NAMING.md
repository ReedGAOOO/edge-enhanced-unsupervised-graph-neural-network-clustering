# Mainline Naming (ECHF Family)

ECHF = **E**dge-**C**alibrated **H**ierarchical **F**usion.

## Canonical names
- `B15_ECHF_main`: former `B15_PATHB_v12_hier` (Path-B + V12 + hierarchical edge attributes, sched=100).
- `B15_ECHF_s60`: former `B15_PATHB_v12_hier_sched60` (same structure, shorter schedule).
- `G15_ECHF_main`: former `G15_default_hetero` (preset for heterogeneous/urban graphs).
- `G15_ECHF_noadapt`: former `G15_noadapt_hetero`.
- `G17_V5_temp15`: former `G17_temp1p5_mainline` (V5 comparator).

## Preset aliases in `tools/run_preset.py`
- `g15_echf_main` (default) <= `g15_default_hetero`
- `g15_echf_noadapt` <= `g15_noadapt_hetero`
- `g17_v5_temp15` <= `g17_temp1p5_mainline`
- `b15_echf_branch` (sched=100)
- `b15_echf_branch_s60`

Old names are still accepted for backward compatibility.
