# Breakthrough experiment plans

Updated 2026-10-15. The [root README](../../README.md) explains findings; [evidence/](evidence/README.md) holds compact
tracked tables. Full training outputs and checkpoints are excluded from Git.

| Plan | Question and scope | Evidence status |
|---|---|---|
| `e1_backbone_gate.json` | Does the longer training budget train the backbone? | Recorded gate runs; includes historical synthetic graphs |
| `e2d_plateau_diag.json` | Why does cold training stick on weak features? | Diagnostic sweeps |
| `e2w_warm_smoke.json` | Does pseudo-partition warm start escape the plateau? | Smoke check, not the main benchmark |
| `e2_synth_ctx_warm.json` | Edge semantics versus architecture on `synth_ctx_v1` | Main E2 evidence; EP2 collapsed and was stopped early |
| `e3_real_dual.json` | Warm/cold starts on five real graphs | Original E3 tables select each model by its own SE loss |
| `e4_b45_collapse_ablation.json` | Which B45 component causes collapse? | Component ablations on Photo/AIFB |
| `e6_b31_robust.json` | Does removing the gate preserve gains and avoid collapse? | B31 synthetic/real runs, including ACM |
| `e7_ctx_edge.json` | Can endpoint products unlock contextual edges? | Synthetic context controls |
| `e8_ctx_real.json` | Does B31 plus context help real graphs? | Real-graph context follow-up |
| `e9_fixed_ruler.json` | Do message-only learned factors improve the SE objective? | 44/92 planned runs recorded: all real jobs; 48 synthetic jobs have no recorded results |
| `e9x_machine_check.json` | How much do two seed-0 runs drift on another GPU? | 2/2 recorded; current-host launcher, no remote orchestration |

E5 headroom, E10 SE alignment and the B45 gate diagnosis are standalone tools, not missing numbered plan files.
Their commands are documented in [tools/README.md](../../tools/README.md).

## Launching and resuming

```bash
# Preview the 44 E9 real jobs without training
.venv/bin/python tools/run_breakthrough.py --plan configs/breakthrough/e9_fixed_ruler.json --root_path data \
  --only_tags b31fr_warm,b31fr_nowarm,b31frctx_warm,b31frctx_nowarm --dry_run

# Run that subset on the current machine
.venv/bin/python tools/run_breakthrough.py --plan configs/breakthrough/e9_fixed_ruler.json --root_path data \
  --only_tags b31fr_warm,b31fr_nowarm,b31frctx_warm,b31frctx_nowarm --gpu 0 --max_parallel 4
```

Use `--only_tags b31fr_synth,b31frctx_synth` for the 48 planned E9 synthetic jobs. Their presence in a JSON plan does
not establish completion. `--dry_run` prints commands without writing output files, even when matching runs are cached.
A run with an existing `<dataset>_metrics.json` is cached without validating the code/config
revision; use `--force` or a new plan `name` after changing either. A filtered invocation rewrites that plan's `runs.csv`
and `summary.csv` for the selected jobs, although per-run directories remain. Preserve ledgers before changing the subset.

Each run writes `stdout.txt` (first line: exact command), a training log and metrics JSON. The plan root holds
`plan.json`, `runs.csv` and `summary.csv`. Checkpoints are under `checkpoints/breakthrough_v1/...`.
`--max_parallel` controls processes on one GPU, not agents or machines. Four or fewer jobs is a practical starting point
for larger graphs; reduce it if memory contention develops. `--threads` caps CPU library threads per process.

## Prerequisites and interpretation

- Prepare every dataset named by the selected jobs. See [data/README.md](../../data/README.md). Urban raw city files
  are external inputs. The generator options for the documented synthetic suite are `--feature_sep 0.35 --homophily 0.45,0.55`.
- EP1/EP2 jobs require patched local expert code copies under `results/breakthrough_v1/_code/`. Those ignored copies
  are not included in a clone; archived original workspaces do not contain the same warm-start hook.
- Checkpoints are selected by training loss. `oracle_best_nmi` uses labels to choose an evaluation epoch and is retained
  for reference comparison only. It is not the reported label-free checkpoint score.
- Combined E6/E8/E9 tables choose each dataset/seed's warm/cold start using the edge-free baseline's lower
  `best_train_loss_mean`, then apply that start to every model. The report falls back to a model's own loss if the
  baseline pair is missing; inspect `dual_start_detail.csv` when reporting a new subset.
- The benchmark supplies class counts/cluster budgets. E5/E10 use labels for diagnosis and do not train the models.
- E9 `msg_only` skips learned scalar SE weighting but retains augmentation, base weights, assignment residuals and
  the edge regularizer. Its SE graph is not a frozen original adjacency.

## Regenerating the real-graph report

With the E3/E6/E8/E9 output ledgers and classic-floor CSVs present:

```bash
.venv/bin/python tools/report_e3_real.py \
  --extra_dirs results/breakthrough_v1/e6_b31_robust,results/breakthrough_v1/e8_ctx_real,results/breakthrough_v1/e9_fixed_ruler \
  --extra_classic results/breakthrough_v1/e6_b31_robust/classic_acm_v2.csv \
  --datasets cora,photo,dblp_magnn_author_v2,entities_aifb,acm_multiplex_v2 \
  --out_dir results/breakthrough_v1/combined_real
```

`--out_dir` writes `real_table.md` and `dual_start_detail.csv` separately from the source experiments. It does not change
where input CSVs are read. The E9 machine check has only two warm runs and should be kept separate from this report.
