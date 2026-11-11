# Depth sweep (plan rows R1–R4): how to run it

Written 2026-11-11 for Akash, who runs the CARC jobs. This sweep is CPU only. Nothing has been submitted yet; every
submission needs Akash's OK.

**What it does.** For each cache layer (0–33: 0 = embeddings, 32 = final post-norm, 33 = raw block 32) and each
training seed in `failure_prob/protocol.py` (0, 1, 2), it:

1. trains one probe on the `train` split;
2. scores every `seen_val` rollout;
3. hands the per-step scores to an evaluator, together with each rollout's horizon, `min(length, task_min_step)`.

It writes a table and a depth-curve plot. It **never reads `unseen_test`**: the loader wrapper refuses that split
before any file opens, and the tests check that no unseen file is ever opened.

| Plan row | Run |
|---|---|
| R1 depth curve | `SWEEP_POOL=mean SWEEP_NORMALIZE=0`, all 34 layers |
| R3 pooling | `SWEEP_POOL=first` and `last`, at the best layer (or all layers: it is cheap) |
| R4 raw vs standardized | `SWEEP_NORMALIZE=1` (z-score with the cache's train-split statistics) vs `0` |
| R2 fine sweep | needs replay or stored neighbours (plan note); not covered here |

## TEMPORARY pieces

Both default plugins are placeholders, and anything they produce is **not a result**. Every row, `summary.csv`
and the plot title say TEMPORARY.

- **Probe (until B6, Aniruth's probe spec): `temporary_mlp`.** The registered `single_layer_mlp`
  (D → 64 → 1, per timestep), trained with `failure_prob/engine/trainer.py`: Adam at lr 1e-3, 5 epochs, 8 rollouts per
  batch, BCE on the rollout label at every step. No early stopping and no selection on seen_val.
- **Evaluator (until B4, Rhea's evaluator): `temporary_max`.** The rollout score is the maximum per-step score
  before the horizon. It reports ROC-AUC, PR-AUC and TPR at 5% FPR. This is not SAFE's exact protocol.

**Swapping them changes no other code.** Put a factory in any module and pass it by name:
`--probe pkg.module:make_probe` (or `SWEEP_PROBE=...`). The same works for `--evaluator`. The contracts are in
`failure_prob/engine/sweep_plugins.py`:

- `Probe.fit(train_records, seed)` returns a fitted probe.
- The fitted probe's `step_scores(records)` returns one score per timestep of each rollout.
- `Evaluator(EvalInput)` returns a dict of metrics.

The probe and evaluator names are part of the output folder name and of `config.json`, so results from different
plugins never mix.

## Files

- `failure_prob/engine/depth_sweep.py`: the sweep, the unseen lock, and aggregation (tables and plot).
- `failure_prob/engine/sweep_plugins.py`: the contracts, the TEMPORARY placeholders, and name resolution.
- `scripts/depth_sweep.py`: the command-line tool, with `run` and `aggregate`.
- `scripts/carc/depth_sweep.slurm`: one array task per layer.
- `scripts/carc/depth_sweep_aggregate.slurm`: aggregation, run after the array.
- `tests/test_depth_sweep.py` and `tests/synthetic_cache.py`: synthetic tests (14) covering:
  - the planted-signal layer is found;
  - unseen files are never opened;
  - horizon and full-length scores reach the evaluator;
  - plugins can be swapped by name;
  - normalization uses train statistics;
  - reruns skip finished rows and refuse a different configuration;
  - both Slurm scripts run end to end with a fake Slurm environment.

  They pass under torch 2.14 (`.venv-kit310`) and under torch 2.2.0 with numpy 1.26 (what the CARC kit env uses).

## On CARC (CPU jobs; Akash submits)

Run everything from the root of a Safe-Research checkout at the commit that holds this file. The Python
environment defaults to the collection kit's venv (`/scratch1/<user>/safe_cache_work/venv310`). It needs torch,
numpy, pandas and scikit-learn; matplotlib is needed only for the plot. Set `SWEEP_PYTHON` to use another
environment.

```bash
export SWEEP_POOL=mean SWEEP_NORMALIZE=0        # CACHE_DIR defaults to /scratch1/<user>/safe_cache/v1
# 1. tiny test on 2 layers (sizes the full run on real data)
sbatch --account=<account> --export=ALL --array=0-1 scripts/carc/depth_sweep.slurm
#    check:  sacct -j <job id> --format=JobID,Elapsed,MaxRSS,State
# 2. all 34 layers (the tiny test's rows are reused: finished (layer, seed) rows are skipped)
sbatch --account=<account> --export=ALL scripts/carc/depth_sweep.slurm
# 3. tables + plot, after every array task succeeded
sbatch --account=<account> --export=ALL --dependency=afterok:<array job id> scripts/carc/depth_sweep_aggregate.slurm
```

Output goes to `/scratch1/<user>/depth_sweep/<pool>-<raw|trainnorm>-train_<horizon>-<probe>-<evaluator>/`:

- `config.json`;
- `rows/L<LL>_seed<S>.json`;
- `results.csv`, `summary.csv` and `depth_curve.png`.

To put it somewhere else, set `SWEEP_OUT` (one folder) or `SWEEP_OUT_ROOT`.

## Sizing (TEMPORARY probe)

Measured 2026-11-11 with a 2-layer run (3 seeds each) on a **full-size synthetic cache**: 4096-dim, 210 train and 140
seen_val rollouts, 86k train steps, float16. It ran with 4 threads on a laptop CPU (Apple M1).

| Per layer | Measured |
|---|---|
| Load train + seen_val (one layer, one pooling) | 5–12 s |
| Train, per seed | 3–5 s |
| Score seen_val, per seed | 0.3–1.2 s |
| Peak memory | 2.8–3.4 GB |

The job requests per array task are 4 CPUs, 8 GB and 30 min: about 2.5× the memory and well over 10× the time, to
allow for a slower CPU and for `/scratch1` reads. The whole array is 34 short tasks.

The real probe (B6) may cost much more, for example with a hyperparameter budget. When it lands, re-run the tiny
test and adjust `--mem` and `--time`.

The data files on CARC are real, but these timings come from synthetic data on a laptop. The tiny test in step 1 is
what confirms them on CARC.
