# Depth sweep (plan rows R1–R4): how to run it

Written 2026-11-11 for Akash, who runs the CARC jobs. This sweep is CPU only, and every submission needs Akash's OK.
The 2-layer test (layers 0-1) ran on CARC on 2026-11-11; the full run is next.

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

Run everything from the root of a Safe-Research checkout of `akash/extraction`. A separate clone is fine: the sweep
needs nothing from the collection kit's clone or its `config.sh`. It uses the clone's own
`collection/load_cache.py` and puts the clone's root first on `sys.path`, so `failure_prob` imports without an
install or `PYTHONPATH`.

The job runs the interpreter at `SWEEP_PYTHON` directly; no environment is activated. It needs torch, numpy, pandas
and scikit-learn (the collection kit's venv has them), plus matplotlib for the plot only. The default is the kit's
default venv (`/scratch1/<user>/safe_cache_work/venv310`); if yours lives elsewhere, export `SWEEP_PYTHON`.

```bash
mkdir -p logs                                    # in git, but Slurm needs it to exist before the job starts
export SWEEP_PYTHON=<path to venv310>/bin/python
export SWEEP_POOL=mean SWEEP_NORMALIZE=0         # CACHE_DIR defaults to /scratch1/<user>/safe_cache/v1
sbatch --account=<account> --export=ALL --array=0-1 scripts/carc/depth_sweep.slurm     # tiny test
sbatch --account=<account> --export=ALL --array=2-33 scripts/carc/depth_sweep.slurm    # the rest
sbatch --account=<account> --export=ALL --dependency=afterok:<array job id> scripts/carc/depth_sweep_aggregate.slurm
#   check:  sacct -j <job id> --format=JobID,Elapsed,MaxRSS,State
```

Output goes to `/scratch1/<user>/depth_sweep/<pool>-<raw|trainnorm>-train_<horizon>-<probe>-<evaluator>/`:

- `config.json`;
- `rows/L<LL>_seed<S>.json`;
- `results.csv`, `summary.csv` and `depth_curve.png`.

To put it somewhere else, set `SWEEP_OUT` (one folder) or `SWEEP_OUT_ROOT`. Every file and folder the sweep writes
is 0644 / 0755, and the aggregate job also opens up rows written before that rule existed. The team also needs
execute permission on each parent folder, starting with `/scratch1/<user>`.

**Logs.** Each task writes `logs/depth_sweep-<array job>_<task>.out` in the checkout and prints that path first.
Every line is timestamped, including lines before and after the imports and before and after the cache opens, so a
stall shows where it happened. To see where a running task is stuck, run `scancel --signal=USR1 <job>_<task>`,
which prints its Python stack into the log. A task that is cancelled or runs out of time also prints its stack, and
writes no further rows.

**Reruns reuse rows, and say so.** Each row ends with `"completed": true` and records the Slurm job, array task, log
path, commit and write time. A rerun reuses a row only when those fields are present, and logs each reuse:
`layer L seed S: reusing row from job J_T, commit C, written ...`. Rows from the first CARC test, written before
these fields existed, are reused and logged as legacy rows. Any other row is moved to `rows/_discarded/` and
recomputed. `--overwrite` moves rows aside the same way and never deletes them. Job IDs and log paths stay in
`rows/*.json` and are dropped from `results.csv`, so the tables can be shared without them.

**No bytecode writes.** Every CARC job sets `PYTHONDONTWRITEBYTECODE=1`, in this sweep and in the collection kit. In
the first test, two array tasks starting together hung on the same `__pycache__` file in the checkout on `/scratch1`.
The setting stops jobs writing `.pyc` files; they still read the venv's existing ones, so imports stay fast. A
per-job `PYTHONPYCACHEPREFIX` would recompile torch, pandas and scikit-learn in every job, about 3x slower to import.

## Sizing (TEMPORARY probe)

Measured on CARC on 2026-11-11 with the real cache, layers 0 and 1, 3 seeds each, 4 CPUs:

| Per task (one layer) | CARC, measured |
|---|---|
| Total | 39-46 s |
| Load train + seen_val (one layer, one pooling) | 2.4-13.5 s |
| Train, per seed | 4.3-9.6 s |
| Score seen_val, per seed | 0.2-1.2 s |
| Peak memory | about 4.1 GB |

Each array task requests 4 CPUs, 6 GB and 4 min: 1.5x the memory and about 5x the time. The time margin is there
because all 32 tasks of the full run start together and read `/scratch1` at once, so loading may be slower than with
2 tasks. A stuck task still ends within minutes. If a task runs out of time, rerun that layer alone, for example
`--array=17`; the finished layers are not touched.

Seeds are deterministic: on CARC, layer 0 seed 0 run twice gave identical metrics.

The real probe (B6) may cost much more, for example with a hyperparameter budget. When it lands, re-run the tiny
test and adjust `--mem` and `--time`.
