# Depth sweep (plan rows R1–R4): how to run it

Written 2026-11-11 for Akash, who runs the CARC jobs. This sweep is CPU only, and every submission needs Akash's OK.
R1 (mean pooling, raw features) ran on CARC on 2026-11-11 with the TEMPORARY probe and evaluator: all 34 layers ×
3 seeds, 102 rows. Ten layers stalled reading `/scratch1` on the first attempt and were rerun throttled, with the
stalling nodes excluded. The fixes below come from that run.

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

- `failure_prob/engine/depth_sweep.py`: the sweep, the unseen lock, the sequential file reader, and aggregation.
- `failure_prob/engine/sweep_plugins.py`: the contracts, the TEMPORARY placeholders, and name resolution.
- `scripts/depth_sweep.py`: the command-line tool, with `run` and `aggregate`, and the start-up watchdog.
- `scripts/carc/submit_depth_sweep.sh`: submits the array (throttled, optional node exclusions) and the aggregate job.
- `scripts/carc/depth_sweep.slurm`: one array task per layer.
- `scripts/carc/depth_sweep_aggregate.slurm`: aggregation, after the array.
- `tests/test_depth_sweep.py` and `tests/synthetic_cache.py`: 38 synthetic tests. They cover:
  - the unseen lock;
  - the sequential reader returning exactly the memmap values, and its stall retries;
  - the watchdog, SIGTERM handling and the copying of the local log;
  - row reuse, temp-file cleanup and file modes;
  - the submit script and both Slurm scripts with a fake Slurm;
  - bootstrap CIs, reference lines and the plot layout.

  They pass under torch 2.14 (`.venv-kit310`) and under torch 2.2.0 with numpy 1.26 (the CARC kit env).

## On CARC (CPU jobs; Akash submits)

Run everything from a Safe-Research checkout of `akash/extraction`. A separate clone is fine: the sweep needs nothing
from the collection kit's clone or its `config.sh`. It uses the clone's own `collection/load_cache.py` and puts the
clone's root first on `sys.path`, so `failure_prob` imports without an install or `PYTHONPATH`. If you can, keep the
clone off `/scratch1` (for example in your home folder); then only the cache is read from `/scratch1`.

The job runs the interpreter at `SWEEP_PYTHON` directly; no environment is activated. It needs torch, numpy, pandas
and scikit-learn (the collection kit's venv has them), plus matplotlib for the plot only. The default is the kit's
default venv (`/scratch1/<user>/safe_cache_work/venv310`); if yours lives elsewhere, export `SWEEP_PYTHON`.

```bash
export SWEEP_PYTHON=<path to venv310>/bin/python
export SWEEP_POOL=mean SWEEP_NORMALIZE=0         # CACHE_DIR defaults to /scratch1/<user>/safe_cache/v1
scripts/carc/submit_depth_sweep.sh --account <account> --dry-run         # shows the two sbatch commands
scripts/carc/submit_depth_sweep.sh --account <account>                   # layers 0-33, 6 at a time
#   rerun some layers, avoiding bad nodes:
scripts/carc/submit_depth_sweep.sh --account <account> --layers 10,16 --max-parallel 3 --exclude <node>,<node>
#   check:  sacct -j <array job id> --format=JobID,Elapsed,MaxRSS,State,ExitCode,NodeList
```

- **Aggregation:** the aggregate job runs after the array whatever its outcome (`afterany`). It writes the tables and
  the plot from the rows that exist. If any (layer, seed) row is missing, it lists the missing ones with a ready
  `--layers` value and exits 4.
- **Exclusions:** `EXCLUDE_NODES` sets a default for `--exclude`.
- **Plain sbatch:** it works too. Give the limit yourself, as `--array=<layers>%6`, because a command-line `--array`
  replaces the script's default of `0-33%6`.

Output goes to `/scratch1/<user>/depth_sweep/<pool>-<raw|trainnorm>-train_<horizon>-<probe>-<evaluator>/`:

- `config.json`;
- `rows/L<LL>_seed<S>.json`;
- `results.csv`, `summary.csv` and `depth_curve.png`.

To put it somewhere else, set `SWEEP_OUT` (one folder) or `SWEEP_OUT_ROOT`. Every file and folder the sweep writes
is 0644 / 0755, and the aggregate job also opens up rows written before that rule existed. The team also needs
execute permission on each parent folder, starting with `/scratch1/<user>`.

**Reading the cache.** Each layer file (about 0.7 GB for train and 0.5 GB for seen_val with mean pooling) is read
into memory in one sequential pass of 64 MB chunks. It is no longer memory-mapped and sliced rollout by rollout,
which turned into many small reads. The values are identical, and a test checks that.
- **Progress:** MB read is logged every 5 s, then the total and MB/s.
- **Stalls:** if a chunk makes no progress for 30 s, it is retried on a new file handle, up to 2 times. After that
  the task exits with code 75.
- **Settings:** `SWEEP_READ_STALL_S`, `SWEEP_READ_RETRIES`, `SWEEP_READ_CHUNK_MB`.

**Start-up watchdog.** Start-up and loading must finish within `SWEEP_LOAD_TIMEOUT` (default 120 s). Start-up means
the imports and opening the cache; loading means reading the layer's data. A normal task needs 5-15 s for this. If
it isn't done in time, the task logs `WATCHDOG: ...` and the Python stack of every thread, then exits with code 75,
instead of waiting for the 4-minute limit. A second thread prints that stack, so it works even while the main thread
is stuck in uninterruptible I/O.

**Logs.** Each task logs to a file on the node's local disk and copies it into the Slurm log at
`logs/depth_sweep-<array job>_<task>.out` when it ends, also when cancelled or out of time. A stalled `/scratch1`
cannot leave the log half-written, and the first line of the Slurm log names the local file. If the checkout itself
is on `/scratch1`, the Slurm log is too, so a stall at the very start can still leave it empty. Keep the clone
elsewhere to avoid that.
- **Format:** every line is timestamped.
- **Stack of a running task:** `scancel --signal=USR1 <job>_<task>` prints it into the log.
- **Commits:** `logs/` is in git only through `logs/.gitkeep`; everything else in it is ignored, because logs contain
  job IDs and node names.

**Reruns reuse rows, and say so.** Each row ends with `"completed": true` and records the Slurm job, array task, log
path, commit and write time. A rerun reuses a row only when those fields are present, and logs each reuse:
`layer L seed S: reusing row from job J_T, commit C, written ...`.
- **Legacy rows:** rows from the first CARC test, written before these fields existed, are reused and logged as
  legacy rows.
- **Other rows:** moved to `rows/_discarded/` and recomputed. `--overwrite` moves rows aside the same way and never
  deletes them.
- **Temp files:** temp files left by tasks killed mid-write (`.config.json.<random>`, `.<random>`) are moved to
  `_discarded/tmp/` at the start of every run and aggregation, but only once they are 15 minutes old. A task that is
  writing one now is never disturbed.
- **Job IDs:** job IDs and log paths stay in `rows/*.json` and are dropped from `results.csv`, so the tables can be
  shared without them.
- **Compatibility:** `config.json` is unchanged by these fixes, and a test pins its keys. The finished rows are
  reused.

**No bytecode writes.** Every CARC job sets `PYTHONDONTWRITEBYTECODE=1`, in this sweep and in the collection kit. In
the first test, two array tasks starting together hung on the same `__pycache__` file in the checkout on `/scratch1`.
The setting stops jobs writing `.pyc` files; they still read the venv's existing ones, so imports stay fast. A
per-job `PYTHONPYCACHEPREFIX` would recompile torch, pandas and scikit-learn in every job, about 3x slower to import.

## Plot, CIs and reference lines

`depth_curve.png` shows the mean over seeds per layer with a band of ± one seed std, and bootstrap CI bars.

**Index 33 (raw block 32, `block32_prenorm`).** It is the same block as layer 32, read before the final RMSNorm,
so it is drawn as an orange diamond beside x = 32, not as a 34th layer. It has its own std and CI bars. A layer
named `<block>_prenorm` always goes next to its `<block>_postnorm` partner; 33-layer caches have none.

**Bootstrap CIs over seen_val episodes.** Each row also saves what its evaluator saw, as
`rows/L<LL>_seed<S>.scores.npz`: per-step scores, labels, horizons, task and trajectory IDs. At aggregation time,
for each layer:
- **Resampling:** episodes are drawn with replacement 1,000 times, separately among failures and successes, so
  both class counts stay fixed.
- **Paired seeds:** every seed is re-scored on the same resample by the folder's own evaluator, so B4's evaluator
  is used once it lands.
- **The CI:** the metrics are averaged over seeds, and the CI is the 95% percentile interval of that average: the
  uncertainty of the plotted point that comes from which episodes seen_val happens to contain.

`summary.csv` gets `val_<metric>_ci95_low`/`_high` for every metric, plus `bootstrap_resamples`. Rows written
before scores were saved, like the TEMPORARY R1 run, get empty CI columns, and the log says which layers.
Settings: `--bootstrap N` (0 = none) or `SWEEP_BOOTSTRAP`, and `--ci 0.95`.

The aggregate job runs the resampling on 4 CPUs (4 GB, 30 min). With the TEMPORARY evaluator the full sweep is
about 7 CPU-minutes. A slower B4 evaluator scales that linearly; lower `SWEEP_BOOTSTRAP` or raise `--time` if needed.

**Reference lines.** Use `--ref "label=value"` (repeatable), or `SWEEP_REFS="label=value;label=value"` for the
aggregate job. For example, SAFE's final-layer baseline from Adarsh's reproduction, and the all-layer average:
```bash
export SWEEP_REFS="SAFE final layer (reproduction)=<value>;All-layer average=<value>"
```
Values are for the plotted metric (`--metric`, default roc_auc). They are only comparable when computed on
seen_val, at the same horizon, with the same evaluator. They are stored per metric in `references.json`, so later
aggregations keep them without the flag.

## When /scratch1 stalls

See also runbook section 8 (`07_RUNBOOK_MODAL.md`). In the full run, about 30 tasks started together. Each read about
1 GB from `/scratch1`, and ten tasks stalled. A stalled task is in uninterruptible I/O (state D): its log stops after
`cache open`, or stays empty, and Slurm reports `JOB NOT ENDING WITH SIGNALS`. Such a process ignores every signal,
SIGKILL included, so it cannot print its own stack. The watchdog's stack and the node-local log are the evidence.
Stalls clustered on some nodes, where every task stalled, but load mattered too.

1. Find the stalled tasks and their nodes: `sacct -j <array job id> --format=JobID,State,ExitCode,NodeList`. Exit
   code 75 or TIMEOUT means a stall.
2. Rerun only those layers, throttled harder and avoiding those nodes:
   `submit_depth_sweep.sh --account <account> --layers <list> --max-parallel 3 --exclude <nodes>`. Finished rows are
   reused, and the new aggregate job runs afterwards.
3. If a node stalls again, tell CARC support (node name and time).

## Sizing (TEMPORARY probe)

Measured on CARC on 2026-11-11 with the real cache, 4 CPUs, before the sequential reader:

| Per task (one layer) | CARC, measured |
|---|---|
| Total | 39-50 s |
| Load train + seen_val (one layer, one pooling) | 2.4-13.5 s |
| Train, per seed | 4.3-9.6 s |
| Score seen_val, per seed | 0.2-1.2 s |
| Peak memory | about 4.1 GB |

Each array task requests 4 CPUs, 6 GB and 4 min. The sequential reader keeps one layer file (at most 0.7 GB) in
memory only until that split's rollouts are built, so peak memory should stay about the same. Check MaxRSS on the
next run.

Seeds are deterministic: on CARC, layer 0 seed 0 run twice gave identical metrics.

The real probe (B6) may cost much more, for example with a hyperparameter budget. When it lands, run 2 layers first
(`--layers 0-1`) and adjust `--mem` and `--time`.
