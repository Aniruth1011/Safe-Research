#!/bin/bash
#SBATCH --job-name=depth-sweep
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=00:04:00
#SBATCH --array=0-33%6
#SBATCH --output=logs/depth_sweep-%A_%a.out
# CPU only. Depth sweep (plan rows R1-R4): one array task per cache layer, each trains the probe for every seed in
# failure_prob/protocol.py (0, 1, 2) on train and scores seen_val. unseen_test is never read.
#
# Submit with scripts/carc/submit_depth_sweep.sh: it sets the layers, the number of tasks running at once (default 6;
# about 30 tasks reading /scratch1 together stalled some of them), optional node exclusions, and chains the
# aggregate job. A plain sbatch also works; then always give the limit yourself, as --array=<layers>%6, because a
# command-line --array replaces the default above.
#
# Optional settings (export them and submit with --export=ALL): SWEEP_POOL, SWEEP_NORMALIZE, SWEEP_TRAIN_HORIZON,
# SWEEP_PROBE, SWEEP_EVALUATOR, SWEEP_OUT, SWEEP_OUT_ROOT, CACHE_DIR (default /scratch1/$USER/safe_cache/v1),
# SWEEP_PYTHON (default /scratch1/$USER/safe_cache_work/venv310/bin/python),
# SWEEP_LOAD_TIMEOUT (default 120 s: start-up and loading must finish by then, or the task exits with code 75),
# SWEEP_READ_STALL_S / SWEEP_READ_RETRIES / SWEEP_READ_CHUNK_MB (default 30 s / 2 / 64 MB, see read_npy_sequential).
#
# Logging: the task logs to a file on the node's local disk ($TMPDIR or /tmp) and copies it into the Slurm log
# logs/depth_sweep-<array job>_<task>.out at the end, also when cancelled or out of time, so a stalled /scratch1 cannot
# leave the log empty. A process stuck in uninterruptible I/O (state D) ignores every signal, even SIGKILL, so its
# own stack cannot be printed; the start-up watchdog prints it from a second thread instead.
#
# Sizing (TEMPORARY probe), measured on CARC 2026-11-11 with the real cache, 4 CPUs: 39-50 s per task, peak about
# 4.1 GB. Requests: 6 GB and 4 min. Re-measure with a 2-layer run when the real probe (plan row B6) lands.
set -euo pipefail
die() { echo "depth_sweep.slurm: $*" >&2; exit 1; }
[ -n "${SLURM_JOB_ID:-}" ] || die "submit with sbatch (CPU job); never run this on a login node"
ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }

# --- log to the node's local disk first; copy into the Slurm log (fd 3, in logs/ on the checkout's disk) at exit ----
TASK="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export SWEEP_LOG="${SLURM_SUBMIT_DIR:?}/logs/depth_sweep-$TASK.out"
LOCAL_LOG="$(mktemp "${TMPDIR:-/tmp}/depth_sweep-$TASK.XXXXXX.log")"
exec 3>&1
echo "[$(ts)] depth_sweep.slurm start; logging to node-local $LOCAL_LOG, copied here when the task ends" >&3
exec >"$LOCAL_LOG" 2>&1
PY_PID=""
copied=0
copy_log() {
  [ "$copied" = 1 ] && return 0
  copied=1
  cat "$LOCAL_LOG" >&3 || true
  rm -f "$LOCAL_LOG"
  chmod 644 "$SWEEP_LOG" 2>/dev/null || true
}
on_term() {
  echo "[$(ts)] depth_sweep.slurm: got SIGTERM (cancelled or out of time); copying the log and exiting"
  [ -n "$PY_PID" ] && kill -TERM "$PY_PID" 2>/dev/null
  sleep 2                                                     # let Python print its stack if it can
  copy_log
  exit 143
}
trap copy_log EXIT
trap on_term TERM
echo "[$(ts)] depth_sweep.slurm start: task $TASK, log $SWEEP_LOG (node-local copy $LOCAL_LOG)"

cd "$SLURM_SUBMIT_DIR"
[ -f scripts/depth_sweep.py ] || die "submit from the root of the Safe-Research checkout"
umask 022                                                    # results and logs are shared with the team
# Never write .pyc files: concurrent array tasks once hung on a shared __pycache__ file in the checkout on /scratch1.
# Existing .pyc files (the venv's, compiled at install) are still read, so imports stay fast.
export PYTHONDONTWRITEBYTECODE=1

PY="${SWEEP_PYTHON:-/scratch1/${USER:-$(id -un)}/safe_cache_work/venv310/bin/python}"
[ -x "$PY" ] || die "no Python at $PY (set SWEEP_PYTHON to an env with torch, numpy, pandas, scikit-learn)"
echo "[$(ts)] checking that $PY imports torch, numpy, pandas, scikit-learn"
# -I: do not search the current folder (on /scratch1) for modules; -B: no bytecode
LIMIT=(); command -v timeout >/dev/null && LIMIT=(timeout -k 10 "${SWEEP_LOAD_TIMEOUT:-120}")
${LIMIT[@]+"${LIMIT[@]}"} "$PY" -I -B -c "import torch, numpy, pandas, sklearn" \
  || die "$PY could not import torch/numpy/pandas/scikit-learn within ${SWEEP_LOAD_TIMEOUT:-120} s"
echo "[$(ts)] imports ok"

export CUDA_VISIBLE_DEVICES=""                               # CPU job: never touch a GPU
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
LAYER="${SLURM_ARRAY_TASK_ID:?submit as an array job (see scripts/carc/submit_depth_sweep.sh)}"
ARGS=(run --layers "$LAYER" --pool "${SWEEP_POOL:-mean}" --train-horizon "${SWEEP_TRAIN_HORIZON:-full}"
      --probe "${SWEEP_PROBE:-temporary_mlp}" --evaluator "${SWEEP_EVALUATOR:-temporary_max}"
      --threads "${SLURM_CPUS_PER_TASK:-4}")
[ "${SWEEP_NORMALIZE:-0}" = 1 ] && ARGS+=(--normalize)
[ -n "${SWEEP_OUT:-}" ] && ARGS+=(--out "$SWEEP_OUT")

echo "[$(ts)] layer $LAYER, commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown), python $PY"
# In the background, so the TERM trap runs at once instead of after Python exits (bash defers traps otherwise).
"$PY" scripts/depth_sweep.py "${ARGS[@]}" &
PY_PID=$!
rc=0
wait "$PY_PID" || rc=$?
PY_PID=""
case $rc in
  0)  echo "[$(ts)] layer $LAYER done" ;;
  75) echo "[$(ts)] layer $LAYER: start-up or a read from the file system stalled (exit 75); rerun this layer," \
           "maybe excluding this node ($(hostname -s 2>/dev/null || echo unknown))" ;;
  *)  echo "[$(ts)] layer $LAYER failed with exit code $rc" ;;
esac
# Measured time and memory of this task afterwards: sacct -j <job id> --format=JobID,Elapsed,MaxRSS,State,NodeList
exit "$rc"
