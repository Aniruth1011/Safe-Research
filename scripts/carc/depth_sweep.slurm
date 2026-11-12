#!/bin/bash
#SBATCH --job-name=depth-sweep
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=00:04:00
#SBATCH --array=0-33
#SBATCH --output=logs/depth_sweep-%A_%a.out
# CPU only. Depth sweep (plan rows R1-R4): one array task per cache layer, each trains the probe for every seed in
# failure_prob/protocol.py (0, 1, 2) on train and scores seen_val. unseen_test is never read.
#
# Submit from the root of the Safe-Research checkout; logs go to its logs/ folder (in git, so it exists; Slurm does not
# create it). Set the variables in the shell and pass --export=ALL
# (sbatch --export splits its value on commas):
#   export SWEEP_POOL=mean SWEEP_NORMALIZE=0          # R1; R4 = SWEEP_NORMALIZE=1; R3 = SWEEP_POOL=first|last
#   sbatch --account=<account> --export=ALL scripts/carc/depth_sweep.slurm
#   # the tiny test on 2 layers instead of the array:
#   sbatch --account=<account> --export=ALL --array=0-1 scripts/carc/depth_sweep.slurm
# Then, once every array task has finished:
#   sbatch --account=<account> --export=ALL --dependency=afterok:<array job id> scripts/carc/depth_sweep_aggregate.slurm
#
# Optional: CACHE_DIR (default /scratch1/$USER/safe_cache/v1), SWEEP_OUT_ROOT (default /scratch1/$USER/depth_sweep),
# SWEEP_PYTHON (default: the collection kit's venv, /scratch1/$USER/safe_cache_work/venv310/bin/python),
# SWEEP_PROBE / SWEEP_EVALUATOR (default: the TEMPORARY placeholders), SWEEP_TRAIN_HORIZON (full | task_min_step).
#
# Sizing (TEMPORARY probe), measured on CARC 2026-11-11 with the real cache, layers 0-1, 4 CPUs: 39-46 s per task
# (load 2.4-13.5 s, fit 4.3-9.6 s per seed, score 0.2-1.2 s per seed), peak memory about 4.1 GB. Requests: 6 GB
# (1.5x) and 4 min (about 5x): 32 tasks starting together all read /scratch1 at once, so loading may be slower than
# in the 2-task test, and a stuck task still dies within minutes. Re-measure with --array=0-1 when the real probe
# (plan row B6) lands.
set -euo pipefail
die() { echo "depth_sweep.slurm: $*" >&2; exit 1; }
[ -n "${SLURM_JOB_ID:-}" ] || die "submit with sbatch (CPU job); never run this on a login node"
cd "${SLURM_SUBMIT_DIR:?}"
[ -f scripts/depth_sweep.py ] || die "submit from the root of the Safe-Research checkout"
umask 022                                                    # results and logs are shared with the team
# Never write .pyc files: concurrent array tasks once hung on a shared __pycache__ file in the checkout on /scratch1.
# Existing .pyc files (the venv's, compiled at install) are still read, so imports stay fast.
export PYTHONDONTWRITEBYTECODE=1
ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }
export SWEEP_LOG="$PWD/logs/depth_sweep-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.out"
chmod 644 "$SWEEP_LOG" 2>/dev/null || true
echo "[$(ts)] depth_sweep.slurm start: log $SWEEP_LOG"

PY="${SWEEP_PYTHON:-/scratch1/${USER:-$(id -un)}/safe_cache_work/venv310/bin/python}"
[ -x "$PY" ] || die "no Python at $PY (set SWEEP_PYTHON to an env with torch, numpy, pandas, scikit-learn)"
echo "[$(ts)] checking that $PY imports torch, numpy, pandas, scikit-learn"
"$PY" -c "import torch, numpy, pandas, sklearn" || die "$PY lacks torch/numpy/pandas/scikit-learn"
echo "[$(ts)] imports ok"

export CUDA_VISIBLE_DEVICES=""                               # CPU job: never touch a GPU
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
LAYER="${SLURM_ARRAY_TASK_ID:?submit as an array job (--array=0-33, or --array=0-1 for the tiny test)}"
ARGS=(run --layers "$LAYER" --pool "${SWEEP_POOL:-mean}" --train-horizon "${SWEEP_TRAIN_HORIZON:-full}"
      --probe "${SWEEP_PROBE:-temporary_mlp}" --evaluator "${SWEEP_EVALUATOR:-temporary_max}"
      --threads "${SLURM_CPUS_PER_TASK:-4}")
[ "${SWEEP_NORMALIZE:-0}" = 1 ] && ARGS+=(--normalize)
[ -n "${SWEEP_OUT:-}" ] && ARGS+=(--out "$SWEEP_OUT")

echo "[$(ts)] layer $LAYER, commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown), python $PY"
"$PY" scripts/depth_sweep.py "${ARGS[@]}"
# Measured time and memory of this task afterwards: sacct -j <job id> --format=JobID,Elapsed,MaxRSS,State
