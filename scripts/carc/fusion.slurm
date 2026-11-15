#!/bin/bash
#SBATCH --job-name=fusion
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fusion-%j.out
# CPU only. Dhyan's rows (A9 check, F1, F2, F3, F6) on the shared cache; train + seen_val only, unseen_test is never
# read (on the shared cache it is permission-locked anyway). Submit from the root of the Safe-Research checkout;
# everything after the script name is passed through:
#
#   export SBATCH_ACCOUNT=<project account>   # the team's CARC project (sacctmgr show assoc user=$USER format=account)
#   export CACHE=/scratch1/akashrav/safe_cache/v1   # full path: the parent folder is pass-through only
#   mkdir -p logs
#   sbatch scripts/carc/fusion.slurm a9 --cache "$CACHE" --out "$TMPDIR/a9_fs" --layers 33
#   sbatch scripts/carc/fusion.slurm search --model concat_fusion --layers A B --cache "$CACHE" \
#          --output-dir outputs/fusion/F1 --label "F1 two-layer concat"
#   sbatch scripts/carc/fusion.slurm kcurve --model concat_fusion --layers A B C D --reference-layer R \
#          --cache "$CACHE" --output-dir outputs/fusion/F6
#   sbatch scripts/carc/fusion.slurm run ...       (scripts/run_fusion.py, one configuration)
#   sbatch scripts/carc/fusion.slurm compare ...   (scripts/fusion_compare.py)
#
# Out of time? Resubmit the same command with --resume: finished seeds and grid points are kept.
# Sizing: the runner prints its memory estimate before loading and refuses when it exceeds 90% of --mem. K <= 2 fits
# 16G, K = 3-4 needs about 32G (train at full horizon, float32). Override on the command line, e.g.
# `sbatch --mem=16G --time=01:00:00 scripts/carc/fusion.slurm ...`. Measure afterwards with
#   sacct -j <job id> --format=JobID,Elapsed,MaxRSS,State
# Settings: FUSION_PYTHON (default: the kit's venv, $WORK/venv310/bin/python as in collection/carc/config.sh, else the
# python3 on PATH of the submitting shell, e.g. an activated conda env; submit with --export=ALL, Slurm's default).
set -euo pipefail
die() { echo "fusion.slurm: $*" >&2; exit 1; }
[ -n "${SLURM_JOB_ID:-}" ] || die "submit with sbatch (CPU job); never run this on a login node"
ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }

cd "${SLURM_SUBMIT_DIR:?}"
[ -f scripts/fusion_experiments.py ] || die "submit from the root of the Safe-Research checkout"
[ $# -ge 1 ] || die "usage: sbatch scripts/carc/fusion.slurm {a9|search|kcurve|run|compare} <args>"
CMD="$1"; shift
case "$CMD" in
  a9)      SCRIPT=(scripts/a9_verify_export.py) ;;
  search)  SCRIPT=(scripts/fusion_experiments.py search) ;;
  kcurve)  SCRIPT=(scripts/fusion_experiments.py kcurve) ;;
  run)     SCRIPT=(scripts/run_fusion.py) ;;
  compare) SCRIPT=(scripts/fusion_compare.py) ;;
  *) die "unknown command $CMD (a9|search|kcurve|run|compare)" ;;
esac

umask 022                                                    # results are shared with the team
export PYTHONDONTWRITEBYTECODE=1                             # concurrent jobs once hung on a shared __pycache__
export CUDA_VISIBLE_DEVICES=""                               # CPU job: never touch a GPU
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
WORK_DEFAULT="/scratch1/${USER:-$(id -un)}/safe_cache_work"
PY="${FUSION_PYTHON:-${WORK:-$WORK_DEFAULT}/venv310/bin/python}"
if [ -z "${FUSION_PYTHON:-}" ] && [ ! -x "$PY" ]; then
  PY="$(command -v python3 || command -v python || true)"
  echo "note: no kit venv at ${WORK:-$WORK_DEFAULT}/venv310; using ${PY:-nothing} from PATH (set FUSION_PYTHON to choose)"
fi
[ -n "$PY" ] && [ -x "$PY" ] || die "no Python at '$PY' (set FUSION_PYTHON to an env with torch, numpy, pandas, scikit-learn)"
"$PY" -B -c "import torch, numpy, pandas, sklearn" || die "$PY cannot import torch/numpy/pandas/scikit-learn"

echo "[$(ts)] fusion.slurm $CMD: job $SLURM_JOB_ID, commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)," \
     "account ${SLURM_JOB_ACCOUNT:-?}, $SLURM_CPUS_PER_TASK CPUs, mem ${SLURM_MEM_PER_NODE:-?} MB, python $PY"
# Python in the background so a cancel / time-out (SIGTERM) reaches it at once; outputs are written atomically.
"$PY" -B "${SCRIPT[@]}" "$@" &
PY_PID=$!
trap 'echo "[$(ts)] got SIGTERM (cancelled or out of time): resubmit with --resume"; kill -TERM "$PY_PID" 2>/dev/null' TERM
rc=0
wait "$PY_PID" || rc=$?
[ "$rc" -gt 128 ] && { wait "$PY_PID" 2>/dev/null || true; }   # interrupted by the trap: let Python finish exiting
echo "[$(ts)] fusion.slurm $CMD finished with exit code $rc"
exit "$rc"
