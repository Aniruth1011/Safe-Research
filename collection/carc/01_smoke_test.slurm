#!/bin/bash
#SBATCH --job-name=safe-smoke
#SBATCH --partition=debug
#SBATCH -n 1
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Smoke test: runs ../smoke_test.sh (the same sequence Modal runs) with SMOKE_FULL=1, i.e. env checks, model load,
# 2 episodes with a resume check, the L32 capture check, an independent repeat + compare-runs, and a tiny cache
# build + verify. Timings and disk use land in $SMOKE_ROOT/timing_summary.txt for sizing the full run.
# GPU syntax: `-n 1 --gpus=<type>:1` (`--gpus-per-task` is rejected on CARC). #SBATCH lines cannot expand
# variables, so the type is written out here; it must equal GPU_TYPE in carc/config.sh (checked below). To use
# another type, override on the command line: sbatch --gpus=<type>:1 ...
# Submit from inside collection/ after editing carc/config.sh:
#   mkdir -p logs && sbatch --account=<account> carc/01_smoke_test.slurm
set -euo pipefail
# Never write .pyc files from a job: concurrent jobs once hung on a shared __pycache__ file on /scratch1.
# Existing .pyc files (the venv's, compiled at install) are still read, so imports stay fast.
export PYTHONDONTWRITEBYTECODE=1
source carc/config.sh
set +u; source carc/activate_env.sh; set -u
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi "$GPU_TYPE" \
  || { echo "GPU is not GPU_TYPE=$GPU_TYPE from carc/config.sh"; exit 1; }
export SMOKE_FULL="${SMOKE_FULL:-1}"
bash smoke_test.sh
