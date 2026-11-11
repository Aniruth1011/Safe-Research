#!/bin/bash
#SBATCH --job-name=depth-sweep-aggregate
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=depth_sweep_aggregate-%j.out
# CPU only. Collects the depth sweep's rows into results.csv and summary.csv and draws depth_curve.png.
# Submit from the root of the Safe-Research checkout with the SAME SWEEP_* variables as the sweep itself:
#   sbatch --account=<account> --export=ALL --dependency=afterok:<array job id> scripts/carc/depth_sweep_aggregate.slurm
# If matplotlib is missing from the environment, the tables are still written and the job exits 3; draw the plot
# elsewhere with `python scripts/depth_sweep.py aggregate --out <copy of the folder>`.
set -euo pipefail
die() { echo "depth_sweep_aggregate.slurm: $*" >&2; exit 1; }
[ -n "${SLURM_JOB_ID:-}" ] || die "submit with sbatch (CPU job); never run this on a login node"
cd "${SLURM_SUBMIT_DIR:?}"
[ -f scripts/depth_sweep.py ] || die "submit from the root of the Safe-Research checkout"
PY="${SWEEP_PYTHON:-/scratch1/${USER:-$(id -un)}/safe_cache_work/venv310/bin/python}"
[ -x "$PY" ] || die "no Python at $PY (set SWEEP_PYTHON)"
export CUDA_VISIBLE_DEVICES=""

ARGS=(aggregate --pool "${SWEEP_POOL:-mean}" --train-horizon "${SWEEP_TRAIN_HORIZON:-full}"
      --probe "${SWEEP_PROBE:-temporary_mlp}" --evaluator "${SWEEP_EVALUATOR:-temporary_max}")
[ "${SWEEP_NORMALIZE:-0}" = 1 ] && ARGS+=(--normalize)
[ -n "${SWEEP_OUT:-}" ] && ARGS+=(--out "$SWEEP_OUT")
"$PY" scripts/depth_sweep.py "${ARGS[@]}"
