#!/bin/bash
#SBATCH --job-name=depth-sweep-aggregate
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/depth_sweep_aggregate-%j.out
# CPU only. Collects the depth sweep's rows into results.csv and summary.csv and draws depth_curve.png.
# scripts/carc/submit_depth_sweep.sh submits it after the array (--dependency=afterany, so it runs even when some
# layers failed). By hand, from the root of the checkout with the SAME SWEEP_* variables as the sweep itself:
#   sbatch --account=<account> --export=ALL --dependency=afterany:<array job id> scripts/carc/depth_sweep_aggregate.slurm
# Bootstrap CIs (SWEEP_BOOTSTRAP resamples of seen_val episodes, default 1000; 0 = none) use all 4 CPUs. With the
# TEMPORARY evaluator, 34 layers x 3 seeds x 1000 resamples is about 7 CPU-minutes; a slower real evaluator (B4)
# scales that. Reference lines: SWEEP_REFS='SAFE final layer (reproduction)=0.81;All-layer average=0.79' (values of
# the plotted metric, roc_auc by default, on seen_val with the same evaluator); kept in references.json.
# It checks that every layer in SWEEP_EXPECT_LAYERS (default 0-33) has a row for every seed. If some are missing it
# still writes the (partial) tables, lists the missing layers and how to rerun them, and exits 4.
# If matplotlib is missing from the environment, the tables are still written and the job exits 3; draw the plot
# elsewhere with `python scripts/depth_sweep.py aggregate --out <copy of the folder>`.
set -euo pipefail
die() { echo "depth_sweep_aggregate.slurm: $*" >&2; exit 1; }
[ -n "${SLURM_JOB_ID:-}" ] || die "submit with sbatch (CPU job); never run this on a login node"
cd "${SLURM_SUBMIT_DIR:?}"
[ -f scripts/depth_sweep.py ] || die "submit from the root of the Safe-Research checkout"
umask 022                                                    # tables and plot are shared with the team
export PYTHONDONTWRITEBYTECODE=1                             # never write .pyc files into the shared checkout
export SWEEP_LOG="$PWD/logs/depth_sweep_aggregate-${SLURM_JOB_ID}.out"
chmod 644 "$SWEEP_LOG" 2>/dev/null || true
echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] depth_sweep_aggregate.slurm start: log $SWEEP_LOG"
PY="${SWEEP_PYTHON:-/scratch1/${USER:-$(id -un)}/safe_cache_work/venv310/bin/python}"
[ -x "$PY" ] || die "no Python at $PY (set SWEEP_PYTHON)"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # parallel over processes

ARGS=(aggregate --pool "${SWEEP_POOL:-mean}" --train-horizon "${SWEEP_TRAIN_HORIZON:-full}"
      --probe "${SWEEP_PROBE:-temporary_mlp}" --evaluator "${SWEEP_EVALUATOR:-temporary_max}")
[ "${SWEEP_NORMALIZE:-0}" = 1 ] && ARGS+=(--normalize)
[ -n "${SWEEP_OUT:-}" ] && ARGS+=(--out "$SWEEP_OUT")
ARGS+=(--expect-layers "${SWEEP_EXPECT_LAYERS:-0-33}")
"$PY" scripts/depth_sweep.py "${ARGS[@]}"
