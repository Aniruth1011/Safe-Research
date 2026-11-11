#!/bin/bash
#SBATCH --job-name=safe-build-cache
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
# CPU-only. Refuses to run unless all 10 tasks x 50 episodes are complete.
#   sbatch --account=<account> --dependency=afterok:<rollout_array_job_id> carc/03_build_cache.slurm
set -euo pipefail
source carc/config.sh
set +u; source carc/activate_env.sh; set -u

mkdir -p "$CACHE_ROOT"
# Capture check on ALL 500 episodes before packaging (L32 must equal SAFE's own last-layer pkl)
python check_cache.py capture --rollout-dir "$ROLLOUT_ROOT/$RUN_NOTE/libero_10" | tail -5
python build_cache.py \
  --rollout-dir "$ROLLOUT_ROOT/$RUN_NOTE/libero_10" \
  --cache-root "$CACHE_ROOT" --version "$CACHE_VERSION" \
  --checkpoint "$(cat "$WORK/checkpoint_path.txt")" \
  --provenance-files "$WORK/pip_freeze.txt" "$WORK/openvla_commit.txt" "$WORK/libero_commit.txt" "$WORK/checkpoint_path.txt"

python check_cache.py verify --cache "$CACHE_ROOT/$CACHE_VERSION"
chmod -R g+rX "$CACHE_ROOT/$CACHE_VERSION"   # readable by your CARC project group
du -sh "$CACHE_ROOT/$CACHE_VERSION"
