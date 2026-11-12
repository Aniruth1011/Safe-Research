#!/bin/bash
#SBATCH --job-name=safe-build-modal-cache
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
# CPU-only. Builds and verifies the cache from tasks pulled by 04_pull_from_modal.slurm.
#   sbatch --account=<account> carc/05_build_cache_from_modal.slurm
# Refuses unless all 10 task folders are present and the image provenance of every workspace used is identical.
set -euo pipefail
# Never write .pyc files from a job: concurrent jobs once hung on a shared __pycache__ file on /scratch1.
# Existing .pyc files (the venv's, compiled at install) are still read, so imports stay fast.
export PYTHONDONTWRITEBYTECODE=1
source carc/config.sh
set +u; source carc/activate_env.sh; set -u
echo "cache will be written to $CACHE_ROOT/$CACHE_VERSION"

RUN_DIR="${PULL_ROOT:-/scratch1/${USER:-$(id -un)}/rollouts}/$RUN_NOTE"
FLAT="$RUN_DIR/libero_10"          # one flat view (symlinks) of all task folders, the layout build_cache.py reads
for t in 00 01 02 03 04 05 06 07 08 09; do
  [ -d "$RUN_DIR/task$t/libero_10" ] || { echo "missing $RUN_DIR/task$t: pull it first (04_pull_from_modal.slurm)"; exit 1; }
done

# Every workspace that ran tasks must have built the identical image (same pip freeze, commits, builder version).
mapfile -t PROFILES < <(ls "$RUN_DIR/image_provenance")
for p in "${PROFILES[@]:1}"; do
  diff -r "$RUN_DIR/image_provenance/${PROFILES[0]}" "$RUN_DIR/image_provenance/$p" \
    || { echo "image provenance differs between workspaces ${PROFILES[0]} and $p"; exit 1; }
done

rm -rf "$FLAT" && mkdir -p "$FLAT/provenance"
for t in 00 01 02 03 04 05 06 07 08 09; do
  for f in "$RUN_DIR/task$t/libero_10"/task*; do ln -s "$f" "$FLAT/"; done
  for f in "$RUN_DIR/task$t/libero_10/provenance"/*; do [ -e "$f" ] && ln -s "$f" "$FLAT/provenance/"; done
  cp "$RUN_DIR/task$t/TASK_META.json" "$RUN_DIR/task_meta_task$t.json"
done

mkdir -p "$CACHE_ROOT"
python check_cache.py capture --rollout-dir "$FLAT" | tail -5
P0="$RUN_DIR/image_provenance/${PROFILES[0]}"
python build_cache.py \
  --rollout-dir "$FLAT" \
  --cache-root "$CACHE_ROOT" --version "$CACHE_VERSION" \
  --checkpoint "$(cat "$P0/checkpoint_path.txt")" \
  --provenance-files "$P0"/*.txt "$RUN_DIR"/task_meta_task*.json "$RUN_DIR/pull_log.tsv"

python check_cache.py verify --cache "$CACHE_ROOT/$CACHE_VERSION"
chmod -R g+rX "$CACHE_ROOT/$CACHE_VERSION"   # readable by your CARC project group
du -sh "$CACHE_ROOT/$CACHE_VERSION"
echo "Cache verified. Tell Akash before anything is deleted from any Modal Volume."
