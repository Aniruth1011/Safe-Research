#!/bin/bash
#SBATCH --job-name=safe-pull-modal
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# CPU-only. Pulls finished tasks from a Modal Volume into /scratch1/<user>/rollouts/<run>/taskNN and verifies them
# (checksums, 50 episodes or listed invalid markers, A10, written by that profile). Never run on a login node.
#   sbatch --account=<account> --export=ALL,PROFILE=friend,TASKS=0-4 carc/04_pull_from_modal.slurm
# Needs the Modal CLI (MODAL_BIN or `modal` on PATH) with the profile configured. Nothing on Modal is deleted.
set -euo pipefail
source carc/config.sh
: "${PROFILE:?set PROFILE (e.g. friend or akashrav) with --export}"
: "${TASKS:?set TASKS (e.g. 0-4) with --export}"
bash carc/pull_from_modal.sh --profile "$PROFILE" --tasks "$TASKS" --run-note "$RUN_NOTE"
du -sh "${PULL_ROOT:-/scratch1/${USER:-$(id -un)}/rollouts}/$RUN_NOTE"
