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
#   export PROFILE=friend TASKS=0,1,3,4 MODAL_BIN=<path to your modal CLI>
#   sbatch --account=<account> --export=ALL carc/04_pull_from_modal.slurm
# Set the variables in the shell and pass only --export=ALL: sbatch --export splits its value on commas, so
# --export=ALL,TASKS=0,1,3,4 would deliver only TASKS=0.
# Needs the Modal CLI (MODAL_BIN or `modal` on PATH) with the profile configured. Nothing on Modal is deleted.
set -euo pipefail
# Never write .pyc files from a job: concurrent jobs once hung on a shared __pycache__ file on /scratch1.
# Existing .pyc files (the venv's, compiled at install) are still read, so imports stay fast.
export PYTHONDONTWRITEBYTECODE=1
source carc/config.sh
: "${PROFILE:?set PROFILE (e.g. friend or akashrav) with --export}"
: "${TASKS:?set TASKS (e.g. 0-4) with --export}"
bash carc/pull_from_modal.sh --profile "$PROFILE" --tasks "$TASKS" --run-note "$RUN_NOTE"
du -sh "${PULL_ROOT:-/scratch1/${USER:-$(id -un)}/rollouts}/$RUN_NOTE"
