#!/bin/bash
# C1/C2: SAFE's original MLP (indep) / LSTM on the shared cache, SAFE's grid:
#   lr {1e-4,3e-4,1e-3} x lambda_reg {1e-3,1e-2,1e-1,1} x seed {0,1,2} = 36 array tasks.
# Submit from the repo root (or set REPO), after mkdir -p /scratch1/$USER/safe_baselines/slurm:
#   sbatch --export=ALL,MODEL=indep --job-name=c1-mlp  scripts/carc/safe_baseline.slurm
#   sbatch --export=ALL,MODEL=lstm  --job-name=c2-lstm --time=08:00:00 scripts/carc/safe_baseline.slurm
# Resubmit failed indices only: sbatch --array=7,19 --export=ALL,MODEL=indep ...
# Only train + seen_val are loaded (dataset=openvla_cache); unseen_test is never touched.
# Measured 2026-11-14 on 4 CPUs (b05-08): indep ~2.4 s/epoch -> ~45 min for 1000 epochs;
#   lstm ~19 s/epoch -> ~5.3 h for 1000 epochs (hence --time=08:00:00, or --partition=gpu --gres=gpu:1).
#SBATCH --account=yzhao010_1531
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-35
#SBATCH --output=/scratch1/%u/safe_baselines/slurm/%x-%A_%a.out
set -euo pipefail

module purge
module load python/3.10.16

export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline MPLBACKEND=Agg OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=/scratch1/$USER/tmp; mkdir -p "$TMPDIR"

MODEL=${MODEL:?set MODEL=indep or lstm}
CACHE_ROOT=${CACHE_ROOT:-/scratch1/akashrav/safe_cache/v1}
REPO=${REPO:-$SLURM_SUBMIT_DIR}
ENV_DIR=${ENV_DIR:-/scratch1/$USER/safe_env}
EXTRA_ARGS=${EXTRA_ARGS:-}      # e.g. "train.eval_save_ckpt=True" for the winners re-run

LRS=(1e-4 3e-4 1e-3); LAMS=(1e-3 1e-2 1e-1 1)
i=$SLURM_ARRAY_TASK_ID; SEED=$((i % 3)); LAM=${LAMS[$(((i / 3) % 4))]}; LR=${LRS[$((i / 12))]}
OUT=/scratch1/$USER/safe_baselines/$MODEL
mkdir -p "$OUT/wandb" "$OUT/logs"   # wandb falls back to a temp dir if wandb_dir does not exist

source "$ENV_DIR/bin/activate"
cd "$REPO"
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) model=$MODEL lr=$LR lambda_reg=$LAM seed=$SEED"

python -m failure_prob.train \
  dataset=openvla_cache dataset.cache_root="$CACHE_ROOT" dataset.layer=32 dataset.token_idx_rel=mean \
  model="$MODEL" model.batch_size=64 model.lr="$LR" model.lambda_reg="$LAM" \
  train.seed="$SEED" train.wandb_group_name="c_safe_$MODEL" train.exp_suffix="${MODEL}_lr${LR}_lam${LAM}_s${SEED}" \
  train.wandb_dir="$OUT/wandb" train.logs_save_root="$OUT/logs" \
  hydra.run.dir="$OUT/hydra/lr${LR}_lam${LAM}_s${SEED}" $EXTRA_ARGS
