#!/bin/bash
# C1-C3: SAFE's original models on the shared cache (dataset=openvla_cache: train + seen_val only,
# unseen_test is never touched). Grids are SAFE's own (scripts/batch_training/submit_openvla_libero.bash).
#
#   C1 MODEL=indep  lr {1e-4,3e-4,1e-3} x lambda_reg {1e-3,1e-2,1e-1,1} x seed {0,1,2}   array 0-35
#                   index = 12*lr_idx + 3*lam_idx + seed
#   C2 MODEL=lstm   same grid and index                                                  array 0-35
#   C3 MODEL=embed  cosine k-NN, topk {1,5,10} x cumsum {False,True}, seeds 0-1-2 in one run   array 0-5
#                   index = 2*topk_idx + cumsum_idx
#
# Submit from the repo root (or set REPO), after mkdir -p /scratch1/$USER/safe_baselines/slurm:
#   sbatch --export=ALL,MODEL=indep --job-name=c1-mlp  scripts/carc/safe_baseline.slurm
#   sbatch --export=ALL,MODEL=lstm  --job-name=c2-lstm --time=08:00:00 scripts/carc/safe_baseline.slurm
#   sbatch --export=ALL,MODEL=embed --job-name=c3-knn  --array=0-5 --cpus-per-task=16 --mem=64G --time=<measured> \
#          scripts/carc/safe_baseline.slurm
# Resubmit failed indices only: sbatch --array=7,19 --export=ALL,MODEL=indep ...
# Winners re-run with checkpoints (separate OUT, so the selection folder never gets duplicate runs):
#   sbatch --array=a,b,c --export=ALL,MODEL=indep,OUT=/scratch1/$USER/safe_baselines/winners/indep,EXTRA_ARGS=train.eval_save_ckpt=True ...
#
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

MODEL=${MODEL:?set MODEL=indep, lstm or embed}
CACHE_ROOT=${CACHE_ROOT:-/scratch1/akashrav/safe_cache/v1}
REPO=${REPO:-$SLURM_SUBMIT_DIR}
ENV_DIR=${ENV_DIR:-/scratch1/$USER/safe_env}
OUT=${OUT:-/scratch1/$USER/safe_baselines/$MODEL}
EXTRA_ARGS=${EXTRA_ARGS:-}      # e.g. "train.eval_save_ckpt=True" for the winners re-run
i=$SLURM_ARRAY_TASK_ID

case "$MODEL" in
  indep|lstm)
    LRS=(1e-4 3e-4 1e-3); LAMS=(1e-3 1e-2 1e-1 1)
    (( i < 36 )) || { echo "index $i out of range 0-35 for $MODEL" >&2; exit 2; }
    SEED=$((i % 3)); LAM=${LAMS[$(((i / 3) % 4))]}; LR=${LRS[$((i / 12))]}
    TAG="lr${LR}_lam${LAM}_s${SEED}"
    MODEL_ARGS=(model.batch_size=64 model.lr="$LR" model.lambda_reg="$LAM")
    ;;
  embed)
    TOPKS=(1 5 10); CUMSUMS=(False True)
    (( i < 6 )) || { echo "index $i out of range 0-5 for embed" >&2; exit 2; }
    TOPK=${TOPKS[$((i / 2))]}; CUMSUM=${CUMSUMS[$((i % 2))]}
    SEED=0-1-2                     # k-NN is deterministic; train.py loops over the seeds in one process
    TAG="topk${TOPK}_cumsum${CUMSUM}"
    MODEL_ARGS=(model.n_epochs=1 model.distance=cosine model.use_success_only=False
                model.topk="$TOPK" model.cumsum="$CUMSUM")
    ;;
  *) echo "MODEL must be indep, lstm or embed, got '$MODEL'" >&2; exit 2 ;;
esac

mkdir -p "$OUT/wandb" "$OUT/logs"   # wandb falls back to a temp dir if wandb_dir does not exist
source "$ENV_DIR/bin/activate"
cd "$REPO"
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) model=$MODEL $TAG seed=$SEED out=$OUT"

python -m failure_prob.train \
  dataset=openvla_cache dataset.cache_root="$CACHE_ROOT" dataset.layer=32 dataset.token_idx_rel=mean \
  model="$MODEL" "${MODEL_ARGS[@]}" \
  train.seed="$SEED" train.wandb_group_name="c_safe_$MODEL" train.exp_suffix="${MODEL}_${TAG}" \
  train.wandb_dir="$OUT/wandb" train.logs_save_root="$OUT/logs" \
  hydra.run.dir="$OUT/hydra/$TAG" $EXTRA_ARGS
