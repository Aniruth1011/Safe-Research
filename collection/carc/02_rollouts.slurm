#!/bin/bash
#SBATCH --job-name=safe-rollouts
#SBATCH --partition=gpu
#SBATCH -n 1
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x-%A_%a.out
# One array element per LIBERO-10 task (50 episodes each), all on the same GPU type.
# The env is re-seeded per task (env.seed(0)) and episodes use LIBERO's fixed initial states, so splitting by task
# does not change what each episode sees. --resume True makes resubmission continue where a job stopped:
# an episode only counts as done once its .csv exists (written after .pkl and .layers.pt).
# GPU syntax: `-n 1 --gpus=<type>:1` (`--gpus-per-task` is rejected on CARC). #SBATCH lines cannot expand variables,
# so the type is written out; it must equal GPU_TYPE in carc/config.sh (checked below; override with sbatch --gpus=...).
#   mkdir -p logs && sbatch --account=<account> carc/02_rollouts.slurm
# Rerun only failed tasks:  sbatch --account=<account> --array=3,7 carc/02_rollouts.slurm
set -euo pipefail
source carc/config.sh
set +u; source carc/activate_env.sh; set -u
CKPT=$(cat "$WORK/checkpoint_path.txt")
TASK=${SLURM_ARRAY_TASK_ID}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi "$GPU_TYPE" \
  || { echo "GPU is not GPU_TYPE=$GPU_TYPE from carc/config.sh"; exit 1; }

cd "$OPENVLA_DIR"
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla --pretrained_checkpoint "$CKPT" \
  --task_suite_name libero_10 --center_crop True --attn_implementation flash_attention_2 \
  --output_hidden_states True --save_all_layers True --all_layers_pooling "$POOLING" \
  --task_start_index "$TASK" --task_end_index "$TASK" --num_trials_per_task 50 --resume True \
  --seed "$SEED" --use_wandb False --save_logs True \
  --save_root "$ROLLOUT_ROOT" --run_id_note "$RUN_NOTE"

DONE=$(ls "$ROLLOUT_ROOT/$RUN_NOTE/libero_10"/task${TASK}--ep*--succ*.csv 2>/dev/null | wc -l)
echo ">>> task $TASK: $DONE/50 episodes complete"
