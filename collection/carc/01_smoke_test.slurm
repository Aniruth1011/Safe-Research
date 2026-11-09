#!/bin/bash
#SBATCH --job-name=safe-smoke
#SBATCH --partition=debug
#SBATCH --gpus-per-task=a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Smoke test on task 0. Checks, in order:
#   1. RESUME: run 1 episode, then ask for 2 with --resume True -> episode 0 must NOT be rerun, episode 1 must appear.
#   2. CAPTURE: all-layer L32 must reproduce SAFE's own last-layer .pkl exactly (first/last token) for each episode.
#   3. REPEAT: an independent second run of the same 2 episodes, compared for outcome flips / action divergence.
#      (a sanity check, not a reproducibility guarantee)
#   4. BUILD: a tiny cache is built and verified; timings and disk use are printed for sizing the full run.
# Submit from inside collection/ after editing carc/config.sh:
#   mkdir -p logs && sbatch --account=<acct> carc/01_smoke_test.slurm
set -euo pipefail
source carc/config.sh
module purge; module load conda 2>/dev/null || module load anaconda3 2>/dev/null || true
eval "$(conda shell.bash hook)"; conda activate "$CONDA_ENV"
CKPT=$(cat "$WORK/checkpoint_path.txt")
SMOKE="$ROLLOUT_ROOT/smoke-$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

rollout () {  # $1 = run note, $2 = number of episodes
  (cd "$OPENVLA_DIR" && python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla --pretrained_checkpoint "$CKPT" \
    --task_suite_name libero_10 --center_crop True \
    --output_hidden_states True --save_all_layers True --all_layers_pooling "$POOLING" \
    --task_start_index 0 --task_end_index 0 --num_trials_per_task "$2" --resume True \
    --seed "$SEED" --use_wandb False --save_logs True \
    --save_root "$SMOKE" --run_id_note "$1")
}

A="$SMOKE/rep_a/libero_10"; B="$SMOKE/rep_b/libero_10"

echo "=== 1. RESUME ==="
T0=$(date +%s); rollout rep_a 1; T1=$(date +%s)
EP0=$(ls "$A"/task0--ep0--succ*.csv); STAMP0=$(stat -c %Y "$EP0")
rollout rep_a 2; T2=$(date +%s)
[ "$(stat -c %Y "$EP0")" = "$STAMP0" ] || { echo "RESUME FAIL: episode 0 was rerun"; exit 1; }
ls "$A"/task0--ep1--succ*.csv >/dev/null || { echo "RESUME FAIL: episode 1 missing"; exit 1; }
[ "$(ls "$A"/task0--ep*--succ*.csv | wc -l)" = 2 ] || { echo "RESUME FAIL: wrong episode count"; exit 1; }
echo "RESUME ok"

echo "=== 2. CAPTURE ==="
cd "$KIT_DIR"
python check_cache.py capture --rollout-dir "$A"

echo "=== 3. REPEAT ==="
T3=$(date +%s); rollout rep_b 2; T4=$(date +%s)
cd "$KIT_DIR"
python check_cache.py compare-runs --a "$A" --b "$B"

echo "=== 4. BUILD ==="
python build_cache.py --rollout-dir "$A" --cache-root "$SMOKE/cache" \
  --expect-tasks 1 --expect-episodes 2 --unseen-tasks 99 --allow-incomplete
python check_cache.py verify --cache "$SMOKE/cache/v1"

STEPS=$(python -c "import pandas as pd,glob; print(sum(len(pd.read_csv(f)) for f in glob.glob('$B/task0--ep*--succ*.csv')))")
BYTES=$(du -sb "$B" | cut -f1)
cat <<EOF | tee "$SMOKE/timing_summary.txt"
=== SUMMARY (paste this to the team) ===
GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)
first run, 1 episode (includes model load): $((T1 - T0)) s
resume run, 1 more episode:                 $((T2 - T1)) s
repeat run, 2 episodes (includes model load): $((T4 - T3)) s
steps in repeat run: $STEPS  -> ~$(python -c "print(round(($T4-$T3)/max($STEPS,1),3))") s/step upper bound
raw rollout bytes per step: ~$(python -c "print(round($BYTES/max($STEPS,1)/1e6,2))") MB
Full run sizing: steps_total ~= 500 x avg_episode_length (check episode lengths above; failures run to 520)
EOF
du -sh "$A" "$SMOKE/cache"
echo ">>> Smoke test passed. Set --time in 02_rollouts.slurm from the s/step figure above (x 50 episodes x ~520 steps x 1.5)."
