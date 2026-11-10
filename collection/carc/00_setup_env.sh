#!/bin/bash
# One-time setup. Run on a CARC LOGIN node (needs internet), from inside collection/:
#   bash carc/00_setup_env.sh
set -euo pipefail
KIT_SRC="$(cd "$(dirname "$0")/.." && pwd)"
source "$KIT_SRC/carc/config.sh"

mkdir -p "$WORK" "$HF_HOME"

module purge
module load conda 2>/dev/null || module load anaconda3 2>/dev/null || true   # whichever your CARC setup uses
eval "$(conda shell.bash hook)"

if ! conda env list | grep -q "^$CONDA_ENV "; then
  conda create -y -n "$CONDA_ENV" python=3.10
fi
conda activate "$CONDA_ENV"

# --- SAFE's OpenVLA fork, pinned and patched ---
if [ ! -d "$OPENVLA_DIR" ]; then
  git clone https://github.com/vla-safe/openvla.git "$OPENVLA_DIR"
fi
cd "$OPENVLA_DIR"
if ! grep -q save_all_layers experiments/robot/libero/run_libero_eval.py; then
  git apply --check "$KIT_DIR/openvla_all_layers.patch"
  git apply "$KIT_DIR/openvla_all_layers.patch"
fi
git rev-parse HEAD > "$WORK/openvla_commit.txt"

# Versions follow the OpenVLA LIBERO instructions. CUDA 12.1 wheels run on A40/A100/L40S.
pip install "torch==2.2.0" "torchvision==0.17.0" "torchaudio==2.2.0" --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install pandas scikit-learn
pip install packaging ninja
# flash-attn is optional: if this fails, add `--attn_implementation sdpa` in 02_rollouts.slurm
# (and keep that choice for ALL tasks: it changes numerics).
pip install "flash-attn==2.5.5" --no-build-isolation || echo ">>> flash-attn failed: use --attn_implementation sdpa"

# --- LIBERO ---
if [ ! -d "$LIBERO_DIR" ]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"
fi
(cd "$LIBERO_DIR" && pip install -e . && git rev-parse HEAD > "$WORK/libero_commit.txt")
pip install -r experiments/robot/libero/libero_requirements.txt

# LIBERO asks an interactive question on first import; answer it now so batch jobs don't hang.
echo ">>> If LIBERO asks about a custom dataset path, answer N."
python -c "from libero.libero import benchmark; print('LIBERO ok:', benchmark.get_benchmark_dict()['libero_10']().n_tasks, 'tasks')"

# --- Download and pin the checkpoint (~15 GB) ---
pip install -U "huggingface_hub[cli]"
huggingface-cli download openvla/openvla-7b-finetuned-libero-10
SNAP=$(ls -d "$HF_HOME"/hub/models--openvla--openvla-7b-finetuned-libero-10/snapshots/* | head -1)
echo "$SNAP" > "$WORK/checkpoint_path.txt"
echo ">>> Pinned checkpoint snapshot: $SNAP"

pip freeze > "$WORK/pip_freeze.txt"
echo ">>> Setup done. Next: sbatch carc/01_smoke_test.slurm"
