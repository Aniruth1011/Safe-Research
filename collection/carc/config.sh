# Edit these once. Every other script sources this file.
# ---------------------------------------------------------------------------------------------
# Your CARC Slurm account (the PI/project account you normally pass to --account)
export CARC_ACCOUNT="CHANGE_ME"                     # e.g. "smith_123"

# Where code, model weights and outputs live. /scratch1 is large and fast; /project is shareable with your group.
export WORK="/scratch1/$USER/safe_cache_work"       # clones + conda env + HF cache
export ROLLOUT_ROOT="/scratch1/$USER/safe_rollouts" # raw rollouts (~150-200 GB with all layers, 3 poolings)
export CACHE_ROOT="/project/CHANGE_ME/safe_cache"   # final versioned cache (same size again); share this
export CACHE_VERSION="v1"

export CONDA_ENV="safe-openvla"
export OPENVLA_DIR="$WORK/openvla"                  # vla-safe/openvla with openvla_all_layers.patch applied
export LIBERO_DIR="$WORK/LIBERO"
export KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # this collection/ folder
export HF_HOME="$WORK/hf"                           # keeps the ~15 GB checkpoint out of your home quota

# Pin ONE GPU type for every rollout job (smoke test included) so all 500 episodes use identical hardware.
export GPU_TYPE="a40"                               # A40 48 GB: also on the `debug` partition for the smoke test

export RUN_NOTE="all-layers-v1"                     # rollouts land in $ROLLOUT_ROOT/$RUN_NOTE/libero_10
export POOLING="mean,first,last"                    # use "mean" alone to cut storage ~3x (loses Exp 14)
export SEED=7                                       # SAFE / OpenVLA default

# Headless MuJoCo rendering on GPU nodes
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
