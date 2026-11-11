# Edit these once. Every other script sources this file.
# Every script refuses to run while any value below still contains CHANGE_ME (see kit_check_config at the bottom).
# Write settings as `export NAME=value` lines: kit_check_config finds them by that form.
# ---------------------------------------------------------------------------------------------
# Your CARC Slurm account (the PI/project account you normally pass to --account)
export CARC_ACCOUNT="CHANGE_ME"                     # e.g. "smith_123"

# Where code, model weights and outputs live. /scratch1 is large and fast; /project2 is shareable with your group.
export WORK="/scratch1/${USER:-$(id -un)}/safe_cache_work"       # clones + Python env + HF cache
export ROLLOUT_ROOT="/scratch1/${USER:-$(id -un)}/safe_rollouts" # raw rollouts (~150-200 GB with all layers, 3 poolings)
# CARC project directories are /project2/<account>, NOT /project/<account> (seen with `myquota`, 2026-11-06).
# If `myquota` shows a different project path for your group, edit this line.
export CACHE_ROOT="/project2/$CARC_ACCOUNT/safe_cache"   # final versioned cache (same size again); share this
export CACHE_VERSION="v1"

# --- Python environment (pins and reasons: ../constraints.txt) ------------------------------
# "uv"    = plain virtualenv built with uv, no conda. This is the mode verified on CARC (A40, 2026-11-09).
# "conda" = conda env named $CONDA_ENV (the kit's original behaviour).
export ENV_MODE="uv"
export VENV_DIR="$WORK/venv310"                     # used when ENV_MODE=uv
export CONDA_ENV="safe-openvla"                     # used when ENV_MODE=conda
# uv mode only: full path of a Python 3.10 binary. `module load python` is unreliable in non-interactive
# shells on CARC (it once left a stale 3.11 on PATH), so give the binary itself, e.g. the path that
# `module show python/3.10.16` prints, plus /bin/python3.10. Leave empty to let uv find or download a 3.10.
export PYTHON_BIN=""
# CUDA toolkit root; $CUDA_HOME/bin/nvcc must exist. flash-attn cannot be built without it, and
# `module load cuda` does not set it in non-interactive shells. Use CARC's CUDA 12.4 toolkit (the one the
# verified build used): `module show cuda/12.4.0` prints its root, under /apps/spack/2406/.../cuda-12.4.0-<hash>.
# Setup checks $CUDA_HOME/bin/nvcc before installing anything.
export CUDA_HOME="${CUDA_HOME:-CHANGE_ME}"          # e.g. the root printed by `module show cuda/12.4.0`

# --- Pinned sources (do not change without telling the team: they define the dataset) ---------
export OPENVLA_REPO="https://github.com/vla-safe/openvla.git"
export OPENVLA_COMMIT="300dce26d44f407c725695d16cd445755c92cbd1"   # the commit openvla_all_layers.patch is made for
export OPENVLA_DIR="$WORK/openvla"                  # vla-safe/openvla with openvla_all_layers.patch applied
export LIBERO_REPO="https://github.com/Lifelong-Robot-Learning/LIBERO.git"
# LIBERO upstream master on 2026-11-10 (last commit dated 2025-03-15, so the CARC clone of 2026-11-06 was
# this commit too, but its hash was never written down there: owner to confirm).
export LIBERO_COMMIT="8f1084e3132a39270c3a13ebe37270a43ece2a01"
export LIBERO_DIR="$WORK/LIBERO"
export LIBERO_CONFIG_PATH="$WORK/libero_config"     # LIBERO's config.yaml lives here instead of ~/.libero
export CHECKPOINT_REPO="openvla/openvla-7b-finetuned-libero-10"
export PROCESSOR_CODE_REPO="openvla/openvla-7b"     # base repo: the processor's code files are loaded from it
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # this collection/ folder
export KIT_DIR
export HF_HOME="$WORK/hf"                           # keeps the ~15 GB checkpoint out of your home quota

# Pin ONE GPU type for every rollout job (smoke test included) so all 500 episodes use identical hardware.
export GPU_TYPE="a40"                               # A40 48 GB: also on the `debug` partition for the smoke test

export RUN_NOTE="all-layers-v1"                     # rollouts land in $ROLLOUT_ROOT/$RUN_NOTE/libero_10
export POOLING="mean,first,last"                    # use "mean" alone to cut storage ~3x (loses Exp 14)
export SEED=7                                       # SAFE / OpenVLA default

# Headless MuJoCo rendering on GPU nodes
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# ---------------------------------------------------------------------------------------------
# Guard used by 00_setup_env.sh and activate_env.sh. Returns non-zero (and says why) if a setting above
# is still a CHANGE_ME placeholder, is empty, or is malformed.
KIT_CONFIG_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

kit_check_config() {
  local name value bad=0
  if [ ! -r "$KIT_CONFIG_FILE" ]; then
    echo "config.sh: cannot read $KIT_CONFIG_FILE to check it." >&2
    return 1
  fi
  while read -r name; do
    value="${!name-}"
    case "$value" in
      *CHANGE_ME*)
        echo "config.sh: $name is still a CHANGE_ME placeholder ($value). Edit $KIT_CONFIG_FILE first." >&2
        bad=1 ;;
      "")
        if [ "$name" != "PYTHON_BIN" ]; then
          echo "config.sh: $name is empty. Set it in $KIT_CONFIG_FILE." >&2
          bad=1
        fi ;;
    esac
  done < <(sed -n 's/^export \([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' "$KIT_CONFIG_FILE")
  case "${ENV_MODE-}" in
    uv|conda) ;;
    *) echo "config.sh: ENV_MODE must be \"uv\" or \"conda\" (got \"${ENV_MODE-}\")." >&2; bad=1 ;;
  esac
  for name in OPENVLA_COMMIT LIBERO_COMMIT; do
    if ! [[ "${!name-}" =~ ^[0-9a-f]{40}$ ]]; then
      echo "config.sh: $name must be a full 40-character commit hash (got \"${!name-}\")." >&2
      bad=1
    fi
  done
  return "$bad"
}
