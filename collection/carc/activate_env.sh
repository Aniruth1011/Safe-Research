#!/bin/bash
# Activate the collection environment. SOURCE this file (do not execute it), after config.sh:
#   source carc/config.sh
#   source carc/activate_env.sh
# It activates the uv virtualenv ($VENV_DIR) or the conda env ($CONDA_ENV) chosen by ENV_MODE, and exports
# what every job needs: CUDA_HOME (+ PATH), HF_HOME, HF_HUB_OFFLINE, MUJOCO_GL / PYOPENGL_PLATFORM and
# LIBERO_CONFIG_PATH. It stops with a message if config.sh is unfinished or the environment does not exist.
# Safe under `set -euo pipefail`. No `module load` is needed in uv mode.
# shellcheck shell=bash

_kit_err() { echo "activate_env.sh: ERROR: $*" >&2; }

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  _kit_err "this file must be sourced, not executed: 'source carc/activate_env.sh' (after carc/config.sh)."
  exit 2
fi

if ! declare -F kit_check_config >/dev/null 2>&1; then
  _kit_err "config.sh is not loaded. Run 'source carc/config.sh' before 'source carc/activate_env.sh'."
  unset -f _kit_err
  return 1 2>/dev/null || exit 1
fi
if ! kit_check_config; then
  _kit_err "fix the settings listed above, then try again."
  unset -f _kit_err
  return 1 2>/dev/null || exit 1
fi

# conda's and virtualenv's activation code may read unset variables, so `set -u` is switched off around
# those calls only and restored afterwards.
case $- in *u*) _kit_nounset=1 ;; *) _kit_nounset=0 ;; esac
_kit_restore_nounset() { if [ "$_kit_nounset" = 1 ]; then set -u; fi; }
_kit_ok=1

# CUDA first, so the environment's own bin/ ends up in front of it on PATH.
export CUDA_HOME
case ":$PATH:" in
  *":$CUDA_HOME/bin:"*) ;;
  *) export PATH="$CUDA_HOME/bin:$PATH" ;;
esac

case "$ENV_MODE" in
  uv)
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
      _kit_err "no uv virtualenv at VENV_DIR=$VENV_DIR (missing bin/activate)." \
               "Build it once with: bash carc/00_setup_env.sh"
      _kit_ok=0
    else
      set +u
      # shellcheck disable=SC1091
      source "$VENV_DIR/bin/activate"
      _kit_restore_nounset
    fi
    ;;
  conda)
    set +u
    if command -v module >/dev/null 2>&1; then
      module purge || true
      module load conda 2>/dev/null || module load anaconda3 2>/dev/null || true   # whichever your CARC setup uses
    fi
    if ! command -v conda >/dev/null 2>&1; then
      _kit_err "ENV_MODE=conda but no 'conda' command was found (after trying 'module load conda')." \
               "Use ENV_MODE=uv in carc/config.sh, or make conda available."
      _kit_ok=0
    else
      eval "$(conda shell.bash hook)"
      _kit_envs="$(conda env list)"
      if ! awk -v n="$CONDA_ENV" '$1 == n { found = 1 } END { exit !found }' <<<"$_kit_envs"; then
        _kit_err "no conda env named CONDA_ENV=$CONDA_ENV. Build it once with: bash carc/00_setup_env.sh"
        _kit_ok=0
      else
        conda activate "$CONDA_ENV"
      fi
      unset _kit_envs
    fi
    _kit_restore_nounset
    ;;
  *)
    _kit_err "unknown ENV_MODE='${ENV_MODE-}'. Set ENV_MODE to \"uv\" or \"conda\" in carc/config.sh."
    _kit_ok=0
    ;;
esac

if [ "$_kit_ok" = 1 ]; then
  # Model files: read only from the pinned local cache. 00_setup_env.sh downloads everything, so jobs never
  # fetch a newer copy of the processor code by accident. Set HF_HUB_OFFLINE=0 before sourcing to override.
  export HF_HOME
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

  # Rendering: EGL only. CARC has no libOSMesa, and a different backend changes the pixels the policy sees,
  # so one dataset must never mix backends.
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
  if [ "$MUJOCO_GL" != "egl" ] || [ "$PYOPENGL_PLATFORM" != "egl" ]; then
    _kit_err "MUJOCO_GL and PYOPENGL_PLATFORM must both be 'egl' (got '$MUJOCO_GL' and '$PYOPENGL_PLATFORM')."
    _kit_ok=0
  fi

  # LIBERO reads its config from here. Without the file it asks a question on first import, which a batch
  # job cannot answer. 00_setup_env.sh writes it (and sets KIT_SETUP=1 so this warning stays quiet meanwhile).
  export LIBERO_CONFIG_PATH
  if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ] && [ "${KIT_SETUP:-0}" != 1 ]; then
    echo "activate_env.sh: WARNING: $LIBERO_CONFIG_PATH/config.yaml is missing; LIBERO will stop to ask a" \
         "question on first import. Run: bash carc/00_setup_env.sh" >&2
  fi
fi

if [ "$_kit_ok" = 1 ]; then
  echo "activate_env.sh: ENV_MODE=$ENV_MODE python=$(command -v python || echo '<none>') CUDA_HOME=$CUDA_HOME"
  unset -f _kit_err _kit_restore_nounset
  unset _kit_ok _kit_nounset
else
  unset -f _kit_err _kit_restore_nounset
  unset _kit_ok _kit_nounset
  return 1 2>/dev/null || exit 1
fi
