#!/bin/bash
# One-time setup. Needs internet: run on a CARC login node, or (better for the long installs) inside an
# interactive CPU allocation on the `main` partition. From inside collection/, after editing carc/config.sh:
#   bash carc/00_setup_env.sh            # build everything; safe to rerun, finished steps are skipped
#   bash carc/00_setup_env.sh --verify   # install nothing, only check an existing environment
# What it pins and why: ../constraints.txt (one comment per pin). Every install below goes through it.
set -euo pipefail
KIT_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=collection/carc/config.sh
source "$KIT_SRC/carc/config.sh"

die() { echo "00_setup_env.sh: ERROR: $*" >&2; exit 1; }

kit_check_config || die "carc/config.sh is not ready (see the lines above). Nothing was installed."

MODE="setup"
case "${1:-}" in
  "") ;;
  --verify) MODE="verify" ;;
  *) die "unknown argument '$1'. Usage: bash carc/00_setup_env.sh [--verify]" ;;
esac

CONSTRAINTS="$KIT_DIR/constraints.txt"   # single source of truth for every pinned version
PATCH="$KIT_DIR/openvla_all_layers.patch"
[ -f "$CONSTRAINTS" ] || die "constraints file not found: $CONSTRAINTS"
[ -f "$PATCH" ] || die "patch not found: $PATCH"
# Versions are never repeated in this script: installs name packages only and -c "$CONSTRAINTS" picks the
# version. The flash-attn version is read here only for the messages and the import check below.
FLASH_ATTN_PIN="$(sed -n 's/^flash-attn==\([0-9][0-9.]*\).*/\1/p' "$CONSTRAINTS")"
[ -n "$FLASH_ATTN_PIN" ] || die "$CONSTRAINTS has no exact 'flash-attn==X' pin"

# The installer for each mode. Both take the same arguments (install / check / freeze).
if [ "$ENV_MODE" = "uv" ]; then
  if ! command -v uv >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
  fi
  command -v uv >/dev/null 2>&1 || die "ENV_MODE=uv but 'uv' was not found. Install it once (goes to" \
    "~/.local/bin, no admin rights needed): curl -LsSf https://astral.sh/uv/install.sh | sh"
  PIP=(uv pip)
else
  PIP=(python -m pip)
fi

# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
flash_attn_fail() {  # $1 = what went wrong
  cat >&2 <<EOF

00_setup_env.sh: ERROR: flash-attn $FLASH_ATTN_PIN is not usable: $1

Setup stops here on purpose.
Do NOT switch to another attention implementation to get past this.
SAFE runs OpenVLA with attn_implementation="flash_attention_2". A different implementation changes the
hidden-state features, so a dataset collected that way (or with a mix of both) cannot be compared with
SAFE or with the rest of the team's runs.

Fix the build instead:
  - CUDA_HOME in carc/config.sh must be a CUDA toolkit root with bin/nvcc (now: $CUDA_HOME).
    'module load cuda' does not set it in non-interactive shells on CARC.
  - torch (the +cu121 build pinned in constraints.txt) must already be installed (the build reads it),
    plus packaging, ninja, setuptools, wheel.
  - Then rerun: bash carc/00_setup_env.sh
EOF
  exit 1
}

check_python_310() {
  local v
  v="$(python -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
  case "$v" in
    3.10.*) echo ">>> Python $v ($(command -v python))" ;;
    *) die "the environment's Python is $v, but this project needs 3.10.x. In uv mode set PYTHON_BIN in" \
           "carc/config.sh to a python3.10 binary and delete $VENV_DIR; in conda mode recreate $CONDA_ENV." ;;
  esac
}

check_pins() {  # every installed version must satisfy ../constraints.txt (the single source of truth)
  python - "$CONSTRAINTS" <<'PY'
import sys
from importlib import metadata

from packaging.requirements import Requirement
from packaging.version import Version

bad = []
with open(sys.argv[1]) as fh:
    for raw in fh:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        req = Requirement(line)
        try:
            have = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            have = None
        good = have is not None and req.specifier.contains(Version(have), prereleases=True)
        print(f"  {'ok ' if good else 'BAD'} {req.name:<20} installed {have or 'NOT INSTALLED':<16} required {req.specifier}")
        if not good:
            bad.append(req.name)
# The CUDA build cannot be expressed in a constraints file (no index URLs there): check it here.
for name in ("torch", "torchvision", "torchaudio"):
    try:
        have = metadata.version(name)
    except metadata.PackageNotFoundError:
        continue
    if not have.endswith("+cu121"):
        print(f"  BAD {name:<20} installed {have:<16} required the +cu121 build (PyTorch cu121 index)")
        bad.append(name)
if bad:
    sys.exit(
        "PIN MISMATCH: " + ", ".join(bad) + ". These versions (collection/constraints.txt) are the ones "
        "verified to work together. Do not upgrade them; reinstall through carc/00_setup_env.sh."
    )
PY
}

check_flash_attn() {
  # What a login node (no GPU) can check: the package is the pinned version and its compiled CUDA extension
  # (flash_attn_2_cuda) loads against the installed torch.
  # What ONLY a GPU node can check: torch.cuda.is_available(), transformers' is_flash_attn_2_available()
  # (it needs a visible GPU), and that the kernels actually run. On a GPU node the second half below does the
  # first two; the smoke test (01_smoke_test.slurm) does the third by loading and running the model.
  python -c '
import sys
from importlib import metadata
import flash_attn, flash_attn_2_cuda  # the Python package and its compiled CUDA extension
assert metadata.version("flash-attn") == sys.argv[1], (metadata.version("flash-attn"), sys.argv[1])
import torch
if torch.cuda.is_available():
    from transformers.utils import is_flash_attn_2_available
    assert is_flash_attn_2_available(), "transformers cannot use flash-attn on this GPU node"
    print(">>> flash-attn", sys.argv[1], "usable by transformers on", torch.cuda.get_device_name(0))
else:
    print(">>> flash-attn", sys.argv[1], "package and CUDA extension import (no GPU here; the smoke test runs it on one)")
' "$FLASH_ATTN_PIN" </dev/null || flash_attn_fail "the import check failed (see the Python error above)"
}

check_libero() {  # stdin is /dev/null: if LIBERO still wanted to ask its question this fails instead of hanging
  [ -f "$LIBERO_CONFIG_PATH/config.yaml" ] || die "LIBERO config missing: $LIBERO_CONFIG_PATH/config.yaml"
  python -c '
import os
from libero.libero import benchmark, get_libero_path
root = get_libero_path("benchmark_root")
want = os.path.join(os.environ["LIBERO_DIR"], "libero", "libero")
assert os.path.isdir(root) and os.path.samefile(root, want), (
    "LIBERO config points at %s, expected %s. Delete %s/config.yaml and rerun carc/00_setup_env.sh"
    % (root, want, os.environ["LIBERO_CONFIG_PATH"]))
for key in ("bddl_files", "init_states"):
    assert os.path.isdir(get_libero_path(key)), "LIBERO %s folder missing: %s" % (key, get_libero_path(key))
n = benchmark.get_benchmark_dict()["libero_10"]().n_tasks
assert n == 10, "libero_10 should have 10 tasks, got %d" % n
print("LIBERO ok:", n, "tasks")
' </dev/null || die "LIBERO import test failed (see the Python error above)."
  # Not checked here: creating a simulator env imports robosuite/mujoco, whose EGL start-up needs a GPU node.
}

check_repos() {
  local head
  head="$(git -C "$OPENVLA_DIR" rev-parse HEAD)"
  [ "$head" = "$OPENVLA_COMMIT" ] || die "$OPENVLA_DIR is at $head, expected OPENVLA_COMMIT=$OPENVLA_COMMIT"
  git -C "$OPENVLA_DIR" apply --reverse --check "$PATCH" 2>/dev/null \
    || die "$OPENVLA_DIR does not contain exactly this kit's openvla_all_layers.patch"
  head="$(git -C "$LIBERO_DIR" rev-parse HEAD)"
  [ "$head" = "$LIBERO_COMMIT" ] || die "$LIBERO_DIR is at $head, expected LIBERO_COMMIT=$LIBERO_COMMIT"
  echo ">>> openvla @ $OPENVLA_COMMIT + patch, LIBERO @ $LIBERO_COMMIT"
}

check_checkpoint() {
  local snap
  [ -s "$WORK/checkpoint_path.txt" ] || die "$WORK/checkpoint_path.txt missing: the checkpoint was not downloaded"
  snap="$(cat "$WORK/checkpoint_path.txt")"
  [ -f "$snap/config.json" ] || die "checkpoint snapshot incomplete or missing: $snap"
  # Jobs run with HF_HUB_OFFLINE=1. The checkpoint's processor class lives in the BASE repo
  # ($PROCESSOR_CODE_REPO), so this fails unless that repo's code files are in the local cache too.
  HF_HUB_OFFLINE=1 python -c '
import sys
from transformers import AutoProcessor
p = AutoProcessor.from_pretrained(sys.argv[1], trust_remote_code=True)
print(">>> processor loads offline:", type(p).__name__)
' "$snap" </dev/null || die "the processor does not load offline from $snap. The code files of" \
    "$PROCESSOR_CODE_REPO must be in $HF_HOME; rerun carc/00_setup_env.sh with internet access."
}

verify_env() {
  echo ">>> Verifying the environment (ENV_MODE=$ENV_MODE)"
  check_python_310
  check_pins || die "pinned versions do not match (table above)."
  "${PIP[@]}" check || die "'pip check' reports conflicting packages (above)."
  check_flash_attn
  check_repos
  check_libero
  check_checkpoint
  echo ">>> Environment verified."
}

clone_at() {  # $1 = url, $2 = dir, $3 = full commit hash, $4 = name for messages
  local head
  if [ ! -d "$2/.git" ]; then
    git clone "$1" "$2"
  fi
  head="$(git -C "$2" rev-parse HEAD)"
  if [ "$head" != "$3" ]; then
    if [ -n "$(git -C "$2" status --porcelain)" ]; then
      die "$4 clone $2 is at $head (expected $3) and has local changes. Move that folder away and rerun."
    fi
    git -C "$2" cat-file -e "$3^{commit}" 2>/dev/null || git -C "$2" fetch origin
    git -C "$2" cat-file -e "$3^{commit}" 2>/dev/null \
      || die "$4 commit $3 does not exist in $1 (deleted or force-pushed upstream?). The kit's patch and" \
             "the dataset are pinned to it: do not substitute another commit; tell the team."
    git -C "$2" checkout --detach "$3"
  fi
  head="$(git -C "$2" rev-parse HEAD)"
  [ "$head" = "$3" ] || die "$4 clone $2 is at $head, expected $3"
}

recorded_rev() {  # $1 = HF repo id; prints the revision an earlier run recorded, or nothing
  if [ -f "$WORK/checkpoint_commit.txt" ]; then
    awk -v r="$1" '$1 == r { print $2; exit }' "$WORK/checkpoint_commit.txt"
  fi
}

hf_snapshot() {  # $1 = HF repo id, $2 = revision ("" = current main), $3 = all | code; prints the snapshot dir
  HF_HUB_OFFLINE=0 python - "$1" "$2" "$3" <<'PY' | tail -n 1
import sys
from huggingface_hub import snapshot_download

repo, revision, what = sys.argv[1:4]
kwargs = {}
if what == "code":  # everything except model weights (a few MB: configs, tokenizer, the processor's .py files)
    kwargs["ignore_patterns"] = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt", "*.h5", "*.msgpack", "*.onnx"]
print(snapshot_download(repo_id=repo, revision=revision or None, **kwargs))
PY
}

# ----------------------------------------------------------------------------------------------
# --verify: activate, check, stop
# ----------------------------------------------------------------------------------------------
if [ "$MODE" = "verify" ]; then
  # shellcheck source=collection/carc/activate_env.sh
  source "$KIT_SRC/carc/activate_env.sh"
  verify_env
  exit 0
fi

# ----------------------------------------------------------------------------------------------
# 1. Pre-flight (before anything is installed)
# ----------------------------------------------------------------------------------------------
# flash-attn's build needs the CUDA compiler. Checked first so a wrong CUDA_HOME costs seconds, not an hour.
[ -x "$CUDA_HOME/bin/nvcc" ] || die "CUDA_HOME=$CUDA_HOME has no bin/nvcc. flash-attn cannot be built without" \
  "it. Set CUDA_HOME in carc/config.sh to the CUDA toolkit root ('module show cuda' prints it)."
command -v git >/dev/null 2>&1 || die "git not found"

mkdir -p "$WORK" "$HF_HOME" "$LIBERO_CONFIG_PATH" "$WORK/tmp"
# Keep download/build caches and temp files next to the env, not in $HOME or the login node's /tmp.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORK/cache/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORK/cache/pip}"
export TMPDIR="$WORK/tmp"

# ----------------------------------------------------------------------------------------------
# 2. Create the Python 3.10 environment if it does not exist yet, then activate it
# ----------------------------------------------------------------------------------------------
if [ "$ENV_MODE" = "uv" ]; then
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    if [ -n "$PYTHON_BIN" ]; then
      [ -x "$PYTHON_BIN" ] || die "PYTHON_BIN=$PYTHON_BIN is not an executable file (see carc/config.sh)"
      uv venv --python "$PYTHON_BIN" "$VENV_DIR"
    else
      uv venv --python 3.10 "$VENV_DIR"   # uv finds a 3.10 on PATH or downloads its own; no `module load`
    fi
  fi
else
  set +u   # module/conda shell functions are not written for `set -u`
  if command -v module >/dev/null 2>&1; then
    module purge || true
    module load conda 2>/dev/null || module load anaconda3 2>/dev/null || true   # whichever your CARC setup uses
  fi
  command -v conda >/dev/null 2>&1 || die "ENV_MODE=conda but no 'conda' command was found. Use ENV_MODE=uv."
  eval "$(conda shell.bash hook)"
  CONDA_ENVS="$(conda env list)"
  if ! awk -v n="$CONDA_ENV" '$1 == n { found = 1 } END { exit !found }' <<<"$CONDA_ENVS"; then
    conda create -y -n "$CONDA_ENV" python=3.10
  fi
  set -u
fi
export KIT_SETUP=1   # tells activate_env.sh that LIBERO's config is not expected to exist yet
# shellcheck source=collection/carc/activate_env.sh
source "$KIT_SRC/carc/activate_env.sh"
check_python_310

# ----------------------------------------------------------------------------------------------
# 3. Sources: SAFE's OpenVLA fork at the pinned commit + the kit's patch; LIBERO at the pinned commit
# ----------------------------------------------------------------------------------------------
clone_at "$OPENVLA_REPO" "$OPENVLA_DIR" "$OPENVLA_COMMIT" "OpenVLA"
if git -C "$OPENVLA_DIR" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo ">>> openvla_all_layers.patch is already applied"
elif git -C "$OPENVLA_DIR" apply --check "$PATCH"; then
  git -C "$OPENVLA_DIR" apply "$PATCH"
else
  die "openvla_all_layers.patch neither applies to $OPENVLA_DIR nor is already applied there. The folder has" \
      "other changes (an older version of the patch?). Move it away and rerun to get a fresh clone."
fi
git -C "$OPENVLA_DIR" rev-parse HEAD > "$WORK/openvla_commit.txt"

clone_at "$LIBERO_REPO" "$LIBERO_DIR" "$LIBERO_COMMIT" "LIBERO"
git -C "$LIBERO_DIR" rev-parse HEAD > "$WORK/libero_commit.txt"

# ----------------------------------------------------------------------------------------------
# 4. Packages. Order as verified on CARC. EVERY install line carries -c "$CONSTRAINTS".
#    Versions come from the constraints file only. CUDA 12.1 wheels (PyTorch index) run on A40/A100/L40S/A10.
# ----------------------------------------------------------------------------------------------
"${PIP[@]}" install -c "$CONSTRAINTS" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
"${PIP[@]}" install -c "$CONSTRAINTS" -e "$OPENVLA_DIR"
"${PIP[@]}" install -c "$CONSTRAINTS" pandas scikit-learn
"${PIP[@]}" install -c "$CONSTRAINTS" -e "$LIBERO_DIR"
"${PIP[@]}" install -c "$CONSTRAINTS" -r "$OPENVLA_DIR/experiments/robot/libero/libero_requirements.txt"

# LIBERO's editable install exposes no modules (libero/ has no __init__.py), so `import libero` fails
# without this .pth file that puts the repo root on sys.path.
SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
printf '%s\n' "$LIBERO_DIR" > "$SITE_PACKAGES/libero_repo.pth"

# LIBERO asks "custom dataset path? (Y/N)" on first import and then writes config.yaml. Answer N here, once,
# so no batch job ever meets the question. LIBERO_CONFIG_PATH (config.sh) keeps the file out of ~/.libero.
if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then
  printf 'N\n' | python -c "import libero.libero" \
    || die "LIBERO's first import failed; its config was not written to $LIBERO_CONFIG_PATH"
fi

# ----------------------------------------------------------------------------------------------
# 5. flash-attn: REQUIRED. Installed after the editable install, as the OpenVLA README says.
#    There is no fallback: if it cannot be installed and imported, setup fails (see flash_attn_fail).
# ----------------------------------------------------------------------------------------------
"${PIP[@]}" install -c "$CONSTRAINTS" packaging ninja setuptools wheel
if ! "${PIP[@]}" install -c "$CONSTRAINTS" flash-attn --no-build-isolation; then
  flash_attn_fail "the install/build step failed (see the output above)"
fi
check_flash_attn

# ----------------------------------------------------------------------------------------------
# 6. Model files: the fine-tuned checkpoint (~15 GB) and the base repo's non-weight code files (a few MB).
#    The first run takes the current `main` of each repo and records its commit; reruns reuse the recorded
#    commit, so the pin cannot move by accident. To re-pin on purpose, delete $WORK/checkpoint_commit.txt.
# ----------------------------------------------------------------------------------------------
CKPT_REV="$(recorded_rev "$CHECKPOINT_REPO")"
SNAP="$(hf_snapshot "$CHECKPOINT_REPO" "$CKPT_REV" all)" || die "download of $CHECKPOINT_REPO failed"
[ -f "$SNAP/config.json" ] || die "download of $CHECKPOINT_REPO returned no usable snapshot ('$SNAP')"
CKPT_REV="$(basename "$SNAP")"   # HF cache layout: .../snapshots/<commit hash>

CODE_REV="$(recorded_rev "$PROCESSOR_CODE_REPO")"
CODE_SNAP="$(hf_snapshot "$PROCESSOR_CODE_REPO" "$CODE_REV" code)" || die "download of $PROCESSOR_CODE_REPO failed"
[ -d "$CODE_SNAP" ] || die "download of $PROCESSOR_CODE_REPO returned no usable snapshot ('$CODE_SNAP')"
CODE_REV="$(basename "$CODE_SNAP")"
# Offline, transformers finds the base repo's code through the cache's refs/main. It must be the recorded commit.
CODE_REF="$(dirname "$(dirname "$CODE_SNAP")")/refs/main"
if [ ! -f "$CODE_REF" ] || [ "$(cat "$CODE_REF")" != "$CODE_REV" ]; then
  die "the local cache's 'main' of $PROCESSOR_CODE_REPO ($CODE_REF) is not the recorded commit $CODE_REV." \
      "Jobs would load different processor code than the one recorded."
fi

echo "$SNAP" > "$WORK/checkpoint_path.txt"
printf '%s %s\n%s %s (non-weight code files only)\n' \
  "$CHECKPOINT_REPO" "$CKPT_REV" "$PROCESSOR_CODE_REPO" "$CODE_REV" > "$WORK/checkpoint_commit.txt"
echo ">>> Pinned checkpoint snapshot: $SNAP"
echo ">>> Pinned commits: $CHECKPOINT_REPO@$CKPT_REV, $PROCESSOR_CODE_REPO@$CODE_REV"

# ----------------------------------------------------------------------------------------------
# 7. Verify everything, then record the environment (the cache build copies these files as provenance)
# ----------------------------------------------------------------------------------------------
verify_env
"${PIP[@]}" freeze > "$WORK/pip_freeze.txt"
echo ">>> Setup done. Records in $WORK: pip_freeze.txt openvla_commit.txt libero_commit.txt" \
     "checkpoint_path.txt checkpoint_commit.txt"
echo ">>> Next: sbatch carc/01_smoke_test.slurm"
