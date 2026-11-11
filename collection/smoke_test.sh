#!/usr/bin/env bash
# Portable smoke test for the all-layer collection. Runs the same sequence on CARC (from carc/01_smoke_test.slurm)
# and on Modal (from modal_app.py::smoke, which sets KIT_ON_MODAL=1). Do not submit the full run until it passes.
#
#   1. ENV      Python 3.10, CUDA GPU, pins vs constraints.txt, flash-attn importable (hard fail), MUJOCO_GL=egl and
#               an EGL context, LIBERO imports without its interactive prompt.
#   2. LOAD     The checkpoint loads offline through SAFE's own get_vla/get_processor with flash_attention_2.
#   3. ROLLOUT  SMOKE_EPISODES (default 2) episodes of one task with --save_all_layers: the first N-1, then N with
#               --resume True (episode 0 must not be rerun).
#   4. CAPTURE  check_cache.py capture: L32 must equal SAFE's own .pkl exactly.
#   5. FULL     (SMOKE_FULL=1 only) an independent repeat run + compare-runs, and a tiny cache build + verify.
# Writes $SMOKE_ROOT/timing_summary.txt.
#
# Settings (environment): OPENVLA_DIR (required), CKPT (or $WORK/checkpoint_path.txt), SMOKE_ROOT
# (default $ROLLOUT_ROOT/smoke-<job id or time>), SMOKE_TASK (default 1: a seen task; 0, 3, 5 are the locked
# unseen tasks), SMOKE_EPISODES (default 2), SMOKE_FULL (default 0), SEED (7), POOLING (mean,first,last).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${KIT_ON_MODAL:-0}" != "1" ]; then
  # CARC / any Slurm or shell host: config.sh first (unless the caller already sourced it), then activate_env.sh.
  set +u
  if [ -z "${KIT_DIR:-}" ]; then source "$HERE/carc/config.sh"; fi
  source "$HERE/carc/activate_env.sh"
  set -u
fi

die () { echo "SMOKE FAIL: $*" >&2; exit 1; }
KIT_DIR="${KIT_DIR:-$HERE}"
: "${OPENVLA_DIR:?OPENVLA_DIR must point at vla-safe/openvla@300dce2 with openvla_all_layers.patch applied}"
SEED="${SEED:-7}"
POOLING="${POOLING:-mean,first,last}"
SMOKE_TASK="${SMOKE_TASK:-1}"
SMOKE_EPISODES="${SMOKE_EPISODES:-2}"
SMOKE_FULL="${SMOKE_FULL:-0}"
if [ -z "${CKPT:-}" ]; then
  [ -n "${WORK:-}" ] && [ -f "$WORK/checkpoint_path.txt" ] || die "set CKPT or provide \$WORK/checkpoint_path.txt"
  CKPT="$(cat "$WORK/checkpoint_path.txt")"
fi
if [ -z "${SMOKE_ROOT:-}" ]; then
  : "${ROLLOUT_ROOT:?set SMOKE_ROOT or ROLLOUT_ROOT}"
  SMOKE_ROOT="$ROLLOUT_ROOT/smoke-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}"
fi
export CKPT KIT_DIR SMOKE_ROOT
case "$SMOKE_TASK" in 0|3|5) echo "WARNING: task $SMOKE_TASK is a locked unseen task; prefer a seen task" >&2 ;; esac
[ "$SMOKE_EPISODES" -ge 1 ] || die "SMOKE_EPISODES must be >= 1"
mkdir -p "$SMOKE_ROOT"

echo "=== 1. ENV ==="
[ "${MUJOCO_GL:-}" = "egl" ] || die "MUJOCO_GL must be egl (got '${MUJOCO_GL:-unset}'): one render backend for all episodes"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || die "nvidia-smi failed: no GPU?"
python - "$KIT_DIR/constraints.txt" <<'EOF'
import importlib.metadata as md, sys
from packaging.requirements import Requirement
assert sys.version_info[:2] == (3, 10), f"Python 3.10 required, got {sys.version}"
bad = []
for line in open(sys.argv[1]):
    line = line.split("#", 1)[0].strip()
    if not line:
        continue
    r = Requirement(line)
    try:
        have = md.version(r.name)
    except md.PackageNotFoundError:
        continue
    ok = r.specifier.contains(have, prereleases=True)
    print(f"{'ok ' if ok else 'BAD'} {r.name} {have} ({r.specifier})")
    bad += [] if ok else [r.name]
assert not bad, f"installed versions violate constraints.txt: {bad}"
import torch
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
import flash_attn, flash_attn_2_cuda  # noqa: F401  hard fail: no silent switch to another attention implementation
from transformers.utils import is_flash_attn_2_available
assert is_flash_attn_2_available(), "transformers cannot use flash_attention_2"
print("flash-attn", flash_attn.__version__)
EOF
python -c "import mujoco; c = mujoco.GLContext(64, 64); c.make_current(); c.free(); print('EGL context ok, mujoco', mujoco.__version__)" \
  || die "EGL context failed (MUJOCO_GL=egl)"
python -c "from libero.libero import benchmark; n = benchmark.get_benchmark_dict()['libero_10']().n_tasks; assert n == 10, n; print('LIBERO ok:', n, 'tasks')" \
  < /dev/null || die "LIBERO import failed (if it asked a question, pre-write \$LIBERO_CONFIG_PATH/config.yaml)"

echo "=== 2. LOAD ==="
(cd "$OPENVLA_DIR" && python - <<'EOF'
import os, sys, time
from types import SimpleNamespace
sys.path.insert(0, os.getcwd())
import torch
from experiments.robot.openvla_utils import get_processor, get_vla
cfg = SimpleNamespace(pretrained_checkpoint=os.environ["CKPT"], attn_implementation="flash_attention_2",
                      load_in_8bit=False, load_in_4bit=False)
t0 = time.time()
vla = get_vla(cfg)
processor = get_processor(cfg)
dt = time.time() - t0
impl = getattr(vla.config, "_attn_implementation", None)
assert impl == "flash_attention_2", f"attention implementation is {impl!r}, expected flash_attention_2"
assert "libero_10" in vla.norm_stats or "libero_10_no_noops" in vla.norm_stats, "libero_10 un-norm key missing"
print(f"model + processor loaded in {dt:.1f} s; attn={impl}; "
      f"GPU memory allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
EOF
) || die "model load failed"

rollout () {  # $1 = run note, $2 = number of episodes. SAFE's settings + all-layer capture (same flags as modal_app.py)
  (cd "$OPENVLA_DIR" && python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla --pretrained_checkpoint "$CKPT" \
    --task_suite_name libero_10 --center_crop True --attn_implementation flash_attention_2 \
    --output_hidden_states True --save_all_layers True --all_layers_pooling "$POOLING" \
    --task_start_index "$SMOKE_TASK" --task_end_index "$SMOKE_TASK" --num_trials_per_task "$2" --resume True \
    --seed "$SEED" --use_wandb False --save_logs True \
    --save_root "$SMOKE_ROOT" --run_id_note "$1")
}
mtime () { python -c "import os, sys; print(int(os.path.getmtime(sys.argv[1])))" "$1"; }
count () { find "$1" -maxdepth 1 -name "task${SMOKE_TASK}--ep*--succ*.csv" | wc -l | tr -d ' '; }

A="$SMOKE_ROOT/rep_a/libero_10"; B="$SMOKE_ROOT/rep_b/libero_10"
echo "=== 3. ROLLOUT (task $SMOKE_TASK, $SMOKE_EPISODES episodes, resume check) ==="
T0=$(date +%s); T1=$T0
if [ "$SMOKE_EPISODES" -ge 2 ]; then
  rollout rep_a $((SMOKE_EPISODES - 1)); T1=$(date +%s)
  EP0=$(find "$A" -maxdepth 1 -name "task${SMOKE_TASK}--ep0--succ*.csv" | head -1)
  [ -n "$EP0" ] || die "episode 0 missing after the first run"
  STAMP0=$(mtime "$EP0")
fi
rollout rep_a "$SMOKE_EPISODES"; T2=$(date +%s)
if [ "$SMOKE_EPISODES" -ge 2 ]; then
  [ "$(mtime "$EP0")" = "$STAMP0" ] || die "RESUME: episode 0 was rerun"
fi
[ "$(count "$A")" = "$SMOKE_EPISODES" ] || die "expected $SMOKE_EPISODES episodes in $A, found $(count "$A")"
ls "$A"/task"${SMOKE_TASK}"--ep*--succ*.layers.pt >/dev/null || die "no .layers.pt files: --save_all_layers did not write"
echo "ROLLOUT ok"

echo "=== 4. CAPTURE ==="
python "$KIT_DIR/check_cache.py" capture --rollout-dir "$A"

if [ "$SMOKE_FULL" = "1" ]; then
  echo "=== 5a. REPEAT (sanity check, not a reproducibility guarantee) ==="
  T3=$(date +%s); rollout rep_b "$SMOKE_EPISODES"; T4=$(date +%s)
  python "$KIT_DIR/check_cache.py" compare-runs --a "$A" --b "$B"
  echo "repeat run, $SMOKE_EPISODES episodes (includes model load): $((T4 - T3)) s" > "$SMOKE_ROOT/repeat_timing.txt"
  echo "=== 5b. BUILD ==="
  python "$KIT_DIR/build_cache.py" --rollout-dir "$A" --cache-root "$SMOKE_ROOT/cache" \
    --expect-tasks 1 --expect-episodes "$SMOKE_EPISODES" --unseen-tasks 99 --allow-incomplete
  python "$KIT_DIR/check_cache.py" verify --cache "$SMOKE_ROOT/cache/v1"
fi

STEPS=$(python -c "import glob, sys, pandas as pd; print(sum(len(pd.read_csv(f)) for f in glob.glob(sys.argv[1] + '/task*--ep*--succ*.csv')))" "$A")
BYTES=$(du -sk "$A" | cut -f1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
{
  echo "=== SMOKE SUMMARY ($(date -u +%Y-%m-%dT%H:%MZ)) ==="
  echo "GPU: $GPU_NAME   MUJOCO_GL: $MUJOCO_GL   task: $SMOKE_TASK   episodes: $SMOKE_EPISODES"
  echo "openvla commit: $(git -C "$OPENVLA_DIR" rev-parse --short HEAD)"
  if [ "$SMOKE_EPISODES" -ge 2 ]; then
    echo "first run, $((SMOKE_EPISODES - 1)) episode(s) (includes model load): $((T1 - T0)) s"
    echo "resume run, 1 more episode (includes model load): $((T2 - T1)) s"
  else
    echo "run, 1 episode (includes model load): $((T2 - T0)) s"
  fi
  echo "steps in rep_a: $STEPS -> $(python -c "import sys; print(round(int(sys.argv[1]) / max(int(sys.argv[2]), 1), 3))" "$((T2 - T0))" "$STEPS") s/step upper bound (includes model loads)"
  echo "raw rollout size per step: ~$(python -c "import sys; print(round(int(sys.argv[1]) * 1024 / max(int(sys.argv[2]), 1) / 1e6, 2))" "$BYTES" "$STEPS") MB"
  echo "Full-run sizing: 500 episodes x (s/step) x avg episode length (failures run to 520 steps)."
} | tee "$SMOKE_ROOT/timing_summary.txt"
echo ">>> Smoke test passed."
