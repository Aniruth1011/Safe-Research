#!/bin/bash
# Pull finished tasks from a Modal Volume to CARC /scratch1, verify checksums and completeness. CPU job only:
# run it through carc/04_pull_from_modal.slurm (never on a login node).
#
#   bash carc/pull_from_modal.sh --profile friend --tasks 0-4 [--run-note all-layers-v1] [--volume cs566-collection]
#
# Needs the Modal CLI on PATH (or MODAL_BIN) with the profile already set up (`modal token set ... --profile <name>`).
# Token values are never printed; profiles are referred to by name only.
#
# Layout on the Volume (written by collection/modal_app.py::rollout_task), one folder per task:
#   rollouts/<run>/taskNN/libero_10/...      SAFE's files + .layers.pt + invalid markers + provenance/
#   rollouts/<run>/taskNN/TASK_META.json     workspace profile, GPU, kit commit, timing, status
#   rollouts/<run>/taskNN/SHA256SUMS         sha256 of every file under libero_10/, written when the task completed
# Result on CARC: $PULL_ROOT/<run>/taskNN/ (same layout), plus $PULL_ROOT/<run>/image_provenance/<profile>/.
# A task is moved into place only after every check passes; a failed pull leaves nothing under taskNN/.
# Nothing is ever deleted from the Volume.
set -euo pipefail

PROFILE="" TASKS="" RUN_NOTE="${RUN_NOTE:-all-layers-v1}" VOLUME="${CS566_DATA_VOLUME:-cs566-collection}"
PULL_ROOT="${PULL_ROOT:-/scratch1/${USER:-$(id -un)}/rollouts}"
MODAL_BIN="${MODAL_BIN:-modal}"
EPISODES="${EPISODES_PER_TASK:-50}"
EXPECTED_GPU="NVIDIA A10"

die() { echo "pull_from_modal: $*" >&2; exit 1; }
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --run-note) RUN_NOTE="$2"; shift 2 ;;
    --volume) VOLUME="$2"; shift 2 ;;
    *) die "unknown argument $1" ;;
  esac
done
[ -n "$PROFILE" ] || die "--profile is required (e.g. friend or akashrav)"
[ -n "$TASKS" ] || die "--tasks is required (e.g. 0-4 or 3,7)"
command -v "$MODAL_BIN" >/dev/null || die "Modal CLI not found (set MODAL_BIN or put modal on PATH)"
command -v sha256sum >/dev/null || die "sha256sum not found"
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_OUTSIDE_SLURM:-0}" != "1" ]; then
  die "run this inside a CPU job (carc/04_pull_from_modal.slurm), never on a login node"
fi

expand_tasks() {  # "0-4" / "3,7" -> 0 1 2 3 4 / 3 7
  local part out=""
  IFS=',' read -ra parts <<< "$1"
  for part in "${parts[@]}"; do
    if [[ "$part" =~ ^([0-9])-([0-9])$ ]]; then out+=" $(seq "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" | tr '\n' ' ')"
    elif [[ "$part" =~ ^[0-9]$ ]]; then out+=" $part"
    else die "bad task spec '$part' (tasks are 0..9)"; fi
  done
  echo $out
}

check_task_dir() {  # $1 = folder holding TASK_META.json, SHA256SUMS, libero_10/ ; $2 = task id
  local d="$1" t="$2"
  python3 - "$d" "$t" "$EPISODES" "$EXPECTED_GPU" "$PROFILE" <<'PY'
import json, re, sys
from pathlib import Path
d, t, n_eps, gpu, profile = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
meta = json.loads((d / "TASK_META.json").read_text())
problems = []
if meta.get("task") != t: problems.append(f"TASK_META task {meta.get('task')} != {t}")
if meta.get("status") != "complete": problems.append(f"status is {meta.get('status')!r}, not 'complete'")
if meta.get("gpu_name") != gpu: problems.append(f"gpu_name {meta.get('gpu_name')!r} != {gpu!r}")
atts = meta.get("attempts", [])
if len(atts) != 1 or atts[0].get("first_episode") != 0 or atts[0].get("engineering_resume"):
    problems.append(f"not one uninterrupted attempt from episode 0 (attempts: "
                    f"{[(a.get('first_episode'), a.get('engineering_resume')) for a in atts]})")
if meta.get("workspace_profile") != profile:
    problems.append(f"written by profile {meta.get('workspace_profile')!r}, pulled with {profile!r}")
ep_dir = d / "libero_10"
for e in range(n_eps):
    data = sorted(p.name for p in ep_dir.glob(f"task{t}--ep{e}--succ*"))
    exts = {n.split(".", 1)[1] for n in data}
    invalid = (ep_dir / f"task{t}--ep{e}--invalid.json").exists()
    complete = {"csv", "pkl", "layers.pt"} <= exts
    if invalid and data: problems.append(f"ep{e}: invalid marker AND data files {data}")
    elif not invalid and not complete: problems.append(f"ep{e}: incomplete ({data or 'no files'})")
extra = [p.name for p in ep_dir.glob(f"task*--ep*") if not p.name.startswith(f"task{t}--")
         or int(re.search(r"--ep(\d+)--", p.name).group(1)) >= n_eps]
if extra: problems.append(f"unexpected files: {extra[:5]}")
if problems:
    print("\n".join(f"  task {t}: {p}" for p in problems)); sys.exit(1)
n_inv = len(list(ep_dir.glob(f"task{t}--ep*--invalid.json")))
print(f"  task {t}: {n_eps - n_inv} episodes + {n_inv} invalid markers, GPU {meta['gpu_name']}, profile {profile}, OK")
PY
}

mkdir -p "$PULL_ROOT/$RUN_NOTE"
LOG="$PULL_ROOT/$RUN_NOTE/pull_log.tsv"
[ -f "$LOG" ] || printf 'utc\tprofile\ttask\tresult\n' > "$LOG"

# Image provenance of this workspace (compared across workspaces before the cache is built)
PROV="$PULL_ROOT/$RUN_NOTE/image_provenance/$PROFILE"
mkdir -p "$PROV"
"$MODAL_BIN" volume get --profile "$PROFILE" --force "$VOLUME" work/image_provenance "$PROV.tmp" >/dev/null
rm -rf "$PROV" && mv "$PROV.tmp" "$PROV"
"$MODAL_BIN" volume get --profile "$PROFILE" --force "$VOLUME" work/checkpoint_path.txt "$PROV/checkpoint_path.txt" >/dev/null

failed=0
for t in $(expand_tasks "$TASKS"); do
  tt=$(printf '%02d' "$t")
  final="$PULL_ROOT/$RUN_NOTE/task$tt"
  if [ -e "$final" ]; then
    echo "task $t: $final already exists; checking it again (delete it yourself to re-pull)"
    check_task_dir "$final" "$t" || { failed=1; continue; }
    continue
  fi
  stage="$PULL_ROOT/$RUN_NOTE/.incoming-task$tt"
  rm -rf "$stage"
  echo "task $t: pulling rollouts/$RUN_NOTE/task$tt from volume $VOLUME (profile $PROFILE)"
  if ! "$MODAL_BIN" volume get --profile "$PROFILE" "$VOLUME" "rollouts/$RUN_NOTE/task$tt" "$stage" >/dev/null; then
    echo "task $t: download failed"; printf '%s\t%s\t%s\tdownload-failed\n' "$(date -u +%FT%TZ)" "$PROFILE" "$t" >> "$LOG"
    failed=1; continue
  fi
  # `modal volume get` of a folder may nest it under its own name; normalise to $stage/{TASK_META.json,...}
  [ -f "$stage/TASK_META.json" ] || { [ -d "$stage/task$tt" ] && stage="$stage/task$tt"; }
  if ! ( cd "$stage" && sha256sum --quiet -c SHA256SUMS ); then
    echo "task $t: CHECKSUM MISMATCH"; printf '%s\t%s\t%s\tchecksum-failed\n' "$(date -u +%FT%TZ)" "$PROFILE" "$t" >> "$LOG"
    failed=1; continue
  fi
  listed=$(cut -c67- "$stage/SHA256SUMS" | sort); present=$(cd "$stage" && find libero_10 -type f | sort)
  if [ "$listed" != "$present" ]; then
    echo "task $t: files on disk differ from SHA256SUMS"; failed=1; continue
  fi
  if ! check_task_dir "$stage" "$t"; then
    printf '%s\t%s\t%s\tincomplete\n' "$(date -u +%FT%TZ)" "$PROFILE" "$t" >> "$LOG"; failed=1; continue
  fi
  mv "$stage" "$final"
  rm -rf "$PULL_ROOT/$RUN_NOTE/.incoming-task$tt"
  printf '%s\t%s\t%s\tok\n' "$(date -u +%FT%TZ)" "$PROFILE" "$t" >> "$LOG"
done

[ "$failed" = 0 ] || die "some tasks failed (see above and $LOG); nothing was deleted on Modal"
echo "pull_from_modal: all requested tasks verified under $PULL_ROOT/$RUN_NOTE"
