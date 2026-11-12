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
# Result on CARC: $PULL_ROOT/<run>/taskNN/ (same layout), plus $PULL_ROOT/<run>/image_provenance/<profile>/ (a
# directory). Reruns are safe: a task already pulled and still verified is skipped; anything partial, mismatched or
# malformed (also leftovers of an interrupted pull) is moved to $PULL_ROOT/<run>/_discarded/<utc>/, never deleted.
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

RUN_DIR="$PULL_ROOT/$RUN_NOTE"
UTC=$(date -u +%Y%m%dT%H%M%SZ)
DISCARD="$RUN_DIR/_discarded/$UTC"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/pull_log.tsv"
[ -f "$LOG" ] || printf 'utc\tprofile\ttask\tresult\n' > "$LOG"
logrow() { printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$PROFILE" "$1" "$2" >> "$LOG"; }
discard() {  # move $1 aside as _discarded/<utc>/$2 (never delete), and say why
  mkdir -p "$DISCARD"
  mv "$1" "$DISCARD/$2"
  echo "  moved $1 -> $DISCARD/$2 ($3)"
}

# `modal volume get VOL REMOTE DEST` (modal 1.6): if DEST is an existing directory, files land in DEST/<basename of
# REMOTE>/...; if DEST does not exist, EVERY file is written to DEST itself (a folder collapses into one file); an
# existing DEST without --force is an error. So: always download into a fresh, existing, empty directory, then locate
# the payload inside it (with or without the <basename> level).
fetch() {  # $1 = remote path, $2 = fresh local dir (created here) ; prints the payload path
  mkdir -p "$2"
  "$MODAL_BIN" volume get --profile "$PROFILE" "$VOLUME" "$1" "$2/" >/dev/null || return 1
  local base; base=$(basename "$1")
  if [ -e "$2/$base" ]; then echo "$2/$base"; else echo "$2"; fi
}

# Leftovers of an interrupted earlier pull are moved aside, never reused.
for d in "$RUN_DIR"/.incoming-*; do [ -e "$d" ] && discard "$d" "$(basename "$d")" "left over from an interrupted pull"; done

# ---- image provenance of this workspace: a directory image_provenance/<profile>/ (pip freeze, commits, fingerprint,
#      checkpoint_path.txt), compared across workspaces by 05_build_cache_from_modal.slurm
PROV="$RUN_DIR/image_provenance/$PROFILE"
mkdir -p "$RUN_DIR/image_provenance"
INC="$RUN_DIR/.incoming-provenance-$PROFILE"
P=$(fetch work/image_provenance "$INC") || die "could not download work/image_provenance from volume $VOLUME"
[ -f "$P/image_fingerprint.txt" ] || die "work/image_provenance in volume $VOLUME has no image_fingerprint.txt"
INC_C="$RUN_DIR/.incoming-checkpoint-$PROFILE"
CK=$(fetch work/checkpoint_path.txt "$INC_C") || die "could not download work/checkpoint_path.txt"
[ -f "$CK" ] || CK="$CK/checkpoint_path.txt"
mv "$CK" "$P/checkpoint_path.txt"
rmdir "$INC_C" 2>/dev/null || true
if [ -d "$PROV" ] && diff -r -q "$PROV" "$P" >/dev/null; then
  echo "image provenance for $PROFILE: unchanged (already in place)"
  rm -rf "$INC"   # a byte-identical duplicate of what is already in place
else
  [ -e "$PROV" ] && discard "$PROV" "image_provenance-$PROFILE" "differs from the volume or is not a directory"
  mv "$P" "$PROV"
  rm -rf "$INC"   # only empty helper folders remain here
  echo "image provenance for $PROFILE: $(cat "$PROV/image_fingerprint.txt")"
fi

verify_task() {  # $1 = task folder, $2 = task id: checksums, file list, completeness, GPU, profile, one attempt
  local d="$1" t="$2"
  [ -f "$d/TASK_META.json" ] && [ -f "$d/SHA256SUMS" ] && [ -d "$d/libero_10" ] || { echo "  task $t: TASK_META.json/SHA256SUMS/libero_10 missing"; return 1; }
  ( cd "$d" && sha256sum --quiet -c SHA256SUMS ) || { echo "  task $t: CHECKSUM MISMATCH"; return 1; }
  local listed present
  listed=$(cut -c67- "$d/SHA256SUMS" | sort); present=$(cd "$d" && find libero_10 -type f | sort)
  [ "$listed" = "$present" ] || { echo "  task $t: files on disk differ from SHA256SUMS"; return 1; }
  check_task_dir "$d" "$t"
}

failed=0
for t in $(expand_tasks "$TASKS"); do
  tt=$(printf '%02d' "$t")
  final="$RUN_DIR/task$tt"
  if [ -e "$final" ]; then
    if verify_task "$final" "$t"; then
      echo "task $t: already pulled and verified; skipped"; logrow "$t" "already-verified"; continue
    fi
    discard "$final" "task$tt" "failed verification"
  fi
  inc="$RUN_DIR/.incoming-task$tt"
  echo "task $t: pulling rollouts/$RUN_NOTE/task$tt from volume $VOLUME (profile $PROFILE)"
  if ! src=$(fetch "rollouts/$RUN_NOTE/task$tt" "$inc"); then
    echo "task $t: download failed"; logrow "$t" "download-failed"; failed=1
    [ -e "$inc" ] && discard "$inc" "incoming-task$tt" "download failed"
    continue
  fi
  if ! verify_task "$src" "$t"; then
    logrow "$t" "verification-failed"; failed=1
    discard "$inc" "incoming-task$tt" "failed verification after download"
    continue
  fi
  mv "$src" "$final"
  if [ -e "$inc" ]; then  # only when the CLI nested the folder: the helper directory must now be empty
    rmdir "$inc" 2>/dev/null || discard "$inc" "incoming-task$tt-rest" "unexpected extra files next to the task folder"
  fi
  logrow "$t" "ok"
done

[ "$failed" = 0 ] || die "some tasks failed (see above and $LOG); nothing was deleted, on Modal or here"
echo "pull_from_modal: all requested tasks verified under $RUN_DIR"
