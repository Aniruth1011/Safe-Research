#!/bin/bash
# Submit the depth sweep's array job and its aggregate job, throttled. Run on a CARC login node from anywhere in the
# Safe-Research checkout; this only calls sbatch (the jobs themselves run on compute nodes).
#
#   export SWEEP_PYTHON=<venv>/bin/python SWEEP_POOL=mean SWEEP_NORMALIZE=0     # plus any other SWEEP_* setting
#   scripts/carc/submit_depth_sweep.sh --account <account> [--layers 0-33,avg] [--max-parallel 6] \
#       [--exclude <node>,<node>] [--time 00:04:00] [--no-aggregate] [--dry-run]
#
# --layers        layer indices, e.g. 0-33 or 10,16,19-20, plus "avg" for the all-layer average baseline (plan row
#                 S5; default 0-33,avg; finished rows are reused either way). "avg" reads all 32 member layers
#                 (~37 GB), so it runs as its own task with $AVG_TIME (default 00:30:00) and $AVG_MEM (default 8G).
# --max-parallel  array tasks running at once (default 6). About 30 tasks reading /scratch1 together stalled some.
# --exclude       nodes to avoid (default: $EXCLUDE_NODES). Pass the same list again when rerunning.
# --dry-run       print the sbatch commands without submitting
# The aggregate job runs after the array whatever its outcome (afterany) and exits 4 if any layer is still missing.
set -euo pipefail
die() { echo "submit_depth_sweep.sh: $*" >&2; exit 1; }
[ -z "${SLURM_JOB_ID:-}" ] || die "run this on a login node, not inside a job"

ACCOUNT="" LAYERS="0-33,avg" MAXP=6 EXCLUDE="${EXCLUDE_NODES:-}" TIME="" AGG=1 DRY=0
while [ $# -gt 0 ]; do
  case $1 in
    --account) ACCOUNT=${2:?}; shift 2 ;;
    --layers) LAYERS=${2:?}; shift 2 ;;
    --max-parallel) MAXP=${2:?}; shift 2 ;;
    --exclude) EXCLUDE=${2:?}; shift 2 ;;
    --time) TIME=${2:?}; shift 2 ;;
    --no-aggregate) AGG=0; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
done
[ -n "$ACCOUNT" ] || die "--account is required"
[[ $LAYERS =~ ^([0-9,-]|avg)+$ ]] || die "--layers must look like 0-33,avg or 10,16,19-20 (got '$LAYERS')"
AVG=0
if [[ ,$LAYERS, == *,avg,* ]]; then
  AVG=1
  LAYERS=$(echo ",$LAYERS," | sed 's/,avg,/,/g; s/^,*//; s/,*$//')
fi
[[ $MAXP =~ ^[1-9][0-9]*$ ]] || die "--max-parallel must be a positive integer (got '$MAXP')"
NODES_RE='^[][A-Za-z0-9,_.-]*$'                              # node names, ranges like b[01-03], commas
[[ $EXCLUDE =~ $NODES_RE ]] || die "--exclude must be a comma-separated node list without spaces"
[[ $TIME =~ ^[0-9:-]*$ ]] || die "--time must look like 00:04:00 (got '$TIME')"
cd "$(dirname "$0")/../.."                                   # sbatch's submit folder = the checkout root
[ -f scripts/depth_sweep.py ] || die "cannot find the checkout root"
mkdir -p logs

COMMON=(--account="$ACCOUNT")
[ -n "$EXCLUDE" ] && COMMON+=(--exclude="$EXCLUDE")
CMDS=()
if [ -n "$LAYERS" ]; then
  SWEEP=(sbatch --parsable "${COMMON[@]}" --export=ALL --array="$LAYERS%$MAXP")
  [ -n "$TIME" ] && SWEEP+=(--time="$TIME")
  CMDS+=("${SWEEP[*]} scripts/carc/depth_sweep.slurm")
fi
if [ "$AVG" = 1 ]; then
  CMDS+=("sbatch --parsable ${COMMON[*]} --export=ALL,SWEEP_LAYER=avg --array=0 --time=${AVG_TIME:-00:30:00} --mem=${AVG_MEM:-8G} scripts/carc/depth_sweep.slurm")
fi
[ ${#CMDS[@]} -gt 0 ] || die "no layers to run"

if [ "$DRY" = 1 ]; then
  printf '%s\n' "${CMDS[@]}"
  [ "$AGG" = 1 ] && echo "sbatch --parsable ${COMMON[*]} --export=ALL --dependency=afterany:<job ids> scripts/carc/depth_sweep_aggregate.slurm"
  exit 0
fi
JOBS=()
for cmd in "${CMDS[@]}"; do
  read -r -a argv <<< "$cmd"                              # no spaces inside any argument (checked above)
  id=$("${argv[@]}")
  JOBS+=("${id%%;*}")
  echo "job ${id%%;*}: $cmd"
done
DEP=$(IFS=:; echo "${JOBS[*]}")
if [ "$AGG" = 1 ]; then
  AJOB=$(sbatch --parsable "${COMMON[@]}" --export=ALL --dependency="afterany:$DEP" scripts/carc/depth_sweep_aggregate.slurm)
  echo "aggregate job ${AJOB%%;*}: runs after job(s) $DEP; exits 4 and lists the layers to rerun if any are missing"
fi
echo "check: sacct -j $(IFS=,; echo "${JOBS[*]}") --format=JobID,Elapsed,MaxRSS,State,ExitCode,NodeList"
