#!/bin/bash
# Submit the depth sweep's array job and its aggregate job, throttled. Run on a CARC login node from anywhere in the
# Safe-Research checkout; this only calls sbatch (the jobs themselves run on compute nodes).
#
#   export SWEEP_PYTHON=<venv>/bin/python SWEEP_POOL=mean SWEEP_NORMALIZE=0     # plus any other SWEEP_* setting
#   scripts/carc/submit_depth_sweep.sh --account <account> [--layers 0-33] [--max-parallel 6] \
#       [--exclude <node>,<node>] [--time 00:04:00] [--no-aggregate] [--dry-run]
#
# --layers        Slurm array indices, e.g. 0-33 or 10,16,19-20 (default 0-33; finished rows are reused either way)
# --max-parallel  array tasks running at once (default 6). About 30 tasks reading /scratch1 together stalled some.
# --exclude       nodes to avoid (default: $EXCLUDE_NODES). Pass the same list again when rerunning.
# --dry-run       print the sbatch commands without submitting
# The aggregate job runs after the array whatever its outcome (afterany) and exits 4 if any layer is still missing.
set -euo pipefail
die() { echo "submit_depth_sweep.sh: $*" >&2; exit 1; }
[ -z "${SLURM_JOB_ID:-}" ] || die "run this on a login node, not inside a job"

ACCOUNT="" LAYERS="0-33" MAXP=6 EXCLUDE="${EXCLUDE_NODES:-}" TIME="" AGG=1 DRY=0
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
[[ $LAYERS =~ ^[0-9,-]+$ ]] || die "--layers must look like 0-33 or 10,16,19-20 (got '$LAYERS')"
[[ $MAXP =~ ^[1-9][0-9]*$ ]] || die "--max-parallel must be a positive integer (got '$MAXP')"
cd "$(dirname "$0")/../.."                                   # sbatch's submit folder = the checkout root
[ -f scripts/depth_sweep.py ] || die "cannot find the checkout root"
mkdir -p logs

COMMON=(--account="$ACCOUNT" --export=ALL)
[ -n "$EXCLUDE" ] && COMMON+=(--exclude="$EXCLUDE")
SWEEP=(sbatch --parsable "${COMMON[@]}" --array="$LAYERS%$MAXP")
[ -n "$TIME" ] && SWEEP+=(--time="$TIME")
SWEEP+=(scripts/carc/depth_sweep.slurm)

if [ "$DRY" = 1 ]; then
  echo "${SWEEP[*]}"
  [ "$AGG" = 1 ] && echo "sbatch --parsable ${COMMON[*]} --dependency=afterany:<array job id> scripts/carc/depth_sweep_aggregate.slurm"
  exit 0
fi
JOB=$("${SWEEP[@]}")
JOB=${JOB%%;*}
echo "array job $JOB: layers $LAYERS, at most $MAXP at once${EXCLUDE:+, excluding $EXCLUDE}"
if [ "$AGG" = 1 ]; then
  AJOB=$(sbatch --parsable "${COMMON[@]}" --dependency="afterany:$JOB" scripts/carc/depth_sweep_aggregate.slurm)
  echo "aggregate job ${AJOB%%;*}: runs after the array; exits 4 and lists the layers to rerun if any are missing"
fi
echo "check: sacct -j $JOB --format=JobID,Elapsed,MaxRSS,State,ExitCode,NodeList"
