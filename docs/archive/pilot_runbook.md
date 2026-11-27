> **Archived (2026-11-10).** Written for the retired `akash/collector` pipeline (`akash-ravi/openvla`, tag `archive/collector-2026-11`). Collection now uses the kit in `collection/`. Kept for reference only.

# Pilot runbook (A5/A6) — DRAFT, FOR THE CARC AGENT TO REVIEW

Status: **draft, nothing here has been run on CARC.** Items marked *(UNVERIFIED)* are guesses to confirm against CARC docs or by a quick check before use. Do not submit jobs, start A8, download >20 GB, or delete anything without asking Akash (hard rule 5).

Code: `vla-safe/openvla` fork, branch `akash/collector` (remote `akash-ravi/openvla`). Docs: `Safe-Research`, branch `akash/extraction`.

## 0. Preconditions
- T1–T5 done (account, storage ≥150 GB, env, EGL rendering, single inference).
- Two conda envs: **collection** (fork, torch 2.2.0, transformers 4.40.1, LIBERO, flash-attn) and **detector** (Safe-Research, `pip install -e . --no-deps` if it ever shares the collection env).
- Checkpoint already on disk (compute nodes run with `HF_HUB_OFFLINE=1`). Record the HF revision hash.
- Pilot tasks = **seen** tasks only, from the P0-2 split manifest. Do not score detectors on anything here (hard rule 1).
- CPU sanity on the login node (no GPU): in the fork, `python -m pytest -q tests` (needs transformers==4.40.1 for the capture and dry-run tests) and `python experiments/robot/libero/audit_collection.py --help`.

## 1. Job template
`experiments/robot/libero/slurm/collect_task.sbatch` (fork). Header: `#!/bin/bash -l`, `-A <account>`, `-p debug -n 1 --gpus=a40:1 -t 01:00:00`, `--requeue`, `--open-mode=append`, plus `--cpus-per-task=8 --mem=48G` *(UNVERIFIED: from handoff T2)*. It sets `HF_HUB_OFFLINE=1`, `MUJOCO_GL=egl`, and requires an explicit `CUDA_HOME`. One job = one task = one episode range. Fill in `CKPT`, `COLLECT_ENV`, `CUDA_HOME`, `OUT_ROOT` (scratch, e.g. `/scratch1/<user>/data/pilot`).

## 2. Steps
1. **Smoke job (1 task, 2 episodes)** — debug partition:
   ```bash
   mkdir -p logs
   sbatch -J smoke_t0 --export=ALL,TASK_ID=0,EP_START=0,EP_END=2,OUT_ROOT=$OUT_ROOT,CKPT=$CKPT,COLLECT_ENV=$ENV,CUDA_HOME=$CUDA_HOME \
       experiments/robot/libero/slurm/collect_task.sbatch
   ```
   Check the log for: bf16 supported, GPU name, `[capture] keys=[...]`, no exceptions, steps/s per episode. Then `audit_collection.py $OUT_ROOT --num-tasks 10 --n-episodes 2 --expect-seed 7` (full mode); with only task 0 collected expect `MISSING_TASK` flags for the others — pass `--num-tasks 1` instead.
2. **GPU verification jobs** (same header, `-t 00:30:00` is enough *(UNVERIFIED)*): T6 (`VLA_CKPT=$CKPT OPENVLA_FORK_DIR=$PWD python -m pytest tests/gpu/test_layer_mapping.py -s` from Safe-Research, collection env) and the SAFE parity check (`python experiments/robot/libero/parity_check.py --checkpoint $CKPT --task_id 0 --n_episodes 2 --workdir $SCRATCH/parity`). Gate: both pass before step 3. If the norm-recompute is not bitwise identical, report the numbers; do not loosen `VLA_NORM_ATOL`.
3. **Pilot collection**: one debug job per seen task, 10–20 episodes each (`EP_START=0 EP_END=<N>`). Choose N so a job fits in the walltime with the 20% margin: run `project_collection.py` on the smoke audit JSON (`--partition debug`) and use its `episodes_per_job`. Keep `ATTN` identical for every job. Set `TOKEN_LEVEL_EPS=1` for at most one or two tasks (verification episodes; large) so pooling check #4 has data.
4. **Resume-safe reruns**: if a job times out, is preempted or is requeued, **re-submit the identical command**. The collector skips episodes whose `.pt` validates, reclaims locks of dead jobs, repairs missing sidecars, and removes only temp files older than 24 h with no live owner. Never `rm` anything in `$OUT_ROOT`; bad files are moved to `$OUT_ROOT/_quarantine/` automatically. An in-flight episode is lost at timeout (restarts from its first step); sizing with the margin keeps this rare. Do **not** run two jobs with overlapping episode ranges on purpose (locks make it safe, but it wastes GPU time).
5. **Audit + projection** (CPU, login node OK; output **outside** `$OUT_ROOT`):
   ```bash
   D=$(date +%Y%m%d); mkdir -p $HOME/audits
   python experiments/robot/libero/audit_collection.py $OUT_ROOT --num-tasks 10 --n-episodes <N> --expect-seed 7 \
       --json $HOME/audits/pilot_$D.json > $HOME/audits/pilot_$D.md; echo "audit exit=$?"
   python experiments/robot/libero/project_collection.py $HOME/audits/pilot_$D.json --episodes-per-task 50 --partition debug --json $HOME/audits/proj_$D.json
   ```
   `--num-tasks` should cover the pilot's task ids; tasks not in the pilot show as `MISSING_TASK` (expected, note it). Use `--fast` for quick progress checks while jobs run (sidecars only; cannot see corrupt `.pt`).
6. **Fill `docs/archive/pilot_report.md`** (checks #1–#12) from: T6 output (#1–#3), the `token_level_sample` verification episodes (#4), the audit Markdown (#5–#8, #10–#12), parity output (#9). Record date, GPU type, commit for every number (hard rule 7).

## 3. Pass / stop criteria
- All pilot checks pass → report to Akash and **wait**; A8 (full collection) needs explicit approval.
- Any of these → stop and report, do not "fix" by deleting or editing data: audit flags `FAILED_VALIDATION`, `SIDECAR_MISMATCH`, `INCONSISTENT_META`, `DUPLICATE_EPISODES`, `MISSING_CLASS` on a pilot task, `TOKEN_OUT_OF_RANGE` / `TOKEN_*_EDGE` above threshold; a parity or T6 failure; `INVALID_EPISODE` (read the exception type; a deterministic crash is a bug, not a failure episode).
- Invalid episodes are never counted as failures.

## 4. Known limitations / open questions *(for the reviewer)*
- Slurm option spelling (`--gpus=a40:1`, `-p debug`), CPUs/memory, and whether `debug` allows an A40 *(UNVERIFIED)*.
- `draccus` list syntax `--task_ids "[3]"` for `Optional[List[int]]` *(UNVERIFIED)*; if it fails, use `--task_ids=[3]`.
- The collector's `inputs` (A2, lossless images) and several metadata fields (A3) are still TODO; the pilot storage projection will use assumed values for inputs until A2 lands.
- The CPU dry run (`tests/test_collect_dryrun.py`) exercises collector → files → audit with a mock env and a tiny Llama; it does not cover OpenVLA's multimodal path, flash-attn numerics, or real LIBERO.
