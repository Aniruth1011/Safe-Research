# Status board

Last updated: **2026-11-11** (orchestrator). Read this first; the other files in this folder explain the why.

## Where we are

| Phase | What | State |
|---|---|---|
| 0a | Git cleanup (plan row G0) | **Done and pushed 2026-11-10** |
| 0 | Understanding pack (`01`–`06` in this folder) | **Done 2026-11-10** |
| 1 | The five kit fixes, as a follow-up PR | **PR [#3](https://github.com/Aniruth1011/Safe-Research/pull/3) ready for review (2026-11-11)**, head `90f8007` |
| 2 | Modal smoke test (≤ $3) | **PASSED 2026-11-11** on A10 (`akashrav`), image `0f65545c…` validated. Spend ≤ $0.54 |
| 3 | Full collection (500 episodes) | **Done 2026-11-11** (plan rows A5, A6, A8): 10/10 tasks on NVIDIA A10, pulled to CARC, cache `v1` verified |

## Done

- **Verified the repo state against the brief** (2026-11-10).
- **PR #2's test suite, first run by us:** 17 passed on Python 3.10.22 (macOS CPU, torch 2.14.1, `upstream/main` `40a9f81`, 2026-11-10).
- **Retired collector's tests still pass:** 36 passed, 1 skipped (Python 3.10, torch 2.2.0, transformers 4.40.1, `akash/collector` `7b40c4c`, 2026-11-10).
- **Git cleanup, pushed 2026-11-10 with Akash's OK:**
  - Fork `main` fast-forwarded to upstream `40a9f81`.
  - `backup/extraction-pre-rebase-2026-11-10` (`ad558cc`, the old tip) pushed to the fork first, as insurance.
  - `akash/extraction` force-pushed (`--force-with-lease`, old fork tip `b684948` is contained in the backup): upstream/main + 4 commits, then the CARC log.
  - `akash/kit-fixes` pushed (upstream/main + `collection/constraints.txt` + T6).
  - `akash/setup-py` deleted locally and on the fork (it was merged upstream).
  - OpenVLA fork: `akash/kit-patch` (`b5cb926` = `300dce2` + Rhea's patch, Rhea as author) and tag `archive/collector-2026-11` pushed.
- **CARC log committed** as `docs/compute_notes_carc.md`, scrubbed for the public repo (no account/project IDs, user names, host names, Slurm job IDs or SU balances).
- **Decisions confirmed by Akash (2026-11-10):** A10 is the single GPU type, Modal only. `VLA_Task_Plan_v5.xlsx` is current (v4 outdated).
- **Tooling on the laptop:** `uv`, two Python 3.10 virtual environments (`.venv-kit310`, `.venv-ov310`).

## Phase 1 result (2026-11-10, CPU, laptop)

- `akash/kit-fixes` @ `6b50ca8` (pushed): all five fixes merged; `collection/openvla_all_layers.patch` regenerated from
  openvla `akash/kit-patch` @ `bf38c80` (pushed) with `git diff 300dce2 akash/kit-patch`; it applies cleanly to `300dce2`.
- **Kit tests:** `pytest -q` = **111 passed, 2 skipped** (Python 3.10.22, `.venv-kit310`). Skips: T6 (needs a GPU) and the
  Modal import test (`modal` not in that venv; run by hand with the Modal CLI's Python: the app builds 4 functions and 3 entrypoints, nothing sent to Modal).
- **Patch-side tests:** `pytest -q tests/` = **67 passed** (Python 3.10, torch 2.2.0, transformers 4.40.1, `.venv-ov310`), including one test that runs fixes 3+4+5 together in the real episode loop.
- Not testable on a laptop (the smoke test checks them): EGL on Modal, flash-attn in the Modal image, the real OpenVLA config reporting `flash_attention_2`, A10 speed.
- Incident: the previous session's F2 agent revived after its usage limit and wrote into `sr-fix2` at the same time as ours; it was stopped, and the two versions were reconciled file by file (F2b).

## Blocked / waiting on Akash

Nothing. Collection is complete (see "Cache v1" below).

## Cache v1 (CARC, built and verified 2026-11-11)

- All 10 tasks pulled and verified (`friend` 0–7, `akashrav` 8, 9), then `05_build_cache_from_modal.slurm`.
- `check_cache.py verify`: **341 files, 0 corrupted**. Splits: **train 210 / seen_val 140 / unseen_test 150** episodes
  (unseen tasks 0, 3, 5); **198,600 rows** (one per step).
- Location: `CACHE_ROOT/v1`, with `CACHE_ROOT` now defaulting to `/scratch1/<user>/safe_cache` (kit `2db85ab`; an
  exported `CACHE_ROOT` overrides it). `/scratch1` is not backed up.
- Modal Volumes (Akash, 2026-11-11): tasks 0–9 **kept** on both Volumes as a backup, plus `smoke/`, `work/` and
  `cs566-hf-cache`. Only wave 1's discarded A10G run of task 2 was deleted from `friend`.

## Wave 2 result (kit `6cb1781`, 2026-11-11 05:36–09:41 UTC)

Task 2 rerun from episode 0 plus tasks 5, 6 in `friend`; tasks 8, 9 in `akashrav`; then task 7 in `friend` (rule:
friend only if its headroom under the $28 stop is at least $3.60; it was $28 - $22.03 = $5.97).

| Task | Workspace | Outcome (success/failure) | Steps | Container h | Cost (measured) | GPU | Fingerprint |
|---|---|---|---|---|---|---|---|
| 2 | `friend` | 25 / 25 | 20,287 | 2.03 | $2.68 | A10 | match |
| 5 | `friend` | 37 / 13 | 14,578 | 1.49 | $1.97 | A10 | match |
| 6 | `friend` | 22 / 28 | 20,413 | 2.09 | $2.76 | A10 | match |
| 7 | `friend` | 34 / 16 | 19,049 | 1.97 | $2.60 | A10 | match |
| 8 | `akashrav` | 10 / 40 | 25,348 | 2.54 | $3.37 | A10 | match |
| 9 | `akashrav` | 26 / 24 | 20,402 | 2.07 | $2.74 | A10 | match |

All 50/50 episodes per task, 0 invalid, one uninterrupted attempt from episode 0 each; failures always ran to the 520
cap. The per-container GPU check saw only `NVIDIA A10` in wave 2 (no re-requests). Task 2's wave-1 A10G output is in
`friend`'s Volume under `_discarded/` (recorded in its `TASK_META.json`).

**Whole collection:** 500 episodes, 263 success / 237 failure (52.6%), 198,600 steps, about 166 GB of `.layers.pt`.
Pull plan: `friend` tasks 0–7, `akashrav` tasks 8, 9 (one pull job at a time), then `05_build_cache_from_modal.slurm`.

## Wave 1 result (`friend`, kit `8300892`, 2026-11-11 03:03–05:24 UTC)

| Task | Outcome (success/failure) | Steps | Container h | Cost (measured) | GPU | Fingerprint |
|---|---|---|---|---|---|---|
| 0 | 32 / 18 | 19,386 | 1.98 | $2.62 | A10 | match |
| 1 | 37 / 13 | 17,250 | 1.79 | $2.36 | A10 | match |
| 2 | 26 / 24 | 19,875 | 2.29 | $3.04 | **A10G** | match |
| 3 | 23 / 27 | 20,591 | 2.08 | $2.76 | A10 | match |
| 4 | 17 / 33 | 21,296 | 2.17 | $2.87 | A10 | match |

All 50/50 episodes per task, 0 invalid, 0 discarded, one uninterrupted attempt each. Failures always ran to the 520 cap.
**Task 2 is not usable** (another GPU type): do not pull it; the puller and build_cache refuse it anyway. Modal served
`gpu="A10"` with an A10G for 1 of 5 containers; the run-level guard missed it (simultaneous container start).
Fixed in `6cb1781`: every container requires `NVIDIA A10` before touching outputs; the launcher re-requests.

Spend: `friend` ~$13.9 (function time + start-up), upper bound $14.34 (watchdog). `akashrav` ≤ $0.55.

## Decisions (Akash, 2026-11-11)

- **No resume for the dataset.** Every task runs as one uninterrupted process (episodes 0–49). Implemented in
  `190a3d5`: partial outputs are moved to `_discarded/`, never deleted, and the task restarts at episode 0;
  engineering-only resume flag; puller refuses anything else. Tests pass on torch 2.14 and 2.2.0. The image is
  unchanged (`akashrav` fingerprint still `0f65545c…`), so no new smoke test.
- Profile name `akashrav` stays (no alias).
- Wave 1 approved in `friend`, hard stop at $28 spent in that workspace. Report each task as it finishes, then
  both workspaces' spend and a wave-2 recommendation; wait for OK before wave 2.

## Smoke test result (Modal A10, 2026-11-11, kit `99adba5`, task 1, `--full`)

- Checks: environment (pins, flash-attn, EGL, LIBERO) pass; model load pass (7.9 s, 14.09 GiB, `flash_attention_2`);
  rollout + resume pass; capture 2/2 pass; repeat + `compare-runs` ran (1 flip, explained by resume); tiny build + verify
  pass (108 files).
- **MEASURED (A10):** 0.363 s/step (520-step episode); 98.5–191.6 s per episode (268–520 steps); peak GPU memory
  16,301 MiB (`nvidia-smi`); `.layers.pt` 0.836 MB/step; ~0.89 MB/step for all files. About 1.2× the A40's s/step.
- First attempt failed on a kit bug (torch 2.2 `mmap` needs a `str` path); fixed and retried once, as approved.
- **Spend (akashrav), upper bounds from container lifetimes × published rates:** ≤ $0.54. Compare with the dashboard.
- Image builder pinned to 2025.06 (2026.10 rejects Python 3.10). Validated image fingerprint
  `0f65545c78b2372a4765db3ba616b759e5c5529e5daceda31cd2eea05dc9faf7`, committed in `collection/smoke_validated_images.json`.

## Decisions (Akash, 2026-11-10)

- **PR #3** stays a draft until the Modal smoke test passes; then add the smoke results to it and mark it ready. Mention
  only `@Aniruth1011`; do not tag Rhea.
- **No history rewrite.** The CARC account ID and user name are not credentials; rewriting shared history is riskier.
  **Delete the fork's `backup/extraction-pre-rebase-2026-11-10` after PR #3 merges.**
- **Full-run budget plan:** recheck the A10 price (done: $0.000306/s ~ $1.10/h, 2026-11-10); set each task function's
  timeout to the measured smoke timing x 1.5 (not 4 h); run the collection in **two waves of 5 tasks** and report spend
  after wave 1; if the projected total exceeds what is left of the $30, stop and ask (Akash will switch to a teammate's
  workspace for the rest).

## Next

1. Team: confirm read access to the cache on CARC (it is under Akash's `/scratch1`; copy to `/project2/<group>` if needed).
2. Depth sweep R1–R4: [`08_DEPTH_SWEEP.md`](08_DEPTH_SWEEP.md). A TEMPORARY R1 run (placeholder probe and
   metric, not a result) completed on CARC on 2026-11-11: 34 layers × 3 seeds. The sweep now also has bootstrap CIs,
   a per-task breakdown and the S5 all-layer average (`--layers avg`; definition for Rhea's review in
   [`09_S5_ALL_LAYER_AVERAGE.md`](09_S5_ALL_LAYER_AVERAGE.md)). Waiting for B6 (probe spec) and B4 (evaluator).
   No CARC runs until Akash says. SAFE reference line: only from C4 (Adarsh's SAFE model re-scored by our evaluator).
3. After PR #3 merges: delete the fork's `backup/extraction-pre-rebase-2026-11-10`.

## Budget

| Item | Spent | Notes |
|---|---|---|
| Modal (`akashrav`) | ~$6.2 (≤ $6.90) of $30 | Smoke test + tasks 8, 9, 2026-11-11 (watchdog upper bound) |
| Modal (`friend`) | ~$24.3 (≤ $24.80) of $30 | Tasks 0–7 incl. the discarded A10G run of task 2, 2026-11-11 |
| CARC | 0 SU by the orchestrator | No CARC jobs will be submitted |

## Contradictions and surprises (2026-11-10)

- The brief says "L4" in three places but its Compute section says "A10 by default". **Resolved:** Akash confirmed A10 (2026-11-10); plan v5 says A10 everywhere.
- The local `upstream/main` was stale (`88d9c50`); after fetching it is `40a9f81`, as the brief says.
- `protocol.py` lives in `failure_prob/`, not in `collection/`.
- The repo's `.gitignore` contains `env/` (and `ENV/`), so any directory named `env` is ignored on every OS, including `docs/env/`; those files need `git add -f`. New files avoid directories named `env`.
- The current plan is `VLA_Task_Plan_v5.xlsx` (in the project folder since 2026-11-10). The review doc still cites v2.
- The CARC log is now `docs/compute_notes_carc.md` (scrubbed); `docs/compute_notes.md` is a different, older file.
- The kit's setup script tells users to switch to `sdpa` if flash-attn fails to install. Fix 3 removes that.
