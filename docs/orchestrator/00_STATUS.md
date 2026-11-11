# Status board

Last updated: **2026-11-10** (orchestrator). Read this first; the other files in this folder explain the why.

## Where we are

| Phase | What | State |
|---|---|---|
| 0a | Git cleanup (plan row G0) | **Done and pushed 2026-11-10** |
| 0 | Understanding pack (`01`–`06` in this folder) | **Done 2026-11-10** |
| 1 | The five kit fixes, as a follow-up PR | **PR [#3](https://github.com/Aniruth1011/Safe-Research/pull/3) ready for review (2026-11-11)**, head `90f8007` |
| 2 | Modal smoke test (≤ $3) | **PASSED 2026-11-11** on A10 (`akashrav`), image `0f65545c…` validated. Spend ≤ $0.54 |
| 3 | Full collection (500 episodes) | Out of scope until Akash approves, the team answers the checklist, and the fix PR is merged |

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

1. **Wave 1 approved (tasks 0–4, `friend`, hard stop at $28), but the `friend` Modal profile is not configured on
   this Mac** (`modal profile list` shows only `akashrav`, 2026-11-11). Akash adds it himself; then: download (CPU),
   image check, launch.

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

1. On approval: run `download_model`, then the full smoke test (`--full`); report results and actual spend.
2. Smoke passes → set the rollout timeout from it, add results to PR #3, mark it ready for review.
3. Full collection (A8) still needs: PR #3 merged, the team's checklist answers, and Akash's approval of wave 1.

## Budget

| Item | Spent | Notes |
|---|---|---|
| Modal (`akashrav`) | **≤ $0.54** of $30 | Smoke test, 2026-11-11 (upper bound) |
| Modal (`friend`) | $0.00 of $30 | Full run planned here |
| CARC | 0 SU by the orchestrator | No CARC jobs will be submitted |

## Contradictions and surprises (2026-11-10)

- The brief says "L4" in three places but its Compute section says "A10 by default". **Resolved:** Akash confirmed A10 (2026-11-10); plan v5 says A10 everywhere.
- The local `upstream/main` was stale (`88d9c50`); after fetching it is `40a9f81`, as the brief says.
- `protocol.py` lives in `failure_prob/`, not in `collection/`.
- The repo's `.gitignore` contains `env/` (and `ENV/`), so any directory named `env` is ignored on every OS, including `docs/env/`; those files need `git add -f`. New files avoid directories named `env`.
- The current plan is `VLA_Task_Plan_v5.xlsx` (in the project folder since 2026-11-10). The review doc still cites v2.
- The CARC log is now `docs/compute_notes_carc.md` (scrubbed); `docs/compute_notes.md` is a different, older file.
- The kit's setup script tells users to switch to `sdpa` if flash-attn fails to install. Fix 3 removes that.
