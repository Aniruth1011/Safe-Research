# Status board

Last updated: **2026-11-10** (orchestrator). Read this first; the other files in this folder explain the why.

## Where we are

| Phase | What | State |
|---|---|---|
| 0a | Git cleanup (plan row G0) | **Done and pushed 2026-11-10** |
| 0 | Understanding pack (`01`–`06` in this folder) | **Done 2026-11-10** |
| 1 | The five kit fixes, as a follow-up PR | In progress (F1–F5 restarted 2026-11-10; first attempt left nothing) |
| 2 | Modal smoke test (≤ $3) | Not started. Needs Phase 1 + Akash's approval |
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

## In progress

| Subagent | Task | Effort |
|---|---|---|
| F1 | Modal app (A10), portable smoke script, CARC job syntax | high |
| F2 | Pins in every install path, uv fallback, flash-attn install hard-fail | high |
| F3 | Run-time attention guard, run fingerprint, mixed-dataset refusal | high |
| F4 | Invalid episodes: marker, resume, cap; cache build accepts them | high |
| F5 | Raw block 32 as layer index 33 | high |

## Blocked / waiting on Akash

1. **IDs in public git history (found 2026-11-10 by D4).** The CARC account ID and user name are in files that were
   public on the fork BEFORE today: `docs/env/env_freeze.txt`, the two archived pilot docs (Safe-Research), and
   `experiments/robot/libero/{slurm/collect_task.sbatch,audit_collection.py}` (OpenVLA `akash/collector` = tag
   `archive/collector-2026-11`). The tip of `akash/extraction` is scrubbed (`7ef726d`, pushed). Old commits, the fork
   `backup/extraction-pre-rebase-2026-11-10` branch and the collector branch/tag still contain them. Removing them
   needs a history rewrite + force-push / delete (Akash's decision). `main` and `akash/kit-fixes` are clean.
2. **Budget headroom (D4's estimate, not measured).** Top of the A10 range is ~$26–28 for the full run; with the
   ≤ $3 smoke test this crosses the $25 stop-and-ask line, and the worst case touches the $30 ceiling. Image builds,
   Volume storage and the CPU cache build are not counted. The smoke test (A5/A6) must settle this before A8.

Next gate: approval to open the fix PR.

## Next

1. Review and integrate the subagents' work; run every test on Python 3.10.
2. Gate: all tests pass → ask Akash to approve opening the fix PR (tag Rhea and Aniruth).
3. Phase 2 estimate and approval request for the Modal smoke test.

## Budget

| Item | Spent | Notes |
|---|---|---|
| Modal | **$0.00** of $30 | Nothing launched. Stop-and-ask line: $25 |
| CARC | 0 SU by the orchestrator | No CARC jobs will be submitted |

## Contradictions and surprises (2026-11-10)

- The brief says "L4" in three places but its Compute section says "A10 by default". **Resolved:** Akash confirmed A10 (2026-11-10); plan v5 says A10 everywhere.
- The local `upstream/main` was stale (`88d9c50`); after fetching it is `40a9f81`, as the brief says.
- `protocol.py` lives in `failure_prob/`, not in `collection/`.
- The repo's `.gitignore` contains `env/` (and `ENV/`), so any directory named `env` is ignored on every OS, including `docs/env/`; those files need `git add -f`. New files avoid directories named `env`.
- The current plan is `VLA_Task_Plan_v5.xlsx` (in the project folder since 2026-11-10). The review doc still cites v2.
- The CARC log is now `docs/compute_notes_carc.md` (scrubbed); `docs/compute_notes.md` is a different, older file.
- The kit's setup script tells users to switch to `sdpa` if flash-attn fails to install. Fix 3 removes that.
