# Runbook: the full collection on Modal, pulled to CARC

Written 2026-11-10 for whoever runs the collection (Akash). One A10 per task, waves of 5 tasks, two Modal
workspaces: **`friend`** (main) and **`akashrav`** (backup), each with a $30 cap. Commands run from the root of the
Safe-Research checkout at the commit named in PR #3, unless the step says CARC. Measured smoke numbers:
[`00_STATUS.md`](00_STATUS.md).

**Rules.** Refer to workspaces by profile name only. Never print, paste, log or commit a token; `~/.modal.toml`
lives in your home directory, never in a repo. Never delete anything from a Modal Volume until Akash confirms that the
cache on CARC verified. Stop and ask Akash after every wave.

## 1. Profiles

- `modal profile list` shows the profile names (never the secrets). Pick the workspace **per command**:
  `MODAL_PROFILE=friend modal run ...` for `modal run`, `--profile friend` for `modal volume ...`. Do not switch the
  global default (`modal profile activate`) in the middle of a run.
- Adding a workspace: its owner creates a token in that workspace; you type
  `modal token set --token-id <id> --token-secret <secret> --profile friend` yourself, in your own terminal.

## 2. Once per workspace, before its first GPU run (CPU only, cents)

1. `MODAL_PROFILE=friend modal run collection/modal_app.py::download_model`. This builds the image (pinned Modal
   image builder 2025.06, Python 3.10, the pins in `collection/constraints.txt`, openvla `300dce2` + the kit patch),
   downloads the checkpoint into this workspace's `cs566-hf-cache` Volume and copies the image provenance into
   `cs566-collection:work/image_provenance`.
2. Check that both workspaces built the identical image:
   `modal volume get --profile friend cs566-collection work/image_provenance prov-friend`, the same for `akashrav`,
   then `diff -r prov-friend prov-akashrav` must print nothing. (The CARC build job checks this again.)
3. **Only a smoke-validated image may run full-run tasks.** An image is identified by its content fingerprint
   (`image_fingerprint.txt`: hash of pip freeze, OS packages, Python, repo commits, patch and builder version), because
   Modal image IDs differ per workspace. A full smoke test (`MODAL_PROFILE=<p> modal run collection/modal_app.py::smoke
   --full`) records its fingerprint as validated; the fingerprint of the passing smoke run is also committed in
   `collection/smoke_validated_images.json`. The launcher (on CPU, before any GPU) and every task refuse an image whose
   fingerprint is in neither place. If the `diff -r` above shows a difference, the other workspace's image is a
   different image: it needs its own full smoke test (about $1, Akash's OK) before it runs any task. Each
   `TASK_META.json` records the fingerprint, the Modal image ID and which smoke pass validated it.

## 3. Launch a wave

```
MODAL_PROFILE=friend modal run --detach collection/modal_app.py::rollouts \
    --tasks 0-4 --sec-per-episode <smoke s/episode> --confirm-full-run
```

- At most 5 tasks per launch (enforced). Plan: wave 1 = tasks 0–4, wave 2 = tasks 5–9.
- The launch refuses without `MODAL_PROFILE` or from a modified tree, and prints the cost estimate first.
- Each task writes `rollouts/all-layers-v1/taskNN/` in that workspace's Volume: SAFE's files, `.layers.pt`, invalid
  markers, `TASK_META.json` (workspace, GPU, kit commit, every attempt's container and seconds) and, once complete,
  `SHA256SUMS`. Relaunching the same tasks in the same workspace resumes them (`--resume`).
- Timeout per task = 1.5 × the worst-case task time measured in the smoke test: 1.5 × (30 s + 50 × 191.6 s) =
  14,415 s (4.0 h). Expected per task, from the smoke measurements: 1.4–2.7 container-hours (all-success to
  all-failure), about **$1.80–3.50 per task, $9–18 per wave of 5**; if every container hit the timeout, about $27 per wave.
- Use `--sec-per-episode 152` for the launch estimate. That figure is an estimate: 0.363 s/step at about 400 steps per
  episode, assuming the published ~54% LIBERO-10 success rate.
- **Resume changes trajectories** (smoke test, 2026-11-11): an episode run in a fresh process after `--resume` can
  differ from the same episode run right after the previous one. Each attempt in `TASK_META.json` records its
  `first_episode`. Report any resumed task to Akash.

## 4. Check spend (after every wave, before asking for the next)

- `MODAL_PROFILE=friend modal run collection/modal_app.py::status` prints per task: status, episodes, invalid,
  workspace, GPU, container-hours, and dollars, then the spend per workspace and the projected cost of the
  unfinished tasks. Rate: A10 $0.000306/s + 2 cores + 16 GiB = **~$1.32 per container-hour** (modal.com/pricing,
  2026-11-10), plus ~3 min per container for start-up (estimate).
- Compare with the workspace's **Usage** page on modal.com (the source of truth), and report both to Akash.
- If the projection exceeds what is left of that workspace's $30: stop and ask Akash.

## 5. Pull finished tasks to CARC (CPU job; Akash runs it)

- Once on CARC: install the Modal CLI for your user (`uv tool install modal`), add **both** profiles with
  `modal token set ... --profile friend` and `... --profile akashrav`, then `chmod 600 ~/.modal.toml`.
- From `collection/`: `sbatch --account=<account> --export=ALL,PROFILE=friend,TASKS=0-4 carc/04_pull_from_modal.slurm`.
  **Always in a CPU job, never on the login node** (the script refuses outside Slurm).
- Each task lands in `/scratch1/<user>/rollouts/all-layers-v1/taskNN/` only if its checksums match, `TASK_META` says
  complete on an NVIDIA A10 in that profile, and every episode has its files or an invalid marker. Failures leave
  nothing in place and are listed in `pull_log.tsv`.

## 6. Build and verify the cache on CARC (CPU job)

`sbatch --account=<account> carc/05_build_cache_from_modal.slurm` once all 10 tasks are pulled. It refuses if a
task folder is missing or the image provenance differs between workspaces, links the task folders into one flat
view, then runs the capture check, `build_cache.py` (which also refuses mixed GPU, attention or render backend, and
checks invalid markers) and `check_cache.py verify`. Then tell Akash; only after his OK may Volumes be cleaned.

## 7. A credit limit is hit mid-wave

1. Containers in that workspace stop. Their tasks keep `TASK_META` status `running` (or `partial`); `status` shows
   which. Do not relaunch in that workspace.
2. Pull the tasks that **are** complete from that workspace (section 5). The puller refuses the unfinished ones.
3. In the other workspace: run section 2 (download, image check) if not done yet, then launch **only the unfinished
   tasks**, e.g. `--tasks 7,9`. They start from scratch there (each workspace has its own Volume). A task is never
   continued across workspaces; `rollout_task` refuses that.
4. Pull them from the other profile. Each task's `TASK_META.json` records the workspace and GPU that produced it, and
   the cache manifest keeps all of them. The kit's mixing guard still requires A10 everywhere.
5. Leave the abandoned partial files in the old Volume until Akash confirms the CARC cache verified.
