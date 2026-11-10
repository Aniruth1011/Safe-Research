# Repo map: the two repositories, the branches that matter, and where things live

Written 2026-11-10. Sources: `git remote -v`, `git branch -a`, `git log --oneline -5 <branch>` and `git tag`, run in
both local clones on 2026-11-10; the files themselves; the orchestrator's shared context; the team task plan
(`VLA_Task_Plan_v5`, sheet "Task Plan"). Commit hashes are short hashes as of that date. For live state, read
[`00_STATUS.md`](00_STATUS.md).

In one sentence: **the research code lives in `Safe-Research`, and the code that runs the 7B policy lives in a
separate `openvla` fork. Rhea's kit connects them with a patch file.**

## 1. The two repositories

| Repo | Our fork (remote `origin`) | Upstream (remote `upstream`) | What it holds |
|---|---|---|---|
| Safe-Research | `github.com/akash-ravi/Safe-Research` | `github.com/Aniruth1011/Safe-Research` (the team repo, public) | SAFE's detector code, the team's framework, Rhea's collection kit, tests, docs |
| openvla | `github.com/akash-ravi/openvla` | `github.com/vla-safe/openvla` (SAFE's OpenVLA fork; push disabled locally) | SAFE's rollout script `experiments/robot/libero/run_libero_eval.py`, which runs the frozen policy in LIBERO |

- Akash has **read** access to the team repo, so every change goes there as a PR from the fork. Aniruth merges.
- The openvla fork is pinned at **`300dce2`** ("add safe changes"), which is also `upstream/main` there as of 2026-11-10.
  We never push to `vla-safe/openvla`.

## 2. Safe-Research branches (checked 2026-11-10)

| Branch | Tip | What it is | Status |
|---|---|---|---|
| `upstream/main` | `40a9f81` | PR #1 (Python 3.10 pin, `b76ea64`) + **PR #2, Rhea's collection kit** (`a2741c0`, `a7c85db`, `e41c612`). Merged 2026-11-09 18:24 Pacific (= 2026-11-10 UTC), without the five fixes. | The team's truth. Never rewritten. |
| `main` (local and fork) | `40a9f81` | Equal to `upstream/main`. The fork's `main` was pushed to match on 2026-11-10 (plan row G0). | Keep in sync; no own commits. |
| `akash/extraction` | `11655fa` | `40a9f81` + 4 commits: `e65c6e8` env docs (`docs/env/`), `7ac65e4` T6 GPU test (`tests/gpu/`), `fa9f49d` `docs/compute_notes.md`, `11655fa` archived pilot docs (`docs/archive/`). | Docs branch. This file's folder (`docs/orchestrator/`) sits here, uncommitted. |
| `akash/kit-fixes` | `08b883e` | `40a9f81` + `ec452b0` (`collection/constraints.txt`) + `08b883e` (T6, `tests/gpu/`). | Base of the follow-up fix PR (plan row A1). |
| `backup/extraction-pre-rebase-2026-11-10` | `ad558cc` | The old `akash/extraction` before it was rebased onto PR #2: 13 commits on top of `88d9c50` (the fork's pre-PR #2 base). | Safety copy only. Do not build on it. |

Also present locally: `wip/fix1` … `wip/fix5` (one per must-fix, all at `08b883e` when checked). `akash/setup-py`
(PR #1) is merged and deleted, locally and on the fork. There are no tags in this repo.

## 3. openvla branches (checked 2026-11-10)

| Branch / tag | Tip | What it is | Status |
|---|---|---|---|
| `upstream/main` | `300dce2` | SAFE's OpenVLA fork. | Pinned base. |
| `akash/kit-patch` | `b5cb926` | `300dce2` + Rhea's patch as one commit (author: Rhea). Touches exactly two files: `experiments/robot/libero/layer_cache.py` (new, 117 lines) and `run_libero_eval.py` (+36/−1). | Patch-side fixes go here as commits on top. |
| `akash/collector` = tag `archive/collector-2026-11` | `7b40c4c` | Akash's own collector, built on `300dce2`. | **Retired as a pipeline.** Its tested pieces (feature capture, resume/audit code, mock-env tests) may be reused. Do not extend the branch. |

`wip/ov-fix3`, `wip/ov-fix4`, `wip/ov-fix5` are the local work branches for the patch-side fixes.

## 4. How the kit patch is produced

The kit ships `collection/openvla_all_layers.patch` in Safe-Research. It is **generated, never hand-edited**:

```bash
cd openvla-fork            # on akash/kit-patch, with any fix commits on top
git diff 300dce2 HEAD > ../Safe-Research/collection/openvla_all_layers.patch
```

Checked 2026-11-10: `git diff 300dce2 akash/kit-patch` is byte-identical to the patch file on `upstream/main`.
Anyone can check that the patch still applies with `git apply --check` on a clean `300dce2` checkout; the kit's setup
script does exactly that (`collection/carc/00_setup_env.sh:24-27`).

To change what runs on the GPU: commit on `akash/kit-patch` (or a `wip/ov-fixN` branch merged into it), regenerate the
patch, and commit the regenerated file in Safe-Research.

## 5. Safe-Research directory map

| Path | What lives there | Who owns it |
|---|---|---|
| `failure_prob/model/`, `failure_prob/conf/`, `failure_prob/utils/`, `failure_prob/train.py` | SAFE's original detector code, Hydra configs and helpers | **SAFE's original files. Nobody edits them.** |
| `failure_prob/data/` | SAFE's originals (`openvla.py` with `split_rollouts`, `pizero*.py`, `open_pizero.py`, `utils.py`) **plus** three team files added in `88d9c50`: `feature_store.py`, `adapters.py`, `synthetic.py` | SAFE's files: nobody edits. The team files: the pipeline owners. |
| `failure_prob/models/` | Team detectors: `single_layer_mlp`, `safe_mlp`, `safe_lstm`, `concat_fusion`, `gated_fusion`, `residual_fusion`, `registry.py` | Team (fusion: Dhyan, Aniruth) |
| `failure_prob/engine/` | Team trainer, metrics, run tracking (`tracking.py` made Windows-safe in PR #2) | Team |
| `failure_prob/protocol.py` | Fixed seeds and split constants (added in PR #2) | Rhea |
| `collection/` | Rhea's kit: the patch file, `build_cache.py`, `check_cache.py`, `load_cache.py`, `carc/` Slurm scripts, `README.md`; on `akash/kit-fixes` also `constraints.txt` | Rhea (kit); Akash (fix PR) |
| `scripts/` | `train.py`, `evaluate.py`, `probe_layers.py` (synthetic-only today), `list_models.py`; SAFE's originals `eval_conformal_figure.py`, `get_wandb_metrics.py`, `visualize_features.py`, `batch_training/` | Team; SAFE's originals untouched |
| `configs/`, `experiments/` | Experiment and model YAMLs, experiment index and manifests | Team |
| `tests/` | `test_framework.py`, `test_shared_cache.py` (kit tests on fake rollouts), `test_tracking_portable.py`; `tests/gpu/` (T6, GPU only) on `akash/kit-fixes` and `akash/extraction` | Team; new kit tests are `tests/test_kit_<topic>.py` |
| `docs/` | On `main`: six framework docs (`architecture.md`, `experiment_protocol.md`, ...). On `akash/extraction`: also `env/`, `compute_notes.md`, `archive/`, and `orchestrator/` (uncommitted) | Team; Akash for the extra folders |

Measured 2026-11-10 (macOS CPU, Python 3.10.22, `upstream/main` `40a9f81`): `python -m pytest -q` = 17 passed.

## 6. Gotcha: directories named `env` are ignored by git

`.gitignore` (lines 105–111) ignores Python virtualenv folders, including `env/` (line 107) and `ENV/` (line 109).
Checked 2026-11-10 with `git check-ignore -v --no-index`:

- `env/` matches any directory named `env` **on every OS** (git matches it case-sensitively on Linux too).
- On macOS git runs with `core.ignorecase=true`, so `ENV/` additionally matches `env`, `Env`, and so on.

`docs/env/` is tracked only because its files were force-added. Any **new** file under a directory called `env` is
silently skipped by `git add`. Do not create directories named `env`; that is why the pin file is
`collection/constraints.txt`, not under an `env/` folder.

## 7. Local-only folders

The working copies under `CS566/worktrees/` and the Python 3.10 virtualenvs next to the repos exist only on Akash's
laptop; they are not in either repo and nothing here depends on them.
