# Shared feature cache: OpenVLA-7B on LIBERO-10, all 33 hidden-state layers

This kit runs OpenVLA once on CARC and saves **every** hidden-state layer at every step. It then packages the result into a versioned, checksummed cache with a frozen split, so all five of us train detectors on identical features and labels.

## What's in the kit

| File | What it does |
|---|---|
| `openvla_all_layers.patch` | Patch for `vla-safe/openvla`. Adds `--save_all_layers True`, which writes `task*--ep*--succ*.layers.pt` = (T, pools, 33, 4096) bfloat16 next to SAFE's usual files. SAFE's own `.pkl`/`.csv` outputs are unchanged, so the original SAFE baselines (Exp 1–3) still run on the same rollouts. |
| `build_cache.py` | Raw rollouts → immutable cache (`features/<pool>/<split>/L<NN>.npy`, `episodes.csv`, `splits.json`, train-only `norm_stats/`, `manifest.json` + `CHECKSUMS.sha256`). |
| `load_cache.py` | Read the cache from Python, or `export` chosen layers into the repo's `FeatureStore` format (one store per split) for `scripts/train.py --feature-root`. `unseen_test` is locked unless `--allow-unseen` (Exp 37 only). |
| `check_cache.py` | `verify` a downloaded copy against checksums; `compare-runs` checks two rollout runs for outcome flips and action divergence. |
| `carc/` | `config.sh` (edit once) → `00_setup_env.sh` → `01_smoke_test.slurm` → `02_rollouts.slurm` → `03_build_cache.slurm`. |
| `../tests/test_shared_cache.py` | End-to-end tests on fake rollouts (no GPU needed); `layer_cache.py` is loaded straight from the patch. |

## Conventions frozen by this cache (agree on these before running)

- **Layer index** = HuggingFace `hidden_states` index, **L0 = embeddings, L1–L32 = decoder layers**:
  - `0` is the embeddings.
  - `1…31` are the outputs of decoder layers 1–31, *before* any normalization.
  - `32` is the output of decoder layer 32 **after the final RMSNorm**. Checked in the source of the pinned `transformers==4.40.1` (`LlamaModel.forward` applies `self.norm` and then appends the last hidden state). This is the same tensor SAFE stores as `s[-1]`, so our final-layer baseline matches SAFE. `check_cache.py capture` confirms it on real data: L32 must equal SAFE's own `.pkl` exactly.
  - So spreadsheet **L4 = index 4, L32 = index 32**.
  - Because only L32 is normalized, raw scales differ across layers.
- **Stored features are unnormalized.** `norm_stats/` holds train-split-only statistics as an *optional* preprocessing step. Whether to use it is a shared evaluator choice, applied identically to every model. `load_cache.py` normalizes only when you pass `normalize=True` / `--normalize`.
- **Pooling over the 7 action tokens:**
  - `mean` = SAFE `token_idx_rel=mean`
  - `first` = SAFE `token_idx_rel=0.0`
  - `last` = SAFE `token_idx_rel=1.0`
- **Split (one fixed protocol):**
  - Unseen tasks are **0, 3, 5**. These are the tasks SAFE's own `failure_prob.data.openvla.split_rollouts` picks after `seed_everything(0)`. I verified this by running SAFE's code (`tests/test_shared_cache.py::test_safe_seed0_task_split`). Seeds 1 and 2 pick {5, 7, 8} and {6, 8, 9}.
  - **Only the task-level split matches SAFE.** SAFE assigns seen episodes to train/val with `torch.randperm` over an *unsorted* `glob`, so even SAFE's own assignment changes with file order. Ours is about 60/40 per seen task (SAFE's ratio), stratified by outcome, and saved explicitly. So it is *not* SAFE's seed-0 train/val split.
  - `splits.json` stores the **exact trajectory IDs**, the **task names**, and this provenance note. Never regenerate the split from a seed; to rebuild, pass `--split-file`.
  - **Label results honestly:** "single fixed split (SAFE seed-0 unseen tasks)". This is not SAFE's mean over 3 splits, so don't put our numbers side by side with SAFE's reported means as if they were the same protocol.
- **Label:** `failure = 1 - success`, per trajectory.
- **Evaluation horizon:** `episodes.csv:task_min_step` is SAFE's per-task minimum rollout length. `records(..., horizon="task_min_step")` truncates to it. The shared evaluator should use it so a model can't win just by noticing that failed episodes run longer.
- **Unseen-task features** are in separate `unseen_test/` folders. Don't distribute them until the locked final run (Exp 37).

## Fixed seeds (`failure_prob/protocol.py`)

| Seed | Value | Used by |
|---|---|---|
| Rollout seed | 7, plus LIBERO's fixed initial states and greedy decoding | `carc/config.sh` `SEED` → `run_libero_eval.py --seed` |
| Split | unseen tasks 0, 3, 5; seen tasks 60/40 train/seen_val, stratified; generated once with split seed 0 | `build_cache.py` defaults → **explicit IDs in `splits.json`** (the real source of truth) |
| Training seeds | 0, 1, 2 | `scripts/train.py --seeds` default; one run per seed |

`tests/test_shared_cache.py` fails if any of these drift apart.

## Running it on CARC

0. Clone this repo on CARC and `cd collection`. Edit `carc/config.sh`: set `CARC_ACCOUNT`, put `CACHE_ROOT` under your group's `/project` directory, and check the GPU type.
1. Set up the environment once, on a login node: `bash carc/00_setup_env.sh`.
   - Clones the fork, applies the patch, installs the pinned versions (torch 2.2.0, transformers 4.40.1), installs LIBERO, and downloads and pins the checkpoint snapshot.
   - LIBERO asks a question on first import. Answer **N**.
2. Run the smoke test (debug partition, under an hour) and **don't submit the full run until it passes**:
   `mkdir -p logs && sbatch --account=$CARC_ACCOUNT carc/01_smoke_test.slurm`
   - **Resume:** runs 1 episode, then asks for 2 with `--resume True`. Episode 0 must not be rerun.
   - **Capture:** L32 must reproduce SAFE's own `.pkl` exactly, every layer must differ from its neighbor, and there must be no NaNs.
   - **Repeat:** an independent second run, compared for outcome flips and action divergence. This is a sanity check, not a reproducibility guarantee.
   - **Build:** builds and verifies a tiny cache, then writes `timing_summary.txt` (seconds per step, MB per step). Share that with the team, and set `--time` in `02_rollouts.slurm` from it.
3. Run the rollouts, 10 parallel jobs (one per task), all on A40:
   `sbatch --account=$CARC_ACCOUNT carc/02_rollouts.slurm`
   - To rerun only unfinished tasks: `--array=3,7`. Resume is automatic.
4. Build and verify the cache:
   `sbatch --account=$CARC_ACCOUNT --dependency=afterok:<array_job_id> carc/03_build_cache.slurm`
   - Refuses to run unless all 500 episodes exist.
   - Runs the capture check on every episode.
   - Copies pip freeze and the OpenVLA, LIBERO and checkpoint versions into `provenance/` *before* checksumming.
   - **Detector training starts only after this job's `verify` passes.**

Each `01`–`03` script reads `carc/config.sh` by relative path, so submit it from inside this folder.

**Storage (rough):** about 0.8 MB per step for 3 poolings × 33 layers. At roughly 350 steps × 500 episodes, that's ~150 GB of raw rollouts, plus about the same again for the cache. Setting `POOLING="mean"` in `config.sh` cuts this ~3×, but drops Exp 14.

## Running on Modal (A10)

`modal_app.py` runs the same collection on [Modal](https://modal.com), on **one A10 per task** (`gpu="A10"`). A10 is the single GPU type for the whole dataset: never mix GPU types across the 500 episodes. Every `modal run` costs money, so agree on the spend first.

- **Image:** CUDA 12.1 devel base, Python 3.10, every `pip install` constrained by `constraints.txt` (the single source of truth for pins; the build fails if a required pin is missing or not honored). `vla-safe/openvla` is checked out at exactly `300dce2` (the build fails otherwise), with `openvla_all_layers.patch` applied by `git apply`. LIBERO is at a pinned commit, with a `.pth` file and a pre-written config, so there is no interactive prompt. flash-attn 2.5.5 is installed last (`CUDA_HOME` set). Rendering is headless EGL (`MUJOCO_GL=egl`).
- **Volumes:** `cs566-hf-cache` holds the checkpoint and the small code files of `openvla/openvla-7b`, so the processor loads offline. `cs566-collection` holds the rollouts, the smoke outputs, the cache and the provenance files. Provenance files can contain host names and job IDs, so never commit anything from the volumes to this public repo.

Run from the repo root:

1. `modal run collection/modal_app.py::download_model` downloads the checkpoint into the volume (CPU only).
2. `modal run collection/modal_app.py::smoke` runs `smoke_test.sh` on one A10, on task 1 (a seen task) with 2 episodes. It checks the environment (pins, flash-attn, EGL, LIBERO), loads the model, rolls out with `--save_all_layers` (including a resume check) and runs the L32 capture check. Pass `--full` to add the repeat run and the tiny cache build. It prints `timing_summary.txt`. The A10 speed is unmeasured until this runs.
3. `modal run --detach collection/modal_app.py::rollouts --confirm-full-run --sec-per-episode <from smoke>` runs tasks 0–9, 50 episodes each, with seed 7 and `--resume`, one container per task (4 h timeout each). Without `--confirm-full-run` (or `CS566_CONFIRM_FULL_RUN=1`), it only prints the GPU-hour and cost estimate and exits. Relaunching continues where it stopped. `--tasks 3,7` reruns only those tasks. Each container refuses to run if the GPU name or the image differs from what earlier tasks of the same run recorded.
4. `modal run --detach collection/modal_app.py::build_cache` runs the same capture check, then `build_cache.py`, then `check_cache.py verify` on the volume (CPU only).

The smoke test is the same script on CARC (`carc/01_smoke_test.slurm`) and on Modal. On Modal it does not source `carc/activate_env.sh` (`KIT_ON_MODAL=1`).

## Getting it to teammates

Each file is one layer × one pooling × one split, about 0.4–0.7 GB with full-size data. People download only what they need.

- **On CARC**, if everyone is in the same project group, `03_build_cache.slurm` makes the cache group-readable (`chmod g+rX`).
- **Off CARC (Kaggle, Colab):** upload `manifest.json`, `CHECKSUMS.sha256`, `episodes.csv`, `splits.json`, `norm_stats/` and the needed `features/<pool>/{train,seen_val}/L*.npy` files to a private Kaggle Dataset or a private Hugging Face dataset. Check that platform's size limits first.
- After downloading, everyone runs `python check_cache.py verify --cache <dir>`. Withheld unseen files and layers you didn't download are reported but don't count as errors. Corrupted files do.

## Using it

```python
from load_cache import SharedCache
c = SharedCache("cache/v1")
train = c.records("train", layers=[4], pool="mean", normalize=True)     # list of FeatureStore-style records
val   = c.records("seen_val", layers=[4], pool="mean", normalize=True)
```

Or export to `FeatureStore` folders for `scripts/train.py` (which runs the protocol training seeds 0, 1, 2 by default):

```bash
python load_cache.py export --cache cache/v1 --out fs_L4_L24 --layers 4 24 32 --pool mean --normalize
python scripts/train.py --model single_layer_mlp --layers 4 --feature-root fs_L4_L24/train
```

`scripts/train.py` still scores on the data it trains on, so its numbers are a format check, not a result. Sheet numbers need an evaluator that scores `seen_val` at the episode level with the `task_min_step` horizon, as SAFE does.
