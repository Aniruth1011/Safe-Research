# Shared feature cache: OpenVLA-7B on LIBERO-10, every hidden-state layer

This kit runs OpenVLA once (on Modal, one A10 per task; the CARC scripts are kept as the verified reference setup) and saves **every** hidden-state layer at every step. It then packages the result into a versioned, checksummed cache with a frozen split, so all five of us train detectors on identical features and labels.

## What's in the kit

| File | What it does |
|---|---|
| `openvla_all_layers.patch` | Patch for `vla-safe/openvla`. Adds `--save_all_layers True`, which writes `task*--ep*--succ*.layers.pt` = (T, pools, 34, 4096) bfloat16 (33 with `--save_raw_block32 False`) next to SAFE's usual files. With the flag it also refuses to run without `flash_attention_2`, records a run fingerprint, and writes invalid-episode markers. SAFE's own `.pkl`/`.csv` outputs are unchanged, so the original SAFE baselines (Exp 1–3) still run on the same rollouts. Regenerated from the openvla fork's `akash/kit-patch` with `git diff 300dce2 akash/kit-patch`; never edit it by hand. |
| `build_cache.py` | Raw rollouts → immutable cache (`features/<pool>/<split>/L<NN>.npy`, `episodes.csv`, `splits.json`, train-only `norm_stats/`, `manifest.json` + `CHECKSUMS.sha256`). |
| `load_cache.py` | Read the cache from Python, or `export` chosen layers into the repo's `FeatureStore` format (one store per split) for `scripts/train.py --feature-root`. `unseen_test` is locked unless `--allow-unseen` (Exp 37 only). |
| `check_cache.py` | `capture` checks rollouts against SAFE's own `.pkl`; `verify` checks a downloaded copy against checksums; `compare-runs` checks two rollout runs for outcome flips and action divergence. |
| `constraints.txt` | The single source of truth for every package pin of the collection environment. Every install path (CARC setup, Modal image) uses it with `pip install -c`. |
| `modal_app.py` | The Modal app: image, one A10 function per task, a CPU function that builds the cache. See "Running on Modal (A10)". |
| `smoke_test.sh` | The smoke sequence (env checks, model load, resume, capture, optional repeat and tiny build). Same script on Modal and CARC. |
| `carc/` | `config.sh` (edit once) → `00_setup_env.sh` → `01_smoke_test.slurm` → `02_rollouts.slurm` → `03_build_cache.slurm`; jobs source `activate_env.sh` (uv or conda). |
| `../tests/test_shared_cache.py`, `../tests/test_kit_*.py` | End-to-end tests on fake rollouts and checks of the scripts (no GPU needed); `layer_cache.py` is loaded straight from the patch. The patch carries its own tests (`tests/test_kit_*.py` in the openvla checkout). `../tests/gpu/test_layer_mapping.py` (T6) is the GPU proof of the layer convention. |

## Conventions frozen by this cache (agree on these before running)

- **Layer index** = HuggingFace `hidden_states` index, **L0 = embeddings, L1–L32 = decoder layers**:
  - `0` is the embeddings.
  - `1…31` are the outputs of decoder layers 1–31, *before* any normalization.
  - `32` is the output of decoder layer 32 **after the final RMSNorm**. Checked in the source of the pinned `transformers==4.40.1` (`LlamaModel.forward` applies `self.norm` and then appends the last hidden state). This is the same tensor SAFE stores as `s[-1]`, so our final-layer baseline matches SAFE. `check_cache.py capture` confirms it on real data: at L32, the `first`/`last` poolings must equal SAFE's own `.pkl` bit for bit, and `mean` must match up to bf16 rounding.
  - So spreadsheet **L4 = index 4, L32 = index 32**.
  - Because only L32 is normalized, raw scales differ across layers.
- **Index 33 = raw block 32 (pending team approval, review checklist item 3).** `33` is the output of decoder layer 32 *before* the final RMSNorm. `hidden_states` does not contain it, so the patch captures it with a forward pre-hook on `model.language_model.model.norm`, once per generated token, and pools it like the other layers. The files are then (T, pools, **34**, 4096). Indices 0–32 do not change, and the L32 capture check still uses index 32. It is on by default with `--save_all_layers True`. `--save_raw_block32 False` turns it off and writes the old 33-layer files. Every `.layers.pt` and `manifest.json` lists `layer_names` (`emb`, `block01`…`block31`, `block32_postnorm`, `block32_prenorm`). A file without `layer_names` is read as the old 33-layer convention. `build_cache.py` and `check_cache.py` refuse a dataset that mixes 33- and 34-layer files. GPU proof: `tests/gpu/test_layer_mapping.py` (T6).
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

The team's GPU runs use Modal (next section); CARC GPU queues were days long in November 2026. These steps are the verified reference setup. Never mix A40 (CARC) and A10 (Modal) rollouts in one dataset: `build_cache.py` refuses a mixed GPU type.

0. Clone this repo on CARC and `cd collection`. Edit `carc/config.sh`: set `CARC_ACCOUNT` (scripts refuse to run while anything is still `CHANGE_ME`), check that `CACHE_ROOT` is under your group's project directory (CARC uses `/project2/<group>`, not `/project/<group>`), check the GPU type, and pick `ENV_MODE`:
   - `uv` (default, the mode verified on CARC): a plain Python 3.10 virtualenv at `$VENV_DIR`, built with [uv](https://docs.astral.sh/uv/). No conda and no `module load`. Install uv once if it is missing (goes to `~/.local/bin`, no admin rights): `curl -LsSf https://astral.sh/uv/install.sh | sh`. Set `PYTHON_BIN` to a python3.10 binary, or leave it empty and uv finds or downloads one.
   - `conda`: a conda env named `$CONDA_ENV` with Python 3.10 (the kit's original behaviour). Same packages, same constraints file; the verified CARC build (2026-11-09) used `uv`, not this mode.
   - `CUDA_HOME` must be a CUDA toolkit root containing `bin/nvcc`: flash-attn is built against it, and `module load cuda` does not set it in batch shells.
1. Set up the environment once, on a login node (or an interactive CPU allocation): `bash carc/00_setup_env.sh`. It is safe to rerun (finished steps are skipped). `bash carc/00_setup_env.sh --verify` installs nothing and checks an existing environment: Python 3.10, every installed version against `constraints.txt`, `pip check`, flash-attn and its CUDA extension import (on a GPU node also that `transformers` can use it), both repos at their pinned commits with the patch applied, LIBERO imports without asking anything, and the processor loads offline.
   - **Versions:** `constraints.txt` is the single source of truth for every pinned package, with one comment per pin saying why. Every install passes `-c constraints.txt` and names packages without versions; the CUDA 12.1 wheels of torch/torchvision/torchaudio come from `--index-url https://download.pytorch.org/whl/cu121` on that one install line (a constraints file cannot carry an index URL), and setup checks they are `+cu121` builds. Do not change a version without telling the team: the versions are part of what defines the dataset.
   - **Sources:** clones `vla-safe/openvla` and checks out exactly `OPENVLA_COMMIT` (the commit the patch is made for; setup stops if that commit is missing upstream rather than substituting another), applies the patch, and clones LIBERO at `LIBERO_COMMIT`.
   - **flash-attn is required.** If it does not build or import (including its compiled `flash_attn_2_cuda` extension), setup stops with an error. Fix the build, usually `CUDA_HOME`; never switch the attention implementation, because SAFE's features are collected with `flash_attention_2` and the whole dataset must use one implementation.
   - **LIBERO:** `pip install -e LIBERO` makes nothing importable (`libero/` has no `__init__.py`), so setup writes a `libero_repo.pth` file into site-packages. LIBERO also asks a question on its first import, which a batch job cannot answer; setup answers **N** once and the answer is stored in `$LIBERO_CONFIG_PATH/config.yaml` (not `~/.libero`).
   - **Model files:** downloads the checkpoint and the non-weight code files of the base repo `openvla/openvla-7b` (the checkpoint's processor class lives there), and records both commits in `$WORK/checkpoint_commit.txt`; reruns reuse the recorded commits. Jobs run with `HF_HUB_OFFLINE=1`, so they load exactly these files.
   - **Records:** `$WORK/{pip_freeze,openvla_commit,libero_commit,checkpoint_path,checkpoint_commit}.txt`. The cache build copies them into the cache as provenance. Download caches, temp files and the ~15 GB Hugging Face cache stay under `$WORK`, out of your home quota.
   - Jobs activate the environment with `source carc/config.sh && source carc/activate_env.sh`, which picks the uv venv or the conda env from `ENV_MODE`, exports `CUDA_HOME`, `HF_HOME`, `HF_HUB_OFFLINE=1` and `LIBERO_CONFIG_PATH`, requires `MUJOCO_GL=egl` (CARC has no OSMesa, and mixing render backends changes the images the policy sees), and fails loudly on an unknown `ENV_MODE` or a missing env. Rendering itself needs a GPU node, so only the smoke test checks it.
2. Run the smoke test (debug partition, under an hour) and **don't submit the full run until it passes**:
   `mkdir -p logs && sbatch --account=$CARC_ACCOUNT carc/01_smoke_test.slurm`
   - **Resume:** runs 1 episode, then asks for 2 with `--resume True`. Episode 0 must not be rerun.
   - **Capture:** L32 must reproduce SAFE's own `.pkl` (first/last bit for bit, mean up to bf16 rounding), every layer must differ from its neighbor, and there must be no NaNs.
   - **Repeat:** an independent second run, compared for outcome flips and action divergence. This is a sanity check, not a reproducibility guarantee.
   - **Build:** builds and verifies a tiny cache, then writes `timing_summary.txt` (seconds per step, MB per step). Share that with the team, and set `--time` in `02_rollouts.slurm` from it.
3. Run the rollouts, 10 parallel jobs (one per task), all on A40 (one GPU type for the whole dataset):
   `sbatch --account=$CARC_ACCOUNT carc/02_rollouts.slurm`
   - To rerun only unfinished tasks: `--array=3,7`. Resume is automatic.
4. Build and verify the cache:
   `sbatch --account=$CARC_ACCOUNT --dependency=afterok:<array_job_id> carc/03_build_cache.slurm`
   - Refuses to run unless each of the 500 episodes either exists or is explained by an invalid-episode marker (see "Invalid episodes"), and unless every episode has the same attention implementation, GPU type, render backend and layer names.
   - Runs the capture check on every episode.
   - Copies pip freeze and the OpenVLA, LIBERO and checkpoint versions into `provenance/` *before* checksumming.
   - **Detector training starts only after this job's `verify` passes.**

Each `01`–`03` script reads `carc/config.sh` by relative path, so submit it from inside this folder.

**Storage (rough, computed, not measured):** about 0.84 MB per step for 3 poolings × 34 layers (0.81 MB with 33). At roughly 350 steps × 500 episodes, that's ~155 GB of raw rollouts, plus about the same again for the cache. Setting `POOLING="mean"` in `config.sh` cuts this ~3×, but drops Exp 14.

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

## Invalid episodes

With `--save_all_layers`, an episode that raises an ordinary exception does not kill the job. The patched script writes `task{T}--ep{E}--invalid.json` (atomic) **instead of** that episode's `.csv`/`.pkl`/`.layers.pt`/`.mp4`, resets the env and continues. The marker is a JSON object: `format` (1), `kind`, `task_suite_name`, `task_id`, `episode_idx`, `init_state_index`, `seed`, `step_index` (env step incl. the settling steps), `num_steps_wait`, `policy_steps_completed`, `exception_type`, `exception_message`, `traceback`, `timestamp_utc`, `job`, `invalid_count_in_job_range`, `max_invalid_episodes`.
- `--resume` counts a marker as done. To retry an episode, delete its marker.
- Systemic errors are never recorded; they stop the job: Ctrl-C/SystemExit, MemoryError, CUDA OOM and CUDA/cuBLAS/cuDNN/driver errors, device-side asserts, flash-attn/EGL/OpenGL errors, import errors, a full disk. Errors before the first episode stop it too.
- More than `--max_invalid_episodes` (default 3) markers in the job's task range also stop the job. If the cause was systemic, delete the markers before resuming.
- `build_cache.py` requires every missing episode to have exactly one marker, refuses an episode that has both a marker and data, and lists invalid episodes under `invalid_episodes` in `manifest.json`. `--max-invalid N` sets a dataset-wide limit. `check_cache.py capture`/`compare-runs` list the markers.

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
