# Compute and environments: where the GPU work runs, what it costs, and how the envs are built

Written 2026-11-10. Sources: the orchestrator's shared context, the CARC setup and verification log (2026-11-06 to
2026-11-09, now [`docs/compute_notes_carc.md`](../compute_notes_carc.md), scrubbed), `docs/env/{README.md,constraints.txt,env_freeze.txt}`, `docs/compute_notes.md`,
`setup.py`, and the team task plan (`VLA_Task_Plan_v5`, cited "per plan v5"). For live
state and money spent, read [`00_STATUS.md`](00_STATUS.md).

In one sentence: **all GPU work runs on Modal on one A10 type; CARC keeps the finished cache and a reference
environment; the 7B policy runs in a pinned "collection" env that must never share packages with the detector env.**

Every number below says whether it is **measured** (with date and GPU) or an **estimate**.

## 1. The compute decision

| Decision | Detail | Since |
|---|---|---|
| **Modal only** for GPU work | Smoke test, full collection (500 episodes) and the GPU layer-hook test (T6). No CARC GPU jobs. | Akash, 2026-11-10 |
| **A10 is the single GPU type** | All 500 episodes on A10. Never mix GPU types in one dataset. | Confirmed by Akash, 2026-11-10 |
| Budget | Hard ceiling **$30** (the Modal Starter plan's monthly credit). **Stop and ask Akash at $25.** | Akash, 2026-11-10 |
| Spent so far | **$0.00** on Modal as of 2026-11-10. Nothing has been launched. | 2026-11-10 |
| Smoke test | **≤ $3**, needs Akash's approval before it runs. | Plan row A5 |

Per plan v5, rows P0-3, A1, A5, A6 and A8 now say Modal A10, and the plan records the 2026-11-10 decision "Modal
only for GPU work; CARC = shared storage only; default Modal GPU = A10". Older text that says "L4" (plan v4, the
orchestrator brief v2) or "CARC A40 as backup" is outdated. Per plan v5, A6 may still compare A10 with L4 "if
measured"; that comparison would be a measurement, not a second data source.

### Why A10 and not L4

OpenVLA picks one action at a time (batch size 1) and decodes 7 action tokens greedily. At batch 1, each token
reads all ~15 GB of bf16 weights from GPU memory and does very little arithmetic with them. So speed is set by
**memory bandwidth**, not by compute (FLOPs).

| GPU | Memory bandwidth (vendor spec) | Memory | Relative time per action (estimate) |
|---|---|---|---|
| A40 (CARC, where we measured) | ~696 GB/s | 48 GB | 1.00 (measured baseline) |
| A10 (Modal) | ~600 GB/s | 24 GB | ~1.16x slower |
| L4 (Modal) | ~300 GB/s | 24 GB | ~2.3x slower |

An L4 is cheaper per hour but roughly twice as slow, so it costs more in total and takes twice as long. The A10's
24 GB holds the measured 14.95 GiB peak with room to spare.

### Time and cost on A10: ESTIMATES, with the arithmetic

These are **estimates**, scaled from the A40 measurements in section 2. The smoke test replaces them with real numbers.

1. Seconds per episode on A40, **measured**: 81 s (264 steps) to 157 s (520 steps, the step cap).
2. 500 episodes on A40: 500 x 81 s = 40,500 s = **11.3 h**; 500 x 157 s = 78,500 s = **21.8 h**. This is the
   context's "~11–22 GPU-hours on A40" (estimate: real episodes lie between these two lengths, and the mix
   depends on the success rate).
3. Scale by bandwidth, 696 / 600 ≈ 1.16: **~13–25 A10-hours** (estimate). The shared context rounds this to
   ~12–24 h. Scaling the whole step slightly overstates it, because ~31 ms of each ~304 ms step is simulation
   and Python, not GPU decoding (273 ms per model call vs 304 ms per rollout step, both measured).
4. Cost at an assumed ~$1.10 per A10-hour (the rate behind the brief's figures; recheck Modal's price list
   before A6): **~$13–28** (estimate). The shared context says ~$13–26.
5. For comparison, L4 (estimate): 11–22 h x ~2 = ~22–44 h; at an assumed ~$0.80/h, ~$18–35.

**Budget warning.** The top of the A10 range (~$26–28), plus the smoke test (≤ $3), is **above the $25
stop-and-ask line and close to the $30 ceiling**. The full run is only safe if the smoke test shows A10 nearer the
low end. This is exactly what A6 decides. Not counted yet: Modal image builds, the CPU cache-build function,
Modal Volume storage, and copying the cache out to CARC.

### The smoke test (A5): plan and what it measures

Not started. Runs after the five kit fixes (A1) and only with Akash's approval. Budget **≤ $3** (estimate: about
2.5 A10-hours at ~$1.10/h, including model download and load).

What it checks (per plan v5, row A5). Per plan v5, the separate parity check is retired: the kit runs SAFE's own
script, and its capture check covers parity.

- **Headless render** with EGL inside the Modal container.
- **Capture:** the kit's layer 32 must equal SAFE's own `.pkl` hidden states exactly.
- **Resume:** a stopped job restarts and skips finished episodes.
- **Repeat:** the same episode run twice gives the same output.
- **Cache build** from the smoke episodes, then cache verify.
- **T6** (`tests/gpu/test_layer_mapping.py`): the layer-hook test, which was bitwise exact on the A40.

What it measures on the A10, to replace every estimate above: model load time, ms per action, steps per second
in a rollout, peak GPU memory, seconds per episode, bytes per step on disk, and the actual dollars spent. A6 then
turns these into the full-run GPU-hours, cost and storage numbers for Akash to approve.

## 2. Measured facts (CARC, NVIDIA A40)

All measured on **2026-11-09**, CARC debug partition, **NVIDIA A40** (46,068 MiB, bf16 supported),
`vla-safe/openvla@300dce2`, `flash_attention_2`, bf16, `MUJOCO_GL=egl`, Akash's collection env (section 4).
The rollouts were 2 episodes of one LIBERO-10 task with default settings plus `output_hidden_states`; they were
an engineering check only, not data.

| Quantity | Value | Notes |
|---|---|---|
| Model load | **24.9 s** | 4 safetensors shards, from the local HF cache |
| Time per action (model call) | **273 ms** median | n = 20; min 271.1, max 275.4 ms |
| Rollout speed | **3.29 steps/s** | 784 steps in 238 s over 2 episodes (~304 ms per step incl. simulation) |
| Peak GPU memory | **14.74 GiB** allocated / 14.95 GiB reserved | Measured during the single-action check |
| Seconds per episode | **157 s** (520 steps, hit the cap); **81 s** (264 steps) | 3.31 and 3.24 steps/s |
| SAFE's own files, per step | **~58.5 KB** | `.pkl` 57,345 B/step (7 x 4096 bf16 hidden states + small), `.csv` ~910 B/step; plus an `.mp4` of 91–145 KB per episode |
| Hidden states shape per action | 7 tokens x 33 layers x 4096 | Matches the kit's layer convention |
| T6 layer-hook test | **Passed, bitwise exact** | Tolerance 0.0; every block hook equals `hidden_states[k]`; the final norm applied to its pre-hook input equals index 32; tokens unchanged with hooks attached |

Derived numbers (**estimates**, computed from the measurements above, 2026-11-10):

| Quantity | Value | Arithmetic |
|---|---|---|
| Kit's all-layer file, per step | **~0.81 MB** (0.84 MB with index 33) | 33 layers x 3 poolings x 4096 x 2 bytes (bf16) = 811,008 B; 34 layers = 835,584 B |
| Raw `.layers.pt` for 500 episodes | **~120–160 GB** | ~150k–200k steps (depends on episode lengths) x ~0.81 MB |
| Cache (float16) for 500 episodes | **about the same again** | Same element count and 2 bytes per value |
| SAFE's own files for 500 episodes | **~10 GB** (estimate) | ~150k–200k steps x ~58.5 KB |
| Full run on A40 | **~11–22 GPU-hours** | Section 1, step 2 |

Not measured anywhere yet: anything on an A10, anything on Modal, and the kit's own `.layers.pt` size on disk
(the 0.81 MB figure is computed, not read off a file).

## 3. CARC's role now

- **Shared storage** for the finished, verified cache, so the whole team can read it. The cache goes to the user
  scratch space (`/scratch1/<user>/...`); environments and the HF cache live in the group project space
  (`/project2/<group>/<user>/...`). These are placeholders; the real paths are in Akash's private notes.
- **Reference environment:** the collection env that produced every measurement in section 2. It is the known-good
  answer when the Modal image misbehaves. (Per plan v5, row P0-3 says "CARC for shared storage only"; keeping the
  env as a reference costs no GPU time.)
- **No GPU jobs.** On 2026-11-06 the `gpu` partition's start estimates were about 4 days out for every GPU type,
  and one 1-hour debug job waited about 20 hours. The kit's Slurm scripts are still being fixed (fix 1) so that
  teammates are not misled. Working syntax, measured 2026-11-06: `-n 1 --gpus=a40:1` (the documented
  `--gpus-per-task` form is rejected without `-n`).

## 4. The two environments

There are two Python environments, and they must stay separate.

| | **Collection env** (runs the 7B policy) | **Detector env** (this repo: probes, fusion, baselines) |
|---|---|---|
| Used for | Rollouts, hidden-state capture, T6 | Training and evaluating detectors on the cache, on CPU or a small GPU |
| Python | 3.10 (3.10.16 on CARC) | 3.10 only (`setup.py`: `>=3.10,<3.11`; 3.11+ fails on a dataclass default) |
| torch | **2.2.0 (cu121)**, torchvision 0.17.0, torchaudio 2.2.0 | Unpinned, so recent (laptop: 2.14.1 CPU) |
| numpy | **< 2** (1.26.4) | Recent (laptop: 2.2.6) |
| Key pins | transformers 4.40.1, tokenizers 0.19.1, timm 0.9.10, peft 0.11.1, tensorflow 2.15.0, flash-attn 2.5.5, robosuite 1.4.0, draccus 0.8.0 | Declared in `setup.py`: pandas, scipy, hydra-core, omegaconf, scikit-learn, wandb, ... |
| Built from | `vla-safe/openvla` `pyproject.toml` + LIBERO + `collection/constraints.txt` | `pip install -e .` in `Safe-Research` |
| Where it runs | Modal image (being built by fix 1); CARC reference env | Laptop, CARC, or any machine |
| Verified | CARC, 2026-11-09 (section 2) | Laptop, 2026-11-10: 17 tests passed (Python 3.10.22, macOS CPU) |

**Why separate.** The policy needs old pins (torch 2.2.0, transformers 4.40.1, numpy < 2, TensorFlow 2.15).
`setup.py` here asks for an unpinned `torch`, so installing this repo into the collection env would upgrade or
replace the pinned torch and break the policy. If one env is ever unavoidable, install this repo with
`pip install -e . --no-deps`. The two envs share only files: the cache is written by one and read by the other.

**Single source of truth for pins: `collection/constraints.txt`.** It is committed on `akash/kit-fixes`; an
in-progress fix (fix 2) is completing it and wiring it into every install path (conda, uv, Modal image).
`docs/env/constraints.txt` and `docs/env/env_freeze.txt` are the record of the verified CARC env, not the
source to install from.

### Building the collection env: what was needed beyond the fork's pins

Verified on CARC, 2026-11-06 to 2026-11-09. None of these changed a version pinned by the OpenVLA fork; they pin
packages the fork leaves open, or fix the install.

| Fix | Why |
|---|---|
| `numpy<2` (→ 1.26.4) | TensorFlow 2.15 fails to import with numpy 2.x. |
| `protobuf>=4.25,<5` | TensorFlow 2.15 requires protobuf < 5. |
| `tensorflow-metadata<1.16` (→ 1.13.1) | The newest version (1.21.0) ships protos that need protobuf ≥ 5.26: `cannot import name 'runtime_version' from 'google.protobuf'`. |
| `mujoco<3` (→ 2.3.7) | robosuite 1.4.0 with mujoco 3.x fails with a bare `AssertionError` in `get_joint_qpos_addr`. |
| Install order | torch from the cu121 index first, then the editable `openvla`, then LIBERO and its requirements, then flash-attn last, always with `-c constraints.txt`. |
| `CUDA_HOME` for flash-attn | The flash-attn 2.5.5 build fails with "CUDA_HOME environment variable is not set" unless `CUDA_HOME` and `PATH` point at a CUDA 12.x toolkit. Install with `--no-build-isolation` after `packaging` and `ninja`. (On CARC it then finished in ~10 s, apparently from a prebuilt wheel.) |
| LIBERO `.pth` file | `pip install -e LIBERO` maps no modules (`libero/` has no `__init__.py`). Add a `.pth` file in `site-packages` containing the LIBERO repo root. |
| LIBERO first-import prompt | The first `import libero.libero` asks an interactive question about a dataset path. Set `LIBERO_CONFIG_PATH` to a writable directory and answer `N` once, or pre-write the config. A batch job or container cannot answer the prompt. |
| Offline HF cache needs code files | The fine-tuned checkpoint's processor points (`auto_map`) at code in the base repo `openvla/openvla-7b`. With `HF_HUB_OFFLINE=1`, also pre-fetch that repo's small non-weight files (~2.5 MB, no safetensors). |
| Caches in big storage | Set `HF_HOME`, `UV_CACHE_DIR`, `PIP_CACHE_DIR`, `TMPDIR` to a large disk (the HF cache alone was ~15 GB). |

### Rendering: EGL vs OSMesa

MuJoCo renders the camera images headless with either **EGL** (GPU) or **OSMesa** (software). On CARC, EGL worked
on the A40 (2026-11-09, no fallback needed); OSMesa could not work there (`libOSMesa` is not installed), and EGL
fails on CPU-only nodes, so anything that imports robosuite needs a GPU node.

The two backends can give **different pixels**, so the policy sees different inputs. The whole dataset must use one
backend, and the kit records it per job (orchestrator choice E3; `build_cache.py` refuses a mixed dataset). EGL on
Modal has **not been tested yet**; the smoke test checks it.

## 5. Storage plan (pending team approval)

This is review-checklist item 7. The team has not approved it as of 2026-11-10. All sizes are **estimates** from
section 2.

| Stage | What | Size (estimate) | Where |
|---|---|---|---|
| During collection | SAFE's `.pkl` / `.csv` / `.mp4` + the kit's raw `.layers.pt` (bf16, 3 poolings) | ~10 GB + ~120–160 GB | Modal Volume |
| Cache build | Versioned, checksummed float16 cache, all 3 poolings, train-only normalization stats | ~120–160 GB | Modal Volume |
| After the cache is built **and verified** | Delete the raw `.layers.pt`; keep SAFE's own files | ~150 GB left (shared-context figure) | Modal Volume |
| Hand-off | Copy the cache (and SAFE's files) to CARC | ~150 GB | `/scratch1/<user>/...` |

Peak on the Modal Volume is raw plus cache together, roughly **250–330 GB** (estimate), until the raw files are
deleted.

Alternatives on the checklist: mean pooling only (~50 GB, but drops the pooling check R3), or a subset of layers.
Keeping all three poolings is what makes R3 possible.

## 6. Gotchas

1. **Python 3.10 only, everywhere.** This repo fails on ≥ 3.11 (`TrainConfig` mutable dataclass default). On CARC,
   `which python` returned a stale 3.11 after `module load`; call the 3.10 interpreter by its full path.
2. **Never install this repo into the collection env** without `--no-deps`: unpinned `torch` replaces 2.2.0.
3. **Unpinned transitive deps break silently.** `tensorflow-metadata` and `mujoco` were the culprits. Always install
   with `-c collection/constraints.txt`.
4. **flash-attn must be present.** SAFE uses `flash_attention_2`; hidden states differ numerically between
   `flash_attention_2`, `sdpa` and `eager`. Fix 3 makes a missing flash-attn a hard failure. The kit's
   `00_setup_env.sh` currently suggests `--attn_implementation sdpa` as a fallback; do not follow it.
5. **One GPU type, one attention implementation, one render backend** for all 500 episodes. Mixing any of them
   makes the dataset inconsistent.
6. **`CUDA_HOME` is not set by `module load cuda`** in non-interactive shells. Set it explicitly in batch scripts
   and image builds. Use `#!/bin/bash -l` for Slurm scripts.
7. **LIBERO needs a `.pth` file and a pre-answered first-import prompt**, or the import fails or hangs in a job.
8. **Offline runs need the base repo's code files** (`openvla/openvla-7b`, ~2.5 MB), not only the fine-tuned weights.
9. **EGL needs a GPU node.** CPU-only machines cannot import robosuite with `MUJOCO_GL=egl`. A failed EGL fallback
   to OSMesa can leave `MUJOCO_GL=osmesa` set for later steps.
10. **Any directory named `env` is ignored by git** (the repo `.gitignore` has `env/`, on every OS; `ENV/` adds case
    variants on macOS). `docs/env/` is tracked only because it was force-added. Do not put new files under `env/`.
11. **draccus re-parses `sys.argv`.** SAFE's `eval_libero` is wrapped with `@draccus.wrap()`, so a wrapper script's
    own flags leak into it (`ambiguous option: --attn`). This broke a parity-check script on 2026-11-09.
12. **CARC scratch purge policy is unknown** (not found as of 2026-11-06). Check it before relying on `/scratch1`
    as the long-term home of the cache.
13. **Debug queue waits on CARC ranged from 26 minutes to ~20 hours** for a 1-hour A40 job (2026-11-06 to
    2026-11-09). This is why GPU work moved to Modal.
14. **The repo is public.** Never commit account or project IDs, usernames inside paths, hostnames, job IDs or
    compute balances. Use `/project2/<group>` and `/scratch1/<user>`.
