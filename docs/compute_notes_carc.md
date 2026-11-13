# CARC setup and verification log (Akash, 2026-11-06 to 2026-11-09)

Scrubbed for this public repo: account and project IDs, user names, host names, Slurm job IDs and SU balances are
removed. Paths are shown as `/project2/<group>` and `/scratch1/<user>`. Measurements, GPU type, package versions and
environment fixes are unchanged from the original log.

**Status since 2026-11-10:** all GPU work moved to Modal (A10). CARC is now used only for shared storage and as the
reference environment. These notes stay because they hold the only measurements we have so far, and they explain
the pins and workarounds the kit now uses.

## Result (2026-11-09, NVIDIA A40, openvla @ `300dce2`)

Third smoke run, on the `debug` partition (6 min 15 s). All gates passed, and job 2 (T6) then ran automatically.

| Step | Result | Numbers |
|---|---|---|
| bf16 GPU | PASS | NVIDIA A40, 46068 MiB, `is_bf16_supported()=True`, driver 580.159.04 |
| Headless render | PASS | `MUJOCO_GL=egl`, no fallback needed. 256x256 frame mean 69.0, std 43.6; the saved PNG shows robot, basket and items (checked by eye) |
| Model load + one action | PASS | `flash_attention_2`, bf16. Load 24.9 s. Action `[0.002 -0.148 0 0 0 -0.001 0.996]`. hidden_states 7 x 33 x 4096 (first entry (1, 285, 4096)). Median latency 273.4 ms/action (min 271.1, max 275.4, n=20). Peak GPU memory 14.74 GiB allocated / 14.95 GiB reserved |
| Mini rollout | PASS (engineering check only) | 2 episodes of task 0, default settings + `output_hidden_states`. ep0: 520 steps (the cap), 157 s, 3.31 steps/s. ep1: 264 steps, 81 s, 3.24 steps/s. Overall 784 steps / 238 s = 3.29 steps/s (~304 ms/step vs 273 ms per model call). Script wall time 269 s including imports and load |
| Bytes per step | measured | SAFE `.pkl` 57,345 B/step (= 7 x 4096 bf16 hidden states + a little; ep0 29,819,656 B / 520 steps, ep1 15,139,494 B / 264 steps); `.csv` ~910 B/step; `.mp4` 145 KB / 91 KB per episode. ~58.5 KB/step in total, 46 MB for the 2 episodes |
| T6 layer mapping | PASS | `tests/gpu/test_layer_mapping.py`: 1 passed in 29.9 s with `VLA_NORM_ATOL=0.0`. tokens_ref == tokens_hooked == tokens without hidden states. block_max_abs_diff 0.0 for blocks 1–31; RMSNorm check max_abs / max_rel / rel_l2 = 0.0 and norm_ratio 1.0 for all 7 steps. **Bitwise exact** |
| Parity check | NOT RUN (script bug) | The retired collector's `parity_check.py` exited with rc=2 after 18 s: draccus's `@draccus.wrap()` on `eval_libero` re-parses `sys.argv`, so the parity script's own `--attn` flag became ambiguous (`--attn_implementation`, `--attn_avg_token`). Parity was retired with the collector |

## Environment (verified on CARC)

- Python 3.10.16 (from the `python/3.10.16` module, called by its explicit binary path; `module load` is unreliable in
  non-interactive shells), a `uv` virtual environment.
- Installed versions: torch 2.2.0+cu121, torchvision 0.17.0+cu121, transformers 4.40.1, tokenizers 0.19.1, timm 0.9.10,
  peft 0.11.1, tensorflow 2.15.0, robosuite 1.4.0, **flash-attn 2.5.5** (imports, including `flash_attn_2_cuda`),
  mujoco 2.3.7, numpy 1.26.4. All 10 `libero_10` tasks load.
- Attention implementation: **`flash_attention_2`** (the fork's default). sdpa was not adopted (team decision).
- Repos used: `vla-safe/openvla` @ `300dce2`, LIBERO, the retired collector (`akash/collector`), Safe-Research.

### Fixes that were needed (none changes a pin of the fork)

1. **Python 3.11 by accident.** A stale shell hash after `module load` gave a 3.11 venv; rebuilt on 3.10.
2. **numpy 2.x breaks TensorFlow 2.15** → constraint `numpy<2`.
3. **protobuf clash.** `tensorflow-metadata` was left unpinned and resolved to 1.21.0, whose protos need
   protobuf >= 5.26, while `tensorflow==2.15.0` forces protobuf 4.25.x. Symptom:
   `ImportError: cannot import name 'runtime_version' from 'google.protobuf'` (reached through `tensorflow_datasets`,
   which `openvla_utils.py` imports). Fix: constraints `protobuf>=4.25,<5` and `tensorflow-metadata<1.16`
   (resolved to tensorflow-metadata 1.13.1, absl-py 1.4.0, googleapis-common-protos 1.75.0). `pip check` clean.
4. **robosuite 1.4.0 vs mujoco 3.x.** With mujoco 3.17.0, robosuite's `binding_utils.py:521` (`get_joint_qpos_addr`)
   raises `AssertionError` after the EGL context is created. Fix: constraint `mujoco<3` (resolved to 2.3.7).
5. **LIBERO editable install maps no modules** (`libero/` has no `__init__.py`). Fix: a `.pth` file that points at
   the LIBERO repo root.
6. **LIBERO asks an interactive question on first import.** Answer `N` once, or pre-write its config and set
   `LIBERO_CONFIG_PATH`.
7. **flash-attn build needs `CUDA_HOME`.** The first attempt failed with "nvcc not found" because `module load cuda`
   did not set it. With `CUDA_HOME` pointing at the CUDA 12.4 module's install directory, the install took ~10 s
   (it seems to have fetched a prebuilt wheel rather than compiling). Install flash-attn after the editable install.
8. **Offline processor loading.** `AutoProcessor` for the fine-tuned checkpoint needs `processing_prismatic.py` and
   the other non-weight code files of the base repo `openvla/openvla-7b` (via `auto_map`). Fetch those files (~2.5 MB)
   into `HF_HOME` before going offline.
9. **Render fallback leaks.** A smoke script that falls back to OSMesa must not leave `MUJOCO_GL=osmesa` set for
   later steps. OSMesa does not work on the CARC GPU nodes anyway (no `libOSMesa`); EGL with the NVIDIA `libEGL` works.

## Slurm on CARC (Discovery)

- GPU types in the `gpu` partition (2-day max): A100 (2 per node), A40 (2), L40S (3–4), V100 (2), P100 (2).
  bf16-capable: A100, A40, L40S.
- **Working syntax:** `-A <account> -p gpu -n 1 --gpus=a40:1 -c 8 --mem=48G -t ...`. The documented
  `--gpus-per-task` form fails without `-n`.
- `debug` partition: 1 h max, only A40 (and P100) GPUs.
- SU cost per GPU-minute (CARC docs): A100 / A40 8, V100 / P100 4.
- Limits (CARC "Running Jobs" doc; whether they are per user or total is not stated): gpu 48 h, 400 CPUs, 36 GPUs,
  36 running jobs, 100 queued; debug 1 h, 48 CPUs, 4 GPUs, 5 jobs. Default wall time 1 h.
- CARC's login banner forbids coding agents on login nodes: installs, clones and downloads ran inside CPU allocations.

### Observed queue waits (why GPU work moved to Modal)

- `gpu` partition `--test-only` start estimates on 2026-11-06: about 4 days out for every GPU type.
- `debug` A40, 1 h jobs: one smoke run waited ~20 h (2026-11-06 → 2026-11-07); another started after an estimated
  1–2 h (2026-11-09); job 2 waited 26 min (2026-11-09, from logs).
- On 2026-11-10 the Slurm controller was unreachable, so there is no fresher estimate.

## Storage

- Quotas: `/project2/<group>` 1 TiB and 6M files, shared by the lab; `/scratch1/<user>` 9.766 TiB, 20M files;
  `/scratch2` 10 TiB; `/home1` 100 GiB.
- The environment, `HF_HOME` (15 GB) and the uv/pip caches lived under `/project2/<group>/<user>/`; rollout data under
  `/scratch1/<user>/data/`.
- **Scratch purge policy: not found** (not shown by `myquota`; the CARC doc URL tried returned 404). Ask CARC before
  relying on `/scratch1` for the shared cache.

## Operations lessons

- If the laptop sleeps and the VPN drops, the SSH control master exits and its socket disappears
  (`ControlPersist` only keeps a healthy master alive). Slurm jobs are unaffected; only local watchers die.
- Idle CPU allocations held while disconnected ran to their time limit and wasted SUs. Use short allocations, and
  chain work inside Slurm (`--dependency=afterany:<job>` with a small CPU gate job that reads the previous log and
  submits the next job only on PASS) instead of holding allocations while waiting.
