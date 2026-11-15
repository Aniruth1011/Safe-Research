# CARC Discovery collection environment (venv310)

Python 3.10.16, built with `uv`. Locked package list: `env_freeze.txt` (`uv pip freeze`). Extra constraints on
UNPINNED transitive deps: `constraints.txt`. Every package pinned in the vla-safe/openvla `pyproject.toml`
(torch 2.2.0, transformers 4.40.1, tensorflow 2.15.0, ...) is kept at its pinned version.

## Build order
1. `module load python/3.10.16`; call the interpreter by explicit path (`.../python-3.10.16-*/bin/python3.10`).
   `which python` can return a stale hashed 3.11.9 after `module load` in a non-interactive shell.
2. `uv venv --python <py3.10> venv310`
3. `uv pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121 -c constraints.txt`
4. `uv pip install -c constraints.txt -e openvla`, then `pandas scikit-learn`, `-e LIBERO`,
   `-r openvla/experiments/robot/libero/libero_requirements.txt`
5. flash-attn (after the editable install), see below.
6. Keep all caches in project space: `HF_HOME`, `UV_CACHE_DIR`, `PIP_CACHE_DIR`, `TMPDIR`.

## Constraints and why
- `numpy<2`: TensorFlow 2.15 breaks on numpy 2.x (`numpy.core.umath failed to import`).
- `protobuf>=4.25,<5`: TensorFlow 2.15 requires protobuf<5.
- `tensorflow-metadata<1.16`: tensorflow-metadata 1.21.0 (unpinned, newest) ships protos needing
  protobuf>=5.26 (`cannot import name 'runtime_version' from 'google.protobuf'`) although its metadata
  allows protobuf 4.x. It resolves to 1.13.1 (with absl-py 1.4.0, googleapis-common-protos added).

## Workarounds
- **LIBERO `.pth`**: `pip install -e LIBERO` maps no modules (`libero/` has no `__init__.py`). Fix:
  `echo <path>/LIBERO > venv310/lib/python3.10/site-packages/libero_repo.pth`.
- **LIBERO config prompt**: first `import libero.libero` asks about a dataset path; answer `N` once with
  `export LIBERO_CONFIG_PATH=<dir>` set so the config lands in project space, not `~/.libero`.
- **flash-attn 2.5.5 and CUDA_HOME**: `module load cuda/12.4.0` did not put nvcc on PATH in a non-interactive
  shell (build died with `CUDA_HOME environment variable is not set`). Set explicitly:
  `export CUDA_HOME=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/cuda-12.4.0-qwgpfv7; export PATH=$CUDA_HOME/bin:$PATH`
  then `uv pip install -c constraints.txt "flash-attn==2.5.5" --no-build-isolation` (finished in ~10 s,
  apparently a prebuilt wheel). Packaging/ninja installed first. Attention impl stays `flash_attention_2`.
- **Slurm GPU syntax**: `-n 1 --gpus=a40:1` works; `--gpus-per-task` without `-n` is rejected.
- Batch scripts: use `#!/bin/bash -l` and set CUDA_HOME/PATH explicitly.

## Later fixes (2026-11-09)
- `mujoco<3` (-> 2.3.7): robosuite 1.4.0 + mujoco 3.x fails in `get_joint_qpos_addr` with a bare `AssertionError`
  (binding_utils.py:521). Unpinned transitive dep; robosuite stays at its pin.
- **Offline HF cache needs the base repo's code files.** The fine-tuned checkpoint's `auto_map` points at
  `openvla/openvla-7b--processing_prismatic...`. With `HF_HUB_OFFLINE=1` also pre-fetch the small non-weight files
  (2.5 MB, no safetensors) of `openvla/openvla-7b` into `HF_HOME` in a CPU job.
- mujoco's EGL init fails on CPU nodes (no EGL device). Anything importing robosuite needs a GPU node.
- `libOSMesa` is not installed; the osmesa fallback cannot work here.
