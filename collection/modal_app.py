"""Modal app for the shared all-layer collection (OpenVLA-7B-finetuned-libero-10 on LIBERO-10).

Every `modal run` below spends money (GPU or CPU time). Get Akash's OK first. Run from the repo root.

    modal run collection/modal_app.py::download_model     # CPU: checkpoint + base-repo code files -> HF volume
    modal run collection/modal_app.py::smoke              # 1 A10: task 1, 2 episodes, smoke_test.sh
    modal run --detach collection/modal_app.py::rollouts --confirm-full-run   # 10 A10s, one per task
    modal run --detach collection/modal_app.py::rollouts --confirm-full-run --tasks 3,7   # rerun some tasks
    modal run --detach collection/modal_app.py::build_cache   # CPU: capture check -> build_cache -> verify

`rollouts` refuses to start without `--confirm-full-run` (or CS566_CONFIRM_FULL_RUN=1) and always prints the
GPU-hour estimate first. Use `--detach` for anything long: without it, closing the terminal stops the app.

Design:
- ONE GPU type for all 500 episodes: A10 (Akash, 2026-11-10). `gpu="A10"` is the string listed in Modal's GPU guide
  (https://modal.com/docs/guide/gpu, checked 2026-11-10; the guide lists no "A10G" value). Each rollout container
  also records the GPU name it got and refuses to run if an earlier task of the same run used a different one.
- All pins come from `collection/constraints.txt` (E4): every pip install below passes `-c` with that file, and the
  image build fails unless the packages in REQUIRED_EXACT_PINS are pinned with `==` there and installed at that version.
- openvla = vla-safe/openvla at exactly OPENVLA_COMMIT, plus `openvla_all_layers.patch` via `git apply` (never edited).
- Outputs live in Modal Volumes, never in this repo. Provenance files there may contain hostnames or job IDs:
  do not copy anything from the volumes into the public repo.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

# Modal's 2026.10 image builder only supports Python >= 3.11 for `add_python`, and this repo needs exactly 3.10
# (the collection env verified on CARC). Builder 2025.06 still supports 3.10. Pin it here, in the client, so every
# workspace (each has its own default builder setting) builds the identical image from the same commit.
IMAGE_BUILDER_VERSION = "2025.06"
if os.environ.get("MODAL_IMAGE_BUILDER_VERSION", IMAGE_BUILDER_VERSION) != IMAGE_BUILDER_VERSION:
    raise RuntimeError(f"MODAL_IMAGE_BUILDER_VERSION must be {IMAGE_BUILDER_VERSION} (or unset) for this app")
os.environ["MODAL_IMAGE_BUILDER_VERSION"] = IMAGE_BUILDER_VERSION

import modal  # noqa: E402  (after the builder pin)

# ----------------------------------------------------------------------------------------------------------------
# Constants (tests/test_kit_modal.py reads several of these by name)
# ----------------------------------------------------------------------------------------------------------------
APP_NAME = "cs566-vla-collection"
GPU = "A10"  # the single GPU type for the whole dataset

OPENVLA_REPO = "https://github.com/vla-safe/openvla.git"
OPENVLA_COMMIT = "300dce26d44f407c725695d16cd445755c92cbd1"  # = 300dce2, SAFE's fork as pinned by the kit
LIBERO_REPO = "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"  # master HEAD on 2026-11-10 (git ls-remote)

CHECKPOINT_REPO = "openvla/openvla-7b-finetuned-libero-10"
BASE_REPO = "openvla/openvla-7b"  # its auto_map code files are needed for offline processor loading
BASE_REPO_IGNORE = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt", "*.h5", "*.msgpack", "*.onnx"]

N_TASKS = 10
EPISODES_PER_TASK = 50
SEED = 7
POOLING = "mean,first,last"
RUN_NOTE = "all-layers-v1"
CACHE_VERSION = "v1"

# Timeouts. CARC A40 (measured 2026-11-09, openvla@300dce2): 81-157 s per episode, so one task (50 episodes) took
# an estimated 1.1-2.2 h there. A10 speed is NOT measured yet; the smoke test measures it. 4 h per task leaves ~1.8x
# headroom over the slowest A40 figure. A task that hits the timeout loses only its in-progress episode: relaunch it
# and --resume continues. So the timeout bounds the cost of a hang (10 x 4 h) instead of having to fit every task.
ROLLOUT_TIMEOUT_S = 4 * 3600
SMOKE_TIMEOUT_S = 45 * 60  # 3 model loads + 2 episodes (+2 more with --full) + checks
BUILD_TIMEOUT_S = 8 * 3600  # same as CARC's 03_build_cache.slurm
DOWNLOAD_TIMEOUT_S = 60 * 60

# Prices from https://modal.com/pricing, checked 2026-11-10: A10 $0.000306/s; CPU $0.0000131/core/s; RAM $0.00000222/GiB/s.
A10_USD_PER_HOUR = 0.000306 * 3600  # ~= $1.10
A40_SEC_PER_EPISODE = (81, 157)  # measured on CARC A40, 2026-11-09
A10_GPU_HOURS_ESTIMATE = (12, 24)  # ESTIMATE (not measured) for all 500 episodes, from the compute plan 2026-11-10
ROLLOUT_CPU_CORES = 2.0  # physical cores (= 4 vCPU), billed in addition to the GPU
ROLLOUT_MEMORY_MIB = 16384

REQUIRED_EXACT_PINS = [  # must appear as `name==version` in constraints.txt and be installed at that version
    "torch", "torchvision", "transformers", "tokenizers", "tensorflow", "flash-attn", "robosuite",
]
REQUIRED_RANGE_PINS = ["numpy", "protobuf", "tensorflow-metadata", "mujoco"]  # must be constrained (any specifier)

# ----------------------------------------------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------------------------------------------
KIT_LOCAL = Path(__file__).resolve().parent  # this collection/ folder
BUILD_DIR = "/kit-build"  # files baked into the image for build steps
KIT_DIR = "/kit"  # the whole collection/ folder, mounted at container start
OPENVLA_DIR = "/opt/openvla"
LIBERO_DIR = "/opt/LIBERO"
LIBERO_CONFIG_PATH = "/opt/libero_config"
PROVENANCE_DIR = "/opt/provenance"

HF_MOUNT = "/vol/hf"
DATA_MOUNT = "/vol/data"
WORK_DIR = f"{DATA_MOUNT}/work"
ROLLOUT_ROOT = f"{DATA_MOUNT}/rollouts"
SMOKE_ROOT = f"{DATA_MOUNT}/smoke"
CACHE_ROOT = f"{DATA_MOUNT}/cache"
CHECKPOINT_PATH_FILE = f"{WORK_DIR}/checkpoint_path.txt"

hf_volume = modal.Volume.from_name(os.environ.get("CS566_HF_VOLUME", "cs566-hf-cache"), create_if_missing=True)
data_volume = modal.Volume.from_name(os.environ.get("CS566_DATA_VOLUME", "cs566-collection"), create_if_missing=True)
VOLUMES = {HF_MOUNT: hf_volume, DATA_MOUNT: data_volume}


# ----------------------------------------------------------------------------------------------------------------
# Image
# ----------------------------------------------------------------------------------------------------------------
def _finish_image() -> None:
    """Build step (runs once, inside the image): LIBERO .pth + config, pin check, provenance files."""
    import importlib.metadata as md
    import sysconfig

    import yaml
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    # LIBERO: `pip install -e` maps no modules (libero/ has no __init__.py), so point a .pth at the repo root.
    Path(sysconfig.get_paths()["purelib"], "libero_repo.pth").write_text(LIBERO_DIR + "\n")
    # LIBERO asks an interactive question on first import unless its config exists. Write the default answer ("N").
    root = f"{LIBERO_DIR}/libero/libero"
    cfg = {"benchmark_root": root, "bddl_files": f"{root}/bddl_files", "init_states": f"{root}/init_files",
           "datasets": f"{root}/../datasets", "assets": f"{root}/assets"}
    Path(LIBERO_CONFIG_PATH).mkdir(parents=True, exist_ok=True)
    Path(LIBERO_CONFIG_PATH, "config.yaml").write_text(yaml.safe_dump(cfg))
    # stdin closed: if LIBERO still prompts, input() raises EOFError and the build fails here, not in a GPU job.
    subprocess.run([sys.executable, "-c", "from libero.libero import benchmark; "
                    "n = benchmark.get_benchmark_dict()['libero_10']().n_tasks; assert n == 10, n; print('LIBERO ok')"],
                   check=True, stdin=subprocess.DEVNULL)

    # Pins: constraints.txt is the single source of truth; verify the installed set against it.
    reqs = {}
    for line in Path(BUILD_DIR, "constraints.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            r = Requirement(line)
            reqs[canonicalize_name(r.name)] = r
    errors = []
    for name in REQUIRED_EXACT_PINS:
        r = reqs.get(canonicalize_name(name))
        if r is None or not any(s.operator == "==" for s in r.specifier):
            errors.append(f"constraints.txt must pin {name} with ==")
    for name in REQUIRED_RANGE_PINS:
        if canonicalize_name(name) not in reqs:
            errors.append(f"constraints.txt must constrain {name}")
    for key, r in reqs.items():
        try:
            have = md.version(r.name)
        except md.PackageNotFoundError:
            if key in {canonicalize_name(n) for n in REQUIRED_EXACT_PINS + REQUIRED_RANGE_PINS}:
                errors.append(f"{r.name} is not installed")
            continue
        if not r.specifier.contains(have, prereleases=True):
            errors.append(f"{r.name} {have} violates constraint {r}")
        print(f"pin ok: {r.name} {have} ({r.specifier})")
    if errors:
        raise SystemExit("pin check failed:\n  " + "\n  ".join(errors))

    Path(PROVENANCE_DIR).mkdir(parents=True, exist_ok=True)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout
    Path(PROVENANCE_DIR, "pip_freeze.txt").write_text(freeze)
    Path(PROVENANCE_DIR, "image_builder_version.txt").write_text(IMAGE_BUILDER_VERSION + "\n")
    for name, repo in (("openvla", OPENVLA_DIR), ("libero", LIBERO_DIR)):
        head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        Path(PROVENANCE_DIR, f"{name}_commit.txt").write_text(head.stdout)
    diff = subprocess.run(["git", "-C", OPENVLA_DIR, "diff", "--stat"], check=True, capture_output=True, text=True)
    Path(PROVENANCE_DIR, "openvla_patch_stat.txt").write_text(diff.stdout)


image = (
    # CUDA *devel* base: nvcc is present if flash-attn has to compile from source instead of fetching its prebuilt wheel.
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git", "build-essential",
        # headless MuJoCo rendering through EGL (GLVND loader; the NVIDIA EGL driver comes from the host)
        "libegl1", "libgl1", "libglvnd0", "libglx0", "libgles2", "libglib2.0-0",
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "MAX_JOBS": "8",  # bounds memory if flash-attn compiles from source
        "NVIDIA_DRIVER_CAPABILITIES": "all",  # includes "graphics", needed for EGL
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "LIBERO_CONFIG_PATH": LIBERO_CONFIG_PATH,
        "HF_HOME": HF_MOUNT,
        "HF_MODULES_CACHE": "/tmp/hf_modules",  # trust_remote_code modules stay container-local, not in the volume
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_MODE": "disabled",
        "KIT_ON_MODAL": "1",  # tells smoke_test.sh not to source carc/activate_env.sh
        "KIT_DIR": KIT_DIR,
        "OPENVLA_DIR": OPENVLA_DIR,
        "LIBERO_DIR": LIBERO_DIR,
        "SEED": str(SEED),
        "POOLING": POOLING,
    })
    .run_commands(
        "mkdir -p /usr/share/glvnd/egl_vendor.d && printf '%s\\n' "
        "'{\"file_format_version\": \"1.0.0\", \"ICD\": {\"library_path\": \"libEGL_nvidia.so.0\"}}' "
        "> /usr/share/glvnd/egl_vendor.d/10_nvidia.json",
    )
    .add_local_file(KIT_LOCAL / "constraints.txt", f"{BUILD_DIR}/constraints.txt", copy=True)
    .add_local_file(KIT_LOCAL / "openvla_all_layers.patch", f"{BUILD_DIR}/openvla_all_layers.patch", copy=True)
    .run_commands(
        # Exact commit or fail: checkout errors if the commit is missing, and the rev-parse comparison double-checks.
        f"git clone {OPENVLA_REPO} {OPENVLA_DIR}",
        f"cd {OPENVLA_DIR} && git cat-file -e {OPENVLA_COMMIT}^{{commit}} && git checkout --detach {OPENVLA_COMMIT}"
        f" && test \"$(git rev-parse HEAD)\" = {OPENVLA_COMMIT}",
        f"cd {OPENVLA_DIR} && git apply --check {BUILD_DIR}/openvla_all_layers.patch"
        f" && git apply {BUILD_DIR}/openvla_all_layers.patch",
        f"git clone {LIBERO_REPO} {LIBERO_DIR}",
        f"cd {LIBERO_DIR} && git checkout --detach {LIBERO_COMMIT} && test \"$(git rev-parse HEAD)\" = {LIBERO_COMMIT}",
    )
    .run_commands(
        # Versions come from constraints.txt (-c). Order follows SAFE's/OpenVLA's README: torch, -e openvla, extras,
        # LIBERO, LIBERO requirements, then flash-attn last with --no-build-isolation.
        "python -m pip install -c /kit-build/constraints.txt packaging ninja wheel setuptools",
        "python -m pip install -c /kit-build/constraints.txt torch torchvision torchaudio"
        " --index-url https://download.pytorch.org/whl/cu121",
        "python -m pip install -c /kit-build/constraints.txt -e /opt/openvla",
        "python -m pip install -c /kit-build/constraints.txt pandas scikit-learn",
        "python -m pip install -c /kit-build/constraints.txt -e /opt/LIBERO",
        "python -m pip install -c /kit-build/constraints.txt -r /opt/openvla/experiments/robot/libero/libero_requirements.txt",
        "python -m pip install -c /kit-build/constraints.txt flash-attn --no-build-isolation",
        "python -m pip check",
    )
    .run_function(_finish_image)
    .add_local_dir(KIT_LOCAL, KIT_DIR, ignore=["**/__pycache__", "**/*.pyc"])  # mounted at start: no rebuild on edits
)

app = modal.App(APP_NAME, image=image)


# ----------------------------------------------------------------------------------------------------------------
# Helpers (run inside containers)
# ----------------------------------------------------------------------------------------------------------------
def rollout_args(ckpt: str, task_id: int, episodes: int, save_root: str, run_note: str) -> list[str]:
    """SAFE's run_libero_eval.py with SAFE's settings + the kit's all-layer capture. Same flags as smoke_test.sh."""
    return [
        "python", "experiments/robot/libero/run_libero_eval.py",
        "--model_family", "openvla",
        "--pretrained_checkpoint", ckpt,
        "--task_suite_name", "libero_10",
        "--center_crop", "True",
        "--attn_implementation", "flash_attention_2",
        "--output_hidden_states", "True",
        "--save_all_layers", "True",
        "--all_layers_pooling", POOLING,
        "--task_start_index", str(task_id),
        "--task_end_index", str(task_id),
        "--num_trials_per_task", str(episodes),
        "--resume", "True",
        "--seed", str(SEED),
        "--use_wandb", "False",
        "--save_logs", "True",
        "--save_root", save_root,
        "--run_id_note", run_note,
    ]


def _gpu_name() -> str:
    out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         check=True, capture_output=True, text=True).stdout
    return out.strip().splitlines()[0].strip()


def _checkpoint() -> str:
    p = Path(CHECKPOINT_PATH_FILE)
    if not p.exists():
        raise RuntimeError(f"{p} missing: run `modal run collection/modal_app.py::download_model` first")
    ckpt = p.read_text().strip()
    if not Path(ckpt, "config.json").exists():
        raise RuntimeError(f"checkpoint snapshot {ckpt} is not in the HF volume")
    return ckpt


def _guard_one_environment(run_note: str) -> None:
    """All tasks of one run must share the GPU type and the exact image (pip freeze + repo commits)."""
    if os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("MUJOCO_GL must be egl for every episode")
    ref = Path(WORK_DIR, "provenance", run_note)
    ref.mkdir(parents=True, exist_ok=True)
    current = {f.name: f.read_text() for f in Path(PROVENANCE_DIR).iterdir()}
    current["gpu_name.txt"] = _gpu_name() + "\n"
    for name, text in current.items():
        f = ref / name
        if not f.exists():
            f.write_text(text)
        elif f.read_text() != text:
            raise RuntimeError(f"{name} differs from the one recorded for run '{run_note}' in {ref}: "
                               "one GPU type and one environment for the whole dataset")
    data_volume.commit()


def _run(cmd: list[str], cwd: str | None = None, env: dict | None = None) -> None:
    print("+", " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True, env={**os.environ, **(env or {})})


OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


# ----------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------
@app.function(volumes=VOLUMES, timeout=DOWNLOAD_TIMEOUT_S, cpu=1.0, memory=2048)
def download_model() -> str:
    """CPU: fine-tuned checkpoint + the non-weight code files of the base repo into the HF volume (idempotent)."""
    from huggingface_hub import snapshot_download

    snap = snapshot_download(CHECKPOINT_REPO)
    snapshot_download(BASE_REPO, ignore_patterns=BASE_REPO_IGNORE)
    Path(WORK_DIR).mkdir(parents=True, exist_ok=True)
    old = Path(CHECKPOINT_PATH_FILE).read_text().strip() if Path(CHECKPOINT_PATH_FILE).exists() else None
    if old and old != snap:
        raise RuntimeError(f"checkpoint snapshot changed ({old} -> {snap}); one checkpoint for the whole dataset")
    Path(CHECKPOINT_PATH_FILE).write_text(snap + "\n")
    hf_volume.commit()
    data_volume.commit()
    print("checkpoint snapshot:", snap)
    return snap


@app.function(gpu=GPU, volumes=VOLUMES, timeout=SMOKE_TIMEOUT_S, cpu=ROLLOUT_CPU_CORES, memory=ROLLOUT_MEMORY_MIB)
def smoke_remote(task_id: int = 1, episodes: int = 2, full: bool = False) -> str:
    """Runs collection/smoke_test.sh on one A10 and returns its timing summary."""
    out = f"{SMOKE_ROOT}/{time.strftime('%Y%m%d-%H%M%S')}"
    env = {**OFFLINE, "CKPT": _checkpoint(), "SMOKE_ROOT": out, "SMOKE_TASK": str(task_id),
           "SMOKE_EPISODES": str(episodes), "SMOKE_FULL": "1" if full else "0"}
    try:
        _run(["bash", f"{KIT_DIR}/smoke_test.sh"], env=env)
    finally:
        data_volume.commit()
    return Path(out, "timing_summary.txt").read_text()


@app.function(gpu=GPU, volumes=VOLUMES, timeout=ROLLOUT_TIMEOUT_S, cpu=ROLLOUT_CPU_CORES, memory=ROLLOUT_MEMORY_MIB,
              max_containers=N_TASKS)
def rollout_task(task_id: int, episodes: int = EPISODES_PER_TASK, run_note: str = RUN_NOTE) -> dict:
    """One LIBERO-10 task (50 episodes, seed 7, --resume) on one A10; outputs go to the data volume."""
    assert 0 <= task_id < N_TASKS, task_id
    _guard_one_environment(run_note)
    t0 = time.time()
    try:
        _run(rollout_args(_checkpoint(), task_id, episodes, ROLLOUT_ROOT, run_note), cwd=OPENVLA_DIR, env=OFFLINE)
    finally:
        data_volume.commit()
    done = len(list(Path(ROLLOUT_ROOT, run_note, "libero_10").glob(f"task{task_id}--ep*--succ*.csv")))
    return {"task": task_id, "complete": done, "expected": episodes, "seconds": round(time.time() - t0)}


@app.function(volumes=VOLUMES, timeout=BUILD_TIMEOUT_S, cpu=2.0, memory=32768)
def build_cache_remote(run_note: str = RUN_NOTE, version: str = CACHE_VERSION, extra_build_args: str = "") -> None:
    """CPU: capture check on every episode -> build_cache.py -> check_cache.py verify, all on the data volume."""
    rollout_dir = f"{ROLLOUT_ROOT}/{run_note}/libero_10"
    prov = Path(WORK_DIR, "provenance", run_note)
    prov_files = sorted(str(p) for p in prov.glob("*.txt")) + [CHECKPOINT_PATH_FILE]
    try:
        _run(["python", f"{KIT_DIR}/check_cache.py", "capture", "--rollout-dir", rollout_dir])
        _run(["python", f"{KIT_DIR}/build_cache.py", "--rollout-dir", rollout_dir, "--cache-root", CACHE_ROOT,
              "--version", version, "--checkpoint", _checkpoint(), "--provenance-files", *prov_files,
              *shlex.split(extra_build_args)])
        _run(["python", f"{KIT_DIR}/check_cache.py", "verify", "--cache", f"{CACHE_ROOT}/{version}"])
    finally:
        data_volume.commit()


# ----------------------------------------------------------------------------------------------------------------
# Local entrypoints (run on your machine; they start the remote functions above)
# ----------------------------------------------------------------------------------------------------------------
def parse_tasks(spec: str) -> list[int]:
    """'0-9' or '3,7' or '1' -> sorted task ids in 0..9."""
    ids: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            ids.update(range(int(a), int(b) + 1))
        elif part:
            ids.add(int(part))
    if not ids or min(ids) < 0 or max(ids) >= N_TASKS:
        raise ValueError(f"tasks must be within 0..{N_TASKS - 1}, got {spec!r}")
    return sorted(ids)


def cost_estimate(n_tasks: int, sec_per_episode: float = 0.0) -> str:
    episodes = n_tasks * EPISODES_PER_TASK
    lines = [f"Launching {n_tasks} task(s) x {EPISODES_PER_TASK} episodes = {episodes} episodes, one {GPU} each."]
    a40 = [episodes * s / 3600 for s in A40_SEC_PER_EPISODE]
    lines.append(f"  A40 reference (MEASURED on CARC 2026-11-09, {A40_SEC_PER_EPISODE[0]}-{A40_SEC_PER_EPISODE[1]} s/episode):"
                 f" {a40[0]:.1f}-{a40[1]:.1f} GPU-h")
    if sec_per_episode > 0:
        h = episodes * sec_per_episode / 3600
        lines.append(f"  {GPU} from your smoke figure ({sec_per_episode:.0f} s/episode): {h:.1f} GPU-h ~ ${h * A10_USD_PER_HOUR:.0f}")
    else:
        lo, hi = (x * n_tasks / N_TASKS for x in A10_GPU_HOURS_ESTIMATE)
        lines.append(f"  {GPU} ESTIMATE (not measured; run the smoke test and pass --sec-per-episode): {lo:.0f}-{hi:.0f}"
                     f" GPU-h ~ ${lo * A10_USD_PER_HOUR:.0f}-{hi * A10_USD_PER_HOUR:.0f} GPU only")
    worst = n_tasks * ROLLOUT_TIMEOUT_S / 3600
    lines.append(f"  Upper bound if every container runs to its {ROLLOUT_TIMEOUT_S // 3600} h timeout: {worst:.0f} GPU-h"
                 f" ~ ${worst * A10_USD_PER_HOUR:.0f} GPU only.")
    lines.append(f"  Plus CPU/RAM per container ({ROLLOUT_CPU_CORES} cores, {ROLLOUT_MEMORY_MIB // 1024} GiB) and volume storage."
                 f" GPU price ${A10_USD_PER_HOUR:.2f}/h (modal.com/pricing, 2026-11-10).")
    return "\n".join(lines)


@app.local_entrypoint()
def smoke(task: int = 1, episodes: int = 2, full: bool = False):
    """1 task, 2 episodes by default (task 1: a seen task; 0, 3, 5 are the locked unseen tasks)."""
    print(f"Smoke test on one {GPU}: at most {SMOKE_TIMEOUT_S // 60} min ~ "
          f"${SMOKE_TIMEOUT_S / 3600 * A10_USD_PER_HOUR:.2f} GPU (upper bound).")
    download_model.remote()
    print(smoke_remote.remote(task, episodes, full))


@app.local_entrypoint()
def rollouts(confirm_full_run: bool = False, tasks: str = "0-9", sec_per_episode: float = 0.0,
             run_note: str = RUN_NOTE):
    task_ids = parse_tasks(tasks)
    print(cost_estimate(len(task_ids), sec_per_episode))
    if not (confirm_full_run or os.environ.get("CS566_CONFIRM_FULL_RUN") == "1"):
        print("Refusing to launch: pass --confirm-full-run (or set CS566_CONFIRM_FULL_RUN=1) after reading the "
              "estimate above and getting the OK for the spend.")
        sys.exit(2)
    download_model.remote()
    for res in rollout_task.map(task_ids, kwargs={"run_note": run_note}, return_exceptions=True):
        print(res)


@app.local_entrypoint()
def build_cache(run_note: str = RUN_NOTE, version: str = CACHE_VERSION, extra_build_args: str = ""):
    build_cache_remote.remote(run_note, version, extra_build_args)
