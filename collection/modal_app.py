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

import calendar
import json
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

# Timeouts. MEASURED on Modal A10, smoke test 2026-11-11 (kit 99adba5, image 0f65545c): 0.363 s/step on a 520-step
# (failed, capped) episode = 188.6-191.6 s per worst-case episode; ~30 s per process for imports + model load.
# Worst-case task = 30 + 50 x 191.6 s = 9,610 s (2.67 h). Timeout = 1.5 x that (Akash, 2026-11-10) = 14,415 s (4.0 h).
# A task that hits the timeout keeps its finished episodes (TASK_META stays "running"); see the runbook.
A10_WORST_EPISODE_S = 191.6  # MEASURED, 520 steps (the cap), Modal A10, 2026-11-11
A10_SEC_PER_EPISODE = (98.5, 191.6)  # MEASURED range, 268-520 steps, Modal A10, 2026-11-11
A10_PROCESS_OVERHEAD_S = 30  # MEASURED (resume run total 221 s - 191.6 s episode), imports + model load
ROLLOUT_TIMEOUT_S = int(1.5 * (A10_PROCESS_OVERHEAD_S + 50 * A10_WORST_EPISODE_S))  # = 14,415 s
SMOKE_TIMEOUT_S = 45 * 60  # 3 model loads + 2 episodes (+2 more with --full) + checks
BUILD_TIMEOUT_S = 8 * 3600  # same as CARC's 03_build_cache.slurm
DOWNLOAD_TIMEOUT_S = 60 * 60

# Prices from https://modal.com/pricing, checked 2026-11-10: A10 $0.000306/s; CPU $0.0000131/core/s; RAM $0.00000222/GiB/s.
A10_USD_PER_HOUR = 0.000306 * 3600  # ~= $1.10
A40_SEC_PER_EPISODE = (81, 157)  # measured on CARC A40, 2026-11-09
ROLLOUT_CPU_CORES = 2.0  # physical cores (= 4 vCPU), billed in addition to the GPU
CPU_USD_PER_CORE_HOUR = 0.0000131 * 3600
RAM_USD_PER_GIB_HOUR = 0.00000222 * 3600
WAVE_SIZE = 5  # Akash 2026-11-10: launch at most 5 tasks at a time; report spend and get approval between waves
CONTAINER_OVERHEAD_S = 180  # ESTIMATE: boot + image pull per container, billed but not inside the function's own timing
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
FINGERPRINT_FILE = "image_fingerprint.txt"
SMOKE_VALIDATED_FILE = "smoke_validated_images.json"  # committed record of image fingerprints a full smoke test passed on


def image_fingerprint(prov_dir: Path) -> str:
    """Content hash of an image: pip freeze, OS packages, Python, repo commits, patch, builder version.
    Modal image IDs are per workspace, so the same image built in two workspaces is compared by this hash."""
    import hashlib
    h = hashlib.sha256()
    for f in sorted(p for p in Path(prov_dir).iterdir() if p.is_file() and p.name != FINGERPRINT_FILE):
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


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
    dpkg = subprocess.run(["dpkg-query", "-W", "-f=${Package}=${Version}\n"], check=True, capture_output=True, text=True)
    Path(PROVENANCE_DIR, "dpkg_packages.txt").write_text("".join(sorted(dpkg.stdout.splitlines(keepends=True))))
    Path(PROVENANCE_DIR, "python_version.txt").write_text(sys.version + "\n")
    Path(PROVENANCE_DIR, FINGERPRINT_FILE).write_text(image_fingerprint(Path(PROVENANCE_DIR)) + "\n")


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
def rollout_args(ckpt: str, task_id: int, episodes: int, save_root: str, run_note: str, resume: bool = False) -> list[str]:
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
        "--resume", "True" if resume else "False",  # dataset runs: never (one uninterrupted process per task)
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
    # This workspace's image provenance (pip freeze, repo commits, builder version) into the volume, so images built
    # in different workspaces can be compared file by file (`modal volume get ... work/image_provenance`).
    dst = Path(WORK_DIR, "image_provenance")
    dst.mkdir(parents=True, exist_ok=True)
    for f in Path(PROVENANCE_DIR).iterdir():
        (dst / f.name).write_text(f.read_text())
    hf_volume.commit()
    data_volume.commit()
    print("checkpoint snapshot:", snap)
    return snap


@app.function(gpu=GPU, volumes=VOLUMES, timeout=SMOKE_TIMEOUT_S, cpu=ROLLOUT_CPU_CORES, memory=ROLLOUT_MEMORY_MIB)
def smoke_remote(task_id: int = 1, episodes: int = 2, full: bool = False, workspace_profile: str = "",
                 kit_commit: str = "") -> str:
    """Runs collection/smoke_test.sh on one A10 and returns its timing summary. A FULL pass (resume, capture, repeat,
    build) records this image's fingerprint as validated in this workspace's volume (work/smoke_passes/)."""
    out = f"{SMOKE_ROOT}/{time.strftime('%Y%m%d-%H%M%S')}"
    env = {**OFFLINE, "CKPT": _checkpoint(), "SMOKE_ROOT": out, "SMOKE_TASK": str(task_id),
           "SMOKE_EPISODES": str(episodes), "SMOKE_FULL": "1" if full else "0"}
    try:
        _run(["bash", f"{KIT_DIR}/smoke_test.sh"], env=env)  # raises unless every smoke check passed
        summary = Path(out, "timing_summary.txt").read_text()
        if full:
            fp, image_id = current_image()
            rec = {"image_fingerprint": fp, "modal_image_id": image_id, "workspace_profile": workspace_profile,
                   "kit_commit": kit_commit, "gpu_name": _gpu_name(), "smoke_dir": out, "task": task_id,
                   "episodes": episodes, "full": True, "passed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "timing_summary": summary}
            Path(WORK_DIR, "smoke_passes").mkdir(parents=True, exist_ok=True)
            _write_json_atomic(Path(WORK_DIR, "smoke_passes", f"{fp}.json"), rec)
            summary += f"\nimage fingerprint {fp} (Modal image {image_id}) recorded as smoke-validated\n"
    finally:
        data_volume.commit()
    return summary


def current_image() -> tuple[str, str | None]:
    """(content fingerprint, Modal image ID if the runtime exposes it) of the image this container runs."""
    fp = Path(PROVENANCE_DIR, FINGERPRINT_FILE).read_text().strip()
    if fp != image_fingerprint(Path(PROVENANCE_DIR)):
        raise RuntimeError("image provenance does not match its recorded fingerprint")
    return fp, os.environ.get("MODAL_IMAGE_ID")


def smoke_validation(fp: str) -> dict | None:
    """The full smoke pass recorded for this image fingerprint: in this workspace's volume, or committed in
    collection/smoke_validated_images.json (a pass in another workspace on a byte-identical image)."""
    local = Path(WORK_DIR, "smoke_passes", f"{fp}.json")
    if local.exists():
        return {"source": "this workspace", **json.loads(local.read_text())}
    committed = Path(KIT_DIR, SMOKE_VALIDATED_FILE)
    if committed.exists():
        rec = json.loads(committed.read_text()).get(fp)
        if rec:
            return {"source": f"committed {SMOKE_VALIDATED_FILE}", **rec}
    return None


@app.function(volumes=VOLUMES, timeout=300, cpu=0.25, memory=512)
def image_check() -> dict:
    """CPU: this workspace's image fingerprint and its smoke validation (checked before any GPU is launched)."""
    data_volume.reload()
    fp, image_id = current_image()
    return {"image_fingerprint": fp, "modal_image_id": image_id, "smoke_validation": smoke_validation(fp)}


def usd_per_container_hour() -> float:
    return A10_USD_PER_HOUR + ROLLOUT_CPU_CORES * CPU_USD_PER_CORE_HOUR + ROLLOUT_MEMORY_MIB / 1024 * RAM_USD_PER_GIB_HOUR


def task_dir(run_note: str, task_id: int) -> Path:
    """One folder per task, so finished tasks can be pulled independently: rollouts/<run>/taskNN/."""
    return Path(ROLLOUT_ROOT, run_note, f"task{task_id:02d}")


def _write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def episode_accounting(ep_dir: Path, task_id: int, episodes: int) -> dict:
    """Per episode: complete (.csv + .pkl + .layers.pt), invalid (marker only) or missing."""
    complete, invalid, missing = [], [], []
    for e in range(episodes):
        exts = {p.name.split(".", 1)[1] for p in ep_dir.glob(f"task{task_id}--ep{e}--succ*")}
        if (ep_dir / f"task{task_id}--ep{e}--invalid.json").exists() and not exts:
            invalid.append(e)
        elif {"csv", "pkl", "layers.pt"} <= exts:
            complete.append(e)
        else:
            missing.append(e)
    return {"complete": len(complete), "invalid": invalid, "missing": missing}


def write_sha256sums(tdir: Path) -> int:
    """`sha256sum -c` format, paths relative to the task folder (only libero_10/ is covered)."""
    import hashlib
    lines = []
    for f in sorted(p for p in (tdir / "libero_10").rglob("*") if p.is_file() and not p.name.endswith(".tmp")):
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {f.relative_to(tdir)}")
    (tdir / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return len(lines)


DISCARD_REASONS = {
    "running": "previous attempt never finished (container stopped: Modal preemption/restart, timeout, credit limit or kill)",
    "partial": "previous attempt ended with missing episodes",
    "failed": "previous attempt failed",
    "engineering-complete": "engineering (resumed) output is never part of the dataset",
    None: "outputs present without TASK_META",
}


def discard_partial(tdir: Path, discard_dir: Path, reason: str, utc: str) -> dict | None:
    """Move a task's partial outputs (libero_10/, SHA256SUMS, a copy of TASK_META.json) to discard_dir. Never deletes.
    Returns the record for TASK_META["discarded_attempts"], or None if there was nothing to move."""
    import shutil
    items = [n for n in ("libero_10", "SHA256SUMS") if (tdir / n).exists()]
    if not items:
        return None
    discard_dir.mkdir(parents=True, exist_ok=False)
    files = sum(1 for p in (tdir / "libero_10").rglob("*") if p.is_file()) if (tdir / "libero_10").exists() else 0
    for n in items:
        shutil.move(str(tdir / n), str(discard_dir / n))
    if (tdir / "TASK_META.json").exists():
        shutil.copy2(tdir / "TASK_META.json", discard_dir / "TASK_META.json")
    return {"discarded_utc": utc, "moved_to": str(discard_dir), "reason": reason, "files_moved": files}


@app.function(gpu=GPU, volumes=VOLUMES, timeout=ROLLOUT_TIMEOUT_S, cpu=ROLLOUT_CPU_CORES, memory=ROLLOUT_MEMORY_MIB,
              max_containers=WAVE_SIZE)
def rollout_task(task_id: int, workspace_profile: str, kit_commit: str, episodes: int = EPISODES_PER_TASK,
                 run_note: str = RUN_NOTE, engineering_resume: bool = False) -> dict:
    """One LIBERO-10 task (50 episodes, seed 7) on one A10, into its own folder rollouts/<run>/taskNN/.

    Dataset policy (Akash, 2026-11-11): every task runs as ONE uninterrupted process, episodes 0-49 in order, because
    an episode's trajectory depends on the episodes before it in the same process (smoke test 2026-11-11). So any
    partial output found at start, from an earlier attempt, a Modal preemption/restart of this same input, a timeout
    or a credit limit, is moved to rollouts/<run>/_discarded/taskNN-<utc>/ (never deleted) and the task restarts at
    episode 0 without --resume. TASK_META.json records every discarded attempt and why.
    `engineering_resume=True` keeps partial output and passes --resume: engineering only. The dataset launcher never
    sets it, and its output is marked "engineering-complete", which the CARC puller and the cache build refuse."""
    assert 0 <= task_id < N_TASKS, task_id
    tdir = task_dir(run_note, task_id)
    tdir.mkdir(parents=True, exist_ok=True)
    meta_path = tdir / "TASK_META.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get("status") == "complete":
        return meta
    if meta and meta.get("workspace_profile") != workspace_profile:
        raise RuntimeError(f"task {task_id} was started in workspace {meta.get('workspace_profile')!r}; "
                           "a partial task is rerun from scratch in a fresh volume, never continued across workspaces")
    fp, image_id = current_image()
    validation = smoke_validation(fp)
    if validation is None:
        raise RuntimeError(f"image {fp[:16]} (Modal {image_id}) has not passed a full smoke test; run "
                           "`modal run collection/modal_app.py::smoke --full` in this workspace first")
    if engineering_resume and meta.get("image_fingerprint") not in (None, fp):
        raise RuntimeError(f"task {task_id} was started on image {meta['image_fingerprint'][:16]}, now {fp[:16]}")
    _guard_one_environment(run_note)
    gpu = _gpu_name()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not engineering_resume:
        reason = DISCARD_REASONS.get(meta.get("status"), f"previous status {meta.get('status')!r}")
        rec = discard_partial(tdir, Path(ROLLOUT_ROOT, run_note, "_discarded", f"task{task_id:02d}-{now.replace(':', '')}"),
                              reason, now)
        if rec or meta.get("attempts"):
            rec = rec or {"discarded_utc": now, "moved_to": None, "reason": reason, "files_moved": 0}
            meta.setdefault("discarded_attempts", []).append({**rec, "attempts": meta.get("attempts", []),
                                                              "previous_status": meta.get("status")})
            meta["attempts"] = []
            data_volume.commit()
    done_before = episode_accounting(tdir / "libero_10", task_id, episodes)
    attempt = {"container_id": os.environ.get("MODAL_TASK_ID"), "started_utc": now,
               # 0 for every dataset attempt; > 0 only for an engineering resume
               "first_episode": done_before["complete"] + len(done_before["invalid"]),
               "engineering_resume": engineering_resume,
               "workspace_profile": workspace_profile, "gpu_name": gpu, "kit_commit": kit_commit}
    attempt["modal_image_id"] = image_id
    meta.update({"task": task_id, "run_note": run_note, "workspace_profile": workspace_profile, "gpu_name": gpu,
                 "kit_commit": kit_commit, "image_builder_version": IMAGE_BUILDER_VERSION,
                 "image_fingerprint": fp, "modal_image_id": image_id,
                 "smoke_validation": {k: validation.get(k) for k in ("source", "workspace_profile", "passed_utc",
                                                                     "modal_image_id", "kit_commit")},
                 "episodes_expected": episodes, "timeout_s": ROLLOUT_TIMEOUT_S, "cpu_cores": ROLLOUT_CPU_CORES,
                 "memory_mib": ROLLOUT_MEMORY_MIB, "status": "running",
                 "dataset_policy": "engineering resume (not dataset)" if engineering_resume
                 else "uninterrupted: episodes 0-49 in one process, no --resume"})
    meta.setdefault("attempts", []).append(attempt)
    _write_json_atomic(meta_path, meta)
    data_volume.commit()
    t0, error = time.time(), None
    try:
        _run(rollout_args(_checkpoint(), task_id, episodes, str(tdir.parent), tdir.name, resume=engineering_resume),
             cwd=OPENVLA_DIR, env=OFFLINE)
    except subprocess.CalledProcessError as e:
        error = f"run_libero_eval.py exited with {e.returncode}"
    finally:
        acct = episode_accounting(tdir / "libero_10", task_id, episodes)
        done = not acct["missing"] and error is None
        attempt.update({"finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "seconds": round(time.time() - t0), "error": error})
        status = ("engineering-complete" if engineering_resume else "complete") if done else ("failed" if error else "partial")
        meta.update({"episodes_complete": acct["complete"], "episodes_invalid": acct["invalid"],
                     "episodes_missing": acct["missing"], "status": status})
        if done:
            meta["sha256sums_files"] = write_sha256sums(tdir)
        _write_json_atomic(meta_path, meta)
        data_volume.commit()
    if error:
        raise RuntimeError(f"task {task_id}: {error}; TASK_META status {meta['status']}")
    return meta


@app.function(volumes=VOLUMES, timeout=600, cpu=0.25, memory=512)
def task_status(run_note: str = RUN_NOTE) -> list[dict]:
    """CPU: every task's TASK_META.json in this workspace's volume (for spend reports)."""
    data_volume.reload()
    out = []
    for t in range(N_TASKS):
        f = task_dir(run_note, t) / "TASK_META.json"
        out.append(json.loads(f.read_text()) if f.exists() else {"task": t, "status": "not started"})
    return out


@app.function(volumes=VOLUMES, timeout=BUILD_TIMEOUT_S, cpu=2.0, memory=32768)
def build_cache_remote(run_note: str = RUN_NOTE, version: str = CACHE_VERSION, extra_build_args: str = "") -> None:
    """CPU: capture check on every episode -> build_cache.py -> check_cache.py verify, all on the data volume."""
    rollout_dir = "/tmp/flat/libero_10"  # flat view (symlinks) of rollouts/<run>/taskNN/libero_10, as build_cache reads
    Path(rollout_dir, "provenance").mkdir(parents=True, exist_ok=True)
    for t in range(N_TASKS):
        src = task_dir(run_note, t) / "libero_10"
        if json.loads((task_dir(run_note, t) / "TASK_META.json").read_text()).get("status") != "complete":
            raise RuntimeError(f"task {t} is not complete in this workspace; build on CARC from pulled tasks instead")
        for f in src.glob("task*"):
            os.symlink(f, Path(rollout_dir, f.name))
        for f in (src / "provenance").glob("*"):
            os.symlink(f, Path(rollout_dir, "provenance", f.name))
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
        lo, hi = (episodes * x / 3600 for x in A10_SEC_PER_EPISODE)
        lines.append(f"  {GPU} from the MEASURED smoke range ({A10_SEC_PER_EPISODE[0]:.0f}-{A10_SEC_PER_EPISODE[1]:.0f} s/episode)"
                     f": {lo:.1f}-{hi:.1f} container-h ~ ${lo * usd_per_container_hour():.0f}"
                     f"-{hi * usd_per_container_hour():.0f}")
    worst = n_tasks * ROLLOUT_TIMEOUT_S / 3600
    lines.append(f"  Upper bound if every container runs to its {ROLLOUT_TIMEOUT_S / 3600:.1f} h timeout: {worst:.1f} container-h"
                 f" ~ ${worst * usd_per_container_hour():.0f}.")
    lines.append(f"  Container rate ${usd_per_container_hour():.3f}/h = A10 ${A10_USD_PER_HOUR:.2f}/h + {ROLLOUT_CPU_CORES} cores"
                 f" + {ROLLOUT_MEMORY_MIB // 1024} GiB (modal.com/pricing, 2026-11-10); volume storage is free under 1 TiB."
                 " Episode range = the 4 smoke episodes (268-520 steps); real tasks vary.")
    return "\n".join(lines)


@app.local_entrypoint()
def smoke(task: int = 1, episodes: int = 2, full: bool = False):
    """1 task, 2 episodes by default (task 1: a seen task; 0, 3, 5 are the locked unseen tasks)."""
    print(f"Smoke test on one {GPU}: at most {SMOKE_TIMEOUT_S // 60} min ~ "
          f"${SMOKE_TIMEOUT_S / 3600 * A10_USD_PER_HOUR:.2f} GPU (upper bound).")
    profile, commit = _launch_context()
    download_model.remote()
    print(smoke_remote.remote(task, episodes, full, profile, commit))


def _launch_context() -> tuple[str, str]:
    """(workspace profile, kit commit). The profile must be chosen explicitly; the tree must be clean so the
    commit recorded in every TASK_META.json is the code that ran."""
    profile = os.environ.get("MODAL_PROFILE", "")
    if not profile:
        raise SystemExit("Set MODAL_PROFILE (e.g. MODAL_PROFILE=friend) so the workspace is chosen explicitly.")
    repo = KIT_LOCAL.parent
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                           check=True, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f"Refusing to launch from a modified tree (commit first):\n{dirty}")
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True,
                            text=True).stdout.strip()
    return profile, commit


@app.local_entrypoint()
def rollouts(confirm_full_run: bool = False, tasks: str = "", sec_per_episode: float = 0.0,
             run_note: str = RUN_NOTE, restart_stale: bool = False):
    """Launch ONE wave (at most WAVE_SIZE tasks), e.g. MODAL_PROFILE=friend modal run --detach
    collection/modal_app.py::rollouts --tasks 0-4 --sec-per-episode <smoke> --confirm-full-run"""
    if not tasks:
        raise SystemExit(f"Pass --tasks explicitly (a wave of at most {WAVE_SIZE} tasks, e.g. 0-4).")
    task_ids = parse_tasks(tasks)
    if len(task_ids) > WAVE_SIZE:
        raise SystemExit(f"{len(task_ids)} tasks requested; launch waves of at most {WAVE_SIZE} and report spend between them.")
    profile, commit = _launch_context()
    print(f"Workspace profile: {profile}   kit commit: {commit[:12]}   run: {run_note}   tasks: {task_ids}")
    print(cost_estimate(len(task_ids), sec_per_episode))
    if not (confirm_full_run or os.environ.get("CS566_CONFIRM_FULL_RUN") == "1"):
        print("Refusing to launch: pass --confirm-full-run (or set CS566_CONFIRM_FULL_RUN=1) after reading the "
              "estimate above and getting the OK for the spend.")
        sys.exit(2)
    download_model.remote()
    img = image_check.remote()
    if img["smoke_validation"] is None:
        raise SystemExit(f"Image {img['image_fingerprint'][:16]} (Modal {img['modal_image_id']}) in workspace {profile} "
                         "has not passed a full smoke test. Run `modal run collection/modal_app.py::smoke --full` here "
                         "first (needs Akash's OK for the spend). No GPU was started.")
    print(f"Image {img['image_fingerprint'][:16]} validated by smoke test: {img['smoke_validation']['source']}, "
          f"{img['smoke_validation'].get('passed_utc')}")
    states = {m["task"]: m.get("status") for m in task_status.remote(run_note)}
    running = [t for t in task_ids if states.get(t) == "running"]
    if running and not restart_stale:
        raise SystemExit(f"Tasks {running} are marked 'running' in workspace {profile}. If `modal app list` shows no app "
                         "still running them, relaunch with --restart-stale: their partial outputs are moved to "
                         "_discarded/ and they restart at episode 0. No GPU was started.")
    skipped = [t for t in task_ids if states.get(t) == "complete"]
    if skipped:
        print(f"Already complete, not relaunched: {skipped}")
        task_ids = [t for t in task_ids if t not in skipped]
    for res in rollout_task.map(task_ids, kwargs={"run_note": run_note, "workspace_profile": profile,
                                                  "kit_commit": commit}, return_exceptions=True,
                                 order_outputs=False):
        print(res if isinstance(res, Exception) else
              {k: res.get(k) for k in ("task", "status", "episodes_complete", "episodes_invalid", "episodes_missing")})
    print(spend_report(task_status.remote(run_note)))


def spend_report(metas: list[dict]) -> str:
    """Container-hours x published rates, per workspace, plus a projection for the tasks not yet complete.
    Estimates: add CONTAINER_OVERHEAD_S per attempt for boot/image pull; check the Modal dashboard as well."""
    rate = usd_per_container_hour()
    lines = [f"{'task':>4} {'status':<12} {'done':>4} {'inv':>3} {'workspace':<10} {'gpu':<12} {'hours':>6} {'USD':>6}"]
    per_ws: dict[str, float] = {}
    done_hours = []
    for m in metas:
        secs_by_ws: dict[str, float] = {}
        for a in m.get("attempts", []) + [x for d in m.get("discarded_attempts", []) for x in d.get("attempts", [])]:
            secs = a.get("seconds")
            if secs is None and a.get("started_utc"):  # still running or killed: count up to now (upper bound)
                secs = time.time() - calendar.timegm(time.strptime(a["started_utc"], "%Y-%m-%dT%H:%M:%SZ"))
            secs_by_ws[a.get("workspace_profile", "?")] = secs_by_ws.get(a.get("workspace_profile", "?"), 0) \
                + (secs or 0) + CONTAINER_OVERHEAD_S
        hours = sum(secs_by_ws.values()) / 3600
        for ws, sec in secs_by_ws.items():
            per_ws[ws] = per_ws.get(ws, 0.0) + sec / 3600 * rate
        if m.get("status") == "complete":
            done_hours.append(hours)
        lines.append(f"{m['task']:>4} {m.get('status', '?'):<12} {m.get('episodes_complete', 0):>4} "
                     f"{len(m.get('episodes_invalid', [])):>3} {m.get('workspace_profile', '-'):<10} "
                     f"{m.get('gpu_name', '-'):<12} {hours:>6.2f} {hours * rate:>6.2f}")
    lines.append(f"Rate: ${rate:.3f}/container-hour (A10 + {ROLLOUT_CPU_CORES} cores + {ROLLOUT_MEMORY_MIB // 1024} GiB), "
                 f"+{CONTAINER_OVERHEAD_S} s per attempt (estimate).")
    for ws, usd in sorted(per_ws.items()):
        lines.append(f"Spent so far on rollouts, workspace {ws}: ~${usd:.2f}")
    remaining = [m["task"] for m in metas if m.get("status") != "complete"]
    if remaining and done_hours:
        per_task = sum(done_hours) / len(done_hours)
        lines.append(f"Projected for the {len(remaining)} unfinished task(s) {remaining}: ~{per_task * len(remaining):.1f} h "
                     f"~ ${per_task * len(remaining) * rate:.2f} (mean of {len(done_hours)} completed task(s)).")
    return "\n".join(lines)


@app.local_entrypoint()
def status(run_note: str = RUN_NOTE):
    """Per-task status and spend in the workspace selected by MODAL_PROFILE (CPU only, ~free)."""
    print(f"Workspace profile: {os.environ.get('MODAL_PROFILE', '(default profile)')}")
    print(spend_report(task_status.remote(run_note)))


@app.local_entrypoint()
def build_cache(run_note: str = RUN_NOTE, version: str = CACHE_VERSION, extra_build_args: str = ""):
    build_cache_remote.remote(run_note, version, extra_build_args)
