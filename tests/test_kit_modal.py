"""CPU-only checks for the Modal app, the portable smoke test and the CARC Slurm scripts (must-fix 1).

Nothing here contacts Modal. Most checks are static (AST / text / `bash -n`). The import test runs only when the
`modal` package is importable; importing modal_app.py just constructs the App object (no network, no cost).
"""
import ast
import importlib.util
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "collection"
MODAL_APP = KIT / "modal_app.py"
SMOKE = KIT / "smoke_test.sh"
SLURMS = [KIT / "carc" / f for f in ("01_smoke_test.slurm", "02_rollouts.slurm", "03_build_cache.slurm")]
GPU_SLURMS = SLURMS[:2]
CONSTRAINTS_FLAG = "-c /kit-build/constraints.txt"


def _src() -> str:
    return MODAL_APP.read_text()


def _tree() -> ast.Module:
    return ast.parse(_src())


def _constant(name):
    for node in _tree().body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            # literals and constant arithmetic such as 4 * 3600; no names, no builtins
            return eval(compile(ast.Expression(node.value), "modal_app.py", "eval"), {"__builtins__": {}}, {})
    raise AssertionError(f"{name} not assigned at module level in modal_app.py")


def _function(name) -> ast.FunctionDef:
    for node in _tree().body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found in modal_app.py")


def _code_lines(text: str):
    return [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]


# ---------------------------------------------------------------- modal_app.py
def test_gpu_is_a10_everywhere():
    assert _constant("GPU") == "A10"
    gpu_kwargs = []
    for node in ast.walk(_tree()):
        if isinstance(node, ast.keyword) and node.arg == "gpu":
            gpu_kwargs.append(node.value)
    assert gpu_kwargs, "no gpu= keyword found"
    for v in gpu_kwargs:  # every gpu= must use the single GPU constant
        assert isinstance(v, ast.Name) and v.id == "GPU", ast.dump(v)


def test_every_pip_install_uses_constraints():
    lines = [ln for ln in _code_lines(_src()) if "python -m pip install" in ln]
    assert len(lines) >= 6
    for ln in lines:
        assert CONSTRAINTS_FLAG in ln, ln
    assert ".pip_install(" not in _src() and ".uv_pip_install(" not in _src(), "use explicit pip with -c constraints"
    assert 'KIT_LOCAL / "constraints.txt", f"{BUILD_DIR}/constraints.txt"' in _src()
    assert _constant("BUILD_DIR") == "/kit-build"


def test_flash_attn_installed_last_without_build_isolation_and_cuda_home():
    src = _src()
    pip_lines = [ln for ln in _code_lines(src) if "python -m pip install" in ln]
    assert "flash-attn" in pip_lines[-1] and "--no-build-isolation" in pip_lines[-1]
    assert '"CUDA_HOME": "/usr/local/cuda"' in src
    assert re.search(r'from_registry\("nvidia/cuda:12\.1[.\d]*-cudnn\d-devel-ubuntu22\.04", add_python="3\.10"\)', src)
    for name in ("torch", "torchvision", "transformers", "tokenizers", "tensorflow", "flash-attn", "robosuite"):
        assert name in _constant("REQUIRED_EXACT_PINS")
    for name in ("numpy", "protobuf", "tensorflow-metadata", "mujoco"):
        assert name in _constant("REQUIRED_RANGE_PINS")


def test_openvla_pinned_and_patch_applied_with_git_apply():
    src = _src()
    commit = _constant("OPENVLA_COMMIT")
    assert commit.startswith("300dce2") and len(commit) == 40
    assert "git checkout --detach {OPENVLA_COMMIT}" in src
    assert 'test \\"$(git rev-parse HEAD)\\" = {OPENVLA_COMMIT}' in src
    assert "git apply --check {BUILD_DIR}/openvla_all_layers.patch" in src
    assert "git apply {BUILD_DIR}/openvla_all_layers.patch" in src
    assert "patch -p" not in src and "sed -i" not in src


def test_libero_pth_config_egl_and_offline_hf():
    src = _src()
    assert '"libero_repo.pth"' in src and "config.yaml" in src and "stdin=subprocess.DEVNULL" in src
    assert _constant("LIBERO_COMMIT") and len(_constant("LIBERO_COMMIT")) == 40
    assert '"MUJOCO_GL": "egl"' in src and '"libegl1"' in src
    assert _constant("BASE_REPO") == "openvla/openvla-7b"
    assert "*.safetensors" in _constant("BASE_REPO_IGNORE")
    assert '"HF_HOME": HF_MOUNT' in src and '"HF_HUB_OFFLINE": "1"' in src


def test_rollout_settings_are_safes_plus_all_layers():
    assert _constant("SEED") == 7 and _constant("EPISODES_PER_TASK") == 50 and _constant("N_TASKS") == 10
    flags = [c.value for c in ast.walk(_function("rollout_args"))
             if isinstance(c, ast.Constant) and isinstance(c.value, str) and c.value.startswith("--")]
    for f in ("--save_all_layers", "--output_hidden_states", "--resume", "--seed", "--center_crop"):
        assert f in flags
    smoke_flags = re.findall(r"(--[a-z_]+)", SMOKE.read_text().split("rollout () {", 1)[1].split("\n}", 1)[0])
    assert sorted(set(flags)) == sorted(set(smoke_flags)), "modal_app.py and smoke_test.sh must use the same flags"


def test_timeouts_are_bounded_and_explained():
    # Akash 2026-11-10: task timeout = 1.5 x the worst-case task time MEASURED in the Modal smoke test
    worst_ep, overhead = _constant("A10_WORST_EPISODE_S"), _constant("A10_PROCESS_OVERHEAD_S")
    assert (worst_ep, overhead) == (191.6, 30)                       # smoke test 2026-11-11, 520-step episode
    assert "ROLLOUT_TIMEOUT_S = int(1.5 * (A10_PROCESS_OVERHEAD_S + 50 * A10_WORST_EPISODE_S))" in _src()
    assert int(1.5 * (overhead + 50 * worst_ep)) == 14415
    assert _constant("SMOKE_TIMEOUT_S") <= 3600
    assert _constant("BUILD_TIMEOUT_S") <= 24 * 3600


def test_full_run_is_opt_in():
    fn = _function("rollouts")
    args = [a.arg for a in fn.args.args]
    defaults = dict(zip(args[-len(fn.args.defaults):], fn.args.defaults))
    assert "confirm_full_run" in defaults and defaults["confirm_full_run"].value is False
    body = ast.unparse(fn)
    assert body.index("cost_estimate(") < body.index("confirm_full_run or")  # estimate printed before the check
    assert body.index("sys.exit(2)") < body.index("rollout_task.map(")
    assert "CS566_CONFIRM_FULL_RUN" in body


def test_cpu_function_runs_capture_build_verify():
    body = ast.unparse(_function("build_cache_remote"))
    i, j, k = body.index("check_cache.py', 'capture'"), body.index("build_cache.py'"), body.index("'verify'")
    assert i < j < k


@pytest.mark.skipif(importlib.util.find_spec("modal") is None, reason="modal not installed")
def test_modal_app_constructs():
    spec = importlib.util.spec_from_file_location("modal_app_under_test", MODAL_APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # builds the App/Image definitions only; nothing is sent to Modal
    assert mod.app.name == "cs566-vla-collection"
    assert mod.parse_tasks("0-9") == list(range(10)) and mod.parse_tasks("3,7") == [3, 7]
    with pytest.raises(ValueError):
        mod.parse_tasks("10")
    assert "ESTIMATE" in mod.cost_estimate(10)


# ---------------------------------------------------------------- shell scripts
@pytest.mark.parametrize("path", [SMOKE, *SLURMS], ids=lambda p: p.name)
def test_bash_syntax(path):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    subprocess.run([bash, "-n", str(path)], check=True)


def test_smoke_script_contents():
    s = SMOKE.read_text()
    assert 'if [ "${KIT_ON_MODAL:-0}" != "1" ]; then' in s
    assert 'source "$HERE/carc/activate_env.sh"' in s
    assert s.index('source "$HERE/carc/config.sh"') < s.index('source "$HERE/carc/activate_env.sh"')
    assert "MUJOCO_GL" in s and "import flash_attn, flash_attn_2_cuda" in s
    assert "--save_all_layers True" in s and 'check_cache.py" capture' in s
    assert "< /dev/null" in s  # LIBERO import must not wait on a prompt
    assert "attn_implementation sdpa" not in s and "--attn_implementation flash_attention_2" in s


@pytest.mark.parametrize("path", GPU_SLURMS, ids=lambda p: p.name)
def test_slurm_gpu_syntax(path):
    s = path.read_text()
    assert "#SBATCH -n 1\n" in s and "#SBATCH --gpus=a40:1\n" in s
    assert "--gpus-per-task" not in _code_lines(s) and "#SBATCH --gpus-per-task" not in s
    assert 'grep -qi "$GPU_TYPE"' in s  # hard-coded #SBATCH type must match config.sh


@pytest.mark.parametrize("path", SLURMS, ids=lambda p: p.name)
def test_slurm_sources_config_then_activate(path):
    s = path.read_text()
    assert s.index("source carc/config.sh") < s.index("source carc/activate_env.sh")
    assert "conda activate" not in s  # activation lives in carc/activate_env.sh (conda or uv)


def test_smoke_slurm_calls_smoke_script():
    assert "bash smoke_test.sh" in SLURMS[0].read_text()


@pytest.mark.parametrize("path", [MODAL_APP, SMOKE, *SLURMS, KIT / "README.md"], ids=lambda p: p.name)
def test_no_site_specific_identifiers(path):
    s = path.read_text()
    for m in re.finditer(r"--account=(\S+)", s):
        assert m.group(1)[0] in "<$", f"use a placeholder or variable account, got {m.group(0)}"
    assert not re.search(r"/(scratch1|project2?)/(?![<$])[A-Za-z0-9_]+", s), "use /scratch1/<user>, /project2/<group>"
    assert not re.search(r"\b[a-z]+_\d{4}\b", s), "looks like a Slurm account/project ID"


def test_image_builder_pinned_for_python_310():
    # Modal's 2026.10 builder rejects add_python="3.10"; 2025.06 is pinned in the client so every workspace
    # builds the same image (found 2026-11-10 when the first smoke launch failed locally before any spend).
    src = _src()
    assert 'IMAGE_BUILDER_VERSION = "2025.06"' in src
    assert src.index('os.environ["MODAL_IMAGE_BUILDER_VERSION"] = IMAGE_BUILDER_VERSION') < src.index("import modal  #")
    assert 'add_python="3.10"' in src
    assert '"image_builder_version.txt"' in src


# ---------------------------------------------------------------- workspace-portable full run (2026-11-10)
def _helpers(**consts):
    """Exec the pure helper functions of modal_app.py without importing modal."""
    import ast as _ast
    import calendar, hashlib, json as _json, os as _os, time as _time  # noqa: F401,E401
    tree = _ast.parse(_src())
    names = {"usd_per_container_hour", "task_dir", "episode_accounting", "write_sha256sums", "spend_report",
             "image_fingerprint"}
    ns = {"Path": Path, "json": _json, "os": _os, "time": _time, "calendar": calendar,
          "A10_USD_PER_HOUR": 0.000306 * 3600, "CPU_USD_PER_CORE_HOUR": 0.0000131 * 3600,
          "RAM_USD_PER_GIB_HOUR": 0.00000222 * 3600, "ROLLOUT_CPU_CORES": 2.0, "ROLLOUT_MEMORY_MIB": 16384,
          "CONTAINER_OVERHEAD_S": 180, "ROLLOUT_ROOT": "/vol/data/rollouts",
          "FINGERPRINT_FILE": "image_fingerprint.txt", **consts}
    code = [n for n in tree.body if isinstance(n, _ast.FunctionDef) and n.name in names]
    exec(compile(_ast.Module(body=code, type_ignores=[]), "modal_app_helpers", "exec"), ns)
    return ns


def test_one_folder_per_task_and_wave_limit():
    src = _src()
    assert 'return Path(ROLLOUT_ROOT, run_note, f"task{task_id:02d}")' in src
    assert "WAVE_SIZE = 5" in src and "max_containers=WAVE_SIZE" in src
    assert "if len(task_ids) > WAVE_SIZE:" in src
    assert 'os.environ.get("MODAL_PROFILE", "")' in src            # workspace chosen explicitly
    assert '"status", "--porcelain"' in src                          # launched only from a committed tree
    assert "never continued across workspaces" in src


def test_episode_accounting_and_checksums(tmp_path):
    h = _helpers()
    ep = tmp_path / "task01" / "libero_10"
    (ep / "provenance").mkdir(parents=True)
    for e, exts in ((0, ("csv", "pkl", "layers.pt", "mp4")), (2, ("mp4", "pkl"))):
        for x in exts:
            (ep / f"task1--ep{e}--succ0.{x}").write_text(x)
    (ep / "task1--ep1--invalid.json").write_text("{}")
    (ep / "task1--ep0--succ0.layers.pt.tmp").write_text("partial")
    acct = h["episode_accounting"](ep, 1, 3)
    assert acct == {"complete": 1, "invalid": [1], "missing": [2]}
    n = h["write_sha256sums"](tmp_path / "task01")
    lines = (tmp_path / "task01" / "SHA256SUMS").read_text().splitlines()
    assert n == len(lines) == 7 and all(len(l.split("  ")[0]) == 64 for l in lines)
    assert not any(l.endswith(".tmp") for l in lines)


def test_spend_report_per_workspace_and_projection():
    h = _helpers()
    rate = h["usd_per_container_hour"]()
    assert abs(rate - 1.3238) < 1e-3                                  # A10 + 2 cores + 16 GiB, modal.com/pricing 2026-11-10
    metas = [
        {"task": 0, "status": "complete", "workspace_profile": "friend", "gpu_name": "NVIDIA A10",
         "episodes_complete": 50, "episodes_invalid": [], "attempts": [{"workspace_profile": "friend", "seconds": 3420}]},
        {"task": 1, "status": "complete", "workspace_profile": "akashrav", "gpu_name": "NVIDIA A10",
         "episodes_complete": 49, "episodes_invalid": [7], "attempts": [{"workspace_profile": "akashrav", "seconds": 3420}]},
        {"task": 2, "status": "not started"},
    ]
    rep = h["spend_report"](metas)
    assert "workspace akashrav: ~$1.32" in rep and "workspace friend: ~$1.32" in rep   # (3420 + 180) s each
    assert "Projected for the 1 unfinished task(s) [2]: ~1.0 h ~ $1.32" in rep


def test_image_fingerprint_is_a_content_hash(tmp_path):
    fp = _helpers()["image_fingerprint"]
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        (d / "pip_freeze.txt").write_text("torch==2.2.0+cu121\n")
        (d / "dpkg_packages.txt").write_text("libegl1=1.4.0\n")
    (a / "image_fingerprint.txt").write_text("ignored\n")             # its own file is not hashed
    assert fp(a) == fp(b) and len(fp(a)) == 64
    (b / "dpkg_packages.txt").write_text("libegl1=1.4.1\n")            # an OS package drift changes the image
    assert fp(a) != fp(b)


def test_full_run_requires_a_smoke_validated_image():
    src = _src()
    assert 'Path(PROVENANCE_DIR, FINGERPRINT_FILE).write_text(image_fingerprint(Path(PROVENANCE_DIR))' in src
    assert '"dpkg_packages.txt"' in src and '"python_version.txt"' in src
    # only a FULL smoke pass validates an image, recorded per workspace volume
    assert "if full:" in src and 'Path(WORK_DIR, "smoke_passes", f"{fp}.json")' in src
    # rollout_task refuses an unvalidated image and records the image in TASK_META
    body = src[src.index("def rollout_task("):src.index("def task_status(")]
    assert "if validation is None:" in body and "raise RuntimeError" in body
    assert '"image_fingerprint": fp, "modal_image_id": image_id' in body
    assert "a task runs on one image" in body
    # the launcher checks on CPU before any GPU starts
    launch = src[src.index("def rollouts("):src.index("def spend_report(")]
    assert launch.index("image_check.remote()") < launch.index("rollout_task.map(")


def test_smoke_validated_image_is_committed():
    import json as _json
    rec = _json.loads((MODAL_APP.parent / "smoke_validated_images.json").read_text())
    (fp, r), = rec.items()
    assert len(fp) == 64 and r["full"] is True and r["gpu_name"] == "NVIDIA A10"
    assert r["modal_image_id"].startswith("im-") and len(r["kit_commit"]) == 40
