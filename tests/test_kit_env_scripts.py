"""Tests for the CARC environment scripts: collection/carc/{config.sh, activate_env.sh, 00_setup_env.sh}.

CPU only, no network, nothing is installed. The real scripts are executed: the config guard, activate_env.sh,
and dry runs of 00_setup_env.sh in a sandbox with stand-ins for `uv`, `conda`, the package index and the
Hugging Face hub (small local git repos play OpenVLA and LIBERO). This exercises the control flow under
`set -euo pipefail`; it cannot prove that the real installs work on CARC. Versions are read from
collection/constraints.txt, never repeated here. Static checks of the pins: test_kit_env_pins.py.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "collection"
CARC = KIT / "carc"
SHELL_FILES = sorted({*KIT.rglob("*.sh"), *KIT.rglob("*.slurm")})
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(
    BASH is None or shutil.which("git") is None or os.name == "nt", reason="needs bash and git (POSIX)"
)

OPENVLA_COMMIT = "300dce26d44f407c725695d16cd445755c92cbd1"   # the commit openvla_all_layers.patch is made for


def verified_versions():
    """The versions the verified CARC build had installed, read from constraints.txt: the exact pin, or the
    version named by "verified X" in the line's comment (ranges, and the +cu121 PyTorch wheels)."""
    out = {}
    for raw in (KIT / "constraints.txt").read_text().splitlines():
        code, _, comment = raw.partition("#")
        code = code.strip()
        if not code:
            continue
        name = re.match(r"[A-Za-z0-9_.-]+", code).group(0)
        verified = re.search(r"\bverified (\S+?)[;,)]?(?:\s|$)", comment)
        exact = re.fullmatch(r"[A-Za-z0-9_.-]+==(\S+)", code)
        assert verified or exact, f"no installable version for {code!r}"
        out[name] = verified.group(1) if verified else exact.group(1)
    return out


PINS = verified_versions()


def logical_lines(path):
    """Non-comment lines of a shell file, with backslash continuations joined."""
    out, buf = [], ""
    for raw in Path(path).read_text().splitlines():
        if raw.rstrip().endswith("\\"):
            buf += raw.rstrip()[:-1] + " "
            continue
        line, buf = buf + raw, ""
        if line.strip() and not line.strip().startswith("#"):
            out.append(line)
    return out


def run(cmd, env=None, cwd=None):
    return subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True, timeout=300)


def base_env(tmp_path, extra_path=()):
    """A clean environment: nothing leaks in from the developer's shell except PATH."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    path = os.pathsep.join([*map(str, extra_path), os.environ.get("PATH", "/usr/bin:/bin")])
    return {"PATH": path, "HOME": str(home), "USER": "tester", "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1", "LC_ALL": "C"}


def write_config(dst, **overrides):
    """The REAL config.sh with some `export NAME=...` lines replaced (so the real guard code is what runs)."""
    text = (CARC / "config.sh").read_text()
    for name, value in overrides.items():
        text, n = re.subn(rf"(?m)^export {name}=.*$", lambda m: f'export {name}="{value}"', text)
        assert n == 1, f"config.sh should define {name} exactly once with `export {name}=`"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text)
    return dst


# ---------------------------------------------------------------------------------------------- static checks
@pytest.mark.parametrize("path", SHELL_FILES, ids=lambda p: str(p.relative_to(KIT)))
def test_bash_syntax(path):
    r = run([BASH, "-n", str(path)])
    assert r.returncode == 0, r.stderr


def test_config_pins_and_paths():
    lines = logical_lines(CARC / "config.sh")
    exported = {m.group(1): m.group(2) for ln in lines if (m := re.match(r'export (\w+)=(.*)', ln))}
    for name in ("ENV_MODE", "VENV_DIR", "CONDA_ENV", "CUDA_HOME", "PYTHON_BIN", "LIBERO_COMMIT", "OPENVLA_COMMIT",
                 "LIBERO_CONFIG_PATH", "HF_HOME", "MUJOCO_GL", "PYOPENGL_PLATFORM"):
        assert name in exported, f"config.sh must export {name}"
    assert exported["ENV_MODE"].startswith('"uv"')                       # the verified mode is the default
    assert OPENVLA_COMMIT in exported["OPENVLA_COMMIT"]
    assert re.search(r'"[0-9a-f]{40}"', exported["LIBERO_COMMIT"])
    assert exported["CACHE_ROOT"].startswith('"/project2/')
    assert not any("/project/" in ln for ln in lines), "CARC project space is /project2/<account>"
    assert "CHANGE_ME" in exported["CARC_ACCOUNT"]                       # account-specific: stays a placeholder
    setup = (CARC / "00_setup_env.sh").read_text()
    assert '"$OPENVLA_COMMIT"' in setup and '"$LIBERO_COMMIT"' in setup


# ---------------------------------------------------------------------------------------------- CHANGE_ME guard
def test_guard_rejects_placeholders_and_accepts_a_filled_config(tmp_path):
    env = base_env(tmp_path)
    r = run([BASH, "-euo", "pipefail", "-c", f'source "{CARC / "config.sh"}"; kit_check_config'], env=env)
    assert r.returncode != 0
    assert "CARC_ACCOUNT" in r.stderr and "CACHE_ROOT" in r.stderr and "CUDA_HOME" in r.stderr and "CHANGE_ME" in r.stderr

    good = write_config(tmp_path / "kit/carc/config.sh", CARC_ACCOUNT="acct_123", CUDA_HOME="/opt/cuda-12.4")
    r = run([BASH, "-euo", "pipefail", "-c", f'source "{good}"; cd /; kit_check_config; echo "$CACHE_ROOT"'], env=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "/project2/acct_123/safe_cache"


@pytest.mark.parametrize("name,value,needle", [
    ("CUDA_HOME", "/apps/CHANGE_ME/cuda", "CUDA_HOME"),
    ("RUN_NOTE", "CHANGE_ME", "RUN_NOTE"),
    ("ENV_MODE", "pip", "ENV_MODE"),
    ("HF_HOME", "", "HF_HOME"),
    ("LIBERO_COMMIT", "8f1084e", "LIBERO_COMMIT"),
])
def test_guard_checks_every_setting(tmp_path, name, value, needle):
    cfg = write_config(tmp_path / "kit/carc/config.sh", CARC_ACCOUNT="acct_123", **{name: value})
    r = run([BASH, "-euo", "pipefail", "-c", f'source "{cfg}"; kit_check_config'], env=base_env(tmp_path))
    assert r.returncode != 0 and needle in r.stderr, r.stderr


def fake_metadata(site, versions):
    """Fake installed-package metadata for `versions`, plus a copy of the real `packaging` library."""
    import packaging
    write_tree(site, {f"{n.replace('-', '_')}-{v}.dist-info/METADATA": f"Metadata-Version: 2.1\nName: {n}\nVersion: {v}\n"
                      for n, v in versions.items()})
    if not (Path(site) / "packaging").exists():
        shutil.copytree(Path(packaging.__file__).parent, Path(site) / "packaging",
                        ignore=shutil.ignore_patterns("__pycache__"))


def write_tree(root, files):
    for rel, content in files.items():
        path = Path(root) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def make_stub(directory, name, body):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)
    return path


def test_scripts_refuse_to_run_with_the_shipped_placeholders(tmp_path):
    """The real scripts with the real (unedited) config must stop before touching anything."""
    stubs, log = tmp_path / "stubs", tmp_path / "calls.log"
    for tool in ("uv", "conda", "git", "pip", "python", "curl"):
        make_stub(stubs, tool, f'echo "{tool} $*" >> "{log}"\nexit 99\n')
    env = base_env(tmp_path, extra_path=[stubs])
    r = run([BASH, str(CARC / "00_setup_env.sh")], env=env, cwd=KIT)
    assert r.returncode != 0 and "CHANGE_ME" in r.stderr and "Nothing was installed" in r.stderr
    r2 = run([BASH, "-euo", "pipefail", "-c", "source carc/config.sh; source carc/activate_env.sh; echo REACHED"],
             env=env, cwd=KIT)
    assert r2.returncode != 0 and "CHANGE_ME" in r2.stderr and "REACHED" not in r2.stdout
    assert not log.exists(), f"tools were called before the guard: {log.read_text()}"


# ---------------------------------------------------------------------------------------------- activate_env.sh
ACTIVATE_PROBE = (
    'source "{cfg}"; source "{act}"; '
    'echo "VENV=${{VIRTUAL_ENV:-}} CONDA=${{CONDA_PREFIX:-}} CUDA=$CUDA_HOME HF=$HF_HOME OFFLINE=$HF_HUB_OFFLINE '
    'GL=$MUJOCO_GL/$PYOPENGL_PLATFORM LIBERO=$LIBERO_CONFIG_PATH"; echo "PATH=$PATH"; '
    '( echo "$THIS_IS_UNSET" ) 2>/dev/null && echo NOUNSET_LOST || echo NOUNSET_KEPT; echo ACTIVATED'
)


def activate(tmp_path, env, **overrides):
    work = tmp_path / "work"
    settings = dict(CARC_ACCOUNT="acct_123", WORK=work, ROLLOUT_ROOT=tmp_path / "rollouts",
                    CACHE_ROOT=tmp_path / "cache", CUDA_HOME=tmp_path / "cuda")
    settings.update(overrides)
    cfg = write_config(tmp_path / "kit/carc/config.sh", **settings)
    probe = ACTIVATE_PROBE.format(cfg=cfg, act=CARC / "activate_env.sh")
    return run([BASH, "-euo", "pipefail", "-c", probe], env=env), work


def test_activate_fails_cleanly_when_the_uv_env_is_missing(tmp_path):
    r, work = activate(tmp_path, base_env(tmp_path), ENV_MODE="uv")
    assert r.returncode != 0
    assert "ERROR" in r.stderr and str(work / "venv310") in r.stderr and "00_setup_env.sh" in r.stderr
    assert "unbound variable" not in r.stderr and "VENV=" not in r.stdout


def test_activate_uv_env(tmp_path):
    venv = tmp_path / "work/venv310"
    (venv / "bin").mkdir(parents=True)
    # like a real activate script; it reads an unset variable on purpose (must not trip the caller's `set -u`)
    (venv / "bin/activate").write_text(
        f'_old="$SOME_UNSET_VARIABLE_FROM_ACTIVATE"\nexport VIRTUAL_ENV="{venv}"\nexport PATH="$VIRTUAL_ENV/bin:$PATH"\n')
    (tmp_path / "work/libero_config").mkdir()
    (tmp_path / "work/libero_config/config.yaml").write_text("benchmark_root: x\n")
    r, work = activate(tmp_path, base_env(tmp_path), ENV_MODE="uv")
    out = r.stdout
    assert f"VENV={venv} " in out and f"CUDA={tmp_path / 'cuda'} " in out and f"HF={work / 'hf'} " in out
    assert "OFFLINE=1 " in out and "GL=egl/egl " in out and f"LIBERO={work / 'libero_config'}" in out
    path = re.search(r"^PATH=(.*)$", out, re.M).group(1).split(":")
    assert path[0] == str(venv / "bin") and str(tmp_path / "cuda/bin") in path
    assert r.returncode == 0 and "ACTIVATED" in out, r.stderr
    assert "NOUNSET_KEPT" in out, "activate_env.sh must leave the caller's `set -u` on"
    assert "WARNING" not in r.stderr


def test_activate_warns_when_libero_config_is_missing(tmp_path):
    venv = tmp_path / "work/venv310"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/activate").write_text(f'export VIRTUAL_ENV="{venv}"\n')
    r, _ = activate(tmp_path, base_env(tmp_path), ENV_MODE="uv")
    assert r.returncode == 0 and f"VENV={venv} " in r.stdout, r.stderr
    assert "WARNING" in r.stderr and "config.yaml" in r.stderr


def test_activate_rejects_an_unknown_env_mode(tmp_path):
    r, _ = activate(tmp_path, base_env(tmp_path), ENV_MODE="system")
    assert r.returncode != 0 and "ENV_MODE" in r.stderr and "ACTIVATED" not in r.stdout


def test_activate_refuses_to_be_executed(tmp_path):
    r = run([BASH, str(CARC / "activate_env.sh")], env=base_env(tmp_path))
    assert r.returncode != 0 and "sourced" in r.stderr


def test_activate_needs_config_first(tmp_path):
    r = run([BASH, "-euo", "pipefail", "-c", f'source "{CARC / "activate_env.sh"}"; echo REACHED'],
            env=base_env(tmp_path))
    assert r.returncode != 0 and "config.sh" in r.stderr and "REACHED" not in r.stdout


CONDA_STUB = r'''echo "conda $*" >> "$STUB_LOG"
case "$1" in
  shell.bash)
    cat <<'EOF'
conda() {
  if [ "$1" = "activate" ]; then
    _unused="$CONDA_STUB_READS_AN_UNSET_VARIABLE"
    export CONDA_PREFIX="$STUB_CONDA_ROOT/envs/$2"
    export PATH="$CONDA_PREFIX/bin:$PATH"
  else
    command conda "$@"
  fi
}
EOF
    ;;
  env)
    echo "# conda environments:"
    echo "base                     $STUB_CONDA_ROOT"
    for d in "$STUB_CONDA_ROOT"/envs/*/; do
      if [ -d "$d" ]; then echo "$(basename "$d")            $d"; fi
    done
    ;;
  create)   # conda create -y -n NAME python=3.10
    [ "$5" = "python=3.10" ] || exit 3
    "$STUB_REAL_PYTHON" -m venv --without-pip "$STUB_CONDA_ROOT/envs/$4"
    site="$("$STUB_CONDA_ROOT/envs/$4/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    cp -R "$STUB_FAKE_SITE"/. "$site"/
    ;;
esac
exit 0
'''


def conda_env(tmp_path, with_conda=True):
    stubs = tmp_path / "stubs"
    stubs.mkdir(exist_ok=True)
    if with_conda:
        make_stub(stubs, "conda", CONDA_STUB)
    env = base_env(tmp_path, extra_path=[stubs])
    if not with_conda:
        env["PATH"] = f"{stubs}:/usr/bin:/bin"
        if any((Path(p) / "conda").exists() for p in ("/usr/bin", "/bin")):
            pytest.skip("a real conda is installed in a system directory")
    env.update(STUB_LOG=str(tmp_path / "calls.log"), STUB_CONDA_ROOT=str(tmp_path / "conda"),
               STUB_REAL_PYTHON=sys.executable, STUB_FAKE_SITE=str(tmp_path / "fake_site"))
    (tmp_path / "conda/envs").mkdir(parents=True, exist_ok=True)
    return env


def test_activate_conda_missing_command_or_env_fails_cleanly(tmp_path):
    r, _ = activate(tmp_path, conda_env(tmp_path, with_conda=False), ENV_MODE="conda")
    assert r.returncode != 0 and "ERROR" in r.stderr and "conda" in r.stderr and "VENV=" not in r.stdout
    r, _ = activate(tmp_path, conda_env(tmp_path), ENV_MODE="conda")
    assert r.returncode != 0 and "safe-openvla" in r.stderr and "00_setup_env.sh" in r.stderr
    assert "unbound variable" not in r.stderr and "VENV=" not in r.stdout


def test_activate_conda_env(tmp_path):
    env = conda_env(tmp_path)
    (tmp_path / "conda/envs/safe-openvla/bin").mkdir(parents=True)
    (tmp_path / "conda/envs/safe-openvla-other/bin").mkdir(parents=True)   # a name that only shares the prefix
    r, _ = activate(tmp_path, env, ENV_MODE="conda")
    assert r.returncode == 0 and f"CONDA={tmp_path / 'conda/envs/safe-openvla'} " in r.stdout, r.stderr
    assert "NOUNSET_KEPT" in r.stdout and "OFFLINE=1 " in r.stdout and "ACTIVATED" in r.stdout


# ---------------------------------------------------------------------------------------------- setup dry runs
LIBERO_INIT = '''import os
assert "LIBERO_CONFIG_PATH" in os.environ, "LIBERO_CONFIG_PATH must be exported (else LIBERO writes ~/.libero)"
libero_config_path = os.environ["LIBERO_CONFIG_PATH"]
config_file = os.path.join(libero_config_path, "config.yaml")


def get_libero_path(key):
    with open(config_file) as f:
        return dict(line.rstrip("\\n").split(": ", 1) for line in f)[key]


if not os.path.exists(libero_config_path):
    os.makedirs(libero_config_path)
if not os.path.exists(config_file):   # same flow as the real libero/libero/__init__.py
    answer = input("Do you want to specify a custom path for the dataset folder? (Y/N): ").lower()
    assert answer == "n", "the setup script must answer N, got %r" % answer
    root = os.path.dirname(os.path.abspath(__file__))
    paths = {"benchmark_root": root, "bddl_files": os.path.join(root, "./bddl_files"),
             "init_states": os.path.join(root, "./init_files"), "datasets": os.path.join(root, "../datasets")}
    with open(config_file, "w") as f:
        f.write("".join("%s: %s\\n" % kv for kv in paths.items()))
'''
LIBERO_BENCHMARK = '''from libero.libero import get_libero_path


class _Libero10:
    n_tasks = 10


def get_benchmark_dict():
    return {"libero_10": _Libero10}
'''
FAKE_HF_HUB = '''import hashlib
import os


def snapshot_download(repo_id, revision=None, ignore_patterns=None, **kwargs):
    assert os.environ.get("HF_HUB_OFFLINE") == "0", "downloads must switch offline mode off"
    main = hashlib.sha1((repo_id + os.environ.get("STUB_HF_MAIN", "")).encode()).hexdigest()
    rev = revision or main
    root = os.path.join(os.environ["HF_HOME"], "hub", "models--" + repo_id.replace("/", "--"))
    snap = os.path.join(root, "snapshots", rev)
    os.makedirs(snap, exist_ok=True)
    if revision is None:
        os.makedirs(os.path.join(root, "refs"), exist_ok=True)
        with open(os.path.join(root, "refs", "main"), "w") as f:
            f.write(rev)
    with open(os.path.join(snap, "config.json"), "w") as f:
        f.write("{}")
    with open(os.environ["STUB_LOG"], "a") as f:
        f.write("hf %s revision=%s ignore=%s\\n" % (repo_id, revision, ignore_patterns))
    print("progress noise on stdout")
    return snap
'''
FAKE_TRANSFORMERS = '''import os


class AutoProcessor:
    @classmethod
    def from_pretrained(cls, path, trust_remote_code=False):
        assert os.environ.get("HF_HUB_OFFLINE") == "1", "the processor check must run offline"
        assert trust_remote_code and os.path.isfile(os.path.join(path, "config.json")), path
        base = os.path.join(os.environ["HF_HOME"], "hub", "models--openvla--openvla-7b", "refs", "main")
        assert os.path.isfile(base), "base repo code files are not in the cache"
        return cls()
'''
FAKE_PIP = '''import os
import sys

args = sys.argv[1:]
with open(os.environ["STUB_LOG"], "a") as f:
    f.write("pip " + " ".join(args) + "\\n")
if args[0] == "install" and any("flash-attn" in a for a in args):
    sys.exit(int(os.environ.get("STUB_FLASH_RC", "0")))
if args[0] == "freeze":
    print("fake-package==1.0")
'''
UV_STUB = r'''echo "uv $*" >> "$STUB_LOG"
case "$1" in
  venv)
    dir="${@: -1}"
    "$STUB_REAL_PYTHON" -m venv --without-pip "$dir"
    site="$("$dir/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    cp -R "$STUB_FAKE_SITE"/. "$site"/
    ;;
  pip)
    case "$2" in
      install) case "$*" in *flash-attn*) exit "${STUB_FLASH_RC:-0}" ;; esac ;;
      freeze) echo "fake-package==1.0" ;;
    esac
    ;;
esac
exit 0
'''


def git(cwd, *args):
    r = run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", *args],
            cwd=cwd, env={"PATH": os.environ["PATH"], "HOME": str(cwd), "GIT_CONFIG_GLOBAL": os.devnull,
                          "GIT_CONFIG_NOSYSTEM": "1"})
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


class Sandbox:
    """A copy of the kit's scripts plus stand-ins for everything they would download or install."""

    def __init__(self, tmp, env_mode="uv", versions=None, drop=(), **config):
        self.tmp, self.work, self.log = tmp, tmp / "work", tmp / "calls.log"
        self.kit = tmp / "kit"
        (self.kit / "carc").mkdir(parents=True)
        for name in ("00_setup_env.sh", "activate_env.sh"):
            shutil.copy(CARC / name, self.kit / "carc" / name)
        shutil.copy(KIT / "constraints.txt", self.kit / "constraints.txt")

        # "OpenVLA": pinned commit A, then one more commit so that the clone's HEAD is NOT the pin.
        ov = tmp / "src/openvla"
        write_tree(ov, {"experiments/robot/libero/run_libero_eval.py": "print('eval')\n",
                        "experiments/robot/libero/libero_requirements.txt": "robosuite\n"})
        git(ov, "init", "-q")
        git(ov, "add", "-A")
        git(ov, "commit", "-q", "-m", "pinned")
        self.openvla_commit = git(ov, "rev-parse", "HEAD")
        write_tree(ov, {"experiments/robot/libero/run_libero_eval.py": "print('eval')\nsave_all_layers = True\n",
                        "experiments/robot/libero/layer_cache.py": "LAYERS = 33\n"})
        git(ov, "add", "-A")
        (self.kit / "openvla_all_layers.patch").write_text(git(ov, "diff", "--cached") + "\n")
        git(ov, "reset", "-q", "--hard", self.openvla_commit)
        write_tree(ov, {"README.md": "newer upstream commit\n"})
        git(ov, "add", "-A")
        git(ov, "commit", "-q", "-m", "after the pin")

        lib = tmp / "src/LIBERO"
        write_tree(lib, {"libero/libero/__init__.py": LIBERO_INIT, "libero/libero/benchmark/__init__.py": LIBERO_BENCHMARK,
                         "libero/libero/bddl_files/.keep": "", "libero/libero/init_files/.keep": "", "setup.py": ""})
        git(lib, "init", "-q")
        git(lib, "add", "-A")
        git(lib, "commit", "-q", "-m", "libero")
        self.libero_commit = git(lib, "rev-parse", "HEAD")

        site = tmp / "fake_site"
        files = {"flash_attn/__init__.py": "", "flash_attn_2_cuda.py": "",
                 "torch/__init__.py": "class cuda:\n    is_available = staticmethod(lambda: False)\n",
                 "huggingface_hub/__init__.py": FAKE_HF_HUB, "transformers/__init__.py": FAKE_TRANSFORMERS,
                 "pip/__init__.py": "", "pip/__main__.py": FAKE_PIP}
        write_tree(site, {k: v for k, v in files.items() if k not in drop})
        fake_metadata(site, {**PINS, **(versions or {})})

        cuda = tmp / "cuda"
        make_stub(cuda / "bin", "nvcc", "exit 0\n")
        stubs = tmp / "stubs"
        make_stub(stubs, "uv", UV_STUB)
        make_stub(stubs, "conda", CONDA_STUB)
        (tmp / "conda/envs").mkdir(parents=True)
        self.cuda = cuda
        settings = dict(CARC_ACCOUNT="acct_123", WORK=self.work, ROLLOUT_ROOT=tmp / "rollouts",
                        CACHE_ROOT=tmp / "cache", ENV_MODE=env_mode, CUDA_HOME=cuda, OPENVLA_REPO=ov,
                        OPENVLA_COMMIT=self.openvla_commit, LIBERO_REPO=lib, LIBERO_COMMIT=self.libero_commit)
        write_config(self.kit / "carc/config.sh", **{**settings, **config})
        self.env = base_env(tmp, extra_path=[stubs])
        self.env.update(STUB_LOG=str(self.log), STUB_REAL_PYTHON=sys.executable, STUB_FAKE_SITE=str(site),
                        STUB_CONDA_ROOT=str(tmp / "conda"))
        self.python = (self.work / "venv310/bin/python" if env_mode == "uv"
                       else tmp / "conda/envs/safe-openvla/bin/python")

    def setup(self, *args, **env):
        return run([BASH, "carc/00_setup_env.sh", *args], env={**self.env, **env}, cwd=self.kit)

    def calls(self, prefix):
        return [ln for ln in self.log.read_text().splitlines() if ln.startswith(prefix)] if self.log.exists() else []

    def records(self):
        return {p.name: p.read_text() for p in sorted(self.work.glob("*.txt"))}


def check_install_calls(sb, installs):
    assert len(installs) == 7, installs
    for ln in installs:
        assert f"-c {sb.kit / 'constraints.txt'}" in ln, f"install without constraints: {ln}"
    assert "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121" in installs[0]
    assert f"-e {sb.work / 'openvla'}" in installs[1] and f"-e {sb.work / 'LIBERO'}" in installs[3]
    assert "libero_requirements.txt" in installs[4]
    assert "flash-attn --no-build-isolation" in installs[6]   # last, after the editable installs
    assert not [ln for ln in installs if "==" in ln], "versions come from -c constraints.txt only"


def check_setup_results(sb):
    rec = sb.records()
    assert set(rec) == {"pip_freeze.txt", "openvla_commit.txt", "libero_commit.txt", "checkpoint_path.txt",
                        "checkpoint_commit.txt"}
    assert rec["openvla_commit.txt"].strip() == sb.openvla_commit
    assert rec["libero_commit.txt"].strip() == sb.libero_commit
    assert "fake-package==1.0" in rec["pip_freeze.txt"]
    snap = Path(rec["checkpoint_path.txt"].strip())
    assert (snap / "config.json").is_file() and "openvla-7b-finetuned-libero-10" in str(snap)
    pins = [ln.split() for ln in rec["checkpoint_commit.txt"].splitlines()]
    assert [p[0] for p in pins] == ["openvla/openvla-7b-finetuned-libero-10", "openvla/openvla-7b"]
    assert pins[0][1] == snap.name and all(re.fullmatch(r"[0-9a-f]{40}", p[1]) for p in pins)
    # sources: exactly the pinned commit (not the newer HEAD), with the patch applied once
    ov = sb.work / "openvla"
    assert git(ov, "rev-parse", "HEAD") == sb.openvla_commit and not (ov / "README.md").exists()
    assert (ov / "experiments/robot/libero/run_libero_eval.py").read_text().count("save_all_layers") == 1
    assert (ov / "experiments/robot/libero/layer_cache.py").is_file()
    # LIBERO: .pth file + pre-answered config => imports with no terminal attached
    site = Path(run([str(sb.python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"]).stdout.strip())
    assert (site / "libero_repo.pth").read_text().strip() == str(sb.work / "LIBERO")
    assert (sb.work / "libero_config/config.yaml").is_file()
    r = subprocess.run([str(sb.python), "-c", "from libero.libero import benchmark"], stdin=subprocess.DEVNULL,
                       env={**sb.env, "LIBERO_CONFIG_PATH": str(sb.work / "libero_config")},
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    # model files: full checkpoint, and the base repo WITHOUT weights
    hf = sb.calls("hf ")
    assert hf[0].startswith("hf openvla/openvla-7b-finetuned-libero-10 revision=None ignore=None")
    assert hf[1].startswith("hf openvla/openvla-7b revision=None") and "*.safetensors" in hf[1]


def test_setup_dry_run_uv_is_complete_and_idempotent(tmp_path):
    sb = Sandbox(tmp_path, env_mode="uv")
    r = sb.setup()
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "Setup done" in r.stdout and "LIBERO ok: 10 tasks" in r.stdout and "Environment verified" in r.stdout
    assert sb.calls("uv venv") == [f"uv venv --python 3.10 {sb.work / 'venv310'}"]
    check_install_calls(sb, sb.calls("uv pip install"))
    assert sb.calls("uv pip check") and sb.calls("uv pip freeze") and not sb.calls("conda")
    check_setup_results(sb)
    first = sb.records()

    # Rerun, with the hub's `main` now pointing somewhere else: nothing is rebuilt and no pin moves.
    r = sb.setup(STUB_HF_MAIN="moved")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "already applied" in r.stdout and len(sb.calls("uv venv")) == 1
    assert sb.records() == first
    check_setup_results(sb)
    assert all("revision=None" not in ln for ln in sb.calls("hf ")[2:]) and len(sb.calls("hf ")) == 4

    # --verify: checks only, installs nothing
    before = len(sb.calls("uv pip install"))
    r = sb.setup("--verify")
    assert r.returncode == 0 and "Environment verified" in r.stdout, r.stderr
    assert len(sb.calls("uv pip install")) == before
    assert sb.setup("--bogus").returncode != 0


def test_setup_dry_run_uv_uses_the_explicit_python_binary(tmp_path):
    sb = Sandbox(tmp_path, env_mode="uv", PYTHON_BIN=sys.executable)
    r = sb.setup()
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert sb.calls("uv venv") == [f"uv venv --python {sys.executable} {sb.work / 'venv310'}"]


def test_setup_dry_run_conda(tmp_path):
    sb = Sandbox(tmp_path, env_mode="conda")
    r = sb.setup()
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "conda create -y -n safe-openvla python=3.10" in sb.calls("conda")
    check_install_calls(sb, sb.calls("pip install"))
    assert sb.calls("pip check") and sb.calls("pip freeze") and not sb.calls("uv")
    check_setup_results(sb)
    r = sb.setup()
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert len([c for c in sb.calls("conda") if c.startswith("conda create")]) == 1


def assert_flash_attn_hard_failure(sb, r):
    assert r.returncode != 0
    assert f"flash-attn {PINS['flash-attn']} is not usable" in r.stderr and re.search(r"Do NOT switch to \w+", r.stderr)
    assert "flash_attention_2" in r.stderr and "Setup done" not in r.stdout
    assert "pip_freeze.txt" not in sb.records(), "a failed setup must not leave a 'finished' record"


def test_setup_fails_hard_when_flash_attn_does_not_install(tmp_path):
    sb = Sandbox(tmp_path)
    r = sb.setup(STUB_FLASH_RC="1")
    assert_flash_attn_hard_failure(sb, r)
    assert not sb.calls("hf "), "nothing may be downloaded after the failure"


def test_setup_fails_hard_when_flash_attn_cuda_extension_does_not_import(tmp_path):
    sb = Sandbox(tmp_path, drop=("flash_attn_2_cuda.py",))
    r = sb.setup()
    assert_flash_attn_hard_failure(sb, r)
    assert "flash_attn_2_cuda" in r.stderr


def run_check_pins(site, cwd):
    """Run the Python block of check_pins() from 00_setup_env.sh against the fake package metadata in `site`."""
    text = (CARC / "00_setup_env.sh").read_text()
    code = text.split("check_pins() {", 1)[1].split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    prog = f"import sys; sys.path.insert(0, {str(site)!r})\n" + code
    return run([sys.executable, "-S", "-c", prog, str(KIT / "constraints.txt")], cwd=cwd)


@pytest.mark.parametrize("name,version", [
    ("numpy", "2.1.0"), ("protobuf", "5.29.0"), ("tensorflow-metadata", "1.21.0"), ("mujoco", "3.17.0"),
    ("torch", "2.2.0"), ("transformers", "4.41.0"), ("flash-attn", "2.6.3"),
])
def test_version_check_names_the_mismatch(tmp_path, name, version):
    """The check_pins block, run on its own against fake package metadata (no install needed)."""
    site = tmp_path / "site"
    fake_metadata(site, {**PINS, name: version})
    r = run_check_pins(site, tmp_path)
    assert r.returncode != 0 and "PIN MISMATCH" in r.stderr, r.stdout + r.stderr
    assert name in r.stderr.split("PIN MISMATCH")[1].split(". ")[0]
    assert re.search(rf"BAD {re.escape(name)}\s", r.stdout) and r.stdout.count("BAD") == 1


def test_version_check_accepts_the_verified_versions(tmp_path):
    site = tmp_path / "site"
    fake_metadata(site, PINS)
    r = run_check_pins(site, tmp_path)
    assert r.returncode == 0 and "BAD" not in r.stdout, r.stdout + r.stderr
    assert r.stdout.count("ok ") == len(PINS)   # one line per constrained package


def test_setup_stops_on_a_pin_mismatch(tmp_path):
    sb = Sandbox(tmp_path, versions={"numpy": "2.1.0"})
    r = sb.setup()
    assert r.returncode != 0 and "PIN MISMATCH: numpy" in r.stderr
    assert "pip_freeze.txt" not in sb.records()


def test_setup_fails_loudly_when_the_pinned_openvla_commit_is_missing(tmp_path):
    sb = Sandbox(tmp_path, OPENVLA_COMMIT="0" * 40)
    r = sb.setup()
    assert r.returncode != 0 and "0" * 40 in r.stderr and "does not exist" in r.stderr
    assert not sb.calls("uv pip install"), "nothing may be installed from an unpinned checkout"
    assert not (sb.work / "openvla/experiments/robot/libero/layer_cache.py").exists(), "patch applied to the wrong commit"


def test_setup_needs_nvcc_before_installing_anything(tmp_path):
    sb = Sandbox(tmp_path)
    (sb.cuda / "bin/nvcc").unlink()
    r = sb.setup()
    assert r.returncode != 0 and "nvcc" in r.stderr and "CUDA_HOME" in r.stderr
    assert not sb.log.exists() and not sb.work.exists()
