"""collection/carc/pull_from_modal.sh against a fake Modal CLI and a fake Volume (CPU only, no network).

Checks the CARC side of the workspace-portable run: a task is moved into /scratch1/.../taskNN only if its checksums
match, its TASK_META says complete on an A10 in the pulled profile, and every episode has its files or an invalid
marker. Nothing is ever deleted on the (fake) Volume.
"""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PULL = ROOT / "collection" / "carc" / "pull_from_modal.sh"
RUN = "all-layers-v1"
EPS = 3
pytestmark = pytest.mark.skipif(shutil.which("bash") is None or shutil.which("sha256sum") is None,
                                reason="needs bash and sha256sum")

FAKE_MODAL = """#!/bin/bash
# fake `modal volume get [--profile P] [--force] VOLUME REMOTE DEST`: copies from $FAKE_VOLUME_ROOT/<profile>/
set -euo pipefail
[ "$1 $2" = "volume get" ] || { echo "fake modal: unsupported $*" >&2; exit 2; }
shift 2; profile=""; args=()
while [ $# -gt 0 ]; do case "$1" in --profile) profile="$2"; shift 2;; --force) shift;; *) args+=("$1"); shift;; esac; done
src="$FAKE_VOLUME_ROOT/$profile/${args[1]}"; dst="${args[2]}"
[ -e "$src" ] || { echo "not found: ${args[1]}" >&2; exit 1; }
rm -rf "$dst"; mkdir -p "$(dirname "$dst")"; cp -R "$src" "$dst"
"""


def make_task(vol: Path, profile: str, t: int, status="complete", gpu="NVIDIA A10", invalid=(), drop=(), meta_profile=None):
    tdir = vol / profile / "rollouts" / RUN / f"task{t:02d}"
    ep = tdir / "libero_10"
    (ep / "provenance").mkdir(parents=True)
    (ep / "provenance" / f"provenance--tasks{t}-{t}--joblocal.json").write_text("{}")
    for e in range(EPS):
        if e in invalid:
            (ep / f"task{t}--ep{e}--invalid.json").write_text(json.dumps({"task_id": t, "episode_idx": e}))
            continue
        for ext in ("csv", "pkl", "layers.pt", "mp4"):
            if (e, ext) not in drop:
                (ep / f"task{t}--ep{e}--succ1.{ext}").write_bytes(f"{t}-{e}-{ext}".encode())
    lines = []
    for f in sorted(p for p in ep.rglob("*") if p.is_file()):
        lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(tdir)}")
    (tdir / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    (tdir / "TASK_META.json").write_text(json.dumps({"task": t, "status": status, "gpu_name": gpu,
                                                     "workspace_profile": meta_profile or profile}))
    work = vol / profile / "work"
    (work / "image_provenance").mkdir(parents=True, exist_ok=True)
    (work / "image_provenance" / "pip_freeze.txt").write_text("torch==2.2.0+cu121\n")
    (work / "checkpoint_path.txt").write_text("/vol/hf/snapshots/abc\n")
    return tdir


@pytest.fixture()
def env(tmp_path):
    fake = tmp_path / "bin" / "modal"
    fake.parent.mkdir()
    fake.write_text(FAKE_MODAL)
    fake.chmod(0o755)
    return dict(os.environ, MODAL_BIN=str(fake), FAKE_VOLUME_ROOT=str(tmp_path / "vol"),
                PULL_ROOT=str(tmp_path / "scratch"), EPISODES_PER_TASK=str(EPS), ALLOW_OUTSIDE_SLURM="1")


def pull(env, *args):
    return subprocess.run(["bash", str(PULL), *args], env=env, capture_output=True, text=True)


def final(env, t):
    return Path(env["PULL_ROOT"]) / RUN / f"task{t:02d}"


def test_complete_tasks_are_verified_and_moved_into_place(env, tmp_path):
    for t in (0, 1):
        make_task(tmp_path / "vol", "friend", t, invalid=(2,) if t == 1 else ())
    r = pull(env, "--profile", "friend", "--tasks", "0-1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (final(env, 0) / "TASK_META.json").exists() and (final(env, 1) / "libero_10" / "task1--ep2--invalid.json").exists()
    assert (Path(env["PULL_ROOT"]) / RUN / "image_provenance" / "friend" / "pip_freeze.txt").exists()
    assert "\tok" in (Path(env["PULL_ROOT"]) / RUN / "pull_log.tsv").read_text()
    assert (tmp_path / "vol" / "friend" / "rollouts" / RUN / "task00").exists()  # nothing deleted on the volume


@pytest.mark.parametrize("kw, needle", [
    (dict(status="running"), "status is 'running'"),          # killed by a timeout or a credit limit
    (dict(gpu="NVIDIA L4"), "gpu_name"),
    (dict(drop={(1, "layers.pt")}), "ep1: incomplete"),
    (dict(meta_profile="akashrav"), "written by profile"),
])
def test_unusable_tasks_are_refused_and_not_moved(env, tmp_path, kw, needle):
    make_task(tmp_path / "vol", "friend", 3, **kw)
    r = pull(env, "--profile", "friend", "--tasks", "3")
    assert r.returncode != 0 and needle in r.stdout + r.stderr, r.stdout + r.stderr
    assert not final(env, 3).exists()


def test_checksum_mismatch_is_refused(env, tmp_path):
    tdir = make_task(tmp_path / "vol", "friend", 4)
    (tdir / "libero_10" / "task4--ep0--succ1.pkl").write_bytes(b"corrupted in transfer")
    r = pull(env, "--profile", "friend", "--tasks", "4")
    assert r.returncode != 0 and "CHECKSUM MISMATCH" in r.stdout + r.stderr
    assert not final(env, 4).exists()


def test_refuses_outside_a_slurm_job(env, tmp_path):
    make_task(tmp_path / "vol", "friend", 0)
    e = {k: v for k, v in env.items() if k not in ("ALLOW_OUTSIDE_SLURM", "SLURM_JOB_ID")}
    r = pull(e, "--profile", "friend", "--tasks", "0")
    assert r.returncode != 0 and "never on a login node" in r.stderr
