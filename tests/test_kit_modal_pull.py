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
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PULL = ROOT / "collection" / "carc" / "pull_from_modal.sh"
RUN = "all-layers-v1"
EPS = 3
pytestmark = pytest.mark.skipif(shutil.which("bash") is None or shutil.which("sha256sum") is None,
                                reason="needs bash and sha256sum")

FAKE_MODAL = r"""#!{python}
# Fake `modal volume get [--profile P] [--force] VOLUME REMOTE DEST`, a port of modal 1.6's cli/_download.py:
#   entries = REMOTE itself if it is a file, else every file/dir below it (recursive); rel = path relative to
#   REMOTE's parent (so it starts with REMOTE's basename); output = DEST/rel if DEST is an existing directory, else
#   DEST itself for EVERY entry; an existing output is an error without --force (with --force it is replaced).
import os, shutil, sys
from pathlib import Path, PurePosixPath
a = sys.argv[1:]
assert a[:2] == ["volume", "get"], a
a = a[2:]; profile = None; force = False; pos = []
while a:
    x = a.pop(0)
    if x == "--profile": profile = a.pop(0)
    elif x == "--force": force = True
    else: pos.append(x)
vol, remote, dest = pos[0], pos[1].rstrip("/"), Path(pos[2] if len(pos) > 2 else ".")
root = Path(os.environ["FAKE_VOLUME_ROOT"]) / profile
src = root / remote
if not src.exists():
    sys.exit(f"No such file or directory: {remote}")
entries = [src] if src.is_file() else sorted(src.rglob("*"))
start = PurePosixPath(remote).parent
for e in entries:
    rel = PurePosixPath(e.relative_to(root).as_posix()).relative_to(start)
    out = dest / rel if dest.is_dir() else dest
    if out.exists():
        if not force:
            sys.exit(f"UsageError: Output path '{out}' already exists. Use --force to overwrite the output directory")
        os.remove(out) if out.is_file() else shutil.rmtree(out)
    if e.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)   # raises FileExistsError if a parent is a file
        shutil.copyfile(e, out)
    else:
        out.mkdir(parents=True, exist_ok=True)
"""


def make_task(vol: Path, profile: str, t: int, status="complete", gpu="NVIDIA A10", invalid=(), drop=(), meta_profile=None,
              attempts=({"first_episode": 0, "engineering_resume": False},)):
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
                                                     "workspace_profile": meta_profile or profile,
                                                     "attempts": list(attempts)}))
    work = vol / profile / "work"
    (work / "image_provenance").mkdir(parents=True, exist_ok=True)
    (work / "image_provenance" / "pip_freeze.txt").write_text("torch==2.2.0+cu121\n")
    (work / "image_provenance" / "image_fingerprint.txt").write_text("0f65545c" + "0" * 56 + "\n")
    (work / "checkpoint_path.txt").write_text("/vol/hf/snapshots/abc\n")
    return tdir


@pytest.fixture()
def env(tmp_path):
    fake = tmp_path / "bin" / "modal"
    fake.parent.mkdir()
    fake.write_text(FAKE_MODAL.replace("{python}", sys.executable))
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
    # dataset policy 2026-11-11: one uninterrupted attempt from episode 0, never resumed
    (dict(attempts=({"first_episode": 0}, {"first_episode": 17})), "not one uninterrupted attempt"),
    (dict(attempts=({"first_episode": 17},)), "not one uninterrupted attempt"),
    (dict(attempts=({"first_episode": 0, "engineering_resume": True},)), "not one uninterrupted attempt"),
    (dict(status="engineering-complete"), "status is 'engineering-complete'"),
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


# ---------------------------------------------------------------- real `modal volume get` semantics (CARC job 14000077)
def fake_get(env, *args):
    return subprocess.run([env["MODAL_BIN"], "volume", "get", "--profile", "friend", *args], env=env,
                          capture_output=True, text=True)


def test_fake_cli_matches_real_modal_semantics(env, tmp_path):
    make_task(tmp_path / "vol", "friend", 0)
    d = tmp_path / "dl"
    # a folder into a NON-existing destination collapses into one file (each entry overwrites it)
    assert fake_get(env, "--force", "cs566-collection", "work/image_provenance", str(d / "a")).returncode == 0
    assert (d / "a").is_file()
    # ... and without --force the second entry is an error
    (tmp_path / "vol" / "friend" / "work" / "image_provenance" / "dpkg_packages.txt").write_text("x\n")
    r = fake_get(env, "cs566-collection", "work/image_provenance", str(d / "b"))
    assert r.returncode != 0 and "already exists" in r.stdout + r.stderr
    # a folder into an EXISTING directory nests under the folder's own name
    (d / "c").mkdir(parents=True)
    assert fake_get(env, "cs566-collection", "work/image_provenance", str(d / "c")).returncode == 0
    assert (d / "c" / "image_provenance" / "pip_freeze.txt").is_file()
    # a single file into an existing directory keeps its name
    assert fake_get(env, "cs566-collection", "work/checkpoint_path.txt", str(d / "c")).returncode == 0
    assert (d / "c" / "checkpoint_path.txt").is_file()


def test_reproduces_carc_job_14000077_with_the_old_command_sequence(env, tmp_path):
    make_task(tmp_path / "vol", "friend", 0)
    prov = tmp_path / "scratch" / RUN / "image_provenance" / "friend"
    old = f"""set -e
mkdir -p "{prov}"
"$MODAL_BIN" volume get --profile friend --force cs566-collection work/image_provenance "{prov}.tmp"
rm -rf "{prov}" && mv "{prov}.tmp" "{prov}"
"$MODAL_BIN" volume get --profile friend --force cs566-collection work/checkpoint_path.txt "{prov}/checkpoint_path.txt"
"""
    r = subprocess.run(["bash", "-c", old], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and ("File exists" in r.stderr or "Not a directory" in r.stderr), r.stderr
    assert prov.is_file()                                       # the state found on CARC


def test_rerun_from_the_failed_carc_state_moves_the_bad_file_aside(env, tmp_path):
    for t in (0, 1, 3, 4):
        make_task(tmp_path / "vol", "friend", t)
    run = Path(env["PULL_ROOT"]) / RUN
    (run / "image_provenance").mkdir(parents=True)
    (run / "image_provenance" / "friend").write_text("x" * 8248)          # the 8,248-byte FILE left on CARC
    (run / "pull_log.tsv").write_text("utc\tprofile\ttask\tresult\n")  # header only
    r = pull(env, "--profile", "friend", "--tasks", "0,1,3,4")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (run / "image_provenance" / "friend").is_dir()
    assert (run / "image_provenance" / "friend" / "checkpoint_path.txt").is_file()
    assert (run / "image_provenance" / "friend" / "image_fingerprint.txt").is_file()
    moved = list((run / "_discarded").glob("*/image_provenance-friend"))
    assert len(moved) == 1 and moved[0].is_file() and moved[0].stat().st_size == 8248   # kept, not deleted
    assert all((run / f"task{t:02d}" / "TASK_META.json").exists() for t in (0, 1, 3, 4))
    assert not list(run.glob(".incoming-*"))
    # rerun: everything already verified is skipped, nothing moved
    r = pull(env, "--profile", "friend", "--tasks", "0,1,3,4")
    assert r.returncode == 0 and r.stdout.count("already pulled and verified; skipped") == 4, r.stdout + r.stderr
    assert "unchanged (already in place)" in r.stdout
    assert len(list((run / "_discarded").glob("*/*"))) == 1


def test_a_corrupted_or_partial_local_task_is_moved_aside_and_repulled(env, tmp_path):
    make_task(tmp_path / "vol", "friend", 5)
    assert pull(env, "--profile", "friend", "--tasks", "5").returncode == 0
    run = Path(env["PULL_ROOT"]) / RUN
    (run / "task05" / "libero_10" / "task5--ep0--succ1.pkl").write_bytes(b"damaged on disk")
    (run / ".incoming-task07").mkdir()                                     # leftover of an interrupted pull
    r = pull(env, "--profile", "friend", "--tasks", "5")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "CHECKSUM MISMATCH" in r.stdout and (run / "task05" / "TASK_META.json").exists()
    assert len(list((run / "_discarded").glob("*/task05"))) == 1
    assert len(list((run / "_discarded").glob("*/.incoming-task07"))) == 1


def test_comma_task_list_through_the_slurm_job(env, tmp_path):
    for t in (0, 1, 3, 4):
        make_task(tmp_path / "vol", "friend", t)
    e = dict(env, PROFILE="friend", TASKS="0,1,3,4", SLURM_JOB_ID="1")     # exported, then --export=ALL
    r = subprocess.run(["bash", "carc/04_pull_from_modal.slurm"], cwd=ROOT / "collection", env=e,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(p.name for p in (Path(env["PULL_ROOT"]) / RUN).glob("task0*")) == ["task00", "task01", "task03", "task04"]
