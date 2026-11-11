"""Must-fix 4: build_cache.py / check_cache.py accept documented invalid episodes (task{T}--ep{E}--invalid.json).
CPU only; reuses the fake rollouts of tests/test_shared_cache.py."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_shared_cache import KIT, run, write_rollouts  # noqa: E402


def make_invalid(raw: Path, t: int, e: int, exc="mujoco.FatalError", **over):
    """What the patched run_libero_eval.py leaves behind: no succ* files, one marker."""
    for p in raw.glob(f"task{t}--ep{e}--succ*"):
        p.unlink()
    rec = {"format": 1, "kind": "invalid_episode", "task_suite_name": "libero_10", "task_id": t, "episode_idx": e,
           "init_state_index": e, "seed": 7, "step_index": 42, "num_steps_wait": 10, "policy_steps_completed": 32,
           "exception_type": exc, "exception_message": "unstable simulation", "traceback": "Traceback ...",
           "timestamp_utc": "2026-11-10T00:00:00+00:00", "job": {"pid": 1}, "invalid_count_in_job_range": 1,
           "max_invalid_episodes": 3}
    rec.update(over)
    path = raw / f"task{t}--ep{e}--invalid.json"
    path.write_text(json.dumps(rec))
    return path


def build(raw, out, *extra):
    return run(KIT / "build_cache.py", "--rollout-dir", raw, "--cache-root", out, "--expect-episodes", 4, *extra)


def test_invalid_episodes_accepted_and_listed_in_manifest(tmp_path):
    raw = tmp_path / "raw"
    write_rollouts(raw)
    make_invalid(raw, 7, 2)
    make_invalid(raw, 1, 3, exc="builtins.ValueError")
    r = build(raw, tmp_path / "c")
    assert r.returncode == 0, r.stderr + r.stdout
    assert "2 invalid episodes" in r.stdout
    cache = tmp_path / "c" / "v1"
    man = json.loads((cache / "manifest.json").read_text())
    inv = {(d["task_id"], d["episode_idx"]): d for d in man["invalid_episodes"]}
    assert set(inv) == {(1, 3), (7, 2)}
    assert inv[(7, 2)]["exception_type"] == "mujoco.FatalError" and inv[(1, 3)]["exception_type"] == "builtins.ValueError"
    assert inv[(7, 2)]["trajectory_id"] == "libero_10/t07/e02" and inv[(7, 2)]["marker"] == "task7--ep2--invalid.json"
    eps = pd.read_csv(cache / "episodes.csv")
    assert len(eps) == 38 and not ((eps.task_id == 7) & (eps.episode_idx == 2)).any()
    assert man["per_task"]["7"]["episodes"] == 3
    sp = json.loads((cache / "splits.json").read_text())["trajectories"]
    assert "libero_10/t07/e02" not in sum(sp.values(), [])
    assert run(KIT / "check_cache.py", "verify", "--cache", cache).returncode == 0


def test_missing_episode_without_marker_is_refused(tmp_path):
    raw = tmp_path / "raw"
    write_rollouts(raw)
    make_invalid(raw, 7, 2)
    for p in raw.glob("task4--ep1--succ*"):  # vanished without a marker
        p.unlink()
    r = build(raw, tmp_path / "c")
    assert r.returncode != 0 and "neither a .csv nor an invalid marker" in r.stderr and "task4--ep1" in r.stderr
    assert not (tmp_path / "c" / "v1").exists()


def test_marker_next_to_data_is_refused_even_with_allow_incomplete(tmp_path):
    raw = tmp_path / "raw"
    write_rollouts(raw)
    make_invalid(raw, 2, 0)
    (raw / "task2--ep0--succ1.pkl").write_bytes(b"stale")
    r = build(raw, tmp_path / "c", "--allow-incomplete")
    assert r.returncode != 0 and "both an invalid marker and data files" in r.stderr


def test_max_invalid_threshold(tmp_path):
    raw = tmp_path / "raw"
    write_rollouts(raw)
    make_invalid(raw, 7, 2)
    make_invalid(raw, 8, 0)
    r = build(raw, tmp_path / "c", "--max-invalid", "1")
    assert r.returncode != 0 and "exceed --max-invalid 1" in r.stderr and "task8--ep0--invalid.json" in r.stderr
    assert build(raw, tmp_path / "c", "--max-invalid", "2").returncode == 0


def test_bad_markers_are_refused(tmp_path):
    raw = tmp_path / "raw"
    write_rollouts(raw)
    make_invalid(raw, 3, 1, episode_idx=2)  # content names another episode: cannot explain task3--ep1
    r = build(raw, tmp_path / "c")
    assert r.returncode != 0 and "does not match the file name" in r.stderr
    (raw / "task3--ep1--invalid.json").write_text("{not json")
    r = build(raw, tmp_path / "c")
    assert r.returncode != 0 and "unreadable marker" in r.stderr
    make_invalid(raw, 3, 1)
    make_invalid(raw, 3, 9)  # outside --expect-episodes 4
    r = build(raw, tmp_path / "c")
    assert r.returncode != 0 and "outside 0..3" in r.stderr


def test_marker_tmp_files_are_ignored(tmp_path):
    raw = tmp_path / "raw"
    write_rollouts(raw)
    (raw / "task3--ep1--invalid.json.123.tmp").write_text("partial")
    r = build(raw, tmp_path / "c")
    assert r.returncode == 0, r.stderr
    assert json.loads((tmp_path / "c" / "v1" / "manifest.json").read_text())["invalid_episodes"] == []


def test_check_cache_reports_invalid_episodes(tmp_path):
    raw = tmp_path / "raw"
    write_rollouts(raw, n_tasks=2, n_eps=3)
    make_invalid(raw, 1, 2)
    r = run(KIT / "check_cache.py", "capture", "--rollout-dir", raw)
    assert r.returncode == 0 and "5/5 episodes pass" in r.stdout
    assert "1 invalid episodes" in r.stdout and "task1--ep2 (mujoco.FatalError)" in r.stdout
    b = tmp_path / "b"
    write_rollouts(b, n_tasks=2, n_eps=3)
    r = run(KIT / "check_cache.py", "compare-runs", "--a", raw, "--b", b)
    assert r.returncode == 0 and "run a: 1 invalid episodes" in r.stdout
