"""E3 dataset-consistency guard in collection/build_cache.py: every episode's `run_fingerprint` (written by the patched
run_libero_eval.py --save_all_layers, see experiments/robot/libero/run_guards.py on the openvla side) must agree on
attention implementation (= flash_attention_2), GPU name and MuJoCo render backend. No GPU or simulator needed."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_shared_cache import KIT, run, write_rollouts  # noqa: E402

sys.path.insert(0, str(KIT))
import build_cache  # noqa: E402

GOOD = {"attn_implementation": "flash_attention_2", "gpu_name": "NVIDIA A10", "render_backend": {"MUJOCO_GL": "egl"}}


def set_fp(raw: Path, pattern: str, fp):
    """Rewrite run_fingerprint in the matching episodes (fp=None removes it)."""
    paths = sorted(raw.glob(pattern))
    assert paths, pattern
    for p in paths:
        payload = torch.load(p, weights_only=False)
        if fp is None:
            payload.pop("run_fingerprint", None)
        else:
            payload["run_fingerprint"] = fp
        torch.save(payload, p)
    return [p.name for p in paths]


def build(raw, cache_root, *extra):
    return run(KIT / "build_cache.py", "--rollout-dir", raw, "--cache-root", cache_root, "--expect-episodes", 4, *extra)


@pytest.fixture()
def raw(tmp_path):
    r = tmp_path / "raw"
    write_rollouts(r)
    return r


def test_consistent_dataset_builds_and_manifest_records_fingerprint(raw, tmp_path):
    fp = dict(GOOD, versions={"torch": "2.2.0", "transformers": "4.40.1", "flash_attn": "2.5.5"})
    set_fp(raw, "task*--ep*--*.layers.pt", fp)
    r = build(raw, tmp_path / "c")
    assert r.returncode == 0, r.stderr + r.stdout
    rec = json.loads((tmp_path / "c" / "v1" / "manifest.json").read_text())["run_fingerprint"]
    assert rec["attn_implementation"] == "flash_attention_2" and rec["gpu_name"] == "NVIDIA A10"
    assert rec["render_backend"] == {"MUJOCO_GL": "egl"}
    assert rec["versions_seen"] == [fp["versions"]]
    assert rec["episodes_with_fingerprint"] == 40
    assert rec["allow_missing_fingerprint"] is False and rec["episodes_missing_fingerprint"] == []


@pytest.mark.parametrize("key, other, needle", [
    ("gpu_name", "NVIDIA L4", "mixed gpu_name"),
    ("render_backend", {"MUJOCO_GL": "osmesa"}, "mixed render_backend"),
    ("render_backend", {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "osmesa"}, "mixed render_backend"),
    ("attn_implementation", "sdpa", "mixed attn_implementation"),
])
def test_mixed_dataset_is_refused(raw, tmp_path, key, other, needle):
    odd = set_fp(raw, "task4--ep*--*.layers.pt", dict(GOOD, **{key: other}))
    r = build(raw, tmp_path / "c")
    msg = r.stderr + r.stdout
    assert r.returncode != 0 and needle in msg, msg
    assert json.dumps(other, sort_keys=True) in msg and json.dumps(GOOD[key], sort_keys=True) in msg  # both values
    assert odd[0] in msg and "x4" in msg and "x36" in msg  # example episodes + counts
    assert not (tmp_path / "c" / "v1").exists()


def test_uniform_but_not_flash_attention_2_is_refused(raw, tmp_path):
    set_fp(raw, "task*--ep*--*.layers.pt", dict(GOOD, attn_implementation="eager"))
    r = build(raw, tmp_path / "c")
    msg = r.stderr + r.stdout
    assert r.returncode != 0 and "attn_implementation \"eager\" is not 'flash_attention_2'" in msg, msg
    assert not (tmp_path / "c" / "v1").exists()


def test_missing_fingerprint_refused_by_default_and_recorded_when_allowed(raw, tmp_path):
    gone = set_fp(raw, "task7--ep2--*.layers.pt", None)
    r = build(raw, tmp_path / "c")
    msg = r.stderr + r.stdout
    assert r.returncode != 0 and "1/40 episodes have no run_fingerprint" in msg and gone[0] in msg, msg
    assert "--allow-missing-fingerprint" in msg
    r = build(raw, tmp_path / "c", "--allow-missing-fingerprint")
    assert r.returncode == 0, r.stderr + r.stdout
    rec = json.loads((tmp_path / "c" / "v1" / "manifest.json").read_text())["run_fingerprint"]
    assert rec["allow_missing_fingerprint"] is True and rec["episodes_missing_fingerprint"] == gone
    assert rec["episodes_with_fingerprint"] == 39 and rec["gpu_name"] == "NVIDIA A10"


def test_allow_missing_does_not_excuse_a_mixed_dataset(raw, tmp_path):
    set_fp(raw, "task1--ep0--*.layers.pt", None)
    set_fp(raw, "task2--ep*--*.layers.pt", dict(GOOD, gpu_name="NVIDIA A40"))
    r = build(raw, tmp_path / "c", "--allow-missing-fingerprint")
    assert r.returncode != 0 and "mixed gpu_name" in (r.stderr + r.stdout)


def test_check_function_with_no_fingerprints_at_all(raw):
    set_fp(raw, "task*--ep*--*.layers.pt", None)
    df = build_cache.discover(raw)
    with pytest.raises(SystemExit, match="40/40 episodes have no run_fingerprint"):
        build_cache.check_run_fingerprints(df, allow_missing=False)
    rec = build_cache.check_run_fingerprints(df, allow_missing=True)
    assert rec["attn_implementation"] is None and rec["episodes_with_fingerprint"] == 0
    assert len(rec["episodes_missing_fingerprint"]) == 40 and isinstance(df, pd.DataFrame)
