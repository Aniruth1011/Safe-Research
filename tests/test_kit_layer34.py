"""Must-fix 5: 34-layer rollouts (index 33 = raw block 32, before the final RMSNorm) in build/check/load_cache.
Datasets come from test_shared_cache.write_rollouts (33 layers, unchanged); `to_34` then appends index 33 and
`layer_names` the way the patched run_libero_eval.py does. No GPU needed."""
import json

import pytest
import torch

from test_shared_cache import D, KIT, POOLS, _lc, run, write_rollouts
from build_cache import LEGACY_LAYER_NAMES
from load_cache import SharedCache

NAMES34 = LEGACY_LAYER_NAMES + ["block32_prenorm"]


def to_34(path, seed=0):
    """Append a distinct raw-block-32 layer (index 33) and layer_names to one .layers.pt."""
    p = torch.load(path, weights_only=False)
    f = p["features"]
    raw = torch.randn(f.shape[0], f.shape[1], 1, f.shape[3], generator=torch.Generator().manual_seed(seed)) * 3 - 7
    p["features"], p["layer_names"] = torch.cat([f, raw.to(f.dtype)], dim=2), NAMES34
    torch.save(p, path)


def rollouts34(root, n_tasks=10, n_eps=4, only=None):
    write_rollouts(root, n_tasks=n_tasks, n_eps=n_eps)
    for i, path in enumerate(sorted(root.glob("*.layers.pt"))):
        if only is None or path.name.startswith(only):
            to_34(path, seed=i)


def build(raw, cache_root, *extra):
    return run(KIT / "build_cache.py", "--rollout-dir", raw, "--cache-root", cache_root, "--expect-episodes", 4, *extra)


def test_build_load_34_layers(tmp_path):
    raw = tmp_path / "raw"
    rollouts34(raw)
    r = build(raw, tmp_path / "c")
    assert r.returncode == 0, r.stderr + r.stdout
    cache = tmp_path / "c" / "v1"
    man = json.loads((cache / "manifest.json").read_text())
    assert man["layer_names"] == NAMES34 and man["num_hidden_states"] == 34
    assert (cache / "features" / "mean" / "train" / "L33.npy").exists()
    c = SharedCache(cache)
    assert c.layer_names[32] == "block32_postnorm" and c.layer_names[33] == "block32_prenorm"
    ep = c.episodes[c.episodes.split == "train"].iloc[0]
    rec = c.records("train", layers=[33, 32], pool="last")[0]["features"][0]
    feats = torch.load(raw / f"task{ep.task_id}--ep{ep.episode_idx}--succ{ep.success}.layers.pt",
                       weights_only=False)["features"]
    assert torch.equal(rec[:, 0], feats[:, 2, 33].float()) and torch.equal(rec[:, 1], feats[:, 2, 32].float())
    with pytest.raises(ValueError, match="not in this cache"):
        c.records("train", layers=[34])
    assert run(KIT / "check_cache.py", "verify", "--cache", cache).returncode == 0


def test_capture_check_uses_index_32_in_34_layer_files(tmp_path):
    raw = tmp_path / "raw"
    rollouts34(raw, n_tasks=2, n_eps=2)
    r = run(KIT / "check_cache.py", "capture", "--rollout-dir", raw)
    assert r.returncode == 0 and "4/4 episodes pass" in r.stdout, r.stdout + r.stderr
    p = next(raw.glob("task1--ep0--*.layers.pt"))
    payload = torch.load(p, weights_only=False)
    payload["features"][:, :, 32] = payload["features"][:, :, 33]  # wrong tensor saved as L32
    torch.save(payload, p)
    r = run(KIT / "check_cache.py", "capture", "--rollout-dir", raw)
    assert r.returncode == 1 and "L32 != SAFE pkl" in r.stdout and "identical to previous layer: [33]" in r.stdout


def test_mixed_33_34_dataset_refused(tmp_path):
    raw = tmp_path / "raw"
    rollouts34(raw, only="task7--")
    r = build(raw, tmp_path / "c")
    assert r.returncode != 0 and "mixed layer conventions" in (r.stderr + r.stdout)
    assert not (tmp_path / "c" / "v1").exists()
    small = tmp_path / "small"
    rollouts34(small, n_tasks=2, n_eps=2, only="task1--")
    r = run(KIT / "check_cache.py", "capture", "--rollout-dir", small)
    assert r.returncode == 1 and "mixed layer conventions" in r.stdout


def test_legacy_files_without_layer_names(tmp_path):
    raw = tmp_path / "raw"
    write_rollouts(raw)
    for path in raw.glob("*.layers.pt"):  # strip the key if the regenerated patch already writes it
        p = torch.load(path, weights_only=False)
        if p.pop("layer_names", None) is not None:
            torch.save(p, path)
    r = build(raw, tmp_path / "c")
    assert r.returncode == 0, r.stderr + r.stdout
    man_path = tmp_path / "c" / "v1" / "manifest.json"
    man = json.loads(man_path.read_text())
    assert man["layer_names"] == LEGACY_LAYER_NAMES and man["num_hidden_states"] == 33
    del man["layer_names"]  # a cache built before this key existed
    man_path.write_text(json.dumps(man))
    assert SharedCache(tmp_path / "c" / "v1").layer_names == LEGACY_LAYER_NAMES
    # 34 layers without layer_names is ambiguous -> refused
    path = next(raw.glob("task2--ep1--*.layers.pt"))
    to_34(path)
    p = torch.load(path, weights_only=False)
    del p["layer_names"]
    torch.save(p, path)
    r = build(raw, tmp_path / "c", "--version", "v2")
    assert r.returncode != 0 and "only 33-layer files may omit layer_names" in (r.stderr + r.stdout)


def test_compare_runs_refuses_mixed(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_rollouts(a, n_tasks=1, n_eps=2)
    rollouts34(b, n_tasks=1, n_eps=2)
    r = run(KIT / "check_cache.py", "compare-runs", "--a", a, "--b", b)
    assert r.returncode == 1 and "different layer conventions" in r.stdout


def test_export_layer_33_to_feature_store(tmp_path):
    import os
    import subprocess
    import sys
    raw = tmp_path / "raw"
    rollouts34(raw)
    assert build(raw, tmp_path / "c").returncode == 0
    env = dict(os.environ, PYTHONPATH=str(KIT.parent))
    r = subprocess.run([sys.executable, str(KIT / "load_cache.py"), "export", "--cache", str(tmp_path / "c" / "v1"),
                        "--out", str(tmp_path / "fs"), "--layers", "32", "33"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    from failure_prob.data.feature_store import FeatureStore
    store = FeatureStore(tmp_path / "fs" / "train")
    assert store.metadata().num_layers == 34 and "33=block32_prenorm" in store.metadata().layer_indexing
    assert store.load([33])[0]["features"].shape[2:] == (1, D)


def test_patch_layer_names_match_kit():
    """Once the orchestrator regenerates openvla_all_layers.patch, its names must be the ones the kit expects."""
    if not hasattr(_lc, "make_layer_names"):
        pytest.skip("openvla_all_layers.patch predates layer_names (regenerated by the orchestrator)")
    assert _lc.make_layer_names(32, raw_block32=False) == LEGACY_LAYER_NAMES
    assert _lc.make_layer_names(32, raw_block32=True) == NAMES34
    assert POOLS == ["mean", "first", "last"]
