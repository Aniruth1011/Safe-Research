"""B5: standardization is fit on seen-train only, applied identically everywhere, and safe on constant features."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from test_shared_cache import KIT, REPO, run, write_rollouts

sys.path.insert(0, str(KIT))
from load_cache import SharedCache  # noqa: E402


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("b5")
    write_rollouts(tmp / "raw")
    r = run(KIT / "build_cache.py", "--rollout-dir", tmp / "raw", "--cache-root", tmp / "c", "--expect-episodes", 4)
    assert r.returncode == 0, r.stderr
    return tmp / "c" / "v1"


def _rows(cache, pool, split, layer):
    return np.load(cache / "features" / pool / split / f"L{layer:02d}.npy").astype(np.float64)


@pytest.mark.parametrize("pool", ["mean", "first", "last"])
@pytest.mark.parametrize("layer", [0, 4, 32])
def test_stats_are_train_only(cache, pool, layer):
    z = np.load(cache / "norm_stats" / f"{pool}.npz")
    i = list(z["layers"]).index(layer)
    train = _rows(cache, pool, "train", layer)
    assert np.allclose(z["mean"][i], train.mean(0), atol=1e-3, rtol=1e-4)
    assert np.allclose(z["std"][i], train.std(0), atol=1e-3, rtol=1e-4)
    assert int(z["n_rows"]) == train.shape[0]


def test_stats_are_not_fit_on_validation(cache):
    """Leakage guard: the stored stats must differ from stats that include seen_val (or unseen) rows."""
    z = np.load(cache / "norm_stats" / "mean.npz")
    i = list(z["layers"]).index(4)
    both = np.concatenate([_rows(cache, "mean", "train", 4), _rows(cache, "mean", "seen_val", 4)])
    assert not np.allclose(z["mean"][i], both.mean(0), atol=1e-4)
    every = np.concatenate([both, _rows(cache, "mean", "unseen_test", 4)])
    assert not np.allclose(z["mean"][i], every.mean(0), atol=1e-4)


def test_normalized_train_is_unit_scale_and_val_uses_train_stats(cache):
    c = SharedCache(cache)
    tr = np.concatenate([r["features"][0, :, 0].numpy() for r in c.records("train", [16], normalize=True)])
    _, raw_std = c.norm_stats("mean", [16])
    varying = raw_std[0] > 1e-6                     # constant dims standardize to exactly 0 (floor), the rest to unit scale
    assert varying.any()
    assert np.allclose(tr.mean(0), 0, atol=1e-2)
    assert np.allclose(tr.std(0)[varying], 1, atol=1e-2) and np.all(tr[:, ~varying] == 0)
    raw_val = np.concatenate([r["features"][0, :, 0].numpy() for r in c.records("seen_val", [16])])
    norm_val = np.concatenate([r["features"][0, :, 0].numpy() for r in c.records("seen_val", [16], normalize=True)])
    mean, std = c.norm_stats("mean", [16])
    assert np.allclose(norm_val, (raw_val - mean[0]) / std[0], atol=1e-4)   # train stats, not val's own


def test_constant_feature_standardizes_to_zero_not_inf(cache, tmp_path):
    """Real cache v1: layer 0 with 'first' pooling is the same prompt token every step, so std is exactly 0
    in all 4096 dims. The 1e-6 floor must turn it into 0, never inf/nan."""
    import shutil
    c2 = tmp_path / "v1"
    shutil.copytree(cache, c2)
    for split in ("train", "seen_val"):
        f = c2 / "features" / "first" / split / "L00.npy"
        a = np.load(f)
        np.save(f, np.full_like(a, 0.5))
    z = dict(np.load(c2 / "norm_stats" / "first.npz"))
    i = list(z["layers"]).index(0)
    z["mean"][i] = 0.5
    z["std"][i] = 0.0
    np.savez(c2 / "norm_stats" / "first.npz", **z)
    c = SharedCache(c2)
    _, std = c.norm_stats("first", [0])
    assert np.all(std == 1e-6)
    for split in ("train", "seen_val"):
        x = np.concatenate([r["features"].numpy().ravel() for r in c.records(split, [0], pool="first", normalize=True)])
        assert np.all(np.isfinite(x)) and np.all(x == 0)


def test_train_py_normalize_switch(cache, tmp_path):
    """Raw vs standardized is one flag; the run records which one it used."""
    for flag in ([], ["--normalize"]):
        r = subprocess.run([sys.executable, str(REPO / "scripts" / "train.py"), "--model", "single_layer_mlp",
                            "--layers", "4", "--cache", str(cache), "--seeds", "0", "--epochs", "1",
                            "--output-root", str(tmp_path / "runs"), *flag], capture_output=True, text=True, cwd=tmp_path)
        assert r.returncode == 0, r.stderr
        run_dir = Path(r.stdout.strip().splitlines()[-1])
        cfg = json.loads((run_dir / "config.json").read_text())
        assert cfg["normalize"] is bool(flag)
        m = json.loads((run_dir / "metrics.json").read_text())
        assert np.isfinite(m["roc_auc"]) and m["eval_split"] == "seen_val"


def test_check_norm_stats_script(cache):
    r = subprocess.run([sys.executable, "-I", str(REPO / "scripts" / "check_norm_stats.py"), str(cache)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("=== pool:") == 3 and "smallest std in pool 'mean'" in r.stdout
