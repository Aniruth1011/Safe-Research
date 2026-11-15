"""A9: review of collection/load_cache.py (records + export) - layer keys 0..33, normalization, unseen lock.

Each test pins one property the team relies on. Tests marked xfail(strict=True) are review findings that are still
open in load_cache.py (see docs/reviews/A9_loader_export_review.md); when a fix lands they XPASS, strict mode turns
that into a failure, and the marker should be removed so the test guards the fix from then on.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from test_kit_layer34 import NAMES34, build, rollouts34
from test_shared_cache import D, KIT, POOLS, REPO
from load_cache import SharedCache
from failure_prob.data.feature_store import FeatureStore

ENV = dict(os.environ, PYTHONPATH=str(REPO))


@pytest.fixture(scope="module")
def c34(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("a9")
    rollouts34(tmp / "raw")
    r = build(tmp / "raw", tmp / "c")
    assert r.returncode == 0, r.stderr + r.stdout
    return tmp, tmp / "raw", tmp / "c" / "v1"


def export(cache, out, *args):
    return subprocess.run([sys.executable, str(KIT / "load_cache.py"), "export", "--cache", str(cache),
                           "--out", str(out), *map(str, args)], capture_output=True, text=True, env=ENV)


def raw_layers(raw, ep):
    return torch.load(raw / f"task{ep.task_id}--ep{ep.episode_idx}--succ{ep.success}.layers.pt",
                      weights_only=False)["features"]  # (T, P, 34, D)


# --------------------------------------------------------------------------- layer keys 0..33

def test_every_layer_key_reads_its_own_layer(c34):
    _, raw, cache = c34
    c = SharedCache(cache)
    assert c.layer_names == NAMES34 and len(c.layer_names) == 34
    assert c.layer_names[0] == "emb" and c.layer_names[32] == "block32_postnorm" and c.layer_names[33] == "block32_prenorm"
    for split in ("train", "seen_val"):
        ep = c.episodes[c.episodes.split == split].iloc[1]  # not the first row: exercises row_start offsets
        feats = raw_layers(raw, ep)
        for pool in POOLS:
            recs = c.records(split, layers=list(range(34)), pool=pool)
            rec = next(r for r in recs if r["trajectory_ids"] == [ep.trajectory_id])["features"][0]
            assert rec.shape == (ep.length, 34, D)
            # fp16 cache vs bf16 source: every bf16 value in this fixture is exactly representable in fp16
            assert torch.equal(rec, feats[:, POOLS.index(pool)].float()), (split, pool)
    rec = c.records("train", layers=[32, 33])[0]["features"][0]
    assert not torch.equal(rec[:, 0], rec[:, 1])  # post-norm L32 and raw block 32 (L33) are different tensors


def test_requested_layer_order_is_kept(c34):
    _, _, cache = c34
    c = SharedCache(cache)
    a = c.records("train", layers=[33, 0, 16])[0]["features"][0]
    b = c.records("train", layers=[0, 16, 33])[0]["features"][0]
    assert torch.equal(a[:, 0], b[:, 2]) and torch.equal(a[:, 1], b[:, 0]) and torch.equal(a[:, 2], b[:, 1])


@pytest.mark.parametrize("bad", [[34], [-1], [0, 34]])
def test_out_of_range_layers_refused(c34, bad):
    with pytest.raises(ValueError, match="not in this cache"):
        SharedCache(c34[2]).records("train", layers=bad)


def test_export_layer_keys_and_metadata(c34):
    tmp, _, cache = c34
    out = tmp / "fs_keys"
    r = export(cache, out, "--layers", 0, 32, 33)
    assert r.returncode == 0, r.stderr
    store = FeatureStore(out / "train")
    meta = store.metadata()
    assert meta.layers == (0, 32, 33) and meta.num_layers == 34 and "33=block32_prenorm" in meta.layer_indexing
    assert meta.split_version == "v1:train" and meta.dtype == "float32"
    recs = SharedCache(cache).records("train", layers=[33, 0], horizon="task_min_step")  # the export default
    got = store.load([33, 0])  # FeatureStore re-indexes by layer key, not by position
    assert all(torch.equal(g["features"], r["features"]) for g, r in zip(got, recs))
    with pytest.raises(ValueError):
        store.load([16])  # not exported
    with pytest.raises(ValueError):
        store.validate_layers([34])


# --------------------------------------------------------------------------- normalization

def test_norm_stats_are_train_only_per_layer(c34):
    _, _, cache = c34
    z = np.load(cache / "norm_stats" / "mean.npz")
    assert z["layers"].tolist() == list(range(34))
    eps = pd.read_csv(cache / "episodes.csv")
    assert int(z["n_rows"]) == int(eps[eps.split == "train"].length.sum())  # train rows only, full horizon
    for l in (0, 16, 33):
        tr = np.load(cache / "features" / "mean" / "train" / f"L{l:02d}.npy").astype(np.float64)
        assert np.allclose(z["mean"][l], tr.mean(0), atol=1e-2)
        assert np.allclose(z["std"][l], tr.std(0), atol=1e-2)


@pytest.mark.parametrize("split", ["train", "seen_val"])
def test_normalize_uses_the_right_layer_stats_in_any_order(c34, split):
    _, _, cache = c34
    c = SharedCache(cache)
    layers = [33, 4, 16]  # unsorted on purpose
    mean, std = c.norm_stats("mean", layers)
    raw = c.records(split, layers=layers)
    norm = c.records(split, layers=layers, normalize=True)
    for r, n in zip(raw, norm):
        assert torch.allclose(n["features"], (r["features"] - torch.from_numpy(mean)) / torch.from_numpy(std),
                              atol=1e-5)
    if split == "train":
        x = torch.cat([n["features"][0] for n in norm]).double()  # (rows, K, D)
        assert torch.allclose(x.mean(0), torch.zeros_like(x[0]), atol=1e-3)


def test_normalized_export_matches_records(c34):
    tmp, _, cache = c34
    out = tmp / "fs_norm"
    assert export(cache, out, "--layers", 33, 4, "--normalize").returncode == 0
    store = FeatureStore(out / "seen_val")
    assert store.metadata().pooling == "mean+trainnorm+h=task_min_step"
    recs = SharedCache(cache).records("seen_val", layers=[33, 4], normalize=True, horizon="task_min_step")
    assert all(torch.equal(g["features"], r["features"]) for g, r in zip(store.load(), recs))


# --------------------------------------------------------------------------- unseen lock + splits

def test_unseen_lock(c34):
    tmp, _, cache = c34
    c = SharedCache(cache)
    with pytest.raises(PermissionError, match="locked"):
        c.records("unseen_test", layers=[33])
    with pytest.raises(PermissionError, match="locked"):
        c.records("unseen_test", layers=[33], normalize=True, horizon="task_min_step")
    r = export(cache, tmp / "fs_unseen", "--layers", 33, "--splits", "train", "unseen_test")
    assert r.returncode != 0 and "locked" in r.stderr
    out = tmp / "fs_default"
    assert export(cache, out, "--layers", 33).returncode == 0
    assert sorted(p.name for p in out.iterdir()) == ["seen_val", "train"]
    unseen = set(json.loads((cache / "splits.json").read_text())["trajectories"]["unseen_test"])
    for split in ("train", "seen_val"):
        ids = {r["trajectory_ids"][0] for r in FeatureStore(out / split).load()}
        assert ids and not ids & unseen
        assert {int(t.split("/t")[1][:2]) for t in ids}.isdisjoint({0, 3, 5})
    # explicit opt-in still works (final locked benchmark only)
    assert len(c.records("unseen_test", layers=[33], allow_unseen=True)) == len(unseen)


def test_episodes_csv_agrees_with_splits_json(c34):
    _, _, cache = c34
    sp = json.loads((cache / "splits.json").read_text())["trajectories"]
    eps = pd.read_csv(cache / "episodes.csv")
    owner = {t: s for s, ids in sp.items() for t in ids}
    assert len(owner) == sum(len(v) for v in sp.values())  # no trajectory in two splits
    assert (eps.trajectory_id.map(owner) == eps.split).all()
    assert set(eps[eps.split == "unseen_test"].task_id) == {0, 3, 5}


def test_task_min_step_horizon_in_records(c34):
    _, _, cache = c34
    c = SharedCache(cache)
    eps = c.episodes[c.episodes.split == "seen_val"]
    recs = c.records("seen_val", layers=[33], horizon="task_min_step")
    assert [r["features"].shape[1] for r in recs] == eps.task_min_step.tolist()
    assert (eps.length > eps.task_min_step).any()  # fixture really has episodes that get cut


# --------------------------------------------------------------------------- fixed findings (A9-1/2/3)

def test_export_can_write_task_min_step_horizon(c34):
    tmp, _, cache = c34
    out = tmp / "fs_horizon"
    r = export(cache, out, "--layers", 33, "--horizon", "task_min_step")
    assert r.returncode == 0, r.stderr
    eps = pd.read_csv(cache / "episodes.csv").set_index("trajectory_id")
    for rec in FeatureStore(out / "seen_val").load():
        assert rec["features"].shape[1] == eps.loc[rec["trajectory_ids"][0], "task_min_step"]


def test_unknown_horizon_refused(c34):
    with pytest.raises(ValueError):
        SharedCache(c34[2]).records("train", layers=[33], horizon="task-min-step")


def test_duplicate_layers_refused(c34):
    with pytest.raises(ValueError):
        SharedCache(c34[2]).records("train", layers=[33, 33])


def test_export_default_is_task_min_step_and_full_is_labelled(c34):
    tmp, _, cache = c34
    eps = pd.read_csv(cache / "episodes.csv").set_index("trajectory_id")
    r = export(cache, tmp / "fs_default", "--layers", 0)
    assert r.returncode == 0, r.stderr
    store = FeatureStore(tmp / "fs_default" / "train")
    assert store.metadata().pooling == "mean+h=task_min_step"
    for rec in store.load():
        ep = eps.loc[rec["trajectory_ids"][0]]
        assert rec["features"].shape[1] == min(ep.length, ep.task_min_step)
    r = export(cache, tmp / "fs_full", "--layers", 0, "--horizon", "full")
    assert r.returncode == 0 and "length reveals the label" in r.stderr
    assert FeatureStore(tmp / "fs_full" / "train").metadata().pooling == "mean+h=full"


def test_unknown_pool_and_missing_norm_stats_layer_are_readable(c34):
    cache = SharedCache(c34[2])
    with pytest.raises(ValueError, match="not in this cache"):
        cache.records("train", layers=[0], pool="Mean")
    with pytest.raises(ValueError, match="no statistics for layers"):
        cache.norm_stats("mean", [0, 99])


def test_a9_verify_script_passes_on_a_good_cache(c34):
    tmp, _, cache = c34
    script = KIT.parent / "scripts" / "a9_verify_export.py"
    r = subprocess.run([sys.executable, str(script), "--cache", str(cache), "--out", str(tmp / "verify"),
                        "--layers", "33", "0"], capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "A9 PASS" in r.stdout and not (tmp / "verify" / "unseen_test").exists()
    r = subprocess.run([sys.executable, str(script), "--cache", str(cache), "--out", str(tmp / "verify")],
                       capture_output=True, text=True, env=ENV)
    assert r.returncode == 2 and "exists" in r.stderr  # never reuses / overwrites a folder
