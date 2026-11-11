"""Tests for collection/ (rollout patch helper, cache builder, checks, FeatureStore export) and the fixed seeds.
No GPU, simulator or OpenVLA weights needed. layer_cache.py is loaded straight out of openvla_all_layers.patch,
so the code tested is exactly the code applied on CARC.
"""
import importlib.util
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "collection"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(KIT))
from load_cache import SharedCache  # noqa: E402
from failure_prob import protocol as PROTO  # noqa: E402


def _load_layer_cache_from_patch():
    text = (KIT / "openvla_all_layers.patch").read_text()
    start = text.index("+++ b/experiments/robot/libero/layer_cache.py")
    body = text[start:].split("\n@@", 1)[1].split("\n", 1)[1]
    end = body.find("\ndiff --git")
    body = body if end < 0 else body[:end]
    src = "\n".join(line[1:] for line in body.splitlines() if line.startswith("+"))
    import tempfile
    path = Path(tempfile.mkdtemp()) / "layer_cache_from_patch.py"
    path.write_text(src + "\n")
    spec = importlib.util.spec_from_file_location("layer_cache_from_patch", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_lc = _load_layer_cache_from_patch()
pool_all_layers, save_episode_layers = _lc.pool_all_layers, _lc.save_episode_layers

N_LAYERS, D, N_TOK, POOLS = 33, 16, 7, ["mean", "first", "last"]
COLS = ["action/dx", "action/dy", "action/dz", "action/droll", "action/dpitch", "action/dyaw", "action/dgripper"]


def fake_step(gen, scale=1.0):
    # token 0 sees the whole prompt (seq=5), later tokens use the KV cache (seq=1); values encode (token, layer)
    toks = []
    for k in range(N_TOK):
        seq = 5 if k == 0 else 1
        toks.append(tuple(torch.randn(1, seq, D, generator=gen).to(torch.bfloat16) * scale + (k * 100 + l)
                          for l in range(N_LAYERS)))
    return tuple(toks)


def write_rollouts(root: Path, n_tasks=10, n_eps=4, seed=0, flip=None):
    root.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(seed)
    for t in range(n_tasks):
        for e in range(n_eps):
            T = 6 + (t + e) % 4
            succ = int((t + e) % 2 == 0)
            if flip and (t, e) in flip:
                succ = 1 - succ
            steps, safe_last = [], []
            for _ in range(T):  # same two computations as the patched run_libero_eval.py
                hs = fake_step(gen)
                safe_last.append(torch.stack([s[-1][0, -1, :] for s in hs], dim=0))   # SAFE's (7, D)
                steps.append(pool_all_layers(hs, POOLS))                              # ours (P, 33, D)
            stem = root / f"task{t}--ep{e}--succ{succ}"
            pickle.dump({"hidden_states": torch.stack(safe_last), "task_id": t}, open(stem.with_suffix(".pkl"), "wb"))
            save_episode_layers(stem.with_suffix(".layers.pt"), steps, POOLS,
                                {"task_id": t, "episode_idx": e, "episode_success": bool(succ),
                                 "task_description": f"fake task {t}", "task_suite_name": "libero_10"})
            acts = np.zeros((T, 7)) + (0.5 if flip and (t, e) in flip else 0.0) * (np.arange(T)[:, None] >= 2)
            pd.DataFrame(acts, columns=COLS).to_csv(stem.with_suffix(".csv"), index=False)
    (root / "provenance").mkdir(exist_ok=True)
    (root / "provenance" / "provenance--tasks0-9--joblocal.json").write_text("{}")


def run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)
    return r


def test_pooling_matches_definition():
    gen = torch.Generator().manual_seed(1)
    step = fake_step(gen, scale=0.0)  # exact values k*100 + l
    out = pool_all_layers(step, POOLS)
    assert out.shape == (3, N_LAYERS, D) and out.dtype == torch.bfloat16
    layer = 4
    assert torch.allclose(out[1, layer].float(), torch.full((D,), 0 * 100 + layer, dtype=torch.float32))   # first token
    assert torch.allclose(out[2, layer].float(), torch.full((D,), 6 * 100 + layer, dtype=torch.float32))   # last token
    assert torch.allclose(out[0, layer].float(), torch.full((D,), 300 + layer, dtype=torch.float32))       # mean of 0..6 *100


@pytest.fixture()
def built(tmp_path):
    raw = tmp_path / "rollouts" / "libero_10"
    write_rollouts(raw)
    r = run(KIT / "build_cache.py", "--rollout-dir", raw, "--cache-root", tmp_path / "cache",
            "--expect-episodes", 4)
    assert r.returncode == 0, r.stderr + r.stdout
    return tmp_path, raw, tmp_path / "cache" / "v1"


def test_build_verify_load(built):
    tmp, raw, cache = built
    man = json.loads((cache / "manifest.json").read_text())
    assert man["splits"]["unseen_test"]["trajectories"] == 12  # tasks 0,3,5 x 4
    eps = pd.read_csv(cache / "episodes.csv")
    sp = json.loads((cache / "splits.json").read_text())["trajectories"]
    assert set(eps[eps.split == "unseen_test"].task_id) == {0, 3, 5}
    assert not set(sp["train"]) & set(sp["seen_val"])
    assert len(sp["train"]) + len(sp["seen_val"]) + len(sp["unseen_test"]) == 40
    # each seen task: stratified 60/40 of 2 succ + 2 fail -> 1 of each class in val
    val = eps[eps.split == "seen_val"]
    assert (val.groupby("task_id").failure.nunique() == 2).all()
    # task_min_step = min length per task
    assert (eps.task_min_step == eps.groupby("task_id").length.transform("min")).all()

    assert run(KIT / "check_cache.py", "verify", "--cache", cache).returncode == 0

    # feature values survive bf16 -> fp16 and land at the right rows, in the requested layer order
    c = SharedCache(cache)
    recs = c.records("train", layers=[32, 4], pool="first")
    ep = eps[eps.split == "train"].iloc[0]
    raw_feats = torch.load(raw / f"task{ep.task_id}--ep{ep.episode_idx}--succ{ep.success}.layers.pt",
                           weights_only=False)["features"]
    got = recs[0]["features"][0]  # (T, 2, D)
    assert got.shape == (ep.length, 2, D)
    assert torch.equal(got[:, 0], raw_feats[:, 1, 32].float())
    assert torch.equal(got[:, 1], raw_feats[:, 1, 4].float())
    assert recs[0]["labels"].item() == ep.failure and recs[0]["trajectory_ids"] == [ep.trajectory_id]

    # SAFE horizon truncation
    short = c.records("train", layers=[4], horizon="task_min_step")
    assert all(r["features"].shape[1] <= e for r, e in zip(short, eps[eps.split == "train"].task_min_step))

    # train-only normalisation stats
    z = np.load(cache / "norm_stats" / "mean.npz")
    train_rows = np.load(cache / "features" / "mean" / "train" / "L04.npy").astype(np.float64)
    assert np.allclose(z["mean"][4], train_rows.mean(0), atol=1e-2)


def test_withheld_unseen_and_corruption(built):
    tmp, raw, cache = built
    import shutil
    shutil.rmtree(cache / "features" / "mean" / "unseen_test")
    r = run(KIT / "check_cache.py", "verify", "--cache", cache)
    assert r.returncode == 0 and "unseen-test files withheld" in r.stdout
    with pytest.raises(PermissionError, match="locked"):
        SharedCache(cache).records("unseen_test", layers=[4], pool="mean")
    with pytest.raises(FileNotFoundError, match="withheld"):
        SharedCache(cache).records("unseen_test", layers=[4], pool="mean", allow_unseen=True)
    f = cache / "features" / "mean" / "train" / "L04.npy"
    b = bytearray(f.read_bytes()); b[-1] ^= 0xFF; f.write_bytes(bytes(b))
    r = run(KIT / "check_cache.py", "verify", "--cache", cache)
    assert r.returncode == 1 and "CORRUPTED" in r.stdout


def test_rebuild_with_same_split_file_is_identical(built):
    tmp, raw, cache = built
    r = run(KIT / "build_cache.py", "--rollout-dir", raw, "--cache-root", tmp / "cache", "--version", "v1b",
            "--split-file", cache / "splits.json", "--expect-episodes", 4)
    assert r.returncode == 0, r.stderr
    a = json.loads((cache / "splits.json").read_text())["trajectories"]
    b = json.loads((tmp / "cache" / "v1b" / "splits.json").read_text())["trajectories"]
    assert a == b
    # immutability: refuses to overwrite
    r = run(KIT / "build_cache.py", "--rollout-dir", raw, "--cache-root", tmp / "cache", "--expect-episodes", 4)
    assert r.returncode != 0 and "immutable" in (r.stderr + r.stdout)


def test_incomplete_rollouts_rejected(tmp_path):
    raw = tmp_path / "raw"
    write_rollouts(raw, n_tasks=10, n_eps=4)
    next(raw.glob("task7--ep2--*.layers.pt")).unlink()
    r = run(KIT / "build_cache.py", "--rollout-dir", raw, "--cache-root", tmp_path / "c", "--expect-episodes", 4)
    assert r.returncode != 0 and "missing" in (r.stderr + r.stdout)
    assert not (tmp_path / "c" / "v1").exists()


def test_task_names_and_provenance_recorded(tmp_path):
    raw = tmp_path / "raw"
    write_rollouts(raw)
    extra = tmp_path / "libero_commit.txt"
    extra.write_text("abc123\n")
    r = run(KIT / "build_cache.py", "--rollout-dir", raw, "--cache-root", tmp_path / "c", "--expect-episodes", 4,
            "--provenance-files", extra)
    assert r.returncode == 0, r.stderr
    cache = tmp_path / "c" / "v1"
    eps = pd.read_csv(cache / "episodes.csv")
    assert (eps.task_name == "fake task " + eps.task_id.astype(str)).all()
    sp = json.loads((cache / "splits.json").read_text())
    assert sp["task_names"]["3"] == "fake task 3" and "NOT SAFE's" in sp["provenance_of_task_split"]
    man = json.loads((cache / "manifest.json").read_text())
    assert man["task_suite_name"] == "libero_10" and man["features_are_unnormalized"] is True
    assert "provenance/libero_commit.txt" in man["files"]  # checksummed, not copied in afterwards
    assert not any("\\" in k for k in man["files"]), "manifest paths must use / on every OS"


def test_capture_check(tmp_path):
    raw = tmp_path / "raw"
    write_rollouts(raw, n_tasks=2, n_eps=2)
    r = run(KIT / "check_cache.py", "capture", "--rollout-dir", raw)
    assert r.returncode == 0 and "4/4 episodes pass" in r.stdout, r.stdout + r.stderr
    # simulate a wrong layer being saved as "L32": capture check must catch it
    p = next(raw.glob("task1--ep0--*.layers.pt"))
    payload = torch.load(p, weights_only=False)
    payload["features"][:, :, 32] = payload["features"][:, :, 31]
    torch.save(payload, p)
    r = run(KIT / "check_cache.py", "capture", "--rollout-dir", raw)
    assert r.returncode == 1 and "L32 != SAFE pkl" in r.stdout and "identical to previous layer: [32]" in r.stdout


def test_safe_seed0_task_split():
    """Run SAFE's own split code (in this repo): seed 0 must pick unseen tasks {0,3,5} for 10 tasks."""
    safe_repo = str(REPO)
    code = (
        "from types import SimpleNamespace as NS\n"
        "from failure_prob.conf import OpenvlaDatasetConfig\n"
        "from failure_prob.utils.random import seed_everything\n"
        "from failure_prob.data.openvla import split_rollouts\n"
        "cfg = NS(dataset=OpenvlaDatasetConfig(data_path='x'))\n"
        "rs = [NS(task_id=t, episode_success=e % 2) for t in range(10) for e in range(50)]\n"
        "seed_everything(0)\n"
        "print(sorted({r.task_id for r in split_rollouts(cfg, rs)['val_unseen']}))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=safe_repo,
                       env=dict(os.environ, PYTHONPATH=safe_repo))
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == "[0, 3, 5]"


def test_compare_runs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_rollouts(a, n_tasks=1, n_eps=3, seed=0)
    write_rollouts(b, n_tasks=1, n_eps=3, seed=0, flip={(0, 1)})
    r = run(KIT / "check_cache.py", "compare-runs", "--a", a, "--b", b)
    assert r.returncode == 0, r.stderr
    assert "outcome flips: 1" in r.stdout and "diverged: 1" in r.stdout


def test_feature_store_export_and_train_with_protocol_seeds(built):
    tmp, raw, cache = built
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, str(KIT / "load_cache.py"), "export", "--cache", str(cache),
                        "--out", str(tmp / "fs"), "--layers", "4", "24", "32"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in (tmp / "fs").iterdir()) == ["seen_val", "train"]  # unseen not exported by default
    r = subprocess.run([sys.executable, str(KIT / "load_cache.py"), "export", "--cache", str(cache),
                        "--out", str(tmp / "fs2"), "--layers", "4", "--splits", "unseen_test"],
                       capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "locked" in r.stderr
    from failure_prob.data.feature_store import FeatureStore
    recs = FeatureStore(tmp / "fs" / "train").load([4, 32])  # L32 passes the store's layer validation
    assert recs[0]["features"].shape[2:] == (2, D)
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "train.py"), "--model", "single_layer_mlp",
                        "--layers", "4", "--feature-root", str(tmp / "fs" / "train"),
                        "--output-root", str(tmp / "runs")], capture_output=True, text=True, env=env, cwd=tmp)
    assert r.returncode == 0, r.stderr
    runs = sorted(p.name for p in (tmp / "runs").iterdir() if p.is_dir())
    assert [n.split("_seed")[1].split("_")[0] for n in runs] == [str(s) for s in PROTO.TRAIN_SEEDS]


def test_fixed_seeds_consistent_everywhere():
    """failure_prob/protocol.py is the single source; the CARC config, cache builder and train CLI must agree."""
    import re
    cfg = (KIT / "carc" / "config.sh").read_text()
    assert re.search(r"^export SEED=(\d+)", cfg, re.M).group(1) == str(PROTO.ROLLOUT_SEED)
    for slurm in ("../smoke_test.sh", "02_rollouts.slurm"):  # 01_smoke_test.slurm runs smoke_test.sh
        assert '--seed "$SEED"' in (KIT / "carc" / slurm).read_text()
    src = (KIT / "build_cache.py").read_text()
    assert f'"--unseen-tasks", type=int, nargs="+", default={list(PROTO.UNSEEN_TASKS)}' in src
    assert f'"--seen-train-ratio", type=float, default={PROTO.SEEN_TRAIN_RATIO}' in src
    assert f'"--split-seed", type=int, default={PROTO.SPLIT_SEED}' in src
    train_src = (REPO / "scripts" / "train.py").read_text()
    assert "default=list(TRAIN_SEEDS)" in train_src
