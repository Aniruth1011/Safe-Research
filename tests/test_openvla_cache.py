"""Tests for failure_prob/data/openvla_cache.py: SAFE's original pipeline fed from the shared cache (C1-C3).

Uses a small fake cache built by collection/build_cache.py (10 tasks x 4 episodes, D=16), which contains
unseen_test features too, so the tests can check they are never read.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from test_shared_cache import KIT, REPO, run, write_rollouts

sys.path.insert(0, str(REPO))
from failure_prob.data import openvla_cache  # noqa: E402
from failure_prob.protocol import UNSEEN_TASKS  # noqa: E402
from failure_prob.utils.random import seed_everything  # noqa: E402


@pytest.fixture(scope="module")
def cache_root(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("oc")
    raw = tmp / "rollouts" / "libero_10"
    write_rollouts(raw)
    r = run(KIT / "build_cache.py", "--rollout-dir", raw, "--cache-root", tmp / "cache", "--expect-episodes", 4)
    assert r.returncode == 0, r.stderr + r.stdout
    root = tmp / "cache" / "v1"
    assert any((root / "features" / "mean" / "unseen_test").glob("L*.npy")), "fake cache should hold unseen files"
    return root


def _cfg(root, pool="mean", layer=32):
    return OmegaConf.create({"dataset": {"cache_root": str(root), "layer": layer, "token_idx_rel": pool,
                                         "expected_split_sizes": None}})


@pytest.fixture()
def no_unseen(monkeypatch):
    """Fail the test if anything asks the cache for unseen_test."""
    from load_cache import SharedCache
    orig_records, orig_array = SharedCache.records, SharedCache._array

    def records(self, split, *a, **k):
        assert split != "unseen_test", "unseen_test was requested"
        assert not k.get("allow_unseen", False)
        return orig_records(self, split, *a, **k)

    def _array(self, pool, split, layer):
        assert split != "unseen_test", "unseen_test features were opened"
        return orig_array(self, pool, split, layer)

    monkeypatch.setattr(SharedCache, "records", records)
    monkeypatch.setattr(SharedCache, "_array", _array)


def test_only_train_and_val_seen_matching_splits_json(cache_root, no_unseen):
    cfg = _cfg(cache_root)
    rollouts = openvla_cache.load_rollouts(cfg)
    out = openvla_cache.split_rollouts(cfg, rollouts)
    assert set(out) == {"train", "val_seen"}
    frozen = json.loads((cache_root / "splits.json").read_text())["trajectories"]
    assert sorted(r.trajectory_id for r in out["train"]) == sorted(frozen["train"])
    assert sorted(r.trajectory_id for r in out["val_seen"]) == sorted(frozen["seen_val"])
    assert not {r.task_id for r in rollouts} & set(UNSEEN_TASKS)
    assert len(rollouts) == len(frozen["train"]) + len(frozen["seen_val"])


def test_task_min_step_from_episodes_csv_and_seed_independent_split(cache_root, no_unseen):
    cfg = _cfg(cache_root)
    rollouts = openvla_cache.load_rollouts(cfg)
    eps = pd.read_csv(cache_root / "episodes.csv").set_index("trajectory_id")
    for r in rollouts:
        assert r.task_min_step == int(eps.loc[r.trajectory_id, "task_min_step"])
        assert r.episode_success == int(eps.loc[r.trajectory_id, "success"])
        assert r.hidden_states.shape[0] == int(eps.loc[r.trajectory_id, "length"])  # full horizon
    splits = []
    for seed in (0, 1, 2):
        seed_everything(seed)
        out = openvla_cache.split_rollouts(cfg, rollouts)
        splits.append({k: [r.trajectory_id for r in v] for k, v in out.items()})
    assert splits[0] == splits[1] == splits[2]


@pytest.mark.parametrize("rel,pool", [("mean", "mean"), (0.0, "first"), (1.0, "last"), ("0.0", "first")])
def test_token_idx_rel_maps_to_cache_pool(cache_root, rel, pool, no_unseen):
    assert openvla_cache.pool_name(rel) == pool
    layer = 32
    rollouts = openvla_cache.load_rollouts(_cfg(cache_root, pool=rel, layer=layer))
    arr = np.load(cache_root / "features" / pool / "train" / f"L{layer:02d}.npy")
    eps = pd.read_csv(cache_root / "episodes.csv").set_index("trajectory_id")
    r = next(r for r in rollouts if r.cache_split == "train")
    ep = eps.loc[r.trajectory_id]
    np.testing.assert_array_equal(r.hidden_states.numpy(), arr[ep.row_start:ep.row_end].astype(np.float32))


def test_bad_pool_and_wrong_sizes_raise(cache_root):
    with pytest.raises(ValueError):
        openvla_cache.pool_name(0.5)
    cfg = _cfg(cache_root)
    rollouts = openvla_cache.load_rollouts(cfg)
    cfg.dataset.expected_split_sizes = [210, 140]
    with pytest.raises(RuntimeError, match="split sizes"):
        openvla_cache.split_rollouts(cfg, rollouts)


# Runs SAFE's unchanged training entry point in a subprocess and records every key it passes to wandb.log.
_RUNNER = """
import json, sys, wandb
keys = set()
_init = wandb.init
def init(*a, **k):  # wandb.init rebinds wandb.log, so wrap it after every init
    run = _init(*a, **k)
    _log = wandb.log
    def log(d, *la, **lk):
        keys.update(d)
        return _log(d, *la, **lk)
    wandb.log = log
    return run
wandb.init = init
out = sys.argv.pop(1)
from failure_prob.train import main
try:
    main()
finally:
    json.dump(sorted(keys), open(out, "w"))
"""


@pytest.mark.parametrize("model,extra", [
    ("indep", ["model.n_epochs=2"]),
    ("lstm", ["model.n_epochs=2"]),
    ("embed", ["model.n_epochs=1", "model.distance=cosine", "model.use_success_only=False", "model.topk=1"]),
])
def test_safe_train_cpu_smoke(cache_root, tmp_path, model, extra):
    keys_file = tmp_path / "keys.json"
    cmd = [sys.executable, "-c", _RUNNER, str(keys_file),
           "dataset=openvla_cache", f"dataset.cache_root={cache_root}", "dataset.expected_split_sizes=null",
           f"model={model}", "model.batch_size=8", *extra, "train.seed=0", "train.eval_save_ckpt=True",
           f"train.wandb_dir={tmp_path}", f"train.logs_save_root={tmp_path / 'logs'}",
           f"hydra.run.dir={tmp_path / 'hydra'}"]
    env = {**__import__("os").environ, "WANDB_MODE": "disabled", "MPLBACKEND": "Agg",
           "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "train: 14 rollouts" in r.stdout and "val_seen: 14 rollouts" in r.stdout
    keys = json.loads(keys_file.read_text())
    assert "falert_early_roc_auc/model_val_seen" in keys
    assert not [k for k in keys if "unseen" in k]

    # The score dump reloads the checkpoint (k-NN: refits) and scores seen_val only
    (run_dir,) = {p.parent for p in (tmp_path / "logs").glob("**/config.yaml")}
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "dump_safe_scores.py"), "--run-dir", str(run_dir),
                        "--seed", "0", "--out-dir", str(tmp_path / "scores"), "--device", "cpu"],
                       cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    df = pd.read_csv(tmp_path / "scores" / f"scores_{model}_s0.csv")
    frozen = json.loads((cache_root / "splits.json").read_text())["trajectories"]
    assert sorted(df.trajectory_id) == sorted(frozen["seen_val"])
    assert (df.score_max_by_tms >= df.score_at_tms).all()
    npz = np.load(tmp_path / "scores" / f"scores_{model}_s0.npz")
    assert sorted(npz.files) == sorted(df.trajectory_id)
