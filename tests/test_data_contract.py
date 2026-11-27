"""B8: synthetic data is generated from the data contract, real code agrees with it, and drift fails loudly."""
import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from failure_prob.data import contract as C
from failure_prob.data.synthetic_cache import make_synthetic_cache

from test_shared_cache import KIT, REPO, _lc

sys.path.insert(0, str(KIT))
from load_cache import SharedCache  # noqa: E402


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return make_synthetic_cache(tmp_path_factory.mktemp("b8"), n_episodes=6, dim=16)


@pytest.fixture()
def broken(cache, tmp_path):
    dst = tmp_path / "v1"
    shutil.copytree(cache, dst)
    return dst


def problems_of(root):
    return C.validate_cache(root)[0]


# ---- the generator follows the contract, and the rest of the code agrees with the contract ---------------------
def test_synthetic_cache_satisfies_contract(cache):
    problems, notes = C.validate_cache(cache)
    assert problems == [] and notes == []


def test_contract_layer_names_match_the_rollout_patch():
    """collection/openvla_all_layers.patch is what writes layer_names on CARC/Modal."""
    assert _lc.make_layer_names(32, raw_block32=True) == C.LAYER_NAMES
    assert C.NUM_LAYERS == 34 and C.LAYER_NAMES[32] == "block32_postnorm" and C.LAYER_NAMES[33] == "block32_prenorm"


def test_contract_matches_protocol_and_builder():
    from failure_prob import protocol
    assert tuple(protocol.UNSEEN_TASKS) == C.UNSEEN_TASKS
    src = (KIT / "build_cache.py").read_text()
    cols = ast.literal_eval(src.split("cols = ", 1)[1].split("\n        df[cols]", 1)[0].replace("\n", " "))
    assert cols == C.EPISODE_COLUMNS


def test_loaders_accept_synthetic_cache(cache):
    from failure_prob.data.cache_splits import load_split_records
    c = SharedCache(cache)
    recs = c.records("train", [0, 32, 33], pool="first", normalize=True)
    assert recs and recs[0]["features"].shape[2:] == (3, 16)
    assert np.isfinite(np.concatenate([r["features"].numpy().ravel() for r in recs])).all()  # constant L0-first -> 0
    val = load_split_records(cache, "seen_val", [31, 33])
    assert len(val) == sum(pd.read_csv(cache / "episodes.csv").split == "seen_val")


def test_planted_signal_is_learnable(cache):
    """Failed episodes drift in late layers, not in early ones (so downstream tests can see a real effect)."""
    from failure_prob.engine.episode_metrics import episode_metrics
    eps = pd.read_csv(cache / "episodes.csv")
    val = eps[eps.split == "seen_val"]

    def auc(layer):
        x = np.load(cache / "features" / "mean" / "seen_val" / f"L{layer:02d}.npy").astype(np.float64)
        mu = np.load(cache / "norm_stats" / "mean.npz")["mean"][layer]
        direction = x[[r for e in val.itertuples() if e.failure for r in range(e.row_start, e.row_end)]].mean(0) - mu
        scores = [((x[e.row_start:e.row_end] - mu) @ direction) for e in val.itertuples()]
        return episode_metrics(scores, val.failure.to_numpy(), np.minimum(val.length, val.task_min_step))["roc_auc"]
    assert auc(31) > 0.8 and auc(31) > auc(2)


# ---- drift: each break is caught, with a message that says what broke ---------------------------------------------
def _edit_json(path, fn):
    d = json.loads(path.read_text())
    fn(d)
    path.write_text(json.dumps(d))


def _edit_csv(path, fn):
    df = pd.read_csv(path)
    df = fn(df)
    df.to_csv(path, index=False)


def _resave(path, fn):
    a = np.load(path)
    np.save(path, fn(a))


DRIFTS = {
    "layer file missing": (lambda r: (r / "features/mean/train/L05.npy").unlink(), "L05.npy: missing"),
    "layer renamed": (lambda r: _edit_json(r / "manifest.json", lambda d: d["layer_names"].__setitem__(33, "block32_raw")),
                      "layer_names differ"),
    "34th layer dropped": (lambda r: _edit_json(r / "manifest.json", lambda d: d.update(num_hidden_states=33)),
                           "num_hidden_states"),
    "features float32": (lambda r: _resave(r / "features/last/seen_val/L10.npy", lambda a: a.astype(np.float32)), "dtype"),
    "feature dim changed": (lambda r: _resave(r / "features/first/train/L20.npy", lambda a: a[:, :-1]), "shape"),
    "pooling dropped": (lambda r: _edit_json(r / "manifest.json", lambda d: d.update(pools_exported=["mean"])),
                        "pools_exported"),
    "column renamed": (lambda r: _edit_csv(r / "episodes.csv", lambda df: df.rename(columns={"task_min_step": "horizon"})),
                       "columns"),
    "row range shifted": (lambda r: _edit_csv(r / "episodes.csv", lambda df: df.assign(row_end=df.row_end + (df.index == 3))),
                          "row range"),
    "label flipped": (lambda r: _edit_csv(r / "episodes.csv", lambda df: df.assign(failure=df.success)), "1 - success"),
    "horizon wrong": (lambda r: _edit_csv(r / "episodes.csv", lambda df: df.assign(task_min_step=df.length)),
                      "task_min_step"),
    "split overlap": (lambda r: _edit_json(r / "splits.json", lambda d: d["trajectories"]["seen_val"].append(
        d["trajectories"]["train"][0])), "in both"),
    "unseen tasks changed": (lambda r: _edit_csv(r / "episodes.csv", lambda df: df.assign(
        split=np.where(df.task_id == 1, "unseen_test", df.split))), "unseen tasks"),
    "stats fit beyond train": (lambda r: _patch_n_rows(r), "train only"),
}


def _patch_n_rows(root):
    p = root / "norm_stats" / "mean.npz"
    z = dict(np.load(p))
    z["n_rows"] = np.array(int(z["n_rows"]) + 7)
    np.savez(p, **z)


@pytest.mark.parametrize("name", list(DRIFTS))
def test_drift_is_caught(broken, name):
    mutate, expected = DRIFTS[name]
    mutate(broken)
    problems = problems_of(broken)
    assert problems, f"{name}: not detected"
    assert any(expected in p for p in problems), f"{name}: got {problems}"


def test_withheld_unseen_is_a_note_not_a_problem(broken):
    for p in C.POOLS:
        shutil.rmtree(broken / "features" / p / "unseen_test")
    problems, notes = C.validate_cache(broken)
    assert problems == [] and "withheld" in notes[0]


# ---- the checker itself ------------------------------------------------------------------------------------------
def test_checker_is_standard_library_only():
    """It must run on a CARC login node, where numpy may not be installed."""
    tree = ast.parse((REPO / "failure_prob" / "data" / "contract.py").read_text(), feature_version=(3, 6))
    mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert mods <= {"argparse", "ast", "csv", "json", "re", "sys", "zipfile", "pathlib", "typing"}, mods


def test_cli(cache, broken):
    script = REPO / "failure_prob" / "data" / "contract.py"
    ok = subprocess.run([sys.executable, "-I", str(script), str(cache)], capture_output=True, text=True)
    assert ok.returncode == 0 and "CONTRACT OK" in ok.stdout
    real = subprocess.run([sys.executable, "-I", str(script), str(cache), "--expect-real"], capture_output=True, text=True)
    assert real.returncode == 1 and "10 tasks x 50" in real.stdout
    (broken / "features/mean/train/L05.npy").unlink()
    bad = subprocess.run([sys.executable, "-I", str(script), str(broken)], capture_output=True, text=True)
    assert bad.returncode == 1 and "CONTRACT FAILED" in bad.stdout
