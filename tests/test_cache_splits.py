"""B1: train.py / evaluate.py read their episodes from the cache's splits.json and score seen_val with B4."""
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from failure_prob.data.cache_splits import SplitError, horizons, load_split_records, load_splits
from failure_prob.engine.scoring import EVALUATOR_NAME

from test_shared_cache import KIT, REPO, run, write_rollouts


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("b1")
    raw = tmp / "raw"
    write_rollouts(raw)                                   # 10 tasks x 4 episodes, the real rollout format
    r = run(KIT / "build_cache.py", "--rollout-dir", raw, "--cache-root", tmp / "c", "--expect-episodes", 4)
    assert r.returncode == 0, r.stderr
    c = tmp / "c" / "v1"
    shutil.rmtree(c / "features" / "mean" / "unseen_test")   # anything that touched unseen data would now fail
    return c


def _copy(cache, tmp_path):
    dst = tmp_path / "v1"
    shutil.copytree(cache, dst)
    return dst


def test_load_splits_matches_cache(cache):
    s = load_splits(cache)
    assert sum(len(v) for v in s.values()) == 40
    eps = pd.read_csv(cache / "episodes.csv")
    assert set(eps[eps.split == "unseen_test"].task_id) == {0, 3, 5}
    assert s["seen_val"] == json.loads((cache / "splits.json").read_text())["trajectories"]["seen_val"]


def test_episodes_csv_must_agree_with_splits_json(cache, tmp_path):
    c = _copy(cache, tmp_path)
    eps = pd.read_csv(c / "episodes.csv")
    i = eps.index[eps.split == "train"][0]
    eps.loc[i, "split"] = "seen_val"
    eps.to_csv(c / "episodes.csv", index=False)
    with pytest.raises(SplitError, match="disagrees with splits.json"):
        load_splits(c)


def test_overlapping_splits_rejected(cache, tmp_path):
    c = _copy(cache, tmp_path)
    doc = json.loads((c / "splits.json").read_text())
    doc["trajectories"]["seen_val"].append(doc["trajectories"]["train"][0])
    (c / "splits.json").write_text(json.dumps(doc))
    with pytest.raises(SplitError, match="in both"):
        load_splits(c)


def test_records_follow_splits_json_and_carry_horizon(cache):
    order = load_splits(cache)["seen_val"]
    recs = load_split_records(cache, "seen_val", [4, 32])
    assert [r["trajectory_ids"][0] for r in recs] == order
    eps = pd.read_csv(cache / "episodes.csv").set_index("trajectory_id")
    for r in recs:
        t = r["trajectory_ids"][0]
        assert r["task_min_steps"] == [int(eps.loc[t, "task_min_step"])]
        assert r["features"].shape[1] == eps.loc[t, "length"] and r["features"].shape[2] == 2
    assert (horizons(recs) == [min(eps.loc[t, "length"], eps.loc[t, "task_min_step"]) for t in order]).all()


def test_unseen_is_locked(cache):
    with pytest.raises(PermissionError, match="locked"):
        load_split_records(cache, "unseen_test", [4])


def _train(args, cwd):
    return subprocess.run([sys.executable, str(REPO / "scripts" / "train.py"), *map(str, args)],
                          capture_output=True, text=True, cwd=cwd)


def _index_rows(cwd):
    with open(cwd / "experiments" / "index.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_train_on_train_score_seen_val_then_evaluate(cache, tmp_path):
    r = _train(["--model", "single_layer_mlp", "--layers", 4, "--cache", cache, "--seeds", 0, "--epochs", 2,
                "--output-root", tmp_path / "runs"], tmp_path)
    assert r.returncode == 0, r.stderr
    run_dir = Path(r.stdout.strip().splitlines()[-1])
    m = json.loads((run_dir / "metrics.json").read_text())
    assert m["eval_split"] == "seen_val" and m["metric_level"] == "episode" and m["evaluator"] == EVALUATOR_NAME
    assert m["n_episodes"] == len(load_splits(cache)["seen_val"])
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["train_episodes"] == len(load_splits(cache)["train"]) and cfg["splits_json_sha256"]
    row = _index_rows(tmp_path)[-1]
    assert row["val_roc_auc"] == str(m["roc_auc"]) and row["split_version"] == "v1"

    ev = subprocess.run([sys.executable, str(REPO / "scripts" / "evaluate.py"), "--checkpoint", str(run_dir / "best.pt")],
                        capture_output=True, text=True, cwd=tmp_path)
    assert ev.returncode == 0, ev.stderr
    out = json.loads(ev.stdout)
    assert out["split"] == "seen_val" and out["roc_auc"] == pytest.approx(m["roc_auc"], abs=1e-12)

    locked = subprocess.run([sys.executable, str(REPO / "scripts" / "evaluate.py"), "--checkpoint",
                             str(run_dir / "best.pt"), "--split", "unseen_test"], capture_output=True, text=True, cwd=tmp_path)
    assert locked.returncode != 0 and "locked" in locked.stderr


def test_legacy_modes_are_not_reported_as_validation(tmp_path):
    r = _train(["--model", "single_layer_mlp", "--layers", 4, "--synthetic", "--seeds", 0,
                "--output-root", tmp_path / "runs"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert "NOT validation" in r.stderr
    m = json.loads((Path(r.stdout.strip().splitlines()[-1]) / "metrics.json").read_text())
    assert m["eval_split"].startswith("train (NOT validation")
    assert _index_rows(tmp_path)[-1]["val_roc_auc"] == ""


def test_cli_rejects_mixed_or_missing_sources(cache, tmp_path):
    assert _train(["--model", "single_layer_mlp", "--layers", 4], tmp_path).returncode != 0
    assert _train(["--model", "single_layer_mlp", "--layers", 4, "--cache", cache, "--synthetic"], tmp_path).returncode != 0
    assert _train(["--model", "single_layer_mlp", "--layers", 4, "--synthetic", "--normalize"], tmp_path).returncode != 0
