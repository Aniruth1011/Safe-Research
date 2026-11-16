"""scripts/select_layers.py (S1-S6) on a fake depth-sweep folder in the sweep's real file format."""
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from failure_prob.engine.episode_metrics import episode_metrics

REPO = Path(__file__).resolve().parents[1]
N_EPS, SEEDS = 140, (0, 1, 2)


def strength(layer):
    if layer == "avg":
        return 0.9
    if layer in (30, 31):
        return 1.3
    if 23 <= layer <= 29 or layer in (32, 33):
        return 1.0
    return 0.5


def make_sweep(root: Path) -> Path:
    """rows/L{LL}_seed{S}.scores.npz exactly as failure_prob/engine/depth_sweep.py::_save_eval_input writes them."""
    rng = np.random.default_rng(0)
    y = (rng.random(N_EPS) < 0.5).astype(int)
    tasks = np.repeat(np.arange(7), 20)
    lengths = np.where(y == 1, 520, rng.integers(150, 400, N_EPS))
    horizons = np.minimum(lengths, 140)
    rows = root / "rows"
    rows.mkdir(parents=True)
    for layer in list(range(34)) + ["avg"]:
        for s in SEEDS:
            r = np.random.default_rng([s, 99 if layer == "avg" else layer])
            steps = [r.normal(size=n) * 0.6 + np.linspace(0, strength(layer) * y[i] + r.normal(0, 0.5), n)
                     for i, n in enumerate(lengths)]
            stem = "Lavg" if layer == "avg" else f"L{layer:02d}"
            np.savez_compressed(rows / f"{stem}_seed{s}.scores.npz",
                                step_scores=np.concatenate(steps).astype(np.float32),
                                offsets=np.concatenate([[0], np.cumsum(lengths)]), labels=y.astype(np.float32),
                                horizons=horizons, task_ids=tasks,
                                trajectory_ids=np.array([f"libero_10/t{t:02d}/e{i:02d}" for i, t in enumerate(tasks)]))
    return root


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    sweep = make_sweep(tmp_path_factory.mktemp("sweep"))
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "select_layers.py"), "--sweep", str(sweep)],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    with open(sweep / "rhea_S1_S6.csv", newline="") as f:
        rows = {row["Task ID"] or row["S.No"]: row for row in csv.DictReader(f)}
    return sweep, r.stdout, rows


def layer_roc(sweep, layer):
    vals = []
    for s in SEEDS:
        z = np.load(sweep / "rows" / f"L{layer:02d}_seed{s}.scores.npz")
        off = z["offsets"]
        scores = [z["step_scores"][off[i]:off[i + 1]] for i in range(len(off) - 1)]
        vals.append(episode_metrics(scores, z["labels"].astype(int), z["horizons"])["roc_auc"])
    return np.mean(vals)


def test_all_rows_written(result):
    _, out, rows = result
    assert {"S1", "S2", "S3", "S4", "S5", "S6", "ref"} <= set(rows)
    assert "=== Message for Dhyan ===" in out and "Best single layer" in out


def test_s1_is_top_two_candidates_and_excludes_l0_l32(result):
    sweep, _, rows = result
    cands = list(range(1, 32)) + [33]
    top2 = sorted(cands, key=lambda l: -layer_roc(sweep, l))[:2]
    assert rows["S1"]["Layers"] == f"L{top2[0]} + L{top2[1]}"
    assert "L32" not in rows["S1"]["Layers"] and "L0 " not in rows["S1"]["Layers"]


def test_per_layer_numbers_match_b4_evaluator(result):
    sweep, _, rows = result
    best = rows["ref"]["Layers"]
    l = int(best[1:])
    assert rows["ref"]["Seen Val ROC-AUC (%)"].startswith(f"{100 * layer_roc(sweep, l):.1f}")


def test_s3_one_layer_per_depth_band(result):
    _, _, rows = result
    layers = [int(x.strip()[1:]) for x in rows["S3"]["Layers"].split("+")]
    assert len(layers) == 3
    assert 1 <= layers[0] <= 11 and 12 <= layers[1] <= 22 and (23 <= layers[2] <= 31 or layers[2] == 33)


def test_s4_starts_from_best_and_s5_is_the_avg_pseudo_layer(result):
    _, _, rows = result
    assert rows["S4"]["Layers"].split(" + ")[0] == rows["ref"]["Layers"]
    assert rows["S5"]["Layers"].startswith("avg_z")


def test_rejects_folder_without_rows(tmp_path):
    (tmp_path / "rows").mkdir()
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "select_layers.py"), "--sweep", str(tmp_path)],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode != 0 and "no rows" in (r.stderr + r.stdout)
