"""F1/F2/F3 hyper-parameter search and F6 K-ablation drivers (synthetic 34-layer cache, no GPU)."""
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

import pandas as pd
import pytest

from test_fusion_runner import cache, cfg  # noqa: F401  (module-scoped cache fixture)
from test_shared_cache import REPO
import load_cache
from failure_prob import fusion_experiments as FX
from failure_prob import fusion_runner as FR


def count_loads(monkeypatch):
    n = {"loads": 0, "splits": []}
    orig = FR.load_cache_splits

    def spy(c):
        n["loads"] += 1
        return orig(c)
    monkeypatch.setattr(FR, "load_cache_splits", spy)
    orig_arr = load_cache.SharedCache._array
    monkeypatch.setattr(load_cache.SharedCache, "_array",
                        lambda self, pool, split, layer: (n["splits"].append(split), orig_arr(self, pool, split, layer))[1])
    return n


def test_search_loads_once_runs_every_point_and_picks_best(cache, tmp_path, monkeypatch):  # noqa: F811
    n = count_loads(monkeypatch)
    out = tmp_path / "F1"
    res = FX.hparam_search(cfg(out, model="concat_fusion", cache=str(cache), seeds=[0, 1], epochs=1,
                               save_checkpoints=False),
                           grid={"lr": [1e-3, 3e-4], "dropout": [0.0, 0.2]}, label="F1 two-layer concat")
    assert n["loads"] == 1 and set(n["splits"]) == {"train", "seen_val"}
    table = pd.read_csv(out / "search.csv")
    assert len(table) == 4 and table.selected.sum() == 1
    assert table.roc_auc_mean.max() == pytest.approx(table[table.selected].roc_auc_mean.iloc[0])
    best = json.loads((out / "best.json").read_text())
    assert best["selected"] == res["selected"] and best["n_grid_points"] == 4
    assert "selection on seen_val" in best["sheet_row"]["Notes"] and best["sheet_row"]["Experiment"] == "F1 two-layer concat"
    for run in table.run:
        assert (pd.read_csv(f"{run}/sheet_row.csv").shape[0] == 1)
    assert list(pd.read_csv(out / "sheet_rows.csv").columns) == FR.SHEET_COLUMNS


def test_search_refuses_foreign_folder_and_bad_grid(tmp_path):
    foreign = tmp_path / "mine"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        FX.hparam_search(cfg(foreign, synthetic=True, seeds=[0], epochs=1, overwrite=True), grid={"lr": [1e-3]})
    assert (foreign / "notes.txt").exists()
    with pytest.raises(ValueError, match="cannot search"):
        FX.hparam_search(cfg(tmp_path / "a", synthetic=True), grid={"batch_size": [4]})
    with pytest.raises(ValueError, match="unique"):
        FX.hparam_search(cfg(tmp_path / "b", synthetic=True), grid={"lr": [1e-3, 1e-3]})
    with pytest.raises(ValueError, match="lr > 0"):
        FX.hparam_search(cfg(tmp_path / "c", synthetic=True), grid={"lr": [0.0]})


def test_search_rerun_with_overwrite(tmp_path):
    out = tmp_path / "s"
    FX.hparam_search(cfg(out, synthetic=True, seeds=[0], epochs=1), grid={"lr": [1e-3]})
    with pytest.raises(FileExistsError):
        FX.hparam_search(cfg(out, synthetic=True, seeds=[0], epochs=1), grid={"lr": [1e-3]})
    FX.hparam_search(cfg(out, synthetic=True, seeds=[0], epochs=1, overwrite=True), grid={"lr": [1e-3]})


def test_k_ablation_uses_prefixes_of_the_ranked_layers(cache, tmp_path, monkeypatch):  # noqa: F811
    n = count_loads(monkeypatch)
    out = tmp_path / "F6"
    rows = FX.k_ablation(cfg(out, model="static_weight_fusion", proj_dim=8, layers=[16, 33, 4], cache=str(cache),
                             seeds=[0], epochs=1, save_checkpoints=False), ks=[1, 2, 3])
    assert n["loads"] == 1
    curve = pd.read_csv(out / "k_curve.csv")
    assert curve.K.tolist() == [1, 2, 3] and curve.layers.tolist() == ["16", "16/33", "16/33/4"]
    assert curve.delta_vs_K1_roc_auc_pp.iloc[0] == 0
    assert curve.Parameters.is_monotonic_increasing  # one projection per layer
    for k, names in [(1, ["block16"]), (3, ["block16", "block32_prenorm", "block04"])]:
        conf = json.loads((out / f"K{k}" / "config.json").read_text())
        assert conf["config"]["layers"] == [16, 33, 4][:k] and conf["provenance"]["layer_names"] == names


def test_k_ablation_validation(tmp_path):
    with pytest.raises(ValueError, match="ks must lie"):
        FX.k_ablation(cfg(tmp_path / "a", synthetic=True, layers=[1, 2]), ks=[1, 2, 3])
    with pytest.raises(ValueError, match="ks must lie"):
        FX.k_ablation(cfg(tmp_path / "b", synthetic=True, layers=[1, 2]), ks=[0, 1])
    with pytest.raises(ValueError, match="init_weights"):
        FX.k_ablation(cfg(tmp_path / "c", synthetic=True, layers=[1, 2], init_weights=[1, 1]), ks=[1, 2])
    with pytest.raises(ValueError, match="duplicate"):
        FX.k_ablation(cfg(tmp_path / "d", synthetic=True, layers=[1, 1]), ks=[1, 2])


def test_first_k_is_a_view():
    import torch
    rec = {"features": torch.zeros(1, 5, 4, 3), "mask": torch.ones(1, 5, dtype=torch.bool)}
    sub = FX.first_k({"train": [rec]}, 2)["train"][0]["features"]
    assert sub.shape == (1, 5, 2, 3) and sub.data_ptr() == rec["features"].data_ptr()


def test_cli_search_and_kcurve(cache, tmp_path):  # noqa: F811
    script = REPO / "scripts" / "fusion_experiments.py"
    r = subprocess.run([sys.executable, str(script), "search", "--model", "concat_fusion", "--layers", "16", "33",
                        "--cache", str(cache), "--output-dir", str(tmp_path / "s"), "--epochs", "1", "--seeds", "0",
                        "--lrs", "1e-3", "--dropouts", "0", "--weight-decays", "0", "--no-checkpoints"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "s" / "best.json").exists()
    r = subprocess.run([sys.executable, str(script), "kcurve", "--model", "concat_fusion", "--layers", "16", "33",
                        "--ks", "1", "2", "3", "--cache", str(cache), "--output-dir", str(tmp_path / "k")],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "ks must lie" in r.stderr


def test_search_computes_ci_only_for_the_winner_and_resumes(cache, tmp_path, monkeypatch):  # noqa: F811
    out = tmp_path / "S"
    base = cfg(out, model="concat_fusion", cache=str(cache), seeds=[0], epochs=1, save_checkpoints=False,
               bootstrap=100)
    FX.hparam_search(base, grid={"lr": [1e-3, 3e-4]})
    best = json.loads((out / "best.json").read_text())
    summaries = {r: json.loads((Path(r) / "summary.json").read_text()) for r in pd.read_csv(out / "search.csv").run}
    assert [("ci" in s["metrics"]["roc_auc"]) for r, s in summaries.items()].count(True) == 1
    assert "ci" in summaries[best["run"]]["metrics"]["roc_auc"] and "CI [" in best["sheet_row"]["Notes"]
    n = count_loads(monkeypatch)
    FX.hparam_search(replace(base, resume=True), grid={"lr": [1e-3, 3e-4]})  # all finished: nothing loaded
    assert n["loads"] == 0
    with pytest.raises(FileExistsError, match="not a k_ablation"):
        FX.k_ablation(replace(base, resume=True, layers=[16, 33]), ks=[1, 2])


def test_k_ablation_reference_layer(cache, tmp_path):  # noqa: F811
    out = tmp_path / "F6ref"
    rows = FX.k_ablation(cfg(out, model="concat_fusion", layers=[16, 33], cache=str(cache), seeds=[0], epochs=1,
                             save_checkpoints=False, bootstrap=0), ks=[1, 2], reference_layer=4)
    curve = pd.read_csv(out / "k_curve.csv")
    assert curve.run.tolist() == ["K1", "K2", "ref_L4"] and (out / "ref_L4" / "summary.json").exists()
    ref = curve.set_index("run").loc["ref_L4"]
    assert ref.layers == "4" and ref.delta_vs_best_single_roc_auc_pp == 0
    conf = json.loads((out / "ref_L4" / "config.json").read_text())
    assert conf["config"]["layers"] == [4] and conf["provenance"]["layer_names"] == ["block04"]
    with pytest.raises(ValueError, match="already in the ranked"):
        FX.k_ablation(cfg(tmp_path / "bad", synthetic=True, layers=[1, 2, 3]), ks=[1, 2, 3], reference_layer=2)
    rows = FX.k_ablation(cfg(tmp_path / "same", synthetic=True, layers=[1, 2], seeds=[0], epochs=1, bootstrap=0),
                         ks=[1, 2], reference_layer=1)  # reference == first ranked: no extra run
    assert [r["run"] for r in rows] == ["K1", "K2"]


@pytest.mark.parametrize("model", ["gated_fusion", "residual_fusion", "static_weight_fusion"])
def test_k_ablation_with_every_fusion_model(tmp_path, model):
    rows = FX.k_ablation(cfg(tmp_path / model, model=model, synthetic=True, layers=[1, 2, 3], seeds=[0], epochs=1,
                             bootstrap=0), ks=[1, 2, 3])
    assert [r["K"] for r in rows] == [1, 2, 3]


# --------------------------------------------------------------------------- paired comparison

def test_compare_fusion_run_with_sweep_style_files(cache, tmp_path):  # noqa: F811
    from failure_prob import fusion_compare as FC
    a = tmp_path / "a"
    FR.run(cfg(a, cache=str(cache), layers=[16, 33], seeds=[0, 1], epochs=1, bootstrap=0, save_checkpoints=False))
    # B: written like the depth sweep (probabilities over the FULL rollout, horizons = min(length, task_min_step))
    b = FR.load_step_scores(a / "seed0" / FR.STEP_SCORES)
    sweep = tmp_path / "sweep" / "rows"
    sweep.mkdir(parents=True)
    for s in (0, 1):
        full = [np.concatenate([1 / (1 + np.exp(-np.clip(x, -50, 50))), np.full(3, 0.99)]) for x in b.step_scores]  # padded tail
        lengths = np.array([len(x) for x in full])
        np.savez(sweep / f"L16_seed{s}.scores.npz", step_scores=np.concatenate(full).astype(np.float32),
                 offsets=np.concatenate([[0], np.cumsum(lengths)]), labels=b.labels.astype(np.float32),
                 horizons=b.horizons, task_ids=b.task_ids, trajectory_ids=np.array(b.trajectory_ids))
    side_a, _ = FC.load_side(str(a))
    side_b, _ = FC.load_side(str(sweep / "L16_seed*.scores.npz"))
    res = FC.compare(side_a, side_b, FR.Scorer(), n=100)
    m = res["metrics"]["roc_auc"]
    assert m["ci"][0] <= m["delta"] <= m["ci"][1] and 0 <= m["p_a_not_better"] <= 1
    same = FC.compare(side_a, side_a, FR.Scorer(), n=50)["metrics"]["roc_auc"]
    assert same["delta"] == 0 and same["ci"] == [0.0, 0.0] and same["p_a_not_better"] == 1.0
    other = FR.EvalInput(b.step_scores[:-1], b.labels[:-1], b.horizons[:-1], b.task_ids[:-1], b.trajectory_ids[:-1])
    with pytest.raises(ValueError, match="different"):
        FC.compare(side_a, [other], FR.Scorer(), n=10)


def test_compare_cli(cache, tmp_path):  # noqa: F811
    a = tmp_path / "a"
    FR.run(cfg(a, cache=str(cache), layers=[16], seeds=[0], epochs=1, bootstrap=0, save_checkpoints=False))
    script = REPO / "scripts" / "fusion_compare.py"
    r = subprocess.run([sys.executable, str(script), "--a", str(a), "--b", str(a), "--n", "20", "--out",
                        str(tmp_path / "c.json")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "delta  +0.00 pp" in r.stdout and json.loads((tmp_path / "c.json").read_text())["n_resamples"] == 20
    r = subprocess.run([sys.executable, str(script), "--a", str(a), "--b", str(tmp_path / "none*.npz")],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "no score files" in r.stderr


def test_slurm_script_syntax_and_dispatch():
    slurm = REPO / "scripts" / "carc" / "fusion.slurm"
    assert subprocess.run(["bash", "-n", str(slurm)]).returncode == 0
    r = subprocess.run(["bash", str(slurm), "search"], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert r.returncode != 0 and "submit with sbatch" in r.stderr  # refuses to run outside a Slurm job
    text = slurm.read_text()
    assert "/home/" not in text and "/scratch1/${USER" in text and "PYTHONDONTWRITEBYTECODE=1" in text  # no personal paths
