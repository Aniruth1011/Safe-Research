"""Fusion runner (F1, F2, F3, F6) on train/seen_val. Uses a tiny 34-layer cache built by the real build_cache.py (no GPU)."""
import json
import math
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from test_kit_layer34 import build, rollouts34
from test_shared_cache import D, REPO
import load_cache
from failure_prob import fusion_runner as FR
from failure_prob.data.synthetic import make_synthetic_records
from failure_prob.protocol import TRAIN_SEEDS


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fusion_cache")
    rollouts34(tmp / "raw")
    r = build(tmp / "raw", tmp / "c")
    assert r.returncode == 0, r.stderr + r.stdout
    return tmp / "c" / "v1"


def cfg(out, **kw):
    base = dict(model="static_weight_fusion", layers=[16, 33], output_dir=str(out), epochs=2, batch_size=4,
                early_stopping_patience=0)  # the tiny test cache has 1 episode per task x outcome: no B6 holdout
    return FR.RunConfig(**(base | kw))


# --------------------------------------------------------------------------- end to end on the cache

def test_cache_run_end_to_end_and_never_reads_unseen(cache, tmp_path, monkeypatch):
    seen = []
    orig = load_cache.SharedCache._array
    monkeypatch.setattr(load_cache.SharedCache, "_array",
                        lambda self, pool, split, layer: (seen.append(split), orig(self, pool, split, layer))[1])
    out = tmp_path / "run"
    summary = FR.run(cfg(out, cache=str(cache)))
    assert set(seen) == {"train", "seen_val"}
    assert summary["seeds"] == list(TRAIN_SEEDS) and not summary["synthetic"]
    assert summary["split_label"] == "single fixed split (SAFE seed-0 unseen tasks)"
    assert len(summary["metrics"]["layer_weights"]["mean"]) == 2
    conf = json.loads((out / "config.json").read_text())
    assert conf["provenance"]["layer_names"] == ["block16", "block32_prenorm"]
    # default scorer is B4 (not provisional); the test config differs from B6 (2 epochs, no early stopping)
    assert "episode_scorer" not in conf["provisional"] and "B6" in conf["provisional"]["probe"]
    assert "epochs" in conf["provisional"]["probe"] and "early_stopping_patience" in conf["provisional"]["probe"]

    eps = pd.read_csv(cache / "episodes.csv")
    val = eps[eps.split == "seen_val"].set_index("trajectory_id")
    for s in TRAIN_SEEDS:
        m = json.loads((out / f"seed{s}" / "metrics.json").read_text())
        assert m["seen_val"]["n_episodes"] == len(val) and 0 <= m["seen_val"]["roc_auc"] <= 1
        assert set(m["seen_val_per_task"]) == {str(t) for t in val.task_id.unique()}
        scores = pd.read_csv(out / f"seed{s}" / "seen_val_episode_scores.csv")
        assert set(scores.trajectory_id) == set(val.index)
        assert (scores.n_steps.values == val.loc[scores.trajectory_id, "task_min_step"].values).all()  # horizon
        assert scores.prob.between(0, 1).all() and np.isfinite(scores.score).all()
        assert not set(scores.trajectory_id) & set(eps[eps.split == "unseen_test"].trajectory_id)
        ck = torch.load(out / f"seed{s}" / "model.pt", weights_only=False)
        assert ck["seed"] == s and ck["model_kwargs"]["num_layers"] == 2


def test_same_seed_is_reproducible(cache, tmp_path):
    a = FR.run(cfg(tmp_path / "a", cache=str(cache), seeds=[1], save_checkpoints=False))
    b = FR.run(cfg(tmp_path / "b", cache=str(cache), seeds=[1], save_checkpoints=False))
    assert a["metrics"]["roc_auc"]["per_seed"] == b["metrics"]["roc_auc"]["per_seed"]
    sa = pd.read_csv(tmp_path / "a" / "seed1" / "seen_val_episode_scores.csv")
    sb = pd.read_csv(tmp_path / "b" / "seed1" / "seen_val_episode_scores.csv")
    assert np.allclose(sa.score, sb.score)


@pytest.mark.parametrize("model, extra", [("concat_fusion", {"head": "linear"}),
                                          ("static_weight_fusion", {"learn_weights": False}),
                                          ("single_layer_mlp", {})])
def test_other_models_and_options(cache, tmp_path, model, extra):
    layers = [33] if model == "single_layer_mlp" else [0, 16, 33]
    s = FR.run(cfg(tmp_path / "r", cache=str(cache), model=model, layers=layers, seeds=[0],
                   select_best_epoch=True, aggregation="mean", save_checkpoints=False, **extra))
    m = json.loads((tmp_path / "r" / "seed0" / "metrics.json").read_text())
    assert m["selection"] == "best seen_val ROC-AUC" and 1 <= m["chosen_epoch"] <= 2
    if extra.get("learn_weights") is False:
        assert np.allclose(m["layer_weights"], 1 / 3)  # frozen uniform mix
    assert not math.isnan(s["metrics"]["roc_auc"]["mean"])


def test_full_horizon_is_longer_and_warned(cache, tmp_path, capsys):
    FR.run(cfg(tmp_path / "f", cache=str(cache), horizon="full", seeds=[0], save_checkpoints=False))
    assert "not reportable" in capsys.readouterr().err
    scores = pd.read_csv(tmp_path / "f" / "seed0" / "seen_val_episode_scores.csv")
    eps = pd.read_csv(cache / "episodes.csv").set_index("trajectory_id")
    assert (scores.n_steps.values == eps.loc[scores.trajectory_id, "length"].values).all()


def test_cli_on_cache(cache, tmp_path):
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "run_fusion.py"), "--model", "concat_fusion",
                        "--layers", "4", "33", "--cache", str(cache), "--output-dir", str(tmp_path / "cli"),
                        "--epochs", "1", "--seeds", "2", "--pool", "last", "--no-normalize",
                        "--early-stopping-patience", "0"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    conf = json.loads((tmp_path / "cli" / "config.json").read_text())["config"]
    assert conf["pool"] == "last" and conf["normalize"] is False and conf["seeds"] == [2]


# --------------------------------------------------------------------------- refusals

@pytest.mark.parametrize("kw, msg", [
    ({"layers": [16, 16]}, "duplicate layers"),
    ({"layers": []}, "at least one layer"),
    ({"layers": [34]}, "not in this cache"),
    ({"layers": [-1]}, "not in this cache"),
    ({"horizon": "task-min-step"}, "horizon must be one of"),
    ({"aggregation": "median"}, "aggregation must be one of"),
    ({"seeds": [0, 0]}, "seeds must be"),
    ({"model": "nope"}, "Unknown model"),
    ({"epochs": 0}, "epochs must be"),
    ({"init_weights": [1.0]}, "init_weights has 1 values"),
])
def test_bad_config_refused(cache, tmp_path, kw, msg):
    with pytest.raises(ValueError, match=msg):
        FR.run(cfg(tmp_path / "x", cache=str(cache), **kw))


def test_needs_exactly_one_source(tmp_path):
    with pytest.raises(ValueError, match="exactly one data source"):
        FR.run(cfg(tmp_path / "x"))
    with pytest.raises(ValueError, match="exactly one data source"):
        FR.run(cfg(tmp_path / "x", cache="c", synthetic=True))


def test_refuses_to_overwrite_results(tmp_path):
    out = tmp_path / "o"
    FR.run(cfg(out, synthetic=True, seeds=[0], epochs=1))
    with pytest.raises(FileExistsError, match="overwrite"):
        FR.run(cfg(out, synthetic=True, seeds=[0], epochs=1))
    FR.run(cfg(out, synthetic=True, seeds=[1], epochs=1, overwrite=True))
    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == ["seed1"]  # stale seed0 removed


def test_overwrite_only_deletes_our_own_run_folders(tmp_path, monkeypatch):
    foreign = tmp_path / "notes"
    foreign.mkdir()
    (foreign / "thesis.tex").write_text("keep me")
    with pytest.raises(FileExistsError, match="refusing to delete"):
        FR.run(cfg(foreign, synthetic=True, seeds=[0], epochs=1, overwrite=True))
    assert (foreign / "thesis.tex").read_text() == "keep me"

    out = tmp_path / "run"
    FR.run(cfg(out, synthetic=True, seeds=[0], epochs=1))
    (out / "seed0" / "my_analysis.ipynb").write_text("{}")  # inside a seed folder: still ours to replace
    (out / "plot.png").write_bytes(b"x")  # user file next to our outputs -> refuse
    with pytest.raises(FileExistsError, match="refusing to delete"):
        FR.run(cfg(out, synthetic=True, seeds=[0], epochs=1, overwrite=True))
    assert (out / "plot.png").exists()

    monkeypatch.chdir(out)  # cwd is protected even if it is a run folder
    (out / "plot.png").unlink()
    with pytest.raises(FileExistsError, match="refusing to delete"):
        FR.run(cfg(".", synthetic=True, seeds=[0], epochs=1, overwrite=True))
    assert (out / "config.json").exists()


def test_output_path_that_is_a_file_is_refused(tmp_path):
    f = tmp_path / "out.txt"
    f.write_text("x")
    with pytest.raises(FileExistsError, match="not an output folder"):
        FR.run(cfg(f, synthetic=True, seeds=[0], epochs=1, overwrite=True))


def test_synthetic_summary_does_not_claim_a_horizon(tmp_path):
    s = FR.run(cfg(tmp_path / "s", synthetic=True, seeds=[0], epochs=1))
    assert s["synthetic"] and s["horizon"] == "n/a (synthetic)"


def test_normalization_blowup_is_warned(cache, tmp_path, capsys, monkeypatch):
    """A dimension with ~0 train std (A9 note: std floor 1e-6) blows seen_val up by ~1e6; the runner must say so."""
    orig = load_cache.SharedCache.norm_stats

    def tiny_std(self, pool, layers):
        mean, std = orig(self, pool, layers)
        std = std.copy()
        std[:, 0] = 1e-6
        return mean, std
    # layer 33 has no ~0-std dimension in this fixture (layer 16 does: bf16 rounding makes some train dims constant,
    # and seen_val then reaches ~2e6 - exactly the A9 std-floor note), so it is the clean control
    run33 = dict(layers=[33], seeds=[0], epochs=1, save_checkpoints=False, cache=str(cache))
    FR.run(cfg(tmp_path / "ok", **run33))
    assert "~0 train std" not in capsys.readouterr().err
    monkeypatch.setattr(load_cache.SharedCache, "norm_stats", tiny_std)
    FR.run(cfg(tmp_path / "w", **run33))
    assert "~0 train std" in capsys.readouterr().err
    monkeypatch.setattr(load_cache.SharedCache, "norm_stats", orig)
    FR.run(cfg(tmp_path / "real", cache=str(cache), layers=[16], seeds=[0], epochs=1, save_checkpoints=False))
    assert "~0 train std" in capsys.readouterr().err  # the real near-zero-std dims in layer 16 are caught too


def _recs(prefix, n=6, k=2, d=4, seed=0):
    rs = make_synthetic_records(n=n, k=k, d=d, seed=seed)
    for r in rs:
        r["trajectory_ids"] = [f"{prefix}{r['trajectory_ids'][0]}"]
    return rs


def test_split_checks():
    ok = {"train": _recs("a/"), "seen_val": _recs("b/")}
    assert FR.check_splits(ok, 2) == 4
    with pytest.raises(ValueError, match="share trajectories"):
        FR.check_splits({"train": _recs("a/"), "seen_val": _recs("a/")}, 2)
    with pytest.raises(ValueError, match="expected exactly the splits"):
        FR.check_splits({"train": ok["train"], "unseen_test": ok["seen_val"]}, 2)
    with pytest.raises(ValueError, match="no episodes"):
        FR.check_splits({"train": ok["train"], "seen_val": []}, 2)
    with pytest.raises(ValueError, match=r"expected \[1,T,3,D\]"):
        FR.check_splits(ok, 3)
    dup = _recs("b/")
    dup[1]["trajectory_ids"] = dup[0]["trajectory_ids"]
    with pytest.raises(ValueError, match="duplicate trajectory"):
        FR.check_splits({"train": ok["train"], "seen_val": dup}, 2)
    bad = _recs("b/")
    bad[0]["features"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        FR.check_splits({"train": ok["train"], "seen_val": bad}, 2)
    mixed = _recs("b/", d=5)
    with pytest.raises(ValueError, match="inconsistent feature dims"):
        FR.check_splits({"train": ok["train"], "seen_val": mixed}, 2)


# --------------------------------------------------------------------------- scoring

def test_aggregation_in_logit_space():
    x = torch.tensor([-2.0, 3.0, 0.0])
    assert FR.aggregate(x, "max") == pytest.approx(3.0)
    assert FR.aggregate(x, "last") == pytest.approx(0.0)
    assert FR.to_prob(FR.aggregate(x, "mean")) == pytest.approx(torch.sigmoid(x.double()).mean().item())
    assert FR.aggregate(torch.tensor([5.0]), "mean") == pytest.approx(5.0)  # one step: mean == that step
    with pytest.raises(ValueError, match="no valid timesteps"):
        FR.aggregate(torch.tensor([]), "max")
    with pytest.raises(ValueError, match="non-finite"):
        FR.aggregate(torch.tensor([1.0, float("inf")]), "max")


@pytest.mark.parametrize("how", ["max", "last", "mean"])
def test_confident_episodes_do_not_tie(how):
    """sigmoid saturates to 1.0 (float32 at x>~17, float64 at x>~37); scores must still rank 20 < 40 < 80."""
    s = [FR.aggregate(torch.tensor([v, v]), how) for v in (20.0, 40.0, 80.0)]
    assert s[0] < s[1] < s[2] and all(np.isfinite(s))
    assert FR.episode_metrics([0, 1], s[:2])["roc_auc"] == 1.0
    lo = [FR.aggregate(torch.tensor([v, v]), how) for v in (-80.0, -40.0)]
    assert lo[0] < lo[1]


def test_episode_scores_ignore_padding():
    """A padded step with a huge logit must not raise an episode's max score."""
    class Probe(torch.nn.Module):
        def forward(self, batch):
            from failure_prob.models.base import Output
            return Output(batch["features"][:, :, 0, 0])
    long_ = {"features": torch.zeros(1, 5, 1, 1), "mask": torch.ones(1, 5, dtype=torch.bool),
             "labels": torch.tensor([1.0]), "task_ids": [1], "trajectory_ids": ["long"]}
    short = {"features": torch.full((1, 2, 1, 1), -4.0), "mask": torch.ones(1, 2, dtype=torch.bool),
             "labels": torch.tensor([0.0]), "task_ids": [1], "trajectory_ids": ["short"]}
    rows = FR.episode_scores(Probe(), [long_, short], batch_size=2, how="max")
    assert rows[1]["n_steps"] == 2 and rows[1]["score"] == pytest.approx(-4.0)
    assert rows[0]["score"] == pytest.approx(0.0) and rows[0]["prob"] == pytest.approx(0.5)


def test_episode_metrics_edge_cases():
    m = FR.episode_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0 and m["tpr_at_5_fpr"] == 1.0
    one = FR.episode_metrics([1, 1], [0.3, 0.4])
    assert math.isnan(one["roc_auc"]) and one["n_failures"] == 2
    tie = FR.episode_metrics([0, 1, 0, 1], [0.5] * 4)
    assert tie["roc_auc"] == 0.5
    with pytest.raises(ValueError, match="same length"):
        FR.episode_metrics([0, 1], [0.5])
    with pytest.raises(ValueError, match="non-finite"):
        FR.episode_metrics([0, 1], [0.5, float("nan")])


def test_summary_handles_nan_and_single_seed():
    per_seed = {0: {"seen_val": {"roc_auc": 0.7, "pr_auc": 0.6, "tpr_at_5_fpr": 0.1}},
                1: {"seen_val": {"roc_auc": float("nan"), "pr_auc": 0.5, "tpr_at_5_fpr": 0.0}}}
    s = FR.summarize(per_seed)
    assert s["roc_auc"]["mean"] == 0.7 and math.isnan(s["roc_auc"]["std"]) and s["roc_auc"]["n_seeds_defined"] == 1
    assert s["pr_auc"]["n_seeds_defined"] == 2
    assert s["pr_auc"]["std"] == pytest.approx(np.std([0.6, 0.5], ddof=1))


# --------------------------------------------------------------------------- protocol shared with the depth sweep

def test_train_full_horizon_seen_val_task_min_step_by_default(cache, monkeypatch, tmp_path):
    got = {}
    orig = FR.load_cache_splits

    def spy(c):
        splits, prov = orig(c)
        got.update(splits)
        return splits, prov
    monkeypatch.setattr(FR, "load_cache_splits", spy)
    FR.run(cfg(tmp_path / "r", cache=str(cache), seeds=[0], save_checkpoints=False))
    eps = pd.read_csv(cache / "episodes.csv").set_index("trajectory_id")
    for r in got["train"]:
        assert r["features"].shape[1] == eps.loc[r["trajectory_ids"][0], "length"]
    for r in got["seen_val"]:
        ep = eps.loc[r["trajectory_ids"][0]]
        assert r["features"].shape[1] == min(ep.length, ep.task_min_step)
    conf = json.loads((tmp_path / "r" / "config.json").read_text())
    assert conf["provenance"]["horizons"] == {"train": "full", "seen_val": "task_min_step"}
    assert json.loads((tmp_path / "r" / "summary.json").read_text())["train_horizon"] == "full"


def test_train_horizon_can_match_seen_val(cache, tmp_path):
    s = FR.run(cfg(tmp_path / "r", cache=str(cache), seeds=[0], train_horizon="task_min_step",
                   save_checkpoints=False))
    assert s["train_horizon"] == "task_min_step"
    with pytest.raises(ValueError, match="train_horizon"):
        FR.validate_config(cfg(tmp_path / "x", synthetic=True, train_horizon="min"))


# --------------------------------------------------------------------------- pluggable evaluator (B4)

class RecordingEvaluator:
    """Stands in for B4: same contract as the depth sweep's evaluators."""
    name = "test_eval"
    temporary = False
    calls = []

    def __call__(self, data):
        RecordingEvaluator.calls.append(data)
        scores = [float(np.max(s[:h])) for s, h in zip(data.step_scores, data.horizons)]
        return FR.episode_metrics(data.labels, scores) | {"extra_metric": 1.0}


def make_recording_evaluator():
    return RecordingEvaluator()


class NoAucEvaluator:
    name, temporary = "no_auc", True

    def __call__(self, data):
        return {"something": 0.5}


def test_plugged_in_evaluator_gets_eval_input_and_replaces_scorer(cache, tmp_path):
    RecordingEvaluator.calls.clear()
    out = tmp_path / "ev"
    s = FR.run(cfg(out, cache=str(cache), seeds=[0], save_checkpoints=False, bootstrap=0,
                   evaluator="test_fusion_runner:make_recording_evaluator"))
    assert "episode_scorer" not in s["provisional"] and "probe" in s["provisional"]  # B6 still pending
    assert s["evaluator"] == "test_eval" and "extra_metric" in s["metrics"]
    eps = pd.read_csv(cache / "episodes.csv").set_index("trajectory_id")
    n_val = int((eps.split == "seen_val").sum())
    val_calls = [d for d in RecordingEvaluator.calls if len(d.labels) == n_val
                 and set(eps.loc[d.trajectory_ids, "split"]) == {"seen_val"}]
    assert len(val_calls) == 2 + 1  # one per epoch (learning curve) + the final seen_val score
    data = val_calls[-1]
    for sc, h, tid in zip(data.step_scores, data.horizons, data.trajectory_ids):
        ep = eps.loc[tid]
        assert len(sc) == h == min(ep.length, ep.task_min_step) and ep.split == "seen_val"
    assert set(data.labels.tolist()) <= {0, 1} and len(data.task_ids) == len(data.labels)
    m = json.loads((out / "seed0" / "metrics.json").read_text())
    builtin = FR.run(cfg(tmp_path / "b", cache=str(cache), seeds=[0], save_checkpoints=False))
    assert m["seen_val"]["roc_auc"] == pytest.approx(builtin["metrics"]["roc_auc"]["mean"])  # same model, same max


@pytest.mark.parametrize("spec, msg", [("nope", "unknown evaluator"), ("no_such_module:f", "cannot load"),
                                       ("test_fusion_runner:missing", "cannot load")])
def test_bad_evaluator_spec_fails_before_loading(tmp_path, spec, msg):
    with pytest.raises(ValueError, match=msg):
        FR.run(cfg(tmp_path / "x", synthetic=True, seeds=[0], evaluator=spec))
    assert not (tmp_path / "x").exists()


def test_evaluator_without_roc_auc_is_refused(tmp_path):
    with pytest.raises(ValueError, match="roc_auc"):
        FR.run(cfg(tmp_path / "x", synthetic=True, seeds=[0], epochs=1,
                   evaluator="test_fusion_runner:NoAucEvaluator"))


# --------------------------------------------------------------------------- sheet row: percent, parameters, latency

def test_sheet_row_in_percent_with_parameters_and_latency(cache, tmp_path):
    out = tmp_path / "sheet"
    s = FR.run(cfg(out, cache=str(cache), seeds=[0, 1], save_checkpoints=False), label="F3 test")
    row = s["sheet_row"]
    assert row["Experiment"] == "F3 test"
    assert row["Seen Val ROC-AUC (%)"] == pytest.approx(100 * s["metrics"]["roc_auc"]["mean"], abs=0.006)
    assert 0 <= row["Seen Val ROC-AUC (%)"] <= 100 and row["Parameters"] > 0 and row["Latency (ms)"] > 0
    assert "Unseen Test ROC-AUC (%)" not in row  # locked until N3
    csv = pd.read_csv(out / "sheet_row.csv")
    assert list(csv.columns) == FR.SHEET_COLUMNS and len(csv) == 1
    FR.run(cfg(out, cache=str(cache), seeds=[0], save_checkpoints=False, overwrite=True))  # still our folder


def test_proj_dim_reaches_the_model(cache, tmp_path):
    out = tmp_path / "p"
    FR.run(cfg(out, cache=str(cache), seeds=[0], proj_dim=8))
    ck = torch.load(out / "seed0" / "model.pt", weights_only=False)
    assert ck["model_kwargs"]["proj_dim"] == 8 and any(k.startswith("proj.") for k in ck["model_state"])


# --------------------------------------------------------------------------- bootstrap CIs, step-score files

def test_bootstrap_ci_written_and_matches_sweep_file_layout(cache, tmp_path):
    out = tmp_path / "boot"
    s = FR.run(cfg(out, cache=str(cache), seeds=[0, 1], save_checkpoints=False, bootstrap=200))
    for k in ("roc_auc", "pr_auc", "tpr_at_5_fpr"):
        lo, hi = s["metrics"][k]["ci"]
        assert 0 <= lo <= hi <= 1
    lo, hi = s["metrics"]["roc_auc"]["ci"]
    assert lo <= s["metrics"]["roc_auc"]["mean"] <= hi
    assert s["bootstrap"]["n_resamples"] == 200 and "stratified" in s["bootstrap"]["scheme"]
    assert "95% CI [" in pd.read_csv(out / "sheet_row.csv").Notes.iloc[0]
    with np.load(out / "seed0" / FR.STEP_SCORES) as z:  # the depth sweep's rows/*.scores.npz keys
        assert set(z.files) == {"step_scores", "offsets", "labels", "horizons", "task_ids", "trajectory_ids"}
    d = FR.load_step_scores(out / "seed0" / FR.STEP_SCORES)
    csv = pd.read_csv(out / "seed0" / "seen_val_episode_scores.csv").set_index("trajectory_id")
    for sc, tid in zip(d.step_scores, d.trajectory_ids):  # episode score = max of the saved step logits
        assert float(np.max(sc)) == pytest.approx(csv.loc[tid, "score"], abs=1e-5)
    again = FR.add_bootstrap(out, 200)  # deterministic for a fixed seed
    assert again["metrics"]["roc_auc"]["ci"] == s["metrics"]["roc_auc"]["ci"]


def test_bootstrap_off_and_bad_settings(tmp_path):
    s = FR.run(cfg(tmp_path / "nb", synthetic=True, seeds=[0], epochs=1, bootstrap=0))
    assert "ci" not in s["metrics"]["roc_auc"]
    for kw in ({"bootstrap": -1}, {"ci": 1.0}, {"latency_device": "tpu"}, {"std_floor": 0.0},
               {"std_floor": 1e-2, "normalize": False}, {"overwrite": True, "resume": True}):
        with pytest.raises(ValueError):
            FR.validate_config(cfg(tmp_path / "x", synthetic=True, **kw))


def test_stratified_resamples_keep_class_counts():
    y = np.array([1, 0, 0, 1, 0, 0, 0])
    for idx in FR.stratified_indices(y, 50, 0):
        assert len(idx) == len(y) and y[idx].sum() == y.sum()


# --------------------------------------------------------------------------- memory, sequential reads, std floor

def test_memory_check_refuses_before_loading(cache, tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "100")  # 0.1 GB job
    with pytest.raises(ValueError, match="exceeds 90%"):
        FR.run(cfg(tmp_path / "m", cache=str(cache), seeds=[0]))
    FR.run(cfg(tmp_path / "m2", cache=str(cache), seeds=[0], epochs=1, memory_check=False, bootstrap=0,
               save_checkpoints=False))


def test_memory_limit_sources(monkeypatch):
    monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "2048")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    assert FR.memory_limit_gb() == pytest.approx(8.0)
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "16384")
    assert FR.memory_limit_gb() == pytest.approx(16.0)


def test_sequential_preload_gives_identical_records_and_frees_files(cache, tmp_path):
    splits, _ = FR.load_cache_splits(cfg(tmp_path, cache=str(cache), layers=[16, 33]))
    ref = load_cache.SharedCache(cache)
    for split, horizon in (("train", "full"), ("seen_val", "task_min_step")):
        want = ref.records(split, [16, 33], normalize=True, horizon=horizon)
        assert all(torch.equal(a["features"], b["features"]) for a, b in zip(splits[split], want))


def test_preload_refuses_unseen(cache):
    with pytest.raises(PermissionError):
        FR._preload_sequential(load_cache.SharedCache(cache), "mean", "unseen_test", [16])


def test_std_floor_caps_the_blowup(cache, tmp_path, capsys):
    splits, prov = FR.load_cache_splits(cfg(tmp_path, cache=str(cache), layers=[16], std_floor=1e-2))
    peak = max(float(r["features"].abs().max()) for recs in splits.values() for r in recs)
    assert peak < 1e4 and prov["std_floor"] == 1e-2  # the loader's 1e-6 floor gives ~2e6 on this fixture
    c = load_cache.SharedCache(cache)
    mean, std = c.norm_stats("mean", [16])
    raw = c.records("train", [16], normalize=False)[0]["features"]
    want = (raw - torch.from_numpy(mean)) / torch.from_numpy(np.maximum(std, 1e-2))
    assert torch.allclose(splits["train"][0]["features"], want)


# --------------------------------------------------------------------------- resume

def test_resume_keeps_finished_seeds_and_refuses_a_changed_config(cache, tmp_path, monkeypatch):
    out = tmp_path / "res"
    FR.run(cfg(out, cache=str(cache), seeds=[0], bootstrap=0, save_checkpoints=False))
    before = (out / "seed0" / "metrics.json").read_text()
    trained = []
    orig = FR.train_one
    monkeypatch.setattr(FR, "train_one", lambda m, tr, va, c, seed, ev=None: (trained.append(seed),
                                                                             orig(m, tr, va, c, seed, ev))[1])
    s = FR.run(cfg(out, cache=str(cache), seeds=[0, 1], bootstrap=0, save_checkpoints=False, resume=True))
    assert trained == [1] and (out / "seed0" / "metrics.json").read_text() == before
    assert s["seeds"] == [0, 1] and s["metrics"]["roc_auc"]["n_seeds_defined"] == 2
    loads = []
    monkeypatch.setattr(FR, "load_cache_splits", lambda c: loads.append(1))
    FR.run(cfg(out, cache=str(cache), seeds=[0, 1], bootstrap=50, resume=True))  # all done: no data read
    assert loads == [] and "ci" in json.loads((out / "summary.json").read_text())["metrics"]["roc_auc"]
    with pytest.raises(FileExistsError, match="different configuration"):
        FR.run(cfg(out, cache=str(cache), seeds=[0], lr=0.5, resume=True))
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "x.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="not written by this runner"):
        FR.run(cfg(foreign, cache=str(cache), seeds=[0], resume=True))


def test_resume_on_an_empty_folder_is_a_fresh_run(tmp_path):
    s = FR.run(cfg(tmp_path / "fresh", synthetic=True, seeds=[0], epochs=1, bootstrap=0, resume=True))
    assert s["seeds"] == [0]


# --------------------------------------------------------------------------- other fusion models (F4/F5 may win F6)

@pytest.mark.parametrize("model", ["gated_fusion", "residual_fusion"])
def test_runner_works_with_the_other_fusion_models(cache, tmp_path, model):
    s = FR.run(cfg(tmp_path / model, model=model, cache=str(cache), seeds=[0], epochs=1, bootstrap=20,
                   save_checkpoints=False))
    assert s["sheet_row"]["Parameters"] > 0 and "ci" in s["metrics"]["roc_auc"]


@pytest.mark.parametrize("model", ["concat_fusion", "static_weight_fusion"])
def test_latency_is_one_call_per_timestep_and_one_step_scores_match(model):
    """The online latency times a one-step call; that is valid only if a step's score does not depend on other steps."""
    recs = make_synthetic_records(n=4, t=9, k=2, d=8, seed=1)
    c = FR.RunConfig(model=model, layers=[0, 1], output_dir="unused", synthetic=True, proj_dim=4 if model ==
                     "static_weight_fusion" else None)
    m = FR.get_model(model, **FR.model_kwargs(c, 8))
    full = FR.timestep_logits(m, recs, batch_size=4)
    for r, ref in zip(recs, full):
        steps = [FR.timestep_logits(m, [{**r, "features": r["features"][:, t:t + 1], "mask": r["mask"][:, t:t + 1]}],
                                    1)[0] for t in range(int(r["mask"].sum()))]
        assert torch.allclose(torch.cat(steps), ref, atol=1e-6)
    m.train()
    lat = FR.measure_latency(m, recs, steps_per_episode=3, warmup=1)
    assert m.training  # mode restored
    assert lat["n_calls"] == 4 * 3 and lat["ms_per_step"] > 0 and lat["ms_per_step_amortized"] > 0
    assert "one call per step" in lat["definition"]


def test_duplicate_layers_only_with_the_flag_and_copied_from_one_read(cache, tmp_path, monkeypatch):
    """Row 35 (K4, duplicate-strongest-layer control) feeds one layer several times into the winning fusion."""
    with pytest.raises(ValueError, match="allow-duplicate-layers"):
        FR.validate_config(cfg(tmp_path / "x", cache=str(cache), layers=[33, 33]))
    reads = []
    orig = FR._preload_sequential
    monkeypatch.setattr(FR, "_preload_sequential", lambda c, pool, split, layers: (reads.append(list(layers)),
                                                                                    orig(c, pool, split, layers))[1])
    c = cfg(tmp_path / "dup", cache=str(cache), layers=[33, 33, 16], allow_duplicate_layers=True, seeds=[0])
    splits, prov = FR.load_cache_splits(c)
    assert reads == [[33, 16], [33, 16]]  # each layer file read once per split
    single, _ = FR.load_cache_splits(cfg(tmp_path / "one", cache=str(cache), layers=[33, 16]))
    for a, b in zip(splits["seen_val"], single["seen_val"]):
        assert torch.equal(a["features"][:, :, 0], b["features"][:, :, 0])
        assert torch.equal(a["features"][:, :, 1], b["features"][:, :, 0])
        assert torch.equal(a["features"][:, :, 2], b["features"][:, :, 1])
    assert prov["duplicate_layers"] and len(prov["layer_names"]) == 3
    s = FR.run(c)
    assert s["layers"] == [33, 33, 16]


# --------------------------------------------------------------------------- B6 probe settings + early stopping

def test_defaults_are_the_locked_b6_probe_and_the_b4_scorer():
    c = FR.RunConfig(model="concat_fusion", layers=[31], output_dir="x", synthetic=True)
    assert FR.b6_settings_of(c) == FR.B6_SETTINGS
    assert c.evaluator == FR.B4_EVALUATOR
    assert FR.provisional_flags(FR.resolve_evaluator(c.evaluator), c) == {}  # nothing provisional at B6 + B4
    flags = FR.provisional_flags(None, FR.RunConfig(model="concat_fusion", layers=[31], output_dir="x", lr=1e-3))
    assert "lr" in flags["probe"] and "episode_scorer" in flags


def _es_records(n=40, seed=0):
    recs = make_synthetic_records(n=n, t=6, k=2, d=8, seed=seed)
    for i, r in enumerate(recs):
        r["task_ids"] = [i % 4]
    return recs


def test_holdout_split_is_stratified_seeded_and_disjoint():
    recs = _es_records()
    fit, held = FR.holdout_split(recs, 0.15, seed=1)
    ids = lambda rs: [r["trajectory_ids"][0] for r in rs]  # noqa: E731
    assert not set(ids(fit)) & set(ids(held)) and len(fit) + len(held) == len(recs)
    groups = {(r["task_ids"][0], int(r["labels"].item())) for r in recs}
    assert {(r["task_ids"][0], int(r["labels"].item())) for r in held} == groups  # every task x outcome held out
    assert ids(FR.holdout_split(recs, 0.15, seed=1)[1]) == ids(held)               # same seed, same holdout
    assert ids(FR.holdout_split(recs, 0.15, seed=2)[1]) != ids(held)
    with pytest.raises(ValueError, match="both outcomes"):
        FR.holdout_split([r for r in recs if r["labels"].item() == 1], 0.15, seed=0)


def test_early_stopping_restores_the_best_holdout_epoch_and_never_uses_seen_val():
    recs = _es_records(seed=3)
    val = _es_records(n=12, seed=9)
    c = FR.RunConfig(model="concat_fusion", layers=[0, 1], output_dir="x", synthetic=True, epochs=60,
                     early_stopping_patience=3, lr=3e-2)
    m = FR.get_model("concat_fusion", **FR.model_kwargs(c, 8))
    out = FR.train_one(m, recs, val, c, seed=0)
    losses = [h["holdout_loss"] for h in out["history"]]
    best = int(np.argmin(losses)) + 1
    assert out["chosen_epoch"] == best and out["epochs_run"] == min(60, best + 3)
    assert out["n_fit_rollouts"] + out["n_holdout_rollouts"] == len(recs) and out["n_holdout_rollouts"] > 0
    _, held = FR.holdout_split(recs, 0.15, 0)
    assert FR.holdout_loss(m, held, 8) == pytest.approx(losses[best - 1], rel=1e-5)  # best weights restored
    with pytest.raises(ValueError, match="choose one"):
        FR.validate_config(FR.RunConfig(model="concat_fusion", layers=[0], output_dir="x", synthetic=True,
                                        select_best_epoch=True))
