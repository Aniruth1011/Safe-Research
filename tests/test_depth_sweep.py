"""Depth sweep (R1-R4) on synthetic caches: pipeline, unseen_test lock, horizon, plugins, reruns. CPU only."""
import json
import textwrap

import numpy as np
import pandas as pd
import pytest

from failure_prob.engine import depth_sweep as ds
from failure_prob.engine.sweep_plugins import EvalInput, TemporaryMaxScoreEvaluator, TemporaryMLPProbe
from synthetic_cache import make_synthetic_cache


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return make_synthetic_cache(tmp_path_factory.mktemp("cache") / "v1", n_layers=4, dim=16, signal_layer=2)


def cfg(cache, **kw):
    return ds.SweepConfig(cache_dir=str(cache), **{"seeds": (0, 1), **kw})


def test_sweep_peaks_at_the_planted_layer_and_aggregates(cache, tmp_path):
    out = tmp_path / "out"
    for layer in range(4):
        ds.run_layer(cfg(cache), layer, out, log=lambda m: None)
    written = ds.aggregate(out, plot=False)
    summary = pd.read_csv(written["summary"])
    assert list(summary.layer) == [0, 1, 2, 3] and set(summary.n_seeds) == {2}
    auc = summary.set_index("layer").val_roc_auc_mean
    assert auc.idxmax() == 2 and auc[2] > 0.8 and auc.drop(2).max() < auc[2]
    assert summary.TEMPORARY.all()                                   # placeholder probe/metric are flagged
    results = pd.read_csv(written["results"])
    assert len(results) == 8 and set(results.eval_split) == {"seen_val"} and results.TEMPORARY.all()
    assert (results.n_val_rollouts == 28).all()                      # 7 seen tasks x 4 seen_val episodes
    assert "cache_dir" not in json.loads((out / "config.json").read_text())   # no local paths in outputs


def test_plot_is_written_and_marked_temporary(cache, tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "out"
    for layer in (0, 2):
        ds.run_layer(cfg(cache, seeds=(0,)), layer, out, log=lambda m: None)
    written = ds.aggregate(out)
    assert written["plot"].exists() and written["plot"].stat().st_size > 0


def test_unseen_test_is_never_opened(cache, tmp_path, monkeypatch):
    import builtins, os
    opened = []
    real_load, real_open, real_os_open = np.load, builtins.open, os.open
    # every way a file gets opened: np.load (the loader), open() (headers), os.open (the sequential reader)
    monkeypatch.setattr(np, "load", lambda p, *a, **k: (opened.append(str(p)), real_load(p, *a, **k))[1])
    monkeypatch.setattr(builtins, "open", lambda p, *a, **k: (opened.append(str(p)), real_open(p, *a, **k))[1])
    monkeypatch.setattr(os, "open", lambda p, *a, **k: (opened.append(str(p)), real_os_open(p, *a, **k))[1])
    ds.run_layer(cfg(cache, seeds=(0,)), 2, tmp_path / "out", log=lambda m: None)
    monkeypatch.undo()
    opened = [p for p in opened if p.startswith(str(cache))]               # cache files only (not the out folder)
    assert sum(p.endswith("/train/L02.npy") for p in opened) >= 2           # header + data
    assert any("/train/" in p for p in opened) and any("/seen_val/" in p for p in opened)
    assert not [p for p in opened if "unseen_test" in p]
    c = ds.SeenOnlyCache(cache)
    with pytest.raises(PermissionError):
        c.records("unseen_test", 2, "mean", False, "full")
    with pytest.raises(PermissionError):
        c._cache._array("mean", "unseen_test", 2)                   # the loader-level guard too
    with pytest.raises(PermissionError):
        c._cache.records("unseen_test", [2], allow_unseen=True)


def test_refuses_a_cache_with_an_unseen_task_in_train(tmp_path):
    bad = make_synthetic_cache(tmp_path / "bad", n_layers=2, dim=4, unseen_tasks=())  # task 0 lands in train
    with pytest.raises(ValueError, match="unseen tasks"):
        ds.SeenOnlyCache(bad)


@pytest.fixture
def plugin_module(tmp_path, monkeypatch):
    """A probe and evaluator living in their own module, plugged in by 'module:factory' only."""
    (tmp_path / "my_plugins.py").write_text(textwrap.dedent("""
        import numpy as np
        SEEN = {}

        class _Fitted:
            def __init__(self, w): self.w = w
            def step_scores(self, records):
                return [r["features"][0, :, 0].numpy() @ self.w for r in records]

        class MeanDiffProbe:
            name, temporary = "mean_diff", False
            def fit(self, train_records, seed):
                SEEN["train"] = train_records
                x = {0: [], 1: []}
                for r in train_records:
                    x[int(r["labels"][0])].append(r["features"][0, :, 0].numpy())
                return _Fitted(np.concatenate(x[1]).mean(0) - np.concatenate(x[0]).mean(0))

        class CapturingEvaluator:
            name, temporary = "capture", False
            def __call__(self, data):
                SEEN["eval"] = data
                return {"n": float(len(data.labels))}

        def make_probe(): return MeanDiffProbe()
        def make_evaluator(): return CapturingEvaluator()
    """))
    monkeypatch.syspath_prepend(str(tmp_path))
    import importlib
    import my_plugins
    importlib.reload(my_plugins)
    return my_plugins


def test_plugins_swap_by_name_and_get_the_horizon(cache, tmp_path, plugin_module):
    c = cfg(cache, seeds=(0,), probe="my_plugins:make_probe", evaluator="my_plugins:make_evaluator")
    rows = ds.run_layer(c, 2, tmp_path / "out", log=lambda m: None)
    assert rows[0]["probe"] == "mean_diff" and rows[0]["evaluator"] == "capture" and rows[0]["TEMPORARY"] is False
    data: EvalInput = plugin_module.SEEN["eval"]
    ep = pd.read_csv(cache / "episodes.csv").set_index("trajectory_id")
    assert set(ep.loc[data.trajectory_ids, "split"]) == {"seen_val"}
    assert list(data.horizons) == [min(ep.at[t, "length"], ep.at[t, "task_min_step"]) for t in data.trajectory_ids]
    assert [len(s) for s in data.step_scores] == [ep.at[t, "length"] for t in data.trajectory_ids]  # full rollouts
    assert {r["trajectory_ids"][0].split("--")[0] for r in plugin_module.SEEN["train"]} <= {f"task{t}" for t in (1, 2, 4, 6, 7, 8, 9)}
    assert set(ep.loc[[r["trajectory_ids"][0] for r in plugin_module.SEEN["train"]], "split"]) == {"train"}


def test_normalize_uses_train_split_statistics(cache, tmp_path, plugin_module):
    c = cfg(cache, seeds=(0,), normalize=True, probe="my_plugins:make_probe", evaluator="my_plugins:make_evaluator")
    ds.run_layer(c, 1, tmp_path / "out", log=lambda m: None)
    x = np.concatenate([r["features"][0, :, 0].numpy() for r in plugin_module.SEEN["train"]])
    assert np.abs(x.mean(0)).max() < 0.05 and np.abs(x.std(0) - 1).max() < 0.05


def test_pool_flag_reads_that_pool(cache, tmp_path, monkeypatch):
    import os
    opened = []
    real_os_open = os.open
    monkeypatch.setattr(os, "open", lambda p, *a, **k: (opened.append(str(p)), real_os_open(p, *a, **k))[1])
    ds.run_layer(cfg(cache, seeds=(0,), pool="first"), 0, tmp_path / "out", log=lambda m: None)
    monkeypatch.undo()
    feats = [p for p in opened if "/features/" in p]
    assert feats and all("/features/first/" in p for p in feats)


def test_placeholder_evaluator_ignores_steps_after_the_horizon():
    late_spike = [np.array([0.1, 0.1, 0.9]), np.array([0.2, 0.2, 0.0]), np.array([0.3, 0.3]), np.array([0.0, 0.0])]
    data = EvalInput(late_spike, np.array([0, 1, 1, 0]), np.array([2, 2, 2, 2]), np.zeros(4, int), list("abcd"))
    assert TemporaryMaxScoreEvaluator()(data)["roc_auc"] == 1.0   # with the step-3 spike counted it would be 0.75


def test_reruns_reuse_completed_rows_log_each_reuse_and_refuse_a_different_configuration(cache, tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setenv("SLURM_JOB_ID", "1001")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "1000")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    (row,) = ds.run_layer(cfg(cache, seeds=(0,)), 1, out, log=lambda m: None)
    assert row["completed"] is True and row["slurm_job_id"] == "1001" and row["slurm_array_task_id"] == "1"
    assert row["written_utc"].endswith("Z")
    msgs = []
    assert ds.run_layer(cfg(cache, seeds=(0,)), 1, out, log=msgs.append) == []      # reused
    assert any(m.startswith("layer 1 seed 0: reusing row from job 1000_1, commit ") for m in msgs), msgs
    assert len(ds.run_layer(cfg(cache, seeds=(0, 1)), 1, out, log=lambda m: None)) == 1  # only the new seed
    with pytest.raises(ValueError, match="different configuration"):
        ds.run_layer(cfg(cache, seeds=(0,), pool="first"), 1, out, log=lambda m: None)
    with pytest.raises(ValueError, match="not in this cache"):
        ds.run_layer(cfg(cache, seeds=(0,)), 4, out, log=lambda m: None)


def test_legacy_rows_are_reused_and_unfinished_rows_are_moved_aside(cache, tmp_path, monkeypatch):
    for k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"):
        monkeypatch.delenv(k, raising=False)
    out = tmp_path / "out"
    rows = {r["seed"]: r for r in ds.run_layer(cfg(cache, seeds=(0, 1, 2)), 1, out, log=lambda m: None)}
    assert rows[0]["slurm_job_id"] is None and rows[0]["completed"] is True       # outside Slurm: still reusable
    path = lambda s: out / "rows" / f"L01_seed{s}.json"
    legacy = {k: v for k, v in rows[0].items() if k not in ds.JOB_FIELDS + ("completed", "written_utc")}
    path(0).write_text(json.dumps(legacy))                                         # a row from before this change
    path(1).write_text(json.dumps({**rows[1], "completed": False}))                 # never finished
    path(2).write_text(json.dumps({k: v for k, v in rows[2].items() if k != "slurm_job_id"}))  # job fields missing
    msgs = []
    redone = ds.run_layer(cfg(cache, seeds=(0, 1, 2)), 1, out, log=msgs.append)
    assert sorted(r["seed"] for r in redone) == [1, 2]
    assert any("seed 0: reusing legacy row (written before job IDs were recorded)" in m for m in msgs), msgs
    assert sum("moved to" in m and "recomputing" in m for m in msgs) == 2
    assert sorted(f.name.split(".")[0] for f in (out / "rows" / "_discarded").iterdir()) == ["L01_seed1", "L01_seed2"]
    assert json.loads(path(0).read_text()) == legacy                               # legacy row left as it was
    assert all(json.loads(path(s).read_text())["completed"] is True for s in (1, 2))


def test_files_are_shared_readable_and_results_csv_has_no_job_fields(cache, tmp_path, monkeypatch):
    import os, stat
    monkeypatch.setenv("SLURM_JOB_ID", "1001")
    monkeypatch.setenv("SWEEP_LOG", "/somewhere/logs/x.out")
    old = os.umask(0o077)                                                          # a strict umask, as on CARC
    try:
        out = tmp_path / "out"
        ds.run_layer(cfg(cache, seeds=(0,)), 1, out, log=lambda m: None)
        stale = out / "rows" / "L01_seed0.json"
        os.chmod(stale, 0o600)                                                     # like the rows written before
        written = ds.aggregate(out, plot=False)
    finally:
        os.umask(old)
    mode = lambda p: stat.S_IMODE(p.stat().st_mode)
    assert mode(out) == mode(out / "rows") == 0o755
    assert {mode(p) for p in [stale, out / "config.json", *written.values()]} == {0o644}
    results = pd.read_csv(written["results"])
    assert not set(ds.JOB_FIELDS) & set(results.columns) and results.completed.all()


def test_sigterm_stops_the_run_before_a_row_is_written(cache, tmp_path):
    """A cancelled task (scancel / time limit) dumps its stack and exits without writing the row it was working on."""
    import os, signal, subprocess, sys, time
    (tmp_path / "slow_plugins.py").write_text(textwrap.dedent("""
        import time
        class SlowProbe:
            name, temporary = "slow", False
            def fit(self, train_records, seed):
                print("FITTING", flush=True)
                time.sleep(60)
        def make_probe(): return SlowProbe()
    """))
    out = tmp_path / "out"
    env = {**os.environ, "PYTHONPATH": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.Popen([sys.executable, str(ds.REPO_ROOT / "scripts" / "depth_sweep.py"), "run", "--cache",
                             str(cache), "--out", str(out), "--layers", "1", "--seeds", "0", "--probe",
                             "slow_plugins:make_probe"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    lines, t0 = [], time.time()
    for line in proc.stdout:
        lines.append(line)
        if "FITTING" in line or time.time() - t0 > 120:
            break
    proc.send_signal(signal.SIGTERM)
    rest, _ = proc.communicate(timeout=60)
    text = "".join(lines) + rest
    assert proc.returncode == 128 + signal.SIGTERM, text
    assert "received signal 15" in text and "slow_plugins.py" in text           # the stack shows where it was
    assert not list((out / "rows").glob("*.json")) if (out / "rows").exists() else True
    assert any("importing failure_prob" in l for l in lines) and "PYTHONDONTWRITEBYTECODE=1" in lines[0]


def test_temporary_probe_is_seeded():
    pytest.importorskip("torch")
    from synthetic_cache import make_synthetic_cache  # noqa: F401  (fixture module importable)
    rng = np.random.default_rng(0)
    import torch
    recs = [{"features": torch.from_numpy(rng.standard_normal((1, 5, 1, 4)).astype(np.float32)),
             "mask": torch.ones(1, 5, dtype=torch.bool), "labels": torch.tensor([float(i % 2)]),
             "task_ids": [1], "trajectory_ids": [f"t{i}"]} for i in range(6)]
    p = TemporaryMLPProbe(epochs=1)
    a, b, c = (p.fit(recs, s).step_scores(recs) for s in (0, 0, 1))
    assert all(np.array_equal(x, y) for x, y in zip(a, b)) and not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_cli_run_and_aggregate(cache, tmp_path, capsys):
    import importlib.util, sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("depth_sweep_cli", Path(ds.REPO_ROOT) / "scripts" / "depth_sweep.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    out = tmp_path / "out"
    cli.main(["run", "--cache", str(cache), "--out", str(out), "--layers", "0", "2", "--seeds", "0", "--threads", "1"])
    cli.main(["aggregate", "--out", str(out), "--no-plot"])
    assert len(pd.read_csv(out / "summary.csv")) == 2
    assert "TEMPORARY" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------ CARC job scripts
SLURM_DIR = ds.REPO_ROOT / "scripts" / "carc"


def _bash():
    import shutil
    return shutil.which("bash")


@pytest.mark.parametrize("name", ["depth_sweep.slurm", "depth_sweep_aggregate.slurm"])
def test_slurm_scripts_parse_and_refuse_outside_slurm(name, tmp_path):
    import os, subprocess
    path = SLURM_DIR / name
    assert subprocess.run([_bash(), "-n", str(path)]).returncode == 0
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    r = subprocess.run([_bash(), str(path)], env=env, capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode != 0 and "never run this on a login node" in r.stderr


def test_slurm_array_task_and_aggregate_end_to_end(cache, tmp_path):
    """Both job scripts with a fake Slurm environment, on the synthetic cache; folders derived the same way."""
    import os, subprocess, sys
    env = {**os.environ, "SLURM_JOB_ID": "1", "SLURM_ARRAY_JOB_ID": "7", "SLURM_SUBMIT_DIR": str(ds.REPO_ROOT),
           "SLURM_CPUS_PER_TASK": "1", "SWEEP_PYTHON": sys.executable, "CACHE_DIR": str(cache),
           "SWEEP_OUT_ROOT": str(tmp_path / "sweeps"), "SWEEP_POOL": "first", "SWEEP_NORMALIZE": "1",
           "SWEEP_EXPECT_LAYERS": "0,2"}
    env.pop("PYTHONDONTWRITEBYTECODE", None)                          # the script must set it itself
    for layer in ("0", "2", "2"):
        r = subprocess.run([_bash(), str(SLURM_DIR / "depth_sweep.slurm")], env={**env, "SLURM_ARRAY_TASK_ID": layer},
                           capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stdout + r.stderr
        assert f"log {ds.REPO_ROOT}/logs/depth_sweep-7_{layer}.out" in r.stdout
        assert "PYTHONDONTWRITEBYTECODE=1" in r.stdout
        assert f"layer {layer} seed 2:" in r.stdout                      # computed the first time, reused the second
        assert "logging to node-local" in r.stdout and f"layer {layer} done" in r.stdout   # local log copied back
    assert "layer 2 seed 2: reusing row from job 7_2" in r.stdout
    r = subprocess.run([_bash(), str(SLURM_DIR / "depth_sweep_aggregate.slurm")], env=env, capture_output=True,
                       text=True, timeout=600)
    assert r.returncode in (0, 3), r.stdout + r.stderr                 # 3 = no matplotlib, tables still written
    out = tmp_path / "sweeps" / ds.SweepConfig(cache_dir="", pool="first", normalize=True).label()
    summary = pd.read_csv(out / "summary.csv")
    assert list(summary.layer) == [0, 2] and set(summary.n_seeds) == {3}   # seeds 0, 1, 2 from protocol.TRAIN_SEEDS


# ------------------------------------------------------------------------------------- file-system friendly loading
def test_sequential_reader_gives_exactly_the_memmap_values(cache, monkeypatch):
    monkeypatch.setenv("SWEEP_READ_CHUNK_MB", "0.003")                 # many small chunks
    msgs = []
    seen_only = ds.SeenOnlyCache(cache, log=msgs.append)
    plain = ds.load_cache_module().SharedCache(cache)                 # np.load(mmap_mode="r") + per-episode slices
    for split, norm, horizon in [("train", False, "full"), ("seen_val", True, "full"), ("train", False, "task_min_step")]:
        a = seen_only.records(split, 2, "mean", norm, horizon)
        b = plain.records(split, [2], pool="mean", normalize=norm, horizon=horizon)
        assert len(a) == len(b) and all(x["features"].equal(y["features"]) for x, y in zip(a, b))
        assert [x["trajectory_ids"] for x in a] == [y["trajectory_ids"] for y in b]
    assert any(m.startswith("read mean/train/L02.npy:") and "MB/s" in m for m in msgs), msgs
    assert seen_only._mm == {}                                         # arrays dropped once records exist


def test_reader_retries_a_stalled_chunk_on_a_new_handle(tmp_path, monkeypatch):
    import os, time
    arr = np.arange(10_000, dtype=np.float16).reshape(100, 100)
    np.save(tmp_path / "L00.npy", arr)
    real, calls = os.pread, []

    def flaky(fd, n, off):
        calls.append(off)
        if len(calls) == 1:
            time.sleep(1.5)                                            # the first read hangs
        return real(fd, n, off)

    monkeypatch.setattr(os, "pread", flaky)
    msgs = []
    out = ds.read_npy_sequential(tmp_path / "L00.npy", log=msgs.append, chunk_mb=0.01, stall_s=0.3, retries=2)
    assert np.array_equal(out, arr)
    assert any("no progress for 0 s at 0 MB; retrying on a new file handle (1/2)" in m for m in msgs), msgs


def test_reader_gives_up_after_the_retries(tmp_path, monkeypatch):
    import os, time
    np.save(tmp_path / "L00.npy", np.zeros((10, 10), np.float16))
    monkeypatch.setattr(os, "pread", lambda fd, n, off: time.sleep(3) or b"")
    with pytest.raises(ds.ReadStalled, match="after 1 retries"):
        ds.read_npy_sequential(tmp_path / "L00.npy", log=lambda m: None, stall_s=0.2, retries=1)


def _cli(*args, env=None, timeout=120):
    import os, subprocess, sys
    return subprocess.run([sys.executable, str(ds.REPO_ROOT / "scripts" / "depth_sweep.py"), *args],
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(env or {})}, capture_output=True,
                          text=True, timeout=timeout)


def test_startup_watchdog_exits_75_with_a_stack_instead_of_waiting_for_the_time_limit(cache, tmp_path):
    slow = tmp_path / "slow_loader.py"
    slow.write_text("import time\nclass SharedCache:\n    def __init__(self, root):\n        time.sleep(60)\n")
    t0 = __import__("time").time()
    r = _cli("run", "--cache", str(cache), "--out", str(tmp_path / "out"), "--layers", "1", "--seeds", "0",
             "--loader", str(slow), env={"SWEEP_LOAD_TIMEOUT": "15"})   # long enough for the imports
    assert r.returncode == 75, r.stdout + r.stderr
    assert __import__("time").time() - t0 < 50
    assert "WATCHDOG: start-up (imports, opening the cache) and loading the first layer did not finish within 15 s" in r.stdout
    assert "slow_loader.py" in r.stdout                                 # the stack shows where it was stuck


def test_aggregate_lists_missing_layers_and_exits_4(cache, tmp_path):
    out = tmp_path / "out"
    for layer in (0, 2):
        ds.run_layer(cfg(cache, seeds=(0, 1)), layer, out, log=lambda m: None)
    (out / "rows" / "L02_seed1.json").unlink()
    r = _cli("aggregate", "--out", str(out), "--no-plot", "--expect-layers", "0-3")
    assert r.returncode == 4, r.stdout + r.stderr
    assert "MISSING: layer 1 seeds [0, 1]" in r.stdout and "MISSING: layer 2 seeds [1]" in r.stdout
    assert "LAYERS=1,2,3" in r.stdout and (out / "summary.csv").exists()   # partial tables still written
    r = _cli("aggregate", "--out", str(out), "--no-plot", "--expect-layers", "0")
    assert r.returncode == 0 and "all expected layers present" in r.stdout


def test_config_json_is_unchanged_so_finished_rows_stay_reusable(cache, tmp_path):
    """config.json must keep exactly these keys: a different config.json makes the sweep refuse the folder and its rows."""
    ds.run_layer(cfg(cache, seeds=(0,)), 0, tmp_path / "out", log=lambda m: None)
    doc = json.loads((tmp_path / "out" / "config.json").read_text())
    assert sorted(doc) == ["cache", "evaluator", "evaluator_name", "evaluator_temporary", "normalize", "pool", "probe",
                           "probe_name", "probe_temporary", "seeds", "train_horizon"]
    assert sorted(doc["cache"]) == ["cache_version", "created_utc", "manifest_sha256"]


def test_slurm_task_copies_its_log_and_stops_python_on_sigterm(cache, tmp_path):
    import os, signal, subprocess, sys, time
    (tmp_path / "slow_plugins.py").write_text(textwrap.dedent("""
        import time
        class SlowProbe:
            name, temporary = "slow", False
            def fit(self, train_records, seed):
                print("FITTING", flush=True)
                time.sleep(60)
        def make_probe(): return SlowProbe()
    """))
    log_dir = tmp_path / "local"
    log_dir.mkdir()
    env = {**os.environ, "SLURM_JOB_ID": "1", "SLURM_ARRAY_JOB_ID": "7", "SLURM_ARRAY_TASK_ID": "1",
           "SLURM_SUBMIT_DIR": str(ds.REPO_ROOT), "SLURM_CPUS_PER_TASK": "1", "SWEEP_PYTHON": sys.executable,
           "CACHE_DIR": str(cache), "SWEEP_OUT": str(tmp_path / "out"), "SWEEP_PROBE": "slow_plugins:make_probe",
           "PYTHONPATH": str(tmp_path), "TMPDIR": str(log_dir)}
    slurm_log = tmp_path / "slurm.out"
    with open(slurm_log, "w") as f:
        proc = subprocess.Popen([_bash(), str(SLURM_DIR / "depth_sweep.slurm")], env=env, stdout=f, stderr=f)
        t0 = time.time()
        while time.time() - t0 < 120:                                   # the local log fills; the Slurm log does not
            local = list(log_dir.glob("depth_sweep-7_1.*.log"))
            if local and "FITTING" in local[0].read_text():
                break
            time.sleep(0.2)
        assert "FITTING" not in slurm_log.read_text()
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=60) == 143
    text = slurm_log.read_text()
    assert "got SIGTERM" in text and "received signal 15" in text and "FITTING" in text, text
    assert not list(log_dir.glob("*.log"))                              # local copy removed after copying
    assert not list((tmp_path / "out" / "rows").glob("*.json"))


def test_submit_script_throttles_excludes_and_chains_the_aggregate(tmp_path):
    import os, subprocess
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "sbatch").write_text('#!/bin/bash\necho "$*" >> "$SBATCH_LOG"\n'
                                 'n=$(wc -l < "$SBATCH_LOG"); echo "$((4000 + n));cluster"\n')
    (fake / "sbatch").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.update(PATH=f"{fake}:{env['PATH']}", SBATCH_LOG=str(tmp_path / "calls"), EXCLUDE_NODES="nodeA,nodeB")
    script = str(SLURM_DIR / "submit_depth_sweep.sh")
    r = subprocess.run([_bash(), script, "--account", "acct", "--layers", "10,16,19-20"], env=env, capture_output=True,
                       text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    sweep, agg = (tmp_path / "calls").read_text().splitlines()
    assert "--array=10,16,19-20%6" in sweep and "--exclude=nodeA,nodeB" in sweep and "--account=acct" in sweep
    assert sweep.endswith("scripts/carc/depth_sweep.slurm")
    assert "--dependency=afterany:4001" in agg and "--exclude=nodeA,nodeB" in agg and agg.endswith("_aggregate.slurm")
    assert "array job 4001" in r.stdout and "aggregate job 4002" in r.stdout
    r = subprocess.run([_bash(), script, "--account", "acct", "--max-parallel", "3", "--dry-run"], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0 and "--array=0-33%3" in r.stdout
    assert len((tmp_path / "calls").read_text().splitlines()) == 2    # dry run submitted nothing
    r = subprocess.run([_bash(), script, "--layers", "0-1"], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "--account is required" in r.stderr


def test_slurm_default_array_is_throttled():
    assert "#SBATCH --array=0-33%6" in (SLURM_DIR / "depth_sweep.slurm").read_text().splitlines()


def test_job_logs_cannot_be_committed():
    import shutil, subprocess
    if shutil.which("git") is None or not (ds.REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    git = lambda *a: subprocess.run(["git", "-C", str(ds.REPO_ROOT), *a], capture_output=True, text=True)
    for name in ("logs/depth_sweep-1_2.out", "logs/depth_sweep_aggregate-3.out", "logs/anything.log"):
        assert git("check-ignore", "-q", name).returncode == 0, name
    assert git("check-ignore", "-q", "logs/.gitkeep").returncode == 1
    assert git("ls-files", "logs").stdout.split() == ["logs/.gitkeep"]


def test_leftover_temp_files_are_moved_aside_only_when_stale(cache, tmp_path):
    import os, time
    out = tmp_path / "out"
    ds.run_layer(cfg(cache, seeds=(0,)), 1, out, log=lambda m: None)
    old = time.time() - 3600
    names = {out / ".config.json.3g3gs9sb": old, out / ".fhnuqgyg": old, out / "rows" / ".L01_seed1.json.ab_c1234": old,
             out / ".zy7kq697": time.time()}                        # the last one: a task may be writing it now
    for f, mtime in names.items():
        f.write_text("partial")
        os.utime(f, (mtime, mtime))
    keep = [out / "notes.txt", out / ".hidden_but_not_temp_file"]
    for f in keep:
        f.write_text("x")
        os.utime(f, (old, old))
    msgs = []
    ds.run_layer(cfg(cache, seeds=(0,)), 1, out, log=msgs.append)  # every seed reused; the cleanup still runs
    moved = sorted(p.name for p in (out / "_discarded" / "tmp").iterdir())
    assert moved == [".config.json.3g3gs9sb", ".fhnuqgyg", "rows_.L01_seed1.json.ab_c1234"]
    assert (out / ".zy7kq697").exists() and all(f.exists() for f in keep)
    assert sum("moved leftover temp file" in m for m in msgs) == 3
    (out / ".zy7kq697").unlink()
    ds.aggregate(out, plot=False)                                   # results ignore _discarded/
    assert len(pd.read_csv(out / "results.csv")) == 1


def test_a_failed_write_leaves_no_temp_file(tmp_path, monkeypatch):
    import os
    def boom(*a):
        raise SystemExit(143)                                       # as if SIGTERM arrived mid-write
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(SystemExit):
        ds._write_json_atomic(tmp_path / "rows" / "L00_seed0.json", {"a": 1})
    monkeypatch.undo()
    assert list((tmp_path / "rows").iterdir()) == []
