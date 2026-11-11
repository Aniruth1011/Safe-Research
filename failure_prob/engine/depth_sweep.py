"""Depth sweep over the shared cache (plan rows R1-R4): one probe per layer x training seed, scored on seen_val.

    train on split "train"  ->  score every seen_val rollout  ->  evaluator at the task_min_step horizon

Never reads unseen_test: SeenOnlyCache refuses that split before any feature file is opened, and the sweep also
refuses a cache whose train/seen_val episodes include a task from protocol.UNSEEN_TASKS.

Outputs, in one folder per configuration (pool, normalize, train horizon, probe, evaluator, cache):
    config.json                 the configuration; a second run into the same folder must match it
    rows/L{LL}_seed{S}.json     one row per (layer, seed), written atomically; reruns skip finished rows
    results.csv                 all rows (aggregate)
    summary.csv                 mean / std / n over seeds per layer (aggregate)
    depth_curve.png             mean +- std per layer (aggregate)
Rows made with a TEMPORARY probe or evaluator say so in every file and in the plot title.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from failure_prob.engine.sweep_plugins import EVALUATORS, PROBES, EvalInput, resolve
from failure_prob.protocol import TRAIN_SEEDS, UNSEEN_TASKS

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOADER = REPO_ROOT / "collection" / "load_cache.py"
ALLOWED_SPLITS = ("train", "seen_val")  # unseen_test is never touched by the sweep
METRIC_PREFIX = "val_"


def load_cache_module(path: Path = DEFAULT_LOADER):
    """Import collection/load_cache.py (it is a script folder, not a package)."""
    spec = importlib.util.spec_from_file_location("load_cache", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SeenOnlyCache:
    """SharedCache restricted to train and seen_val. Any request for another split raises before a file is opened."""

    def __init__(self, cache_dir: Path, loader: Path = DEFAULT_LOADER):
        self._cache = load_cache_module(loader).SharedCache(cache_dir)
        inner_array = self._cache._array

        def guarded_array(pool, split, layer):
            _check_split(split)
            return inner_array(pool, split, layer)

        self._cache._array = guarded_array  # second line of defence, in case the loader changes
        self.manifest = self._cache.manifest
        self.episodes = self._cache.episodes
        leaked = sorted(set(self.episodes[self.episodes.split.isin(ALLOWED_SPLITS)].task_id) & set(UNSEEN_TASKS))
        if leaked:
            raise ValueError(f"cache puts unseen tasks {leaked} in train/seen_val; refusing (protocol.UNSEEN_TASKS)")

    def records(self, split: str, layer: int, pool: str, normalize: bool, horizon: str) -> list[dict]:
        _check_split(split)
        return self._cache.records(split, [layer], pool=pool, normalize=normalize, horizon=horizon)

    def layer_names(self) -> list[str]:
        names = self.manifest.get("layer_names")
        if names:
            return list(names)
        n = int(self.manifest.get("num_hidden_states", 33))
        return ["emb"] + [f"block{i:02d}" for i in range(1, 32)] + ["block32_postnorm"] if n == 33 else [f"L{i}" for i in range(n)]

    def identity(self) -> dict:
        root = Path(self._cache.root)
        return {"cache_version": self.manifest.get("cache_version"), "created_utc": self.manifest.get("created_utc"),
                "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()}


def _check_split(split: str) -> None:
    if split not in ALLOWED_SPLITS:
        raise PermissionError(f"the depth sweep never reads {split!r} (allowed: {ALLOWED_SPLITS})")


@dataclass(frozen=True)
class SweepConfig:
    cache_dir: str
    pool: str = "mean"
    normalize: bool = False
    train_horizon: str = "full"          # "full" or "task_min_step"; seen_val is always scored at task_min_step
    probe: str = "temporary_mlp"
    evaluator: str = "temporary_max"
    seeds: tuple[int, ...] = field(default=TRAIN_SEEDS)

    def label(self) -> str:
        safe = lambda s: s.replace(":", "-").replace(".", "_").replace("/", "_")
        return f"{self.pool}-{'trainnorm' if self.normalize else 'raw'}-train_{self.train_horizon}-{safe(self.probe)}-{safe(self.evaluator)}"


def _git_commit() -> str:
    try:
        sha = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
        return (sha or "unknown") + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def _peak_rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6  # bytes on macOS, KiB on Linux


def _write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _check_or_write_config(out_dir: Path, cfg: SweepConfig, cache: SeenOnlyCache, probe, evaluator) -> dict:
    cfg_doc = {**asdict(cfg), "seeds": list(cfg.seeds), "cache": cache.identity(),
               "probe_name": probe.name, "probe_temporary": bool(probe.temporary),
               "evaluator_name": evaluator.name, "evaluator_temporary": bool(evaluator.temporary)}
    cfg_doc.pop("cache_dir")  # a path (may contain a user name); the cache identity above is what matters
    path = out_dir / "config.json"
    if path.exists():
        old = json.loads(path.read_text())
        if {k: v for k, v in old.items() if k != "seeds"} != {k: v for k, v in cfg_doc.items() if k != "seeds"}:
            raise ValueError(f"{out_dir} holds a different configuration ({path}); use another --out folder")
        cfg_doc["seeds"] = sorted(set(old["seeds"]) | set(cfg_doc["seeds"]))
    _write_json_atomic(path, cfg_doc)
    return cfg_doc


def run_layer(cfg: SweepConfig, layer: int, out_dir: Path, loader: Path = DEFAULT_LOADER, overwrite: bool = False,
              log=print) -> list[dict]:
    """Train cfg.seeds probes on one layer and score seen_val. Returns the rows (also written to out_dir/rows/)."""
    out_dir = Path(out_dir)
    cache = SeenOnlyCache(Path(cfg.cache_dir), loader)
    names = cache.layer_names()
    if not 0 <= layer < len(names):
        raise ValueError(f"layer {layer} not in this cache (0..{len(names) - 1})")
    probe, evaluator = resolve(cfg.probe, PROBES), resolve(cfg.evaluator, EVALUATORS)
    cfg_doc = _check_or_write_config(out_dir, cfg, cache, probe, evaluator)
    todo = [s for s in cfg.seeds if overwrite or not (out_dir / "rows" / f"L{layer:02d}_seed{s}.json").exists()]
    if not todo:
        log(f"layer {layer}: all seeds {list(cfg.seeds)} already done; skipped")
        return []

    t0 = time.time()
    train = cache.records("train", layer, cfg.pool, cfg.normalize, cfg.train_horizon)
    val = cache.records("seen_val", layer, cfg.pool, cfg.normalize, "full")
    load_s = time.time() - t0
    ep = cache.episodes.set_index("trajectory_id")
    val_ids = [r["trajectory_ids"][0] for r in val]
    horizons = np.array([min(int(ep.at[t, "length"]), int(ep.at[t, "task_min_step"])) for t in val_ids])
    labels = np.array([float(r["labels"].reshape(-1)[0]) for r in val])
    task_ids = np.array([int(r["task_ids"][0]) for r in val])
    if set(task_ids) & set(UNSEEN_TASKS):
        raise ValueError("seen_val contains an unseen task")  # cannot happen after SeenOnlyCache's check
    log(f"layer {layer} ({names[layer]}): loaded {len(train)} train / {len(val)} seen_val rollouts in {load_s:.1f} s")

    rows = []
    for seed in todo:
        t1 = time.time()
        fitted = probe.fit(train, seed)
        fit_s = time.time() - t1
        t2 = time.time()
        scores = fitted.step_scores(val)
        metrics = evaluator(EvalInput(scores, labels, horizons, task_ids, val_ids))
        score_s = time.time() - t2
        temporary = bool(probe.temporary or evaluator.temporary)
        row = {
            "experiment": "R1-R4_depth_sweep", "layer": layer, "layer_name": names[layer], "seed": seed,
            "pool": cfg.pool, "normalize": cfg.normalize, "train_horizon": cfg.train_horizon,
            "eval_split": "seen_val", "eval_horizon": "task_min_step",
            "probe": probe.name, "evaluator": evaluator.name, "TEMPORARY": temporary,
            "n_train_rollouts": len(train), "n_train_steps": int(sum(r["features"].shape[1] for r in train)),
            "n_val_rollouts": len(val), "n_val_failures": int(labels.sum()),
            **{METRIC_PREFIX + k: v for k, v in metrics.items()},
            "load_s": round(load_s, 2), "fit_s": round(fit_s, 2), "score_s": round(score_s, 2),
            "peak_rss_gb": round(_peak_rss_gb(), 2), "threads": _threads(),
            "cache_version": cfg_doc["cache"]["cache_version"], "cache_manifest_sha256": cfg_doc["cache"]["manifest_sha256"],
            "git_commit": _git_commit(), "python": platform.python_version(), "torch": _torch_version(),
        }
        _write_json_atomic(out_dir / "rows" / f"L{layer:02d}_seed{seed}.json", row)
        shown = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        log(f"layer {layer} seed {seed}: {shown}  (fit {fit_s:.1f} s, score {score_s:.1f} s, peak RSS {row['peak_rss_gb']} GB)"
            + ("  [TEMPORARY probe/evaluator]" if temporary else ""))
        rows.append(row)
    return rows


def _threads() -> int:
    try:
        import torch
        return torch.get_num_threads()
    except ImportError:
        return 0


def _torch_version() -> str:
    try:
        import torch
        return torch.__version__
    except ImportError:
        return "none"


def aggregate(out_dir: Path, metric: str = "roc_auc", plot: bool = True) -> dict:
    """Collect rows into results.csv and summary.csv, and draw depth_curve.png. Returns the paths written."""
    import pandas as pd
    out_dir = Path(out_dir)
    files = sorted((out_dir / "rows").glob("L*_seed*.json"))
    if not files:
        raise FileNotFoundError(f"no rows under {out_dir / 'rows'}")
    df = pd.DataFrame([json.loads(f.read_text()) for f in files]).sort_values(["layer", "seed"])
    if df.eval_split.ne("seen_val").any():
        raise ValueError("rows scored on something other than seen_val")
    if df[["probe", "evaluator", "pool", "normalize", "train_horizon", "cache_manifest_sha256"]].nunique().max() > 1:
        raise ValueError("rows from different configurations in one folder")
    df.to_csv(out_dir / "results.csv", index=False)
    metric_cols = [c for c in df.columns if c.startswith(METRIC_PREFIX)]
    g = df.groupby(["layer", "layer_name"])[metric_cols]
    summary = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std")).join(g.size().rename("n_seeds"))
    summary = summary.reset_index()
    temporary = bool(df.TEMPORARY.any())
    summary.insert(0, "TEMPORARY", temporary)
    summary.to_csv(out_dir / "summary.csv", index=False)
    written = {"results": out_dir / "results.csv", "summary": out_dir / "summary.csv"}
    if plot:
        written["plot"] = _plot(summary, df.iloc[0], METRIC_PREFIX + metric, out_dir / "depth_curve.png", temporary)
    return written


def _plot(summary, first_row, col: str, path: Path, temporary: bool) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    x, m, s = summary.layer.to_numpy(), summary[f"{col}_mean"].to_numpy(), summary[f"{col}_std"].fillna(0).to_numpy()
    ax.plot(x, m, marker="o", ms=3, lw=1.5, color="#2563eb")
    ax.fill_between(x, m - s, m + s, alpha=0.2, color="#2563eb", lw=0)
    ax.set_xlabel("layer index (0 = embeddings, 32 = final post-norm, 33 = raw block 32 if present)")
    ax.set_ylabel(f"seen_val {col.removeprefix(METRIC_PREFIX)} (mean ± std over seeds)")
    ax.set_xticks(x)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(alpha=0.3)
    title = (f"Depth sweep: pool={first_row.pool}, {'train-norm' if first_row.normalize else 'raw'}, "
             f"probe={first_row.probe}, eval={first_row.evaluator}")
    if temporary:
        title = "TEMPORARY probe/metric: NOT A RESULT\n" + title
    ax.set_title(title, fontsize=9, color="#b91c1c" if temporary else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
