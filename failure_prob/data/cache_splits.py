"""B1: training and evaluation read their episodes from the shared cache's splits.json.

splits.json is the source of truth for which trajectory belongs to train / seen_val / unseen_test (it is never
regenerated from a seed). This module checks that the cache's episodes.csv agrees with it, that the splits are
disjoint and complete, and that the unseen tasks are the protocol's, then loads records for one split in the
repo's record format, in splits.json order. unseen_test stays locked unless allow_unseen=True (final benchmark only).

Each record also carries `task_min_steps` ([int]), SAFE's per-task horizon, so evaluation can score each episode
at min(length, task_min_step) with failure_prob.engine.episode_metrics.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from failure_prob.protocol import UNSEEN_TASKS

SPLITS = ("train", "seen_val", "unseen_test")
REPO = Path(__file__).resolve().parents[2]


class SplitError(ValueError):
    pass


def _load_cache_module():
    """collection/load_cache.py is a script folder, not a package: import it by path."""
    path = REPO / "collection" / "load_cache.py"
    spec = importlib.util.spec_from_file_location("collection_load_cache", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_splits(cache_dir: str | Path) -> dict[str, list[str]]:
    """Read and validate splits.json against episodes.csv. Returns {split: [trajectory_id, ...]}."""
    import pandas as pd
    cache_dir = Path(cache_dir)
    doc = json.loads((cache_dir / "splits.json").read_text())
    traj = doc.get("trajectories")
    if not isinstance(traj, dict) or set(traj) != set(SPLITS):
        raise SplitError(f"{cache_dir}/splits.json must have trajectories for exactly {SPLITS}")
    seen: dict[str, str] = {}
    for split in SPLITS:
        ids = traj[split]
        if len(ids) != len(set(ids)):
            raise SplitError(f"splits.json: duplicate trajectory ids inside {split}")
        for t in ids:
            if t in seen:
                raise SplitError(f"splits.json: {t} is in both {seen[t]} and {split}")
            seen[t] = split
    eps = pd.read_csv(cache_dir / "episodes.csv")
    csv_split = dict(zip(eps.trajectory_id, eps.split))
    if set(csv_split) != set(seen):
        missing, extra = sorted(set(csv_split) - set(seen)), sorted(set(seen) - set(csv_split))
        raise SplitError(f"splits.json and episodes.csv list different episodes "
                         f"(not in splits.json: {missing[:3]}, not in episodes.csv: {extra[:3]})")
    wrong = [t for t, s in seen.items() if csv_split[t] != s]
    if wrong:
        raise SplitError(f"episodes.csv disagrees with splits.json for {len(wrong)} episodes, e.g. {wrong[:3]}; "
                         "splits.json is the source of truth: rebuild the cache with --split-file")
    unseen_tasks = tuple(sorted(int(t) for t in eps.loc[eps.split == "unseen_test", "task_id"].unique()))
    if unseen_tasks != tuple(sorted(UNSEEN_TASKS)):
        raise SplitError(f"cache's unseen tasks {unseen_tasks} != protocol {tuple(UNSEEN_TASKS)}")
    seen_tasks = set(eps.loc[eps.split != "unseen_test", "task_id"])
    if seen_tasks & set(unseen_tasks):
        raise SplitError(f"tasks {sorted(seen_tasks & set(unseen_tasks))} appear in both seen and unseen splits")
    return {s: list(traj[s]) for s in SPLITS}


def load_split_records(cache_dir: str | Path, split: str, layers: list[int], pool: str = "mean",
                       normalize: bool = False, allow_unseen: bool = False) -> list[dict]:
    """Full-length records for one split, ordered as in splits.json, each with `task_min_steps`."""
    if split not in SPLITS:
        raise SplitError(f"split must be one of {SPLITS}")
    if split == "unseen_test" and not allow_unseen:
        raise PermissionError("unseen_test is locked: training and model selection use train and seen_val only "
                              "(AGENTS.md); pass allow_unseen=True only for the final locked benchmark")
    cache_dir = Path(cache_dir)
    order = load_splits(cache_dir)[split]
    cache = _load_cache_module().SharedCache(cache_dir)
    records = cache.records(split, list(layers), pool=pool, normalize=normalize, horizon="full",
                            allow_unseen=allow_unseen)
    by_id = {r["trajectory_ids"][0]: r for r in records}
    if set(by_id) != set(order):
        raise SplitError(f"cache records for {split} do not match splits.json")
    min_step = dict(zip(cache.episodes.trajectory_id, cache.episodes.task_min_step))
    out = []
    for t in order:
        r = by_id[t]
        r["task_min_steps"] = [int(min_step[t])]
        out.append(r)
    return out


def horizons(records: list[dict]) -> np.ndarray:
    """Per-record evaluation horizon: min(length, task_min_step)."""
    return np.array([min(int(r["mask"].sum()), int(r["task_min_steps"][0])) for r in records])
