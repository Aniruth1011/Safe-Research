"""SAFE dataset module backed by the shared feature cache (C1-C3).

Lets SAFE's original pipeline (`python -m failure_prob.train dataset=openvla_cache ...`) train and evaluate
on the team's frozen split. Only `train` and `seen_val` are ever read; `unseen_test` (tasks 0/3/5) is locked
until the final benchmark (N3) and is never requested from the cache here.

- Split: taken verbatim from the cache's splits.json / episodes.csv. Seed-independent on purpose (SAFE's own
  split_rollouts re-draws tasks per seed).
- task_min_step: read from episodes.csv (computed over all 50 episodes of a task), never recomputed with
  SAFE's set_task_min_step, which would depend on which episodes are loaded.
- Features: raw (no normalisation), full horizon, one layer (default L32 = SAFE's feature).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from failure_prob.conf import Config
from failure_prob.protocol import UNSEEN_TASKS
from .utils import Rollout

LOADED_SPLITS = ("train", "seen_val")                     # never unseen_test
SPLIT_KEY = {"train": "train", "seen_val": "val_seen"}     # cache split name -> SAFE split name


def pool_name(token_idx_rel) -> str:
    """SAFE's token_idx_rel -> cache pool: 'mean' -> mean, 0.0 -> first, 1.0 -> last."""
    if isinstance(token_idx_rel, str) and token_idx_rel.strip().lower() == "mean":
        return "mean"
    try:
        v = float(token_idx_rel)
    except (TypeError, ValueError):
        v = None
    if v == 0.0:
        return "first"
    if v == 1.0:
        return "last"
    raise ValueError(f"token_idx_rel={token_idx_rel!r} has no cache pool; use 'mean', 0.0 (first) or 1.0 (last)")


def _shared_cache(root):
    kit = str(Path(__file__).resolve().parents[2] / "collection")
    if kit not in sys.path:
        sys.path.insert(0, kit)
    from load_cache import SharedCache
    return SharedCache(root)


def load_rollouts(cfg: Config) -> list[Rollout]:
    cache = _shared_cache(cfg.dataset.cache_root)
    eps = cache.episodes.set_index("trajectory_id")
    pool = pool_name(cfg.dataset.token_idx_rel)
    rollouts = []
    for split in LOADED_SPLITS:
        for rec in cache.records(split, [int(cfg.dataset.layer)], pool=pool):   # full horizon, raw features
            tid = rec["trajectory_ids"][0]
            ep = eps.loc[tid]
            if int(ep.task_id) in UNSEEN_TASKS or ep.split != split:
                raise RuntimeError(f"{tid}: task {ep.task_id} / split {ep.split} must not appear in {split}")
            r = Rollout(
                hidden_states=rec["features"][0, :, 0, :],          # (T, D) float32
                task_suite_name="libero_10",
                task_id=int(ep.task_id),
                task_description=str(ep.task_name),
                episode_idx=int(ep.episode_idx),
                episode_success=int(ep.success),
                mp4_path="",
                task_min_step=int(ep.task_min_step),
            )
            r.trajectory_id = tid
            r.cache_split = SPLIT_KEY[split]
            rollouts.append(r)
    print(f"openvla_cache: loaded {len(rollouts)} rollouts (L{int(cfg.dataset.layer):02d}, pool={pool}) "
          f"from {cfg.dataset.cache_root}")
    return rollouts


def split_rollouts(cfg: Config, all_rollouts: list[Rollout]) -> dict[str, list[Rollout]]:
    """Fixed split from the cache; ignores the training seed on purpose."""
    out = {name: [r for r in all_rollouts if r.cache_split == name] for name in SPLIT_KEY.values()}

    # Must match splits.json exactly (no re-splitting, nothing missing, nothing extra)
    frozen = json.loads((Path(cfg.dataset.cache_root) / "splits.json").read_text())["trajectories"]
    for cache_name, safe_name in SPLIT_KEY.items():
        got = sorted(r.trajectory_id for r in out[safe_name])
        if got != sorted(frozen[cache_name]):
            raise RuntimeError(f"{safe_name}: loaded trajectories differ from splits.json[{cache_name}]")

    expected = cfg.dataset.expected_split_sizes
    if expected is not None:
        sizes = (len(out["train"]), len(out["val_seen"]))
        if sizes != tuple(expected):
            raise RuntimeError(f"split sizes {sizes} != expected {tuple(expected)}")

    for name, rs in out.items():
        print(f"{name}: {len(rs)} rollouts")
    return out
