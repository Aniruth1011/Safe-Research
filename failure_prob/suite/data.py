"""Shared-cache loading into flat, device-resident tensors.

Rows of every episode are stored contiguously (episode order of episodes.csv)
and truncated to min(length, task_min_step). Batches are built by index, so no
padding is materialized for per-timestep models. unseen_test is refused unless
allow_unseen=True, which only the N3 benchmark passes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config as C

SPLITS = ('train', 'seen_val', 'unseen_test')


@dataclass
class SplitData:
    split: str
    layers: tuple
    x: torch.Tensor            # [N, K, D] float32 on device
    starts: torch.Tensor       # [E] long, row offset of each episode in x
    lengths: torch.Tensor      # [E] long, valid length (full for train, task_min_step for eval splits)
    labels: torch.Tensor       # [E] float, failure = 1
    task_ids: np.ndarray       # [E]
    trajectory_ids: list

    @property
    def n_episodes(self):
        return len(self.lengths)

    @property
    def device(self):
        return self.x.device

    def batch(self, episodes: torch.Tensor):
        """Rows of the given episodes (concatenated) and the [B, T] validity mask.

        mask is row-major consistent with rows: padded[mask] = per_row_values.
        """
        episodes = episodes.to(self.lengths.device)
        lengths = self.lengths[episodes]
        T = int(lengths.max())
        pos = torch.arange(T, device=lengths.device)
        mask = pos[None, :] < lengths[:, None]
        rows = (self.starts[episodes][:, None] + pos[None, :])[mask]
        return rows, mask

    def with_features(self, x: torch.Tensor, layers=('embedding',)):
        return SplitData(self.split, tuple(layers), x, self.starts, self.lengths, self.labels,
                         self.task_ids, self.trajectory_ids)


def episodes_table(cache: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(Path(cache or C.CACHE) / 'episodes.csv')


def verify_split_isolation(cache: Path | None = None) -> dict:
    """Cheap consumer checks: episodes.csv agrees with splits.json, tasks are disjoint."""
    cache = Path(cache or C.CACHE)
    ep = episodes_table(cache)
    splits = json.loads((cache / 'splits.json').read_text())
    for split in SPLITS:
        declared = set(splits['trajectories'][split])
        listed = set(ep.trajectory_id[ep.split == split])
        if declared != listed:
            raise ValueError(f'{split}: episodes.csv and splits.json disagree ({len(declared ^ listed)} ids)')
    unseen = set(ep.task_id[ep.split == 'unseen_test'])
    seen = set(ep.task_id[ep.split != 'unseen_test'])
    if unseen & seen or unseen != set(splits['unseen_tasks']) or unseen != set(C.UNSEEN_TASKS):
        raise ValueError(f'task split not isolated: seen={sorted(seen)} unseen={sorted(unseen)}')
    return dict(seen_tasks=sorted(int(t) for t in seen), unseen_tasks=sorted(int(t) for t in unseen),
                counts={s: int((ep.split == s).sum()) for s in SPLITS})


def load_split(split: str, layers, device='cpu', cache: Path | None = None, pool: str = C.POOL,
               normalize: bool | None = None, allow_unseen: bool = False, horizon: str | None = None) -> SplitData:
    """train uses C.TRAIN_HORIZON (full rollouts, as SAFE); evaluation splits use C.EVAL_HORIZON."""
    normalize = C.NORMALIZE if normalize is None else normalize
    horizon = horizon or (C.TRAIN_HORIZON if split == 'train' else C.EVAL_HORIZON)
    if split not in SPLITS:
        raise ValueError(split)
    if split == 'unseen_test' and not allow_unseen:
        raise PermissionError('unseen_test is locked; only the N3 benchmark (allow_unseen=True) may read it')
    cache, layers = Path(cache or C.CACHE), tuple(int(l) for l in layers)
    ep = episodes_table(cache)
    ep = ep[ep.split == split].reset_index(drop=True)
    if not len(ep):
        raise ValueError(f'no episodes in {split}')
    keep = np.minimum(ep.length.to_numpy(), ep.task_min_step.to_numpy()) if horizon == 'task_min_step' \
        else ep.length.to_numpy()
    n = int(keep.sum())
    D = None
    out = None
    if normalize:
        z = np.load(cache / 'norm_stats' / f'{pool}.npz')
        where = {int(l): i for i, l in enumerate(z['layers'])}
        mean = z['mean'][[where[l] for l in layers]].astype(np.float32)
        std = np.maximum(z['std'][[where[l] for l in layers]].astype(np.float32), 1e-6)
    for k, layer in enumerate(layers):
        path = cache / 'features' / pool / split / f'L{layer:02d}.npy'
        arr = np.load(path, mmap_mode='r')
        if out is None:
            D = arr.shape[1]
            out = np.empty((n, len(layers), D), dtype=np.float32)
        cursor = 0
        for start, k_len in zip(ep.row_start.to_numpy(), keep):
            out[cursor:cursor + k_len, k] = arr[start:start + k_len]
            cursor += k_len
        if normalize:
            out[:, k] -= mean[k]
            out[:, k] /= std[k]
    if not np.isfinite(out).all():
        raise ValueError(f'non-finite features in {split} layers {layers}')
    lengths = torch.as_tensor(keep, dtype=torch.long)
    starts = torch.cumsum(lengths, 0) - lengths
    return SplitData(split, layers, torch.from_numpy(out).to(device), starts.to(device), lengths.to(device),
                     torch.as_tensor(ep.failure.to_numpy(), dtype=torch.float32, device=device),
                     ep.task_id.to_numpy(), list(ep.trajectory_id))


def class_weights(train: SplitData) -> tuple[float, float]:
    """SAFE RolloutDataset weights: (w_fail, w_success) = 1/((count+1)/N)."""
    n = train.n_episodes
    n_fail = float(train.labels.sum())
    return n / (n_fail + 1), n / (n - n_fail + 1)
