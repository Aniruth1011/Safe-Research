"""Write a synthetic cache in the exact on-disk layout of collection/build_cache.py (for tests and timing only).

Layout: manifest.json, episodes.csv, features/<pool>/<split>/L<NN>.npy (float16 [rows, D]), norm_stats/<pool>.npz.
Failure rollouts carry a planted signal in `signal_layer` only, so a working sweep peaks there. Synthetic numbers
are never results.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "seen_val", "unseen_test")


def make_synthetic_cache(root: Path, *, n_layers: int = 6, dim: int = 16, episodes_per_task: int = 10,
                         tasks: tuple[int, ...] = tuple(range(10)), unseen_tasks: tuple[int, ...] = (0, 3, 5),
                         min_len: int = 12, max_len: int = 30, signal_layer: int = 2, signal: float = 1.5,
                         pools: tuple[str, ...] = ("mean", "first"), layers_to_write: tuple[int, ...] | None = None,
                         splits_to_write: tuple[str, ...] = SPLITS, failure_len: int | None = None,
                         seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    root = Path(root)
    rows = []
    for t in tasks:
        n_train = int(round(0.6 * episodes_per_task))
        for e in range(episodes_per_task):
            failure = int(rng.random() < 0.5)
            length = failure_len if (failure and failure_len) else int(rng.integers(min_len, max_len + 1))
            split = "unseen_test" if t in unseen_tasks else ("train" if e < n_train else "seen_val")
            rows.append(dict(trajectory_id=f"task{t}--ep{e}", task_id=t, task_name=f"synthetic task {t}", episode_idx=e,
                             success=1 - failure, failure=failure, length=length, split=split))
    df = pd.DataFrame(rows)
    df["task_min_step"] = df.groupby("task_id")["length"].transform("min")
    df["row_start"] = 0
    for s in SPLITS:
        idx = df.index[df.split == s]
        df.loc[idx, "row_start"] = np.concatenate([[0], np.cumsum(df.loc[idx, "length"].to_numpy())[:-1]]) if len(idx) else []
    df["row_end"] = df.row_start + df.length
    root.mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "episodes.csv", index=False)

    direction = rng.standard_normal(dim).astype(np.float32)
    direction /= np.linalg.norm(direction)
    layers = list(range(n_layers)) if layers_to_write is None else list(layers_to_write)
    for p in pools:
        sums = np.zeros((n_layers, dim)); sqs = np.zeros((n_layers, dim)); n_train_rows = 0
        for s in splits_to_write:
            sub = df[df.split == s]
            if not len(sub):
                continue
            d = root / "features" / p / s
            d.mkdir(parents=True, exist_ok=True)
            n_rows = int(sub.length.sum())
            for l in layers:
                mm = np.lib.format.open_memmap(d / f"L{l:02d}.npy", mode="w+", dtype=np.float16, shape=(n_rows, dim))
                for ep in sub.itertuples():
                    block = rng.standard_normal((ep.length, dim), dtype=np.float32) + 0.1 * l
                    if l == signal_layer and ep.failure:
                        block += signal * direction
                    mm[ep.row_start:ep.row_end] = block.astype(np.float16)
                    if s == "train":
                        sums[l] += block.sum(0); sqs[l] += (block.astype(np.float64) ** 2).sum(0)
                mm.flush()
                del mm
            if s == "train":
                n_train_rows = n_rows
        (root / "norm_stats").mkdir(exist_ok=True)
        mean = sums / max(n_train_rows, 1)
        std = np.sqrt(np.maximum(sqs / max(n_train_rows, 1) - mean ** 2, 0.0))
        np.savez(root / "norm_stats" / f"{p}.npz", layers=np.arange(n_layers), mean=mean.astype(np.float32),
                 std=std.astype(np.float32), n_rows=n_train_rows)

    names = ["emb"] + [f"block{i:02d}" for i in range(1, n_layers)]
    manifest = {"cache_version": "synthetic", "created_utc": "2026-01-01T00:00:00+00:00", "dataset": "synthetic",
                "backbone": "synthetic", "checkpoint": "synthetic", "num_hidden_states": n_layers,
                "layer_names": names, "pools_exported": list(pools), "feature_dim": dim,
                "splits": {s: int((df.split == s).sum()) for s in SPLITS}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return root
