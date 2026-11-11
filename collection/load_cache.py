#!/usr/bin/env python
"""Read the shared cache, or export it into Ani's FeatureStore format.

Python use:
    from load_cache import SharedCache
    cache = SharedCache("/path/to/cache/v1")
    train = cache.records("train", layers=[4, 24, 32], pool="mean")      # list of FeatureStore-style records
    val   = cache.records("seen_val", layers=[4, 24, 32], pool="mean", normalize=True)

CLI export (one FeatureStore folder per split, usable with `scripts/train.py --feature-root`):
    python load_cache.py export --cache /path/to/cache/v1 --out /path/to/fs --layers 4 24 32 --pool mean

Records match failure_prob/data/feature_store.py: features [1,T,K,D] float32, mask [1,T] bool,
labels tensor([failure]), task_ids [task], trajectory_ids [id].
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SPLITS = ("train", "seen_val", "unseen_test")


class SharedCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.episodes = pd.read_csv(self.root / "episodes.csv")
        self.layer_names = manifest_layer_names(self.manifest)  # one name per layer index (33 or 34)
        self._mm: dict = {}

    def _array(self, pool: str, split: str, layer: int) -> np.ndarray:
        key = (pool, split, layer)
        if key not in self._mm:
            path = self.root / "features" / pool / split / f"L{layer:02d}.npy"
            if not path.exists():
                hint = " (unseen features are withheld until the locked final benchmark)" if split == "unseen_test" else ""
                raise FileNotFoundError(f"{path} not present in this copy of the cache{hint}")
            self._mm[key] = np.load(path, mmap_mode="r")
        return self._mm[key]

    def norm_stats(self, pool: str, layers: list[int]) -> tuple[np.ndarray, np.ndarray]:
        z = np.load(self.root / "norm_stats" / f"{pool}.npz")
        pos = {int(l): i for i, l in enumerate(z["layers"])}
        idx = [pos[l] for l in layers]
        return z["mean"][idx], np.maximum(z["std"][idx], 1e-6)

    def records(self, split: str, layers: list[int], pool: str = "mean", normalize: bool = False,
                horizon: str = "full", allow_unseen: bool = False) -> list[dict]:
        """horizon='full' keeps whole rollouts; 'task_min_step' truncates to SAFE's per-task horizon.
        unseen_test is locked unless allow_unseen=True (only for the final locked benchmark, Exp 37)."""
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if split == "unseen_test" and not allow_unseen:
            raise PermissionError("unseen_test is locked until the final benchmark (Exp 37); pass allow_unseen=True only then")
        bad = [l for l in layers if not 0 <= l < len(self.layer_names)]
        if bad:
            raise ValueError(f"layers {bad} not in this cache (indices 0..{len(self.layer_names) - 1}: {self.layer_names[-1]})")
        arrays = [self._array(pool, split, l) for l in layers]
        mean = std = None
        if normalize:
            mean, std = self.norm_stats(pool, layers)
        out = []
        for ep in self.episodes[self.episodes.split == split].itertuples():
            stop = ep.row_end if horizon == "full" else ep.row_start + min(ep.length, ep.task_min_step)
            x = np.stack([a[ep.row_start:stop] for a in arrays], axis=1).astype(np.float32)  # (T, K, D)
            if normalize:
                x = (x - mean[None]) / std[None]
            t = x.shape[0]
            out.append({
                "features": torch.from_numpy(x)[None],
                "mask": torch.ones(1, t, dtype=torch.bool),
                "labels": torch.tensor([float(ep.failure)]),
                "task_ids": [int(ep.task_id)],
                "trajectory_ids": [ep.trajectory_id],
            })
        return out


def manifest_layer_names(manifest: dict) -> list[str]:
    """`layer_names` from the manifest; caches built before that key existed use the legacy 33-layer convention."""
    if "layer_names" in manifest:
        return list(manifest["layer_names"])
    if manifest.get("num_hidden_states", 33) != 33:
        raise ValueError(f"manifest has no layer_names but {manifest['num_hidden_states']} layers")
    return ["emb"] + [f"block{i:02d}" for i in range(1, 32)] + ["block32_postnorm"]


def export(cache_root: Path, out: Path, layers: list[int], pool: str, splits: list[str], normalize: bool,
           allow_unseen: bool = False):
    try:
        from failure_prob.data.feature_store import FeatureMetadata, FeatureStore
    except ImportError as e:
        raise SystemExit("Install the team repo first (pip install -e Safe-Research) so FeatureStore is importable") from e
    cache = SharedCache(cache_root)
    m = cache.manifest
    for split in splits:
        recs = cache.records(split, layers, pool, normalize=normalize, allow_unseen=allow_unseen)
        meta = FeatureMetadata(
            backbone=m["backbone"], checkpoint=m["checkpoint"], dataset=m["dataset"],
            split_version=f"{m['cache_version']}:{split}", layers=tuple(layers),
            pooling=pool + ("+trainnorm" if normalize else ""), dtype="float32",
            layer_indexing="hf_hidden_states(0=embeddings,32=final)" + (",33=block32_prenorm" if len(cache.layer_names) == 34 else ""),
            num_layers=len(cache.layer_names),
        )
        FeatureStore(out / split).save(recs, meta)
        print(f"{split}: {len(recs)} trajectories -> {out / split}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--cache", required=True, type=Path)
    e.add_argument("--out", required=True, type=Path)
    e.add_argument("--layers", required=True, type=int, nargs="+")
    e.add_argument("--pool", default="mean")
    e.add_argument("--splits", nargs="+", default=["train", "seen_val"])
    e.add_argument("--normalize", action="store_true", help="z-score with TRAIN-split statistics")
    e.add_argument("--allow-unseen", action="store_true", help="only for the final locked benchmark (Exp 37)")
    args = ap.parse_args()
    export(args.cache, args.out, args.layers, args.pool, args.splits, args.normalize, args.allow_unseen)


if __name__ == "__main__":
    main()
