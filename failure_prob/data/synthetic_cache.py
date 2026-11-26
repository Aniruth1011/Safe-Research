"""B8: synthetic test data generated from the data contract (failure_prob/data/contract.py).

Writes fake rollouts in exactly the format the patched OpenVLA rollout script writes (per episode .layers.pt with
`layer_names`, `pools`, `run_fingerprint`, ..., the SAFE .pkl and the per-step .csv), then builds them into a cache
with the real collection/build_cache.py. So a synthetic cache has the same files, layer names, poolings, splits,
tables and statistics as cache v1, just small. Tests use it instead of hand-rolled fakes, and
tests/test_data_contract.py checks it against the contract, so a change on either side fails a test.

Planted structure, so probes and fusion have something to find (numbers mean nothing about OpenVLA):
  * failed episodes drift over time in layers 20-31 and 33; layer 32 is an RMS-normalised copy of 33 (like the
    real post-norm layer);
  * layer 0 with 'first' pooling is constant (as in v1: the same prompt token every step);
  * failures run to max_len, successes stop earlier (length leakage, which is why task_min_step exists).

    from failure_prob.data.synthetic_cache import make_synthetic_cache
    cache = make_synthetic_cache(tmp_path)          # -> tmp_path/cache/v1, satisfies the contract
"""
from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

from failure_prob.data.contract import LAYER_NAMES, NUM_LAYERS, POOLS, UNSEEN_TASKS

REPO = Path(__file__).resolve().parents[2]
ACTION_COLS = ["action/dx", "action/dy", "action/dz", "action/droll", "action/dpitch", "action/dyaw", "action/dgripper"]
SIGNAL_LAYERS = list(range(20, 32)) + [33]


def write_rollouts(out: Path, n_tasks: int = 10, n_episodes: int = 6, dim: int = 16, max_len: int = 30,
                   seed: int = 0) -> Path:
    import torch
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal((NUM_LAYERS, dim))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    prompt_token = rng.standard_normal(dim)                       # L0 'first': constant across all steps
    pools = list(POOLS)
    for t in range(n_tasks):
        task_vec = rng.standard_normal((NUM_LAYERS, dim)) * 0.5
        for e in range(n_episodes):
            fail = e % 2                                          # both outcomes in every task -> both splits valid
            T = max_len if fail else int(rng.integers(max_len // 2, max_len - 1))
            x = rng.standard_normal((T, NUM_LAYERS, dim)) * (0.2 + np.arange(NUM_LAYERS) / NUM_LAYERS)[None, :, None]
            x = x + task_vec[None]
            if fail:
                ramp = np.linspace(0, 1.5, T)[:, None]
                for layer in SIGNAL_LAYERS:
                    x[:, layer] += ramp * direction[layer]
            x[:, 32] = x[:, 33] / np.sqrt((x[:, 33] ** 2).mean(-1, keepdims=True) + 1e-6)   # "post-norm" copy
            per_pool = {"mean": x, "first": x + 0.1, "last": x - 0.1}
            per_pool["first"] = per_pool["first"].copy()
            per_pool["first"][:, 0] = prompt_token
            feats = np.stack([per_pool[p] for p in pools], axis=1)                         # (T, P, L, D)
            stem = out / f"task{t}--ep{e}--succ{1 - fail}"
            torch.save({"features": torch.from_numpy(feats).float().to(torch.bfloat16), "pools": pools,
                        "layer_names": LAYER_NAMES, "task_id": t, "episode_idx": e, "episode_success": bool(1 - fail),
                        "task_description": f"synthetic task {t}", "task_suite_name": "libero_10",
                        "run_fingerprint": {"attn_implementation": "flash_attention_2", "gpu_name": "synthetic",
                                            "render_backend": "synthetic", "versions": {"synthetic_seed": seed}},
                        "init_state_index": e, "seed": seed},
                       stem.with_suffix(".layers.pt"))
            last = torch.from_numpy(x[:, 32]).float().to(torch.bfloat16)
            with open(stem.with_suffix(".pkl"), "wb") as f:
                pickle.dump({"hidden_states": last[:, None].repeat(1, 7, 1), "task_id": t}, f)
            with open(stem.with_suffix(".csv"), "w") as f:
                f.write(",".join(ACTION_COLS) + "\n" + "\n".join(",".join(["0.0"] * 7) for _ in range(T)) + "\n")
    return out


def make_synthetic_cache(root: str | Path, n_tasks: int = 10, n_episodes: int = 6, dim: int = 16, max_len: int = 30,
                         seed: int = 0, version: str = "v1") -> Path:
    """Fake rollouts -> real build_cache.py -> root/cache/<version>. Returns the cache folder."""
    if n_tasks <= max(UNSEEN_TASKS):
        raise ValueError(f"need more than {max(UNSEEN_TASKS)} tasks so unseen tasks {UNSEEN_TASKS} exist")
    root = Path(root)
    raw = write_rollouts(root / "rollouts", n_tasks, n_episodes, dim, max_len, seed)
    cmd = [sys.executable, str(REPO / "collection" / "build_cache.py"), "--rollout-dir", str(raw),
           "--cache-root", str(root / "cache"), "--version", version, "--expect-tasks", str(n_tasks),
           "--expect-episodes", str(n_episodes), "--checkpoint", "synthetic"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"build_cache.py failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return root / "cache" / version


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build a small contract-conforming synthetic cache.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(make_synthetic_cache(a.out, n_episodes=a.episodes, dim=a.dim, seed=a.seed))
