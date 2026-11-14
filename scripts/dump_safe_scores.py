#!/usr/bin/env python
"""Dump per-episode seen_val failure scores of a trained SAFE baseline (C1-C3), for TPR@5%FPR, bootstrap CIs and C4.

SAFE logs ROC-AUC / PR-AUC but not per-episode scores. Re-run a winning config with train.eval_save_ckpt=True
(one seed per process), then point this script at that run's logs_save_path:

    python scripts/dump_safe_scores.py --run-dir <logs_save_path> --seed 0 --out-dir /scratch1/$USER/safe_baselines/scores

indep / lstm: loads <run-dir>/model_final.ckpt. embed (k-NN): refits on train in-process (deterministic; its
checkpoint holds no features). Scores only seen_val (val_seen); unseen_test is never loaded.

Writes, per model and seed:
  scores_<model>_s<seed>.csv  trajectory_id, task_id, failure, task_min_step, length, score_max_by_tms, score_at_tms
  scores_<model>_s<seed>.npz  full per-step score array per episode, keyed by trajectory_id
score_max_by_tms = max(s[:task_min_step]) (SAFE's falert_early); score_at_tms = s[task_min_step - 1].
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from failure_prob.data import load_rollouts, split_rollouts  # noqa: E402
from failure_prob.data.utils import RolloutDataset  # noqa: E402
from failure_prob.model import get_model  # noqa: E402
from failure_prob.utils.random import seed_everything  # noqa: E402
from failure_prob.utils.routines import model_forward_dataloader  # noqa: E402

SPLIT = "val_seen"


def episode_table(rollouts, scores) -> pd.DataFrame:
    rows = []
    for r, s in zip(rollouts, scores):
        tms = int(r.task_min_step)
        assert 1 <= tms <= len(s), (r.trajectory_id, tms, len(s))
        rows.append({
            "trajectory_id": r.trajectory_id,
            "task_id": r.task_id,
            "failure": 1 - int(r.episode_success),
            "task_min_step": tms,
            "length": len(s),
            "score_max_by_tms": float(np.max(s[:tms])),
            "score_at_tms": float(s[tms - 1]),
        })
    return pd.DataFrame(rows)


def score(cfg, seed: int, run_dir: Path | None, device: str):
    seed_everything(seed)
    rollouts_by_split = split_rollouts(cfg, load_rollouts(cfg))
    assert set(rollouts_by_split) == {"train", SPLIT}, set(rollouts_by_split)
    input_dim = rollouts_by_split["train"][0].hidden_states.shape[-1]

    model = get_model(cfg, input_dim)
    model.to(device)
    if cfg.model.name == "embed":
        train_ds = RolloutDataset(cfg, rollouts_by_split["train"], device="cpu")
        model.train_epoch(None, DataLoader(train_ds, batch_size=cfg.model.batch_size, shuffle=False))
    else:
        ckpt = run_dir / "model_final.ckpt"
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    rollouts = rollouts_by_split[SPLIT]
    loader = DataLoader(RolloutDataset(cfg, rollouts, device="cpu"), batch_size=cfg.model.batch_size, shuffle=False)
    with torch.no_grad():
        s, masks, _ = model_forward_dataloader(model, loader)
    s = s.detach().cpu().numpy()
    lengths = masks.sum(dim=-1).cpu().numpy().astype(int)
    return rollouts, [s[i, :lengths[i]] for i in range(len(lengths))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True, help="logs_save_path with config.yaml (+ model_final.ckpt)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cache-root", default=None, help="override dataset.cache_root from the run's config")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    cfg = OmegaConf.load(args.run_dir / "config.yaml")
    if cfg.dataset.name != "openvla_cache":
        raise SystemExit(f"dataset.name={cfg.dataset.name}; this script only scores openvla_cache runs")
    if args.cache_root:
        cfg.dataset.cache_root = args.cache_root
    cfg.dataset.load_to_cuda = False
    cfg.train.seed = args.seed

    rollouts, scores = score(cfg, args.seed, args.run_dir, args.device)
    df = episode_table(rollouts, scores)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"scores_{cfg.model.name}_s{args.seed}"
    df.to_csv(stem.with_suffix(".csv"), index=False)
    np.savez_compressed(stem.with_suffix(".npz"), **{r.trajectory_id: s for r, s in zip(rollouts, scores)})
    auc = roc_auc_score(df.failure, df.score_max_by_tms)
    auc_at = roc_auc_score(df.failure, df.score_at_tms)
    print(f"{stem}.csv: {len(df)} {SPLIT} episodes; ROC-AUC max_by_tms={auc:.4f} (= SAFE falert_early) "
          f"at_tms={auc_at:.4f}")


if __name__ == "__main__":
    sys.exit(main())
