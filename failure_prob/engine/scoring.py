"""Score a trained detector on cache records at the episode level (B1 + B4), the same way in train.py and evaluate.py."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from failure_prob.data.cache_splits import horizons
from failure_prob.engine.episode_metrics import EpisodeEvaluator, episode_metrics
from failure_prob.engine.trainer import RecordDataset, collate_records

EVALUATOR_NAME = EpisodeEvaluator().name


@torch.no_grad()
def step_scores(model, records, batch_size: int = 8) -> list[np.ndarray]:
    """Per-step failure logits for every record, trimmed to its real length, in record order."""
    model.eval()
    out = []
    loader = DataLoader(RecordDataset(records), batch_size=batch_size, shuffle=False, collate_fn=collate_records)
    for batch in loader:
        logits = model(batch).failure_logits.float().cpu().numpy()
        lengths = batch["mask"].sum(1).tolist()
        out.extend(logits[i, :n].copy() for i, n in enumerate(lengths))
    return out


def evaluate_records(model, records, batch_size: int = 8) -> dict:
    """Episode-level metrics at min(length, task_min_step) (records need `task_min_steps`, see cache_splits)."""
    scores = step_scores(model, records, batch_size)
    labels = np.array([int(r["labels"].reshape(-1)[0]) for r in records])
    task_ids = np.array([r["task_ids"][0] for r in records])
    metrics = episode_metrics(scores, labels, horizons(records), task_ids)
    return {**metrics, "n_episodes": int(len(records)), "n_failures": int(labels.sum()),
            "evaluator": EVALUATOR_NAME, "metric_level": "episode", "horizon": "min(length, task_min_step)"}
