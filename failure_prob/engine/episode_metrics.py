"""B4: episode-level failure-detection metrics at the task_min_step horizon (SAFE-equivalent).

SAFE scores a detector per EPISODE, not per timestep (failure_prob/utils/metrics.py::eval_scores_roc_prc,
"falert_early"):

    episode score = max over steps [0, horizon) of the per-step failure score
    horizon       = task_min_step = the shortest episode length of that task (LIBERO failures always run to 520
                    steps, so looking past the shortest episode would let episode length give the label away)

then ROC-AUC, PR-AUC (sklearn auc(recall, precision), as SAFE computes it, not average precision) and TPR at 5% FPR
over episodes, plus SAFE's task-wise mean ROC-AUC. tests/test_episode_metrics.py checks equality with SAFE's own
eval_scores_roc_prc on identical scores.

Scores may be logits or probabilities: max and ROC/PR are unchanged by any increasing transform.

Two ways to use it:
    episode_metrics(step_scores, labels, horizons, task_ids)            # plain function
    --evaluator failure_prob.engine.episode_metrics:make_evaluator        # depth-sweep plugin (PR #4 contract)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

TPR_FPR_TARGET = 0.05
METRIC_KEYS = ("roc_auc", "pr_auc", "tpr_at_5_fpr", "roc_auc_taskmean")


def episode_scores(step_scores: Sequence[np.ndarray], horizons: Sequence[int]) -> np.ndarray:
    """One score per episode: the max per-step score over steps [0, horizon)."""
    if len(step_scores) != len(horizons):
        raise ValueError(f"{len(step_scores)} score arrays but {len(horizons)} horizons")
    out = np.empty(len(step_scores), dtype=np.float64)
    for i, (s, h) in enumerate(zip(step_scores, horizons)):
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        h = int(h)
        if h < 1:
            raise ValueError(f"episode {i}: horizon must be >= 1, got {h}")
        if len(s) < h:
            raise ValueError(f"episode {i}: {len(s)} step scores but horizon {h}; scores must cover the horizon")
        head = s[:h]
        if not np.all(np.isfinite(head)):
            raise ValueError(f"episode {i}: non-finite score before the horizon")
        out[i] = head.max()
    return out


def _ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    """(ROC-AUC, PR-AUC as SAFE computes it, TPR at 5% FPR). NaN when only one class is present."""
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan"), float("nan"), float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    ok = fpr <= TPR_FPR_TARGET + 1e-12
    return float(auc(fpr, tpr)), float(auc(recall, precision)), float(tpr[ok].max()) if ok.any() else 0.0


def episode_metrics(step_scores: Sequence[np.ndarray], labels, horizons, task_ids=None) -> dict[str, float]:
    """Episode-level metrics. labels: 1 = failure, 0 = success, one per episode.

    Returns floats only: roc_auc (primary), pr_auc, tpr_at_5_fpr, roc_auc_taskmean (mean of per-task ROC-AUCs over
    tasks that have both outcomes; NaN if none, or if task_ids is None).
    """
    y = np.asarray(labels).reshape(-1)
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 (success) or 1 (failure)")
    y = y.astype(int)
    scores = episode_scores(step_scores, horizons)
    if y.size != scores.size:
        raise ValueError(f"{y.size} labels but {scores.size} episodes")
    roc, pr, tpr5 = _ranking_metrics(y, scores)
    taskmean = float("nan")
    if task_ids is not None:
        t = np.asarray(task_ids).reshape(-1)
        if t.size != y.size:
            raise ValueError(f"{t.size} task ids but {y.size} episodes")
        per_task = [_ranking_metrics(y[t == k], scores[t == k])[0] for k in np.unique(t)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            taskmean = float(np.nanmean(per_task)) if not np.all(np.isnan(per_task)) else float("nan")
    return {"roc_auc": roc, "pr_auc": pr, "tpr_at_5_fpr": tpr5, "roc_auc_taskmean": taskmean}


@dataclass
class EpisodeEvaluator:
    """Depth-sweep evaluator plugin (contract in failure_prob/engine/sweep_plugins.py, PR #4).

    Called with an EvalInput: step_scores (full-length per-step scores), labels, horizons (min(length,
    task_min_step)), task_ids, trajectory_ids. Duck-typed, so this module does not import PR #4's code.
    """
    name: str = "episode_max_at_task_min_step_v1"
    temporary: bool = False

    def __call__(self, data) -> dict[str, float]:
        return episode_metrics(data.step_scores, data.labels, data.horizons, data.task_ids)


def make_evaluator() -> EpisodeEvaluator:
    """Factory for `--evaluator failure_prob.engine.episode_metrics:make_evaluator`."""
    return EpisodeEvaluator()
