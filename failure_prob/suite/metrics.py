"""Episode-level metrics. Positive class = failure. Scores are raw (no calibration)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from . import config as C


def episode_scores(step_scores: list[np.ndarray], fraction: float = 1.0, mode: str = 'max') -> np.ndarray:
    """Score of each rollout using only its prefix up to round((h-1)*fraction) (SAFE indexing)."""
    out = []
    for s in step_scores:
        i = int(round((len(s) - 1) * fraction))
        out.append(s[:i + 1].max() if mode == 'max' else s[i])
    return np.asarray(out, dtype=np.float64)


def tpr_at_fpr(labels, scores, fpr_target=C.TARGET_FPR) -> float:
    """Highest ROC operating-point TPR with FPR <= target (no interpolation)."""
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(tpr[fpr <= fpr_target].max())


def binary_metrics(labels, scores) -> dict:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.min() == labels.max():
        return dict(n=len(labels), n_fail=int(labels.sum()), roc_auc=None, pr_auc=None, ap=None, tpr_at_5fpr=None)
    p, r, _ = precision_recall_curve(labels, scores)
    return dict(n=len(labels), n_fail=int(labels.sum()),
                roc_auc=float(roc_auc_score(labels, scores)),
                pr_auc=float(auc(r, p)),
                ap=float(average_precision_score(labels, scores)),
                tpr_at_5fpr=tpr_at_fpr(labels, scores))


def threshold_at_fpr(labels, scores, fpr_target=C.TARGET_FPR) -> float:
    """Smallest threshold whose FPR on these (seen-val) successes is <= target; frozen for N3."""
    neg = np.sort(np.asarray(scores)[np.asarray(labels) == 0])
    k = int(np.floor(fpr_target * len(neg)))            # at most k successes may exceed it
    return float(np.nextafter(neg[len(neg) - 1 - k], np.inf)) if k < len(neg) else float(neg.min())


def thresholded(labels, scores, thr) -> dict:
    labels = np.asarray(labels).astype(int)
    pred = np.asarray(scores) >= thr
    tp, fp = int((pred & (labels == 1)).sum()), int((pred & (labels == 0)).sum())
    n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())
    return dict(threshold=thr, tpr=tp / n_pos if n_pos else None, fpr=fp / n_neg if n_neg else None,
                bal_acc=((tp / n_pos) + 1 - fp / n_neg) / 2 if n_pos and n_neg else None)


def fast_auc(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Vectorized Mann-Whitney AUC; scores [R, n] (ties averaged), labels [R, n]."""
    from scipy.stats import rankdata
    ranks = rankdata(scores, axis=-1)
    pos = labels.sum(-1)
    neg = labels.shape[-1] - pos
    with np.errstate(invalid='ignore', divide='ignore'):
        return ((ranks * labels).sum(-1) - pos * (pos + 1) / 2) / (pos * neg)


def bootstrap_auc_ci(labels, seed_scores: list[np.ndarray], n_boot=2000, seed=0, level=0.95):
    """CI of the seed-mean ROC-AUC, resampling episodes (same resample for every seed)."""
    labels = np.asarray(labels).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(labels), size=(n_boot, len(labels)))
    lab = labels[idx]
    aucs = np.mean([fast_auc(lab, np.asarray(s)[idx]) for s in seed_scores], axis=0)
    aucs = aucs[np.isfinite(aucs)]
    lo, hi = np.quantile(aucs, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)
