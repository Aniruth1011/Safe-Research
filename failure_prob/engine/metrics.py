import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve


def binary_metrics(labels, logits):
    y = np.asarray(labels).astype(int); score = 1 / (1 + np.exp(-np.asarray(logits)))
    pred = (score >= 0.5).astype(int)
    result = {"roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan"), "pr_auc": float(average_precision_score(y, score)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "f1": float(f1_score(y, pred, zero_division=0))}
    fpr, tpr, _ = roc_curve(y, score) if len(np.unique(y)) > 1 else (np.array([0, 1]), np.array([0, 1]), None)
    for target in (0.05, 0.10):
        result[f"tpr_at_{int(target*100)}_fpr"] = float(np.max(tpr[fpr <= target], initial=0.0))
    return result
