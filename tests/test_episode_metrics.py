"""B4 evaluator: SAFE parity, horizon handling, edge cases, and the depth-sweep plugin contract."""
import math
import os
from types import SimpleNamespace as NS

import numpy as np
import pytest
from sklearn.metrics import auc, average_precision_score, precision_recall_curve

from failure_prob.engine.episode_metrics import METRIC_KEYS, episode_metrics, episode_scores, make_evaluator


def fake_episodes(seed=0, n_tasks=5, per_task=20, max_len=60):
    """Failures run to max_len (as in LIBERO), successes stop early; scores drift up for failures."""
    rng = np.random.default_rng(seed)
    scores, labels, lengths, tasks = [], [], [], []
    for t in range(n_tasks):
        for _ in range(per_task):
            f = int(rng.random() < 0.45)
            n = max_len if f else int(rng.integers(20, max_len - 5))
            drift = np.linspace(0, 1.0 if f else 0.3, n)
            scores.append(rng.normal(size=n) * 0.7 + drift)
            labels.append(f); lengths.append(n); tasks.append(t)
    lengths, tasks = np.array(lengths), np.array(tasks)
    task_min_step = np.array([lengths[tasks == t].min() for t in tasks])
    horizons = np.minimum(lengths, task_min_step)
    return scores, np.array(labels), lengths, horizons, tasks


def test_parity_with_safe_eval_scores_roc_prc():
    """Same numbers as SAFE's own scorer (failure_prob/utils/metrics.py) on identical scores."""
    os.environ.setdefault("WANDB_MODE", "disabled")
    from failure_prob.utils.metrics import eval_scores_roc_prc
    for seed in range(3):
        s, y, n, h, t = fake_episodes(seed)
        rollouts = [NS(task_id=int(tt), task_min_step=int(hh), episode_success=1 - int(yy),
                       hidden_states=np.zeros(nn)) for tt, hh, yy, nn in zip(t, h, y, n)]
        safe = eval_scores_roc_prc({"seen_val": rollouts}, {"seen_val": s}, "x", [1.0],
                                   plot_auc_curves=False, plot_score_curves=False)
        ours = episode_metrics(s, y, h, t)
        assert ours["roc_auc"] == pytest.approx(safe["falert_early_roc_auc/x_seen_val"], abs=1e-12)
        assert ours["pr_auc"] == pytest.approx(safe["falert_early_prc_auc/x_seen_val"], abs=1e-12)
        assert ours["roc_auc_taskmean"] == pytest.approx(safe["falert_early_roc_auc_taskwise/x_seen_val"], abs=1e-12)


def test_pr_auc_is_safes_definition_not_average_precision():
    s, y, n, h, t = fake_episodes(4)
    e = episode_scores(s, h)
    p, r, _ = precision_recall_curve(y, e)
    m = episode_metrics(s, y, h, t)
    assert m["pr_auc"] == pytest.approx(auc(r, p), abs=1e-12)
    assert m["pr_auc"] != pytest.approx(average_precision_score(y, e), abs=1e-6)


def test_horizon_blocks_length_leakage():
    """A detector that fires only after the task horizon must score chance, however perfect it is later."""
    s, y, n, h, t = fake_episodes(1)
    late = [np.concatenate([np.zeros(hh), np.full(nn - hh, float(yy))]) for nn, hh, yy in zip(n, h, y)]
    assert episode_metrics(late, y, h, t)["roc_auc"] == 0.5
    assert episode_metrics(late, y, n, t)["roc_auc"] == 1.0      # whole-episode scoring would be fooled


def test_perfect_inverted_and_tpr():
    s, y, n, h, t = fake_episodes(2)
    perfect = [np.full(nn, float(yy)) for nn, yy in zip(n, y)]
    m = episode_metrics(perfect, y, h, t)
    assert m["roc_auc"] == 1.0 and m["tpr_at_5_fpr"] == 1.0
    inverted = [np.full(nn, 1.0 - yy) for nn, yy in zip(n, y)]
    m = episode_metrics(inverted, y, h, t)
    assert m["roc_auc"] == 0.0 and m["tpr_at_5_fpr"] == 0.0


def test_logits_and_probabilities_give_identical_metrics():
    s, y, n, h, t = fake_episodes(3)
    probs = [1 / (1 + np.exp(-x)) for x in s]
    a, b = episode_metrics(s, y, h, t), episode_metrics(probs, y, h, t)
    for k in METRIC_KEYS:
        assert a[k] == pytest.approx(b[k], abs=1e-12)


def test_single_class_and_task_mean():
    s = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
    m = episode_metrics(s, [1, 1], [2, 2], [0, 0])
    assert all(math.isnan(m[k]) for k in METRIC_KEYS)            # bootstrap resamples can be single-class
    # task 0 has both outcomes, task 1 only failures: task mean uses task 0 only
    s = [np.array([0.9]), np.array([0.1]), np.array([0.5]), np.array([0.6])]
    m = episode_metrics(s, [1, 0, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1])
    assert m["roc_auc_taskmean"] == 1.0
    assert math.isnan(episode_metrics(s, [1, 0, 1, 1], [1, 1, 1, 1])["roc_auc_taskmean"])   # no task ids


def test_returns_plain_floats_with_standard_keys():
    s, y, n, h, t = fake_episodes(5)
    m = episode_metrics(s, y, h, t)
    assert tuple(m) == METRIC_KEYS and all(type(v) is float for v in m.values())


@pytest.mark.parametrize("bad, msg", [
    (dict(horizons=[3, 5]), "scores must cover the horizon"),
    (dict(horizons=[0, 2]), "horizon must be >= 1"),
    (dict(scores=[np.array([np.nan, 1.0, 2.0]), np.array([1.0, 2.0, 3.0, 4.0])]), "non-finite"),
    (dict(labels=[1, 2]), "labels must be 0"),
    (dict(labels=[1]), "1 labels but 2 episodes"),
])
def test_rejects_bad_input(bad, msg):
    args = dict(scores=[np.arange(3.0), np.arange(4.0)], labels=[1, 0], horizons=[2, 2])
    args.update(bad)
    with pytest.raises(ValueError, match=msg):
        episode_metrics(args["scores"], args["labels"], args["horizons"])


def test_nan_after_horizon_is_ignored():
    """Scores after the horizon are never read (padding or later steps may hold anything)."""
    s = [np.array([0.9, np.nan]), np.array([0.1, np.inf])]
    assert episode_metrics(s, [1, 0], [1, 1])["roc_auc"] == 1.0


def test_depth_sweep_plugin_contract():
    """PR #4's sweep resolves `module:factory`, needs `name` and `temporary`, and calls it with an EvalInput."""
    ev = make_evaluator()
    assert ev.name == "episode_max_at_task_min_step_v1" and ev.temporary is False
    s, y, n, h, t = fake_episodes(6)
    data = NS(step_scores=s, labels=y, horizons=h, task_ids=t, trajectory_ids=[f"e{i}" for i in range(len(s))])
    assert ev(data) == episode_metrics(s, y, h, t)
