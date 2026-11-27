"""Unit tests for failure_prob.suite: SAFE equivalence, causality, masking, metrics, guards."""
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from failure_prob.suite import config as C
from failure_prob.suite import models as M
from failure_prob.suite.data import SplitData
from failure_prob.suite.metrics import binary_metrics, bootstrap_auc_ci, episode_scores, fast_auc, threshold_at_fpr
from failure_prob.suite.train import Batch, layer_keep, predict, reg_loss, safe_loss, step_scores

try:  # SAFE's model/base.py imports wandb but the tested code paths never call it
    import wandb  # noqa: F401
except ImportError:
    sys.modules['wandb'] = types.ModuleType('wandb')


def toy(B=5, T=7, K=1, D=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.tensor([T, T - 2, 3, T - 1, 1][:B])
    starts = torch.cumsum(lengths, 0) - lengths
    x = torch.randn(int(lengths.sum()), K, D, generator=g)
    labels = torch.tensor([1., 0., 1., 0., 0.][:B])
    return SplitData('train', tuple(range(K)), x, starts, lengths, labels, np.arange(B) % 2, [str(i) for i in range(B)])


def safe_conf():
    """SAFE's original conf dataclasses only import on Python <= 3.10 (mutable defaults)."""
    try:
        import failure_prob.conf as conf
    except ValueError as e:
        pytest.skip(f'original SAFE conf not importable on this Python: {e}')
    return conf


def safe_batch(data):
    b = Batch(data, torch.arange(data.n_episodes), True)
    return b, dict(features=b.x[:, :, 0], valid_masks=b.mask.float(), success_labels=1 - b.labels)


def test_probe_matches_safe_indep_model():
    IndepModelConfig = safe_conf().IndepModelConfig
    from failure_prob.model.indep import IndepModel
    data = toy()
    ref = IndepModel(SimpleNamespace(model=IndepModelConfig(hidden_dim=8, lambda_reg=0.1)), 4)
    ours = M.ProbeMLP(4, 8)
    ours.load_state_dict(ref.state_dict())
    b, sb = safe_batch(data)
    w = [1.7, 0.6]
    ref_loss, _ = ref.forward_compute_loss(sb, w)
    ref_reg, _ = ref.compute_regularization_loss(0.1)
    flat = Batch(data, torch.arange(data.n_episodes), False)
    loss = safe_loss(flat.logits(ours), flat.mask, flat.labels, 'cumsum', w[0], w[1])
    assert torch.allclose(loss, ref_loss, atol=1e-6)
    assert torch.allclose(reg_loss(ours, 0.1), ref_reg, atol=1e-6)
    ref_scores = ref(sb).squeeze(-1)
    ours_scores = step_scores(flat.logits(ours), flat.mask, 'cumsum')
    assert torch.allclose(ours_scores[b.mask], ref_scores[b.mask], atol=1e-5)


def test_lstm_matches_safe_lstm_model():
    LstmModelConfig = safe_conf().LstmModelConfig
    from failure_prob.model.lstm import LstmModel
    data = toy()
    ref = LstmModel(SimpleNamespace(model=LstmModelConfig(hidden_dim=8)), 4)
    ours = M.SafeLSTM(4, 8)
    ours.load_state_dict(ref.state_dict())
    b, sb = safe_batch(data)
    w = [1.7, 0.6]
    ref_loss, _ = ref.forward_compute_loss(sb, w)
    loss = safe_loss(b.logits(ours), b.mask, b.labels, 'bce', w[0], w[1])
    assert torch.allclose(loss, ref_loss, atol=1e-5)
    assert torch.allclose(step_scores(b.logits(ours), b.mask, 'bce')[b.mask], ref(sb).squeeze(-1)[b.mask], atol=1e-6)


def all_models(K=3, D=4):
    yield M.ProbeMLP(D, 8), 1
    for cls in (M.StaticFusion, M.GatedFusion, M.ResidualFusion, M.AttentionFusion):
        yield cls(D, K, 8), K
    yield M.SafeLSTM(D, 8), 1
    yield M.TemporalGRU(D, 8), 1
    yield M.TemporalGRU(D, 8, delta=True), 1
    yield M.CausalTransformer(D, 8, heads=2), 1
    yield M.PromptHead(D, 8), 1
    yield M.AdapterHead(D, 8), 1


@pytest.mark.parametrize('idx', range(11))
def test_causal_padding_and_backward(idx):
    model, K = list(all_models())[idx]
    torch.manual_seed(0)
    data = toy(K=K)
    full = predict(model, data)
    # Prefix-only inputs give identical scores (no future leakage), batch padding has no effect.
    cut = SplitData('x', data.layers, data.x, data.starts, torch.clamp(data.lengths - 2, min=1), data.labels,
                    data.task_ids, data.trajectory_ids)
    for f, c in zip(full, predict(model, cut)):
        np.testing.assert_allclose(f[:len(c)], c, rtol=1e-4, atol=1e-5)
    alone = predict(model, SplitData('x', data.layers, data.x, data.starts[2:3], data.lengths[2:3],
                                     data.labels[2:3], data.task_ids[2:3], ['2']))
    np.testing.assert_allclose(alone[0], full[2], rtol=1e-4, atol=1e-5)
    model.train()
    b = Batch(data, torch.arange(data.n_episodes), model.sequential)
    keep = layer_keep(b.x.shape[0], K, 0.5, torch.Generator().manual_seed(0), 'cpu') if K > 1 else None
    loss = safe_loss(b.logits(model, keep), b.mask, b.labels, model.objective, 1.0, 1.0) + reg_loss(model, 0.01)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_fusion_k1_static_equals_probe_architecture():
    assert M.StaticFusion(4096, 1).n_params() - 1 == M.ProbeMLP(4096).n_params() == 1049089


def test_adapter_matches_prompt_params():
    p, a = M.PromptHead(256), M.AdapterHead(256)
    assert abs(p.n_params() - a.n_params()) / p.n_params() < 0.05


def test_width_for_params():
    target = M.GatedFusion(4096, 3).n_params()
    w = M.width_for_params(4096, target)
    assert abs(M.ProbeMLP(4096, w).n_params() - target) / target < 0.05


def test_layer_keep_never_empty():
    keep = layer_keep(1000, 3, 0.9, torch.Generator().manual_seed(0), 'cpu')
    assert keep.any(1).all() and (~keep).any()


def test_metrics_and_threshold():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 0, 1, 0] * 4)
    scores = np.linspace(0, 1, 40) + labels * 0.3
    m = binary_metrics(labels, scores)
    assert 0.5 < m['roc_auc'] <= 1 and np.isclose(fast_auc(labels[None].astype(float), scores[None])[0], m['roc_auc'])
    thr = threshold_at_fpr(labels, scores)
    assert ((scores >= thr) & (labels == 0)).sum() <= 0.05 * (labels == 0).sum()
    lo, hi = bootstrap_auc_ci(labels, [scores, scores], n_boot=100)
    assert lo <= m['roc_auc'] <= hi
    assert binary_metrics([0, 0], [0.1, 0.2])['roc_auc'] is None


def test_episode_scores_use_prefix_only():
    s = [np.array([0., 5., 1., 1., 9.])]
    assert episode_scores(s, 0.25)[0] == 5.0         # index round(4*0.25)=1, max-so-far
    assert episode_scores(s, 0.5, 'at')[0] == 1.0
    assert episode_scores(s, 1.0)[0] == 9.0


def test_unseen_locked():
    from failure_prob.suite.data import load_split
    with pytest.raises(PermissionError):
        load_split('unseen_test', [32])


def test_grid_and_registry():
    assert len(C.grid(C.EXPERIMENTS['F4_GATED'])) == 12
    assert len(C.grid(C.EXPERIMENTS['X4_LDROP'])) == 36
    assert [C.EXPERIMENTS[f'P_L{l:02d}'].row.split('/')[0] for l in (4, 8, 12, 16, 20, 24)] == \
        ['5', '6', '7', '8', '9', '10']
    for stage in C.STAGES.values():
        assert all(n in C.EXPERIMENTS for n in stage)
