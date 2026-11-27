"""Detector heads. All return per-timestep logits; the trainer applies SAFE scoring.

Two objectives, both verbatim from the original SAFE code (see tests):
  cumsum - IndepModel: score_t = sum_{s<=t} sigmoid(logit_s); L1-style loss.
  bce    - LstmModel (cumsum=False): score_t = sigmoid(logit_t); per-step BCE.
Rollout score at a horizon = max of score_t over the prefix (for cumsum this is
the last value). Every model is causal: score_t depends only on inputs <= t.

Per-timestep models implement step_logits(x[n,K,D], keep[n,K] | None).
Sequence models implement seq_logits(x[B,T,K,D], mask[B,T]).
Fusion models also implement embed(x) -> [n, H], the pre-classifier fused
representation consumed (frozen) by the temporal / prompt experiments.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from . import config as C


class Detector(nn.Module):
    objective = 'cumsum'
    sequential = False

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- probes
class ProbeMLP(Detector):
    """SAFE IndepModel (n_layers=2, sigmoid, cumsum); same parameter names."""

    def __init__(self, input_dim: int, hidden: int = C.HIDDEN, **_):
        super().__init__()
        self.projector = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def step_logits(self, x, keep=None):
        return self.projector(x[:, 0]).squeeze(-1)

    def embed(self, x):
        return self.projector[1](self.projector[0](x[:, 0]))


# --------------------------------------------------------------------------- fusion
class FusionBase(Detector):
    """Per-layer projections D->H, a fusion rule, ReLU, Linear(H,1).

    With K=1 and identity fusion this is exactly the SAFE MLP probe, so fusion
    gains are measured against the same head.
    """

    def __init__(self, input_dim: int, n_layers: int, hidden: int = C.HIDDEN, **_):
        super().__init__()
        self.K = n_layers
        self.proj = nn.ModuleList(nn.Linear(input_dim, hidden) for _ in range(n_layers))
        self.out = nn.Linear(hidden, 1)

    def project(self, x):
        return torch.stack([p(x[:, k]) for k, p in enumerate(self.proj)], dim=1)   # [n,K,H]

    def fuse(self, e, keep):
        raise NotImplementedError

    def fusion_weights(self, x):
        return None

    def embed(self, x, keep=None):
        return F.relu(self.fuse(self.project(x), keep))

    def step_logits(self, x, keep=None):
        return self.out(self.embed(x, keep)).squeeze(-1)


def _masked_softmax(logits, keep):
    if keep is not None:
        logits = logits.masked_fill(~keep, float('-inf'))
    return logits.softmax(dim=-1)


class StaticFusion(FusionBase):
    """F3: one learned softmax weight per layer, shared across timesteps/inputs."""

    def __init__(self, input_dim, n_layers, hidden=C.HIDDEN, **_):
        super().__init__(input_dim, n_layers, hidden)
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))

    def fuse(self, e, keep):
        w = _masked_softmax(self.layer_logits.expand(e.shape[0], -1), keep)
        return (w[..., None] * e).sum(1)

    def fusion_weights(self, x):
        return self.layer_logits.softmax(-1).expand(x.shape[0], -1)


class GatedFusion(FusionBase):
    """F4: per-timestep, input-conditioned softmax gates over projected layers."""

    def __init__(self, input_dim, n_layers, hidden=C.HIDDEN, gate_hidden: int = 64, **_):
        super().__init__(input_dim, n_layers, hidden)
        self.gate = nn.Sequential(nn.Linear(n_layers * hidden, gate_hidden), nn.ReLU(),
                                  nn.Linear(gate_hidden, n_layers))

    def _gates(self, e, keep):
        return _masked_softmax(self.gate(e.flatten(1)), keep)

    def fuse(self, e, keep):
        return (self._gates(e, keep)[..., None] * e).sum(1)

    def fusion_weights(self, x):
        return self._gates(self.project(x), None)


class ResidualFusion(FusionBase):
    """F5: z = P_a(x_a) + sum_c alpha_c * W_c relu(P_c(x_c)); layer 0 is the anchor.

    alpha_c start at 0 (ReZero), so training starts from the best single layer
    and learns additive complements; all parts are trained end to end.
    """

    def __init__(self, input_dim, n_layers, hidden=C.HIDDEN, **_):
        super().__init__(input_dim, n_layers, hidden)
        self.branch = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(n_layers - 1))
        self.alpha = nn.Parameter(torch.zeros(n_layers - 1))

    def fuse(self, e, keep):
        terms = [e[:, 0]] + [self.alpha[c] * self.branch[c](F.relu(e[:, c + 1])) for c in range(self.K - 1)]
        terms = torch.stack(terms, 1)
        if keep is not None:
            terms = terms * keep[..., None]
        return terms.sum(1)

    def fusion_weights(self, x):
        e = self.project(x)
        norms = torch.stack([e[:, 0].norm(dim=-1)] + [
            (self.alpha[c] * self.branch[c](F.relu(e[:, c + 1]))).norm(dim=-1) for c in range(self.K - 1)], 1)
        return norms / norms.sum(1, keepdim=True)


class AttentionFusion(FusionBase):
    """X1: self-attention across the K layer tokens of one timestep (never across time)."""

    def __init__(self, input_dim, n_layers, hidden=C.HIDDEN, heads: int = 4, **_):
        super().__init__(input_dim, n_layers, hidden)
        self.layer_embedding = nn.Parameter(torch.randn(n_layers, hidden) * 0.02)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)

    def fuse(self, e, keep):
        tokens = e + self.layer_embedding
        pad = None if keep is None else ~keep
        h, _ = self.attn(tokens, tokens, tokens, key_padding_mask=pad, need_weights=False)
        h = self.norm(tokens + h)
        if keep is None:
            return h.mean(1)
        k = keep[..., None].float()
        return (h * k).sum(1) / k.sum(1)


# --------------------------------------------------------------------------- temporal
class SafeLSTM(Detector):
    """SAFE LstmModel (1 layer, full history, cumsum=False, dropout 0); same names."""
    objective = 'bce'
    sequential = True

    def __init__(self, input_dim, hidden=C.HIDDEN, **_):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, 1, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def seq_logits(self, x, mask):
        out, _ = self.lstm(x[:, :, 0])
        return self.fc(out).squeeze(-1)


class TemporalGRU(Detector):
    """Rows 29/30: causal GRU on the frozen winner embedding (+ first differences)."""
    objective = 'bce'
    sequential = True

    def __init__(self, input_dim, hidden=C.HIDDEN, delta: bool = False, **_):
        super().__init__()
        self.delta = delta
        self.gru = nn.GRU(input_dim * (2 if delta else 1), hidden, 1, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def seq_logits(self, x, mask):
        z = x[:, :, 0]
        if self.delta:   # d_0 = 0; padding is only to the right so valid steps are exact
            d = torch.cat([torch.zeros_like(z[:, :1]), z[:, 1:] - z[:, :-1]], 1)
            z = torch.cat([z, d], -1)
        out, _ = self.gru(z)
        return self.fc(out).squeeze(-1)


def sinusoidal(T, d, device):
    pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class CausalTransformer(Detector):
    """Row 31: one pre-norm block, 4 heads, strict causal mask, sinusoidal positions."""
    objective = 'bce'
    sequential = True

    def __init__(self, input_dim, hidden=C.HIDDEN, heads: int = 4, **_):
        super().__init__()
        self.inp = nn.Linear(input_dim, hidden)
        self.block = nn.TransformerEncoderLayer(hidden, heads, 2 * hidden, dropout=0.0,
                                                batch_first=True, norm_first=True)
        self.fc = nn.Linear(hidden, 1)

    def seq_logits(self, x, mask):
        z = x[:, :, 0]
        T = z.shape[1]
        h = self.inp(z) + sinusoidal(T, self.fc.in_features, z.device)
        causal = torch.triu(torch.full((T, T), float('-inf'), device=z.device), diagonal=1)
        h = self.block(h, src_mask=causal, is_causal=True)
        return self.fc(h).squeeze(-1)


# --------------------------------------------------------------------------- prompts
class _EmbedHead(Detector):
    def __init__(self, input_dim, hidden=C.HIDDEN):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def adapt(self, z):
        raise NotImplementedError

    def step_logits(self, x, keep=None):
        z = x[:, 0]
        return self.head(z + self.adapt(z)).squeeze(-1)


class PromptHead(_EmbedHead):
    """Row 32: input-conditioned mixture of N learned failure prompts added to z."""

    def __init__(self, input_dim, hidden=C.HIDDEN, n_prompts: int = C.N_PROMPTS, **_):
        super().__init__(input_dim, hidden)
        self.query = nn.Linear(input_dim, input_dim)
        self.prompts = nn.Parameter(torch.randn(n_prompts, input_dim) * 0.02)

    def adapt(self, z):
        a = (self.query(z) @ self.prompts.T / math.sqrt(z.shape[-1])).softmax(-1)
        return a @ self.prompts


def matched_adapter_rank(input_dim: int, n_prompts: int = C.N_PROMPTS) -> int:
    prompt_extra = input_dim * input_dim + input_dim + n_prompts * input_dim
    return max(1, round((prompt_extra - input_dim) / (2 * input_dim + 1)))


class AdapterHead(_EmbedHead):
    """Row 32 control: residual bottleneck adapter with prompt-matched parameters."""

    def __init__(self, input_dim, hidden=C.HIDDEN, rank: int | None = None, **_):
        super().__init__(input_dim, hidden)
        rank = rank or matched_adapter_rank(input_dim)
        self.down = nn.Linear(input_dim, rank)
        self.up = nn.Linear(rank, input_dim)

    def adapt(self, z):
        return self.up(F.relu(self.down(z)))


REGISTRY = {
    'probe': ProbeMLP, 'probe_wide': ProbeMLP, 'static': StaticFusion, 'gated': GatedFusion,
    'residual': ResidualFusion, 'attention': AttentionFusion, 'lstm': SafeLSTM,
    'gru': TemporalGRU, 'gru_delta': TemporalGRU, 'causal_tf': CausalTransformer,
    'prompt': PromptHead, 'adapter': AdapterHead,
}
FUSION_MODELS = {'static', 'gated', 'residual', 'attention'}


def build(kind: str, **kwargs) -> Detector:
    if kind == 'gru_delta':
        kwargs['delta'] = True
    return REGISTRY[kind](**kwargs)


def width_for_params(input_dim: int, target: int) -> int:
    """Hidden width of a 2-layer SAFE MLP with ~target trainable parameters."""
    return max(1, round((target - 1) / (input_dim + 2)))
