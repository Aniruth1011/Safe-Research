"""F3 (experiment row 21): static learned layer weights (weighted sum + MLP).

    z[b, t, k]   = P_k(x[b, t, k])            P_k = Linear(D, proj_dim) per layer, or identity when proj_dim is None
    fused[b, t]  = gamma * sum_k softmax(alpha)_k * z[b, t, k]          x: [B, T, K, D]
    logits[b, t] = head(fused[b, t])

``proj_dim`` projects every layer to a shared dimension first (the experiment sheet's "project features to shared
dimension"): each layer gets its own Linear(D, proj_dim), so layers can be mixed in a learned common space. With
proj_dim = hidden_dim and an mlp head, F3 has about the same parameter count as the concatenation of the same K
layers (K * D * proj_dim), which makes "F3 vs concatenation" a parameter-matched comparison. Without projection
(proj_dim=None) it is a pure scalar mix of the z-scored layers:

The K layer weights are *static*: one scalar per layer, shared by every timestep, episode and task (unlike
``gated_fusion``, whose weights depend on the input). Without projection the fused vector keeps the per-layer width
D, so the head has exactly as many parameters as a single-layer probe and fusion only adds K (+1 for gamma).

``learn_weights=False`` freezes the mix at ``init_weights`` (uniform when omitted). The frozen weights live in a
buffer, so they are saved with the checkpoint but are not trainable parameters. Use it for fixed-weight references
(e.g. the uniform average of the chosen layers) with the identical head and training loop.

Features should be z-scored per layer first (``SharedCache.records(..., normalize=True)``); raw layers differ in scale
by orders of magnitude and the softmax weights would then mostly undo scale instead of choosing layers.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from ._fusion import FusionDetector
from .base import Output
from .probe_heads import build_head, check_layer_input
from .registry import register_model


def _normalized_init(init_weights: Sequence[float] | None, num_layers: int, learnable: bool) -> torch.Tensor:
    if init_weights is None:
        return torch.full((num_layers,), 1.0 / num_layers)
    w = torch.as_tensor(list(init_weights), dtype=torch.float32)
    if w.ndim != 1 or w.numel() != num_layers:
        raise ValueError(f"init_weights needs {num_layers} values (one per layer), got {list(init_weights)}")
    if not torch.isfinite(w).all() or (w < 0).any():
        raise ValueError(f"init_weights must be finite and non-negative, got {w.tolist()}")
    if w.sum() <= 0:
        raise ValueError("init_weights must not all be zero")
    if learnable and (w == 0).any():
        # log(0) = -inf would freeze that layer forever and turns into NaN under weight decay
        raise ValueError("learnable init_weights must be strictly positive; use learn_weights=False to drop layers")
    return w / w.sum()


@register_model("static_weight_fusion")
class StaticWeightFusion(FusionDetector):
    def __init__(self, input_dim=32, num_layers=2, hidden_dim=64, dropout=0.0, head="mlp", learn_weights=True,
                 init_weights=None, learn_scale=True, proj_dim=None, **kwargs):
        super().__init__(input_dim, num_layers, hidden_dim, dropout)
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.learn_weights = bool(learn_weights)
        w = _normalized_init(init_weights, num_layers, self.learn_weights)
        if self.learn_weights:
            self.layer_logits = nn.Parameter(torch.log(w))  # softmax(log w) == w, so training starts at init_weights
        else:
            self.register_buffer("fixed_weights", w)
        if self.learn_weights and learn_scale:
            self.gamma = nn.Parameter(torch.ones(()))
        else:
            self.register_buffer("gamma", torch.ones(()))
        if proj_dim is not None and (isinstance(proj_dim, bool) or int(proj_dim) != proj_dim or proj_dim < 1):
            raise ValueError(f"proj_dim must be a positive integer or None, got {proj_dim!r}")
        self.proj_dim = None if proj_dim is None else int(proj_dim)
        self.proj = (None if self.proj_dim is None
                     else nn.ModuleList(nn.Linear(input_dim, self.proj_dim) for _ in range(num_layers)))
        self.head_kind = head
        self.head = build_head(head, self.proj_dim or input_dim, hidden_dim, dropout)

    def layer_weights(self) -> torch.Tensor:
        """Current mixing weights, shape [K], non-negative and summing to 1."""
        if self.learn_weights:
            return torch.softmax(self.layer_logits, dim=0)
        return self.fixed_weights

    def forward(self, batch):
        x = self._layers(batch)
        check_layer_input(x, self.num_layers, self.input_dim)
        w = self.layer_weights().to(x.dtype)
        if self.proj is not None:
            x = torch.stack([p(x[:, :, k]) for k, p in enumerate(self.proj)], dim=2)  # [B, T, K, proj_dim]
        fused = self.gamma.to(x.dtype) * torch.einsum("btkd,k->btd", x, w)
        logits = self.head(self.dropout(fused))
        return Output(logits, auxiliary={"layer_weights": w.detach()})

    def extra_repr(self) -> str:
        return (f"num_layers={self.num_layers}, input_dim={self.input_dim}, head={self.head_kind!r}, "
                f"learn_weights={self.learn_weights}, proj_dim={self.proj_dim}")


def entropy_of(weights: torch.Tensor) -> float:
    """Entropy (nats) of a weight vector: log(K) = uniform, 0 = a single layer. For reporting how peaked a mix is."""
    w = weights.detach().clamp_min(0)
    w = w / w.sum()
    nz = w[w > 0]
    return float(-(nz * nz.log()).sum()) if nz.numel() else math.nan
