"""Probe heads shared by the fusion detectors (F1, F2, F3, F6).

The shared probe spec (B6) decides which head the team reports; until then both are available and every fusion
model takes the same ``head`` argument so swapping the probe never touches the fusion code.

* ``"mlp"``    - Linear(in, hidden) -> ReLU -> Dropout -> Linear(hidden, 1)   (the repo's existing MLPHead)
* ``"linear"`` - Linear(in, 1), i.e. a logistic-regression probe on each timestep
"""
from __future__ import annotations

from torch import nn

from ._fusion import MLPHead

HEADS = ("mlp", "linear")


class LinearHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_head(kind: str, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> nn.Module:
    if kind == "mlp":
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1 for an mlp head, got {hidden_dim}")
        return MLPHead(input_dim, hidden_dim, dropout)
    if kind == "linear":
        return LinearHead(input_dim)
    raise ValueError(f"Unknown probe head {kind!r}; choose one of {HEADS}")


def check_layer_input(x, num_layers: int, input_dim: int) -> None:
    """Fail with a readable message instead of a shape error deep inside nn.Linear."""
    if x.shape[2] != num_layers:
        raise ValueError(f"Expected {num_layers} layers, got {x.shape[2]}")
    if x.shape[3] != input_dim:
        raise ValueError(f"Expected feature dim {input_dim}, got {x.shape[3]}")
