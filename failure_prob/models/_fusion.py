import torch
from torch import nn

from .base import FailureDetector, Output


class LayerInputMixin:
    def _layers(self, batch):
        x = batch["features"]
        if x.ndim == 3:
            x = x.unsqueeze(2)
        if x.ndim != 4:
            raise ValueError(f"features must have shape [B,T,K,D], got {tuple(x.shape)}")
        return x


class FusionDetector(LayerInputMixin, FailureDetector):
    def __init__(self, input_dim: int, num_layers: int, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.input_dim, self.num_layers, self.hidden_dim = input_dim, num_layers, hidden_dim
        self.dropout = nn.Dropout(dropout)


class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)
