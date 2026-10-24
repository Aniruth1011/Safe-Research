from torch import nn
import torch

from ._fusion import FusionDetector
from .base import Output
from .registry import register_model


@register_model("residual_fusion")
class ResidualFusion(FusionDetector):
    def __init__(self, input_dim=32, num_layers=2, hidden_dim=64, dropout=0.0, **kwargs):
        super().__init__(input_dim, num_layers, hidden_dim, dropout)
        self.projections = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_layers)])
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, batch):
        x = self._layers(batch)
        if x.shape[2] != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} layers, got {x.shape[2]}")
        fused = self.projections[0](x[:, :, 0])
        for projection, layer in zip(self.projections[1:], x[:, :, 1:].unbind(2)):
            fused = fused + projection(layer)
        return Output(self.head(self.dropout(fused)).squeeze(-1))
