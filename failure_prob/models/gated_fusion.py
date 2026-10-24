import torch
from torch import nn

from ._fusion import FusionDetector
from .base import Output
from .registry import register_model


@register_model("gated_fusion")
class GatedFusion(FusionDetector):
    def __init__(self, input_dim=32, num_layers=2, hidden_dim=64, dropout=0.0, **kwargs):
        super().__init__(input_dim, num_layers, hidden_dim, dropout)
        self.projections = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_layers)])
        self.gate = nn.Sequential(nn.Linear(input_dim * num_layers, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_layers))
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, batch):
        x = self._layers(batch)
        if x.shape[2] != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} layers, got {x.shape[2]}")
        projected = torch.stack([p(x[:, :, i]) for i, p in enumerate(self.projections)], dim=2)
        weights = torch.softmax(self.gate(x.flatten(2)), dim=-1)
        fused = (projected * weights.unsqueeze(-1)).sum(dim=2)
        return Output(self.head(self.dropout(fused)).squeeze(-1), auxiliary={"gate_weights": weights})
