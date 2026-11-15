"""Concatenation fusion: F1 (row 19, two layers) and F2 (row 20, three layers). The K chosen layers are
concatenated per timestep ([B,T,K*D]) and probed.

``head="mlp"`` (default, unchanged from the original model) or ``head="linear"`` (logistic probe) - see probe_heads.py.
The head's input width grows with K, so compare it against ``static_weight_fusion`` (F3) and a parameter-matched
single-layer probe (K1/K2) before attributing a gain to fusion.
"""
from ._fusion import FusionDetector
from .base import Output
from .probe_heads import build_head, check_layer_input
from .registry import register_model


@register_model("concat_fusion")
class ConcatFusion(FusionDetector):
    def __init__(self, input_dim=32, num_layers=2, hidden_dim=64, dropout=0.0, head="mlp", **kwargs):
        super().__init__(input_dim, num_layers, hidden_dim, dropout)
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.head_kind = head
        self.head = build_head(head, input_dim * num_layers, hidden_dim, dropout)

    def forward(self, batch):
        x = self._layers(batch)
        check_layer_input(x, self.num_layers, self.input_dim)
        return Output(self.head(x.flatten(2)))
