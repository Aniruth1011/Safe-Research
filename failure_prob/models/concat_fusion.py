from ._fusion import FusionDetector, MLPHead
from .base import Output
from .registry import register_model


@register_model("concat_fusion")
class ConcatFusion(FusionDetector):
    def __init__(self, input_dim=32, num_layers=2, hidden_dim=64, dropout=0.0, **kwargs):
        super().__init__(input_dim, num_layers, hidden_dim, dropout)
        self.head = MLPHead(input_dim * num_layers, hidden_dim, dropout)

    def forward(self, batch):
        x = self._layers(batch)
        if x.shape[2] != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} layers, got {x.shape[2]}")
        return Output(self.head(x.flatten(2)))
