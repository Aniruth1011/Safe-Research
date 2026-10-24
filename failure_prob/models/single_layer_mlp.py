from ._fusion import FusionDetector, MLPHead
from .base import Output
from .registry import register_model


@register_model("single_layer_mlp")
class SingleLayerMLP(FusionDetector):
    def __init__(self, input_dim=32, num_layers=1, hidden_dim=64, layer_index=0, dropout=0.0, **kwargs):
        super().__init__(input_dim, num_layers, hidden_dim, dropout)
        self.layer_index = layer_index
        self.head = MLPHead(input_dim, hidden_dim, dropout)

    def forward(self, batch):
        x = self._layers(batch)
        if self.layer_index >= x.shape[2]:
            raise ValueError(f"Requested layer position {self.layer_index}, only {x.shape[2]} supplied")
        return Output(self.head(x[:, :, self.layer_index]), auxiliary={"layer_index": self.layer_index})
