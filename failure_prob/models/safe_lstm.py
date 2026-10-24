from torch import nn

from ._fusion import FusionDetector
from .base import Output
from .registry import register_model


@register_model("safe_lstm")
class SafeLSTM(FusionDetector):
    """Normalized-contract compatibility baseline; original LSTM remains untouched."""
    def __init__(self, input_dim=32, num_layers=1, hidden_dim=64, dropout=0.0, **kwargs):
        super().__init__(input_dim, num_layers, hidden_dim, dropout)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, batch):
        x = self._layers(batch)[:, :, 0]
        out, _ = self.lstm(x)
        return Output(self.head(self.dropout(out)).squeeze(-1))
