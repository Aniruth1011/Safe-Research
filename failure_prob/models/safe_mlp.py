from .single_layer_mlp import SingleLayerMLP
from .registry import register_model


@register_model("safe_mlp")
class SafeMLP(SingleLayerMLP):
    """Compatibility name for SAFE's final-layer independent MLP baseline."""

