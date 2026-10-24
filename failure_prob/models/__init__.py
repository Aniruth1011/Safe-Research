"""Registered detectors for the normalized frozen-feature contract."""

from .base import FailureDetector, Output
from .registry import get_model, list_models, register_model
from . import safe_mlp, safe_lstm, single_layer_mlp, concat_fusion, gated_fusion, residual_fusion

__all__ = ["FailureDetector", "Output", "get_model", "list_models", "register_model"]
