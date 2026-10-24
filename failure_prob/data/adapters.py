"""Adapters between original SAFE rollouts and the normalized research batch."""
from __future__ import annotations

import torch


def rollout_batch_to_features(batch: dict, layer_index: int = 0) -> dict:
    features = batch["features"]
    if features.ndim == 3:
        features = features.unsqueeze(2)
    return {
        "features": features,
        "mask": batch.get("mask", batch.get("valid_masks")).bool(),
        "labels": 1.0 - batch.get("labels", batch["success_labels"]).float(),
        "task_ids": batch.get("task_ids"),
        "trajectory_ids": batch.get("trajectory_ids"),
    }


def select_layers(features: torch.Tensor, layer_positions: list[int]) -> torch.Tensor:
    if features.ndim != 4:
        raise ValueError("Expected features with shape [B,T,K,D]")
    if max(layer_positions, default=-1) >= features.shape[2]:
        raise ValueError(f"Missing requested layer position in feature tensor: {layer_positions}")
    return features[:, :, layer_positions]
