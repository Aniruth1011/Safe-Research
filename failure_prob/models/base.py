from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class Output:
    failure_logits: torch.Tensor
    prediction_level: str = "timestep"
    mask_semantics: str = "mask_true_is_valid"
    auxiliary: dict[str, Any] | None = None


class FailureDetector(nn.Module):
    """Common detector contract for cached VLA features."""

    model_type = "base"

    def forward(self, batch: dict[str, torch.Tensor]) -> Output:
        raise NotImplementedError

    def loss(self, batch: dict[str, torch.Tensor], output: Output | None = None) -> torch.Tensor:
        output = output or self(batch)
        labels = batch["labels"].float()
        logits = output.failure_logits
        mask = batch.get("mask")
        if logits.ndim == 3:
            logits = logits.squeeze(-1)
        target = labels[:, None].expand_as(logits)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if mask is not None:
            losses = losses * mask.to(losses.dtype)
            return losses.sum() / mask.sum().clamp_min(1)
        return losses.mean()

    def predict_logits(self, batch):
        return self(batch).failure_logits
