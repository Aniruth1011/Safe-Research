"""Pluggable probe and evaluator for the depth sweep (plan rows R1-R4).

The sweep (failure_prob/engine/depth_sweep.py) only talks to the two small interfaces below. To plug in the real
pieces, write a module with a factory and pass it by name, e.g.

    python scripts/depth_sweep.py run ... --probe failure_prob.probes.b6:make_probe \
                                          --evaluator failure_prob.eval.b4:make_evaluator

or register a short name in PROBES / EVALUATORS at the bottom of this file. Nothing else changes.

Contracts
---------
Probe.fit(train_records, seed) -> FittedProbe
    train_records: list of load_cache.SharedCache records (features [1,T,1,D] float32, mask, labels, task_ids,
    trajectory_ids), split "train" only. The probe owns preprocessing beyond the --normalize flag, its
    hyperparameters and its seeding (seed is one of protocol.TRAIN_SEEDS). It must not look at any other split.
FittedProbe.step_scores(records) -> list[np.ndarray]
    One 1-D array per record: a failure score for every timestep of the FULL rollout (higher = more likely to fail).
Evaluator(EvalInput) -> dict[str, float]
    Scores seen_val rollouts. Receives the full-length per-step scores plus each episode's horizon
    (min(length, task_min_step)); the evaluator decides how a rollout is scored at that horizon.

Every probe and evaluator says whether it is TEMPORARY. Rows produced by a temporary one are marked as such in
every output file and in the plot title, and must not be reported as results.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np


class FittedProbe(Protocol):
    def step_scores(self, records: Sequence[dict]) -> list[np.ndarray]: ...


class Probe(Protocol):
    name: str
    temporary: bool

    def fit(self, train_records: Sequence[dict], seed: int) -> FittedProbe: ...


@dataclass(frozen=True)
class EvalInput:
    step_scores: list[np.ndarray]      # per episode, full rollout length
    labels: np.ndarray                 # 1 = failure, 0 = success
    horizons: np.ndarray               # per episode: min(length, task_min_step)
    task_ids: np.ndarray
    trajectory_ids: list[str]


class Evaluator(Protocol):
    name: str
    temporary: bool

    def __call__(self, data: EvalInput) -> dict[str, float]: ...


# ---------------------------------------------------------------------------------------------------------------
# TEMPORARY placeholders. Replace with Aniruth's probe spec (plan row B6) and Rhea's evaluator (plan row B4).
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class _TorchFittedProbe:
    model: object

    def step_scores(self, records):
        import torch
        out = []
        self.model.eval()
        with torch.no_grad():
            for r in records:
                logits = self.model({"features": r["features"], "mask": r["mask"]}).failure_logits  # [1, T]
                out.append(torch.sigmoid(logits[0]).numpy().astype(np.float64))
        return out


@dataclass
class TemporaryMLPProbe:
    """TEMPORARY placeholder until the shared probe spec (B6) lands.

    SAFE-style independent per-timestep MLP (registered model `single_layer_mlp`, D -> 64 -> 1), trained with the
    repo's trainer (failure_prob.engine.trainer.train_model: Adam, BCE on the rollout label at every valid step).
    Fixed budget, no early stopping and no model selection on seen_val (the sweep scores seen_val once).
    """
    hidden_dim: int = 64
    epochs: int = 5
    batch_size: int = 8
    lr: float = 1e-3
    name: str = "TEMPORARY_mlp64"
    temporary: bool = True

    def fit(self, train_records, seed):
        import torch
        from failure_prob.engine.trainer import train_model
        from failure_prob.models import get_model
        d = int(train_records[0]["features"].shape[-1])
        torch.manual_seed(seed)  # model init; train_model reseeds the batch order with the same seed
        model = get_model("single_layer_mlp", input_dim=d, num_layers=1, hidden_dim=self.hidden_dim, layer_index=0)
        train_model(model, list(train_records), epochs=self.epochs, batch_size=self.batch_size, lr=self.lr, seed=seed)
        return _TorchFittedProbe(model)


@dataclass
class TemporaryMaxScoreEvaluator:
    """TEMPORARY placeholder until the shared evaluator (B4) lands.

    Rollout score = max per-step score over steps [0, horizon). Then ROC-AUC, PR-AUC (average precision) and the
    TPR at 5% FPR across seen_val rollouts (failure = positive class). Not SAFE's exact protocol.
    """
    name: str = "TEMPORARY_max_at_task_min_step"
    temporary: bool = True

    def __call__(self, data: EvalInput) -> dict[str, float]:
        from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
        scores = np.array([float(np.max(s[:h])) for s, h in zip(data.step_scores, data.horizons)])
        y = data.labels.astype(int)
        if len(np.unique(y)) < 2:
            return {"roc_auc": float("nan"), "pr_auc": float("nan"), "tpr_at_5_fpr": float("nan")}
        fpr, tpr, _ = roc_curve(y, scores)
        return {"roc_auc": float(roc_auc_score(y, scores)), "pr_auc": float(average_precision_score(y, scores)),
                "tpr_at_5_fpr": float(np.max(tpr[fpr <= 0.05], initial=0.0))}


PROBES: dict[str, Callable[[], Probe]] = {"temporary_mlp": TemporaryMLPProbe}
EVALUATORS: dict[str, Callable[[], Evaluator]] = {"temporary_max": TemporaryMaxScoreEvaluator}


def resolve(spec: str, registry: dict[str, Callable]):
    """A registered short name, or 'package.module:factory' (called with no arguments)."""
    if spec in registry:
        obj = registry[spec]()
    elif ":" in spec:
        module, attr = spec.split(":", 1)
        obj = getattr(importlib.import_module(module), attr)()
    else:
        raise KeyError(f"unknown plugin {spec!r}: use one of {sorted(registry)} or 'package.module:factory'")
    for field in ("name", "temporary"):
        if not hasattr(obj, field):
            raise TypeError(f"plugin {spec!r} has no `{field}` attribute")
    return obj
