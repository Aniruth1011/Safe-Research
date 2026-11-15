"""Pluggable probe and evaluator for the depth sweep (plan rows R1-R4).

The sweep (failure_prob/engine/depth_sweep.py) only talks to the two small interfaces below. To plug in the real
pieces, write a module with a factory and pass it by name, e.g.

    python scripts/depth_sweep.py run ... --probe some.module:make_probe \
                                          --evaluator failure_prob.engine.episode_metrics:make_evaluator

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
from dataclasses import dataclass, fields
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
# TEMPORARY placeholders. The probe waits for Aniruth's settings (plan row B6). The evaluator placeholder is kept
# selectable as `temporary_max` (the TEMPORARY R1 run used it); the default is B4's evaluator, `b4`, below.
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


# ---------------------------------------------------------------------------------------------------------------
# The shared probe (plan row B6): the framework model, trained like the fusion models. Settings are PLACEHOLDERS.
# ---------------------------------------------------------------------------------------------------------------
def _framework_step_scores(model, records, batch_size: int) -> list[np.ndarray]:
    """Per-step failure logits of every record, trimmed to its length: the same loop as
    failure_prob.engine.scoring.step_scores (tests check they agree). Kept here because importing scoring pulls in
    failure_prob.data and with it Hydra/omegaconf, which the sweep's CARC environment does not need."""
    import torch
    from torch.utils.data import DataLoader
    from failure_prob.engine.trainer import RecordDataset, collate_records
    model.eval()
    out = []
    with torch.no_grad():
        for batch in DataLoader(RecordDataset(list(records)), batch_size=batch_size, shuffle=False,
                                collate_fn=collate_records):
            logits = model(batch).failure_logits.float().cpu().numpy()
            out.extend(logits[i, :n].astype(np.float64) for i, n in enumerate(batch["mask"].sum(1).tolist()))
    return out


@dataclass
class _FrameworkFitted:
    model: object
    batch_size: int
    fit_info: dict

    def step_scores(self, records):
        return _framework_step_scores(self.model, records, self.batch_size)


def _holdout_split(records, fraction: float, seed: int):
    """(fit, holdout) of TRAIN records: `fraction` of each (task, outcome) group, picked with `seed`."""
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(records):
        groups.setdefault((int(r["task_ids"][0]), int(r["labels"].reshape(-1)[0])), []).append(i)
    rng = np.random.default_rng(seed)
    hold = set()
    for key in sorted(groups):
        idx = groups[key]
        k = int(round(fraction * len(idx)))
        if len(idx) >= 2:
            k = min(max(k, 1), len(idx) - 1)    # keep at least one episode of the group on each side
        else:
            k = 0
        hold.update(rng.permutation(idx)[:k].tolist())
    fit = [r for i, r in enumerate(records) if i not in hold]
    held = [r for i, r in enumerate(records) if i in hold]
    labels = {int(r["labels"].reshape(-1)[0]) for r in held}
    if len(labels) < 2:
        raise ValueError("early-stopping holdout needs both outcomes; raise es_holdout_fraction")
    return fit, held


# Plan row B6, locked by Aniruth on 2026-11-15: the shared probe's settings (framework_mlp's defaults). Early
# stopping watches the loss on 15% of TRAIN (es_monitor "loss", es_min_delta 0); seen_val is never used for it.
B6_SETTINGS = {"hidden_dim": 64, "dropout": 0.1, "lr": 3e-4, "epochs": 150, "batch_size": 8,
               "early_stopping_patience": 10, "es_holdout_fraction": 0.15, "es_monitor": "loss", "es_min_delta": 0.0}


@dataclass
class FrameworkMLPProbe:
    """The shared depth-sweep probe (plan row B6): Aniruth's framework model `single_layer_mlp`
    (failure_prob/models/single_layer_mlp.py), trained with failure_prob/engine/trainer.py the same way
    scripts/train.py trains the fusion models: torch.manual_seed(seed), get_model(...), train_model(...) (Adam,
    BCE on the rollout label at every valid step), then per-step logits scored like failure_prob/engine/scoring.py.

    Every setting is an argument (`--probe-arg key=value`, or SWEEP_PROBE_ARGS in the Slurm jobs). The defaults are
    the locked B6 settings (B6_SETTINGS, from Aniruth, 2026-11-15). Only those exact settings are the real probe
    (name "framework_mlp", not temporary, folder tag "b6"); any other setting is exploratory, so it is named
    "TEMPORARY_framework_mlp", every output says TEMPORARY, and its folder carries a digest of the settings.

    Early stopping is off when early_stopping_patience == 0. When on, it holds out es_holdout_fraction of the TRAIN
    episodes (stratified by task and outcome, chosen with the seed), trains on the rest, checks es_monitor on the
    holdout after every epoch ("loss": the training loss on the holdout, lower is better; "roc_auc": B4's episode
    ROC-AUC at min(length, task_min_step), higher is better), stops after `patience` epochs without an improvement
    larger than es_min_delta, and restores the best epoch's weights. seen_val is never used for any of this.
    """
    hidden_dim: int = B6_SETTINGS["hidden_dim"]
    dropout: float = B6_SETTINGS["dropout"]
    lr: float = B6_SETTINGS["lr"]
    epochs: int = B6_SETTINGS["epochs"]
    batch_size: int = B6_SETTINGS["batch_size"]
    early_stopping_patience: int = B6_SETTINGS["early_stopping_patience"]
    es_holdout_fraction: float = B6_SETTINGS["es_holdout_fraction"]
    es_monitor: str = B6_SETTINGS["es_monitor"]
    es_min_delta: float = B6_SETTINGS["es_min_delta"]

    @property
    def locked(self) -> bool:
        """True only for exactly the locked B6 settings."""
        return self.settings() == B6_SETTINGS

    @property
    def name(self) -> str:
        return "framework_mlp" if self.locked else "TEMPORARY_framework_mlp"

    @property
    def temporary(self) -> bool:
        return not self.locked

    @property
    def folder_tag(self) -> str | None:
        """Short folder tag for the locked settings; None means the sweep tags the folder with a settings digest."""
        return "b6" if self.locked else None

    def __post_init__(self):
        checks = {"hidden_dim": self.hidden_dim >= 1, "dropout": 0.0 <= self.dropout < 1.0, "lr": self.lr > 0,
                  "epochs": self.epochs >= 1, "batch_size": self.batch_size >= 1,
                  "early_stopping_patience": self.early_stopping_patience >= 0,
                  "es_holdout_fraction": 0.0 < self.es_holdout_fraction < 1.0,
                  "es_monitor": self.es_monitor in ("loss", "roc_auc"), "es_min_delta": self.es_min_delta >= 0}
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            raise ValueError(f"framework_mlp: invalid {', '.join(f'{k}={getattr(self, k)!r}' for k in bad)}")

    def settings(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def _monitor(self, model, holdout) -> float:
        import torch
        from failure_prob.engine.trainer import RecordDataset, collate_records
        if self.es_monitor == "roc_auc":
            from failure_prob.engine.episode_metrics import episode_metrics
            if any("task_min_steps" not in r for r in holdout):
                raise ValueError("es_monitor=roc_auc needs records with task_min_steps")
            scores = _framework_step_scores(model, holdout, self.batch_size)
            labels = np.array([int(r["labels"].reshape(-1)[0]) for r in holdout])
            horizons = [min(int(r["mask"].sum()), int(r["task_min_steps"][0])) for r in holdout]
            return episode_metrics(scores, labels, horizons)["roc_auc"]
        model.eval()
        total, steps = 0.0, 0
        with torch.no_grad():
            for batch in torch.utils.data.DataLoader(RecordDataset(holdout), batch_size=self.batch_size,
                                                     collate_fn=collate_records):
                n = int(batch["mask"].sum())
                total += float(model.loss(batch)) * n
                steps += n
        return total / max(steps, 1)

    def fit(self, train_records, seed):
        import copy
        import torch
        from failure_prob.engine.trainer import train_model
        from failure_prob.models import get_model
        records = list(train_records)
        fit_recs, holdout = records, []
        if self.early_stopping_patience > 0:
            fit_recs, holdout = _holdout_split(records, self.es_holdout_fraction, seed)
        d = int(records[0]["features"].shape[-1])
        torch.manual_seed(seed)                                  # as scripts/train.py: model init from the seed
        model = get_model("single_layer_mlp", input_dim=d, num_layers=1, hidden_dim=self.hidden_dim,
                          dropout=self.dropout, layer_index=0)
        sign = -1.0 if self.es_monitor == "loss" else 1.0        # compare as "higher is better"
        track = {"best": None, "best_epoch": 0, "state": None, "monitor": []}

        def on_epoch_end(epoch, m, history):
            rng = torch.random.get_rng_state()                   # evaluating must not change the next shuffle
            value = self._monitor(m, holdout)
            torch.random.set_rng_state(rng)
            track["monitor"].append(value)
            if track["best"] is None or sign * value > sign * track["best"] + self.es_min_delta:
                track.update(best=value, best_epoch=epoch, state=copy.deepcopy(m.state_dict()))
            return epoch - track["best_epoch"] >= self.early_stopping_patience

        history = train_model(model, fit_recs, epochs=self.epochs, batch_size=self.batch_size, lr=self.lr, seed=seed,
                              on_epoch_end=on_epoch_end if holdout else None)
        if holdout:
            model.load_state_dict(track["state"])
        info = {"epochs_run": len(history), "best_epoch": track["best_epoch"] if holdout else len(history),
                "train_loss": [round(float(x), 6) for x in history], "n_fit_rollouts": len(fit_recs),
                "n_holdout_rollouts": len(holdout),
                **({"holdout_" + self.es_monitor: [round(float(x), 6) for x in track["monitor"]]} if holdout else {})}
        return _FrameworkFitted(model, self.batch_size, info)


def make_framework_mlp(**kwargs) -> FrameworkMLPProbe:
    """Factory for `--probe framework_mlp [--probe-arg key=value ...]`. Values may be strings (from the command line);
    they are converted to each setting's type. Unknown keys are refused."""
    types = {f.name: f.type for f in fields(FrameworkMLPProbe)}
    unknown = sorted(set(kwargs) - set(types))
    if unknown:
        raise ValueError(f"framework_mlp: unknown setting(s) {unknown}; settings are {sorted(types)}")
    conv = {"int": int, "float": float, "str": str}
    out = {}
    for k, v in kwargs.items():
        t = conv[types[k] if isinstance(types[k], str) else types[k].__name__]
        if isinstance(v, str) and t is int and not v.lstrip("-").isdigit():
            raise ValueError(f"framework_mlp: {k} must be an integer, got {v!r}")
        out[k] = t(v)
    return FrameworkMLPProbe(**out)


def _b4_evaluator() -> Evaluator:
    """The shared evaluator (plan row B4, failure_prob/engine/episode_metrics.py): SAFE's episode score (max step
    score before min(length, task_min_step)), ROC-AUC, SAFE-style PR-AUC, TPR at 5% FPR, task-mean ROC-AUC."""
    from failure_prob.engine.episode_metrics import make_evaluator
    return make_evaluator()


DEFAULT_EVALUATOR = "b4"
DEFAULT_PROBE = "framework_mlp"   # with its defaults = the locked B6 probe; "temporary_mlp" is the old placeholder
PROBES: dict[str, Callable[..., Probe]] = {"temporary_mlp": TemporaryMLPProbe, "framework_mlp": make_framework_mlp}
EVALUATORS: dict[str, Callable[[], Evaluator]] = {"b4": _b4_evaluator, "temporary_max": TemporaryMaxScoreEvaluator}


def resolve(spec: str, registry: dict[str, Callable], **kwargs):
    """A registered short name, or 'package.module:factory', called with the keyword settings given (if any)."""
    if spec in registry:
        factory = registry[spec]
    elif ":" in spec:
        module, attr = spec.split(":", 1)
        factory = getattr(importlib.import_module(module), attr)
    else:
        raise KeyError(f"unknown plugin {spec!r}: use one of {sorted(registry)} or 'package.module:factory'")
    try:
        obj = factory(**kwargs)
    except TypeError as e:
        if kwargs:
            raise TypeError(f"plugin {spec!r} does not take the settings {sorted(kwargs)}: {e}") from e
        raise
    for field in ("name", "temporary"):
        if not hasattr(obj, field):
            raise TypeError(f"plugin {spec!r} has no `{field}` attribute")
    return obj
