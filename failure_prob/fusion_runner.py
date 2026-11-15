"""Fusion runner (shared by F1, F2, F3, F6): train a fusion detector on `train` and score it on `seen_val`, per episode, for every protocol seed.

Development only: this module has no way to load `unseen_test` (it never passes ``allow_unseen``), and it refuses
records whose split metadata or trajectory IDs say otherwise.

Two pieces are provisional until the shared pieces land, and every output file says so:
  * episode scorer (B4, Rhea): an episode's score aggregates its per-timestep failure logits over the valid
    (unpadded, horizon-truncated) steps; ``max`` (default) flags an episode as soon as any step looks like failure,
    as SAFE's detector does. The shared evaluator plugs in by name (``evaluator="pkg.module:factory"``), with the
    same contract as the depth sweep's evaluators (``Evaluator(EvalInput) -> dict``, see ``EvalInput`` below).
  * probe (B6, Aniruth): head type, width, epochs, lr and weight decay are all arguments; nothing is tuned here.

Protocol (the same as the R1-R4 depth sweep, so fusion rows compare like-for-like with single layers): train on
``train`` at ``train_horizon`` (default ``full``, as the sweep), score every ``seen_val`` episode at ``horizon``
(``task_min_step``, the only reportable horizon).

CLI: scripts/run_fusion.py. Python:
    from failure_prob.fusion_runner import RunConfig, run
    run(RunConfig(model="static_weight_fusion", layers=[16, 33], cache=os.environ["CACHE"],
                  output_dir="outputs/fusion/swf_16_33"))
"""
from __future__ import annotations

import copy
import inspect
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from failure_prob.data.synthetic import make_synthetic_records
from failure_prob.engine.trainer import RecordDataset, collate_records
from failure_prob.models import get_model, list_models
from failure_prob.protocol import TRAIN_SEEDS

REPO = Path(__file__).resolve().parents[1]
HORIZONS = ("task_min_step", "full")
AGGREGATIONS = ("max", "mean", "last")
DEV_SPLITS = ("train", "seen_val")
PROVISIONAL = {
    "episode_scorer": "provisional fusion_runner scorer, pending the shared evaluator (B4)",
    "probe": "provisional head/hyper-parameters, pending the shared probe spec (B6)",
}


@dataclass(frozen=True)
class EvalInput:
    """What a plugged-in evaluator receives; field-for-field the depth sweep's ``sweep_plugins.EvalInput``.

    step_scores: per seen_val episode, the per-step failure scores (here: logits, higher = more likely to fail) over
    the steps this runner kept; horizons: the number of steps the evaluator should use (= len(step_scores[i]), the
    episode was already cut to ``horizon`` when loaded)."""
    step_scores: list
    labels: np.ndarray
    horizons: np.ndarray
    task_ids: np.ndarray
    trajectory_ids: list


def resolve_evaluator(spec: str | None):
    """None -> the built-in provisional scorer. Otherwise a name registered in the depth sweep's
    ``sweep_plugins.EVALUATORS`` (when that module exists) or 'package.module:factory' (called with no arguments).
    The object must be callable and carry ``name`` and ``temporary``."""
    if spec is None:
        return None
    import importlib
    registry = {}
    try:
        registry = dict(importlib.import_module("failure_prob.engine.sweep_plugins").EVALUATORS)
    except (ImportError, AttributeError):
        pass
    if spec in registry:
        obj = registry[spec]()
    elif ":" in spec:
        module, attr = spec.split(":", 1)
        try:
            obj = getattr(importlib.import_module(module), attr)()
        except (ImportError, AttributeError) as e:
            raise ValueError(f"cannot load evaluator {spec!r}: {e}") from None
    else:
        raise ValueError(f"unknown evaluator {spec!r}: use one of {sorted(registry)} or 'package.module:factory'")
    for attr in ("name", "temporary"):
        if not hasattr(obj, attr):
            raise ValueError(f"evaluator {spec!r} has no `{attr}` attribute")
    if not callable(obj):
        raise ValueError(f"evaluator {spec!r} is not callable")
    return obj


def provisional_flags(evaluator) -> dict:
    flags = dict(PROVISIONAL)
    if evaluator is not None:
        if getattr(evaluator, "temporary", True):
            flags["episode_scorer"] = f"TEMPORARY evaluator {evaluator.name!r}, pending the shared evaluator (B4)"
        else:
            del flags["episode_scorer"]
    return flags


@dataclass
class RunConfig:
    model: str
    layers: list[int]
    output_dir: str
    cache: str | None = None            # SharedCache root (…/safe_cache/v1); exactly one of cache / synthetic
    synthetic: bool = False             # CPU fixture with random features, for smoke tests only
    pool: str = "mean"
    normalize: bool = True              # train-only z-scoring (team default for the main comparison)
    horizon: str = "task_min_step"      # seen_val scoring horizon; "full" leaks the label through length
    train_horizon: str = "full"         # train episodes; "full" as in the depth sweep (no seen_val data involved)
    seeds: list[int] = field(default_factory=lambda: list(TRAIN_SEEDS))
    head: str = "mlp"
    hidden_dim: int = 64
    dropout: float = 0.0
    epochs: int = 5                     # the depth sweep's TEMPORARY probe budget (until B6)
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 0.0              # 0 disables clipping
    select_best_epoch: bool = False     # pick the epoch with the best seen_val ROC-AUC (selection on seen_val only)
    aggregation: str = "max"
    evaluator: str | None = None        # None = built-in scorer; else a sweep_plugins name or 'pkg.module:factory'
    learn_weights: bool = True          # static_weight_fusion only
    init_weights: list[float] | None = None  # static_weight_fusion only
    proj_dim: int | None = None         # static_weight_fusion only: per-layer projection to a shared dimension
    std_floor: float | None = None      # normalize with max(train std, std_floor) instead of the loader's 1e-6 floor
    allow_duplicate_layers: bool = False  # K4 control (row 35): the same layer several times, e.g. [33, 33, 33]
    bootstrap: int = 1000               # stratified bootstrap resamples of seen_val episodes for CIs (0 = off)
    ci: float = 0.95                    # CI level
    latency_device: str = "cpu"         # cpu | cuda
    memory_check: bool = True           # refuse to load when the estimated memory exceeds the job's limit
    resume: bool = False                # keep finished seeds of an identical earlier run, train only the missing ones
    overwrite: bool = False
    save_checkpoints: bool = True


# ----------------------------------------------------------------------------------------------- validation

def validate_config(cfg: RunConfig, need_source: bool = True) -> None:
    if cfg.model not in list_models():
        raise ValueError(f"Unknown model {cfg.model!r}; registered: {list_models()}")
    if need_source and bool(cfg.cache) == bool(cfg.synthetic):
        raise ValueError("Give exactly one data source: cache=<SharedCache root> or synthetic=True")
    if not cfg.layers:
        raise ValueError("layers must name at least one layer")
    if any(not isinstance(l, (int, np.integer)) or isinstance(l, bool) for l in cfg.layers):
        raise ValueError(f"layers must be integers, got {cfg.layers}")
    dup = sorted({l for l in cfg.layers if list(cfg.layers).count(l) > 1})
    if dup and not cfg.allow_duplicate_layers:
        raise ValueError(f"duplicate layers {dup}: each layer may appear once (allow_duplicate_layers=True / "
                         f"--allow-duplicate-layers for the duplicate-strongest-layer control)")
    if cfg.horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}, got {cfg.horizon!r}")
    if cfg.train_horizon not in HORIZONS:
        raise ValueError(f"train_horizon must be one of {HORIZONS}, got {cfg.train_horizon!r}")
    if cfg.aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {cfg.aggregation!r}")
    if not cfg.seeds or len(set(cfg.seeds)) != len(cfg.seeds):
        raise ValueError(f"seeds must be non-empty and unique, got {cfg.seeds}")
    for name in ("epochs", "batch_size"):
        if getattr(cfg, name) < 1:
            raise ValueError(f"{name} must be >= 1")
    if cfg.lr <= 0 or cfg.weight_decay < 0 or cfg.grad_clip < 0 or not 0 <= cfg.dropout < 1:
        raise ValueError("need lr > 0, weight_decay >= 0, grad_clip >= 0 and 0 <= dropout < 1")
    if cfg.std_floor is not None and (not cfg.normalize or not cfg.std_floor > 0):
        raise ValueError("std_floor needs normalize=True and a value > 0")
    if cfg.bootstrap < 0 or not 0 < cfg.ci < 1:
        raise ValueError("need bootstrap >= 0 and 0 < ci < 1")
    if cfg.latency_device not in ("cpu", "cuda"):
        raise ValueError(f"latency_device must be cpu or cuda, got {cfg.latency_device!r}")
    if cfg.latency_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("latency_device='cuda' but no GPU is visible")
    if cfg.overwrite and cfg.resume:
        raise ValueError("choose overwrite or resume, not both")
    if cfg.proj_dim is not None and cfg.proj_dim < 1:
        raise ValueError(f"proj_dim must be >= 1, got {cfg.proj_dim}")
    if cfg.init_weights is not None and len(cfg.init_weights) != len(cfg.layers):
        raise ValueError(f"init_weights has {len(cfg.init_weights)} values for {len(cfg.layers)} layers")


def model_kwargs(cfg: RunConfig, input_dim: int) -> dict:
    """Constructor arguments for cfg.model; options the model does not declare are dropped (and reported)."""
    wanted = {"input_dim": input_dim, "num_layers": len(cfg.layers), "hidden_dim": cfg.hidden_dim,
              "dropout": cfg.dropout, "head": cfg.head, "learn_weights": cfg.learn_weights,
              "init_weights": cfg.init_weights, "proj_dim": cfg.proj_dim}
    from failure_prob.models.registry import _MODELS
    params = inspect.signature(_MODELS[cfg.model].__init__).parameters
    kept = {k: v for k, v in wanted.items() if k in params}
    ignored = sorted(k for k, v in wanted.items() if k not in params and v not in (None, True, "mlp"))
    if ignored:
        print(f"note: {cfg.model} does not take {ignored}; ignored", file=sys.stderr)
    return kept


# ----------------------------------------------------------------------------------------------- data

def _shared_cache(root):
    kit = REPO / "collection"
    if str(kit) not in sys.path:
        sys.path.insert(0, str(kit))
    from load_cache import SharedCache  # noqa: E402  (collection/ is a script folder, not a package)
    return SharedCache(root)


def memory_limit_gb() -> float | None:
    """The job's memory limit: Slurm's allocation when inside a job, else the machine's physical memory."""
    env = os.environ
    try:
        if env.get("SLURM_MEM_PER_NODE"):
            return float(env["SLURM_MEM_PER_NODE"]) / 1024
        if env.get("SLURM_MEM_PER_CPU"):
            cpus = int(env.get("SLURM_CPUS_PER_TASK") or env.get("SLURM_CPUS_ON_NODE") or 1)
            return float(env["SLURM_MEM_PER_CPU"]) * cpus / 1024
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    except (ValueError, OSError, AttributeError):
        return None


def _steps(eps, split: str, horizon: str) -> int:
    e = eps[eps.split == split]
    return int(e.length.sum() if horizon == "full" else np.minimum(e.length, e.task_min_step).sum())


def estimate_memory_gb(cache, cfg: RunConfig) -> dict:
    """Peak estimate: float32 records of both splits (K x D per step) + one split's float16 layer files held while
    its records are built + 1 GB for torch/the model. Feature width read from the layer file header."""
    eps = cache.episodes
    path = Path(cfg.cache) / "features" / cfg.pool / "train" / f"L{cfg.layers[0]:02d}.npy"
    d = int(np.load(path, mmap_mode="r").shape[1])
    k = len(cfg.layers)
    recs = sum(_steps(eps, s, cfg.train_horizon if s == "train" else cfg.horizon) for s in DEV_SPLITS) * k * d * 4
    files = max(int(eps[eps.split == s].length.sum()) for s in DEV_SPLITS) * k * d * 2  # whole split files
    return {"records_gb": recs / 1024 ** 3, "files_gb": files / 1024 ** 3,
            "total_gb": (recs + files) / 1024 ** 3 + 1.0, "feature_dim": d}


def _preload_sequential(cache, pool: str, split: str, layers) -> None:
    """Read each layer file of `split` in one sequential pass into the loader's array cache (instead of memory-mapped
    random slices, which stalled on /scratch1 under load). Only train / seen_val are ever read here."""
    if split not in DEV_SPLITS:
        raise PermissionError(f"refusing to read {split}")
    mm = getattr(cache, "_mm", None)
    if not isinstance(mm, dict):
        return
    for layer in layers:
        path = Path(cache.root) / "features" / pool / split / f"L{layer:02d}.npy"
        if not path.exists():
            return  # let the loader raise its own readable error
        t0 = time.time()
        mm[(pool, split, layer)] = np.load(path)
        print(f"read {split}/L{layer:02d} ({path.stat().st_size / 1024 ** 3:.2f} GB) in {time.time() - t0:.0f} s",
              file=sys.stderr, flush=True)


def load_cache_splits(cfg: RunConfig) -> tuple[dict[str, list[dict]], dict]:
    """train + seen_val records from the shared cache, plus provenance. Never requests unseen_test.

    Repeated layers (allow_duplicate_layers) are read once and then copied into their K slots."""
    unique = list(dict.fromkeys(cfg.layers))
    if len(unique) != len(cfg.layers):
        from dataclasses import replace
        out, prov = load_cache_splits(replace(cfg, layers=unique, allow_duplicate_layers=False))
        idx = [unique.index(l) for l in cfg.layers]
        for recs in out.values():
            for r in recs:
                r["features"] = r["features"][:, :, idx].contiguous()
        if prov.get("layer_names"):
            prov["layer_names"] = [prov["layer_names"][i] for i in idx]
        prov["duplicate_layers"] = True
        return out, prov
    cache = _shared_cache(cfg.cache)
    names = getattr(cache, "layer_names", None)
    if names is not None:
        bad = [l for l in cfg.layers if not 0 <= l < len(names)]
        if bad:
            raise ValueError(f"layers {bad} not in this cache (valid 0..{len(names) - 1})")
    pools = cache.manifest.get("pools_exported")
    if pools is not None and cfg.pool not in pools:
        raise ValueError(f"pool {cfg.pool!r} not in this cache (pools_exported: {list(pools)})")
    est = estimate_memory_gb(cache, cfg)
    limit = memory_limit_gb()
    print(f"memory estimate {est['total_gb']:.1f} GB for {len(cfg.layers)} layers (limit "
          f"{'unknown' if limit is None else f'{limit:.1f} GB'})", file=sys.stderr, flush=True)
    if cfg.memory_check and limit is not None and est["total_gb"] > 0.9 * limit:
        raise ValueError(f"estimated {est['total_gb']:.1f} GB exceeds 90% of the {limit:.1f} GB available; request "
                         f"more memory (sbatch --mem) or fewer layers (memory_check=False / --no-memory-check skips)")
    floor_stats = None
    if cfg.std_floor is not None:
        mean, std = cache.norm_stats(cfg.pool, list(cfg.layers))
        floor_stats = (torch.from_numpy(mean), torch.from_numpy(np.maximum(std, cfg.std_floor)))
        print(f"std floor {cfg.std_floor:g}: raises {int((std < cfg.std_floor).sum())} train-std values",
              file=sys.stderr)
    eps = cache.episodes
    out = {}
    for split in DEV_SPLITS:
        horizon = cfg.train_horizon if split == "train" else cfg.horizon
        _preload_sequential(cache, cfg.pool, split, cfg.layers)
        try:
            recs = cache.records(split, layers=list(cfg.layers), pool=cfg.pool,
                                 normalize=cfg.normalize and floor_stats is None, horizon=horizon)
        finally:
            for layer in cfg.layers:  # the records hold their own float32 copies
                getattr(cache, "_mm", {}).pop((cfg.pool, split, layer), None)
        if floor_stats is not None:
            for r in recs:
                r["features"] = (r["features"] - floor_stats[0]) / floor_stats[1]
        rows = eps[eps.split == split].set_index("trajectory_id")
        for r in recs:
            tid = r["trajectory_ids"][0]
            if tid not in rows.index:
                raise ValueError(f"{split}: trajectory {tid} is not listed for this split in episodes.csv")
            ep = rows.loc[tid]
            limit = int(min(ep.length, ep.task_min_step)) if horizon == "task_min_step" else int(ep.length)
            if r["features"].shape[1] != limit:  # guards against a loader that ignores/misreads the horizon
                raise ValueError(f"{split}/{tid}: {r['features'].shape[1]} steps, expected {limit} at horizon "
                                 f"{horizon}")
        out[split] = recs
    if cfg.normalize:
        peak = max(float(r["features"].abs().max()) for recs in out.values() for r in recs)
        if peak > 1e3:
            print(f"WARNING: normalized features reach |x|={peak:.3g}; some dimension has ~0 train std (loader std "
                  f"floor 1e-6). Consider --std-floor 1e-2, or --no-normalize as a comparison.", file=sys.stderr)
    m = cache.manifest
    prov = {"cache_root": str(cfg.cache), "cache_version": m.get("cache_version"), "checkpoint": m.get("checkpoint"),
            "layer_names": [names[l] for l in cfg.layers] if names else None,
            "split_label": "single fixed split (SAFE seed-0 unseen tasks)",
            "horizons": {"train": cfg.train_horizon, "seen_val": cfg.horizon},
            "memory_estimate_gb": round(est["total_gb"], 2), "std_floor": cfg.std_floor}
    return out, prov


def load_synthetic_splits(cfg: RunConfig, d: int = 8) -> tuple[dict[str, list[dict]], dict]:
    out = {}
    for i, split in enumerate(DEV_SPLITS):
        recs = make_synthetic_records(n=16, k=len(cfg.layers), d=d, seed=1000 + i)
        for r in recs:
            r["trajectory_ids"] = [f"{split}/{r['trajectory_ids'][0]}"]
        out[split] = recs
    return out, {"synthetic": True, "note": "random features; numbers are meaningless"}


def check_splits(splits: dict[str, list[dict]], k: int) -> int:
    """Non-empty, disjoint, consistent [1,T,K,D] records. Returns D."""
    if set(splits) != set(DEV_SPLITS):
        raise ValueError(f"expected exactly the splits {DEV_SPLITS}, got {sorted(splits)}")
    ids = {}
    dims = set()
    for split, recs in splits.items():
        if not recs:
            raise ValueError(f"{split} has no episodes")
        for r in recs:
            f = r["features"]
            if f.ndim != 4 or f.shape[0] != 1 or f.shape[2] != k:
                raise ValueError(f"{split}/{r['trajectory_ids']}: features {tuple(f.shape)}, expected [1,T,{k},D]")
            if f.shape[1] < 1 or int(r["mask"].sum()) < 1:
                raise ValueError(f"{split}/{r['trajectory_ids']}: episode has no valid timesteps")
            if not torch.isfinite(f).all():
                raise ValueError(f"{split}/{r['trajectory_ids']}: non-finite features")
            dims.add(int(f.shape[3]))
        ids[split] = [r["trajectory_ids"][0] for r in recs]
        if len(set(ids[split])) != len(ids[split]):
            raise ValueError(f"{split} contains duplicate trajectory ids")
    if len(dims) != 1:
        raise ValueError(f"inconsistent feature dims across records: {sorted(dims)}")
    leak = set(ids["train"]) & set(ids["seen_val"])
    if leak:
        raise ValueError(f"train and seen_val share trajectories: {sorted(leak)[:5]}")
    return dims.pop()


# ----------------------------------------------------------------------------------------------- scoring

@torch.no_grad()
def timestep_logits(model, records, batch_size: int) -> list[torch.Tensor]:
    """Per-episode 1-D tensors of failure logits over the valid timesteps, in record order."""
    model.eval()
    out = []
    loader = DataLoader(RecordDataset(records), batch_size=batch_size, shuffle=False, collate_fn=collate_records)
    for batch in loader:
        logits = model(batch).failure_logits
        if logits.ndim == 3:
            logits = logits.squeeze(-1)
        if logits.shape != batch["mask"].shape:
            raise ValueError(f"model returned logits {tuple(logits.shape)} for mask {tuple(batch['mask'].shape)}")
        for row, mask in zip(logits, batch["mask"]):
            out.append(row[mask].double())
    return out


def aggregate(step_logits: torch.Tensor, how: str) -> float:
    """Episode score in LOGIT space: logit of the episode's failure probability.

    Ranking (ROC/PR-AUC) uses this value. Probabilities saturate (float32 sigmoid(x) == 1.0 for x > ~17), which would
    tie confident episodes and distort the AUC; logits keep them ordered.
      max/last: the max / last step logit (= logit of max / last step probability)
      mean:     logit(mean_t sigmoid(x_t)), computed stably as log(mean p) - log(mean (1-p))
    """
    if step_logits.numel() == 0:
        raise ValueError("cannot score an episode with no valid timesteps")
    x = step_logits.double()
    if not torch.isfinite(x).all():
        raise ValueError("non-finite step logits")
    if how == "max":
        return float(x.max())
    if how == "last":
        return float(x[-1])
    if how == "mean":
        f = torch.nn.functional.logsigmoid
        return float(torch.logsumexp(f(x), 0) - torch.logsumexp(f(-x), 0))
    raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {how!r}")


def to_prob(score: float) -> float:
    return float(torch.sigmoid(torch.tensor(score, dtype=torch.float64)))


def episode_scores(model, records, batch_size: int, how: str) -> list[dict]:
    rows = []
    for r, logits in zip(records, timestep_logits(model, records, batch_size)):
        rows.append({"trajectory_id": r["trajectory_ids"][0], "task_id": (r.get("task_ids") or [None])[0],
                     "failure": int(r["labels"].reshape(-1)[0].item()), "n_steps": int(logits.numel()),
                     "score": (sc := aggregate(logits, how)), "prob": to_prob(sc),
                     "_step_logits": logits.numpy()})  # for a plugged-in evaluator; not written to disk
    return rows


def episode_metrics(labels, scores) -> dict:
    """Episode-level ROC-AUC, PR-AUC, TPR at 5% FPR. Undefined metrics are NaN (single-class input)."""
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("labels and scores must be 1-D and the same length")
    if not np.isfinite(s).all():
        raise ValueError("non-finite episode scores")
    out = {"n_episodes": int(len(y)), "n_failures": int(y.sum())}
    if len(np.unique(y)) < 2:
        return out | {"roc_auc": math.nan, "pr_auc": math.nan, "tpr_at_5_fpr": math.nan}
    fpr, tpr, _ = roc_curve(y, s)
    return out | {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
                  "tpr_at_5_fpr": float(np.max(tpr[fpr <= 0.05], initial=0.0))}


def evaluate(rows: list[dict], evaluator=None) -> dict:
    """Metrics for scored episodes: the built-in ``episode_metrics`` or a plugged-in evaluator (EvalInput contract)."""
    if evaluator is None:
        return episode_metrics(*_labels_scores(rows))
    data = EvalInput(step_scores=[r["_step_logits"] for r in rows],
                     labels=np.array([r["failure"] for r in rows], dtype=np.int64),
                     horizons=np.array([r["n_steps"] for r in rows], dtype=np.int64),
                     task_ids=np.array([-1 if r["task_id"] is None else r["task_id"] for r in rows], dtype=np.int64),
                     trajectory_ids=[r["trajectory_id"] for r in rows])
    res = evaluator(data)
    if not isinstance(res, dict):
        raise ValueError(f"evaluator {evaluator.name!r} returned {type(res).__name__}, expected a dict")
    y = data.labels
    return {"n_episodes": int(len(y)), "n_failures": int(y.sum())} | {k: float(v) for k, v in res.items()}


def per_task_metrics(rows: list[dict], evaluator=None) -> dict:
    tasks = sorted({r["task_id"] for r in rows}, key=lambda t: (t is None, t))
    return {str(t): evaluate([r for r in rows if r["task_id"] == t], evaluator) for t in tasks}


# ----------------------------------------------------------------------------------------------- training

def train_one(model, train_recs, val_recs, cfg: RunConfig, seed: int, evaluator=None) -> dict:
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(RecordDataset(train_recs), batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=collate_records, generator=gen)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError(f"{cfg.model} has no trainable parameters")
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    history, best = [], {"epoch": None, "roc_auc": -math.inf, "state": None}
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            loss = model.loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch} (seed {seed})")
            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            losses.append(loss.item())
        val = evaluate(episode_scores(model, val_recs, cfg.batch_size, cfg.aggregation), evaluator)
        if "roc_auc" not in val:
            raise ValueError("the evaluator returns no 'roc_auc'; the learning curve and --select-best-epoch need it")
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "seen_val_roc_auc": val["roc_auc"]})
        if cfg.select_best_epoch and not math.isnan(val["roc_auc"]) and val["roc_auc"] > best["roc_auc"]:
            best = {"epoch": epoch, "roc_auc": val["roc_auc"], "state": copy.deepcopy(model.state_dict())}
    if cfg.select_best_epoch and best["state"] is not None:
        model.load_state_dict(best["state"])
        chosen = best["epoch"]
    else:  # no selection, or seen_val AUC undefined at every epoch -> keep the last epoch
        chosen = cfg.epochs
    return {"history": history, "chosen_epoch": chosen,
            "selection": "best seen_val ROC-AUC" if cfg.select_best_epoch else "last epoch"}


def _labels_scores(rows):
    return [r["failure"] for r in rows], [r["score"] for r in rows]


# ----------------------------------------------------------------------------------------------- outputs

def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    with os.fdopen(fd, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=_json_default)
        fh.write("\n")
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, torch.Tensor):
        return o.tolist()
    return str(o)


def _write_scores_csv(path: Path, rows: list[dict]) -> None:
    import pandas as pd
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    os.close(fd)
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows], columns=["trajectory_id", "task_id", "failure", "n_steps", "score", "prob"]).to_csv(tmp, index=False)
    os.replace(tmp, path)


def _write_sheet_csv(path: Path, rows: list[dict]) -> None:
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    os.close(fd)
    extra = [c for r in rows for c in r if c not in SHEET_COLUMNS]
    pd.DataFrame(rows, columns=SHEET_COLUMNS + list(dict.fromkeys(extra))).to_csv(tmp, index=False)
    os.replace(tmp, path)


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


RUN_MARKER = "fusion_runner_output"


def _is_previous_run(path: Path) -> bool:
    """True only for a folder this runner created: our marker in config.json and nothing but our own files."""
    try:
        if json.loads((path / "config.json").read_text()).get("marker") != RUN_MARKER:
            return False
    except (OSError, ValueError, AttributeError):
        return False
    allowed = {"config.json", "summary.json", "sheet_row.csv"}
    for entry in path.iterdir():
        if entry.name in allowed and entry.is_file():
            continue
        if entry.is_dir() and not entry.is_symlink() and entry.name.startswith("seed") and entry.name[4:].lstrip("-").isdigit():
            continue
        if entry.name.startswith(("config.json", "summary.json", "sheet_row.csv")) and entry.is_file():  # temp
            continue
        return False
    return True


RESUME_IGNORED = ("overwrite", "resume", "output_dir", "seeds", "save_checkpoints", "bootstrap", "ci",
                  "latency_device", "memory_check")


def _same_run(path: Path, cfg: RunConfig) -> bool:
    try:
        old = json.loads((path / "config.json").read_text())["config"]
    except (OSError, ValueError, KeyError):
        return False
    new = asdict(cfg)
    return all(old.get(k) == v for k, v in new.items() if k not in RESUME_IGNORED)


def _prepare_output(path: Path, overwrite: bool, cfg: RunConfig | None = None) -> bool:
    """Returns True when an earlier identical run is being resumed (its finished seeds are kept)."""
    path = path.expanduser()
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise FileExistsError(f"{path} is a file or symlink, not an output folder")
    if cfg is not None and cfg.resume and path.exists() and any(path.iterdir()):
        if not _is_previous_run(path.resolve()):
            raise FileExistsError(f"cannot resume {path}: it was not written by this runner")
        if not _same_run(path, cfg):
            raise FileExistsError(f"cannot resume {path}: it was run with a different configuration")
        return True
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{path} exists and is not empty; pass overwrite=True (--overwrite) to replace it")
        resolved = path.resolve()
        protected = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve(), REPO.resolve()}
        if resolved in protected or resolved in REPO.resolve().parents or not _is_previous_run(resolved):
            raise FileExistsError(f"refusing to delete {path}: --overwrite only replaces a folder written by "
                                  f"this runner (config.json marker, only seed*/config/summary entries)")
        import shutil
        shutil.rmtree(resolved)
    path.mkdir(parents=True, exist_ok=True)
    return False


@torch.no_grad()
def measure_latency(model, records, max_episodes: int = 20, steps_per_episode: int = 25, warmup: int = 5,
                    device: str = "cpu") -> dict:
    """Detector latency in ms per timestep, measured the way a deployed monitor runs: ONE call per timestep (batch 1,
    a single step [1, 1, K, D]), no autograd, on the current device/threads. The fusion models score each step on its
    own, so a one-step call is exactly the online cost. Covers the detector only (features already extracted), i.e.
    what it adds on top of the VLA forward pass.

    ms_per_step: median over up to `steps_per_episode` evenly spaced steps of up to `max_episodes` seen_val episodes.
    ms_per_step_amortized: whole-episode forward / steps (offline scoring; much smaller, not the online cost)."""
    recs = list(records)[:max_episodes]
    if not recs:
        return {"ms_per_step": math.nan, "n_episodes": 0}
    dev = torch.device(device)
    m = model if dev.type == "cpu" else copy.deepcopy(model).to(dev)
    was_training = m.training
    m.eval()
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)

    def to_dev(b):
        return {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in b.items()}

    def one_step(r, t):
        return to_dev(collate_records([{**r, "features": r["features"][:, t:t + 1], "mask": r["mask"][:, t:t + 1]}]))

    def timed(batch) -> float:
        sync()
        t0 = time.perf_counter()
        m(batch)
        sync()
        return (time.perf_counter() - t0) * 1e3

    online, amortized = [], []
    with torch.inference_mode():
        for _ in range(warmup):
            m(one_step(recs[0], 0))
        for r in recs:
            n = int(r["mask"].sum())
            if n < 1:
                continue
            for t in np.unique(np.linspace(0, n - 1, min(steps_per_episode, n)).astype(int)):
                online.append(timed(one_step(r, int(t))))
            amortized.append(timed(to_dev(collate_records([r]))) / n)
    m.train(was_training)
    return {"ms_per_step": float(np.median(online)) if online else math.nan,
            "ms_per_step_amortized": float(np.median(amortized)) if amortized else math.nan,
            "n_episodes": len(recs), "n_calls": len(online), "device": dev.type, "threads": torch.get_num_threads(),
            "definition": "median detector forward time for one timestep, one call per step (online), batch 1"}


# ----------------------------------------------------------------------------------------------- bootstrap CIs

STEP_SCORES = "seen_val_step_scores.npz"  # same layout as the depth sweep's rows/*.scores.npz


def save_step_scores(path: Path, rows: list[dict]) -> None:
    lengths = np.array([len(r["_step_logits"]) for r in rows], dtype=np.int64)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".npz", dir=path.parent)
    os.close(fd)
    np.savez(tmp, step_scores=np.concatenate([np.asarray(r["_step_logits"], np.float32) for r in rows]),
             offsets=np.concatenate([[0], np.cumsum(lengths)]),
             labels=np.array([r["failure"] for r in rows], np.float32), horizons=lengths,
             task_ids=np.array([-1 if r["task_id"] is None else r["task_id"] for r in rows], np.int64),
             trajectory_ids=np.array([r["trajectory_id"] for r in rows], dtype=str))
    os.replace(tmp, path)


def load_step_scores(path: Path) -> EvalInput:
    """A fusion run's seed<s>/seen_val_step_scores.npz or a depth-sweep rows/L<l>_seed<s>.scores.npz."""
    with np.load(path) as z:
        flat, off = z["step_scores"], z["offsets"]
        return EvalInput([flat[off[i]:off[i + 1]] for i in range(len(off) - 1)], z["labels"].astype(np.int64),
                         z["horizons"].astype(np.int64), z["task_ids"], [str(t) for t in z["trajectory_ids"]])


def align(inputs: list[EvalInput], ref_ids: list[str] | None = None) -> list[EvalInput]:
    """Put every EvalInput in the same episode order; they must cover exactly the same episodes."""
    ref = list(ref_ids if ref_ids is not None else inputs[0].trajectory_ids)
    out = []
    for d in inputs:
        pos = {t: i for i, t in enumerate(d.trajectory_ids)}
        if set(pos) != set(ref) or len(pos) != len(d.trajectory_ids):
            raise ValueError("score files cover different (or duplicated) seen_val episodes")
        o = [pos[t] for t in ref]
        out.append(EvalInput([d.step_scores[i] for i in o], np.asarray(d.labels)[o], np.asarray(d.horizons)[o],
                             np.asarray(d.task_ids)[o], ref))
    if any(not np.array_equal(out[0].labels, d.labels) for d in out):
        raise ValueError("score files disagree on episode labels")
    return out


class Scorer:
    """Metrics of one EvalInput on a subset of its episodes: the plugged-in evaluator, or the built-in scorer
    (episode score = cfg.aggregation of the step scores over each episode's horizon, then episode_metrics)."""

    def __init__(self, evaluator=None, how: str = "max"):
        self.evaluator, self.how = evaluator, how

    def prepare(self, d: EvalInput):
        if self.evaluator is not None:
            return d
        return np.array([aggregate(torch.as_tensor(np.asarray(s[:h], np.float64)), self.how)
                         for s, h in zip(d.step_scores, d.horizons)])

    def __call__(self, d: EvalInput, prepared, idx: np.ndarray) -> dict:
        if self.evaluator is None:
            m = episode_metrics(np.asarray(d.labels)[idx], prepared[idx])
            return {k: v for k, v in m.items() if k not in ("n_episodes", "n_failures")}
        sub = EvalInput([d.step_scores[i] for i in idx], np.asarray(d.labels)[idx], np.asarray(d.horizons)[idx],
                        np.asarray(d.task_ids)[idx], [d.trajectory_ids[i] for i in idx])
        return {k: float(v) for k, v in self.evaluator(sub).items()}


def stratified_indices(labels, n: int, seed: int):
    """n resamples of episode indices with replacement, separately among failures and successes (the depth sweep's
    scheme: class counts are kept, so every metric stays defined)."""
    labels = np.asarray(labels)
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels != 1)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def bootstrap_ci(inputs: list[EvalInput], scorer: Scorer, n: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """Percentile CI of the seed-averaged metrics, resampling seen_val episodes (the same episodes for every seed)."""
    inputs = align(inputs)
    prepared = [scorer.prepare(d) for d in inputs]
    samples: dict[str, list[float]] = {}
    for idx in stratified_indices(inputs[0].labels, n, seed):
        per_seed = [scorer(d, p, idx) for d, p in zip(inputs, prepared)]
        for k in per_seed[0]:
            samples.setdefault(k, []).append(float(np.nanmean([m[k] for m in per_seed])))
    lo, hi = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    out = {}
    for k, v in samples.items():
        v = np.asarray(v)
        v = v[np.isfinite(v)]
        out[k] = [float(np.percentile(v, lo)), float(np.percentile(v, hi))] if len(v) else [math.nan, math.nan]
    return {"ci": ci, "n_resamples": n, "seed": seed, "intervals": out,
            "scheme": "stratified by outcome, same episodes for every seed, percentile CI of the seed mean"}


def add_bootstrap(run_dir: Path, n: int, ci: float = 0.95, seed: int = 0) -> dict:
    """(Re)compute the CI of a finished run from its saved step scores; updates summary.json and sheet_row.csv."""
    run_dir = Path(run_dir)
    conf = json.loads((run_dir / "config.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    seeds = summary["seeds"]
    inputs = [load_step_scores(run_dir / f"seed{s}" / STEP_SCORES) for s in seeds]
    scorer = Scorer(resolve_evaluator(conf["config"].get("evaluator")), conf["config"].get("aggregation", "max"))
    boot = bootstrap_ci(inputs, scorer, n, ci, seed)
    for k, (lo, hi) in boot["intervals"].items():
        if k in summary["metrics"]:
            summary["metrics"][k]["ci"] = [lo, hi]
    summary["bootstrap"] = {k: v for k, v in boot.items() if k != "intervals"}
    summary["sheet_row"] = _with_ci_notes(summary["sheet_row"], boot)
    _write_json(run_dir / "summary.json", summary)
    _write_sheet_csv(run_dir / "sheet_row.csv", [summary["sheet_row"]])
    return summary


def _with_ci_notes(row: dict, boot: dict) -> dict:
    row = dict(row)
    pct = int(round(100 * boot["ci"]))
    parts = [f"{key} {pct}% CI [{100 * boot['intervals'][key][0]:.2f}, {100 * boot['intervals'][key][1]:.2f}]"
             for key, _ in SHEET_METRICS if key in boot["intervals"]]
    notes = [p for p in str(row.get("Notes", "")).split("; ") if p and " CI [" not in p]
    row["Notes"] = "; ".join(notes + parts)
    return row


SHEET_METRICS = (("roc_auc", "Seen Val ROC-AUC (%)"), ("pr_auc", "Seen Val PR-AUC (%)"),
                 ("tpr_at_5_fpr", "Seen Val TPR @ 5% FPR (%)"))
SHEET_COLUMNS = ["Experiment", *(c for _, c in SHEET_METRICS), "Parameters", "Latency (ms)", "Notes"]


def sheet_row(summary_metrics: dict, per_seed: dict[int, dict], label: str) -> dict:
    """One row for the experiments sheet: metrics in percent (0-100, SAFE Table 1 scale), mean over seeds, with the
    std across seeds in Notes. Unseen-test columns are never filled here (locked until N3)."""
    row = {"Experiment": label}
    notes = []
    for key, col in SHEET_METRICS:
        m = summary_metrics.get(key)
        if m is None:
            row[col] = math.nan
            continue
        row[col] = round(100 * m["mean"], 2) if not math.isnan(m["mean"]) else math.nan
        if not math.isnan(m["std"]):
            notes.append(f"{key} {100 * m['mean']:.2f}±{100 * m['std']:.2f}")
    first = next(iter(per_seed.values()))
    row["Parameters"] = int(first["trainable_parameters"])
    lat = [m["latency"]["ms_per_step"] for m in per_seed.values() if m.get("latency")]
    row["Latency (ms)"] = round(float(np.median(lat)), 4) if lat else math.nan
    row["Notes"] = (f"seen_val, mean±std over seeds {sorted(per_seed)}; " + "; ".join(notes)).strip("; ")
    return row


def summarize(per_seed: dict[int, dict]) -> dict:
    first = next(iter(per_seed.values()))["seen_val"]
    keys = [k for k in first if k not in ("n_episodes", "n_failures")]  # whatever the evaluator reports
    out = {}
    for k in keys:
        vals = np.array([m["seen_val"].get(k, math.nan) for m in per_seed.values()], dtype=float)
        ok = vals[~np.isnan(vals)]
        out[k] = {"mean": float(ok.mean()) if ok.size else math.nan, "n_seeds_defined": int(ok.size),
                  "std": float(ok.std(ddof=1)) if ok.size > 1 else math.nan,
                  "per_seed": {str(s): float(v) for s, v in zip(per_seed, vals)}}
    weights = [m["layer_weights"] for m in per_seed.values() if m.get("layer_weights") is not None]
    if weights:
        w = np.array(weights)
        out["layer_weights"] = {"mean": w.mean(0).tolist(), "std": w.std(0).tolist()}
    return out


# ----------------------------------------------------------------------------------------------- entry point

def run(cfg: RunConfig, splits: dict[str, list[dict]] | None = None, provenance: dict | None = None,
        label: str | None = None) -> dict:
    """Train/evaluate cfg.model for every seed; write outputs under cfg.output_dir; return the summary.
    `splits` overrides data loading (tests / callers that already hold records, e.g. fusion_experiments)."""
    validate_config(cfg, need_source=splits is None)
    evaluator = resolve_evaluator(cfg.evaluator)  # fail before any data is read
    provisional = provisional_flags(evaluator)
    out = Path(cfg.output_dir)
    resuming = _prepare_output(out, cfg.overwrite, cfg)
    done = {s for s in cfg.seeds if resuming and (out / f"seed{s}" / "metrics.json").is_file()
            and (out / f"seed{s}" / STEP_SCORES).is_file()}
    if resuming:
        print(f"resume {out}: seeds {sorted(done)} already finished", file=sys.stderr)
    if done == set(cfg.seeds):  # nothing to train: no data needed
        run_info = json.loads((out / "config.json").read_text())
        prov, d = run_info["provenance"], run_info["feature_dim"]
    else:
        if splits is None:
            splits, prov = load_synthetic_splits(cfg) if cfg.synthetic else load_cache_splits(cfg)
        else:
            prov = provenance or {"source": "records passed by caller"}
        d = check_splits(splits, len(cfg.layers))
        run_info = {"marker": RUN_MARKER, "config": asdict(cfg), "provenance": prov, "provisional": provisional,
                    "evaluator": getattr(evaluator, "name", "builtin_" + cfg.aggregation),
                    "git_commit": _git_commit(), "feature_dim": d,
                    "n_episodes": {s: len(r) for s, r in splits.items()}, "synthetic": bool(cfg.synthetic)}
        _write_json(out / "config.json", run_info)
    if cfg.horizon != "task_min_step":
        print("WARNING: horizon='full' lets episode length reveal the label; not reportable", file=sys.stderr)

    per_seed = {}
    for seed in cfg.seeds:
        if seed in done:
            per_seed[seed] = json.loads((out / f"seed{seed}" / "metrics.json").read_text())
            continue
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = get_model(cfg.model, **model_kwargs(cfg, d))
        trained = train_one(model, splits["train"], splits["seen_val"], cfg, seed, evaluator)
        rows = episode_scores(model, splits["seen_val"], cfg.batch_size, cfg.aggregation)
        metrics = {
            "seed": seed, "model": cfg.model, "layers": list(cfg.layers),
            "seen_val": evaluate(rows, evaluator),
            "seen_val_per_task": per_task_metrics(rows, evaluator),
            # train is scored at train_horizon (full by default), so it is a fit check, not comparable to seen_val
            "train": evaluate(episode_scores(model, splits["train"], cfg.batch_size, cfg.aggregation), evaluator),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "layer_weights": model.layer_weights().tolist() if hasattr(model, "layer_weights") else None,
            "latency": measure_latency(model, splits["seen_val"], device=cfg.latency_device),
            **trained, "provisional": provisional, "synthetic": bool(cfg.synthetic),
        }
        seed_dir = out / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        _write_json(seed_dir / "metrics.json", metrics)
        _write_scores_csv(seed_dir / "seen_val_episode_scores.csv", rows)
        save_step_scores(seed_dir / STEP_SCORES, rows)
        if cfg.save_checkpoints:
            torch.save({"model": cfg.model, "model_kwargs": model_kwargs(cfg, d), "model_state": model.state_dict(),
                        "config": asdict(cfg), "seed": seed, "chosen_epoch": trained["chosen_epoch"]},
                       seed_dir / "model.pt")
        per_seed[seed] = metrics
        auc = metrics["seen_val"]["roc_auc"]
        print(f"[{cfg.model} layers={list(cfg.layers)} seed={seed}] seen_val ROC-AUC={auc:.4f} "
              f"(epoch {trained['chosen_epoch']}, {metrics['trainable_parameters']} params)", flush=True)

    summary = {"model": cfg.model, "layers": list(cfg.layers), "seeds": list(cfg.seeds), "metrics": summarize(per_seed),
               "provisional": provisional, "synthetic": bool(cfg.synthetic),
               "evaluator": run_info["evaluator"],
               "horizon": "n/a (synthetic)" if cfg.synthetic else cfg.horizon,
               "train_horizon": "n/a (synthetic)" if cfg.synthetic else cfg.train_horizon,
               "split_label": prov.get("split_label")}
    label = label or f"{cfg.model} layers={'/'.join(map(str, cfg.layers))}"
    summary["sheet_row"] = sheet_row(summary["metrics"], per_seed, label)
    _write_json(out / "summary.json", summary)
    _write_sheet_csv(out / "sheet_row.csv", [summary["sheet_row"]])
    if cfg.bootstrap:
        summary = add_bootstrap(out, cfg.bootstrap, cfg.ci)
        ci = summary["metrics"]["roc_auc"].get("ci")
        if ci:
            print(f"[{label}] seen_val ROC-AUC {100 * summary['metrics']['roc_auc']['mean']:.2f}% "
                  f"({int(round(100 * cfg.ci))}% CI {100 * ci[0]:.2f}-{100 * ci[1]:.2f})", flush=True)
    return summary
