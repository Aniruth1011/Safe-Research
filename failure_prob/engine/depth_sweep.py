"""Depth sweep over the shared cache (plan rows R1-R4): one probe per layer x training seed, scored on seen_val.

    train on split "train"  ->  score every seen_val rollout  ->  evaluator at the task_min_step horizon

Never reads unseen_test: SeenOnlyCache refuses that split before any feature file is opened, and the sweep also
refuses a cache whose train/seen_val episodes include a task from protocol.UNSEEN_TASKS.

Outputs, in one folder per configuration (pool, normalize, train horizon, probe, evaluator, cache):
    config.json                 the configuration; a second run into the same folder must match it
    rows/L{LL}_seed{S}.json     one row per (layer, seed), written atomically, ending in "completed": true and
                                the Slurm job / array task that wrote it; reruns reuse such rows and log each reuse
    rows/_discarded/            rows that were not reusable (moved aside, never deleted), then recomputed
    _discarded/tmp/             temp files left by tasks killed mid-write (moved aside at the start of a run)
    results.csv                 all rows (aggregate)
    summary.csv                 mean / std / n over seeds per layer (aggregate)
    depth_curve.png             mean +- std per layer (aggregate)
Rows made with a TEMPORARY probe or evaluator say so in every file and in the plot title.
Every file and folder written here is made readable by group and others (0644 / 0755), so results can be shared.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import re
import queue
import resource
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from failure_prob.engine.sweep_plugins import EVALUATORS, PROBES, EvalInput, resolve
from failure_prob.protocol import TRAIN_SEEDS, UNSEEN_TASKS

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOADER = REPO_ROOT / "collection" / "load_cache.py"
ALLOWED_SPLITS = ("train", "seen_val")  # unseen_test is never touched by the sweep
METRIC_PREFIX = "val_"
FILE_MODE, DIR_MODE = 0o644, 0o755  # results are shared with the team
# Who wrote a row. Kept in rows/*.json (on CARC only) but left out of results.csv, which may be shared more widely:
# job IDs and log paths (they contain the user name) must not reach the public repo.
JOB_FIELDS = ("slurm_job_id", "slurm_array_job_id", "slurm_array_task_id", "log_path")


def load_cache_module(path: Path = DEFAULT_LOADER):
    """Import collection/load_cache.py (it is a script folder, not a package)."""
    spec = importlib.util.spec_from_file_location("load_cache", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------- file-system friendly reads
class ReadStalled(RuntimeError):
    """A read from the cache made no progress within the stall limit, retries included."""


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def read_npy_sequential(path: Path, log=print, chunk_mb: float | None = None, stall_s: float | None = None,
                        retries: int | None = None, progress_s: float = 5.0) -> np.ndarray:
    """Read a .npy file into memory in one sequential pass of large chunks.

    This replaces np.load(mmap_mode="r") followed by per-episode slicing, which turns into many small page-fault reads
    on a shared file system. Every chunk is read by a helper thread. If a chunk makes no progress for `stall_s`
    seconds, the read is retried on a fresh file handle, up to `retries` times; then ReadStalled is raised. A thread
    stuck in uninterruptible I/O cannot be stopped, only abandoned, so the caller should end the process.
    Progress (MB read) is logged every `progress_s` seconds. Defaults: $SWEEP_READ_CHUNK_MB (64),
    $SWEEP_READ_STALL_S (30), $SWEEP_READ_RETRIES (2).
    """
    chunk = int((chunk_mb if chunk_mb is not None else _env_float("SWEEP_READ_CHUNK_MB", 64)) * (1 << 20))
    stall_s = stall_s if stall_s is not None else _env_float("SWEEP_READ_STALL_S", 30)
    retries = int(retries if retries is not None else _env_float("SWEEP_READ_RETRIES", 2))
    path = Path(path)
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(f)
        offset = f.tell()
        size = os.fstat(f.fileno()).st_size
    nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if size != offset + nbytes:
        raise ValueError(f"{path}: {size} bytes on disk, header says {offset + nbytes}")
    raw = np.empty(nbytes, dtype=np.uint8)
    mb = nbytes / 1e6
    t0 = last = time.time()
    pos = 0
    while pos < nbytes:
        n = min(chunk, nbytes - pos)
        raw[pos:pos + n] = np.frombuffer(_read_chunk(path, offset + pos, n, stall_s, retries, log, pos), np.uint8)
        pos += n
        now = time.time()
        if now - last >= progress_s and pos < nbytes:
            log(f"reading {path.name}: {pos / 1e6:.0f}/{mb:.0f} MB ({now - t0:.1f} s)")
            last = now
    dt = max(time.time() - t0, 1e-9)
    log(f"read {'/'.join(path.parts[-3:])}: {mb:.0f} MB in {dt:.1f} s ({mb / dt:.0f} MB/s)")
    return np.ndarray(shape, dtype=dtype, buffer=raw, order="F" if fortran else "C")


def _read_chunk(path: Path, offset: int, size: int, stall_s: float, retries: int, log, pos: int) -> bytes:
    results: queue.Queue = queue.Queue()

    def worker(attempt: int) -> None:
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                parts, got = [], 0
                while got < size:
                    b = os.pread(fd, size - got, offset + got)
                    if not b:
                        raise EOFError(f"{path}: file ended at byte {offset + got}")
                    parts.append(b)
                    got += len(b)
            finally:
                os.close(fd)
            results.put((attempt, b"".join(parts), None))
        except BaseException as e:  # noqa: BLE001  (reported to the waiting thread)
            results.put((attempt, None, e))

    def start(attempt: int) -> None:
        threading.Thread(target=worker, args=(attempt,), daemon=True, name=f"read-{path.name}-{attempt}").start()

    attempt, running, last_error = 0, 1, None
    start(0)
    while True:
        try:
            _, data, err = results.get(timeout=stall_s)
        except queue.Empty:
            if attempt >= retries:
                raise ReadStalled(f"{path}: no progress for {stall_s:.0f} s at {pos / 1e6:.0f} MB after {retries} "
                                  f"retries (file system stalled?)") from None
            attempt += 1
            running += 1
            log(f"reading {path.name}: no progress for {stall_s:.0f} s at {pos / 1e6:.0f} MB; "
                f"retrying on a new file handle ({attempt}/{retries})")
            start(attempt)
            continue
        running -= 1
        if err is None:
            return data
        last_error = err
        if attempt < retries:
            attempt += 1
            running += 1
            log(f"reading {path.name} at {pos / 1e6:.0f} MB failed ({type(err).__name__}: {err}); "
                f"retrying ({attempt}/{retries})")
            time.sleep(1)
            start(attempt)
        elif running == 0:
            raise last_error


class SeenOnlyCache:
    """SharedCache restricted to train and seen_val. Any request for another split raises before a file is opened.

    Feature files are read whole and sequentially (read_npy_sequential), not memory-mapped; the values handed to the
    loader are identical. They are dropped again once a split's records are built.
    """

    def __init__(self, cache_dir: Path, loader: Path = DEFAULT_LOADER, log=print):
        self._cache = load_cache_module(loader).SharedCache(cache_dir)
        inner_array = self._cache._array
        mm = getattr(self._cache, "_mm", None)  # the loader's per-file cache, keyed (pool, split, layer)
        root = Path(self._cache.root)

        def guarded_array(pool, split, layer):
            _check_split(split)
            path = root / "features" / pool / split / f"L{layer:02d}.npy"
            if isinstance(mm, dict) and (pool, split, layer) not in mm and path.exists():
                mm[(pool, split, layer)] = read_npy_sequential(path, log=log)
            return inner_array(pool, split, layer)

        self._cache._array = guarded_array  # second line of defence, in case the loader changes
        self._mm = mm
        self.manifest = self._cache.manifest
        self.episodes = self._cache.episodes
        leaked = sorted(set(self.episodes[self.episodes.split.isin(ALLOWED_SPLITS)].task_id) & set(UNSEEN_TASKS))
        if leaked:
            raise ValueError(f"cache puts unseen tasks {leaked} in train/seen_val; refusing (protocol.UNSEEN_TASKS)")

    def records(self, split: str, layer: int, pool: str, normalize: bool, horizon: str) -> list[dict]:
        _check_split(split)
        out = self._cache.records(split, [layer], pool=pool, normalize=normalize, horizon=horizon)
        if isinstance(self._mm, dict):
            self._mm.pop((pool, split, layer), None)  # the records hold their own float32 copies
        return out

    def layer_names(self) -> list[str]:
        names = self.manifest.get("layer_names")
        if names:
            return list(names)
        n = int(self.manifest.get("num_hidden_states", 33))
        return ["emb"] + [f"block{i:02d}" for i in range(1, 32)] + ["block32_postnorm"] if n == 33 else [f"L{i}" for i in range(n)]

    def identity(self) -> dict:
        root = Path(self._cache.root)
        return {"cache_version": self.manifest.get("cache_version"), "created_utc": self.manifest.get("created_utc"),
                "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()}


def _check_split(split: str) -> None:
    if split not in ALLOWED_SPLITS:
        raise PermissionError(f"the depth sweep never reads {split!r} (allowed: {ALLOWED_SPLITS})")


@dataclass(frozen=True)
class SweepConfig:
    cache_dir: str
    pool: str = "mean"
    normalize: bool = False
    train_horizon: str = "full"          # "full" or "task_min_step"; seen_val is always scored at task_min_step
    probe: str = "temporary_mlp"
    evaluator: str = "temporary_max"
    seeds: tuple[int, ...] = field(default=TRAIN_SEEDS)

    def label(self) -> str:
        safe = lambda s: s.replace(":", "-").replace(".", "_").replace("/", "_")
        return f"{self.pool}-{'trainnorm' if self.normalize else 'raw'}-train_{self.train_horizon}-{safe(self.probe)}-{safe(self.evaluator)}"


def _git_commit() -> str:
    try:
        sha = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
        return (sha or "unknown") + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def _peak_rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6  # bytes on macOS, KiB on Linux


def _share(path: Path) -> None:
    """Make a file 0644 / a folder 0755. Silently skips paths owned by someone else."""
    try:
        os.chmod(path, DIR_MODE if Path(path).is_dir() else FILE_MODE)
    except OSError:
        pass


def _mkdir_shared(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _share(path)


def _write_json_atomic(path: Path, obj) -> None:
    _mkdir_shared(path.parent)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")  # mkstemp makes 0600 files ...
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")
        os.chmod(tmp, FILE_MODE)                                           # ... so open it up before the rename
        os.replace(tmp, path)
    except BaseException:                                                  # incl. SystemExit from SIGTERM
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# Temp files of _write_json_atomic (".<name>.<random>") and other hidden 8-character tempfile names (".<random>").
_TEMP_NAME = re.compile(r"^\.(?:[A-Za-z0-9_]+\.json\.)?[a-z0-9_]{8}$")
STALE_TEMP_S = 15 * 60  # far longer than a task's time limit, so a task still writing is never disturbed


def move_aside_stale_temp_files(out_dir: Path, log=print, min_age_s: float = STALE_TEMP_S) -> list[Path]:
    """Move temp files left by tasks killed mid-write into out_dir/_discarded/tmp/. Only files older than min_age_s
    are touched: a temp file of a task that is running now must stay where it is."""
    out_dir, moved, now = Path(out_dir), [], time.time()
    for folder in (out_dir, out_dir / "rows"):
        if not folder.is_dir():
            continue
        for f in sorted(folder.iterdir()):
            try:
                stale = f.is_file() and _TEMP_NAME.match(f.name) and now - f.stat().st_mtime >= min_age_s
            except OSError:  # vanished meanwhile (another task renamed it)
                continue
            if stale:
                dest = out_dir / "_discarded" / "tmp" / (f"rows_{f.name}" if folder.name == "rows" else f.name)
                _mkdir_shared(dest.parent)
                try:
                    os.replace(f, dest)
                except OSError:
                    continue
                log(f"moved leftover temp file {f.relative_to(out_dir)} (from a task killed mid-write) to {dest}")
                moved.append(dest)
    return moved


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _job_identity() -> dict:
    """The Slurm job (and array task) running this process; all None outside Slurm."""
    env = os.environ
    return {"slurm_job_id": env.get("SLURM_JOB_ID"), "slurm_array_job_id": env.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": env.get("SLURM_ARRAY_TASK_ID"), "log_path": env.get("SWEEP_LOG")}


def _job_label(row: dict) -> str:
    """'<array job>_<task>' as sacct shows it, else the job ID, else 'no Slurm job'."""
    if row.get("slurm_array_job_id") and row.get("slurm_array_task_id") is not None:
        return f"job {row['slurm_array_job_id']}_{row['slurm_array_task_id']}"
    return f"job {row['slurm_job_id']}" if row.get("slurm_job_id") else "no Slurm job"


def check_row(path: Path, layer: int, seed: int) -> tuple[str, dict | None, str]:
    """Classify an existing row file: ("reuse" | "legacy" | "invalid", row or None, reason).

    reuse   written by this version: "completed": true and the job fields (the job ID may be None outside Slurm)
    legacy  written before job IDs were recorded (no "completed" and no job fields at all); reused, as agreed
    invalid anything else (unreadable, wrong layer/seed, completed missing or false): moved aside and recomputed
    """
    try:
        row = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        return "invalid", None, f"unreadable ({type(e).__name__})"
    if not isinstance(row, dict) or row.get("layer") != layer or row.get("seed") != seed:
        return "invalid", None, "layer/seed do not match the file name"
    if not any(k.startswith(METRIC_PREFIX) for k in row):
        return "invalid", row, "no metrics"
    has_job = all(k in row for k in JOB_FIELDS[:3])
    if row.get("completed") is True and has_job:
        return "reuse", row, ""
    if "completed" not in row and not any(k in row for k in JOB_FIELDS):
        return "legacy", row, ""
    return "invalid", row, "not marked completed by a recorded job"


def _discard(path: Path, reason: str, log) -> None:
    dest = path.parent / "_discarded" / f"{path.stem}.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    _mkdir_shared(dest.parent)
    os.replace(path, dest)
    log(f"{path.name}: {reason}; moved to {dest} and recomputing")


def _check_or_write_config(out_dir: Path, cfg: SweepConfig, cache: SeenOnlyCache, probe, evaluator) -> dict:
    cfg_doc = {**asdict(cfg), "seeds": list(cfg.seeds), "cache": cache.identity(),
               "probe_name": probe.name, "probe_temporary": bool(probe.temporary),
               "evaluator_name": evaluator.name, "evaluator_temporary": bool(evaluator.temporary)}
    cfg_doc.pop("cache_dir")  # a path (may contain a user name); the cache identity above is what matters
    path = out_dir / "config.json"
    if path.exists():
        old = json.loads(path.read_text())
        if {k: v for k, v in old.items() if k != "seeds"} != {k: v for k, v in cfg_doc.items() if k != "seeds"}:
            raise ValueError(f"{out_dir} holds a different configuration ({path}); use another --out folder")
        cfg_doc["seeds"] = sorted(set(old["seeds"]) | set(cfg_doc["seeds"]))
    _write_json_atomic(path, cfg_doc)
    return cfg_doc


def run_layer(cfg: SweepConfig, layer: int, out_dir: Path, loader: Path = DEFAULT_LOADER, overwrite: bool = False,
              log=print, on_loaded=None) -> list[dict]:
    """Train cfg.seeds probes on one layer and score seen_val. Returns the rows (also written to out_dir/rows/).

    on_loaded() is called once the layer's data is in memory, or once it is clear nothing needs loading
    (the CLI's start-up watchdog uses it)."""
    out_dir = Path(out_dir)
    on_loaded = on_loaded or (lambda: None)
    log(f"opening cache {cfg.cache_dir} (loader {loader})")
    t_open = time.time()
    cache = SeenOnlyCache(Path(cfg.cache_dir), loader, log=log)
    log(f"cache open ({time.time() - t_open:.1f} s): {len(cache.episodes)} episodes, "
        f"version {cache.manifest.get('cache_version')}")
    names = cache.layer_names()
    if not 0 <= layer < len(names):
        raise ValueError(f"layer {layer} not in this cache (0..{len(names) - 1})")
    probe, evaluator = resolve(cfg.probe, PROBES), resolve(cfg.evaluator, EVALUATORS)
    _mkdir_shared(out_dir)
    move_aside_stale_temp_files(out_dir, log)
    cfg_doc = _check_or_write_config(out_dir, cfg, cache, probe, evaluator)
    todo = []
    for seed in cfg.seeds:
        path = out_dir / "rows" / f"L{layer:02d}_seed{seed}.json"
        if not path.exists():
            todo.append(seed)
            continue
        if overwrite:
            _discard(path, "--overwrite given", log)
            todo.append(seed)
            continue
        status, row, reason = check_row(path, layer, seed)
        if status == "reuse":
            log(f"layer {layer} seed {seed}: reusing row from {_job_label(row)}, commit {row.get('git_commit')}, "
                f"written {row.get('written_utc')} ({path})")
        elif status == "legacy":
            mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            log(f"layer {layer} seed {seed}: reusing legacy row (written before job IDs were recorded), "
                f"commit {row.get('git_commit')}, file time {mtime} ({path})")
        else:
            _discard(path, reason, log)
            todo.append(seed)
    if not todo:
        log(f"layer {layer}: every seed {list(cfg.seeds)} reused; nothing to compute")
        on_loaded()
        return []

    t0 = time.time()
    train = cache.records("train", layer, cfg.pool, cfg.normalize, cfg.train_horizon)
    val = cache.records("seen_val", layer, cfg.pool, cfg.normalize, "full")
    load_s = time.time() - t0
    on_loaded()
    ep = cache.episodes.set_index("trajectory_id")
    val_ids = [r["trajectory_ids"][0] for r in val]
    horizons = np.array([min(int(ep.at[t, "length"]), int(ep.at[t, "task_min_step"])) for t in val_ids])
    labels = np.array([float(r["labels"].reshape(-1)[0]) for r in val])
    task_ids = np.array([int(r["task_ids"][0]) for r in val])
    if set(task_ids) & set(UNSEEN_TASKS):
        raise ValueError("seen_val contains an unseen task")  # cannot happen after SeenOnlyCache's check
    log(f"layer {layer} ({names[layer]}): loaded {len(train)} train / {len(val)} seen_val rollouts in {load_s:.1f} s")

    rows = []
    for seed in todo:
        t1 = time.time()
        fitted = probe.fit(train, seed)
        fit_s = time.time() - t1
        t2 = time.time()
        scores = fitted.step_scores(val)
        metrics = evaluator(EvalInput(scores, labels, horizons, task_ids, val_ids))
        score_s = time.time() - t2
        temporary = bool(probe.temporary or evaluator.temporary)
        row = {
            "experiment": "R1-R4_depth_sweep", "layer": layer, "layer_name": names[layer], "seed": seed,
            "pool": cfg.pool, "normalize": cfg.normalize, "train_horizon": cfg.train_horizon,
            "eval_split": "seen_val", "eval_horizon": "task_min_step",
            "probe": probe.name, "evaluator": evaluator.name, "TEMPORARY": temporary,
            "n_train_rollouts": len(train), "n_train_steps": int(sum(r["features"].shape[1] for r in train)),
            "n_val_rollouts": len(val), "n_val_failures": int(labels.sum()),
            **{METRIC_PREFIX + k: v for k, v in metrics.items()},
            "load_s": round(load_s, 2), "fit_s": round(fit_s, 2), "score_s": round(score_s, 2),
            "peak_rss_gb": round(_peak_rss_gb(), 2), "threads": _threads(),
            "cache_version": cfg_doc["cache"]["cache_version"], "cache_manifest_sha256": cfg_doc["cache"]["manifest_sha256"],
            "git_commit": _git_commit(), "python": platform.python_version(), "torch": _torch_version(),
            **_job_identity(), "written_utc": _utc_now(),
            "completed": True,  # set only here, after the metrics exist; reruns reuse a row only if it is True
        }
        _write_json_atomic(out_dir / "rows" / f"L{layer:02d}_seed{seed}.json", row)
        shown = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        log(f"layer {layer} seed {seed}: {shown}  (fit {fit_s:.1f} s, score {score_s:.1f} s, peak RSS {row['peak_rss_gb']} GB)"
            + ("  [TEMPORARY probe/evaluator]" if temporary else ""))
        rows.append(row)
    return rows


def _threads() -> int:
    try:
        import torch
        return torch.get_num_threads()
    except ImportError:
        return 0


def _torch_version() -> str:
    try:
        import torch
        return torch.__version__
    except ImportError:
        return "none"


def parse_layers(spec: str) -> list[int]:
    """'0-33' or '2,5,10-12' -> sorted layer list."""
    out: set[int] = set()
    for part in filter(None, (p.strip() for p in spec.split(","))):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return sorted(out)


def missing_rows(out_dir: Path, layers: list[int]) -> dict[int, list[int]]:
    """{layer: [seeds]} for every expected (layer, seed) row that does not exist yet (seeds from config.json)."""
    seeds = json.loads((Path(out_dir) / "config.json").read_text())["seeds"]
    rows = Path(out_dir) / "rows"
    out = {l: [s for s in seeds if not (rows / f"L{l:02d}_seed{s}.json").exists()] for l in layers}
    return {l: s for l, s in out.items() if s}


def aggregate(out_dir: Path, metric: str = "roc_auc", plot: bool = True, log=print) -> dict:
    """Collect rows into results.csv and summary.csv, and draw depth_curve.png. Returns the paths written."""
    import pandas as pd
    out_dir = Path(out_dir)
    move_aside_stale_temp_files(out_dir, log)
    files = sorted((out_dir / "rows").glob("L*_seed*.json"))
    if not files:
        raise FileNotFoundError(f"no rows under {out_dir / 'rows'}")
    df = pd.DataFrame([json.loads(f.read_text()) for f in files]).sort_values(["layer", "seed"])
    df = df.drop(columns=[c for c in JOB_FIELDS if c in df.columns])
    if df.eval_split.ne("seen_val").any():
        raise ValueError("rows scored on something other than seen_val")
    if df[["probe", "evaluator", "pool", "normalize", "train_horizon", "cache_manifest_sha256"]].nunique().max() > 1:
        raise ValueError("rows from different configurations in one folder")
    df.to_csv(out_dir / "results.csv", index=False)
    metric_cols = [c for c in df.columns if c.startswith(METRIC_PREFIX)]
    g = df.groupby(["layer", "layer_name"])[metric_cols]
    summary = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std")).join(g.size().rename("n_seeds"))
    summary = summary.reset_index()
    temporary = bool(df.TEMPORARY.any())
    summary.insert(0, "TEMPORARY", temporary)
    summary.to_csv(out_dir / "summary.csv", index=False)
    written = {"results": out_dir / "results.csv", "summary": out_dir / "summary.csv"}
    if plot:
        written["plot"] = _plot(summary, df.iloc[0], METRIC_PREFIX + metric, out_dir / "depth_curve.png", temporary)
    # open up everything here (also rows written before rows were made 0644); others' files are left alone
    for path in [out_dir, out_dir / "rows", *out_dir.glob("*"), *(out_dir / "rows").rglob("*")]:
        _share(path)
    return written


def _plot(summary, first_row, col: str, path: Path, temporary: bool) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    x, m, s = summary.layer.to_numpy(), summary[f"{col}_mean"].to_numpy(), summary[f"{col}_std"].fillna(0).to_numpy()
    ax.plot(x, m, marker="o", ms=3, lw=1.5, color="#2563eb")
    ax.fill_between(x, m - s, m + s, alpha=0.2, color="#2563eb", lw=0)
    ax.set_xlabel("layer index (0 = embeddings, 32 = final post-norm, 33 = raw block 32 if present)")
    ax.set_ylabel(f"seen_val {col.removeprefix(METRIC_PREFIX)} (mean ± std over seeds)")
    ax.set_xticks(x)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(alpha=0.3)
    title = (f"Depth sweep: pool={first_row.pool}, {'train-norm' if first_row.normalize else 'raw'}, "
             f"probe={first_row.probe}, eval={first_row.evaluator}")
    if temporary:
        title = "TEMPORARY probe/metric: NOT A RESULT\n" + title
    ax.set_title(title, fontsize=9, color="#b91c1c" if temporary else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
