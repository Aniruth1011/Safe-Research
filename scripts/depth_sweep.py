#!/usr/bin/env python
"""Depth sweep (plan rows R1-R4): one probe per cache layer x training seed, trained on train, scored on seen_val.

    # one or more layers (a CARC array task runs one layer: --layers $SLURM_ARRAY_TASK_ID)
    python scripts/depth_sweep.py run --layers 0 1 [--pool mean|first|last] [--normalize] \
        [--cache /scratch1/<user>/safe_cache/v1] [--out <folder>] [--probe temporary_mlp] [--evaluator temporary_max]
    # after all layers: results.csv, summary.csv, depth_curve.png
    python scripts/depth_sweep.py aggregate --out <folder> [--metric roc_auc]

R1 = all layers, R3 = --pool first/last at the best layer, R4 = --normalize (train-split statistics) vs raw.
unseen_test is never read. The default probe and evaluator are TEMPORARY placeholders (see
failure_prob/engine/sweep_plugins.py); swap them with --probe/--evaluator 'package.module:factory'.
"""
from __future__ import annotations

# Only the standard library before the first log line, so a stall while importing shows up in the log.
import argparse
import faulthandler
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

T_START = time.time()


def log(msg: str) -> None:
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {msg}", flush=True)


def _on_sigterm(signum, frame):
    """scancel / time limit: show where the process was, then stop before any further row is written."""
    log(f"received signal {signum} (cancelled or out of time); stack follows; no further rows will be written")
    faulthandler.dump_traceback(all_threads=True)
    raise SystemExit(128 + signum)


EXIT_STALLED = 75  # EX_TEMPFAIL: start-up or loading did not finish in time, or a read stalled; rerun the layer


class StartupWatchdog:
    """Fail fast instead of waiting for the job's time limit when start-up or loading a layer stalls.

    arm(seconds, what) starts the clock; disarm() stops it. If the clock runs out, the process logs a clear message
    and every thread's Python stack, then exits with EXIT_STALLED. That works while the main thread waits for a file
    read in uninterruptible I/O (state D), which a signal cannot interrupt. A C-level backstop
    (faulthandler.dump_traceback_later) fires 15 s later in case the Python-level thread cannot run.
    """

    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self._gen = 0

    def arm(self, seconds: float, what: str) -> None:
        import threading
        if seconds <= 0:
            return
        with self._lock:
            self._gen += 1
            gen = self._gen
        try:
            faulthandler.dump_traceback_later(seconds + 15, exit=True)
        except (ValueError, OSError, AttributeError):  # stderr has no file descriptor (e.g. under pytest)
            pass

        def fire():
            with self._lock:
                if gen != self._gen:
                    return
            log(f"WATCHDOG: {what} did not finish within {seconds:.0f} s (SWEEP_LOAD_TIMEOUT). The file system is "
                f"probably stalled. Exiting with code {EXIT_STALLED} so the task can be rerun, maybe on another node. "
                "Python stack of every thread:")
            sys.stdout.flush()
            faulthandler.dump_traceback(file=sys.stdout, all_threads=True)
            sys.stdout.flush()
            os._exit(EXIT_STALLED)

        t = threading.Timer(seconds, fire)
        t.daemon = True
        t.start()

    def disarm(self) -> None:
        with self._lock:
            self._gen += 1
        faulthandler.cancel_dump_traceback_later()


WATCHDOG = StartupWatchdog()
LOAD_TIMEOUT = float(os.environ.get("SWEEP_LOAD_TIMEOUT", "120") or 0)

if __name__ == "__main__":                                     # not when a test imports this file
    faulthandler.enable()                                      # tracebacks on hard crashes
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)  # `scancel --signal=USR1 <job>`: print the stack
    signal.signal(signal.SIGTERM, _on_sigterm)
    WATCHDOG.arm(LOAD_TIMEOUT, "start-up (imports, opening the cache) and loading the first layer")

log(f"start: pid {os.getpid()}, python {sys.version.split()[0]} ({sys.executable}), cwd {os.getcwd()}, "
    f"log {os.environ.get('SWEEP_LOG') or 'stdout'}, PYTHONDONTWRITEBYTECODE={os.environ.get('PYTHONDONTWRITEBYTECODE', '')}")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout without `pip install -e`
log("importing failure_prob (numpy, the cache loader)")
from failure_prob.engine.depth_sweep import ReadStalled, SweepConfig, aggregate, missing_rows, parse_layers, run_layer  # noqa: E402
from failure_prob.protocol import TRAIN_SEEDS  # noqa: E402
log(f"imports done ({time.time() - T_START:.1f} s since start)")

USER = os.environ.get("USER", "user")
DEFAULT_CACHE = os.environ.get("CACHE_DIR", f"/scratch1/{USER}/safe_cache/v1")
DEFAULT_OUT_ROOT = os.environ.get("SWEEP_OUT_ROOT", f"/scratch1/{USER}/depth_sweep")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="train and score probes for some layers")
    r.add_argument("--layers", type=int, nargs="+", required=True)
    r.add_argument("--cache", default=DEFAULT_CACHE, help="cache version folder (default: $CACHE_DIR or %(default)s)")
    r.add_argument("--out", default=None, help=f"output folder (default: {DEFAULT_OUT_ROOT}/<configuration label>)")
    r.add_argument("--pool", default="mean", choices=["mean", "first", "last"])
    r.add_argument("--normalize", action="store_true", help="z-score with the cache's TRAIN-split statistics")
    r.add_argument("--train-horizon", default="full", choices=["full", "task_min_step"])
    r.add_argument("--probe", default="temporary_mlp", help="registered name or 'package.module:factory'")
    r.add_argument("--evaluator", default="temporary_max", help="registered name or 'package.module:factory'")
    r.add_argument("--seeds", type=int, nargs="+", default=list(TRAIN_SEEDS))
    r.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or None)
    r.add_argument("--loader", type=Path, default=None, help="path to collection/load_cache.py (default: this repo's)")
    r.add_argument("--overwrite", action="store_true", help="redo (layer, seed) rows that already exist")
    a = sub.add_parser("aggregate", help="collect rows into tables and the depth-curve plot")
    a.add_argument("--out", default=None, help="the run's folder (default: derived from the flags below, as for run)")
    # the same configuration flags as `run`, used only to find the default folder
    a.add_argument("--pool", default="mean", choices=["mean", "first", "last"])
    a.add_argument("--normalize", action="store_true")
    a.add_argument("--train-horizon", default="full", choices=["full", "task_min_step"])
    a.add_argument("--probe", default="temporary_mlp")
    a.add_argument("--evaluator", default="temporary_max")
    a.add_argument("--metric", default="roc_auc", help="metric drawn in the plot (any key the evaluator returns)")
    a.add_argument("--no-plot", action="store_true")
    a.add_argument("--expect-layers", default=None,
                   help="e.g. 0-33: list missing (layer, seed) rows, still write the tables, and exit 4 if any")
    args = ap.parse_args(argv)

    def config(cache=DEFAULT_CACHE, seeds=TRAIN_SEEDS):
        return SweepConfig(cache_dir=cache, pool=args.pool, normalize=args.normalize, train_horizon=args.train_horizon,
                           probe=args.probe, evaluator=args.evaluator, seeds=tuple(seeds))

    if args.cmd == "run":
        log("importing torch")
        t = time.time()
        import torch
        log(f"torch {torch.__version__} imported ({time.time() - t:.1f} s)")
        if args.threads:
            torch.set_num_threads(args.threads)
        cfg = config(args.cache, args.seeds)
        out = Path(args.out) if args.out else Path(DEFAULT_OUT_ROOT) / cfg.label()
        log(f"configuration {cfg.label()} -> {out}")
        kw = {"loader": args.loader} if args.loader else {}
        for i, layer in enumerate(args.layers):
            if i:
                WATCHDOG.arm(LOAD_TIMEOUT, f"loading layer {layer}")
            try:
                run_layer(cfg, layer, out, overwrite=args.overwrite, log=log, on_loaded=WATCHDOG.disarm, **kw)
            except ReadStalled as e:
                log(f"READ STALLED: {e}. Exiting with code {EXIT_STALLED}; rerun this layer, maybe on another node.")
                sys.stdout.flush()
                os._exit(EXIT_STALLED)  # a reader thread may be stuck in the kernel; do not wait for it
        WATCHDOG.disarm()
        log(f"done ({time.time() - T_START:.1f} s since start)")
    else:
        WATCHDOG.disarm()
        plot = not args.no_plot
        if plot:
            try:
                import matplotlib  # noqa: F401
            except ImportError:
                plot = False
                print("matplotlib is not installed here: writing the tables only. Draw the plot elsewhere with "
                      "`python scripts/depth_sweep.py aggregate --out <copy of this folder>`.", file=sys.stderr)
        out = Path(args.out) if args.out else Path(DEFAULT_OUT_ROOT) / config().label()
        log(f"aggregating {out}")
        for name, path in aggregate(out, metric=args.metric, plot=plot, log=log).items():
            log(f"{name}: {path}")
        if args.expect_layers:
            missing = missing_rows(out, parse_layers(args.expect_layers))
            if missing:
                for layer, seeds in missing.items():
                    log(f"MISSING: layer {layer} seeds {seeds}")
                log(f"{len(missing)} layer(s) incomplete; the tables above are partial. Rerun them with "
                    f"LAYERS={','.join(map(str, missing))}")
                sys.exit(4)
            log(f"all expected layers present ({args.expect_layers})")
        if not args.no_plot and not plot:
            sys.exit(3)


if __name__ == "__main__":
    main()
