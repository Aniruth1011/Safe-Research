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

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout without `pip install -e`

from failure_prob.engine.depth_sweep import SweepConfig, aggregate, run_layer  # noqa: E402
from failure_prob.protocol import TRAIN_SEEDS  # noqa: E402

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
    args = ap.parse_args(argv)

    def config(cache=DEFAULT_CACHE, seeds=TRAIN_SEEDS):
        return SweepConfig(cache_dir=cache, pool=args.pool, normalize=args.normalize, train_horizon=args.train_horizon,
                           probe=args.probe, evaluator=args.evaluator, seeds=tuple(seeds))

    if args.cmd == "run":
        if args.threads:
            import torch
            torch.set_num_threads(args.threads)
        cfg = config(args.cache, args.seeds)
        out = Path(args.out) if args.out else Path(DEFAULT_OUT_ROOT) / cfg.label()
        print(f"configuration {cfg.label()} -> {out}", flush=True)
        kw = {"loader": args.loader} if args.loader else {}
        for layer in args.layers:
            run_layer(cfg, layer, out, overwrite=args.overwrite, log=lambda m: print(m, flush=True), **kw)
    else:
        plot = not args.no_plot
        if plot:
            try:
                import matplotlib  # noqa: F401
            except ImportError:
                plot = False
                print("matplotlib is not installed here: writing the tables only. Draw the plot elsewhere with "
                      "`python scripts/depth_sweep.py aggregate --out <copy of this folder>`.", file=sys.stderr)
        out = Path(args.out) if args.out else Path(DEFAULT_OUT_ROOT) / config().label()
        for name, path in aggregate(out, metric=args.metric, plot=plot).items():
            print(f"{name}: {path}")
        if not args.no_plot and not plot:
            sys.exit(3)


if __name__ == "__main__":
    main()
