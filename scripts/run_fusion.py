"""Fusion runner CLI (one configuration; F1-F3 searches and F6 are in fusion_experiments.py): train a fusion detector on train, score seen_val per episode, for seeds 0/1/2.

Real cache (CARC; CACHE = the shared cache v1/ folder from the team handoff):
    python scripts/run_fusion.py --model static_weight_fusion --layers 16 33 \
        --cache $CACHE --output-dir outputs/fusion/swf_L16_L33
    python scripts/run_fusion.py --model concat_fusion --layers 16 33 --head linear \
        --cache $CACHE --output-dir outputs/fusion/concat_L16_L33

CPU smoke test (random features, numbers meaningless):
    python scripts/run_fusion.py --model static_weight_fusion --layers 8 16 --synthetic --epochs 2 \
        --output-dir /tmp/fusion_smoke

unseen_test is never loaded. Use --horizon task_min_step (default) for anything you report.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failure_prob.fusion_runner import AGGREGATIONS, B4_EVALUATOR, HORIZONS, RunConfig, run  # noqa: E402
from failure_prob.models import list_models  # noqa: E402
from failure_prob.models.probe_heads import HEADS  # noqa: E402
from failure_prob.protocol import TRAIN_SEEDS  # noqa: E402


def common_parser() -> argparse.ArgumentParser:
    """Every RunConfig flag; shared with scripts/fusion_experiments.py."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model", required=True, choices=list_models())
    p.add_argument("--layers", required=True, type=int, nargs="+", help="cache layer indices (0..33; use 33, not 32, "
                   "for raw block 32 in depth comparisons)")
    p.add_argument("--output-dir", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cache", help="SharedCache root (the cache v1/ folder on CARC or a local copy)")
    src.add_argument("--synthetic", action="store_true", help="random CPU fixture; for smoke tests only")
    p.add_argument("--pool", default="mean", choices=["mean", "first", "last"])
    p.add_argument("--no-normalize", dest="normalize", action="store_false",
                   help="use raw features (sensitivity check only; default z-scores with train-only stats)")
    p.add_argument("--horizon", default="task_min_step", choices=HORIZONS, help="seen_val scoring horizon")
    p.add_argument("--train-horizon", default="full", choices=HORIZONS,
                   help="train episodes (default full, as the depth sweep)")
    p.add_argument("--seeds", type=int, nargs="+", default=list(TRAIN_SEEDS))
    p.add_argument("--head", default="mlp", choices=HEADS)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=150, help="at most (early stopping)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--select-best-epoch", action="store_true",
                   help="keep the epoch with the best seen_val ROC-AUC (needs --early-stopping-patience 0)")
    p.add_argument("--early-stopping-patience", type=int, default=10,
                   help="B6: stop after this many epochs without a lower TRAIN-holdout loss (0 = off)")
    p.add_argument("--es-holdout-fraction", type=float, default=0.15, help="B6: share of train held out")
    p.add_argument("--es-min-delta", type=float, default=0.0)
    p.add_argument("--aggregation", default="max", choices=AGGREGATIONS, help="episode score from step logits")
    p.add_argument("--evaluator", default=B4_EVALUATOR,
                   help="episode scorer: default B4 (the team's SAFE-equivalent evaluator); 'builtin' for the "
                   "fusion_runner scorer, or a depth-sweep registered name / 'package.module:factory'")
    p.add_argument("--allow-duplicate-layers", action="store_true",
                   help="allow a layer more than once, e.g. --layers 33 33 33 (duplicate-strongest-layer control)")
    p.add_argument("--fixed-weights", dest="learn_weights", action="store_false",
                   help="static_weight_fusion: freeze the mix at --init-weights (uniform if omitted)")
    p.add_argument("--init-weights", type=float, nargs="+", help="static_weight_fusion: one value per layer")
    p.add_argument("--proj-dim", type=int, default=None, help="static_weight_fusion: project every layer to this "
                   "shared dimension before the weighted sum (F3 uses 64); omit for a pure scalar mix")
    p.add_argument("--std-floor", type=float, default=None, help="normalize with max(train std, this) instead of "
                   "the loader's 1e-6 floor (e.g. 1e-2 if the run warns that normalized features blow up)")
    p.add_argument("--bootstrap", type=int, default=1000, help="seen_val bootstrap resamples for CIs (0 = off)")
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--latency-device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--no-memory-check", dest="memory_check", action="store_false",
                   help="load even if the memory estimate exceeds the job's limit")
    p.add_argument("--resume", action="store_true", help="keep finished seeds of an identical earlier run "
                   "(e.g. after a Slurm time-out) and train only the missing ones")
    p.add_argument("--overwrite", action="store_true", help="replace a non-empty --output-dir")
    p.add_argument("--no-checkpoints", dest="save_checkpoints", action="store_false")
    return p


def to_config(ns: argparse.Namespace) -> RunConfig:
    fields = set(RunConfig.__dataclass_fields__)
    return RunConfig(**{k: v for k, v in vars(ns).items() if k in fields})


def parse(argv=None) -> RunConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                parents=[common_parser()])
    return to_config(p.parse_args(argv))


def main(argv=None) -> int:
    cfg = parse(argv)
    try:
        run(cfg)
    except (ValueError, FileExistsError, FileNotFoundError, PermissionError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
