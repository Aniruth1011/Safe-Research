"""Dhyan's fusion rows: F1 (row 19), F2 (row 20), F3 (row 21) with the compact hyper-parameter search, and F6 (row 25).

Layer sets come from selection (S1-S4) on seen_val; until then any layers run (numbers are development only).
CACHE = the shared cache v1/ folder (team handoff). Every flag of scripts/run_fusion.py is accepted.

  # F1: two-layer concatenation on the selected complementary pair (S2)
  python scripts/fusion_experiments.py search --model concat_fusion --layers A B --cache $CACHE --output-dir outputs/fusion/F1
  # F2: three-layer concatenation on the early/middle/late set (S3)
  python scripts/fusion_experiments.py search --model concat_fusion --layers A B C --cache $CACHE --output-dir outputs/fusion/F2
  # F3: static learned weights, layers projected to a shared 64-d space
  python scripts/fusion_experiments.py search --model static_weight_fusion --proj-dim 64 --layers A B C --cache $CACHE \
      --output-dir outputs/fusion/F3
  # F6: K = 1..4 on the ranked selection (best first), best fusion architecture and its searched hyper-parameters
  python scripts/fusion_experiments.py kcurve --model concat_fusion --layers A B C D --lr 3e-4 --dropout 0.2 \
      --cache $CACHE --output-dir outputs/fusion/F6

search: grid over --lrs x --dropouts x --weight-decays (default 2x2x2), all seeds each, winner = best mean seen_val
ROC-AUC. Outputs: search.csv (every grid point), best.json, sheet_rows.csv (percent, parameters, latency).
kcurve: one run per K under K<k>/, k_curve.csv and sheet_rows.csv. unseen_test is never loaded.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from failure_prob import fusion_experiments as FX  # noqa: E402
from run_fusion import common_parser, to_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search", parents=[common_parser()], help="F1/F2/F3 with the compact hyper-parameter search")
    s.add_argument("--lrs", type=float, nargs="+", default=FX.DEFAULT_GRID["lr"])
    s.add_argument("--dropouts", type=float, nargs="+", default=FX.DEFAULT_GRID["dropout"])
    s.add_argument("--weight-decays", type=float, nargs="+", default=FX.DEFAULT_GRID["weight_decay"])
    s.add_argument("--label", default=None, help="Experiment name in the sheet row (e.g. 'F1 two-layer concat')")
    k = sub.add_parser("kcurve", parents=[common_parser()], help="F6: K = 1..4 on the ranked layer list")
    k.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4])
    k.add_argument("--reference-layer", type=int, default=None,
                   help="R1's best single layer; also run as K=1 when it is not the first ranked layer")
    k.add_argument("--label", default=None)
    a = ap.parse_args(argv)
    cfg = to_config(a)
    try:
        if a.cmd == "search":
            FX.hparam_search(cfg, {"lr": a.lrs, "dropout": a.dropouts, "weight_decay": a.weight_decays}, label=a.label)
        else:
            FX.k_ablation(cfg, a.ks, label=a.label, reference_layer=a.reference_layer)
    except (ValueError, FileExistsError, FileNotFoundError, PermissionError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
