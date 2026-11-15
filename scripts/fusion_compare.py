"""Paired seen_val comparison with a bootstrap CI of the difference (A - B), e.g. F1 vs the best single layer.

    python scripts/fusion_compare.py --a outputs/fusion/F1/hp_<winner> \
        --b "<depth sweep out>/rows/L16_seed*.scores.npz" --label-a "F1 concat 16+24" --label-b "best single L16"
    python scripts/fusion_compare.py --a outputs/fusion/F3/hp_<winner> --b outputs/fusion/F2/hp_<winner>

A / B: a fusion run folder, or a quoted glob of depth-sweep score files. Both sides are scored by the same evaluator
(default: built-in, episode score = max step score; or --evaluator as for run_fusion.py).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failure_prob import fusion_compare as FC  # noqa: E402
from failure_prob import fusion_runner as FR  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--evaluator", default=None)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the result as JSON here")
    a = ap.parse_args(argv)
    try:
        if a.n < 1 or not 0 < a.ci < 1:
            raise ValueError("need --n >= 1 and 0 < --ci < 1")
        side_a, desc_a = FC.load_side(a.a)
        side_b, desc_b = FC.load_side(a.b)
        res = FC.compare(side_a, side_b, FR.Scorer(FR.resolve_evaluator(a.evaluator)), a.n, a.ci, a.seed)
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    res.update({"a": {"label": a.label_a, "source": desc_a}, "b": {"label": a.label_b, "source": desc_b}})
    pct = int(round(100 * a.ci))
    print(f"{a.label_a} [{desc_a}]  vs  {a.label_b} [{desc_b}]")
    print(f"seen_val: {res['n_episodes']} episodes ({res['n_failures']} failures), {a.n} paired resamples")
    for k, m in res["metrics"].items():
        print(f"  {k:14s} {100 * m['a']:6.2f}% vs {100 * m['b']:6.2f}%  delta {100 * m['delta']:+6.2f} pp  "
              f"{pct}% CI [{100 * m['ci'][0]:+.2f}, {100 * m['ci'][1]:+.2f}]  P(A not better) {m['p_a_not_better']:.3f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
