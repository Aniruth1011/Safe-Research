"""Paired comparison of two detectors on seen_val: "does fusion beat the best single layer?" with a CI.

Each side is a set of per-seed score files on the SAME seen_val episodes:
  * a fusion_runner output folder (its seed<s>/seen_val_step_scores.npz), or
  * a glob of depth-sweep score files, e.g. "<sweep out>/rows/L16_seed*.scores.npz" (same layout).
Both sides are scored by the same evaluator (built-in max-of-step-scores, or a plugged-in one). Every bootstrap
resample draws ONE set of episodes (stratified by outcome) and scores both sides on it, each averaged over its
seeds; the CI is the percentile interval of the difference A - B. `p_a_not_better` is the share of resamples with
A - B <= 0 (one-sided, descriptive). seen_val only: unseen_test is never involved.
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np

from failure_prob import fusion_runner as FR


def load_side(spec: str) -> tuple[list, str]:
    p = Path(spec)
    if p.is_dir():
        summary = json.loads((p / "summary.json").read_text())
        files = [p / f"seed{s}" / FR.STEP_SCORES for s in summary["seeds"]]
        desc = f"fusion run {p.name} ({summary['model']} layers {summary['layers']})"
    else:
        files = sorted(Path(f) for f in glob.glob(spec))
        desc = f"{len(files)} score files matching {spec}"
    missing = [str(f) for f in files if not f.is_file()]
    if not files or missing:
        raise ValueError(f"no score files for {spec!r}" + (f" (missing {missing[:3]})" if missing else ""))
    return [FR.load_step_scores(f) for f in files], desc


def compare(a: list, b: list, scorer: FR.Scorer, n: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    a = FR.align(a)
    ref = a[0].trajectory_ids
    try:
        b = FR.align(b, ref)
    except ValueError as e:
        raise ValueError(f"the two sides were scored on different seen_val episodes: {e}") from None
    if not np.array_equal(a[0].labels, b[0].labels):
        raise ValueError("the two sides disagree on episode labels")
    pa, pb = [scorer.prepare(d) for d in a], [scorer.prepare(d) for d in b]
    full = np.arange(len(ref))

    def side(inputs, prepared, idx):
        per = [scorer(d, p, idx) for d, p in zip(inputs, prepared)]
        return {k: float(np.nanmean([m[k] for m in per])) for k in per[0]}
    point_a, point_b = side(a, pa, full), side(b, pb, full)
    deltas: dict[str, list[float]] = {}
    for idx in FR.stratified_indices(a[0].labels, n, seed):
        ma, mb = side(a, pa, idx), side(b, pb, idx)
        for k in ma:
            if k in mb:
                deltas.setdefault(k, []).append(ma[k] - mb[k])
    lo, hi = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    out = {}
    for k, v in deltas.items():
        v = np.asarray(v)
        v = v[np.isfinite(v)]
        out[k] = {"a": point_a[k], "b": point_b[k], "delta": point_a[k] - point_b[k],
                  "ci": [float(np.percentile(v, lo)), float(np.percentile(v, hi))] if len(v) else [math.nan] * 2,
                  "p_a_not_better": float((v <= 0).mean()) if len(v) else math.nan}
    return {"metrics": out, "n_episodes": len(ref), "n_failures": int(np.sum(a[0].labels)),
            "seeds": {"a": len(a), "b": len(b)}, "n_resamples": n, "ci": ci, "seed": seed,
            "scheme": "paired stratified bootstrap over seen_val episodes; each side averaged over its seeds"}
