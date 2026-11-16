"""Rhea's S1-S6 from a depth-sweep folder (rows/L??_seed?.scores.npz), with no extra training.

    python scripts/select_layers.py --sweep <depth_sweep run folder>   [--margin 0.03] [--min-gain 0.005]

Everything is computed on seen_val episode scores saved by the sweep (B4 evaluator: per-episode max score up to
min(length, task_min_step)); unseen data is never read. Selection and reporting both use seen_val, so the fused
numbers are optimistic until the locked final test.

Per layer: ROC-AUC etc. per seed (B4), mean +- std over seeds.
Candidates for selection: layers 1-31 and 33 (raw block outputs). 0 (embeddings) and 32 (= block 32 after the final
norm, a rescaled copy of 33) are excluded by default; change with --candidates.
Score-level fusion ("late fusion", used for S4's greedy search and as a stand-in number for S1-S4/S6 until the trained
fusion runs exist): within each seed, z-score every layer's episode scores over the seen_val episodes, average the
chosen layers, score with B4; report mean +- std over seeds.

S1  top-2 layers by mean ROC-AUC
S2  best layer + the competitive layer (ROC-AUC >= best - margin) whose errors overlap least with the best layer's
    (errors at each layer's own Youden-optimal threshold; overlap = Jaccard of the two error sets)
S3  best layer in each depth band: early 1-11, middle 12-22, late 23-31 + 33
S4  greedy forward selection on score-level fusion, starting from the best layer; stop when the gain < min-gain or 4
S5  the sweep's all-layer average pseudo-layer ("avg") as trained by the sweep
S6  uniform (equal-weight) score-level average of the S1 set and of the S3 set
Also: top-4 ranking (for Dhyan's F6) and the S2 pair / S3 set (for F1 / F2-F3).
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

try:
    from failure_prob.engine.episode_metrics import episode_metrics
except ImportError:  # run outside the repo: identical definition (see failure_prob/engine/episode_metrics.py)
    from sklearn.metrics import auc, precision_recall_curve

    def episode_metrics(step_scores, labels, horizons, task_ids=None):
        y = np.asarray(labels).astype(int)
        s = np.array([np.max(np.asarray(x)[:h]) for x, h in zip(step_scores, horizons)])
        if len(np.unique(y)) < 2:
            return {"roc_auc": np.nan, "pr_auc": np.nan, "tpr_at_5_fpr": np.nan}
        fpr, tpr, _ = roc_curve(y, s)
        p, r, _ = precision_recall_curve(y, s)
        return {"roc_auc": auc(fpr, tpr), "pr_auc": auc(r, p), "tpr_at_5_fpr": tpr[fpr <= 0.05 + 1e-12].max()}

ROW_RE = re.compile(r"^L(\d\d|avg)_seed(\d+)\.scores\.npz$")
BANDS = {"early": range(1, 12), "middle": range(12, 23), "late": list(range(23, 32)) + [33]}
KEYS = ("roc_auc", "pr_auc", "tpr_at_5_fpr")


def load_sweep(sweep: Path):
    """{layer: {seed: episode_scores}}, labels, task_ids, trajectory_ids (common episode order)."""
    rows = {}
    ref_ids = labels = tasks = None
    for f in sorted((sweep / "rows").glob("L*_seed*.scores.npz")):
        m = ROW_RE.match(f.name)
        if not m:
            continue
        layer = "avg" if m.group(1) == "avg" else int(m.group(1))
        seed = int(m.group(2))
        z = np.load(f)
        off, flat, hor = z["offsets"], z["step_scores"], z["horizons"]
        ep = np.array([flat[off[i]:off[i] + hor[i]].max() for i in range(len(hor))], dtype=np.float64)
        ids = [str(t) for t in z["trajectory_ids"]]
        if ref_ids is None:
            ref_ids, labels, tasks = ids, z["labels"].astype(int), z["task_ids"]
        elif ids != ref_ids:
            order = {t: i for i, t in enumerate(ids)}
            if set(order) != set(ref_ids):
                raise SystemExit(f"{f.name}: different episodes from the other rows")
            ep = ep[[order[t] for t in ref_ids]]
        rows.setdefault(layer, {})[seed] = ep
    if not rows:
        raise SystemExit(f"no rows/L??_seed?.scores.npz files in {sweep}")
    return rows, labels, tasks, ref_ids


def metrics_of(ep_scores, labels, tasks):
    return episode_metrics([np.array([v]) for v in ep_scores], labels, np.ones(len(ep_scores), int), tasks)


def summarize(per_seed: list[dict]) -> dict:
    return {k: (float(np.nanmean([m[k] for m in per_seed])), float(np.nanstd([m[k] for m in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0)
            for k in KEYS}


def zscore(v):
    sd = v.std()
    return (v - v.mean()) / (sd if sd > 0 else 1.0)


def fused(rows, layers, labels, tasks, seeds):
    """Score-level fusion: per seed, mean of z-scored episode scores of `layers`."""
    per_seed = [metrics_of(np.mean([zscore(rows[l][s]) for l in layers], axis=0), labels, tasks) for s in seeds]
    return summarize(per_seed)


def youden_errors(score, labels):
    fpr, tpr, thr = roc_curve(labels, score)
    t = thr[np.argmax(tpr - fpr)]
    return set(np.flatnonzero((score >= t).astype(int) != labels))


def fmt(m, k="roc_auc"):
    return f"{100 * m[k][0]:.1f} ± {100 * m[k][1]:.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", required=True, type=Path)
    ap.add_argument("--candidates", default="1-31,33", help="layers allowed in selections, e.g. '1-31,33'")
    ap.add_argument("--margin", type=float, default=0.03, help="S2: partner must be within this ROC-AUC of the best")
    ap.add_argument("--min-gain", type=float, default=0.005, help="S4: stop when adding a layer gains less")
    ap.add_argument("--out", type=Path, default=None, help="CSV for the sheet (default: <sweep>/rhea_S1_S6.csv)")
    args = ap.parse_args()

    rows, labels, tasks, ids = load_sweep(args.sweep)
    seed_sets = [set(v) for v in rows.values()]
    seeds = sorted(set.intersection(*seed_sets))
    cand = []
    for part in args.candidates.split(","):
        a, _, b = part.partition("-")
        cand += list(range(int(a), int(b) + 1)) if b else [int(a)]
    cand = [l for l in cand if l in rows]
    missing = sorted(set(range(34)) - set(k for k in rows if k != "avg"))
    print(f"sweep: {args.sweep}\nepisodes: {len(labels)} seen_val ({labels.sum()} failures), seeds used: {seeds}")
    if missing:
        print(f"note: layers missing from the sweep: {missing}")

    per_layer = {l: summarize([metrics_of(rows[l][s], labels, tasks) for s in seeds]) for l in rows}
    ranked = sorted(cand, key=lambda l: -per_layer[l]["roc_auc"][0])
    best = ranked[0]
    avg_score = {l: np.mean([zscore(rows[l][s]) for s in seeds], axis=0) for l in cand}

    print("\nPer-layer seen_val ROC-AUC (mean ± std over seeds, %), candidates ranked:")
    for i, l in enumerate(ranked):
        print(f"  {i + 1:>2}. L{l:<3} {fmt(per_layer[l])}")
    if 32 in per_layer:
        print(f"  (L32, post-norm copy of L33, not a candidate: {fmt(per_layer[32])})")
    if 0 in per_layer and 0 not in cand:
        print(f"  (L0, embeddings, not a candidate: {fmt(per_layer[0])})")

    # S1
    s1 = ranked[:2]
    # S2
    err_best = youden_errors(avg_score[best], labels)
    competitive = [l for l in ranked[1:] if per_layer[l]["roc_auc"][0] >= per_layer[best]["roc_auc"][0] - args.margin]
    if not competitive:
        competitive = ranked[1:2]
    overlap = {}
    for l in competitive:
        e = youden_errors(avg_score[l], labels)
        overlap[l] = len(err_best & e) / max(1, len(err_best | e))
    partner = min(competitive, key=lambda l: (overlap[l], -per_layer[l]["roc_auc"][0]))
    s2 = [best, partner]
    print(f"\nS2: error overlap with L{best} (Jaccard, lower = more complementary), layers within {args.margin} "
          f"ROC-AUC of L{best}:")
    for l in sorted(overlap, key=overlap.get)[:8]:
        print(f"  L{l:<3} overlap {overlap[l]:.2f}   ROC-AUC {fmt(per_layer[l])}" + ("   <- picked" if l == partner else ""))
    # S3
    s3 = []
    for band, rng in BANDS.items():
        inband = [l for l in ranked if l in rng]
        if inband:
            s3.append(inband[0])
    # S4 greedy
    s4, trace = [best], [(best, fused(rows, [best], labels, tasks, seeds)["roc_auc"][0])]
    while len(s4) < 4:
        cur = trace[-1][1]
        gains = {l: fused(rows, s4 + [l], labels, tasks, seeds)["roc_auc"][0] for l in cand if l not in s4}
        nxt = max(gains, key=gains.get)
        if gains[nxt] - cur < args.min_gain:
            break
        s4.append(nxt)
        trace.append((nxt, gains[nxt]))

    results = []

    def add(row, task, name, layers, m, note):
        results.append({"S.No": row, "Task ID": task, "Experiment": name,
                        "Layers": " + ".join(f"L{l}" for l in layers),
                        "Seen Val ROC-AUC (%)": fmt(m, "roc_auc"), "Seen Val PR-AUC (%)": fmt(m, "pr_auc"),
                        "Seen Val TPR @ 5% FPR (%)": fmt(m, "tpr_at_5_fpr"), "Notes": note})

    late = "score-level fusion of the sweep's per-layer probes (z-scored, averaged); trained fusion pending (Dhyan F1-F3)"
    add(15, "S1", "Top-2 layers by validation AUC", s1, fused(rows, s1, labels, tasks, seeds), late)
    add(16, "S2", "Complementary top-2 layers", s2, fused(rows, s2, labels, tasks, seeds),
        f"partner = lowest Youden-error overlap (Jaccard {overlap[partner]:.2f}) among layers within {args.margin} "
        f"ROC-AUC of L{best}; {late}")
    add(17, "S3", "Depth-spaced 3-layer set", s3, fused(rows, s3, labels, tasks, seeds),
        f"best layer per band (early 1-11, middle 12-22, late 23-31+33); {late}")
    add(18, "S4", "Greedy forward selection", s4, fused(rows, s4, labels, tasks, seeds),
        "greedy on score-level fusion, gains " + " -> ".join(f"L{l}:{100 * v:.1f}" for l, v in trace)
        + f"; stopped (gain < {args.min_gain} or K=4)")
    if "avg" in per_layer:
        add(38, "S5", "All-layer average baseline", [], per_layer["avg"],
            "MLP probe trained by the depth sweep on mean of z-scored L1-31 + L33 (pseudo-layer 'avg')")
        results[-1]["Layers"] = "avg_z(L1-31, L33)"
    add(39, "S6", "Uniform average of selected layers", s1, fused(rows, s1, labels, tasks, seeds),
        f"equal-weight score average of the S1 set; for the S3 set ({' + '.join(f'L{l}' for l in s3)}): "
        f"{fmt(fused(rows, s3, labels, tasks, seeds))}")
    results.append({"S.No": "ref", "Task ID": "", "Experiment": f"Best single layer (L{best})",
                    "Layers": f"L{best}", "Seen Val ROC-AUC (%)": fmt(per_layer[best], "roc_auc"),
                    "Seen Val PR-AUC (%)": fmt(per_layer[best], "pr_auc"),
                    "Seen Val TPR @ 5% FPR (%)": fmt(per_layer[best], "tpr_at_5_fpr"),
                    "Notes": "for comparison: the selections should beat this"})

    out = args.out or args.sweep / "rhea_S1_S6.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)

    print("\n=== S1-S6 (seen_val, %, mean ± std over seeds) ===")
    for r in results:
        print(f"{r['Task ID'] or r['S.No']:>4} {r['Layers']:<22} ROC {r['Seen Val ROC-AUC (%)']:<12} "
              f"PR {r['Seen Val PR-AUC (%)']:<12} TPR@5 {r['Seen Val TPR @ 5% FPR (%)']:<12}")
    print(f"\nwritten: {out}")
    top4 = ranked[:4]
    print("\n=== Message for Dhyan ===")
    print(f"S2 pair (F1): L{s2[0]} + L{s2[1]}\nS3 set (F2, F3): early L{s3[0]}, middle L{s3[1]}, late L{s3[-1]}"
          if len(s3) == 3 else f"S2 pair (F1): L{s2[0]} + L{s2[1]}\nS3 set: {s3}")
    print("Top-4 ranked (F6), best first: " + ", ".join(f"L{l}" for l in top4))
    print(f"Best single layer: L{best} (seen_val ROC-AUC {fmt(per_layer[best])}%)")
    print(f"S4 greedy set: {' + '.join(f'L{l}' for l in s4)}")
    print(f"Pooling: mean, train-only z-score. All picked on seen_val from the depth sweep ({args.sweep.name}).")


if __name__ == "__main__":
    main()
