#!/usr/bin/env python3
"""R2: compare depth-sweep runs that differ only in probe settings, on the same layers.

    python3 scripts/r2_compare.py <sweep folder> [<sweep folder> ...] [--layers 27-32] [--csv out.csv]

Each folder is one scripts/depth_sweep.py run (config.json + summary.csv). For every probe setting it prints
seen_val ROC-AUC per layer (3-seed mean), the best layer, and where the reference layer (default L31) ranks.
Standard library only, so it runs on a CARC login node. It reads summary.csv; it never reads the cache.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

B6 = {"lr": 3e-4, "hidden_dim": 64, "dropout": 0.1}


def parse_layers(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def settings(folder):
    cfg = json.loads((folder / "config.json").read_text())
    s = cfg.get("probe_settings") or {}
    if not s and "_b6-" in folder.name:
        s = dict(B6)
    return {k: s.get(k) for k in ("lr", "hidden_dim", "dropout")}, cfg


def load(folder, layers):
    rows = {}
    with open(folder / "summary.csv", newline="") as f:
        for r in csv.DictReader(f):
            layer = int(r["layer"])
            if layer in layers:
                rows[layer] = r
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--layers", default="27-32")
    ap.add_argument("--ref", type=int, default=31, help="layer whose rank is reported (default 31)")
    ap.add_argument("--csv", type=Path, default=None, help="also write the long table here")
    a = ap.parse_args(argv)
    layers = parse_layers(a.layers)

    runs, problems = [], []
    for folder in a.folders:
        if not (folder / "summary.csv").exists():
            problems.append(f"{folder}: no summary.csv (not aggregated yet?)")
            continue
        s, cfg = settings(folder)
        rows = load(folder, layers)
        missing = [l for l in layers if l not in rows]
        if missing:
            problems.append(f"{folder.name}: missing layers {missing}")
        if any(r.get("TEMPORARY") == "True" for r in rows.values()) and s == B6:
            problems.append(f"{folder.name}: B6 settings but marked TEMPORARY")
        runs.append((s, cfg, folder, rows))

    # same everything except the probe settings
    keys = ("pool", "normalize", "train_horizon", "evaluator")
    base = {k: runs[0][1].get(k) for k in keys} if runs else {}
    for s, cfg, folder, _ in runs:
        diff = [k for k in keys if cfg.get(k) != base.get(k)]
        if diff:
            problems.append(f"{folder.name}: differs from {runs[0][2].name} in {diff}")

    runs.sort(key=lambda t: (float(t[0]["lr"]), int(t[0]["hidden_dim"]), float(t[0]["dropout"])))
    f = lambda r, k: 100 * float(r[k])
    head = "| lr | hidden | dropout | " + " | ".join(f"L{l}" for l in layers) + f" | best | L{a.ref} rank | best task-mean |"
    print(head)
    print("|" + "---|" * (head.count("|") - 1))
    long, best_counts, ref_ranks = [], {}, []
    for s, cfg, folder, rows in runs:
        roc = {l: f(rows[l], "val_roc_auc_mean") for l in layers if l in rows}
        tm = {l: f(rows[l], "val_roc_auc_taskmean_mean") for l in layers if l in rows}
        order = sorted(roc, key=roc.get, reverse=True)
        best = order[0]
        best_counts[best] = best_counts.get(best, 0) + 1
        rank = order.index(a.ref) + 1 if a.ref in order else None
        ref_ranks.append(rank)
        tag = " (B6)" if s == B6 else ""
        cells = [f"**{roc[l]:.1f}**" if l == best else f"{roc[l]:.1f}" if l in roc else "–" for l in layers]
        tbest = max(tm, key=tm.get)
        print(f"| {s['lr']:g}{tag} | {s['hidden_dim']} | {s['dropout']:g} | " + " | ".join(cells)
              + f" | L{best} | {rank} | L{tbest} ({tm[tbest]:.1f}) |")
        for l in layers:
            if l in rows:
                r = rows[l]
                long.append({"lr": s["lr"], "hidden_dim": s["hidden_dim"], "dropout": s["dropout"], "layer": l,
                             "roc_auc": round(roc[l], 2), "roc_ci_low": round(f(r, "val_roc_auc_ci95_low"), 2),
                             "roc_ci_high": round(f(r, "val_roc_auc_ci95_high"), 2),
                             "roc_taskmean": round(tm[l], 2), "pr_auc": round(f(r, "val_pr_auc_mean"), 2),
                             "folder": folder.name})
    print()
    print("best layer counts:", ", ".join(f"L{l}: {n}/{len(runs)}" for l, n in sorted(best_counts.items())))
    print(f"L{a.ref} rank per setting:", ref_ranks)
    if a.csv and long:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(long[0]))
            w.writeheader()
            w.writerows(long)
        print("wrote", a.csv)
    for p in problems:
        print("WARNING:", p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
