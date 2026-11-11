#!/usr/bin/env python
"""Integrity and reproducibility checks.

verify        Check a (possibly partial) downloaded copy of the cache against manifest.json checksums.
    python check_cache.py verify --cache /path/to/cache/v1            # full sha256
    python check_cache.py verify --cache /path/to/cache/v1 --quick    # sizes + shapes only

capture       Check the all-layer capture against SAFE's own last-layer .pkl from the same episodes.
    python check_cache.py capture --rollout-dir runA/libero_10

compare-runs  Compare two raw rollout folders of the same tasks (e.g. CARC A40 vs a teammate's run):
              outcome labels, lengths, first step where actions diverge, and feature differences.
    python check_cache.py compare-runs --a runA/libero_10 --b runB/libero_10
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NAME_RE = re.compile(r"^task(\d+)--ep(\d+)--succ([01])\.csv$")
ACTION_COLS = ["action/dx", "action/dy", "action/dz", "action/droll", "action/dpitch", "action/dyaw", "action/dgripper"]


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify(cache: Path, quick: bool) -> int:
    manifest = json.loads((cache / "manifest.json").read_text())
    files = manifest.get("files", {})
    if not files:
        print("manifest has no checksums (built with --skip-checksums)")
        return 1
    ok, bad, absent = 0, [], []
    for rel, info in files.items():
        p = cache / rel
        if not p.exists():
            absent.append(rel)
            continue
        if p.stat().st_size != info["bytes"] or (not quick and sha256(p) != info["sha256"]):
            bad.append(rel)
            continue
        if p.suffix == ".npy":
            np.load(p, mmap_mode="r")  # header/shape sanity
        ok += 1
    withheld = [r for r in absent if "/unseen_test/" in r.replace("\\", "/")]
    other_absent = [r for r in absent if r not in withheld]
    print(f"verified {ok} files; {len(bad)} corrupted; {len(other_absent)} not downloaded; "
          f"{len(withheld)} unseen-test files withheld")
    for r in bad:
        print(f"  CORRUPTED: {r}")
    if other_absent:
        print("  not downloaded (fine if you only need some layers):", ", ".join(other_absent[:8]),
              "..." if len(other_absent) > 8 else "")
    return 1 if bad else 0


def index(d: Path) -> dict[tuple[int, int], Path]:
    out = {}
    for csv in d.glob("task*--ep*--succ*.csv"):
        m = NAME_RE.match(csv.name)
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = csv
    return out


def report_invalid(d: Path, label: str = "") -> dict[tuple[int, int], str]:
    """Print the invalid-episode markers (task{T}--ep{E}--invalid.json, must-fix 4) of a rollout folder."""
    out = {}
    for p in sorted(d.glob("task*--ep*--invalid.json")):
        m = re.match(r"^task(\d+)--ep(\d+)--invalid\.json$", p.name)
        if m:
            try:
                out[(int(m.group(1)), int(m.group(2)))] = json.loads(p.read_text()).get("exception_type", "?")
            except (OSError, ValueError):
                out[(int(m.group(1)), int(m.group(2)))] = "UNREADABLE MARKER"
    if out:
        print(f"{label}{len(out)} invalid episodes (no data, skipped): "
              + ", ".join(f"task{t}--ep{e} ({x})" for (t, e), x in sorted(out.items())))
    return out


def compare_runs(a: Path, b: Path, layer: int, pool: str) -> int:
    import torch
    ia, ib = index(a), index(b)
    report_invalid(a, "run a: ")
    report_invalid(b, "run b: ")
    common = sorted(set(ia) & set(ib))
    if not common:
        print("No episodes in common")
        return 1
    rows = []
    for key in common:
        ca, cb = ia[key], ib[key]
        da, db = pd.read_csv(ca), pd.read_csv(cb)
        sa, sb = int(NAME_RE.match(ca.name).group(3)), int(NAME_RE.match(cb.name).group(3))
        n = min(len(da), len(db))
        acts_a, acts_b = da[ACTION_COLS].to_numpy()[:n], db[ACTION_COLS].to_numpy()[:n]
        diff_steps = np.where(np.abs(acts_a - acts_b).max(axis=1) > 1e-6)[0]
        first_div = int(diff_steps[0]) if len(diff_steps) else None
        feat_diff = None
        la, lb = ca.with_suffix(".layers.pt"), cb.with_suffix(".layers.pt")
        if la.exists() and lb.exists():
            pa = torch.load(la, map_location="cpu", weights_only=False, mmap=True)
            pb = torch.load(lb, map_location="cpu", weights_only=False, mmap=True)
            k = first_div if first_div is not None else n
            if k > 0:
                from build_cache import layer_names_of
                if layer_names_of(pa, pa["features"].shape[2]) != layer_names_of(pb, pb["features"].shape[2]):
                    print(f"refusing to compare {la.name}: runs use different layer conventions (33 vs 34 layers?)")
                    return 1
                pi = pa["pools"].index(pool)
                xa = pa["features"][:k, pi, layer].float()
                xb = pb["features"][:k, pi, layer].float()
                feat_diff = float(((xa - xb).norm(dim=-1) / xa.norm(dim=-1).clamp_min(1e-6)).max())
        rows.append({"task": key[0], "episode": key[1], "succ_a": sa, "succ_b": sb, "len_a": len(da), "len_b": len(db),
                     "first_action_divergence_step": first_div,
                     f"max_rel_feat_diff_L{layer}_before_divergence": feat_diff})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    flips = int((df.succ_a != df.succ_b).sum())
    diverged = int(df.first_action_divergence_step.notna().sum())
    print(f"\n{len(df)} common episodes | outcome flips: {flips} | trajectories that diverged: {diverged}")
    print("Identical outcomes on a handful of episodes is a sanity check, not proof of reproducibility.")
    return 0


def capture(rollout_dir: Path) -> int:
    """Cross-check .layers.pt against SAFE's own .pkl from the SAME episode.

    SAFE's pkl holds hidden_states[-1][0,-1,:] for each of the 7 action tokens: (T, 7, D).
    Our layer index 32 must reproduce it: first == pkl[:,0], last == pkl[:,6] exactly, mean == pkl.mean(1)
    up to bfloat16 rounding. Also checks that layers are not accidentally duplicated, and that all episodes share
    one layer convention (`layer_names`; 34-layer files add raw block 32 at index 33, which this check skips).
    """
    import pickle
    import torch
    from build_cache import layer_names_of
    names0 = None
    eps = sorted(rollout_dir.glob("task*--ep*--succ*.csv"))
    report_invalid(rollout_dir)
    if not eps:
        print(f"no completed episodes in {rollout_dir}")
        return 1
    failures = 0
    for csv in eps:
        lp, pp = csv.with_suffix(".layers.pt"), csv.with_suffix(".pkl")
        if not (lp.exists() and pp.exists()):
            print(f"FAIL {csv.stem}: missing .layers.pt or .pkl")
            failures += 1
            continue
        lay = torch.load(lp, map_location="cpu", weights_only=False)
        ref = pickle.load(open(pp, "rb"))["hidden_states"]  # (T, 7, D)
        f, pools = lay["features"], lay["pools"]          # (T, P, 33 or 34, D)
        n_rows = len(pd.read_csv(csv))
        msgs = []
        try:
            names = layer_names_of(lay, f.shape[2])
        except ValueError as e:
            names = None
            msgs.append(str(e))
        names0 = names0 or names
        if names is not None and names != names0:
            msgs.append(f"mixed layer conventions: {len(names)} layers vs {len(names0)} in an earlier episode")
        if not (f.shape[0] == ref.shape[0] == n_rows):
            msgs.append(f"length mismatch layers={f.shape[0]} pkl={ref.shape[0]} csv={n_rows}")
        else:
            last_layer = 32  # SAFE's s[-1] = block 32 after the final norm ("block32_postnorm"), also in 34-layer files
            if "first" in pools and not torch.equal(f[:, pools.index("first"), last_layer], ref[:, 0].to(f.dtype)):
                msgs.append("first-token L32 != SAFE pkl token 0")
            if "last" in pools and not torch.equal(f[:, pools.index("last"), last_layer], ref[:, -1].to(f.dtype)):
                msgs.append("last-token L32 != SAFE pkl token 6")
            if "mean" in pools:
                a = f[:, pools.index("mean"), last_layer].float()
                b = ref.float().mean(1)
                rel = ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6)).max().item()
                if rel > 1e-2:
                    msgs.append(f"mean-pooled L32 differs from SAFE pkl mean by rel {rel:.2e}")
            p0 = pools.index("mean") if "mean" in pools else 0
            x = f[:, p0].float()
            dup = [l for l in range(1, f.shape[2]) if torch.equal(x[:, l], x[:, l - 1])]
            if dup:
                msgs.append(f"layers identical to previous layer: {dup}")
            if not torch.isfinite(x).all():
                msgs.append("non-finite values")
        print(("FAIL " if msgs else "ok   ") + csv.stem + (": " + "; ".join(msgs) if msgs else ""))
        failures += bool(msgs)
    print(f"\n{len(eps) - failures}/{len(eps)} episodes pass the capture check")
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--cache", required=True, type=Path)
    v.add_argument("--quick", action="store_true")
    c = sub.add_parser("compare-runs")
    c.add_argument("--a", required=True, type=Path)
    c.add_argument("--b", required=True, type=Path)
    c.add_argument("--layer", type=int, default=32)
    c.add_argument("--pool", default="mean")
    k = sub.add_parser("capture")
    k.add_argument("--rollout-dir", required=True, type=Path)
    args = ap.parse_args()
    if args.cmd == "verify":
        sys.exit(verify(args.cache, args.quick))
    if args.cmd == "capture":
        sys.exit(capture(args.rollout_dir))
    sys.exit(compare_runs(args.a, args.b, args.layer, args.pool))


if __name__ == "__main__":
    main()
