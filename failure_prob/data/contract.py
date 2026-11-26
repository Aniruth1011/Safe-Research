"""B8: the shared-cache data contract, as code, plus a checker.

The contract is what every loader, probe, sweep and fusion run in this repo assumes about a cache version folder
(e.g. /scratch1/akashrav/safe_cache/v1). `validate_cache()` returns a list of problems (empty = the cache satisfies
the contract). The synthetic cache generator (failure_prob/data/synthetic_cache.py) builds caches FROM these constants,
and tests/test_data_contract.py fails if the code or a cache drifts away from them.

Standard library only (no numpy/pandas/torch), so it runs on a CARC login node against the real cache:

    python3 failure_prob/data/contract.py /scratch1/akashrav/safe_cache/v1 --expect-real

Unreadable unseen_test files (locked to the final benchmark) count as withheld, not as problems.
"""
import argparse
import ast
import csv
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

CONTRACT_VERSION = "cache-v1"

# ---- layers ------------------------------------------------------------------------------------------------------
# Index = HF hidden_states index; 33 = raw output of block 32 before the final RMSNorm (added by PR #3).
LAYER_NAMES = ["emb"] + [f"block{i:02d}" for i in range(1, 32)] + ["block32_postnorm", "block32_prenorm"]
NUM_LAYERS = len(LAYER_NAMES)                     # 34
POOLS = ("first", "last", "mean")                 # over the 7 action tokens (SAFE token_idx_rel 0.0 / 1.0 / mean)
SPLITS = ("train", "seen_val", "unseen_test")
UNSEEN_TASKS = (0, 3, 5)                          # failure_prob/protocol.py
FEATURE_DTYPE = "<f2"                             # float16
STATS_DTYPE = "<f4"                               # norm_stats mean/std float32

# ---- tables ------------------------------------------------------------------------------------------------------
EPISODE_COLUMNS = ["trajectory_id", "task_id", "task_name", "episode_idx", "success", "failure", "length",
                   "task_min_step", "split", "row_start", "row_end"]
TRAJECTORY_ID = re.compile(r"^libero_10/t(\d{2})/e(\d{2})$")
SPLITS_JSON_KEYS = ("trajectories", "unseen_tasks", "task_names", "seen_train_ratio", "split_seed")
NORM_STATS_KEYS = ("layers", "mean", "std", "n_rows")
MANIFEST_KEYS = ("cache_version", "dataset", "num_hidden_states", "layer_names", "layers_exported", "pools_exported",
                 "feature_dim", "dtype", "features_are_unnormalized", "label", "splits", "files")

# ---- only for the real OpenVLA/LIBERO-10 cache (--expect-real) ----------------------------------------------------
REAL = {"n_tasks": 10, "episodes_per_task": 50, "feature_dim": 4096, "max_length": 520}


# ------------------------------------------------------------------------------------------------------------------
def _npy_header(head: bytes) -> dict:
    """Parse a .npy header (dtype descr, fortran_order, shape) from the first bytes of the file."""
    if head[:6] != b"\x93NUMPY":
        raise ValueError("not a .npy file")
    size = 2 if head[6] == 1 else 4
    hlen = int.from_bytes(head[8:8 + size], "little")
    start = 8 + size
    if len(head) < start + hlen:
        raise ValueError("truncated .npy header")
    return ast.literal_eval(head[start:start + hlen].decode("latin1"))


def read_npy_header(path: Path) -> dict:
    with open(path, "rb") as f:
        return _npy_header(f.read(4096))


def _npz_members(path: Path) -> Dict[str, dict]:
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            with z.open(name) as f:
                blob = f.read(4096)
            hdr = _npy_header(blob)
            if hdr["shape"] == () and name == "n_rows.npy":
                data = z.read(name)
                start = 8 + (2 if data[6] == 1 else 4) + int.from_bytes(data[8:8 + (2 if data[6] == 1 else 4)], "little")
                hdr["value"] = int.from_bytes(data[start:start + 8], "little", signed=True)
            out[name[:-4] if name.endswith(".npy") else name] = hdr
    return out


def validate_cache(root, expect_real: bool = False) -> Tuple[List[str], List[str]]:
    """Check one cache version folder against the contract. Returns (problems, notes)."""
    root = Path(root)
    problems = []  # type: List[str]
    notes = []  # type: List[str]
    P = problems.append

    for f in ("manifest.json", "episodes.csv", "splits.json"):
        if not (root / f).exists():
            P(f"missing {f}")
    if problems:
        return problems, notes

    # ---- manifest ----
    man = json.loads((root / "manifest.json").read_text())
    for k in MANIFEST_KEYS:
        if k not in man:
            P(f"manifest.json: missing key {k!r}")
    if man.get("num_hidden_states") != NUM_LAYERS:
        P(f"manifest.json: num_hidden_states {man.get('num_hidden_states')} != {NUM_LAYERS}")
    if man.get("layer_names") != LAYER_NAMES:
        P(f"manifest.json: layer_names differ from the contract (got {str(man.get('layer_names'))[:80]}...)")
    if man.get("layers_exported") != list(range(NUM_LAYERS)):
        P("manifest.json: layers_exported is not 0..33")
    if sorted(man.get("pools_exported") or []) != sorted(POOLS):
        P(f"manifest.json: pools_exported {man.get('pools_exported')} != {list(POOLS)}")
    if not str(man.get("dtype", "")).startswith("float16"):
        P(f"manifest.json: dtype {man.get('dtype')!r} is not float16")
    if man.get("features_are_unnormalized") is not True:
        P("manifest.json: features_are_unnormalized must be true (standardization happens at load time, B5)")
    dim = man.get("feature_dim")
    if not isinstance(dim, int) or dim < 1:
        P(f"manifest.json: feature_dim {dim!r} invalid")
        return problems, notes

    # ---- episodes.csv ----
    with open(root / "episodes.csv", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)
    if cols != EPISODE_COLUMNS:
        P(f"episodes.csv: columns {cols} != contract {EPISODE_COLUMNS}")
        return problems, notes
    eps = []
    for r in rows:
        try:
            eps.append({**r, **{k: int(r[k]) for k in ("task_id", "episode_idx", "success", "failure", "length",
                                                        "task_min_step", "row_start", "row_end")}})
        except ValueError:
            P(f"episodes.csv: non-integer field in row {r.get('trajectory_id')}")
            return problems, notes
    ids = [e["trajectory_id"] for e in eps]
    if len(ids) != len(set(ids)):
        P("episodes.csv: duplicate trajectory_id")
    for e in eps:
        m = TRAJECTORY_ID.match(e["trajectory_id"])
        if not m or (int(m.group(1)), int(m.group(2))) != (e["task_id"], e["episode_idx"]):
            P(f"episodes.csv: trajectory_id {e['trajectory_id']!r} doesn't match task_id/episode_idx")
        if e["success"] not in (0, 1) or e["failure"] != 1 - e["success"]:
            P(f"episodes.csv: {e['trajectory_id']}: failure must be 1 - success (label 1 = failure)")
        if e["split"] not in SPLITS:
            P(f"episodes.csv: {e['trajectory_id']}: unknown split {e['split']!r}")
        if e["length"] < 1 or e["row_end"] - e["row_start"] != e["length"]:
            P(f"episodes.csv: {e['trajectory_id']}: row range [{e['row_start']}, {e['row_end']}) != length {e['length']}")
    by_task = {}  # type: Dict[int, List[dict]]
    for e in eps:
        by_task.setdefault(e["task_id"], []).append(e)
    for t, es in by_task.items():
        mn = min(e["length"] for e in es)
        if any(e["task_min_step"] != mn for e in es):
            P(f"episodes.csv: task {t}: task_min_step must be the task's minimum length ({mn})")
    rows_per_split = {}
    for s in SPLITS:
        es = sorted((e for e in eps if e["split"] == s), key=lambda e: e["row_start"])
        expect = 0
        for e in es:
            if e["row_start"] != expect:
                P(f"episodes.csv: {s} row ranges not contiguous at {e['trajectory_id']} (expected start {expect})")
                break
            expect = e["row_end"]
        rows_per_split[s] = expect
    unseen = tuple(sorted({e["task_id"] for e in eps if e["split"] == "unseen_test"}))
    if unseen != UNSEEN_TASKS:
        P(f"episodes.csv: unseen tasks {unseen} != contract {UNSEEN_TASKS}")
    seen_tasks = {e["task_id"] for e in eps if e["split"] != "unseen_test"}
    if seen_tasks & set(unseen):
        P(f"episodes.csv: tasks {sorted(seen_tasks & set(unseen))} are both seen and unseen")
    for t in sorted(seen_tasks):
        have = {e["split"] for e in by_task[t]}
        if have != {"train", "seen_val"}:
            P(f"episodes.csv: seen task {t} must have both train and seen_val episodes (has {sorted(have)})")

    # ---- splits.json ----
    sp = json.loads((root / "splits.json").read_text())
    for k in SPLITS_JSON_KEYS:
        if k not in sp:
            P(f"splits.json: missing key {k!r}")
    traj = sp.get("trajectories", {})
    if set(traj) != set(SPLITS):
        P(f"splits.json: trajectories must have exactly {SPLITS}")
    else:
        owner = {}  # type: Dict[str, str]
        for s in SPLITS:
            for t in traj[s]:
                if t in owner:
                    P(f"splits.json: {t} is in both {owner[t]} and {s}")
                owner[t] = s
        if set(owner) != set(ids):
            P("splits.json: trajectories don't list exactly the episodes in episodes.csv")
        elif any(owner[e["trajectory_id"]] != e["split"] for e in eps):
            P("splits.json and episodes.csv disagree on which split an episode is in")
    if tuple(sorted(sp.get("unseen_tasks", []))) != UNSEEN_TASKS:
        P(f"splits.json: unseen_tasks {sp.get('unseen_tasks')} != {list(UNSEEN_TASKS)}")

    # ---- manifest split summary ----
    for s in SPLITS:
        summ = (man.get("splits") or {}).get(s, {})
        n = sum(e["split"] == s for e in eps)
        if summ.get("trajectories") != n or summ.get("rows") != rows_per_split[s]:
            P(f"manifest.json: splits.{s} ({summ}) disagrees with episodes.csv ({n} episodes, {rows_per_split[s]} rows)")

    # ---- norm_stats ----
    for p in POOLS:
        f = root / "norm_stats" / f"{p}.npz"
        if not f.exists():
            P(f"missing norm_stats/{p}.npz")
            continue
        mem = _npz_members(f)
        missing = [k for k in NORM_STATS_KEYS if k not in mem]
        if missing:
            P(f"norm_stats/{p}.npz: missing {missing}")
            continue
        if tuple(mem["layers"]["shape"]) != (NUM_LAYERS,):
            P(f"norm_stats/{p}.npz: layers shape {mem['layers']['shape']} != ({NUM_LAYERS},)")
        for k in ("mean", "std"):
            if tuple(mem[k]["shape"]) != (NUM_LAYERS, dim) or mem[k]["descr"] != STATS_DTYPE:
                P(f"norm_stats/{p}.npz: {k} is {mem[k]['descr']} {mem[k]['shape']}, contract {STATS_DTYPE} ({NUM_LAYERS}, {dim})")
        if mem["n_rows"].get("value") not in (None, rows_per_split["train"]):
            P(f"norm_stats/{p}.npz: n_rows {mem['n_rows'].get('value')} != train rows {rows_per_split['train']} "
              "(statistics must be fit on train only)")

    # ---- feature files ----
    withheld = 0
    for p in POOLS:
        for s in SPLITS:
            for layer in range(NUM_LAYERS):
                f = root / "features" / p / s / f"L{layer:02d}.npy"
                try:
                    hdr = read_npy_header(f)
                except (FileNotFoundError, PermissionError) as exc:
                    if s == "unseen_test":
                        withheld += 1
                        continue
                    P(f"features/{p}/{s}/L{layer:02d}.npy: {'missing' if isinstance(exc, FileNotFoundError) else 'unreadable'}")
                    continue
                if hdr["descr"] != FEATURE_DTYPE:
                    P(f"features/{p}/{s}/L{layer:02d}.npy: dtype {hdr['descr']} != {FEATURE_DTYPE} (float16)")
                if tuple(hdr["shape"]) != (rows_per_split[s], dim) or hdr.get("fortran_order"):
                    P(f"features/{p}/{s}/L{layer:02d}.npy: shape {hdr['shape']} != ({rows_per_split[s]}, {dim})")
    if withheld:
        notes.append(f"{withheld} unseen_test feature files withheld or locked (expected until the final benchmark)")

    # ---- real-cache expectations ----
    if expect_real:
        if len(by_task) != REAL["n_tasks"] or any(len(es) != REAL["episodes_per_task"] for es in by_task.values()):
            P(f"real cache: expected {REAL['n_tasks']} tasks x {REAL['episodes_per_task']} episodes")
        if dim != REAL["feature_dim"]:
            P(f"real cache: feature_dim {dim} != {REAL['feature_dim']}")
        if any(e["length"] > REAL["max_length"] for e in eps):
            P(f"real cache: an episode is longer than {REAL['max_length']} steps")
    return problems, notes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache", type=Path, help="cache version folder, e.g. /scratch1/akashrav/safe_cache/v1")
    ap.add_argument("--expect-real", action="store_true", help="also check the real cache's size: 10 x 50, D=4096, <=520 steps")
    args = ap.parse_args(argv)
    problems, notes = validate_cache(args.cache, expect_real=args.expect_real)
    for n in notes:
        print(f"note: {n}")
    if problems:
        print(f"CONTRACT FAILED ({CONTRACT_VERSION}): {len(problems)} problem(s)")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"CONTRACT OK ({CONTRACT_VERSION}): {args.cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
