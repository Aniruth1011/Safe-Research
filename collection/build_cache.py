#!/usr/bin/env python
"""Build the team's versioned shared feature cache from raw LIBERO rollouts.

Input: a rollout folder written by the patched `run_libero_eval.py --save_all_layers True`, containing per episode
    task{T}--ep{E}--succ{S}.csv        (written last; marks the episode complete)
    task{T}--ep{E}--succ{S}.pkl        (SAFE's original last-layer features, unchanged)
    task{T}--ep{E}--succ{S}.layers.pt  ((T, P, L, D) bfloat16, all layers, P poolings; L=34 with raw block 32, else 33)
    task{T}--ep{E}--invalid.json       (INSTEAD of the files above when the episode raised; see README "Invalid episodes")

Output (CACHE_ROOT/<version>/):
    manifest.json         everything about the cache, incl. sha256 of every file
    CHECKSUMS.sha256      same checksums, `sha256sum -c` compatible
    episodes.csv          one row per trajectory: ids, task, success/failure, length, split, row range
    splits.json           EXPLICIT trajectory IDs for train / seen_val / unseen_test
    features/<pool>/<split>/L<NN>.npy    float16 [rows_in_split, D]; one file per layer -> download only what you need
    norm_stats/<pool>.npz mean/std per layer computed on the TRAIN split only
    provenance/           copies of the rollout provenance json files

Unseen-task features live in their own `unseen_test/` folders so they can be withheld until the locked final run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

NAME_RE = re.compile(r"^task(\d+)--ep(\d+)--succ([01])\.csv$")
SPLITS = ("train", "seen_val", "unseen_test")
FP16_MAX = 65504.0
LAYER_CONVENTION = (
    "hf_hidden_states: 0=embeddings, i(1..31)=output of decoder layer i (pre-norm), "
    "32=output of decoder layer 32 after final RMSNorm (SAFE's last layer). Spreadsheet L<i> == index i."
    " 33 (only in 34-layer datasets, see layer_names)=output of decoder layer 32 BEFORE the final RMSNorm."
)


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def discover(rollout_dir: Path) -> pd.DataFrame:
    rows = []
    for csv in sorted(rollout_dir.glob("task*--ep*--succ*.csv")):
        m = NAME_RE.match(csv.name)
        if not m:
            continue
        task, ep, succ = int(m.group(1)), int(m.group(2)), int(m.group(3))
        rows.append({
            "task_id": task, "episode_idx": ep, "success": succ,
            "csv": str(csv), "pkl": str(csv.with_suffix(".pkl")),
            "layers": str(csv.with_suffix(".layers.pt")),
        })
    if not rows:
        sys.exit(f"No completed episodes (task*--ep*--succ*.csv) found in {rollout_dir}")
    df = pd.DataFrame(rows).sort_values(["task_id", "episode_idx"]).reset_index(drop=True)
    dup = df[df.duplicated(["task_id", "episode_idx"], keep=False)]
    if len(dup):
        sys.exit(f"Duplicate episodes (same task/episode, different success flag?):\n{dup[['task_id','episode_idx','csv']]}")
    df["trajectory_id"] = [f"libero_10/t{t:02d}/e{e:02d}" for t, e in zip(df.task_id, df.episode_idx)]
    df["failure"] = 1 - df["success"]
    return df


INVALID_RE = re.compile(r"^task(\d+)--ep(\d+)--invalid\.json$")


def account_invalid_episodes(rollout_dir: Path, df: pd.DataFrame, counts: pd.Series, expect_episodes: int,
                             max_invalid: int | None, problems: list[str]) -> tuple[list[dict], pd.Series]:
    """Must-fix 4: discover `task{T}--ep{E}--invalid.json` markers written by the patched run_libero_eval.py.

    Exits (never bypassable) on an unreadable marker, a marker whose content disagrees with its file name, a marker
    for an episode index outside 0..expect_episodes-1, an episode that has both a marker and data files, or more
    markers than --max-invalid. Appends to `problems` (bypassable with --allow-incomplete, like any incompleteness)
    every episode 0..expect_episodes-1 of a task that has neither a completed .csv nor a marker.
    Returns (one manifest record per invalid episode, `counts` with the invalid episodes added per task)."""
    errors, invalid = [], []
    for p in sorted(rollout_dir.glob("task*--ep*--invalid.json")):
        m = INVALID_RE.match(p.name)
        if not m:
            continue
        t, e = int(m.group(1)), int(m.group(2))
        try:
            rec = json.loads(p.read_text())
        except (OSError, ValueError) as ex:
            errors.append(f"{p.name}: unreadable marker ({ex})")
            continue
        if not isinstance(rec, dict) or rec.get("task_id") != t or rec.get("episode_idx") != e \
                or not rec.get("exception_type"):
            errors.append(f"{p.name}: content does not match the file name or has no exception_type")
            continue
        if not 0 <= e < expect_episodes:
            errors.append(f"{p.name}: episode index outside 0..{expect_episodes - 1}")
        data = sorted(q.name for q in rollout_dir.glob(f"task{t}--ep{e}--succ*"))
        if data:
            errors.append(f"{p.name}: episode has both an invalid marker and data files {data}")
        invalid.append({"trajectory_id": f"libero_10/t{t:02d}/e{e:02d}", "task_id": t, "episode_idx": e,
                        "exception_type": rec["exception_type"],
                        "exception_message": str(rec.get("exception_message", ""))[:500],
                        "step_index": rec.get("step_index"), "timestamp_utc": rec.get("timestamp_utc"),
                        "marker": p.name})
    if errors:
        sys.exit("Inconsistent invalid-episode markers:\n  " + "\n  ".join(errors))
    if max_invalid is not None and len(invalid) > max_invalid:
        sys.exit(f"{len(invalid)} invalid episodes exceed --max-invalid {max_invalid}: "
                 + ", ".join(f"{r['marker']} ({r['exception_type']})" for r in invalid))
    done = set(zip(df.task_id.astype(int), df.episode_idx.astype(int)))
    marked = {(r["task_id"], r["episode_idx"]) for r in invalid}
    tasks = sorted({t for t, _ in done | marked})
    unexplained = [f"task{t}--ep{e}" for t in tasks for e in range(expect_episodes) if (t, e) not in done | marked]
    if unexplained:
        problems.append(f"{len(unexplained)} episodes have neither a .csv nor an invalid marker, e.g. {unexplained[:5]}")
    if invalid:
        print(f"{len(invalid)} invalid episodes (excluded from the cache): "
              + ", ".join(f"t{r['task_id']}/e{r['episode_idx']} {r['exception_type']}" for r in invalid))
    per_task = pd.Series({t: sum(1 for tt, _ in marked if tt == t) for t, _ in marked}, dtype="int64")
    return invalid, counts.add(per_task, fill_value=0).astype("int64")


def make_splits(df: pd.DataFrame, unseen_tasks: list[int], seen_train_ratio: float, seed: int) -> dict[str, list[str]]:
    """Seen tasks: per task, stratified by outcome, `seen_train_ratio` to train and the rest to seen_val."""
    splits = {s: [] for s in SPLITS}
    for task, g in df.groupby("task_id"):
        if task in unseen_tasks:
            splits["unseen_test"] += g.trajectory_id.tolist()
            continue
        rng = np.random.default_rng([seed, int(task)])
        for _, sub in g.groupby("success"):
            ids = sorted(sub.trajectory_id.tolist())
            ids = [ids[i] for i in rng.permutation(len(ids))]
            n_train = int(round(seen_train_ratio * len(ids)))
            splits["train"] += ids[:n_train]
            splits["seen_val"] += ids[n_train:]
    return {k: sorted(v) for k, v in splits.items()}


def load_episode(row, mmap: bool = False) -> tuple[torch.Tensor, dict]:
    # mmap=True only maps the file, so the validation pass doesn't read every byte
    payload = torch.load(row.layers, map_location="cpu", weights_only=False, mmap=mmap)
    return payload["features"], payload


FINGERPRINT_KEY = "run_fingerprint"
FINGERPRINT_MUST_MATCH = ("attn_implementation", "gpu_name", "render_backend")
REQUIRED_ATTN_IMPLEMENTATION = "flash_attention_2"


def check_run_fingerprints(df: pd.DataFrame, allow_missing: bool) -> dict:
    """E3 dataset-consistency guard. Reads `run_fingerprint` (written by run_libero_eval.py --save_all_layers, see
    experiments/robot/libero/run_guards.py) from every episode and exits non-zero if attention implementation, GPU name
    or MuJoCo render backend differ across the dataset, or if any attention implementation is not flash_attention_2.
    Episodes without a fingerprint are refused unless `allow_missing` (recorded in the manifest).
    Returns the record stored in manifest.json under `run_fingerprint`."""
    fps, missing = {}, []
    for row in df.itertuples():
        fp = load_episode(row, mmap=True)[1].get(FINGERPRINT_KEY)
        if fp is None:
            missing.append(Path(row.layers).name)
        else:
            fps[Path(row.layers).name] = fp
    problems = []
    if missing and not allow_missing:
        problems.append(f"{len(missing)}/{len(df)} episodes have no {FINGERPRINT_KEY} (pass --allow-missing-fingerprint "
                        f"to build anyway; it is recorded in the manifest), e.g. {missing[:3]}")
    values = {}
    for key in FINGERPRINT_MUST_MATCH:
        groups = defaultdict(list)
        for name, fp in fps.items():
            groups[json.dumps(fp.get(key), sort_keys=True)].append(name)
        values[key] = groups
        if len(groups) > 1:
            listing = "; ".join(f"{v} x{len(n)} e.g. {n[:3]}" for v, n in sorted(groups.items()))
            problems.append(f"mixed {key} across the dataset: {listing}")
    bad_attn = {v: n for v, n in values["attn_implementation"].items()
                if json.loads(v) != REQUIRED_ATTN_IMPLEMENTATION}
    for v, n in sorted(bad_attn.items()):
        problems.append(f"attn_implementation {v} is not {REQUIRED_ATTN_IMPLEMENTATION!r}: x{len(n)} e.g. {n[:3]}")
    if problems:
        sys.exit("Inconsistent rollout dataset (run fingerprints):\n  " + "\n  ".join(problems))
    record = {key: (json.loads(next(iter(values[key]))) if values[key] else None) for key in FINGERPRINT_MUST_MATCH}
    versions = sorted({json.dumps(fp.get("versions"), sort_keys=True) for fp in fps.values()})
    record.update({
        "versions_seen": [json.loads(v) for v in versions],
        "episodes_with_fingerprint": len(fps),
        "allow_missing_fingerprint": bool(allow_missing),
        "episodes_missing_fingerprint": missing,
    })
    return record


# Files without `layer_names` (written before raw block 32 was added) follow this 33-layer convention.
LEGACY_LAYER_NAMES = ["emb"] + [f"block{i:02d}" for i in range(1, 32)] + ["block32_postnorm"]


def layer_names_of(payload: dict, n_layers: int) -> list[str]:
    """Layer names of one .layers.pt (`layer_names` key; absent = legacy 33-layer convention)."""
    names = payload.get("layer_names")
    if names is None:
        if n_layers != len(LEGACY_LAYER_NAMES):
            raise ValueError(f"no layer_names and {n_layers} layers (only 33-layer files may omit layer_names)")
        return list(LEGACY_LAYER_NAMES)
    if len(names) != n_layers:
        raise ValueError(f"{len(names)} layer_names for {n_layers} saved layers")
    return list(names)


def check_layer_names(path, payload: dict, n_layers: int, expected: list[str] | None) -> list[str]:
    """This file's layer names; exits if they are unreadable or differ from `expected` (mixed 33/34-layer dataset)."""
    try:
        names = layer_names_of(payload, n_layers)
    except ValueError as e:
        sys.exit(f"{path}: {e}")
    if expected is not None and names != expected:
        sys.exit(f"{path}: mixed layer conventions in one dataset ({len(names)} layers, last {names[-1]!r} vs "
                 f"{len(expected)} layers, last {expected[-1]!r}); use rollouts from one --save_raw_block32 setting")
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollout-dir", required=True, type=Path)
    ap.add_argument("--cache-root", required=True, type=Path)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--checkpoint", default="openvla/openvla-7b-finetuned-libero-10")
    ap.add_argument("--unseen-tasks", type=int, nargs="+", default=[0, 3, 5],
                    help="Default = the unseen tasks SAFE's own split_rollouts picks with seed 0 (verified by running it). "
                         "Only the TASK split matches SAFE; train/seen_val assignment is ours (explicit, stratified).")
    ap.add_argument("--seen-train-ratio", type=float, default=0.6, help="SAFE uses 0.6 for OpenVLA")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--split-file", type=Path, help="Reuse an existing splits.json instead of generating one")
    ap.add_argument("--layers", type=int, nargs="+", help="Subset of layer indices to export (default: all)")
    ap.add_argument("--pools", nargs="+", help="Subset of poolings to export (default: all present)")
    ap.add_argument("--expect-tasks", type=int, default=10)
    ap.add_argument("--expect-episodes", type=int, default=50)
    ap.add_argument("--max-invalid", type=int, help="Refuse to build if more episodes than this are invalid (default: no limit)")
    ap.add_argument("--allow-incomplete", action="store_true", help="For smoke tests only")
    ap.add_argument("--skip-checksums", action="store_true", help="For smoke tests only")
    ap.add_argument("--allow-missing-fingerprint", action="store_true",
                    help="Build even if some .layers.pt files have no run_fingerprint (recorded in the manifest)")
    ap.add_argument("--provenance-files", type=Path, nargs="*", default=[],
                    help="Extra files (pip freeze, repo commits) copied into provenance/ BEFORE checksumming")
    args = ap.parse_args()

    out = args.cache_root / args.version
    if out.exists():
        sys.exit(f"{out} already exists. Caches are immutable: pick a new --version or delete it deliberately.")

    df = discover(args.rollout_dir)

    # ---------- completeness ----------
    counts = df.groupby("task_id").size()
    problems = []
    invalid, counts = account_invalid_episodes(args.rollout_dir, df, counts, args.expect_episodes, args.max_invalid, problems)
    if len(counts) != args.expect_tasks:
        problems.append(f"expected {args.expect_tasks} tasks, found {len(counts)}: {counts.index.tolist()}")
    short = counts[counts != args.expect_episodes]
    if len(short):
        problems.append(f"tasks without exactly {args.expect_episodes} episodes: {short.to_dict()}")
    missing = [p for col in ("pkl", "layers") for p in df[col] if not Path(p).exists()]
    if missing:
        problems.append(f"{len(missing)} missing .pkl/.layers.pt files, e.g. {missing[:3]}")
    if problems and not args.allow_incomplete:
        sys.exit("Rollouts incomplete:\n  " + "\n  ".join(problems))
    if missing:
        sys.exit("Cannot build without every .layers.pt/.pkl file:\n  " + "\n  ".join(missing[:10]))
    run_fingerprint = check_run_fingerprints(df, args.allow_missing_fingerprint)  # E3: refuse a mixed dataset

    # ---------- lengths + per-episode consistency ----------
    lengths, names, suites, pools0, shape0 = [], [], [], None, None
    layer_names0 = None
    for row in df.itertuples():
        feats, payload = load_episode(row, mmap=True)
        n_csv = len(pd.read_csv(row.csv))
        if feats.shape[0] != n_csv:
            sys.exit(f"{row.layers}: {feats.shape[0]} feature steps but {n_csv} csv rows")
        if bool(payload.get("episode_success")) != bool(row.success):
            sys.exit(f"{row.layers}: success flag disagrees with filename")
        layer_names0 = check_layer_names(row.layers, payload, feats.shape[2], layer_names0)
        if pools0 is None:
            pools0, shape0 = list(payload["pools"]), tuple(feats.shape[1:])
        elif list(payload["pools"]) != pools0 or tuple(feats.shape[1:]) != shape0:
            sys.exit(f"{row.layers}: pools/shape {payload['pools']} {tuple(feats.shape[1:])} != {pools0} {shape0}")
        lengths.append(int(feats.shape[0]))
        names.append(payload.get("task_description", ""))
        suites.append(payload.get("task_suite_name", ""))
    df["length"] = lengths
    df["task_name"] = names
    for t, g in df.groupby("task_id"):
        if g.task_name.nunique() > 1:
            sys.exit(f"task {t} has inconsistent task descriptions: {g.task_name.unique().tolist()}")
    if len(set(suites)) > 1:
        sys.exit(f"mixed task suites in one rollout folder: {sorted(set(suites))}")
    task_suite = suites[0] if suites else ""
    df["task_min_step"] = df.groupby("task_id")["length"].transform("min")  # SAFE's per-task evaluation horizon

    n_pools, n_layers, dim = shape0
    pools = args.pools or pools0
    layers = sorted(args.layers) if args.layers else list(range(n_layers))
    bad = [p for p in pools if p not in pools0] + [l for l in layers if not 0 <= l < n_layers]
    if bad:
        sys.exit(f"Requested pools/layers not available: {bad}")
    pool_pos = {p: pools0.index(p) for p in pools}

    # ---------- splits ----------
    if args.split_file:
        splits = json.loads(args.split_file.read_text())["trajectories"]
    else:
        splits = make_splits(df, args.unseen_tasks, args.seen_train_ratio, args.split_seed)
    traj_to_split = {t: s for s, ids in splits.items() for t in ids}
    unassigned = set(df.trajectory_id) - set(traj_to_split)
    if unassigned:
        sys.exit(f"{len(unassigned)} trajectories missing from split file, e.g. {sorted(unassigned)[:3]}")
    df["split"] = df.trajectory_id.map(traj_to_split)

    # row ranges inside each split's feature files (episodes in canonical task/episode order)
    df["row_start"] = 0
    for s in SPLITS:
        idx = df.index[df.split == s]
        starts = np.concatenate([[0], np.cumsum(df.loc[idx, "length"].to_numpy())[:-1]]) if len(idx) else []
        df.loc[idx, "row_start"] = starts
    df["row_end"] = df.row_start + df.length

    out.mkdir(parents=True)
    (out / "features").mkdir()
    try:
        # ---------- write features ----------
        rows_per_split = {s: int(df.loc[df.split == s, "length"].sum()) for s in SPLITS}
        mm = {}
        for p in pools:
            for s in SPLITS:
                if rows_per_split[s] == 0:
                    continue
                d = out / "features" / p / s
                d.mkdir(parents=True, exist_ok=True)
                for l in layers:
                    mm[(p, s, l)] = np.lib.format.open_memmap(d / f"L{l:02d}.npy", mode="w+", dtype=np.float16,
                                                              shape=(rows_per_split[s], dim))
        sums = {p: np.zeros((len(layers), dim), np.float64) for p in pools}
        sqs = {p: np.zeros((len(layers), dim), np.float64) for p in pools}
        n_train_rows = 0
        for i, row in enumerate(df.itertuples(), 1):
            feats, _ = load_episode(row)
            f32 = feats.float()
            if not torch.isfinite(f32).all():
                sys.exit(f"{row.layers}: non-finite values in features")
            peak = f32.abs().max().item()
            if peak > FP16_MAX:
                sys.exit(f"{row.layers}: |value| {peak:.1f} exceeds float16 range; rebuild with float32 support")
            for p in pools:
                block = f32[:, pool_pos[p]][:, layers].numpy()  # (T, len(layers), D)
                for j, l in enumerate(layers):
                    mm[(p, row.split, l)][row.row_start:row.row_end] = block[:, j].astype(np.float16)
                if row.split == "train":
                    b64 = block.astype(np.float64)
                    sums[p] += b64.sum(0)
                    sqs[p] += (b64 ** 2).sum(0)
            if row.split == "train":
                n_train_rows += row.length
            if i % 25 == 0 or i == len(df):
                print(f"  wrote {i}/{len(df)} episodes", flush=True)
        for m in mm.values():
            m.flush()
        del mm

        # ---------- train-only normalization stats ----------
        (out / "norm_stats").mkdir()
        if n_train_rows:
            for p in pools:
                mean = sums[p] / n_train_rows
                std = np.sqrt(np.maximum(sqs[p] / n_train_rows - mean ** 2, 0.0))
                np.savez(out / "norm_stats" / f"{p}.npz", layers=np.array(layers), mean=mean.astype(np.float32),
                         std=std.astype(np.float32), n_rows=n_train_rows)

        # ---------- tables ----------
        cols = ["trajectory_id", "task_id", "task_name", "episode_idx", "success", "failure", "length", "task_min_step",
                "split", "row_start", "row_end"]
        df[cols].to_csv(out / "episodes.csv", index=False)
        split_doc = {
            "description": "Explicit trajectory assignment. Do not regenerate from a seed; reuse this file.",
            "unseen_tasks": sorted(int(t) for t in df.loc[df.split == "unseen_test", "task_id"].unique()),
            "task_names": {int(t): g.task_name.iloc[0] for t, g in df.groupby("task_id")},
            "provenance_of_task_split": (
                "Unseen tasks [0,3,5] equal what SAFE's failure_prob.data.openvla.split_rollouts selects after "
                "seed_everything(0) (verified by executing SAFE's code). The train/seen_val assignment is NOT SAFE's: "
                "SAFE assigns via torch.randperm over an unsorted glob order, so its assignment depends on file order. "
                "This is ONE fixed split; it does not reproduce SAFE's aggregate over seeds 0/1/2."),
            "seen_train_ratio": args.seen_train_ratio,
            "split_seed": args.split_seed,
            "stratified_by_outcome": not bool(args.split_file),
            "trajectories": splits,
        }
        (out / "splits.json").write_text(json.dumps(split_doc, indent=2) + "\n")

        prov_dir = args.rollout_dir / "provenance"
        (out / "provenance").mkdir()
        if prov_dir.exists():
            shutil.copytree(prov_dir, out / "provenance", dirs_exist_ok=True)
        for f in args.provenance_files:
            if f.exists():
                shutil.copy2(f, out / "provenance" / f.name)
            else:
                print(f"warning: provenance file {f} not found", file=sys.stderr)

        # ---------- summary + warnings ----------
        summary = {s: {"trajectories": int((df.split == s).sum()),
                       "failures": int(df.loc[df.split == s, "failure"].sum()),
                       "rows": rows_per_split[s]} for s in SPLITS}
        per_task = {int(t): {"name": g.task_name.iloc[0], "episodes": int(len(g)), "successes": int(g.success.sum()),
                             "split": g.split.iloc[0]
                             if g.split.nunique() == 1 else "seen"} for t, g in df.groupby("task_id")}
        warnings = []
        for s in ("train", "seen_val"):
            sub = df[df.split == s]
            if len(sub) and sub.failure.nunique() < 2:
                warnings.append(f"{s} has only one class; ROC-AUC undefined")
        for t, g in df[df.split == "seen_val"].groupby("task_id"):
            if g.failure.nunique() < 2:
                warnings.append(f"seen_val task {t} has a single outcome class (per-task AUC undefined)")

        # ---------- manifest + checksums ----------
        files = sorted(p for p in out.rglob("*") if p.is_file())
        checksums = {} if args.skip_checksums else {p.relative_to(out).as_posix(): sha256(p) for p in files}  # POSIX keys on every OS
        manifest = {
            "cache_version": args.version,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "dataset": "LIBERO-10",
            "task_suite_name": task_suite,
            "libero_version": "see provenance/libero_commit.txt and provenance/provenance--*.json",
            "features_are_unnormalized": True,
            "normalization": "none applied to stored features; norm_stats/ (train split only) is an optional, explicit "
                             "preprocessing step the shared evaluator must choose and apply identically for all models",
            "backbone": "OpenVLA-7B",
            "checkpoint": args.checkpoint,
            "run_fingerprint": run_fingerprint,
            "source_rollout_dir": str(args.rollout_dir.resolve()),
            "layer_convention": LAYER_CONVENTION,
            "num_hidden_states": n_layers,
            "layer_names": layer_names0,
            "layers_exported": layers,
            "pools_exported": pools,
            "pool_meaning": {"mean": "mean over the 7 action tokens (SAFE token_idx_rel=mean)",
                             "first": "first action-token position (SAFE token_idx_rel=0.0)",
                             "last": "last action-token position (SAFE token_idx_rel=1.0)"},
            "feature_dim": dim,
            "dtype": "float16 (converted losslessly in range from bfloat16)",
            "label": "failure = 1 - episode_success (per trajectory)",
            "evaluation_horizon": "episodes.csv:task_min_step = SAFE per-task min rollout length (set_task_min_step)",
            "splits": summary,
            "invalid_episodes": invalid,  # episodes that raised during collection; none of their data is in the cache
            "per_task": per_task,
            "warnings": warnings,
            "files": {k: {"sha256": v, "bytes": (out / k).stat().st_size} for k, v in checksums.items()},
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if checksums:
            (out / "CHECKSUMS.sha256").write_text("".join(f"{v}  {k}\n" for k, v in checksums.items()))
    except BaseException:
        print(f"Build failed; removing partial cache {out}", file=sys.stderr)
        shutil.rmtree(out, ignore_errors=True)
        raise

    print(json.dumps({"cache": str(out), "splits": summary, "warnings": warnings}, indent=2))


if __name__ == "__main__":
    main()
