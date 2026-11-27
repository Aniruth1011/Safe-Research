#!/usr/bin/env python
"""Select SAFE baseline configs (C1-C3) from offline wandb runs, on seen_val only.

Applies SAFE's selection rule (scripts/get_wandb_metrics.py, METRIC_MAP["v2"]) to offline run files:
for every <wandb_dir>/**/offline-run-*/ run, read its config and final-epoch seen_val metrics, group runs by
hyperparameters, average over seeds 0/1/2 and take the argmax of falert_early_roc_auc/model_val_seen.

Offline runs of wandb 0.19 keep config and summary only inside run-<id>.wandb (files/wandb-summary.json and
files/config.yaml appear only after `wandb sync`), so the .wandb records are read directly; the JSON/YAML
files are used when present.

    python scripts/collect_safe_runs.py /scratch1/$USER/safe_baselines/indep/wandb --out /scratch1/$USER/safe_baselines/indep/selection.csv

Fails loudly if a config is missing a seed, a (config, seed) appears twice, or any key mentions unseen data.
Writes <out> (one row per config, mean +- std over seeds) and <out stem>_winner.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SELECT = "falert_early_roc_auc/model_val_seen"
REPORT = [SELECT, "falert_early_prc_auc/model_val_seen", "roc_auc/model_val_seen_tq1.0"]
SEEDS = (0, 1, 2)
# Hyperparameters that define a config, per SAFE model (plus the input features, so pools/layers never mix)
HPARAMS = {
    "indep": ["model.lr", "model.lambda_reg", "model.batch_size", "model.n_epochs"],
    "lstm": ["model.lr", "model.lambda_reg", "model.batch_size", "model.n_epochs"],
    "embed": ["model.distance", "model.topk", "model.cumsum", "model.use_success_only"],
}
INPUT_KEYS = ["dataset.name", "dataset.layer", "dataset.token_idx_rel"]


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _read_wandb_records(path: Path) -> tuple[dict, dict, int | None]:
    """(config, summary, exit_code) from an offline run-<id>.wandb file."""
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal.datastore import DataStore

    cfg, summary, exit_code = {}, {}, None
    ds = DataStore()
    ds.open_for_scan(str(path))
    while (data := ds.scan_data()) is not None:
        rec = pb.Record()
        rec.ParseFromString(data)
        kind = rec.WhichOneof("record_type")
        if kind == "run":
            cfg.update({it.key: json.loads(it.value_json) for it in rec.run.config.update})
        elif kind == "config":
            cfg.update({it.key: json.loads(it.value_json) for it in rec.config.update})
        elif kind == "history":   # last logged value per key (the final evaluation epoch)
            for it in rec.history.item:
                summary[it.key or "/".join(it.nested_key)] = json.loads(it.value_json)
        elif kind == "summary":
            for it in rec.summary.update:
                summary[it.key or "/".join(it.nested_key)] = json.loads(it.value_json)
        elif kind == "exit":
            exit_code = rec.exit.exit_code
    return cfg, summary, exit_code


def read_run(run_dir: Path) -> dict:
    files = run_dir / "files"
    if (files / "wandb-summary.json").exists() and (files / "config.yaml").exists():
        summary = json.loads((files / "wandb-summary.json").read_text())
        cfg = yaml.safe_load((files / "config.yaml").read_text())
        exit_code = 0
    else:
        recs = sorted(run_dir.glob("run-*.wandb"))
        if len(recs) != 1:
            raise SystemExit(f"{run_dir}: expected one run-*.wandb file, found {len(recs)}")
        cfg, summary, exit_code = _read_wandb_records(recs[0])
    if exit_code != 0:
        raise SystemExit(f"{run_dir}: run exited with code {exit_code}; resubmit it or remove it")
    cfg = {k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in cfg.items()}
    cfg = _flatten({k: v for k, v in cfg.items() if k in ("dataset", "model", "train")})
    bad = [k for k in summary if "unseen" in k.lower()]
    if bad:
        raise SystemExit(f"{run_dir}: summary contains unseen keys {bad[:5]}; this run must not be used")
    missing = [k for k in REPORT if k not in summary]
    if missing:
        raise SystemExit(f"{run_dir}: missing {missing} (run did not reach its final evaluation?)")
    return {"run_dir": str(run_dir), **cfg, **{k: summary[k] for k in REPORT}}


def collect(wandb_dir: Path) -> pd.DataFrame:
    runs = sorted(p for p in wandb_dir.glob("**/offline-run-*") if p.is_dir())
    if not runs:
        raise SystemExit(f"no offline-run-* directories under {wandb_dir}")
    return pd.DataFrame([read_run(r) for r in runs])


def select(runs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    models = runs["model.name"].unique()
    if len(models) != 1:
        raise SystemExit(f"runs mix models {list(models)}; point at one model's wandb dir")
    model = models[0]
    keys = INPUT_KEYS + HPARAMS[model]
    runs = runs.copy()
    runs["seed"] = runs["train.seed"].astype(int)
    for k in keys:
        runs[k] = runs[k].astype(str)

    dup = runs[runs.duplicated(keys + ["seed"], keep=False)]
    if len(dup):
        raise SystemExit(f"duplicate (config, seed) runs; remove the stale ones:\n{dup[keys + ['seed', 'run_dir']]}")

    rows = []
    for cfg_vals, g in runs.groupby(keys, sort=True):
        seeds = sorted(g.seed)
        if tuple(seeds) != SEEDS:
            raise SystemExit(f"config {dict(zip(keys, cfg_vals))} has seeds {seeds}, expected {list(SEEDS)}")
        row = dict(zip(keys, cfg_vals))
        row["n_seeds"] = len(g)
        for m in REPORT:
            row[f"{m}__mean"] = g[m].mean()
            row[f"{m}__std"] = g[m].std(ddof=0)
        row["run_dirs"] = ";".join(g.sort_values("seed").run_dir)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(f"{SELECT}__mean", ascending=False).reset_index(drop=True)
    best = table.iloc[0]
    winner = {
        "model": model,
        "selection_metric": SELECT,
        "config": {k: best[k] for k in keys},
        **{m: {"mean": float(best[f"{m}__mean"]), "std": float(best[f"{m}__std"])} for m in REPORT},
        "seeds": list(SEEDS),
        "note": "single fixed split (SAFE seed-0 unseen tasks); seen_val; selection-optimistic",
    }
    return table, winner


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wandb_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="CSV of every config (mean +- std over seeds)")
    args = ap.parse_args(argv)

    table, winner = select(collect(args.wandb_dir))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    win_path = args.out.with_name(args.out.stem + "_winner.json")
    win_path.write_text(json.dumps(winner, indent=2) + "\n")
    m = winner[SELECT]
    print(f"{winner['model']}: {len(table)} configs; winner {winner['config']} "
          f"{SELECT} = {m['mean']:.4f} +- {m['std']:.4f} -> {win_path}")


if __name__ == "__main__":
    sys.exit(main())
