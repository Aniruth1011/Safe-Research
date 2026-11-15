"""Dhyan's fusion experiment rows on top of fusion_runner (seen_val only; unseen_test is never loaded).

  F1 (row 19)  two-layer concatenation     concat_fusion, the selected complementary pair (S2)
  F2 (row 20)  three-layer concatenation   concat_fusion, the selected early/middle/late set (S3)
  F3 (row 21)  static learned weights      static_weight_fusion --proj-dim 64, the selected 3 layers
  F6 (row 25)  number-of-layers ablation   best fusion architecture on the first K = 1, 2, 3, 4 selected layers

Two drivers, both loading the cache ONCE and reusing the records for every configuration:
  * ``hparam_search``: the compact LR / dropout / weight-decay search the protocol asks for inside each leading
    experiment. Every grid point is a full fusion_runner run (all seeds); the winner is the best mean seen_val
    ROC-AUC. Selection and reporting both use seen_val (the protocol's choice), so the winner's numbers are
    optimistic by the selection; the search table keeps every grid point so this stays visible.
  * ``k_ablation``: F6. The ranked layer list comes from selection (S1-S4); K = k uses its first k layers.

CLI: scripts/fusion_experiments.py.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

from failure_prob import fusion_runner as FR

EXPERIMENT_MARKER = "fusion_experiments_output"
DEFAULT_GRID = {"lr": [1e-3, 3e-4], "dropout": [0.0, 0.2], "weight_decay": [0.0, 1e-4]}


def load_once(cfg: FR.RunConfig) -> tuple[dict, dict]:
    """train + seen_val records for cfg.layers, checked. Never requests unseen_test."""
    FR.validate_config(cfg)
    splits, prov = FR.load_synthetic_splits(cfg) if cfg.synthetic else FR.load_cache_splits(cfg)
    FR.check_splits(splits, len(cfg.layers))
    return splits, prov


def first_k(splits: dict, k: int) -> dict:
    """Records restricted to their first k layers (a view; no copy of the features)."""
    out = {}
    for split, recs in splits.items():
        out[split] = [{**r, "features": r["features"][:, :, :k]} for r in recs]
    return out


def pick_layer(splits: dict, pos: int) -> dict:
    """Records restricted to the single layer at position `pos` (a view)."""
    return {split: [{**r, "features": r["features"][:, :, pos:pos + 1]} for r in recs] for split, recs in splits.items()}


def _prepare_root(root: Path, overwrite: bool, kind: str, payload: dict, resume: bool = False) -> None:
    root = root.expanduser()
    marker = root / "experiment.json"
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise FileExistsError(f"{root} is a file or symlink, not an output folder")
    if root.exists() and any(root.iterdir()):
        try:
            old = json.loads(marker.read_text())
        except (OSError, ValueError, AttributeError):
            old = {}
        ours = isinstance(old, dict) and old.get("marker") == EXPERIMENT_MARKER
        if not (overwrite or resume):
            raise FileExistsError(f"{root} exists and is not empty; pass --overwrite to replace it or --resume")
        if not ours:
            raise FileExistsError(f"refusing to reuse {root}: it was not written by fusion_experiments")
        if resume and old.get("kind") != kind:
            raise FileExistsError(f"cannot resume {root}: it holds a {old.get('kind')}, not a {kind}")
        for sub in root.iterdir():  # sub-runs are replaced one by one by fusion_runner's own safe overwrite
            if sub.is_file() and not sub.name.startswith(("experiment.json", "search.csv", "k_curve.csv",
                                                           "best.json", "sheet_rows.csv")):
                raise FileExistsError(f"refusing to reuse {root}: unexpected file {sub.name}")
    root.mkdir(parents=True, exist_ok=True)
    FR._write_json(marker, {"marker": EXPERIMENT_MARKER, "kind": kind, **payload})


def _grid_points(grid: dict) -> list[dict]:
    for name, values in grid.items():
        if name not in ("lr", "dropout", "weight_decay", "hidden_dim", "proj_dim", "epochs"):
            raise ValueError(f"cannot search over {name!r}")
        if not values or len(set(values)) != len(values):
            raise ValueError(f"grid values for {name} must be non-empty and unique, got {values}")
    names = list(grid)
    return [dict(zip(names, combo)) for combo in itertools.product(*(grid[n] for n in names))]


def _tag(point: dict) -> str:
    return "_".join(f"{k}{v:g}" if isinstance(v, float) else f"{k}{v}" for k, v in point.items())


def hparam_search(cfg: FR.RunConfig, grid: dict | None = None, splits: dict | None = None,
                  provenance: dict | None = None, label: str | None = None) -> dict:
    """Every grid point is a full run (all seeds) under <output_dir>/hp_<tag>/. Returns the best point's summary."""
    grid = dict(DEFAULT_GRID if grid is None else grid)
    points = _grid_points(grid)
    for p in points:
        FR.validate_config(replace(cfg, **p), need_source=splits is None)
    root = Path(cfg.output_dir)
    _prepare_root(root, cfg.overwrite, "hparam_search", {"base_config": asdict(cfg), "grid": grid}, cfg.resume)
    finished = cfg.resume and all(_complete(root / f"hp_{_tag(p)}", cfg.seeds) for p in points)
    if splits is None and not finished:
        splits, provenance = load_once(cfg)
    label = label or f"{cfg.model} layers={'/'.join(map(str, cfg.layers))}"
    rows = []
    for p in points:
        sub = replace(cfg, **p, output_dir=str(root / f"hp_{_tag(p)}"), bootstrap=0)  # CI only for the winner
        s = FR.run(sub, splits=splits, provenance=provenance, label=f"{label} [{_tag(p)}]")
        roc = s["metrics"]["roc_auc"]
        rows.append({**p, "roc_auc_mean": roc["mean"], "roc_auc_std": roc["std"],
                     "n_seeds_defined": roc["n_seeds_defined"], "run": sub.output_dir, "summary": s})
    ok = [r for r in rows if not math.isnan(r["roc_auc_mean"])]
    if not ok:
        raise ValueError("seen_val ROC-AUC is undefined for every grid point (single-class seen_val?)")
    best = max(ok, key=lambda r: (r["roc_auc_mean"], -(r["roc_auc_std"] if not math.isnan(r["roc_auc_std"]) else 0)))
    import pandas as pd
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "summary"} for r in rows])
    table["selected"] = table["run"] == best["run"]
    table.sort_values("roc_auc_mean", ascending=False).to_csv(root / "search.csv", index=False)
    chosen = {k: best[k] for k in grid}
    if cfg.bootstrap:
        best["summary"] = FR.add_bootstrap(Path(best["run"]), cfg.bootstrap, cfg.ci)
    sheet = dict(best["summary"]["sheet_row"], Experiment=label)
    sheet["Notes"] = (f"{sheet['Notes']}; hparams {chosen} chosen by mean seen_val ROC-AUC over {len(points)} "
                      f"grid points (selection on seen_val, so optimistic)")
    FR._write_json(root / "best.json", {"selected": chosen, "run": best["run"], "sheet_row": sheet,
                                         "n_grid_points": len(points), "provisional": best["summary"]["provisional"]})
    FR._write_sheet_csv(root / "sheet_rows.csv", [sheet])
    print(f"[search {label}] best {chosen}: seen_val ROC-AUC {100 * best['roc_auc_mean']:.2f}%", flush=True)
    return {"selected": chosen, "summary": best["summary"], "sheet_row": sheet}


def _complete(run_dir: Path, seeds) -> bool:
    return all((run_dir / f"seed{s}" / "metrics.json").is_file() for s in seeds) and (run_dir / "summary.json").is_file()


def k_ablation(cfg: FR.RunConfig, ks=(1, 2, 3, 4), label: str | None = None,
               reference_layer: int | None = None) -> list[dict]:
    """F6: cfg.layers is the RANKED selection (best first); K = k trains on cfg.layers[:k]. One run per k under
    <output_dir>/K<k>/, plus k_curve.csv / sheet_rows.csv. Hyper-parameters are cfg's (pass the searched ones).

    reference_layer: R1's best single layer. When it is not cfg.layers[0], it is also run as K=1 with the same model
    (<output_dir>/ref_L<l>/) and every K gets its change against it, since the sheet compares against the best K=1."""
    ks = sorted(set(int(k) for k in ks))
    if not ks or ks[0] < 1 or ks[-1] > len(cfg.layers):
        raise ValueError(f"ks must lie in 1..{len(cfg.layers)} (the number of ranked layers), got {ks}")
    FR.validate_config(cfg)
    if cfg.init_weights is not None:
        raise ValueError("init_weights is per layer set; it cannot be shared across K (leave it unset for F6)")
    if reference_layer is not None and reference_layer in cfg.layers[1:ks[-1]]:
        raise ValueError(f"reference_layer {reference_layer} is already in the ranked layers; put it first instead")
    use_ref = reference_layer is not None and reference_layer != cfg.layers[0]
    root = Path(cfg.output_dir)
    _prepare_root(root, cfg.overwrite, "k_ablation",
                  {"base_config": asdict(cfg), "ks": ks, "reference_layer": reference_layer}, cfg.resume)
    load_layers = list(cfg.layers[:ks[-1]]) + ([reference_layer] if use_ref else [])
    runs = [(f"K{k}", list(cfg.layers[:k])) for k in ks] + ([(f"ref_L{reference_layer}", [reference_layer])]
                                                              if use_ref else [])
    finished = cfg.resume and all(_complete(root / name, cfg.seeds) for name, _ in runs)
    full, prov = (None, {}) if finished else load_once(replace(cfg, layers=load_layers))
    label = label or f"{cfg.model} K-ablation"
    rows = []
    for name, layers in runs:
        sub = replace(cfg, layers=layers, output_dir=str(root / name))
        p = dict(prov or {})
        if p.get("layer_names"):
            p["layer_names"] = [p["layer_names"][load_layers.index(l)] for l in layers]
        recs = None if full is None else (pick_layer(full, len(load_layers) - 1) if name.startswith("ref")
                                          else first_k(full, len(layers)))
        s = FR.run(sub, splits=recs, provenance=p,
                   label=f"{label} {'reference' if name.startswith('ref') else f'K={len(layers)}'} "
                         f"layers={'/'.join(map(str, layers))}")
        rows.append({"K": len(layers), "run": name, "layers": "/".join(map(str, layers)), **s["sheet_row"]})
    k1 = next((r for r in rows if r["run"] == "K1"), None)
    ref = next((r for r in rows if r["run"].startswith("ref")), None) or k1
    for r in rows:
        r["delta_vs_K1_roc_auc_pp"] = (round(r["Seen Val ROC-AUC (%)"] - k1["Seen Val ROC-AUC (%)"], 2)
                                       if k1 else math.nan)
        r["delta_vs_best_single_roc_auc_pp"] = (round(r["Seen Val ROC-AUC (%)"] - ref["Seen Val ROC-AUC (%)"], 2)
                                                if ref else math.nan)
    import pandas as pd
    pd.DataFrame(rows).to_csv(root / "k_curve.csv", index=False)
    FR._write_sheet_csv(root / "sheet_rows.csv",
                        [{k: v for k, v in r.items() if k not in ("K", "run", "layers") and not k.startswith("delta")}
                         for r in rows])
    print("[F6] K curve:", ", ".join(f"{r['run']}: {r['Seen Val ROC-AUC (%)']}%" for r in rows), flush=True)
    return rows


if __name__ == "__main__":  # pragma: no cover
    sys.exit("use scripts/fusion_experiments.py")
