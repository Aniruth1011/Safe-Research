"""scripts/collect_safe_runs.py: SAFE's seen_val selection rule over offline wandb run files (fake runs)."""
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import collect_safe_runs as C  # noqa: E402


def fake_run(root: Path, idx: int, lr: float, lam: float, seed: int, auc: float, extra_summary=None):
    files = root / "wandb" / f"offline-run-20261114_0000{idx:02d}-x{idx}" / "files"
    files.mkdir(parents=True)
    cfg = {
        "dataset": {"value": {"name": "openvla_cache", "layer": 32, "token_idx_rel": "mean"}},
        "model": {"value": {"name": "indep", "lr": lr, "lambda_reg": lam, "batch_size": 64, "n_epochs": 1000}},
        "train": {"value": {"seed": seed}},
        "_wandb": {"value": {}},
    }
    (files / "config.yaml").write_text(yaml.safe_dump(cfg))
    summary = {C.SELECT: auc, "falert_early_prc_auc/model_val_seen": auc - 0.1,
               "roc_auc/model_val_seen_tq1.0": auc - 0.2, **(extra_summary or {})}
    (files / "wandb-summary.json").write_text(json.dumps(summary))


def _grid(root, skip=None, extra=None):
    i = 0
    for lr, base in ((1e-3, 0.70), (1e-4, 0.80)):
        for seed in (0, 1, 2):
            if skip == (lr, seed):
                continue
            fake_run(root, i, lr, 1e-2, seed, base + 0.01 * seed, extra if i == 0 else None)
            i += 1


def test_selects_best_mean_over_seeds(tmp_path):
    _grid(tmp_path)
    out = tmp_path / "sel.csv"
    C.main([str(tmp_path / "wandb"), "--out", str(out)])
    win = json.loads((tmp_path / "sel_winner.json").read_text())
    assert win["config"]["model.lr"] == "0.0001"
    assert win[C.SELECT]["mean"] == pytest.approx(0.81)
    assert win["seeds"] == [0, 1, 2]


def test_missing_seed_fails(tmp_path):
    _grid(tmp_path, skip=(1e-3, 2))
    with pytest.raises(SystemExit, match="seeds"):
        C.main([str(tmp_path / "wandb"), "--out", str(tmp_path / "sel.csv")])


def test_unseen_key_fails(tmp_path):
    _grid(tmp_path, extra={"falert_early_roc_auc/model_val_unseen": 0.9})
    with pytest.raises(SystemExit, match="unseen"):
        C.main([str(tmp_path / "wandb"), "--out", str(tmp_path / "sel.csv")])


def test_reads_real_offline_wandb_runs(tmp_path, monkeypatch):
    """wandb offline runs keep config + metrics only in run-<id>.wandb; the collector must read those."""
    wandb = pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    for seed in (0, 1, 2):
        run = wandb.init(project="t", dir=str(tmp_path), reinit=True, config={
            "dataset": {"name": "openvla_cache", "layer": 32, "token_idx_rel": "mean"},
            "model": {"name": "lstm", "lr": 3e-4, "lambda_reg": 0.1, "batch_size": 64, "n_epochs": 1000},
            "train": {"seed": seed}})
        wandb.log({C.SELECT: 0.5, "falert_early_prc_auc/model_val_seen": 0.4, "roc_auc/model_val_seen_tq1.0": 0.3})
        wandb.log({C.SELECT: 0.6 + 0.01 * seed, "falert_early_prc_auc/model_val_seen": 0.5,
                   "roc_auc/model_val_seen_tq1.0": 0.4})   # final epoch wins
        run.finish()
    out = tmp_path / "sel.csv"
    C.main([str(tmp_path), "--out", str(out)])
    win = json.loads((tmp_path / "sel_winner.json").read_text())
    assert win["model"] == "lstm" and win[C.SELECT]["mean"] == pytest.approx(0.61)
