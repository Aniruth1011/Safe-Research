from __future__ import annotations

import csv
import json
import os
import tempfile
import fcntl
from datetime import datetime, timezone
from pathlib import Path


INDEX_FIELDS = ["run_id", "experiment_id", "experiment_name", "model", "dataset", "backbone", "backbone_checkpoint", "layers", "seed", "split_version", "status", "start_time", "end_time", "trainable_parameters", "best_epoch", "val_roc_auc", "val_pr_auc", "val_tpr_at_5_fpr", "val_tpr_at_10_fpr", "test_roc_auc", "test_pr_auc", "test_tpr_at_5_fpr", "checkpoint_path", "config_path", "metrics_path", "git_commit", "error_message"]


class RunTracker:
    def __init__(self, output_root="outputs/runs", index_path="experiments/index.csv"):
        self.output_root, self.index_path = Path(output_root), Path(index_path)

    def create(self, run_id, **metadata):
        run_dir = self.output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        row = {field: "" for field in INDEX_FIELDS}
        row.update(run_id=run_id, status="RUNNING", start_time=datetime.now(timezone.utc).isoformat(), **metadata)
        self._update(row)
        return run_dir, row

    def finish(self, row, status, **updates):
        row.update(status=status, end_time=datetime.now(timezone.utc).isoformat(), **updates)
        self._update(row)

    def write_json(self, path, payload):
        path = Path(path)
        fd, temp = tempfile.mkstemp(prefix=path.name, dir=path.parent)
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        os.replace(temp, path)

    def _update(self, row):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.index_path.with_suffix(self.index_path.suffix + ".lock")
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            rows = []
            if self.index_path.exists():
                with self.index_path.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
            rows = [old for old in rows if old.get("run_id") != row["run_id"]] + [{k: str(row.get(k, "")) for k in INDEX_FIELDS}]
            fd, temp = tempfile.mkstemp(prefix="index", dir=self.index_path.parent)
            with os.fdopen(fd, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
                writer.writeheader(); writer.writerows(rows)
            os.replace(temp, self.index_path)
