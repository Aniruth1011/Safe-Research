"""Small, metadata-first store for frozen VLA features."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch


@dataclass(frozen=True)
class FeatureMetadata:
    backbone: str
    checkpoint: str
    dataset: str
    split_version: str
    layers: tuple[int, ...]
    feature_type: str = "hidden_state"
    pooling: str = "documented"
    dtype: str = "float32"
    layer_indexing: str = "zero_based"
    num_layers: int | None = None


class FeatureStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.data_path = self.root / "features.pt"
        self.metadata_path = self.root / "metadata.json"

    def save(self, records: list[dict], metadata: FeatureMetadata) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = []
        for record in records:
            required = {"features", "mask", "labels", "task_ids", "trajectory_ids"}
            missing = required - set(record)
            if missing:
                raise ValueError(f"Feature record missing fields: {sorted(missing)}")
            payload.append({k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in record.items()})
        torch.save(payload, self.data_path)
        self.metadata_path.write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n")

    def metadata(self) -> FeatureMetadata:
        if not self.metadata_path.exists():
            raise FileNotFoundError(self.metadata_path)
        raw = json.loads(self.metadata_path.read_text())
        raw["layers"] = tuple(raw["layers"])
        return FeatureMetadata(**raw)

    def validate_layers(self, layers: list[int] | tuple[int, ...]) -> None:
        meta = self.metadata()
        missing = sorted(set(layers) - set(meta.layers))
        if missing:
            raise ValueError(f"Requested feature layers {missing} are missing; available={list(meta.layers)}")
        if meta.num_layers is not None and any(layer < 0 or layer >= meta.num_layers for layer in layers):
            raise ValueError(f"Layer IDs outside declared architecture depth {meta.num_layers}: {layers}")

    def load(self, layers: list[int] | None = None) -> list[dict]:
        records = torch.load(self.data_path, map_location="cpu", weights_only=False)
        if layers is None:
            return records
        self.validate_layers(layers)
        indices = [self.metadata().layers.index(layer) for layer in layers]
        for record in records:
            features = record["features"]
            if features.ndim != 4:
                raise ValueError("Stored features must be [N,T,K,D]")
            record["features"] = features[:, :, indices]
        return records
