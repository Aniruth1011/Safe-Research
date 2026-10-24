from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


class RecordDataset(Dataset):
    def __init__(self, records): self.records = records
    def __len__(self): return len(self.records)
    def __getitem__(self, i): return self.records[i]


def collate_records(records):
    max_t = max(r["features"].shape[1] for r in records)
    b, k, d = len(records), records[0]["features"].shape[2], records[0]["features"].shape[3]
    x = torch.zeros(b, max_t, k, d); mask = torch.zeros(b, max_t, dtype=torch.bool); labels = torch.zeros(b)
    task_ids, trajectory_ids = [], []
    for i, r in enumerate(records):
        t = r["features"].shape[1]; x[i, :t] = r["features"][0]; mask[i, :t] = r["mask"][0]; labels[i] = r["labels"].reshape(-1)[0]; task_ids += r.get("task_ids", [None]); trajectory_ids += r.get("trajectory_ids", [None])
    return {"features": x, "mask": mask, "labels": labels, "task_ids": task_ids, "trajectory_ids": trajectory_ids}


def train_model(model, records, epochs=2, batch_size=4, lr=1e-3, seed=0):
    torch.manual_seed(seed); loader = DataLoader(RecordDataset(records), batch_size=batch_size, shuffle=True, collate_fn=collate_records)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr); history = []
    for _ in range(epochs):
        model.train(); losses = []
        for batch in loader:
            loss = model.loss(batch); optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(loss.item())
        history.append(sum(losses) / len(losses))
    return history
