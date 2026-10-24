import torch


def make_synthetic_records(n=16, t=7, k=4, d=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    records = []
    for i in range(n):
        length = t - (i % 3)
        features = torch.randn(1, length, k, d, generator=generator)
        mask = torch.ones(1, length, dtype=torch.bool)
        labels = torch.tensor([float(i % 2)])
        records.append({"features": features, "mask": mask, "labels": labels, "task_ids": [i % 10], "trajectory_ids": [f"synthetic-{i}"]})
    return records
