import argparse, json
import torch
from torch.utils.data import DataLoader
from failure_prob.data.synthetic import make_synthetic_records
from failure_prob.engine.metrics import binary_metrics
from failure_prob.engine.trainer import RecordDataset, collate_records
from failure_prob.models import get_model

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); p.add_argument("--split", default="seen_val", choices=["seen_val", "unseen_test"]); p.add_argument("--synthetic", action="store_true"); args=p.parse_args()
    ckpt=torch.load(args.checkpoint, map_location="cpu", weights_only=False); cfg=ckpt["config"]; model=get_model(ckpt["model"], input_dim=cfg["input_dim"], num_layers=cfg["num_layers"]); model.load_state_dict(ckpt["model_state"])
    if not args.synthetic: raise ValueError("Provide a FeatureStore for real evaluation; synthetic evaluation requires --synthetic")
    records=make_synthetic_records(k=cfg["num_layers"], d=cfg["input_dim"], seed=cfg["seed"]+1); loader=DataLoader(RecordDataset(records), batch_size=4, collate_fn=collate_records); ys=[]; scores=[]
    with torch.no_grad():
        for b in loader:
            out=model(b).failure_logits; ys.extend(b["labels"].repeat_interleave(b["mask"].sum(1)).tolist()); scores.extend(out[b["mask"]].tolist())
    print(json.dumps({"split":args.split, **binary_metrics(ys,scores)}, indent=2))
if __name__ == "__main__": main()
