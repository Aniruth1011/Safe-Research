"""Train a normalized-contract detector on cached or synthetic features."""
import argparse, json, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from failure_prob.data.feature_store import FeatureStore
from failure_prob.data.synthetic import make_synthetic_records
from failure_prob.engine.metrics import binary_metrics
from failure_prob.engine.trainer import RecordDataset, collate_records, train_model
from failure_prob.engine.tracking import RunTracker
from failure_prob.models import get_model
from failure_prob.protocol import TRAIN_SEEDS

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--dataset", default="libero10"); p.add_argument("--layers", nargs="+", type=int, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=list(TRAIN_SEEDS), help="one run per seed; default = team protocol seeds")
    p.add_argument("--feature-root"); p.add_argument("--output-root", default="outputs/runs"); p.add_argument("--experiment-id", default="EXP_001"); p.add_argument("--epochs", type=int, default=2); p.add_argument("--batch-size", type=int, default=4); p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    for seed in args.seeds:
        run_one(p, args, seed)

def run_one(p, args, seed):
    args.seed = seed; torch.manual_seed(seed)
    if args.synthetic:
        records = make_synthetic_records(k=len(args.layers), seed=args.seed)
    elif not args.feature_root:
        p.error("--feature-root is required unless --synthetic is supplied")
    else:
        records = FeatureStore(args.feature_root).load(args.layers)
    d = records[0]["features"].shape[-1]; run_id = f"{args.experiment_id}_{args.model}_seed{seed}_{int(time.time())}"
    tracker = RunTracker(args.output_root); run_dir, row = tracker.create(run_id, experiment_id=args.experiment_id, experiment_name=args.model, model=args.model, dataset=args.dataset, layers=json.dumps(args.layers), seed=args.seed, split_version="v1", backbone="openvla", backbone_checkpoint="cached")
    config = vars(args) | {"input_dim": d, "num_layers": len(args.layers)}; (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True, default=str))
    try:
        model = get_model(args.model, input_dim=d, num_layers=len(args.layers), layer_index=0)
        history = train_model(model, records, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
        loader = DataLoader(RecordDataset(records), batch_size=args.batch_size, collate_fn=collate_records); ys=[]; scores=[]; model.eval()
        with torch.no_grad():
            for batch in loader:
                out = model(batch).failure_logits; mask = batch["mask"]; ys.extend(batch["labels"].repeat_interleave(mask.sum(1)).tolist()); scores.extend(out[mask].tolist())
        metrics = binary_metrics(ys, scores); tracker.write_json(run_dir / "metrics.json", {"train_loss": history, **metrics, "synthetic": args.synthetic})
        torch.save({"model_state": model.state_dict(), "model": args.model, "config": config}, run_dir / "best.pt"); torch.save({"model_state": model.state_dict(), "model": args.model, "config": config}, run_dir / "last.pt")
        tracker.finish(row, "COMPLETED", trainable_parameters=sum(p.numel() for p in model.parameters()), best_epoch=len(history), val_roc_auc=metrics["roc_auc"], val_pr_auc=metrics["pr_auc"], checkpoint_path=str(run_dir / "best.pt"), config_path=str(run_dir / "config.json"), metrics_path=str(run_dir / "metrics.json"))
        print(run_dir)
    except Exception as exc:
        tracker.finish(row, "FAILED", error_message=repr(exc)); raise

if __name__ == "__main__": main()
