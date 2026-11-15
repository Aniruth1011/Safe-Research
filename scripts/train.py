"""Train a normalized-contract detector on cached or synthetic features.

    # B1: the real protocol. Trains on the cache's train split, scores seen_val at the episode level (B4).
    python scripts/train.py --model single_layer_mlp --layers 16 --cache /scratch1/akashrav/safe_cache/v1 [--pool mean] [--normalize]

    # legacy inputs: a single FeatureStore, or synthetic records. These score the data they trained on, so their
    # numbers are NOT validation results and are not written to the val_* columns of experiments/index.csv.
    python scripts/train.py --model single_layer_mlp --layers 16 --feature-root <store>
    python scripts/train.py --model single_layer_mlp --layers 16 --synthetic
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from failure_prob.data.cache_splits import load_split_records
from failure_prob.data.feature_store import FeatureStore
from failure_prob.data.synthetic import make_synthetic_records
from failure_prob.engine.metrics import binary_metrics
from failure_prob.engine.scoring import evaluate_records
from failure_prob.engine.trainer import RecordDataset, collate_records, train_model
from failure_prob.engine.tracking import RunTracker
from failure_prob.models import get_model
from failure_prob.protocol import TRAIN_SEEDS

NOT_VALIDATION = "train (NOT validation: these records were also used for training; use --cache)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--dataset", default="libero10"); p.add_argument("--layers", nargs="+", type=int, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=list(TRAIN_SEEDS), help="one run per seed; default = team protocol seeds")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--cache", help="shared cache version folder (e.g. /scratch1/akashrav/safe_cache/v1): train on train, score seen_val")
    src.add_argument("--feature-root"); src.add_argument("--synthetic", action="store_true")
    p.add_argument("--pool", default="mean", choices=["mean", "first", "last"], help="with --cache: pooling over the 7 action tokens")
    p.add_argument("--normalize", action="store_true", help="with --cache: z-score each layer with train-split statistics")
    p.add_argument("--output-root", default="outputs/runs"); p.add_argument("--experiment-id", default="EXP_001"); p.add_argument("--epochs", type=int, default=2); p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    if not (args.cache or args.feature_root or args.synthetic):
        p.error("one of --cache, --feature-root or --synthetic is required")
    if (args.pool != "mean" or args.normalize) and not args.cache:
        p.error("--pool/--normalize only apply with --cache")
    data = load_data(args)
    for seed in args.seeds:
        run_one(args, seed, data)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(args):
    """(train_records, eval_records or None, provenance). Loaded once and shared by every seed."""
    if args.cache:
        cache = Path(args.cache)
        train = load_split_records(cache, "train", args.layers, pool=args.pool, normalize=args.normalize)
        val = load_split_records(cache, "seen_val", args.layers, pool=args.pool, normalize=args.normalize)
        prov = {"cache": str(cache), "cache_manifest_sha256": _sha256(cache / "manifest.json"),
                "splits_json_sha256": _sha256(cache / "splits.json"), "pool": args.pool, "normalize": args.normalize,
                "train_episodes": len(train), "eval_split": "seen_val", "eval_episodes": len(val)}
        return train, val, prov
    if args.feature_root:
        return FeatureStore(args.feature_root).load(args.layers), None, {"feature_root": args.feature_root}
    return None, None, {"synthetic": True}


def legacy_metrics(model, records, batch_size):
    loader = DataLoader(RecordDataset(records), batch_size=batch_size, collate_fn=collate_records); ys = []; scores = []; model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model(batch).failure_logits; mask = batch["mask"]; ys.extend(batch["labels"].repeat_interleave(mask.sum(1)).tolist()); scores.extend(out[mask].tolist())
    return {**binary_metrics(ys, scores), "eval_split": NOT_VALIDATION, "metric_level": "timestep"}


def run_one(args, seed, data):
    train_records, eval_records, prov = data
    torch.manual_seed(seed)
    records = train_records if train_records is not None else make_synthetic_records(k=len(args.layers), seed=seed)
    d = records[0]["features"].shape[-1]; run_id = f"{args.experiment_id}_{args.model}_seed{seed}_{int(time.time())}"
    tracker = RunTracker(args.output_root)
    run_dir, row = tracker.create(run_id, experiment_id=args.experiment_id, experiment_name=args.model, model=args.model, dataset=args.dataset, layers=json.dumps(args.layers), seed=seed, split_version="v1" if args.cache else "", backbone="openvla", backbone_checkpoint="cached")
    config = {k: v for k, v in vars(args).items() if k != "seeds"} | {"seed": seed, "input_dim": d, "num_layers": len(args.layers), **prov}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True, default=str))
    try:
        model = get_model(args.model, input_dim=d, num_layers=len(args.layers), layer_index=0)
        history = train_model(model, records, epochs=args.epochs, batch_size=args.batch_size, seed=seed)
        if eval_records is not None:
            metrics = evaluate_records(model, eval_records, batch_size=args.batch_size)
            metrics["eval_split"] = "seen_val"
            val = {"val_roc_auc": metrics["roc_auc"], "val_pr_auc": metrics["pr_auc"], "val_tpr_at_5_fpr": metrics["tpr_at_5_fpr"]}
        else:
            metrics = legacy_metrics(model, records, args.batch_size)
            val = {}
            print(f"WARNING: metrics for {run_id} are on the training data ({NOT_VALIDATION}); "
                  "not written to the val_* columns", file=sys.stderr)
        tracker.write_json(run_dir / "metrics.json", {"train_loss": history, **metrics, "synthetic": bool(args.synthetic)})
        ckpt = {"model_state": model.state_dict(), "model": args.model, "config": config}
        torch.save(ckpt, run_dir / "best.pt"); torch.save(ckpt, run_dir / "last.pt")
        tracker.finish(row, "COMPLETED", trainable_parameters=sum(p.numel() for p in model.parameters()), best_epoch=len(history), checkpoint_path=str(run_dir / "best.pt"), config_path=str(run_dir / "config.json"), metrics_path=str(run_dir / "metrics.json"), **val)
        print(run_dir)
    except Exception as exc:
        tracker.finish(row, "FAILED", error_message=repr(exc)); raise


if __name__ == "__main__": main()
