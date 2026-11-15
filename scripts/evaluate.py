"""Re-score a checkpoint written by scripts/train.py.

    # B1: episode-level metrics (B4) on the cache's seen_val split, with the layers/pool/normalize the run used
    python scripts/evaluate.py --checkpoint outputs/runs/<run>/best.pt [--cache /scratch1/akashrav/safe_cache/v1]

    # final locked benchmark only (Exp 37): unseen_test needs an explicit flag
    python scripts/evaluate.py --checkpoint ... --split unseen_test --allow-unseen

    # legacy synthetic check
    python scripts/evaluate.py --checkpoint ... --synthetic
"""
import argparse, json
import torch
from torch.utils.data import DataLoader
from failure_prob.data.cache_splits import load_split_records
from failure_prob.data.synthetic import make_synthetic_records
from failure_prob.engine.metrics import binary_metrics
from failure_prob.engine.scoring import evaluate_records
from failure_prob.engine.trainer import RecordDataset, collate_records
from failure_prob.models import get_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True); p.add_argument("--split", default="seen_val", choices=["seen_val", "unseen_test"])
    p.add_argument("--cache", help="cache version folder; defaults to the one recorded in the checkpoint's config")
    p.add_argument("--allow-unseen", action="store_true", help="required for --split unseen_test (final benchmark only)")
    p.add_argument("--synthetic", action="store_true"); p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False); cfg = ckpt["config"]
    model = get_model(ckpt["model"], input_dim=cfg["input_dim"], num_layers=cfg["num_layers"]); model.load_state_dict(ckpt["model_state"])
    if args.synthetic:
        records = make_synthetic_records(k=cfg["num_layers"], d=cfg["input_dim"], seed=cfg["seed"] + 1); loader = DataLoader(RecordDataset(records), batch_size=4, collate_fn=collate_records); ys = []; scores = []
        with torch.no_grad():
            for b in loader:
                out = model(b).failure_logits; ys.extend(b["labels"].repeat_interleave(b["mask"].sum(1)).tolist()); scores.extend(out[b["mask"]].tolist())
        print(json.dumps({"split": "synthetic", **binary_metrics(ys, scores)}, indent=2)); return
    cache = args.cache or cfg.get("cache")
    if not cache:
        p.error("no --cache given and the checkpoint was not trained with --cache")
    if args.split == "unseen_test" and not args.allow_unseen:
        p.error("unseen_test is locked until the final benchmark (Exp 37); add --allow-unseen only then")
    records = load_split_records(cache, args.split, cfg["layers"], pool=cfg.get("pool", "mean"),
                                 normalize=cfg.get("normalize", False), allow_unseen=args.allow_unseen)
    print(json.dumps({"split": args.split, "cache": str(cache), **evaluate_records(model, records, args.batch_size)}, indent=2))


if __name__ == "__main__": main()
