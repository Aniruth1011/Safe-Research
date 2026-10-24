import argparse
from failure_prob.data.synthetic import make_synthetic_records
from failure_prob.engine.trainer import train_model
from failure_prob.models import get_model

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model", default="single_layer_mlp"); p.add_argument("--layers", nargs="+", type=int, required=True); p.add_argument("--synthetic", action="store_true"); args=p.parse_args()
    if not args.synthetic: raise ValueError("Real probing requires a FeatureStore path; use --synthetic for the CPU fixture")
    for i, layer in enumerate(args.layers):
        history=train_model(get_model(args.model, input_dim=8, num_layers=len(args.layers), layer_index=i), make_synthetic_records(k=len(args.layers), seed=layer), epochs=1); print(f"layer={layer} train_loss={history[-1]:.6f} selection_split=seen_val")
if __name__ == "__main__": main()
