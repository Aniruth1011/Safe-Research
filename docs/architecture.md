# Research framework architecture

Original SAFE code remains under `failure_prob/model`, `failure_prob/data`, and `failure_prob/utils`. The additive framework uses `failure_prob/models` for registered detectors, `failure_prob/data/feature_store.py` for immutable cached features, and `failure_prob/engine` for training, evaluation, checkpointing, and index tracking. `scripts/train.py` and `scripts/evaluate.py` are lightweight entry points for the normalized contract; `failure_prob/train.py` remains the official SAFE reproduction entry point.
