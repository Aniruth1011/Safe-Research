# SAFE repository audit

Audit date: 2026-10-24

## Findings

| Component | Existing implementation | Decision |
| --- | --- | --- |
| Baseline models | `failure_prob/model/indep.py` and `lstm.py` implement SAFE's historical detector interfaces and training losses. | Reuse unchanged for official reproduction; wrap only for the new normalized contract. |
| Training | `failure_prob/train.py` is Hydra + W&B based and trains the original rollout models. | Leave unchanged and add a local shared engine for cached-feature experiments. |
| Data loading | `failure_prob/data/*.py` load model-specific rollout formats. `data/utils.py` pads rollouts and splits seen/unseen tasks. | Reuse for SAFE runs; add adapters and `feature_store.py` for frozen features. |
| Splitting | `split_rollouts_by_seen_unseen` creates train/seen-validation/unseen splits, but its task lists are supplied by configs/callers. | Preserve semantics and add explicit versioned split manifests for new runs. |
| Metrics | `failure_prob/utils/metrics.py` and `failure_metrics.py` contain SAFE evaluation logic. | Preserve for baseline reproduction; add compact sklearn metrics for cached-feature CLI. |
| Configuration | Hydra configs live under `failure_prob/conf`; batch commands are under `scripts/batch_training`. | Keep Hydra as the original path; new CLI accepts YAML/flags without replacing Hydra. |
| Checkpoints | Original training saves `model_final.ckpt` when enabled and logs through W&B. | Add immutable per-run `best.pt`, `last.pt`, metadata, and atomic index updates. |
| Feature extraction | SAFE consumes rollout files produced by adapted external VLA repositories. | Add metadata-validated storage and document extraction as a separate dependency. |

## Compatibility decisions

The original SAFE code expects `[B,T,D]`, uses `valid_masks` (one for valid), and labels success as `1`. The research framework uses `features` `[B,T,K,D]` when multiple layers are present, `mask` with the same valid-mask meaning, and `labels` with `1` meaning failure. Adapters make this distinction explicit; no original baseline semantics are silently changed.

The official OpenVLA rollout repository is external to this checkout. This repository therefore validates extraction and detector code with synthetic features and cached files, without downloading weights or datasets during tests.
