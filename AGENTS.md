# AGENTS.md

## Section A — Permanent repository instructions

This repository extends SAFE incrementally. Preserve the original implementations under `failure_prob/model`, `failure_prob/data`, `failure_prob/train.py`, and the Hydra configs. New cached-feature research code belongs under `failure_prob/models`, `failure_prob/data/feature_store.py`, `failure_prob/engine`, and `scripts/`.

### Architecture and conventions

- Use the shared `FailureDetector` contract: `forward(batch)` returns an `Output` containing failure logits, prediction level, and mask semantics.
- Normalized cached-feature batches use `features [B,T,K,D]` (or a structured per-layer dictionary), `mask [B,T]`, failure `labels`, `task_ids`, and `trajectory_ids`.
- Padding masks are `True` for valid timesteps. Models must not use padded values for predictions or loss.
- Model names are registered explicitly with `@register_model`; imports must not load VLA weights or initialize GPUs.
- Training and model selection use train and `seen_val` only. `unseen_test` is final evaluation only and cannot affect layers, thresholds, hyperparameters, or checkpoints.
- Keep feature extraction separate from detector training. Cached features must carry backbone checkpoint, layer indexing, pooling, dtype, dataset, split version, and trajectory metadata.
- Do not commit datasets, cached features, checkpoints, credentials, personal paths, or synthetic metrics as official results.

### Lifecycle and required implementation procedure

Use `REQUESTED -> IMPLEMENTING -> IMPLEMENTED -> VALIDATED -> TRAINED -> EVALUATED`. A coding agent implementing this experiment must read this file, inspect existing implementations, create or reuse the model, register it, add config/tests/docs/commands, run local validation, and update the manifest. `TRAINED` and `EVALUATED` require actual successful commands; tests or synthetic smoke runs do not justify those statuses.

AGENTS.md is an instruction file for the coding agent, not a runtime parser. The agent translates the active experiment into Python and configuration. Training does not consult AGENTS.md.

### Mandatory checks

Run `python -m pytest -q` and a CPU synthetic smoke test before claiming validation. Check registry lookup, forward shapes, masks, backward pass, save/load, split isolation, metadata validation, and atomic experiment tracking. Use the original Hydra/W&B commands when reproducing SAFE baselines.

## Section B — ACTIVE_EXPERIMENT

```yaml
experiment:
  id: EXP_001
  name: adaptive_gated_fusion
  status: requested
  description: >
    Implement a gated multi-layer fusion detector with per-layer projections
    into a common embedding and per-timestep learned fusion weights.
  model_type: gated_fusion
  layers:
    selection: explicit
    values: [8, 16, 24, 32]
  baseline:
    - safe_mlp
    - safe_lstm
    - best_single_layer
    - parameter_matched_final_layer
  requirements:
    - Support batched temporal inputs
    - Support padded-sequence masks
    - Return failure logits
    - Work with the shared training pipeline
  priority: must_do
```
