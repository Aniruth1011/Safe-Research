# Compute notes

## T0 — Repo sanity (laptop, CPU)
- Date: 2026-11-06 · Host: macOS (Darwin, Apple silicon laptop, no GPU) · Python 3.10.22 (uv venv, outside repo) · torch 2.14.1 (CPU)
- Base commit: 88d9c50 (main) + setup.py commit on `akash/extraction`
- `uv pip install -e .` (deps now declared in setup.py): OK
- `python -m pytest -q`: **4 passed, 0 failed** (11.7 s; synthetic data only, no downloads)
- `python scripts/list_models.py`: concat_fusion, gated_fusion, residual_fusion, safe_lstm, safe_mlp, single_layer_mlp

### Known issue
Python ≥3.11 fails on the TrainConfig() dataclass default (`failure_prob/conf/__init__.py:284`, `train: TrainConfig = TrainConfig()` -> "mutable default ... not allowed"); fix = `field(default_factory=TrainConfig)`, pending team approval. Not applied (SAFE baseline file). On 3.12 pytest fails at collection with this error.

## Environment separation on CARC
The detector repo (this repo) must not share an environment with the OpenVLA collection env, which pins `torch==2.2.0` / `transformers==4.40.1` (`vla-safe/openvla`). If they do share one env, install this repo with `pip install -e . --no-deps` so `setup.py` does not upgrade or replace the pinned torch. Preferred: two separate conda envs (collection vs. detector training/probing).

## Collection metadata must record the attention implementation
Hidden states differ numerically between `flash_attention_2`, `sdpa` and `eager`. The collector's `meta` must therefore include `attn_implementation` (default `flash_attention_2`, matching SAFE `run_libero_eval.py:72`), and the full collection (pilot through A8) must use one implementation throughout. T6 records the value in `t6_report.json`. `VLA_NORM_ATOL` stays at 0.0: re-applying the same norm module to the same input is expected to be bitwise identical; if it is not, report the numbers rather than loosening the tolerance.

## Intentional deviations from SAFE's rollout code (collector)
- **Per-episode seeding.** SAFE (`run_libero_eval.py`) seeds once at startup, so skipping already-collected episodes on resume would shift the RNG state. The collector instead re-seeds every episode with `sha256(f"{base_seed}:{task_id}:{init_state_idx}")[:4] % (2**31-1)` and records it as `meta.episode_seed`. Decoding is greedy and the init state is fixed, so actions are not expected to depend on RNG state; the parity check (pilot_report.md #9) verifies this empirically.
- **Resume safety.** Bad files are quarantined to `{root}/_quarantine/` (never deleted); temp files are per-job (`*.pt.{SLURM_JOB_ID|pid}.tmp`); episodes are claimed with O_EXCL lock files.
- **Token probabilities** are stored as `token_probs [T,7]` and `token_entropy [T,7]`, computed by the fork's `compute_token_uncertainty_metrics` (softmax over the action-bin slice of the logits; "prob" = max bin probability, equal to the chosen token's probability under greedy decoding when the argmax is an action-bin token). Needed by SAFE's token-uncertainty baselines and the baseline rows.

## SAFE token-uncertainty slice is off by one (kept as-is for baseline parity)
- True action-token range (`prismatic/vla/action_tokenizer.py`: `token = vocab_size - digitize(action, bins)`, digitize returns 1..n_bins): **[vocab_size − n_bins, vocab_size − 1] = 31744..31999** (vocab_size = `text_config.vocab_size − pad_to_multiple_of` = 32000, n_bins = 256).
- SAFE's slice, `vla-safe/openvla` `experiments/robot/unc_utils.py:29` (`logits[:, vocab_size - n_action_bins + 1 : vocab_size + 1]`): **31745..32000**. It omits 31744 (actions clipped to +1.0) and includes 32000 (a padding token, not a trained action).
- Effect on `token_probs` / `token_entropy` (we keep SAFE's computation unchanged): (a) if the chosen token is 31744, the stored "prob" is the max over the *other* bins, not the chosen token's probability, and its mass is excluded from the softmax; (b) the softmax includes a pad-token logit, which slightly changes probabilities and entropy for every step; (c) if the chosen token is 32000 (pad), the action is clipped by the decoder. The collector records `n_tokens_at_bin_min_edge` (chosen == 31744), `n_tokens_at_pad_edge` (chosen == 32000) and `n_out_of_range_token_steps` (any chosen id outside 31744..31999, derived from `model.config`) per episode to quantify this.

## What the tiny-model tests prove (and what they don't)
`tests/test_feature_capture.py` (fork, branch `akash/collector`; needs `transformers==4.40.1`, runs on CPU) drives `FeatureCapture` through a random 4-layer, hidden-64 `LlamaForCausalLM` with `generate(max_new_tokens=7, output_hidden_states=True, return_dict_in_generate=True)` and KV cache, in bf16 and fp32.
- **Proves (hook and index logic):**
  - the `block_k` layer hook equals `hidden_states[j][k][0,-1]` bit-exactly for k = 1..L−1.
  - Re-applying the norm module to the final-norm pre-hook input equals `hidden_states[j][L][0,-1]` bit-exactly.
  - raw block L comes from the pre-hook and equals `layers[L-1]`'s output.
  - pooled first/last/mean are exactly recomputable from `token_level_sample`.
  - `layer_mean_all` is the per-token float32 mean of the raw blocks, cast to bf16, then pooled.
  - generated tokens are identical with and without hooks, and detaching restores the model.
  - buffers do not leak across consecutive actions.
  - the `verify` check fires on mismatched outputs.
  - module location works through a `language_model` wrapper.
- **Does NOT prove:**
  - OpenVLA's multimodal input path (image patch embeddings prepended to the sequence, `prepare_inputs_for_generation`, the prefill/step split in `PrismaticForConditionalGeneration.forward`).
  - the 32-layer / 4096-dim shapes, real weights, or bf16 numerics at that scale.
  - flash-attention-2 (or sdpa/eager) numerics, which differ between implementations.
  - CUDA behaviour (device buffers, sync, memory).
  - that the real RMSNorm re-application is bitwise identical on GPU (reported, not assumed, by T6).
- **T6 on the GPU (`tests/gpu/test_layer_mapping.py`) still has to pass**, and so does the A6 pilot, before any full collection.

## Sidecar metadata
Each episode `ep{NNN}_seed{S}.pt` gets an atomic `ep{NNN}_seed{S}.meta.json` (written after the `.pt` rename; regenerated on resume if missing or inconsistent): the episode meta plus `pt_bytes`, `feature_bytes`, `input_bytes`, `token_level_bytes`. `audit_collection.py --fast` reads only sidecars (cannot detect a corrupt `.pt`); full mode verifies sidecar == `.pt` meta and flags `SIDECAR_MISMATCH` / `SIDECAR_MISSING` / `SIDECAR_ORPHAN`.

## T1–T7
Not run yet. Require a session with CARC access.
