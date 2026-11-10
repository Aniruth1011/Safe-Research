> **Archived (2026-11-10).** Written for the retired `akash/collector` pipeline (`akash-ravi/openvla`, tag `archive/collector-2026-11`). Collection now uses the kit in `collection/`. Kept for reference only.

# Pilot report (A6) — gates A8 full collection

Status: **not run yet.** Record every number with date, GPU type and commit (hard rule 7).

| # | Check | How | Result |
|---|---|---|---|
| 1 | Hooks preserve generated tokens | Compare token IDs with and without hooks, same execution setup, ≥200 steps | |
| 2 | Every layer index is correct | Per-block equality vs `hidden_states[k]` (T6 / `tests/gpu/test_layer_mapping.py`) on pilot steps | |
| 3 | `final_norm` capture is correct | Per-token `RMSNorm(raw block_32) ≈ final_norm` before pooling (report max abs/rel error) | |
| 4 | Pooling is correct | Recompute first/last/mean from `token_level_sample`, compare | |
| 5 | Alignment and labels survive save/reload | T, s, env_t, action, token IDs, success line up; `label_failure = 1 - success` | |
| 6 | Class balance | Successes and failures per task; flag any task missing a class | |
| 7 | Runtime and storage | Measured steps/sec and bytes/episode → A8 projection | |
| 8 | Interrupted-job recovery | Kill a job mid-episode, resume, verify no duplicates/gaps; also run two concurrent jobs on one task dir and a requeue | |
| 9 | **Parity with SAFE's `run_libero_eval.py`** | `python experiments/robot/libero/parity_check.py --checkpoint $VLA_CKPT --task_id 0 --n_episodes 2 --workdir $SCRATCH/parity` (fork repo, branch `akash/collector`). Same task, init states, checkpoint, attention implementation, and episode-0 seed. Assert identical per-step action token IDs, executed actions, token probs/entropy, env_t, and per-episode length and success. Greedy ⇒ exact equality expected. | |
| 10 | **Out-of-range action tokens** | Per task: count and rate of env steps where any of the 7 chosen token IDs is outside the tokenizer's true action range [vocab_size − n_bins, vocab_size − 1] = 31744..31999, derived from `model.config` (`meta.n_out_of_range_token_steps`, `meta.out_of_range_token_rate`, per-step `steps.token_out_of_range`). The fork clips the action in those steps. SAFE's prob definition is unchanged; this only quantifies the issue. Flag any task with a non-trivial rate. | |
| 11 | **Tokens at the min edge** (chosen id == vocab_size − n_bins, 31744) | Per task: `sum(meta.n_tokens_at_bin_min_edge)` and rate per token (÷ 7·steps). Valid actions (clipped to +1.0), but SAFE's slice excludes this id, so SAFE's `token_prob` is the max over the other bins, not the chosen token's probability. | |
| 12 | **Tokens at the pad edge** (chosen id == vocab_size, 32000) | Per task: `sum(meta.n_tokens_at_pad_edge)` and rate. A padding token, inside SAFE's slice but not a trained action. | |

Feature-match criteria (checks 3–4): cosine similarity alone is not enough. Report token agreement, relative L2 error, norm ratio, and the downstream detector-score difference where applicable.

Attention implementation must be identical for the whole collection (see compute_notes.md).

## Run the audit

`audit_collection.py` is the **evidence source for checks #5–#12**. It is strictly read-only (never deletes, moves or modifies episode files, locks, temp files or the quarantine), reuses the collector's own validation and lock helpers (`collect_io.py`), and reports dataset statistics only, so it is safe to run on all tasks. Both scripts live in the openvla fork, branch `akash/collector`, and run on CPU (collection env, login node is fine).

```bash
cd /path/to/openvla            # fork, branch akash/collector
ROOT=/scratch1/akashrav/data/<run>      # collection root: task00/ ... task09/, _quarantine/
mkdir -p $HOME/audits          # --json must be OUTSIDE $ROOT

D=$(date +%Y%m%d)
python experiments/robot/libero/audit_collection.py $ROOT [--fast] \
    --num-tasks 10 --n-episodes 50 --expect-seed 7 \
    --json $HOME/audits/pilot_$D.json > $HOME/audits/pilot_$D.md
echo "audit exit=$?"     # 0 = clean, 1 = flagged (--warn-only forces 0)
cat $HOME/audits/pilot_$D.md

# pilot: pass --n-episodes N_PILOT (e.g. 20) so unrun init states are not reported as missing
# projection for the full run from the pilot audit
python experiments/robot/libero/project_collection.py $HOME/audits/pilot_<date>.json \
    --episodes-per-task 50 --partition debug --su-per-gpu-minute 8     # or --partition gpu (2 h)
```

Mapping from the pilot checks to audit output:

| Check | Audit evidence |
|---|---|
| #5 alignment/labels survive save/reload | no `FAILED_VALIDATION` flags (validation checks T, s/env_t, token IDs, action shapes, `label_failure = 1 − success`, filename↔meta match) |
| #6 class balance | per-task success/failure counts; `MISSING_CLASS` flags |
| #7 runtime and storage | per-task steps/s, bytes/episode, features vs inputs bytes, bytes on disk; feed `project_collection.py` |
| #8 interrupted-job recovery | `DUPLICATE_EPISODES`, `MISSING_EPISODES`, `STALE_LOCK`, `STALE_TMP`, `QUARANTINED` after the kill/resume/concurrent/requeue tests |
| #9 SAFE parity | not covered by the audit; use `parity_check.py` |
| #10 out-of-range tokens | token checks table: out-of-range step rate per task (`TOKEN_OUT_OF_RANGE`) |
| #11 tokens at min edge (31744) | token checks table (`TOKEN_MIN_EDGE`) |
| #12 tokens at pad edge (32000) | token checks table (`TOKEN_PAD_EDGE`) |

Also covered: invalid episodes grouped by exception type, coverage (missing/duplicate `init_state_idx`), lengths and `max_steps` agreement (failures should be exactly the episodes at `max_steps`; `SUCCESS_AT_MAX` / `FAILURE_NOT_AT_MAX` are flagged), and metadata consistency across all episodes (checkpoint revision, `attn_implementation`, `center_crop`, `dtype`, code commits: any mix is `INCONSISTENT_META`). Token-rate warning thresholds default to 1e-3 (`--oor-rate-warn`, `--edge-rate-warn`). Paste the audit Markdown (or its path) into the Result column of the table above, with date, GPU type and commit.

`--fast` reads only the `.meta.json` sidecars (quick, but a corrupt `.pt` goes undetected); use full mode for the pilot and `--fast` for routine progress checks during A8.
