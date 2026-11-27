# Aniruth experiment suite (`failure_prob.suite`), decided settings

One pipeline for task-plan rows 5–10, 22–24, 28–33, 36 and 37 on the shared
cache `/scratch1/akashrav/safe_cache/v1`. Results are reported as **"single fixed split
(SAFE seed-0 unseen tasks)"**. Earlier table numbers came from one seed and one
hyperparameter point (seed 0, lr 1e-4, reg 1e-3). Earlier fusion numbers came from
20-epoch screening. All of them are superseded by this suite.

## Shared probe / training spec (B6)

| Item | Value | Source |
|---|---|---|
| Features | mean pooling over 7 action tokens, **raw** (no normalization) | SAFE (`normalize_hidden_states: False`); user decision 2026-11-21, deviates from the team handoff's z-score |
| Training horizon | **full rollouts** (failures run 520 steps) | SAFE; user decision 2026-11-21 |
| Evaluation horizon | every metric scores only the first `min(length, task_min_step)` steps | SAFE "by earliest stop"; team handoff (length leaks the label) |
| Epochs / batch / width / optimizer | 1000 / 64 trajectories / 256 / Adam, no LR decay, no clipping | SAFE LIBERO batch script + conf |
| Grid | lr {1e-4, 3e-4, 1e-3} × reg {1e-3, 1e-2, 1e-1, 1} | SAFE |
| Seeds | 0, 1, 2 (initialization + batch order; not new task splits) | team |
| Regularizer | reg × Σ (non-bias params)² | SAFE `compute_regularization_loss` |
| Class weights | w = N/(count+1) per class, from train | SAFE `RolloutDataset` |
| Per-step models | cumulative sigmoid score, SAFE L1-style loss (`IndepModel`) | SAFE MLP |
| Sequence models | per-step sigmoid, per-step BCE (`LstmModel`, cumsum=False) | SAFE LSTM |
| Rollout score | max of step score over the prefix (= last value for cumulative) | SAFE "by earliest stop" |
| Checkpoint | final epoch (no early stopping) | SAFE |
| Selection | config with the best **mean over 3 seeds** of final-epoch seen-val ROC-AUC | — |
| Metrics | episode ROC-AUC (primary), PR-AUC (trapezoid) and AP, TPR at FPR ≤ 5%, episode bootstrap 95% CI | team |

Run v2 (`/scratch1/$USER/safe_suite/v2`) uses these SAFE-faithful settings. Run v1 (z-scored features,
training cut at `task_min_step`) was stopped after ~1.5 h and is not used. Remaining deviations from SAFE are our
own 500 rollouts, one fixed outcome-balanced split (SAFE averages 3 task splits), and fixed mean pooling.
Raw early layers have a much smaller scale (L01 std 0.04 vs L32 0.84), which z-scoring would have removed.

Equivalence to the original SAFE `IndepModel` and `LstmModel` (scores, loss, regularizer) is
unit-tested in `tests/test_suite.py`. All 12 grid points of a seed train side by side on one
GPU. They share the initialization and batch order, so this is the same as running 12 separate
SAFE processes with that seed.

## Experiments

| Exp | Row | Model | Input | Compared with |
|---|---|---|---|---|
| P_L04 … P_L24 | 5–10 | SAFE MLP probe | L4, 8, 12, 16, 20, 24 | P_L32 |
| P_L32 / P_L33 | controls | SAFE MLP | post-norm final (SAFE feature) / raw final block | — |
| P_L01 … P_L31 | sweep | SAFE MLP | every raw depth | pool used to choose layers |
| X2_LSTM | 28 | SAFE LSTM | L32, full causal history | P_L32 |
| F3_STATIC | F4 control | learned static softmax layer weights | 3 selected layers | F4 |
| F4_GATED | 22 | per-step input-conditioned softmax gates (gate MLP 768→64→3) | 3 selected | F3 |
| F5_RESIDUAL | 23 | anchor projection + 2 ReZero nonlinear complement branches | anchor + 2 complements | best single probe |
| X1_ATTN | 24 | 4-head attention across the 3 layer tokens per step (not across time) | 3 selected | F4 |
| K1_SIZEMATCH | N3 control | L32 SAFE MLP, width matched to winner params (≤5%) | L32 | winner |
| X2_GRU | 29 | GRU(256) | frozen winner embedding | winner |
| X2_DELTA | 30 | GRU(256) | [embedding, first difference], d₀ = 0 | X2_GRU |
| X2_CAUSAL | 31 | 1 pre-norm block, 4 heads, causal mask, sinusoidal positions | frozen winner embedding | X2_GRU |
| X3_PROMPT | 32 | 8 learned prompts, input-conditioned mixture added to z, SAFE MLP head | frozen winner embedding | X3_PROMPT_CTRL |
| X3_PROMPT_CTRL | 32 control | residual bottleneck adapter, prompt-matched params | frozen winner embedding | X3_PROMPT |
| X4_LDROP | 33 | winner architecture retrained, per-timestep whole-layer dropout p ∈ {0.1, 0.2, 0.3} (p is part of the grid) | winner layers | winner |

Fusion models project each layer D→256 and fuse. They then apply ReLU and Linear(256,1) with the
cumulative SAFE objective. With one layer this is exactly the SAFE probe, so any gain comes from
the fusion itself. The **embedding** consumed by rows 29–32 is that ReLU(fused) vector, from the
winner's chosen config. The weights are frozen, and seed s uses the winner's seed-s checkpoint.
Reports list trainable and total params.

**Layer selection** (`select-layers`) uses seen_val only:
- anchor = best raw-depth probe (pool L1–L31 and L33; post-norm L32 excluded);
- two complements are added greedily, each one maximizing the seen-val ROC-AUC of the averaged,
  z-scored rollout scores of the chosen probes;
- the layer order is [anchor, c1, c2]. F3/F4/X1 use the same 3 layers, so they compare on equal
  footing.
- A teammate handoff overrides it: `select-layers --handoff file.json` with
  `{"layers":[a,c1,c2],"source":"..."}`. Run this before stage B.

**Winner** (`select-winner`) is the best seed-mean seen-val ROC-AUC among F3/F4/F5/X1.

**N1 (row 36)** uses the saved seen-val curves of every selected experiment. It produces:
- pooled and per-task metrics at 25/50/75/100% of each episode's horizon, at index
  `round((h-1)·f)`, scored both as max-so-far and as the score at that step;
- a check on seed 0 that prefix-only inputs reproduce the scores;
- a seed-val threshold at 5% FPR, per seed;
- params, and CPU latency (1 thread, B=1, T=200, ms per step, including the frozen fusion);
- `results_sheet.csv`, with one row per experiment in the sheet's columns.

**N3 (row 37)** works in two steps:
- `freeze` records finalists, configs, checkpoint SHA-256s, source hash and seen-val thresholds.
- `n3 --allow-unseen` verifies those, checks there are exactly 3 unseen tasks, takes an exclusive
  lock (it can never run twice), then scores unseen_test once.
- The unseen files are readable only by Akash, so they have to grant access or run this step.

## Commands

```bash
bash scripts/suite/setup_env.sh                 # once: CUDA venv in ~/envs/safe-suite
export SUITE_PYTHON=$HOME/envs/safe-suite/bin/python
$SUITE_PYTHON -m pytest -q tests/test_suite.py  # unit tests
$SUITE_PYTHON -m failure_prob.suite smoke       # synthetic end-to-end (SYNTHETIC_ONLY)
$SUITE_PYTHON -m failure_prob.suite plan
bash scripts/suite/submit.sh                    # dry run: prints the 4 dependent sbatch commands
bash scripts/suite/submit.sh --submit           # A (34-task array) -> B (4) -> C (7) -> N1
$SUITE_PYTHON -m failure_prob.suite status
# later, after the team agrees on finalists:
$SUITE_PYTHON -m failure_prob.suite freeze --agreed-by "Aniruth, Rhea, Dhyan, Adarsh, Akash"
$SUITE_PYTHON -m failure_prob.suite n3 --allow-unseen     # once, with unseen read access
```

Any single experiment can also run in an interactive allocation with
`python -m failure_prob.suite train F4_GATED`. Training resumes from a group checkpoint every 50
epochs, so a killed job can be resubmitted.

Storage:
- `SUITE_RUN` (default `/scratch1/$USER/safe_suite/v1`) holds every grid point. It is
  re-creatable, and scratch is not backed up.
- `SUITE_OUT` (default `outputs/suite/`) holds selections, the chosen checkpoints, and the
  N1/N3 reports. It is small and durable.
