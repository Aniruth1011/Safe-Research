# Adarsh's workstream: SAFE baselines (C1–C3), controls and figures

Branch: `adarsh/safe-baselines`. The brief is the handoff doc "Adarsh's workstream: SAFE baselines, controls & figures" (13 Nov 2026). The team sheet's Task Plan tab has status and owners. Results go in its Experiments tab.

**All numbers here are seen_val on a single fixed split (SAFE seed-0 unseen tasks).** They are selection-optimistic, because seen_val is also the selection set. Do not compare them with SAFE's published Table 1 (MLP 73.47, LSTM 72.47, cosine k-NN 69.45 ROC-AUC), which averages 3 splits. Unseen-task numbers come only from N3.

## Status

| Task | Experiments row | Status |
|---|---|---|
| C1 SAFE final-layer MLP | 1 | **Done** on seen_val, including the score dump and TPR@5%FPR |
| C2 SAFE final-layer LSTM | 2 | Running (SLURM 14240601, 36 tasks) |
| C3 cosine k-NN | 3 | **Done: fitted and selected on seen_val.** TPR@5%FPR is pending the checkpointed winner re-run (SLURM 14243438). |
| C4 evaluator cross-check | — | Blocked on B4 (Rhea). The score dump (`scripts/dump_safe_scores.py`, CSV + per-step `.npz`) is ready. |
| C5 end-to-end gate | — | Blocked on B1–B4 |
| K1 / K2 / K3 / K4 | 4+26 / 27 / 34 / 35 | Blocked on the winning fusion (F1–F5). K1 also needs B7; K2 also needs R2. |
| N2 figures | — | Blocked on R1 and F4 |

## How C1–C3 are run

SAFE's own `python -m failure_prob.train` runs unchanged. Its training loop, loss, models (`failure_prob/model/{indep,lstm,embed}`), metrics and selection rule are all SAFE's. The data comes in through a new dataset module, `dataset=openvla_cache`.

- `failure_prob/data/openvla_cache.py`:
  - reads layer 32 (block 32 after the final RMSNorm, which is SAFE's feature), mean pool, from the shared cache `/scratch1/akashrav/safe_cache/v1`;
  - loads only `train` (210 episodes) and `seen_val` (140 episodes). `unseen_test` is never requested, and the tests fail if it is opened;
  - takes the split verbatim from `splits.json`, identical for every seed, and checks it against the file;
  - reads `task_min_step` from `episodes.csv` and does not recompute it;
  - keeps features raw (no normalisation).
- Two guards in SAFE's originals, as named in the handoff:
  - `train.py` falls back to CPU when there is no GPU;
  - `utils/routines.py` skips split-CP and functional-CP when there is no `val_unseen`.
- Reported metrics are limited to the `task_min_step` horizon only:
  - `falert_early_roc_auc` (the selection metric);
  - `falert_early_prc_auc`;
  - `roc_auc/..._tq1.0`.
  
  `falert_end_*` and other full-rollout metrics leak the label through episode length (failures run 520 steps). They are never reported.
- Selection (`scripts/collect_safe_runs.py`) applies SAFE's rule:
  - group runs by hyperparameters;
  - average the final-epoch `falert_early_roc_auc/model_val_seen` over seeds 0/1/2;
  - take the argmax.
  
  It fails on a missing seed, a duplicate run, a failed run or any unseen key. It reads offline wandb runs from their `run-*.wandb` files, because offline runs do not write `wandb-summary.json`.
- Compute: CARC compute nodes only, account `yzhao010_1531`, `WANDB_MODE=offline`. Outputs go to `/scratch1/$USER/safe_baselines/<model>/` and are not committed.

## C1 result: SAFE MLP (`model=indep`)

- **Selected config:** lr = 3e-4, lambda_reg = 1.0, batch 64, 1000 epochs, L32 mean.
- **seen_val falert_early ROC-AUC:** 0.7552 ± 0.0030 (mean ± std over seeds 0/1/2).
- **seen_val falert_early PR-AUC:** 0.7494 ± 0.0042. This is SAFE's trapezoid `auc(recall, precision)`.
- **seen_val ROC-AUC at tq1.0:** 0.7552. It equals falert_early, as expected: SAFE-MLP scores are a cumsum of a sigmoid, so they are monotone.
- **TPR @ 5% FPR:** 0.1690 ± 0.0000. **TPR @ 10% FPR:** 0.3756 ± 0.0351.
  - Computed on the per-episode score `max(s[:task_min_step])`. This is the same score as SAFE's falert_early, with no sigmoid.
  - TPR@x%FPR = max TPR with FPR ≤ x on sklearn's `roc_curve`, the same definition as `failure_prob/engine/metrics.py`.
  - The value is identical across seeds for a real reason. In every seed 4 successful episodes score far higher than the rest (about 370–390 vs about 200). At FPR ≤ 5% only 3 false positives are allowed (5% of 69 successes), and exactly 12 of 71 failures score above that cut, with no ties.
  - Scoring at `task_min_step` (`score_at_tms`) gives the same numbers, because SAFE-MLP scores are monotone.
- **Reproducibility check:**
  - The checkpointed winner re-run (SLURM 14243293) logged exactly the same per-seed ROC-AUC as the selection runs.
  - `dump_safe_scores.py` recomputes the same values from the saved checkpoints.
- **Trainable parameters:** 1,049,089 (Linear 4096→256, ReLU, Linear 256→1, Sigmoid).
- **Latency:** not measured. The handoff does not ask for it.
- The std is the population std (ddof = 0) over the 3 seeds.

Per-seed values for the selected config:

| Seed | ROC-AUC | PR-AUC | TPR@5%FPR | TPR@10%FPR |
|---|---|---|---|---|
| 0 | 0.7589 | 0.7533 | 0.1690 | 0.4225 |
| 1 | 0.7551 | 0.7514 | 0.1690 | 0.3662 |
| 2 | 0.7516 | 0.7435 | 0.1690 | 0.3380 |

All 12 configs (SAFE's grid), sorted by the selection metric:

| lr | lambda_reg | ROC-AUC (mean ± std) | PR-AUC (mean ± std) |
|---|---|---|---|
| 0.0003 | 1 | 0.7552 ± 0.0030 | 0.7494 ± 0.0042 |
| 0.001 | 1 | 0.7498 ± 0.0137 | 0.7479 ± 0.0149 |
| 0.0001 | 0.1 | 0.7431 ± 0.0035 | 0.7401 ± 0.0029 |
| 0.0001 | 0.001 | 0.7419 ± 0.0019 | 0.7422 ± 0.0031 |
| 0.0001 | 0.01 | 0.7416 ± 0.0025 | 0.7412 ± 0.0024 |
| 0.0001 | 1 | 0.7394 ± 0.0090 | 0.7354 ± 0.0075 |
| 0.0003 | 0.1 | 0.7245 ± 0.0082 | 0.7277 ± 0.0034 |
| 0.0003 | 0.01 | 0.7205 ± 0.0044 | 0.7292 ± 0.0057 |
| 0.001 | 0.01 | 0.7195 ± 0.0249 | 0.7241 ± 0.0245 |
| 0.001 | 0.1 | 0.7147 ± 0.0179 | 0.7140 ± 0.0070 |
| 0.0003 | 0.001 | 0.7126 ± 0.0077 | 0.7252 ± 0.0075 |
| 0.001 | 0.001 | 0.6343 ± 0.0951 | 0.7293 ± 0.0172 |

Stored files:
- `experiments/results/C1_safe_mlp/runs.csv`: all 36 runs.
- `selection.csv`: the 12 configs.
- `winner.json`.
- `scores_seen_val_s{0,1,2}.csv`: per-episode seen_val scores of the winner, one file per seed.
- `winner_episode_metrics.csv`: ROC-AUC and TPR@5/10%FPR per seed, for both episode scores.
- `experiments/manifests/C1_safe_mlp.yaml`.
- Not committed: per-step score arrays (`scores_indep_s*.npz`, for C4) and checkpoints. They are in `/scratch1/$USER/safe_baselines/{scores,winners}/`.

Provenance:
- Code commit `ea1420d`, SLURM job 14240563.
- Winner re-run: SLURM 14243293, with `train.eval_save_ckpt=True`.
- 36 tasks on `main`, 8 CPU / 32 GB each, about 20–25 min per run.
- Sheet value for "run directory": `/scratch1/$USER/safe_baselines/indep` (Adarsh's CARC user).

## C3 result: cosine k-NN (`model=embed`)

- **Selected config:** distance = cosine, topk = 10, cumsum = True, use_success_only = False, L32 mean.
- **seen_val falert_early ROC-AUC:** 0.6765 ± 0.0000. **PR-AUC:** 0.6948 ± 0.0000.
  - The std is 0 because SAFE's k-NN is deterministic. Seeds 0/1/2 give identical values, which was checked.
- **seen_val ROC-AUC at tq1.0** (the score exactly at `task_min_step`): 0.7271.
- **TPR @ 5% FPR:** pending the score dump from the checkpointed winner re-run (SLURM 14243438).
- **Trainable parameters:** 0. The model is non-parametric: the bank is the 86,823 train timestep features (4096-d), split into success and failure.
- **Latency:** not measured.

All 6 configs, sorted by the selection metric:

| topk | cumsum | falert_early ROC-AUC | falert_early PR-AUC | ROC-AUC at tq1.0 |
|---|---|---|---|---|
| 10 | True | 0.6765 | 0.6948 | 0.7271 |
| 5 | True | 0.6736 | 0.6920 | 0.7118 |
| 1 | True | 0.6528 | 0.6696 | 0.6889 |
| 10 | False | 0.6107 | 0.5968 | 0.8481 |
| 5 | False | 0.5818 | 0.5887 | 0.8494 |
| 1 | False | 0.5436 | 0.5557 | 0.8373 |

**Finding for B4 (Rhea) and C4: the two episode scores rank the configs very differently.**
- Under SAFE's max-so-far score (falert_early, the handoff's selection metric), cumsum = True wins: 0.68 vs 0.61.
- Under the score at `task_min_step` (tq1.0, the B4 spec), cumsum = False is far better: about 0.85 vs 0.73.
- Both are limited to the `task_min_step` horizon, so neither leaks the label.
- Selection follows the handoff (falert_early). Which score is official is the open decision in B4, and it changes which k-NN config is best.

Stored files:
- `experiments/results/C3_cosine_knn/runs.csv`: all 18 runs (6 configs × 3 seeds).
- `selection.csv`.
- `winner.json`.
- `experiments/manifests/C3_cosine_knn.yaml`.

Provenance:
- Code commit `ea1420d`, SLURM job 14241427.
- 6 tasks on `main`, 8 CPU / 32 GB each, about 45–60 min per task (3 seeds).
- Run directory: `/scratch1/$USER/safe_baselines/embed`.

## Reproduce

```bash
# environment (once, inside an allocation): Python 3.10 venv per the handoff; wandb must be <0.29 (pinned in setup.py)
mkdir -p /scratch1/$USER/safe_baselines/slurm
sbatch --export=ALL,MODEL=indep --job-name=c1-mlp scripts/carc/safe_baseline.slurm                    # C1, 36 tasks
sbatch --export=ALL,MODEL=lstm  --job-name=c2-lstm --time=08:00:00 scripts/carc/safe_baseline.slurm   # C2, 36 tasks
sbatch --export=ALL,MODEL=embed --job-name=c3-knn --array=0-5 --cpus-per-task=8 --mem=32G --time=02:00:00 \
       scripts/carc/safe_baseline.slurm                                                                # C3, 6 tasks
python scripts/collect_safe_runs.py /scratch1/$USER/safe_baselines/indep/wandb --out /scratch1/$USER/safe_baselines/indep/selection.csv
# winner re-run with checkpoints (index = 12*lr_idx + 3*lam_idx + seed), then the score dump on seen_val
sbatch --array=21,22,23 --export=ALL,MODEL=indep,OUT=/scratch1/$USER/safe_baselines/winners/indep,EXTRA_ARGS=train.eval_save_ckpt=True \
       --job-name=c1-win scripts/carc/safe_baseline.slurm
python scripts/dump_safe_scores.py --run-dir <logs_save_path of a winner run> --seed 0 --out-dir /scratch1/$USER/safe_baselines/scores
```

## Measured costs (CARC)

| Model | Speed | Run time |
|---|---|---|
| SAFE MLP | 2.4 s/epoch on 4 CPU | about 20–25 min per 1000-epoch run on 8 CPU |
| SAFE LSTM | 19 s/epoch on 4 CPU | about 5.3 h per run on 4 CPU, hence the 8 h limit |
| Cosine k-NN | 15.3 min per seed on 4 CPU, about 10 GB RAM | — |

## Deviations from the handoff, and why

1. **The collector reads `run-*.wandb`.** The handoff expected `files/wandb-summary.json`, but offline wandb runs do not write it.
2. **C3 is a 6-task array, not one 4 h job.** SAFE's k-NN also scores every padded train step, so 18 sequential evaluations would hit the time limit. The grid is unchanged.
3. **The winner re-run writes to a separate `OUT`,** so the selection folder never gets duplicate runs.
4. **`setup.py` pins `wandb<0.29`.** wandb 0.29 removed `wandb.finish(quiet=...)`, which SAFE's `train.py` calls. The venv uses 0.19.11.
5. **C3 resources are 8 CPU / 32 GB, not 16 / 64,** based on the measured cost.

## Open decisions (defaults in use)

| Decision | Default in use | Who decides |
|---|---|---|
| Pooling for C1–C3 | mean only | Akash |
| Official episode score | max-so-far vs at `task_min_step`: report both in C4 | Rhea, B4 |
| PR-AUC definition | trapezoid vs average precision: report both in C4 | Rhea, B4 |
