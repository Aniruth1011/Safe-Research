# Implementation log

## 2026-10-24 — initial framework

- Work completed: audited SAFE and added additive registry, feature-store, engine, CLI, manifests, documentation, and synthetic tests.
- Files modified: see git diff; original SAFE model/data/training files were preserved.
- Commands executed: repository inspection; `python -m pytest -q` (record final result after implementation).
- Outstanding issues: real OpenVLA intermediate extraction and real-data training require the external adapted OpenVLA checkout, weights, and rollout data.
- Next step: run `python scripts/train.py --model gated_fusion --dataset libero10 --layers 8 16 24 32 --synthetic` to validate the reference experiment.

## 2026-11-14 — C1–C3 SAFE baselines on the shared cache (Adarsh)

- Work completed: SAFE's original MLP (C1), LSTM (C2) and cosine k-NN (C3) trained on the shared cache v1 (L32, mean pool) with SAFE's own grids, selected on seen_val only, winners re-run with checkpoints and their seen_val scores dumped. unseen_test was never loaded.
- Results (seen_val, single fixed split, selection-optimistic; not comparable to SAFE's published numbers):

  | Task | Selected config | falert_early ROC-AUC | PR-AUC | TPR@5%FPR | Params |
  |---|---|---|---|---|---|
  | C1 SAFE MLP | lr 3e-4, lambda_reg 1 | 0.7552 ± 0.0030 | 0.7494 ± 0.0042 | 0.1690 ± 0.0000 | 1,049,089 |
  | C2 SAFE LSTM | lr 3e-4, lambda_reg 1 | 0.7418 ± 0.0214 | 0.7532 ± 0.0119 | 0.2723 ± 0.0332 | 4,458,753 |
  | C3 cosine k-NN | topk 10, cumsum True | 0.6765 ± 0.0000 | 0.6948 ± 0.0000 | 0.3099 ± 0.0000 | 0 |

- Files modified: `failure_prob/data/openvla_cache.py`, `failure_prob/conf/__init__.py`, `failure_prob/conf/dataset/openvla_cache.yaml`, the two guards in `failure_prob/train.py` and `failure_prob/utils/routines.py`, `scripts/carc/safe_baseline.slurm`, `scripts/collect_safe_runs.py`, `scripts/dump_safe_scores.py`, `setup.py` (`wandb<0.29`), tests, `experiments/manifests/C{1,2,3}_*.yaml`, `experiments/results/C{1,2,3}_*/`, `docs/adarsh/README.md`.
- Commands executed: `python -m pytest -q`; `sbatch` of `scripts/carc/safe_baseline.slurm` for MODEL=indep/lstm/embed and the winner re-runs; `scripts/collect_safe_runs.py`; `scripts/dump_safe_scores.py`.
- Outstanding issues: official episode score and PR-AUC definition are open in B4 (C2 and C3 rank much better with the score at `task_min_step`); unseen_test numbers come only from N3.
- Next step: C4 evaluator cross-check once B4 lands. Details in `docs/adarsh/README.md`.
