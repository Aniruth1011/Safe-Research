# Layer fusion results: F1, F2, F3, F6 (seen_val)

Sheet rows 19, 20, 21 and 25 (Dhyan). Date: Nov 15, 2026.

## Setup

- Shared cache v1, mean pooling, train-only z-scoring.
- Train on `train` at the full horizon. Score `seen_val` (140 episodes, 71 failures) at `task_min_step`.
- Probe: the locked **B6** settings:
  - mlp hidden 64, dropout 0.1, lr 3e-4, batch 8, at most 150 epochs;
  - early stopping with patience 10 on the loss of a 15% train holdout, the same episodes per seed as the depth sweep.
- Scorer: **B4** (`failure_prob.engine.episode_metrics`, SAFE-equivalent).
- Seeds 0, 1, 2.
- 95% CIs: stratified bootstrap over seen_val episodes (1000 resamples).
- Layers from selection (Rhea, from Akash's real R1 sweep `framework_mlp_b6-b4`):
  - S2 pair L31 + L23;
  - S3 set L3 + L21 + L31;
  - ranked top 4: L31, L30, L29, L27.
- F1–F3: compact search over lr {3e-4, 1e-3} × dropout {0.1, 0.3} × weight decay {0, 1e-4} (8 points, all seeds each). The winner has the best mean seen_val ROC-AUC, so its numbers are optimistic.
- F6: the F3 architecture with B6 settings.
- unseen_test was not read (locked until N3). Single fixed split (SAFE seed-0 unseen tasks).
- Code: branch `dhyan/a9-f1-f3`, commit `3eba1fe`. Slurm jobs 14270515–14270519 and 14270682.

## Results (mean ± std over 3 seeds)

| Row | Experiment | Layers | ROC-AUC (%) | 95% CI | PR-AUC (%) | TPR @ 5% FPR (%) | Parameters | Latency (ms) | Chosen hparams |
|---|---|---|---|---|---|---|---|---|---|
| ref | Best single layer (depth sweep, B6) | L31 | 77.14 ± 1.8 | | 75.72 | 16.90 | | | B6 |
| ref | Best single layer (this runner, B6) | L31 | 76.52 ± 4.51 | 69.50–83.07 | 75.76 | 24.41 | 262,273 | 0.057 | B6 |
| 19 | **F1** two-layer concat + MLP | L31 + L23 | 74.99 ± 2.29 | 67.84–81.48 | 75.24 | 22.07 | 524,417 | 0.077 | lr 3e-4, do 0.1, wd 0 |
| 20 | **F2** three-layer concat + MLP | L3 + L21 + L31 | 74.49 ± 2.02 | 67.63–81.09 | 76.06 | 25.82 | 786,561 | 0.085 | lr 3e-4, do 0.3, wd 1e-4 |
| 21 | **F3** static learned weights (64-d projection) + MLP | L3 + L21 + L31 | 75.59 ± 0.95 | 67.74–82.60 | 77.05 | 24.41 | 790,853 | 0.175 | lr 3e-4, do 0.1, wd 0 |

### F6: number of layers (F3 architecture, ranked layers)

| K | Layers | ROC-AUC (%) | 95% CI | PR-AUC (%) | TPR @ 5% FPR (%) | Parameters | Latency (ms) | Δ vs K=1 (pp) |
|---|---|---|---|---|---|---|---|---|
| 1 | L31 | **77.38** ± 1.23 | 69.75–84.08 | 76.47 | 23.94 | 266,435 | 0.115 | 0 |
| 2 | L31 + L30 | 76.94 ± 2.92 | 69.47–83.79 | 76.01 | 22.07 | 528,644 | 0.155 | −0.44 |
| 3 | L31 + L30 + L29 | 75.78 ± 4.32 | 68.39–82.66 | 74.89 | 20.66 | 790,853 | 0.186 | −1.60 |
| 4 | L31 + L30 + L29 + L27 | 75.64 ± 3.64 | 67.91–82.94 | 74.28 | 20.66 | 1,053,062 | 0.213 | −1.74 |

## Paired comparisons (same 140 episodes, 1000 paired resamples, B4 scorer)

| A vs B | ROC-AUC Δ (pp) | 95% CI | P(A not better) | PR-AUC Δ | per-task ROC-AUC Δ |
|---|---|---|---|---|---|
| F1 vs best single L31 (sweep) | −2.15 | [−4.83, +0.39] | 0.95 | −0.47 | −0.68 |
| F2 vs F1 | −0.50 | [−3.75, +2.76] | 0.63 | +0.82 | +0.64 |
| F3 vs F2 (concat, same layers, ~same params) | +1.10 | [−1.70, +3.82] | 0.22 | +0.99 | +0.51 |
| F3 vs best single L31 (sweep) | −1.55 | [−4.62, +1.52] | 0.83 | +1.34 | +0.47 |

Sanity check: this runner's L31 matches the sweep's L31 within noise (75.4 vs 77.0, Δ −1.6 pp, CI [−3.7, +0.7], measured with the earlier probe).

## Findings

1. **No fusion beats the best single layer on ROC-AUC.**
   - F1, F2 and F3 are 1.5–2.6 points below L31.
   - Every paired CI includes 0; F1 is the closest to significantly worse (P(not better) 0.95).
2. **Learned static weights (F3) are the best fusion design.**
   - +1.1 pp ROC-AUC and +1.0 pp PR-AUC over concatenation of the same layers, at about the same parameter count.
   - The most stable across seeds (±0.95 vs ±2.0–4.5).
   - The gain is not significant (CI [−1.7, +3.8]).
3. **More layers do not help (F6).** ROC-AUC falls slowly with K (77.4 → 76.9 → 75.8 → 75.6). Variance, parameters and latency grow. The best K is 1.
4. **Fusion detects more failures at a strict threshold.**
   - TPR @ 5% FPR is 22–26% for F1–F3, against 16.9% for the sweep's L31.
   - The CIs are very wide (±15–18 pp), so this is a hint only.
5. **Fusion is cheap:** under 0.22 ms per step on CPU, negligible next to the VLA forward pass.

Caveats:
- Hyper-parameters were chosen on seen_val, so the fusion numbers are slightly optimistic. That makes "fusion does not win" a conservative conclusion.
- The late layers are near-ties in R1 (±0.07 CI).
- These are seen-task results. The unseen-task question is answered only at N3.
