# R3: Pooling check (first / last / mean) at the top layers

Owner: Akash (runs) · Write-up: Rhea (with Akash's OK, Nov 27) · Status: done

## Question
Each OpenVLA step generates 7 action tokens. The cache stores three ways of
pooling a layer's hidden states over those 7 tokens (collection/README.md):
`mean` (SAFE `token_idx_rel=mean`, what every other experiment uses),
`first` (`token_idx_rel=0.0`) and `last` (`token_idx_rel=1.0`).
Does `first` or `last` beat `mean` at the best layers?

## Runs (CARC, Nov 15, by Akash)
Same settings as R1 for every pooling: B6 probe (`framework_mlp`), B4 episode
scorer, train-only standardization, trained on train, scored on seen_val
(140 episodes, 71 failures), seeds 0/1/2, 1000 bootstrap resamples.
unseen_test was not read.

    /scratch1/akashrav/depth_sweep/{mean,first,last}-trainnorm-train_full-framework_mlp_b6-b4/summary.csv

`mean` covers all 34 layers (= R1). `first` and `last` were run on
L27, L29, L30, L31, L32.

## Results (seen_val, ×100, 3-seed means)

| Layer | Pool | ROC-AUC [95% CI] | Per-task ROC | PR-AUC | TPR@5%FPR |
|---|---|---|---|---|---|
| L27 | **mean** | **75.1** [67.5, 82.1] | **69.8** | **74.8** | 20.2 |
|     | first | 71.6 [63.5, 79.3] | 58.4 | 71.1 | 16.0 |
|     | last  | 72.5 [64.2, 79.6] | 65.5 | 74.0 | 26.8 |
| L29 | **mean** | **75.6** [68.1, 82.6] | **70.5** | **75.3** | 21.6 |
|     | first | 69.4 [60.8, 76.8] | 56.1 | 69.7 | 18.3 |
|     | last  | 70.3 [62.3, 77.7] | 61.6 | 71.1 | 21.1 |
| L30 | **mean** | **76.5** [69.5, 82.8] | **70.1** | **76.1** | 20.7 |
|     | first | 71.1 [62.6, 78.4] | 58.1 | 69.5 | 17.4 |
|     | last  | 70.3 [62.7, 77.8] | 61.5 | 71.2 | 20.2 |
| L31 | **mean** | **77.1** [69.9, 83.7] | **68.4** | **75.7** | 16.9 |
|     | first | 71.3 [64.0, 78.4] | 58.6 | 70.3 | 14.6 |
|     | last  | 69.4 [61.4, 77.0] | 61.4 | 71.2 | 23.0 |
| L32 (SAFE's layer) | **mean** | **75.4** [67.9, 82.6] | **67.4** | **75.2** | 17.8 |
|     | first | 66.0 [57.9, 73.7] | 52.9 | 65.7 | 18.3 |
|     | last  | 71.9 [64.2, 79.4] | 64.0 | 73.6 | 19.7 |

## Finding
- **Mean pooling is best at every layer tested**, on ROC-AUC, per-task ROC-AUC
  and PR-AUC (5 layers × 2 alternatives = 10 out of 10 comparisons).
- Pooled ROC drops by 2.6–9.3 points with `first` or `last`. Several single
  gaps sit inside the ±7 CIs, but the direction never flips.
  The per-task drop is larger (first: −10 to −15; last: −3 to −9),
  i.e. `first`/`last` generalise worse across tasks.
- `last` sometimes has a higher TPR@5%FPR (L27 26.8, L31 23.0 vs 20.2, 16.9).
  That is one operating point and its CIs are very wide (L31 mean: 8.9–50.2), so it does not
  change the choice.
- With `mean`, L31 is still the best layer, so the R1 ranking and the S1–S6
  layer sets do not change.

**Decision: keep `mean` pooling everywhere (already the default). Nothing
downstream needs rerunning.**

Not run: `first`/`last` at L28 and L33, and the all-layer average with
`first`/`last`. Given 10/10 comparisons above, not worth the compute.
