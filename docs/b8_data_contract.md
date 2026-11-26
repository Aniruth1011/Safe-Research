# B8: Data contract and synthetic test data

**Status: done (Rhea).** The shared cache's format is now written down as code, checked by a tool that runs anywhere (including a CARC login node), and used to generate the synthetic data the tests run on. If the format changes on either side, a test fails.

## The contract: `failure_prob/data/contract.py`

| Item | Contract (cache-v1) |
|---|---|
| Layers | 34: `emb`, `block01`–`block31`, `block32_postnorm` (= SAFE's feature), `block32_prenorm` |
| Poolings | `first`, `last`, `mean` over the 7 action tokens |
| Features | `features/<pool>/<split>/L<NN>.npy`, float16, shape `[rows in split, D]` |
| Splits | `train`, `seen_val`, `unseen_test`; unseen tasks 0, 3, 5; every seen task has train and seen_val episodes |
| `episodes.csv` | exactly the 11 columns in `EPISODE_COLUMNS`; `failure = 1 - success`; row ranges contiguous per split; `task_min_step` = the task's shortest length |
| `splits.json` | explicit trajectory IDs per split, disjoint, matching `episodes.csv` |
| `norm_stats/<pool>.npz` | `layers`, `mean`, `std` (float32, `[34, D]`), `n_rows` = train rows (train-only fit, B5) |
| `manifest.json` | layer names, poolings, `feature_dim`, float16, unnormalized features, per-split counts |
| Real cache only (`--expect-real`) | 10 tasks × 50 episodes, D = 4096, episodes ≤ 520 steps |

Check any cache, standard library only (Python 3.6+):
```bash
python3 failure_prob/data/contract.py /scratch1/akashrav/safe_cache/v1 --expect-real
```
`CONTRACT OK` or a list of what's wrong. Locked unseen files are reported as withheld, not as problems.

## Synthetic data from the contract: `failure_prob/data/synthetic_cache.py`

`make_synthetic_cache(tmp_path)` writes fake rollouts in the rollout patch's real per-episode format (layer names, poolings, run fingerprint, SAFE `.pkl`, per-step `.csv`). It then builds them with the real `collection/build_cache.py`. The result has the same files and structure as v1, just small (10 tasks × 6 episodes, D = 16 by default). Planted structure gives tests a real effect to find:
- failed episodes drift in layers 20–31 and 33;
- L32 is an RMS-normalized copy of L33;
- L0 with first pooling is constant, as in v1;
- failures run longer than successes.

## What fails if something drifts (`tests/test_data_contract.py`)

- The synthetic cache must pass the contract.
- The contract's layer names must equal what the rollout patch writes. Its columns must equal what `build_cache.py` writes, and its unseen tasks must equal `failure_prob/protocol.py`.
- The loaders (`SharedCache`, `load_split_records`) must accept the synthetic cache.
- 13 deliberate breaks must each be caught with a clear message: a missing layer file, a renamed layer, a dropped 34th layer, float32 features, a changed dimension, a dropped pooling, a renamed column, a shifted row range, flipped labels, a wrong horizon, overlapping splits, changed unseen tasks, and stats fit beyond train.
- The checker must stay standard-library only.
