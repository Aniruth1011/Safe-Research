# B5: Standardization (train-only z-score) and the raw vs standardized switch

**Status: done (Rhea).** Fitting the statistics on train only and the raw/standardized switch were already built.
B5 adds the evidence for both, a std-floor decision based on the real cache, and tests that fail if anything leaks.

## What is standardized, and with what

- **Statistics:** `collection/build_cache.py` computes a per-layer, per-dimension mean and std over the **train split only** (86,823 rows in cache v1). It stores them in `norm_stats/<pool>.npz` and records the row count in `n_rows`. Seen_val and unseen_test rows are never used.
- **Application:** one function, `SharedCache.norm_stats()` in `collection/load_cache.py`, applies them as `(x - mean) / max(std, 1e-6)`. It's used by:
  - `scripts/train.py --cache ... --normalize` (B1)
  - the depth sweep (its `trainnorm` runs and the S5 all-layer average)
  - the fusion runner (all three read through `SharedCache`)
- **Fusion runner:** `scripts/run_fusion.py` standardizes by default (`--no-normalize` for raw). It has an optional `--std-floor` override, which defaults to the loader's 1e-6. Per the decision below, leave it unset for reported results.
- **Switch:** in `train.py` and the loader export, raw is the default. `--normalize` (train.py, load_cache export) or the sweep's `trainnorm`/`raw` setting chooses. Each run records the choice (`normalize` in config.json; `trainnorm`/`raw` in the sweep folder name).
- **Team default:** standardized for the main comparison, raw as a sensitivity check. Akash's R4 (raw) vs R1 (normalized) are within noise: L31 0.778 vs 0.771, L32 0.758 vs 0.754.

## Std floor: keep 1e-6 (checked on cache v1)

`python3 scripts/check_norm_stats.py /scratch1/akashrav/safe_cache/v1` (run 2026-11-25) prints, for each pooling and layer, the min, median and max std and how many of the 4096 dims fall below 1e-6, 1e-3 and 1e-2:

| Pooling | Smallest std | Where the 1e-6 floor acts | Dims with std < 1e-3 |
|---|---|---|---|
| mean | 0.00038 (L32) | nowhere | 1 (L32) |
| last | 0.00041 (L32) | nowhere | 1 (L32) |
| first | 0 (L0) | **L0, all 4096 dims** | L0 (all), 1 (L32) |

- **The floor only matters for L0 with first pooling.** The "first" position is the same prompt token at every step, so its embedding is constant (std exactly 0). With the floor, it standardizes to exactly 0, never inf or nan (`tests/test_standardization.py`). It carries no information either way. L0 is excluded from layer selection, and the sweep uses mean pooling.
- **L32 (after the final RMSNorm) has one nearly constant dimension** (std ≈ 4e-4) in every pooling. The floor doesn't touch it. Standardizing scales it up like every other dimension, so it may amplify small noise in 1 of 4096 inputs. That fits L32 raw vs normalized being within noise. Mention it if L32's normalized result is discussed.
- **Every other layer/pooling has no dimension below 1e-3.**
- **Changing the floor would change nothing** for any layer or pooling except L0-first (whose output stays 0). Keeping it leaves R1, R4, S1–S6 and the fusion results exactly as they are.

**Why per-dimension standardization is needed:** raw spread grows with depth. With mean pooling, the median std is 0.012 at block 1, 0.32 at block 16, 0.81 at block 31 and 0.91 at block 32 before the norm. Some dimensions reach a std of 24 in block 32 before the norm. Without standardization, a probe or an all-layer average (S5) would be dominated by late layers and a few large dimensions.

## Tests (`tests/test_standardization.py`)

- The stored mean/std equal train-only statistics for every pooling and for layers 0, 4 and 32. They do **not** equal statistics computed with seen_val or unseen rows added (leakage guard).
- Standardized train features have mean 0 and std 1 per dimension (exactly 0 for constant dims). Seen_val is standardized with the train statistics, not its own.
- A constant feature (std 0) standardizes to 0, never inf or nan.
- `train.py --normalize` and the raw default both run and record which one was used.
- `scripts/check_norm_stats.py` runs with the standard library only.

## Re-checking a future cache

```bash
python3 scripts/check_norm_stats.py /path/to/safe_cache/vN
```
If any layer other than L0 shows dims under `<1e-6`, revisit the floor before using standardized results for that layer.
