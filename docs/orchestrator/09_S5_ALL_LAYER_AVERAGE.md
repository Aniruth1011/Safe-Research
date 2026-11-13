# S5 all-layer average: implementation for review

For Rhea (owner of plan row S5) to review. Written 2026-11-11. Nothing has run on CARC with it yet.

**What it is.** A Hide-and-Seek-style baseline: one probe trained on features averaged over layers. It is a
pseudo-layer of the depth sweep (`--layers avg`), so it gets the same probe, evaluator, seeds, bootstrap CIs and
per-task breakdown as every single layer. The depth-curve plot draws it as a reference line.

**Definition, as implemented.** Function `avg_records` in `failure_prob/engine/depth_sweep.py`:
- **Members:** layers 1–31 and 33 of cache v1. Index 33 is block 32's output before the final RMSNorm.
- **Excluded:** 0 (embeddings) and 32 (final post-norm output, SAFE's feature).
- **Choosing members by name:** every layer except `emb` and `*_postnorm`, which requires a `*_prenorm` layer to
  exist.
- **Z-scoring:** per member layer, z = (x − mean_train) / std_train, per feature dimension. The statistics come from
  the cache's `norm_stats/<pool>.npz`, computed on the train split only, with std floored at 1e-6 (the loader's
  rule).
- **Features:** the plain mean of the 32 z-scored layers, one 4096-d vector per step, in float32.
- **Pooling:** same as the run (`mean`, `first` or `last` token pooling).
- **`--normalize`:** not applied on top.
- **Record:** each row stores the definition (`layer_definition`: members, excluded layers, statistics).

**Questions for the review.**
1. Is excluding layer 32 (post-norm) right, with 33 (pre-norm) standing in for block 32? Or should 32 also be in,
   or 32 in place of 33?
2. Plain mean, or a weighted mean (e.g. learned weights, which would be a different baseline)?
3. Z-score per feature dimension, as implemented, or per layer with a single scale?
4. Is the embedding layer (0) out on purpose in the Hide-and-Seek setup we follow?

**Where to look.** The sweep doc covers it under "All-layer average"; the tests that compare it with averaging
train-z-scored layers by hand are `test_avg_*` in `tests/test_depth_sweep.py`.
