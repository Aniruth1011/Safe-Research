# A9: review of the loader export (`collection/load_cache.py`, merged in PR #3)

Reviewer: Dhyan. Scope: `SharedCache.records` and `load_cache.py export`. Checked: layer keys 0–33, train-only normalization, the unseen lock.
Tests: `tests/test_a9_loader_export.py` (run with `python -m pytest -q tests/test_a9_loader_export.py`). Data: a synthetic 34-layer cache built by the real `build_cache.py`. Nothing in this review was run on the CARC cache.

## Verdict

The loader reads the right data. Every layer key 0–33 returns its own layer for every pooling and both dev splits. Normalization uses train-only statistics, applied per requested layer in any order. The unseen lock holds for `records` and `export`.

The review found three issues. All three are now **fixed** in `collection/load_cache.py` on Dhyan's branch, and their tests run as normal tests. Two of the notes below are fixed too: unknown pools and missing `norm_stats` layers now give readable errors.

**Done-check on the real cache** (the tracker's "real cache loads through FeatureStore"), run once on a CARC CPU node:

```bash
python scripts/a9_verify_export.py --cache $CACHE --out $TMPDIR/a9_fs --layers 33
```

It exports train and seen_val, loads them back through FeatureStore, and checks that:
- the episode counts match `episodes.csv`;
- every episode has `min(length, task_min_step)` steps;
- the features are bit-identical to `records()`;
- the metadata is right;
- the raw layer files are float16 `[rows, 4096]`, with rows matching `episodes.csv`.

It compares the counts with the numbers published for the shared v1 cache (train 210 / 108 / 86,823, seen_val 140 / 71 / 57,222) and prints a `NOTE`, not a failure, on a different copy. On the shared cache, `unseen_test/` is permission-locked. The loader now reports that as a clear `PermissionError` ("withheld until the locked final benchmark") instead of a bare stat error. This check works with the lock, unlike `check_cache.py verify`.

It also prints the smallest train std per layer (see the std-floor note). It ends with `A9 PASS` or `A9 FAIL` and never touches `unseen_test`.

## Verified (passing tests)

| Property | Test |
|---|---|
| Keys 0..33 read `L00..L33` for mean/first/last in train and seen_val, bit-exact against the source `.layers.pt` (rows taken from inside the split, which checks `row_start`). `layer_names[32]` is `block32_postnorm`, `[33]` is `block32_prenorm`, and the two tensors differ. | `test_every_layer_key_reads_its_own_layer` |
| Requested order is kept (`[33, 0, 16]`). | `test_requested_layer_order_is_kept` |
| Keys outside 0..33, including -1, are refused. | `test_out_of_range_layers_refused` |
| Export metadata: `num_layers=34`, `33=block32_prenorm` is in `layer_indexing`, `FeatureStore.load([33, 0])` re-indexes by key, and unexported or out-of-range keys are refused. | `test_export_layer_keys_and_metadata` |
| `norm_stats` are computed on train rows only (`n_rows` = train rows), per layer, and match the train `.npy` mean and std. | `test_norm_stats_are_train_only_per_layer` |
| `normalize=True` equals `(x - mean[l]) / std[l]` for unsorted layer lists, on train and seen_val. Normalized train data has mean 0. | `test_normalize_uses_the_right_layer_stats_in_any_order` |
| `export --normalize` equals `records(normalize=True)`, and `pooling="mean+trainnorm"`. | `test_normalized_export_matches_records` |
| Unseen lock: `records` raises `PermissionError` (also with `normalize`/`horizon`), and `export --splits unseen_test` fails. The default export holds only train and seen_val, and no exported ID is from tasks 0/3/5. `allow_unseen=True` still works. | `test_unseen_lock` |
| The `episodes.csv` split column equals `splits.json`, and no trajectory is in two splits. | `test_episodes_csv_agrees_with_splits_json` |
| `records(horizon="task_min_step")` cuts every episode to `task_min_step`. | `test_task_min_step_horizon_in_records` |

## Findings (all fixed)

Behaviour change for anyone scripting the export: `export` now writes `task_min_step` episodes by default. Pass `--horizon full` for whole rollouts; it prints a leakage warning. The metadata `pooling` string now ends in `+h=<horizon>`, e.g. `mean+trainnorm+h=task_min_step`. Nothing in the repo parses that string.

**A9-1 (high, fixed): `export` always wrote full-length episodes.** `export()` calls `cache.records(...)` without `horizon`, so it gets the default `"full"`, and the CLI has no `--horizon` option. Failures always run to 520 steps, so any model trained or scored on an export can read the label from episode length. The handoff gives the export command as an alternative to `records(..., horizon="task_min_step")`, which makes this easy to hit. The metadata does not record the horizon either, so nobody can tell afterwards.
Fix: `--horizon {task_min_step,full}` (default `task_min_step`), passed through to `records` and recorded in the metadata.
Tests: `test_export_can_write_task_min_step_horizon`, `test_export_default_is_task_min_step_and_full_is_labelled`.

**A9-2 (medium, fixed): unknown horizons were accepted silently.** `stop = ep.row_end if horizon == "full" else ...` treats any other string as `task_min_step`. A typo like `"task-min-step"` happens to do the right thing, but `"min"` or `"full "` with a trailing space also run without an error.
Fix: `records` raises `ValueError` for anything but `"full"` or `"task_min_step"`.
Test: `test_unknown_horizon_refused`.

**A9-3 (low, fixed): duplicate layer keys were accepted.** `records(layers=[33, 33])` returns K=2 copies of the same layer, and export would write `layers=(33, 33)`. A fusion model would then train on a duplicated layer without any error.
Fix: duplicates are refused next to the range check. The export CLI exits with `error: duplicate layers ...`.
Test: `test_duplicate_layers_refused`.

## Notes (no change needed, worth knowing)

- **Normalization statistics cover the full horizon.** They are computed over full-length train episodes, while results use `task_min_step`. That is fine: it is still train-only and the same for every model. It just means normalized train features at the reporting horizon are not exactly mean 0.
- **The `std` floor is 1e-6, and this does happen (not changed: a data decision for B5/Rhea).** A dimension that is constant on train but varies on seen_val gets scaled by about 1e6. In the synthetic fixture, layer 16 has dimensions whose values round to the same bf16 number on every train row (std = 0), and normalized seen_val values reach 2,000,000. Before the real sweep, check `min(std)` per layer in `norm_stats/*.npz` on the real cache. A larger floor, e.g. `max(std, 1e-3 * median(std))`, would be safer. `fusion_runner` prints a warning when normalized features exceed |x| > 1e3.
- **Missing layers in `norm_stats` (fixed).** If a cache was built with a `--layers` subset, `norm_stats()` now names the missing layers and the ones it has, instead of a bare `KeyError`.
- **Pool names and case (fixed).** `pool` is checked against `manifest["pools_exported"]`, so `pool="Mean"` fails the same way on macOS and on CARC.
- **Memory.** `export` holds every record in memory as float32. That is about 1.4 GB per layer for train at full horizon, so a 4-layer export needs about 10 GB of RAM.

`failure_prob/fusion_runner.py` does not depend on these fixes. It calls `records` directly, validates `horizon` and duplicate layers itself, and checks that every episode has exactly the expected number of steps.
