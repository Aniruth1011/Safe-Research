# Layer fusion: Dhyan's rows (F1, F2, F3, F6)

| Task | Sheet row | Experiment | Code | Layers (from selection) | Compare against |
|---|---|---|---|---|---|
| F1 | 19 | Two-layer concatenation + MLP | `concat_fusion` | complementary pair (S2) | best single-layer MLP (R1/R2) |
| F2 | 20 | Three-layer concatenation + MLP | `concat_fusion` | early / middle / late (S3) | F1 |
| F3 | 21 | Static learned layer weights | `static_weight_fusion --proj-dim 64` | selected 3 layers | F2 (about the same parameter count) |
| F6 | 25 | Number-of-layers ablation, K = 1, 2, 3, 4 | best of F1–F5 | ranked selection, first K | K = 1 |

- **Concatenation** (`failure_prob/models/concat_fusion.py`) joins the K layers per timestep (`[B,T,K·D]`) and probes the result. The head grows with K.
- **Static learned weights** (`failure_prob/models/static_weight_fusion.py`) compute `gamma · Σ_k softmax(α)_k · P_k(x_k)`. Each layer first goes through its own `Linear(D, 64)` into a shared 64-d space ("project features to shared dimension"). The weights are one scalar per layer, the same for every timestep and episode. With `proj_dim = hidden_dim = 64`, F3 has about the same number of parameters as concatenating the same layers (K·4096·64), so F3 vs F2 is a parameter-matched comparison. Without `--proj-dim` it is a pure scalar mix of the z-scored layers, and `--fixed-weights` freezes the mix (uniform, the S6 reference).
- **Probe heads:** both models take `head="mlp"` (default) or `"linear"` (`failure_prob/models/probe_heads.py`). Once B6 fixes the probe, pass its head, width and budget as flags.

## The shared cache on CARC

The cache is `/scratch1/akashrav/safe_cache/v1`: 34 layers (`L00`–`L33`), poolings `mean`/`first`/`last`, float16 `[rows, 4096]` files.

| Split | Tasks | Episodes | Failures | Rows | Access |
|---|---|---|---|---|---|
| train | 7 seen, 30 each | 210 | 108 | 86,823 | read |
| seen_val | 7 seen, 20 each | 140 | 71 | 57,222 | read |
| unseen_test | 0, 3, 5 | 150 | 58 | 54,555 | **locked** (owner only) until N3 |

- **Access.** Jobs run under the team's CARC project account. Check that you are in it with `sacctmgr show assoc user=$USER format=account`, then `export SBATCH_ACCOUNT=<account>` once per shell (or pass `--account`). If the account is missing, or `sbatch` says "Invalid account", ask Akash.
- **Always use the full path.** `/scratch1/akashrav` is pass-through only, so `ls` on it fails by design.
- **The lock is expected.** unseen_test is permission-locked. These tools only read train and seen_val, and they are tested on a copy with the same lock (`tests/test_locked_unseen.py`). `check_cache.py verify` crashes on the lock, so don't run it on the shared cache. The A9 check below is the read-only check to use instead.
- **Read only.** Never write into the cache. Outputs go to your checkout or `/scratch1/$USER`.
- **Use a compute node, not the login node.** Use `sbatch` as below, or `salloc -p main -c 4 --mem=32G -t 01:00:00` for interactive work.
- **Report results** as "single fixed split (SAFE seed-0 unseen tasks)". Every output already says so. Don't put them next to SAFE's published numbers, which average three splits.

## Run on CARC (Slurm, CPU)

`scripts/carc/fusion.slurm` wraps every command below. It defaults to 8 CPUs, 32G and 4 h; override with `sbatch --mem=... --time=...`. Python: `FUSION_PYTHON`, else the kit venv `$WORK/venv310`, else `python3` on your PATH (e.g. an activated conda env).

```bash
cd Safe-Research && mkdir -p logs            # Slurm writes logs/fusion-<job>.out (git-ignored)
export SBATCH_ACCOUNT=<team CARC account>
export CACHE=/scratch1/akashrav/safe_cache/v1
sbatch scripts/carc/fusion.slurm a9 --cache "$CACHE" --out /scratch1/$USER/a9_fs --layers 33   # A9 done-check
sbatch scripts/carc/fusion.slurm search --model concat_fusion --layers A B --cache "$CACHE" \
       --output-dir outputs/fusion/F1 --label "F1 two-layer concat"
sbatch scripts/carc/fusion.slurm kcurve --model concat_fusion --layers A B C D --reference-layer R \
       --cache "$CACHE" --output-dir outputs/fusion/F6
sacct -j <job id> --format=JobID,Elapsed,MaxRSS,State                                      # measured time/memory
```

- **Out of time or cancelled:** resubmit the same command with `--resume`. Finished seeds and grid points are kept, and a folder whose configuration differs is refused.
- **Memory:** the runner prints its memory estimate before loading and stops if the estimate exceeds 90% of the job's `--mem` (`--no-memory-check` skips this).
- **Disk reads:** layer files are read in one sequential pass each. Memory-mapped random reads stalled on `/scratch1` under load in the depth sweep.

The same commands without Slurm:

```bash
# CACHE = the shared cache v1/ folder (team handoff); only train + seen_val files are read
python scripts/fusion_experiments.py search --model concat_fusion --layers A B --cache "$CACHE" \
    --output-dir outputs/fusion/F1 --label "F1 two-layer concat"
python scripts/fusion_experiments.py search --model concat_fusion --layers A B C --cache "$CACHE" \
    --output-dir outputs/fusion/F2 --label "F2 three-layer concat"
python scripts/fusion_experiments.py search --model static_weight_fusion --proj-dim 64 --layers A B C --cache "$CACHE" \
    --output-dir outputs/fusion/F3 --label "F3 static learned weights"
# F6: --layers is the RANKED selection (best first); pass the winner's searched hyper-parameters.
# --reference-layer = R1's best single layer, also run as K=1 when it is not the first ranked layer
python scripts/fusion_experiments.py kcurve --model concat_fusion --layers A B C D --lr 3e-4 --dropout 0.2 \
    --reference-layer R --cache "$CACHE" --output-dir outputs/fusion/F6 --label "F6 K ablation"
# Is the gain real? Paired bootstrap CI of the difference, against the depth sweep's single layer or another run
python scripts/fusion_compare.py --a outputs/fusion/F1/hp_<winner> --b "<sweep out>/rows/L16_seed*.scores.npz" \
    --label-a "F1" --label-b "best single L16" --out outputs/fusion/F1/vs_L16.json
# a single configuration (no search)
python scripts/run_fusion.py --model static_weight_fusion --proj-dim 64 --layers A B C --cache "$CACHE" \
    --output-dir outputs/fusion/one_run
# CPU smoke test, random data
python scripts/run_fusion.py --model static_weight_fusion --layers 8 16 --synthetic --epochs 2 --output-dir /tmp/f
```

Until S1–S3 pick the layers, any layers run, but the numbers are development numbers only. Use layer 33, not 32, for raw block 32.

**Protocol** (the same as Akash's R1–R4 depth sweep, so fusion rows compare like-for-like with single layers):
- Train on `train` at the full horizon (`--train-horizon`).
- Score every `seen_val` rollout at `task_min_step` (`--horizon`; the only reportable horizon).
- Seeds 0/1/2.
- **Probe = the locked B6 settings (default):** mlp hidden 64, dropout 0.1, lr 3e-4, batch 8, at most 150 epochs, early stopping with patience 10 on the loss of a 15% holdout of **train** (stratified by task and outcome, chosen with the seed: the same episodes as the depth sweep's B6 probe), best epoch restored. seen_val is never used for stopping. `--early-stopping-patience 0` turns it off.
- Train-only z-scoring is on by default (`--no-normalize` is the raw-feature check). The sweep's single-layer reference should therefore also be run with `--normalize`.
- Episode score: the max step logit (`--aggregation max|mean|last`).

**Compact hyper-parameter search** (`search`, as the protocol asks within each leading experiment):
- The grid is lr {3e-4, 1e-3} × dropout {0.1, 0.3} × weight decay {0, 1e-4}, centred on B6 (the B6 point is one of the 8). Change it with `--lrs --dropouts --weight-decays`.
- Every point runs all three seeds. The winner has the best mean seen_val ROC-AUC.
- The cache is read once for the whole grid (and once for the whole K curve).
- Selection and reporting both use seen_val, so the winner's numbers are optimistic. `search.csv` keeps every grid point, and the sheet row says so in Notes.

**Confidence intervals:**
- Every run reports 95% bootstrap CIs over seen_val episodes (`--bootstrap 1000 --ci 0.95`). The scheme is the depth sweep's: resample stratified by outcome, use the same episodes for every seed, and take the percentile interval of the seed mean.
- `search` computes the CI only for the winning grid point.
- Per-step scores are saved as `seed<s>/seen_val_step_scores.npz`, in the same layout as the sweep's `rows/*.scores.npz`.
- `scripts/fusion_compare.py` scores both sides with one evaluator on the same resampled episodes. It reports the difference in points with its CI and the share of resamples where A is not better. Use it before claiming that fusion beats a single layer, or that F3 beats F2.

**Normalization floor:**
- The loader floors the train std at 1e-6, so a dimension that is constant on train can blow up to around 1e6 on seen_val. The runner warns when that happens.
- `--std-floor 1e-2` normalizes with `max(std, 1e-2)` instead. Use the same setting on both sides of a comparison.

**Latency:** `--latency-device cuda` measures on a GPU when one is visible. The Slurm job is CPU-only.

**Shared evaluator (B4, merged on main as PR #5): the default.** Runs use `failure_prob.engine.episode_metrics:make_evaluator` unless `--evaluator builtin` asks for the old scorer (SAFE-equivalent; PR-AUC is computed as SAFE does, so it can differ slightly from the built-in scorer; adds `roc_auc_taskmean`). In general: pass `--evaluator package.module:factory`, or a name registered in the depth sweep's `sweep_plugins.EVALUATORS` (PR #4, e.g. `temporary_max`). The contract is the sweep's: `Evaluator(EvalInput) -> dict`. The evaluator gets per-step scores (logits) of every seen_val episode, already cut to `task_min_step`. Once a non-TEMPORARY evaluator is used, the `episode_scorer` provisional flag disappears from every output.

**Duplicate-strongest-layer control (row 35, K4):** `--allow-duplicate-layers --layers 33 33 33` feeds one layer K times into the same fusion model. The layer file is read once per split.

## Output

In `--output-dir` of one run:
- `config.json`: run config, cache version, layer names, the train and seen_val horizons, the evaluator, and the git commit.
- `seed<s>/metrics.json`:
  - seen_val metrics and per-task seen_val metrics;
  - train metrics (a fit check at the train horizon, not comparable with seen_val);
  - learning curve, chosen epoch, trainable parameters, latency, and F3's learned layer weights.
- `seed<s>/seen_val_episode_scores.csv`: one row per episode (`trajectory_id, task_id, failure, n_steps, score, prob`), the input for bootstrap CIs. `score` is in logit space and is what the AUCs rank. `prob` is for reading only (probabilities saturate at 1.0).
- `seed<s>/seen_val_step_scores.npz`: per-step scores for bootstrap CIs and `fusion_compare.py`.
- `seed<s>/model.pt`: the checkpoint.
- `summary.json`: mean, std and bootstrap CI over seeds.
- `sheet_row.csv`: one row in the experiments sheet's columns.
  - Seen Val ROC-AUC / PR-AUC / TPR @ 5% FPR, in **percent (0–100)**, mean over seeds. Notes holds the std and the 95% CI.
  - **Parameters** (trainable).
  - **Latency (ms)**: median time of one detector call on ONE timestep (batch 1, no autograd), on the CPU it ran on. A deployed monitor makes one such call per step. Features are assumed already extracted, so this is what fusion adds on top of the VLA. `metrics.json` also keeps `ms_per_step_amortized` (whole episode in one call, divided by its steps), which is much smaller and is not the online cost.
  - The unseen-test columns stay empty until N3.

`search` adds `search.csv`, `best.json` and `sheet_rows.csv`, plus one `hp_<grid point>/` run per grid point. `kcurve` adds `K<k>/` runs (and `ref_L<l>/` with `--reference-layer`), `k_curve.csv` and `sheet_rows.csv`. `k_curve.csv` gives each K's change in points against K = 1 and against the best single layer.

The runners refuse to overwrite a non-empty folder unless you pass `--overwrite`. Even then they only replace folders they wrote themselves (marker file), and never the current directory, your home folder or the repo.

**Memory:** records are float32, about rows × K × 4096 × 4 bytes, plus one split's float16 layer files while they are read. Train at the full horizon is about 1.4 GB per layer. K ≤ 2 fits in 16G and K = 3–4 needs about 32G; the runner's estimate is printed in every log.

**Tested environments:** the suite passes with the CARC kit's versions (Python 3.10, torch 2.2.0, numpy 1.26.4, pandas 2.3.3, scikit-learn 1.7.2) and with newer ones. It also passes with PR #4 (depth sweep) merged in.

## Guarantees and what is still provisional

- **No unseen data.** Only `train` and `seen_val` are ever requested. Records that overlap between splits, have the wrong shape, are non-finite or repeat an ID are refused. The tests check that no `unseen_test` file is opened, including for `search` and `kcurve`.
- **Horizons are enforced.** Unknown horizons and duplicate or out-of-range layers are refused. Every episode is checked to have exactly the expected number of steps for its split.
- **Padding never reaches a score.**
- **Provisional flags.** Every output's `provisional` field is empty only for the B4 scorer with exactly the B6 probe settings; otherwise it lists what differs (the search's non-B6 grid points keep it on purpose). All numbers are on a single fixed split (SAFE seed-0 unseen tasks).
- **B7 overlap.** B7 (Aniruth) lists "static weights, uniform average". `static_weight_fusion` already covers both (`--fixed-weights` is the uniform average), so agree with Aniruth on one model before F3's real run.
