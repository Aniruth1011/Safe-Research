# Glossary

Written 2026-11-10. Terms used across the project, in alphabetical order. Sources: the orchestrator's shared context
(verified 2026-11-10), the collection kit (`collection/`, PR #2 as merged in `upstream/main` `40a9f81`),
`failure_prob/protocol.py`, and the team task plan v5 (`VLA_Task_Plan_v5`). Plan rows are cited "per plan v5".
Background: [`01_PROJECT_OVERVIEW.md`](01_PROJECT_OVERVIEW.md). Live state: [`00_STATUS.md`](00_STATUS.md).

---

**A10.** The NVIDIA GPU type chosen (Akash, 2026-11-10) as the single GPU type for all collection, on Modal only.
Older text that says "L4" is outdated; the CARC measurements in our notes were taken on an **A40**, a different card.

**A40.** The NVIDIA GPU used on CARC for the 2026-11-09 verification run (measured: 273 ms per action, 3.29 steps/s,
14.74 GiB peak memory). Not used for the dataset: no CARC GPU jobs are planned.

**Action tokens.** The 7 tokens OpenVLA generates per control step, one per action dimension. Their ids are
31744–31999 in the vocabulary. The kit stores hidden states at these 7 positions, reduced by a *pooling*.

**All-layer average.** A control that averages every layer's feature into one vector before the detector (the
Hide-and-Seek style idea). Plan v5 row S5 (Rhea). Its exact definition (which indices, raw or standardized) is
still open: review-checklist item 4.

**Block.** One of the 32 transformer decoder layers of OpenVLA's Llama-2-7B backbone. Block k is
`language_model.model.layers[k-1]` in code (see T6). Block k's raw output is layer index k, except block 32 (see
*raw vs post-norm block 32*).

**bf16 / float16.** bf16 (bfloat16) is the precision the policy runs in and the precision of the raw `.layers.pt`
files. The cache stores features as float16 (`collection/build_cache.py`).

**Cache (shared cache).** The versioned, checksummed dataset built once from the raw rollouts by
`collection/build_cache.py`: `features/<pool>/<split>/L<NN>.npy`, `episodes.csv`, `splits.json`, `norm_stats/`,
`provenance/`, `manifest.json`, `CHECKSUMS.sha256`. Every detector reads this, never the 7B model.

**Capture check.** `python collection/check_cache.py capture <rollout_dir>`: checks that layer index 32 in the kit's
`.layers.pt` reproduces SAFE's own `.pkl` for the same episode (first and last pooling exactly, mean up to bf16
rounding), that no two layers are identical, and that there are no NaNs.

**CARC.** USC's research computing cluster. Used for the 2026-11-09 verification run and planned as shared storage
for the finished cache; not used for GPU collection (queues are days long).

**Controls.** Detectors our method must beat for the claim to hold: best single layer (K2), all-layer average (S5),
parameter-matched final-layer detector (K1), plus S6, K3, K4. See the overview, section 4.

**Depth curve.** Output of R1: seen-val detection quality against layer index. Input to layer selection (S1–S6).

**EGL / OSMesa.** Two ways MuJoCo renders images without a screen. EGL uses the GPU; OSMesa renders in software.
They can produce different pixels, hence different policy inputs, so one dataset must use one backend (choice E3).
EGL worked on CARC on 2026-11-09. Set with `MUJOCO_GL=egl`.

**Episode.** One rollout of the policy from one fixed initial state of one task, until success or `max_steps=520`.
LIBERO-10 has 500 (10 tasks x 50 initial states). Its label is `failure = 1 - success`.

**FeatureStore.** The repository's on-disk format for cached features that `scripts/train.py --feature-root` reads.
Class in `failure_prob/data/feature_store.py` (with `FeatureMetadata`). `collection/load_cache.py export` writes one
FeatureStore folder per split from the cache.

**flash_attention_2.** The attention implementation SAFE runs OpenVLA with (`attn_implementation="flash_attention_2"`,
package `flash-attn` 2.5.5). Other implementations (e.g. `sdpa`) change the numbers slightly, so the whole dataset
must use this one. Kit fix 3 (in progress) makes collection fail if flash-attn is missing instead of switching to `sdpa`.

**Hidden states.** The per-layer activations HuggingFace returns with `output_hidden_states=True`: 33 tensors for
this model (embeddings plus 32 blocks), each 4096 wide.

**Invalid episode.** An episode that crashed during collection. SAFE's script stops the whole job on any exception;
kit fix 4 (in progress) records the crash in a marker file `task{t}--ep{e}--invalid.json` instead, continues, and lets
`build_cache.py` accept a documented list of such gaps (choice E2).

**Kit (collection kit).** Rhea's PR #2: `collection/` (patch, `build_cache.py`, `check_cache.py`, `load_cache.py`,
`carc/` scripts, README) plus `tests/test_shared_cache.py`. Merged without the five fixes; the fixes come as a
follow-up PR (plan v5 row A1).

**Kit fixes (five must-fixes).** In progress: (1) Modal app, (2) environment pins, (3) flash-attn hard-fail,
(4) invalid-episode handling, (5) raw block 32 as an extra layer index. Plan v5 row A1.

**Layer index 0–33.** Our numbering of stored layers, equal to the HuggingFace `hidden_states` index. 0 = embeddings;
1–31 = raw outputs of blocks 1–31; 32 = block 32 *after* the final RMSNorm (= SAFE's feature). 33 = raw block 32
*before* the norm, planned as an extra index by kit fix 5 (in progress, choice E1). "L4" in the plan means index 4.

**LIBERO-10.** The benchmark: 10 long-horizon manipulation tasks in MuJoCo simulation, 50 fixed initial states each.
Also called LIBERO-Long.

**Locked unseen test.** The single final evaluation on unseen tasks 0, 3, 5 (plan v5 row N3, Aniruth). Their
features sit in `unseen_test/` folders; `load_cache.py` refuses them without `--allow-unseen`.

**Modal.** The cloud GPU service used for all GPU work (decided 2026-11-10). Budget: $30 hard ceiling, stop and ask
at $25. Nothing spent as of 2026-11-10.

**`norm_stats/`.** Per-layer mean and standard deviation computed on the **train** split only, stored in the cache.
Optional; applied only with `normalize=True` / `--normalize`. Plan v5 row R4 compares raw vs standardized.

**OpenVLA.** The frozen policy: `openvla/openvla-7b-finetuned-libero-10`, Llama-2-7B backbone, run through SAFE's
fork `vla-safe/openvla` pinned at commit `300dce2`. Never trained.

**Patch (`openvla_all_layers.patch`).** The kit's patch to SAFE's OpenVLA fork. Adds `--save_all_layers` to
`experiments/robot/libero/run_libero_eval.py` and a helper `layer_cache.py`. Never hand-edited; regenerated from a branch.

**Plan row IDs.** Row IDs in the team task plan (per plan v5): **G0** git cleanup (Akash, done 2026-11-10);
**P0-x** spec and setup; **R0-PR2** review of PR #2; **A1–A9** extraction and collection (A1 kit fixes, A5 smoke
test, A6 sizing, A8 full run); **B1–B8** pipeline repair; **C1–C3** SAFE baselines, **C4** cross-check, **C5**
end-to-end gate; **R1–R4** probing sweep; **S1–S6** layer selection; **F1–F6** fusion; **K1–K4** controls;
**N1–N3** analysis (N3 = locked unseen test); **M-PR2** merge of the fix PR; **X1–X6** stretch; **D1–D2**
deliverables (not to be confused with doc agents D1–D6 in `00_STATUS.md`). The kit README's "Exp 14", "Exp 37" use the older
experiment-sheet numbering (the plan's "Old sheet row" column: Exp 14 = R3, Exp 37 = N3).

**Pooling (first / last / mean).** How the 7 action-token hidden states become one vector per layer per step:
`first` (SAFE `token_idx_rel=0.0`), `last` (`1.0`) or `mean` (`mean`, computed in float32). The kit stores all three
by default (`all_layers_pooling = "mean,first,last"`). The primary pooling is chosen once (checklist item 4; R3 checks it).

**Probing sweep.** R1–R4 (Akash): train the same small detector on each layer in turn (R1), refine around the best
(R2), check pooling (R3) and normalization (R4). Settings come from the shared probe spec (B6).

**Provenance.** The record of how data was made: package versions, repo commits, checkpoint, settings. The patch
writes provenance JSON per run; `build_cache.py` copies it (and extra files given with `--provenance-files`) into
the cache's `provenance/` folder before checksumming.

**Raw vs post-norm block 32.** Llama applies a final RMSNorm after block 32. HuggingFace's last hidden state (index
32, SAFE's feature) is the **post-norm** output; the **raw** output before the norm is not in `hidden_states`.
Kit fix 5 (in progress) captures it with a forward pre-hook on the final norm and stores it as index 33.

**Resume.** Re-running collection with `--resume True` skips episodes already complete (their `.csv` exists).

**SAFE.** The baseline paper and codebase (Gu et al., NeurIPS 2025, arXiv 2506.09937) this repo is forked from: a
failure detector on the policy's **final-layer** features, built to generalize to unseen tasks. Original code:
`failure_prob/` (do not edit `model/`, `data/`, `conf/`, `utils/`).

**SAFE `.pkl` / `.csv`.** SAFE's per-episode output files: features (final layer at the 7 action tokens, plus token
probabilities) and a summary row. The kit leaves them unchanged, so the SAFE baselines (C1–C3) read them directly.

**Seen / unseen tasks.** Unseen tasks are 0, 3, 5 (SAFE's seed-0 task split, `failure_prob/protocol.py`
`UNSEEN_TASKS`); the other 7 are seen. Seen episodes split about 60/40 into `train` and `seen_val`, stratified by
outcome, saved as explicit IDs in `splits.json`. All tuning and selection use `seen_val` only.

**Seeds.** `failure_prob/protocol.py`: rollout seed 7, split seed 0 (used once), training seeds 0, 1, 2.

**Smoke test.** A short end-to-end check on one GPU before the full run: headless render, resume, capture check,
repeat run, tiny cache build, plus T6. Plan v5 row A5, on one Modal A10, budget under $3. Not started as of 2026-11-10.

**`splits.json`.** The cache file listing the exact episode IDs in `train`, `seen_val` and `unseen_test`. It, not the
seed, is the source of truth; rebuild with `--split-file`, never by regenerating.

**T6.** `tests/gpu/test_layer_mapping.py`, a GPU-only test proving the layer convention: every `hidden_states[k]`
equals its block hook output, the final norm applied to the pre-hook input equals index 32, and action tokens are
unchanged with hooks attached. Passed **bitwise exact** on CARC (A40), per the shared context dated 2026-11-09.

**`task_min_step`.** The shortest episode length within a task (column in the cache's `episodes.csv`). Evaluation
truncates every episode of that task to it, because failed episodes run to 520 steps and length would leak the label.

**uv.** A fast Python package and virtual-environment tool; kit fix 2 (in progress) adds it as a non-conda install path.

**Worktree.** A separate working copy of a git repository on another branch, under `worktrees/` on Akash's laptop.
Each kit fix is developed in its own worktree.
