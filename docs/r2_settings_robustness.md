# R2: Does the best layer depend on the probe settings?

Owner: Akash · Run and write-up: Rhea (covering, Nov 27) · Status: done

## Question
R1 picked L31 as the best layer using one probe setting (B6: lr 3e-4,
hidden 64, dropout 0.1). Is L31 still the best layer under other reasonable
settings, or is it an artefact of that one choice?

(The plan's original R2, a sweep "around" the best layer at sub-layer
positions, needs replay data we do not have; see
docs/orchestrator/08_DEPTH_SWEEP.md. Every layer is already cached, so R2 is
run as this settings-robustness check instead.)

## Runs (CARC, Nov 27)
- Layers L27–L32 (the top of the R1 depth curve).
- Probe settings: lr {1e-4, 3e-4} × hidden {64, 256} × dropout {0.1, 0.3} = 8.
  B6 (3e-4 / 64 / 0.1) is Akash's R1 run; the other 7 are new.
- Everything else as R1: `framework_mlp` probe with B6's early stopping (15%
  of train, loss), mean pooling, train-only standardization, B4 episode scorer,
  trained on train, scored on seen_val (140 episodes), seeds 0/1/2.
  unseen_test was not read.
- Submitted with `scripts/carc/submit_depth_sweep.sh --layers 27-32`;
  results in `/scratch1/pandita/depth_sweep/mean-trainnorm-*`; compared with
  `scripts/r2_compare.py` (this PR).

## Results (seen_val ROC-AUC ×100, 3-seed means)

| lr | hidden | dropout | L27 | L28 | L29 | L30 | L31 | L32 | Best | Best per-task |
|---|---|---|---|---|---|---|---|---|---|---|
| 1e-4 | 64 | 0.1 | 74.6 | 74.7 | 76.1 | 77.3 | **78.8** | 75.0 | L31 | L31 (71.9) |
| 1e-4 | 64 | 0.3 | 74.6 | 75.2 | 76.3 | 77.2 | **78.7** | 74.7 | L31 | L31 (72.2) |
| 1e-4 | 256 | 0.1 | 74.1 | 74.6 | 76.0 | 77.4 | **78.5** | 75.8 | L31 | L31 (71.4) |
| 1e-4 | 256 | 0.3 | 74.2 | 74.7 | 76.1 | 77.4 | **78.5** | 75.7 | L31 | L31 (71.7) |
| 3e-4 (B6) | 64 | 0.1 | 75.1 | 74.7 | 75.6 | 76.5 | **77.1** | 75.4 | L31 | L29 (70.5) |
| 3e-4 | 64 | 0.3 | 73.0 | 73.3 | 74.1 | 75.7 | **77.7** | 75.1 | L31 | L31 (70.8) |
| 3e-4 | 256 | 0.1 | 74.4 | 76.0 | 76.6 | 76.8 | **77.4** | 73.3 | L31 | L31 (70.8) |
| 3e-4 | 256 | 0.3 | 74.4 | 75.9 | 76.6 | 76.8 | **77.3** | 73.4 | L31 | L31 (70.8) |

## Finding
- **L31 is the best layer under all 8 settings (8/8), and L30 is second in
  all 8.** The R1 ranking is not an artefact of the B6 setting.
- L31 beats L32 (SAFE's final-layer feature) in all 8 settings, by 1.7–4.1
  points. Each single gap is still inside the bootstrap CIs (about ±7 at these
  layers), so this is a consistent direction, not a significant difference.
- On per-task ROC, L31 is also best in 7/8 (B6 is the exception: L29).
- lr 1e-4 gives the highest scores (L31 78.5–78.8 vs 77.1 with B6), but the
  gain is well inside the CIs. B6 stays locked; changing it would be a team
  decision (Aniruth) and would need the whole R1 sweep rerun.

**Decision: best layer = L31. S1 (L31 + L30) and the other S1–S6 layer sets
stand; select_layers.py does not need a rerun.**

Caveat: the settings were compared on seen_val, so seen_val is now used for
both layer and settings choice. unseen_test (N3) remains the only untouched
estimate.
