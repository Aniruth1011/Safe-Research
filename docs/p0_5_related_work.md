# P0-5: Related-work update and narrowed novelty claim

Owner: Rhea · Reviewer: Dhyan · Status: draft for review

Every number below for our own work is **seen_val episode-level ROC-AUC (the
B4 primary metric, which `scripts/select_layers.py` ranks by) on our single
fixed split**. It is not comparable with SAFE's published 73.47 / 72.47 / 69.45,
which are averages over 3 splits. Our bootstrap 95% CIs are about ±7 points,
so the late layers are statistically tied.

## 1. What the three new papers do, and what they mean for us

| Paper | VLA / layer used | Layer study? | What it means for our claim |
|---|---|---|---|
| **Hide-and-Seek** (Park et al., ICML 2026, arXiv 2605.30834) | OpenVLA: **average over all layers** (and DoF). π0 / π0.5: hidden state before the velocity head | **Yes**, App. C.3: each of OpenVLA layers 0–31 vs the all-layer average | **Biggest threat.** The all-layer average is already their default, and they report it matches or beats the best single layer (85.2 / 83.4 bACC seen / unseen). Mid–late layers (20–31) are better and more stable than early ones, and the top layers are within ~3 bACC of each other. "Use intermediate layers / average them" is therefore **not new**. |
| **VLA-TuG** (Podolinsky et al., preprint, arXiv 2611.08425) | π0.5 backbone layer 18 (post-norm), GR00T-N1.7 layer 16; one layer each, pooled per modality | **No**. They state they do not analyse the choice of embedding systematically, and list "which layers contain the most generalizable failure signal" as future work | Leaves **room** for a controlled layer study, and names our question as open. Their contribution is elsewhere: pre-onset labels, calibrated trigger (FPR + early-miss), failure-mode head, LIBERO-Occ / LIBERO-X / real Kinova. |
| **SAE papers**: (a) Rogers, Lin, Brown (Utah, arXiv 2610.32208); (b) Swann et al. (Stanford, arXiv 2603.19183) | (a) OpenVLA-7B on LIBERO-10, **layer 0 residual stream only**, TopK SAE 2048×K=64. (b) π0.5 mainly (OpenVLA secondary), steering at PaliGemma layer 5 | **No** depth comparison in either | (a) Same model + suite as us. Raw vs SAE features give similar outcome AUROC (0.80 vs 0.81), so a cheap early-layer signal exists, but they never compare depths. (b) Interpretability / steering, not failure detection. Both are **complementary**: they ask *what* the features mean, we ask *which depth* to read. |

Plan note: the Task Plan says "SAE paper". (a) is the one that does failure
prediction on OpenVLA + LIBERO-10, so it is the one we compare against; (b) is
cited for context. **Dhyan, please confirm which one was meant.**

## 2. What our own results say so far (seen_val, ROC-AUC ×100)

| Detector | Pooled | Per-task mean |
|---|---|---|
| Best single layer, L31 (block31) | **77.1** | 68.4 |
| SAFE's layer, L32 (post-norm final) | 75.4 | 67.4 |
| S5 all-layer average | 73.7 | **69.8** |
| Score-level selection S1 / S2 / S3 | 77.4 / 77.0 / 77.7 | — |
| Feature fusion F1 / F2 / F3 (PR #11, not merged) | 74.99 / 74.49 / 75.59 | — |
| F6 top-K, K = 1…4 (PR #11, not merged) | 77.38 / 76.94 / 75.78 / 75.64 | — |
| SAFE repro MLP / LSTM / kNN (PR #7, not merged) | 75.52 / 74.18 / 67.65 | 63.4 / 68.4 / 51.9 |

SAFE repro PR-AUC (same runs, for reference): 74.94 / 75.32 / 69.48. All SAFE
repro numbers are 3-seed means, recomputed with our evaluator in C4.

Reading: no selected or fused detector is clearly better than the best single
late layer; all differences are inside the ±7 CI. The all-layer average is
worse pooled but best per-task. Adding layers to the fusion (F6 K=1→4) lowers
the score, which is consistent with extra capacity, not extra signal.

## 3. Narrowed novelty claim

**Old claim (plan note):** "Selected/adaptive layers beat strong single-layer
and all-layer-average controls."

The current evidence does not support this, and Hide-and-Seek already owns
"average all layers". **Proposed revised claim:**

> We give the first controlled comparison, for OpenVLA-7B on LIBERO-10 with
> SAFE's own detector and episode-level protocol, of *where* to read failure
> features: every one of the 34 residual-stream layers, SAFE's final layer,
> the all-layer average, and score- and feature-level layer selection/fusion,
> with matched capacity, train-only normalization, a locked unseen-task split
> and bootstrap CIs. We test whether layer selection or fusion beats the
> best single layer and the all-layer average on **unseen tasks**.

What is new versus the three papers: (1) Hide-and-Seek compares single layers
with the average only under its own contrastive loss and bACC; we test SAFE's
MLP/LSTM and SAFE's metric, and we add selection/fusion methods and
capacity-matched controls. (2) VLA-TuG and both SAE papers use one fixed
layer. (3) We report uncertainty, so a null result ("late layers are tied;
fusion does not help") is itself a usable finding.

Phrasing to **avoid** until the unseen_test numbers are in: "beats SAFE",
"intermediate layers are better", "fusion improves detection".

## 4. What would upgrade the claim

- Final unseen_test run (tasks 0, 3, 5; locked). If selection or the average
  beats L31/L32 there by more than the CI, the original claim comes back.
- Report per-task as well as pooled, since the average wins per-task.
- R2 fine sweep (Akash) → rerun `scripts/select_layers.py`.

## Sources

- Park et al., *Hide-and-Seek in Trajectories*, arXiv 2605.30834 (App. C.3).
- Podolinsky et al., *VLA-TuG: Failure Prediction and Classification in VLA
  Models via Calibrated Latent Probing*, arXiv 2611.08425.
- Rogers, Lin, Brown, *Do VLAs Know They Will Fail?*, arXiv 2610.32208.
- Swann et al., *Sparse Autoencoders Reveal Interpretable and Steerable
  Features in VLA Models*, arXiv 2603.19183.
- Gu et al., *SAFE*, arXiv 2506.09937.
