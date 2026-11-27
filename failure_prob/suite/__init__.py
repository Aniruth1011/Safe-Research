"""Aniruth's LIBERO-10 detector suite on the shared CARC cache (v1).

One compact pipeline for the task-plan rows 5-10, 22-24, 28-33, 36 and 37:
GPU-resident cached features, the original SAFE objectives, the full SAFE
LR x regularization grid trained concurrently per seed, seen-val-only selection,
and a separately gated, single-shot unseen benchmark. See docs/suite.md.
"""
