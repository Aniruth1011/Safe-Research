"""Frozen protocol constants and the experiment registry.

Every number here is the decided setting for this suite; changing one means a
new run root. See docs/suite.md for the rationale of each choice.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# --- Paths (override with environment variables) ---------------------------
CACHE = Path(os.environ.get('SAFE_CACHE', '/scratch1/akashrav/safe_cache/v1'))
# Checkpoints for every grid point (large, re-creatable) live on scratch.
RUN_ROOT = Path(os.environ.get('SUITE_RUN', f"/scratch1/{os.environ.get('USER', 'user')}/safe_suite/v2"))   # v2: SAFE-faithful raw features + full-rollout training
# Selected checkpoints, selections and reports (small, must survive) live in the repo outputs/.
OUT_ROOT = Path(os.environ.get('SUITE_OUT', str(REPO / 'outputs' / 'suite')))

# --- SAFE training protocol (scripts/batch_training/submit_openvla_libero.bash + conf) ---
EPOCHS = 1000
BATCH_SIZE = 64
HIDDEN = 256
LRS = (1e-4, 3e-4, 1e-3)
REGS = (1e-3, 1e-2, 1e-1, 1.0)
SEEDS = (0, 1, 2)
EVAL_EVERY = 25          # SAFE roc_every; history only, selection uses the final epoch
CKPT_EVERY = 10          # per-unit resumable checkpoint cadence (epochs)
POOL = 'mean'
NORMALIZE = False        # SAFE default (normalize_hidden_states=False): raw cached features
TRAIN_HORIZON = 'full'   # SAFE trains on whole rollouts
EVAL_HORIZON = 'task_min_step'   # every reported metric scores only the first task_min_step steps
FRACTIONS = (0.25, 0.5, 0.75, 1.0)
TARGET_FPR = 0.05
UNSEEN_TASKS = (0, 3, 5)

FEATURE_DIM = int(os.environ.get('SUITE_FEATURE_DIM', 4096))   # OpenVLA-7B hidden size

# --- Layer conventions of the shared cache ----------------------------------
SAFE_FINAL = 32                       # post-final-RMSNorm, identical to SAFE's feature
RAW_FINAL = 33                        # raw block-32 output (depth comparisons)
RAW_DEPTH = tuple(range(1, 32)) + (RAW_FINAL,)   # candidate pool for layer selection
COARSE = {5: 4, 6: 8, 7: 12, 8: 16, 9: 20, 10: 24}   # sheet row -> layer
FUSION_CANDIDATES = ('F3_STATIC', 'F4_GATED', 'F5_RESIDUAL', 'X1_ATTN')
LAYER_DROPOUT_P = (0.1, 0.2, 0.3)
N_PROMPTS = 8


@dataclass(frozen=True)
class Experiment:
    name: str
    row: str                  # task-plan sheet row / task id
    model: str                # key in models.build
    layers: str | tuple       # fixed tuple, 'selected', or 'winner' (frozen winner embedding)
    compare: str
    stage: str                # A: independent, B: needs layer selection, C: needs fusion winner
    extra_grid: dict = field(default_factory=dict)
    notes: str = ''


def probe(layer: int, row: str = '-') -> Experiment:
    return Experiment(f'P_L{layer:02d}', row, 'probe', (layer,), 'P_L32 (final-layer SAFE MLP)', 'A',
                      notes='SAFE MLP probe on one cache layer')


EXPERIMENTS: dict[str, Experiment] = {}
for _row, _layer in COARSE.items():
    EXPERIMENTS[f'P_L{_layer:02d}'] = probe(_layer, f'{_row}/R1')
EXPERIMENTS['P_L32'] = probe(SAFE_FINAL, 'control: SAFE MLP / final-layer MLP')
EXPERIMENTS['P_L33'] = probe(RAW_FINAL, 'control: raw final block MLP')
for _layer in RAW_DEPTH:            # full raw-depth sweep: candidate pool for layer selection
    EXPERIMENTS.setdefault(f'P_L{_layer:02d}', probe(_layer, 'sweep'))

EXPERIMENTS.update({e.name: e for e in [
    Experiment('X2_LSTM', '28/X2', 'lstm', (SAFE_FINAL,), 'P_L32', 'A',
               notes='SAFE LstmModel on SAFE final-layer feature, full causal history'),
    Experiment('F3_STATIC', 'control for 22/F4', 'static', 'selected', 'F4_GATED', 'B',
               notes='learned static softmax layer weights (Dhyan F3 equivalent)'),
    Experiment('F4_GATED', '22/F4', 'gated', 'selected', 'F3_STATIC', 'B',
               notes='per-timestep input-conditioned softmax gates'),
    Experiment('F5_RESIDUAL', '23/F5', 'residual', 'selected', 'best single layer probe', 'B',
               notes='anchor projection + 2 ReZero nonlinear complement branches'),
    Experiment('X1_ATTN', '24/X1', 'attention', 'selected', 'F4_GATED', 'B',
               notes='4-head self-attention across the 3 layer tokens per timestep'),
    Experiment('K1_SIZEMATCH', 'control for 37/N3', 'probe_wide', (SAFE_FINAL,), 'winning fusion', 'C',
               notes='L32 SAFE MLP with width matched (<=5%) to the winner trainable params'),
    Experiment('X2_GRU', '29/X2', 'gru', 'winner', 'winning fusion (static over time)', 'C',
               notes='GRU over frozen winner embedding'),
    Experiment('X2_DELTA', '30/X2', 'gru_delta', 'winner', 'X2_GRU', 'C',
               notes='GRU over [embedding, first difference]'),
    Experiment('X2_CAUSAL', '31/X2', 'causal_tf', 'winner', 'X2_GRU', 'C',
               notes='1-block 4-head causal transformer over frozen winner embedding'),
    Experiment('X3_PROMPT', '32/X3', 'prompt', 'winner', 'X3_PROMPT_CTRL', 'C',
               notes='8 learned failure prompts, input-conditioned attention, SAFE MLP head'),
    Experiment('X3_PROMPT_CTRL', 'control for 32/X3', 'adapter', 'winner', 'X3_PROMPT', 'C',
               notes='parameter-matched residual adapter, no prompts'),
    Experiment('X4_LDROP', '33/X4', 'winner_arch', 'selected', 'winning fusion', 'C',
               extra_grid={'layer_dropout': LAYER_DROPOUT_P},
               notes='winner architecture retrained with per-timestep layer dropout'),
]})

STAGES = {
    # ordered longest-first: workers claim in this order, so slow units never form the tail
    'A': ['X2_LSTM', 'P_L32'] + [f'P_L{l:02d}' for l in RAW_DEPTH],
    'B': ['X1_ATTN', 'F5_RESIDUAL', 'F4_GATED', 'F3_STATIC'],
    'C': ['X4_LDROP', 'X2_CAUSAL', 'X2_DELTA', 'X2_GRU', 'K1_SIZEMATCH', 'X3_PROMPT', 'X3_PROMPT_CTRL'],
}
# The six coarse probes the sheet asks for (rows 5-10) plus their two references.
REQUIRED_PROBES = [f'P_L{l:02d}' for l in COARSE.values()] + ['P_L32', 'P_L33']


def grid(exp: Experiment) -> list[dict]:
    """All hyperparameter points of one seed. Seeds are the outer loop."""
    points = [dict(lr=lr, reg=reg) for lr in LRS for reg in REGS]
    for key, values in exp.extra_grid.items():
        points = [dict(p, **{key: v}) for p in points for v in values]
    return points


def config_id(cfg: dict) -> str:
    return '_'.join(f'{k}{v:g}' for k, v in sorted(cfg.items()))
