"""CLI:  python -m failure_prob.suite <command>

  status                         progress of every experiment
  plan                           the decided experiment table
  train EXP [EXP ...]            train all grid points x seeds, then select the config
  worker --target A|B|C          claim-and-train loop (Slurm workers; CPU and GPU share the pool)
  select EXP [EXP ...]           (re)select a config from finished runs
  select-layers [--handoff F]    3 layers for F3/F4/F5/X1 (seen_val only, or a teammate handoff)
  select-winner                  best fusion among F3/F4/F5/X1 -> rows 29-33 + K1
  n1 [--wait]                    row 36: per-task + early-fraction seen-val report, thresholds
  freeze [--agreed-by NAMES]     freeze finalists/hashes/thresholds for N3
  n3 --allow-unseen              row 37: single locked unseen benchmark
  smoke                          end-to-end synthetic test (SYNTHETIC, tiny, CPU)
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

from . import config as C


def device_arg(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Full fp32 everywhere (no TF32) so CPU- and GPU-trained units are numerically comparable.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if name == 'cpu' and os.environ.get('SLURM_CPUS_PER_TASK'):
        torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    return name


def print_plan():
    print(f"{'experiment':16s} {'row':28s} {'stage':5s} {'model':12s} {'layers':10s} compare")
    for n, e in C.EXPERIMENTS.items():
        if e.row == 'sweep':
            continue
        print(f'{n:16s} {e.row:28s} {e.stage:5s} {e.model:12s} {str(e.layers):10s} {e.compare}')
    print(f'(+ {sum(e.row == "sweep" for e in C.EXPERIMENTS.values())} raw-depth sweep probes; stage A = '
          f'{len(C.STAGES["A"])} experiments)')
    print(f'grid: lr {C.LRS} x reg {C.REGS} x seeds {C.SEEDS}; {C.EPOCHS} epochs, batch {C.BATCH_SIZE}, '
          f'width {C.HIDDEN}, Adam; pool={C.POOL}, normalize={C.NORMALIZE}, train horizon={C.TRAIN_HORIZON}, '
          f'eval horizon={C.EVAL_HORIZON}')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m failure_prob.suite', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--seeds', type=int, nargs='+', default=list(C.SEEDS))
    ap.add_argument('--epochs', type=int, default=C.EPOCHS)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('status')
    sub.add_parser('plan')
    p = sub.add_parser('train'); p.add_argument('experiments', nargs='+')
    p = sub.add_parser('worker'); p.add_argument('--target', choices=['A', 'B', 'C'], required=True)
    p.add_argument('--max-units', type=int, default=None, help='units trained together (default: 1 CPU, 12 GPU)')
    p.add_argument('--no-wait', action='store_true', help='exit instead of waiting for an earlier stage')
    p.add_argument('--index', type=int, help='ignored (Slurm array index)')
    p = sub.add_parser('select'); p.add_argument('experiments', nargs='+')
    p = sub.add_parser('select-layers'); p.add_argument('--handoff')
    p = sub.add_parser('select-winner'); p.add_argument('--candidates', nargs='+', default=list(C.FUSION_CANDIDATES))
    p = sub.add_parser('n1'); p.add_argument('--finalists', nargs='+'); p.add_argument('--wait', action='store_true'); p.add_argument('--n-boot', type=int, default=2000)
    p = sub.add_parser('freeze'); p.add_argument('--finalists', nargs='+'); p.add_argument('--agreed-by', default='')
    p = sub.add_parser('n3'); p.add_argument('--allow-unseen', action='store_true')
    sub.add_parser('smoke')
    a = ap.parse_args(argv)
    device = device_arg(a.device)

    if a.cmd == 'plan':
        return print_plan()
    if a.cmd == 'smoke':
        from .smoke import run
        return run()
    from . import evaluate, runner
    if a.cmd == 'status':
        for st in runner.STAGE_ORDER:
            print(f'stage {st}: {runner.stage_summary(st)}')
        for r in runner.status():
            if r['units_done'] or r['selected'] or C.EXPERIMENTS[r['experiment']].row != 'sweep':
                print(f"{r['experiment']:16s} {r['row']:28s} stage {r['stage']} units {r['units_done']}/{r['units']} "
                      f"{r['selected']:22s} {'' if r['val_roc_auc'] is None else round(100 * r['val_roc_auc'], 2)}")
    elif a.cmd == 'train':
        for n in a.experiments:
            stage = C.EXPERIMENTS[n].stage
            if stage != 'A':
                runner.prepare_stage(stage, device)
            runner.train_experiment(n, a.seeds, device, a.epochs)
    elif a.cmd == 'worker':
        mu = a.max_units or (12 if device == 'cuda' else 1)
        return runner.worker(a.target, device, mu, not a.no_wait, a.epochs)
    elif a.cmd == 'select':
        for n in a.experiments:
            runner.select_experiment(n, a.seeds)
    elif a.cmd == 'select-layers':
        runner.select_layers(a.handoff)
    elif a.cmd == 'select-winner':
        runner.select_winner(a.candidates, device)
    elif a.cmd == 'n1':
        if a.wait:
            rc = runner.worker('C', device, 1, True, a.epochs)   # helps finish / waits for C
            if rc:
                return rc
            runner.prepare_stage('N1', device)
        print(evaluate.run_n1(a.finalists, device, a.n_boot))
        if a.wait and os.environ.get('SLURM_JOB_ID'):
            runner.cancel_pending_gpu_workers()
    elif a.cmd == 'freeze':
        evaluate.freeze(a.finalists, a.agreed_by)
    elif a.cmd == 'n3':
        print(evaluate.run_n3(a.allow_unseen, device))


if __name__ == '__main__':
    sys.exit(main())
