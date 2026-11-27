"""End-to-end SYNTHETIC smoke test of the whole suite (tiny dims, 3 epochs, CPU).

Builds a fake cache with the real file layout, then runs stage A (required
probes + LSTM), layer selection, stage B, winner selection, stage C, N1,
freeze and N3. Outputs are SYNTHETIC_ONLY and deleted afterwards.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

SEEN = (1, 2, 4, 6, 7, 8, 9)


def make_cache(root: Path, D=16, n_layers=34, seed=0):
    rng = np.random.default_rng(seed)
    rows, feats = [], {s: [] for s in ('train', 'seen_val', 'unseen_test')}
    counts = dict(train=6, seen_val=4, unseen_test=4)
    signal = rng.normal(size=(n_layers, D))
    for task in range(10):
        splits = ['unseen_test'] if task in C.UNSEEN_TASKS else ['train', 'seen_val']
        lengths = {}
        for split in splits:
            for e in range(counts[split]):
                fail = e % 2
                length = int(rng.integers(12, 20)) + (8 if fail else 0)
                lengths[(split, e)] = (length, fail)
        tms = min(l for l, f in lengths.values())
        for (split, e), (length, fail) in lengths.items():
            t = np.linspace(0, 1, length)[:, None, None]
            x = rng.normal(size=(length, n_layers, D)) + fail * 0.8 * t * signal[None]
            feats[split].append(x.astype(np.float16))
            rows.append(dict(trajectory_id=f'libero_10/t{task:02d}/{split}{e:02d}', task_id=task,
                             task_name=f'task {task}', episode_idx=e, success=1 - fail, failure=fail,
                             length=length, task_min_step=tms, split=split))
    ep = pd.DataFrame(rows)
    for split in feats:
        m = ep.split == split
        lens = ep.length[m].to_numpy()
        ep.loc[m, 'row_start'] = np.cumsum(lens) - lens
        ep.loc[m, 'row_end'] = np.cumsum(lens)
        arr = np.concatenate(feats[split])
        d = root / 'features' / 'mean' / split
        d.mkdir(parents=True)
        for l in range(n_layers):
            np.save(d / f'L{l:02d}.npy', arr[:, l])
        if split == 'train':
            (root / 'norm_stats').mkdir()
            np.savez(root / 'norm_stats' / 'mean.npz', layers=np.arange(n_layers), mean=arr.mean(0).astype(np.float32),
                     std=arr.std(0).astype(np.float32), n_rows=len(arr))
    ep[['row_start', 'row_end']] = ep[['row_start', 'row_end']].astype(int)
    ep.to_csv(root / 'episodes.csv', index=False)
    (root / 'splits.json').write_text(json.dumps(dict(
        unseen_tasks=list(C.UNSEEN_TASKS),
        trajectories={s: list(ep.trajectory_id[ep.split == s]) for s in feats})))
    (root / 'manifest.json').write_text(json.dumps(dict(SYNTHETIC_ONLY=True)))


def run():
    base = Path(os.environ.get('SUITE_SMOKE_DIR', f"/scratch1/{os.environ.get('USER', 'user')}/tmp/suite_smoke_{os.getpid()}"))
    if base.exists():
        raise RuntimeError(f'{base} exists')
    try:
        make_cache(base / 'cache')
        C.CACHE, C.RUN_ROOT, C.OUT_ROOT = base / 'cache', base / 'runs', base / 'out'
        C.FEATURE_DIM = 16
        from . import evaluate, runner
        epochs = 3
        C.STAGES['A'] = ['X2_LSTM'] + C.REQUIRED_PROBES          # smaller sweep for the smoke test
        quiet = lambda *_: None
        # resume path: a unit interrupted after 2 epochs must continue from its checkpoint
        k = runner.units_of('P_L04')[0]
        u = runner.make_units([k], 'cpu')[0]
        spec = runner.resolve('P_L04', 0)
        pool = runner.DataPool('cpu')
        u.train_epoch(pool.for_spec(spec, 'train'), 1.0, 1.0); u.epoch = 2; u.save_ckpt()
        assert runner.make_units([k], 'cpu')[0].epoch == 2
        # one GPU-style worker (12 units at a time) and one CPU-style worker share the pool
        assert runner.worker('A', 'cpu', 12, False, epochs, log=quiet) == 0
        assert runner.stage_done('A')
        assert runner.worker('C', 'cpu', 1, True, epochs, poll=1, log=quiet) == 0
        assert runner.worker('C', 'cpu', 12, False, epochs, log=quiet) == 0      # nothing left: exits
        runner.prepare_stage('N1', 'cpu')
        evaluate.run_n1(device='cpu', n_boot=50)
        evaluate.freeze(agreed_by='SYNTHETIC smoke')
        evaluate.run_n3(allow_unseen=True, device='cpu')
        try:
            evaluate.run_n3(allow_unseen=True, device='cpu')
            raise AssertionError('second N3 run was not refused')
        except FileExistsError:
            pass
        print((C.OUT_ROOT / 'N1' / 'N1_report.md').read_text())
        print('SMOKE OK (SYNTHETIC_ONLY; not results)')
    finally:
        shutil.rmtree(base, ignore_errors=True)
