"""Row 36/N1 (seen-val per-task and early-fraction analysis) and row 37/N3 (locked unseen benchmark)."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from . import config as C
from .data import SplitData, verify_split_isolation
from .metrics import binary_metrics, bootstrap_auc_ci, episode_scores, threshold_at_fpr, thresholded
from .runner import DataPool, _out, load_detector, read_json, selection
from .train import Batch, load_curves, predict, step_scores, write_json

METRIC_KEYS = ('roc_auc', 'pr_auc', 'ap', 'tpr_at_5fpr')


def finalists(names=None) -> list[str]:
    names = names or [n for n in C.EXPERIMENTS if selection(n)]
    missing = [n for n in names if not selection(n)]
    if missing:
        raise RuntimeError(f'not selected: {missing}')
    return names


def _agg(per_seed: list[dict]) -> dict:
    out = {}
    for k in METRIC_KEYS:
        v = [m[k] for m in per_seed if m[k] is not None]
        out[k] = float(np.mean(v)) if v else None
        out[k + '_std'] = float(np.std(v, ddof=1)) if len(v) > 1 else None
    out['n'], out['n_fail'] = per_seed[0]['n'], per_seed[0]['n_fail']
    return out


def _frozen_params(name):
    sel = selection(name)
    spec = sel['spec']
    total = sel['trainable_params']
    if spec['input'] != 'cache':
        total += selection(spec['input']['experiment'])['trainable_params']
    return sel['trainable_params'], total


# --------------------------------------------------------------------------- latency
@torch.no_grad()
def latency_ms(name, T=200, repeats=20) -> dict:
    """CPU, 1 thread, B=1, a T-step rollout incl. any frozen fusion embedding; no VLA/IO/normalization."""
    torch.set_num_threads(1)
    model = load_detector(_out('finalists', name, 'seed0.pt'), 'cpu')
    stack = [model]
    while stack[0].spec['input'] != 'cache':
        stack.insert(0, load_detector(_out('finalists', stack[0].spec['input']['experiment'], 'seed0.pt'), 'cpu'))
    K = len(stack[0].spec['layers'])
    x = torch.randn(T, K, C.FEATURE_DIM)
    data = SplitData('latency', (), x, torch.tensor([0]), torch.tensor([T]), torch.tensor([0.0]),
                     np.array([0]), ['x'])

    def run():
        d = data
        for m in stack[:-1]:
            d = d.with_features(m.embed(d.x)[:, None])
        b = Batch(d, torch.tensor([0]), stack[-1].sequential)
        step_scores(b.logits(stack[-1]), b.mask, stack[-1].objective)

    for _ in range(3):
        run()
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        run()
        times.append(time.perf_counter() - t)
    med = float(np.median(times)) * 1000
    return dict(latency_ms_per_step=med / T, latency_ms_rollout=med, T=T, device='cpu', threads=1)


# --------------------------------------------------------------------------- N1
def _curves(name):
    sel = selection(name)
    return {s: load_curves(_out('finalists', name, f'seed{s}_val_curves.npz')) for s in sel['seeds']}


def verify_causal_prefix(name, device='cpu', fraction=0.25, tol=1e-4):
    """Recompute seed-0 early scores feeding ONLY the prefix and compare to the saved full-run curves."""
    model = load_detector(_out('finalists', name, 'seed0.pt'), device)
    val = DataPool(device).for_spec(model.spec, 'seen_val')
    ranks = ((val.lengths - 1).double() * fraction).round().long()
    prefix = SplitData(val.split, val.layers, val.x, val.starts, ranks + 1, val.labels, val.task_ids,
                       val.trajectory_ids)
    strict = np.array([c.max() for c in predict(model, prefix)])
    saved = episode_scores(load_curves(_out('finalists', name, 'seed0_val_curves.npz'))['curves'], fraction)
    err = float(np.max(np.abs(strict - saved) / (np.abs(saved) + 1e-6)))
    if err > tol:
        raise RuntimeError(f'{name}: prefix-only scores differ from full-run scores (rel err {err:.2e})')
    return err


def run_n1(names=None, device='cpu', n_boot=2000, check_causal=True) -> Path:
    names = finalists(names)
    out = _out('N1')
    out.mkdir(parents=True, exist_ok=True)
    pooled, per_task, sheet, thresholds = [], [], [], {}
    for name in names:
        sel = selection(name)
        curves = _curves(name)
        seeds = sorted(curves)
        labels = curves[seeds[0]]['labels']
        tasks = curves[seeds[0]]['task_ids']
        causal_err = verify_causal_prefix(name, device) if check_causal else None
        for f in C.FRACTIONS:
            for mode in ('max', 'at'):
                sc = {s: episode_scores(curves[s]['curves'], f, mode) for s in seeds}
                agg = _agg([binary_metrics(labels, sc[s]) for s in seeds])
                lo, hi = bootstrap_auc_ci(labels, [sc[s] for s in seeds], n_boot)
                pooled.append(dict(experiment=name, row=sel['row'], fraction=f, score=mode, **agg,
                                   roc_auc_ci95_lo=lo, roc_auc_ci95_hi=hi))
                for t in np.unique(tasks):
                    m = tasks == t
                    per_task.append(dict(experiment=name, fraction=f, score=mode, task_id=int(t),
                                         **_agg([binary_metrics(labels[m], sc[s][m]) for s in seeds])))
        full = {s: episode_scores(curves[s]['curves']) for s in seeds}
        thresholds[name] = {str(s): threshold_at_fpr(labels, full[s]) for s in seeds}
        trainable, total = _frozen_params(name)
        lat = latency_ms(name)
        top = next(r for r in pooled if r['experiment'] == name and r['fraction'] == 1.0 and r['score'] == 'max')
        sheet.append(dict(experiment=name, row=sel['row'], config=sel['config_id'],
                          layers=sel['spec']['layers'] or f"winner:{sel['spec']['input']['experiment']}",
                          seen_val_roc_auc=_pct(top['roc_auc']), seen_val_roc_auc_std=_pct(top['roc_auc_std']),
                          seen_val_roc_auc_ci95=f"[{_pct(top['roc_auc_ci95_lo'])}, {_pct(top['roc_auc_ci95_hi'])}]",
                          seen_val_pr_auc=_pct(top['pr_auc']), seen_val_ap=_pct(top['ap']),
                          seen_val_tpr_at_5fpr=_pct(top['tpr_at_5fpr']),
                          trainable_params=trainable, total_params=total,
                          latency_ms_per_step=round(lat['latency_ms_per_step'], 4),
                          causal_prefix_check_rel_err=causal_err, compare=sel['compare']))
        print(f"N1 {name}: ROC-AUC {sheet[-1]['seen_val_roc_auc']} ± {sheet[-1]['seen_val_roc_auc_std']}")
    _csv(out / 'n1_pooled.csv', pooled)
    _csv(out / 'n1_per_task.csv', per_task)
    _csv(out / 'results_sheet.csv', sheet)
    write_json(out / 'seen_val_thresholds.json', dict(rule=f'per seed, FPR<={C.TARGET_FPR} on seen_val '
                                                          'successes, full-horizon max score', thresholds=thresholds))
    _report(out, sheet, pooled)
    return out


def _pct(x):
    return None if x is None else round(100 * x, 2)


def _csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _report(out, sheet, pooled):
    lines = ['# N1: seen-val results (single fixed split, SAFE seed-0 unseen tasks)', '',
             'Mean ± std over training seeds 0/1/2 of the config chosen by seed-mean seen-val ROC-AUC; '
             '95% CI = episode bootstrap of the seed-mean AUC. Horizon = task_min_step; score = max-so-far.', '',
             '| Exp | Row | ROC-AUC | 95% CI | PR-AUC | TPR@5%FPR | Params | ms/step | vs |',
             '|---|---|---|---|---|---|---|---|---|']
    for r in sheet:
        lines.append(f"| {r['experiment']} | {r['row']} | {r['seen_val_roc_auc']} ± {r['seen_val_roc_auc_std']} | "
                     f"{r['seen_val_roc_auc_ci95']} | {r['seen_val_pr_auc']} | {r['seen_val_tpr_at_5fpr']} | "
                     f"{r['trainable_params']} | {r['latency_ms_per_step']} | {r['compare']} |")
    lines += ['', '## Early-fraction ROC-AUC (max-so-far)', '', '| Exp | 25% | 50% | 75% | 100% |', '|---|---|---|---|---|']
    for r in sheet:
        vals = [next(p for p in pooled if p['experiment'] == r['experiment'] and p['fraction'] == f
                     and p['score'] == 'max')['roc_auc'] for f in C.FRACTIONS]
        lines.append(f"| {r['experiment']} | " + ' | '.join(str(_pct(v)) for v in vals) + ' |')
    (out / 'N1_report.md').write_text('\n'.join(lines) + '\n')


# --------------------------------------------------------------------------- N3
def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _source_hash():
    files = sorted((C.REPO / 'failure_prob' / 'suite').glob('*.py'))
    return hashlib.sha256(b''.join(_sha(f).encode() for f in files)).hexdigest()


def _checkpoint_closure(name):
    """Checkpoints a finalist depends on (itself + frozen winner), all seeds."""
    sel = selection(name)
    names = [name] + ([sel['spec']['input']['experiment']] if sel['spec']['input'] != 'cache' else [])
    return {str(_out('finalists', n, f'seed{s}.pt')): _sha(_out('finalists', n, f'seed{s}.pt'))
            for n in names for s in sel['seeds']}


def freeze(names=None, agreed_by: str = '') -> Path:
    out = _out('N3')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'LOCK').exists():
        raise RuntimeError('N3 already ran; the frozen set cannot change')
    names = finalists(names)
    thr_path = _out('N1', 'seen_val_thresholds.json')
    if not thr_path.exists():
        raise RuntimeError('run n1 first: thresholds are frozen from seen_val')
    thr = read_json(thr_path)['thresholds']
    try:
        commit = subprocess.run(['git', '-C', str(C.REPO), 'rev-parse', 'HEAD'], capture_output=True,
                                text=True).stdout.strip()
    except OSError:
        commit = None
    frozen = dict(created=time.strftime('%Y-%m-%dT%H:%M:%S'), agreed_by=agreed_by, finalists=names,
                  selections={n: selection(n) for n in names}, thresholds={n: thr[n] for n in names},
                  checkpoints={n: _checkpoint_closure(n) for n in names}, suite_source_sha256=_source_hash(),
                  git_commit=commit, layer_selection=read_json(_out('layer_selection.json'))
                  if _out('layer_selection.json').exists() else None,
                  winner=read_json(_out('winner.json')) if _out('winner.json').exists() else None,
                  protocol='single fixed split (SAFE seed-0 unseen tasks)')
    write_json(out / 'frozen.json', frozen)
    print(f'frozen {len(names)} finalists -> {out / "frozen.json"}')
    return out / 'frozen.json'


def run_n3(allow_unseen: bool, device='cpu') -> Path:
    if not allow_unseen:
        raise PermissionError('N3 reads unseen_test; pass --allow-unseen explicitly, once, after the freeze')
    out = _out('N3')
    frozen = read_json(out / 'frozen.json')
    if frozen['suite_source_sha256'] != _source_hash():
        raise RuntimeError('suite source changed since freeze')
    for name, files in frozen['checkpoints'].items():
        for path, digest in files.items():
            if _sha(path) != digest:
                raise RuntimeError(f'checkpoint changed since freeze: {path}')
    iso = verify_split_isolation()
    if len(iso['unseen_tasks']) != 3:
        raise RuntimeError(f"expected exactly 3 unseen tasks, got {iso['unseen_tasks']}")
    fd = os.open(out / 'LOCK', os.O_CREAT | os.O_EXCL | os.O_WRONLY)   # single shot, never released
    os.write(fd, time.strftime('%Y-%m-%dT%H:%M:%S').encode())
    os.close(fd)
    pool = DataPool(device)
    pooled, per_task, raw = [], [], {}
    for name in frozen['finalists']:
        sel = frozen['selections'][name]
        per_seed, per_seed_thr, by_task = [], [], {}
        for s in sel['seeds']:
            model = load_detector(_out('finalists', name, f'seed{s}.pt'), device)
            data = pool.for_spec(model.spec, 'unseen_test', allow_unseen=True)
            curves = predict(model, data)
            labels = data.labels.cpu().numpy()
            raw[f'{name}/seed{s}'] = dict(scores=[float(c.max()) for c in curves], labels=labels.tolist(),
                                          task_ids=data.task_ids.tolist())
            for f in C.FRACTIONS:
                sc = episode_scores(curves, f)
                m = binary_metrics(labels, sc)
                per_seed.append(dict(fraction=f, seed=s, **m))
                if f == 1.0:
                    per_seed_thr.append(thresholded(labels, sc, frozen['thresholds'][name][str(s)]))
                    for t in np.unique(data.task_ids):
                        k = data.task_ids == t
                        by_task.setdefault(int(t), []).append(binary_metrics(labels[k], sc[k]))
        for f in C.FRACTIONS:
            pooled.append(dict(experiment=name, row=sel['row'], fraction=f,
                               **_agg([m for m in per_seed if m['fraction'] == f])))
        pooled[-1].update(frozen_threshold_tpr=float(np.mean([t['tpr'] for t in per_seed_thr])),
                          frozen_threshold_fpr=float(np.mean([t['fpr'] for t in per_seed_thr])))
        for t, ms in by_task.items():
            per_task.append(dict(experiment=name, task_id=t, **_agg(ms)))
        print(f"N3 {name}: unseen ROC-AUC {pooled[-1]['roc_auc']:.4f}")
    _csv(out / 'n3_pooled.csv', pooled)
    _csv(out / 'n3_per_task.csv', per_task)
    write_json(out / 'n3_raw_scores.json', raw)
    write_json(out / 'n3_done.json', dict(finished=time.strftime('%Y-%m-%dT%H:%M:%S'), isolation=iso))
    return out
