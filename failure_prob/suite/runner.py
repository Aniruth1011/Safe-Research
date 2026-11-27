"""Experiment resolution, training, hyperparameter / layer / winner selection.

Layout
  RUN_ROOT/<EXP>/seed<s>/<config_id>/{model.pt,result.json,val_curves.npz}  every grid point
  OUT_ROOT/selections/<EXP>.json         chosen config (seed-mean final-epoch seen-val ROC-AUC)
  OUT_ROOT/finalists/<EXP>/seed<s>.pt    the chosen config's checkpoints (durable copies)
  OUT_ROOT/layer_selection.json          3 layers: [anchor, complement1, complement2]
  OUT_ROOT/winner.json                   best fusion among FUSION_CANDIDATES
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import socket
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import config as C
from .data import load_split
from .metrics import episode_scores, fast_auc
from .models import FUSION_MODELS, build, width_for_params
from .train import load_curves, train_units, write_json


def _out(*parts) -> Path:
    return C.OUT_ROOT.joinpath(*parts)


def read_json(path):
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------- data with memo
class DataPool:
    """Memoizes loaded cache splits and frozen-winner embeddings for one process."""

    def __init__(self, device, cache=None):
        self.device, self.cache, self._memo = device, cache, {}

    def cache_split(self, split, layers, allow_unseen=False):
        key = ('cache', split, tuple(layers))
        if key not in self._memo:
            self._memo[key] = load_split(split, layers, self.device, self.cache, allow_unseen=allow_unseen)
        return self._memo[key]

    def for_spec(self, spec, split, allow_unseen=False):
        src = spec['input']
        if src == 'cache':
            return self.cache_split(split, spec['layers'], allow_unseen)
        key = ('embed', split, src['experiment'], src['seed'])
        if key not in self._memo:
            winner = load_detector(_out('finalists', src['experiment'], f"seed{src['seed']}.pt"), self.device)
            base = self.for_spec(winner.spec, split, allow_unseen)
            self._memo[key] = base.with_features(embed_all(winner, base.x)[:, None])
        return self._memo[key]


@torch.no_grad()
def embed_all(model, x, chunk=16384):
    model.eval()
    return torch.cat([model.embed(x[i:i + chunk]) for i in range(0, len(x), chunk)])


def load_detector(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    spec = ckpt['spec']
    model = build(spec['kind'], **spec['kwargs']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.spec = spec
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# --------------------------------------------------------------------------- resolution
def selected_layers() -> list[int]:
    path = _out('layer_selection.json')
    if not path.exists():
        raise FileNotFoundError(f'{path} missing: run `select-layers` after stage A')
    return read_json(path)['layers']


def winner() -> dict:
    path = _out('winner.json')
    if not path.exists():
        raise FileNotFoundError(f'{path} missing: run `select-winner` after stage B')
    return read_json(path)


def resolve(name: str, seed: int) -> dict:
    """Spec (kind, kwargs, layers, input) for one experiment and seed."""
    exp = C.EXPERIMENTS[name]
    kind, kwargs = exp.model, {}
    if exp.model == 'winner_arch':
        w = winner()
        kind, layers = w['model'], w['layers']
    elif exp.layers == 'selected':
        layers = selected_layers()
    elif exp.layers == 'winner':
        w = winner()
        return dict(kind=kind, kwargs=dict(input_dim=w['embed_dim']), layers=[],
                    input=dict(experiment=w['experiment'], seed=seed))
    else:
        layers = list(exp.layers)
    if kind in FUSION_MODELS:
        kwargs['n_layers'] = len(layers)
    if kind == 'probe_wide':
        target = winner()['trainable_params']
        kwargs['hidden'] = width_for_params(C.FEATURE_DIM, target)
    kwargs['input_dim'] = C.FEATURE_DIM
    return dict(kind=kind, kwargs=kwargs, layers=layers, input='cache')


# --------------------------------------------------------------------------- units
def unit_dir(name, seed, cfg) -> Path:
    return C.RUN_ROOT / name / f'seed{seed}' / C.config_id(cfg)


def units_of(name, seeds=C.SEEDS):
    return [(name, s, cfg) for s in seeds for cfg in C.grid(C.EXPERIMENTS[name])]


def make_units(keys, device):
    """Build Unit objects for (name, seed, cfg) keys that share one data spec."""
    from .train import Unit
    out = []
    for name, seed, cfg in keys:
        spec = resolve(name, seed)
        out.append(Unit(name, seed, cfg, unit_dir(name, seed, cfg),
                        lambda cfg, spec=spec: build(spec['kind'], **spec['kwargs']).to(device),
                        dict(spec, experiment=name, seed=seed, config=cfg)))
    return out


def train_experiment(name: str, seeds=C.SEEDS, device='cpu', epochs=C.EPOCHS, log=print):
    """Train all unfinished units of one experiment in this process (no claims), then select."""
    exp = C.EXPERIMENTS[name]
    log(f'== {name} ({exp.row}): {len(C.grid(exp))} grid points x seeds {list(seeds)}, {epochs} epochs, {device}')
    for seed in seeds:
        keys = [k for k in units_of(name, [seed]) if not (unit_dir(*k) / 'result.json').exists()]
        if not keys:
            continue
        pool = DataPool(device)
        spec = resolve(name, seed)
        train_units(make_units(keys, device), pool.for_spec(spec, 'train'), pool.for_spec(spec, 'seen_val'),
                    epochs, log)
    return select_experiment(name, seeds)


# --------------------------------------------------------------------------- claims / workers
STAGE_ORDER = ('A', 'B', 'C')


def owner_id() -> str:
    if os.environ.get('SLURM_ARRAY_JOB_ID') and os.environ.get('SLURM_ARRAY_TASK_ID'):
        return f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}"
    if os.environ.get('SLURM_JOB_ID'):
        return os.environ['SLURM_JOB_ID']
    return f'local:{socket.gethostname()}:{os.getpid()}'


_ALIVE: dict = {}


def owner_alive(owner: str) -> bool:
    """A claim is live while its Slurm job exists (running, or pending after a requeue)."""
    hit = _ALIVE.get(owner)
    if hit and time.time() - hit[1] < 300:
        return hit[0]
    if owner.startswith('local:'):
        _, host, pid = owner.split(':')
        if host != socket.gethostname():
            alive = True
        else:
            try:
                os.kill(int(pid), 0)
                alive = True
            except OSError:
                alive = False
    else:
        try:
            r = subprocess.run(['squeue', '-h', '-j', owner, '-o', '%T'], capture_output=True, text=True,
                               timeout=120)
            alive = bool(r.stdout.strip()) if r.returncode == 0 else 'Invalid job id' not in r.stderr
        except (OSError, subprocess.TimeoutExpired):
            alive = True                       # cannot tell: never steal
    _ALIVE[owner] = (alive, time.time())
    return alive


def unit_state(d: Path, me: str) -> str:
    if (d / 'result.json').exists():
        return 'done'
    if (d / 'FAILED').exists():
        return 'failed'
    claim = d / 'CLAIM'
    if not claim.exists():
        return 'free'
    owner = claim.read_text().strip()
    if owner == me:
        return 'mine'
    return 'taken' if owner_alive(owner) else 'stale'


def try_claim(d: Path, me: str) -> bool:
    d.mkdir(parents=True, exist_ok=True)
    state = unit_state(d, me)
    if state == 'mine':
        return True
    if state == 'free':
        try:
            fd = os.open(d / 'CLAIM', os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.write(fd, me.encode())
        os.close(fd)
        return True
    if state == 'stale':
        tmp = d / f'CLAIM.{os.getpid()}'
        tmp.write_text(me)
        os.replace(tmp, d / 'CLAIM')
        time.sleep(2)
        return (d / 'CLAIM').read_text().strip() == me
    return False


def stage_units(stage):
    return [k for name in C.STAGES[stage] for k in units_of(name)]


def stage_summary(stage, me='-'):
    counts = {}
    for k in stage_units(stage):
        st = unit_state(unit_dir(*k), me)
        counts[st] = counts.get(st, 0) + 1
    return counts


def stage_done(stage):
    return all((unit_dir(*k) / 'result.json').exists() for k in stage_units(stage))


def stage_failed(stage):
    return [str(unit_dir(*k)) for k in stage_units(stage) if (unit_dir(*k) / 'FAILED').exists()]


_PREPARED = set()


def prepare_stage(stage, device):
    """Selections that a stage needs; idempotent and safe to run concurrently (atomic writes)."""
    if stage in _PREPARED:
        return
    idx = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else len(STAGE_ORDER)
    if idx >= 1:
        for name in C.STAGES['A']:
            if selection(name) is None:
                select_experiment(name)
        if not _out('layer_selection.json').exists():
            select_layers()
    if idx >= 2:
        for name in C.STAGES['B']:
            if selection(name) is None:
                select_experiment(name)
        if not _out('winner.json').exists():
            select_winner(device=device)
    if idx >= 3:
        for name in C.STAGES['C']:
            if selection(name) is None:
                select_experiment(name)
    _PREPARED.add(stage)


def claim_batch(stage, me, max_units, prefer=None):
    """Claim up to max_units free units of one experiment+seed (same data); resume own claims first."""
    keys = stage_units(stage)
    if prefer:
        keys = sorted(keys, key=lambda k: (k[0], k[1]) != prefer)
    mine = [k for k in keys if unit_state(unit_dir(*k), me) == 'mine']
    group = mine[:1]
    pool = mine + keys
    for k in pool:
        if len(group) >= max_units:
            break
        if group and (k[0], k[1]) != (group[0][0], group[0][1]):
            continue
        if k in group:
            continue
        if try_claim(unit_dir(*k), me):
            group.append(k)
    return group


def cancel_unstarted_siblings(log=print):
    """Nothing is left to claim: cancel tasks of this array that never started (restarts=0, pending),
    so dependent stages (afterany) are not held back by idle queue entries. Requeued tasks are kept."""
    array = os.environ.get('SLURM_ARRAY_JOB_ID')
    if not array:
        return
    try:
        r = subprocess.run(['squeue', '-h', '-r', '-j', array, '-t', 'PD', '-o', '%i'], capture_output=True,
                           text=True, timeout=120)
        ids = r.stdout.split()
        cancel = []
        for i in ids:
            info = subprocess.run(['scontrol', 'show', 'job', i], capture_output=True, text=True, timeout=60).stdout
            if 'Restarts=0' in info:
                cancel.append(i)
        if cancel:
            subprocess.run(['scancel', *cancel], timeout=120)
            log(f'cancelled {len(cancel)} never-started sibling task(s) of array {array}')
    except (OSError, subprocess.TimeoutExpired) as e:
        log(f'could not cancel siblings: {e}')


def cancel_pending_gpu_workers(log=print):
    """After N1: queued GPU helpers have nothing left to do."""
    try:
        subprocess.run(['scancel', '-u', os.environ.get('USER', ''), '--state=PENDING',
                        '--name=suite-gpuA,suite-gpuB,suite-gpuC'], timeout=120)
        log('cancelled pending GPU helper jobs')
    except (OSError, subprocess.TimeoutExpired) as e:
        log(f'could not cancel GPU helpers: {e}')


class TrainingError(RuntimeError):
    """Training failed (unit marked FAILED / handed back); not a transient scheduling I/O error."""


def worker(target: str, device: str, max_units: int, wait_for_prereq: bool, epochs=C.EPOCHS,
           poll=300, log=print) -> int:
    """Claim-train loop over stages A..target. Exits 0 when nothing is claimable in its own stages."""
    me = owner_id()
    stages = STAGE_ORDER[:STAGE_ORDER.index(target) + 1]
    log(f'worker {me} target {target} device {device} max_units {max_units}')
    prefer = None
    io_errors = 0
    while True:
        try:
            outcome = _worker_step(stages, target, me, device, max_units, wait_for_prereq, epochs, prefer, log)
        except OSError as e:                 # transient /scratch1 metadata error (seen: Errno 5 on stat)
            io_errors += 1
            if io_errors > 20:
                raise
            log(f'filesystem error while scheduling ({e}); retry {io_errors}/20 in 60 s')
            time.sleep(60)
            continue
        io_errors = 0
        kind, value = outcome
        if kind == 'return':
            return value
        if kind == 'progressed':
            prefer = value
            continue
        time.sleep(poll)


def _worker_step(stages, target, me, device, max_units, wait_for_prereq, epochs, prefer, log):
    """One scheduling pass. Returns ('return', rc) | ('progressed', prefer) | ('sleep', None)."""
    progressed = blocked_on_prereq = False
    for stage in stages:
        if stage_done(stage):
            continue
        prev = STAGE_ORDER[STAGE_ORDER.index(stage) - 1] if stage != 'A' else None
        if prev and not stage_done(prev):
            blocked_on_prereq = True
            break
        prepare_stage(stage, device)
        group = claim_batch(stage, me, max_units, prefer)
        if group:
            name, seed = group[0][0], group[0][1]
            prefer = (name, seed)
            spec = resolve(name, seed)
            pool = DataPool(device)
            train, val = pool.for_spec(spec, 'train'), pool.for_spec(spec, 'seen_val')
            log(f'claimed {len(group)} unit(s) of {name} seed{seed}: {[C.config_id(k[2]) for k in group]}')
            try:
                train_units(make_units(group, device), train, val, epochs, log)
            except Exception as exc:
                for k in group:
                    d = unit_dir(*k)
                    if (d / 'result.json').exists():
                        continue
                    if device == 'cuda':      # GPU-only problem: hand the unit back to CPU workers
                        (d / 'CLAIM').unlink(missing_ok=True)
                        (d / f'GPU_ERROR.{me}').write_text(traceback.format_exc())
                    else:
                        (d / 'FAILED').write_text(traceback.format_exc())
                raise TrainingError(f'{name} seed{seed} failed') from exc
            del pool, train, val
            progressed = True
            break
        if stage != target:
            blocked_on_prereq = True      # earlier stage still running elsewhere
        break
    if all(stage_done(s) for s in stages):
        log(f'all stages up to {target} complete')
        cancel_unstarted_siblings(log)
        return 'return', 0
    if progressed:
        return 'progressed', prefer
    failed = [f for s in stages for f in stage_failed(s)]
    if failed:
        log(f'FAILED units block progress: {failed[:5]} ...')
        return 'return', 1
    if not (blocked_on_prereq and wait_for_prereq):
        log(f'nothing claimable for {me}; exiting ({ {s: stage_summary(s, me) for s in stages} })')
        cancel_unstarted_siblings(log)
        return 'return', 0
    return 'sleep', None


# --------------------------------------------------------------------------- selection
def _stats(values):
    v = [x for x in values if x is not None]
    return dict(mean=float(np.mean(v)) if v else None, std=float(np.std(v, ddof=1)) if len(v) > 1 else None,
                per_seed=values)


def _atomic_copy(src, dst):
    tmp = Path(f'{dst}.{os.getpid()}.tmp')
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def select_experiment(name: str, seeds=C.SEEDS) -> dict:
    """Pick the config with the best seed-mean final-epoch seen-val ROC-AUC (never one seed)."""
    exp = C.EXPERIMENTS[name]
    from .config import config_id
    rows, table = [], {}
    for cfg in C.grid(exp):
        cid = config_id(cfg)
        res = []
        for s in seeds:
            path = C.RUN_ROOT / name / f'seed{s}' / cid / 'result.json'
            if not path.exists():
                raise FileNotFoundError(f'{name}: missing {path}; training incomplete')
            res.append(read_json(path))
        table[cid] = res
        aucs = [r['val']['roc_auc'] for r in res]
        rows.append(dict(config_id=cid, **cfg, mean_val_roc_auc=float(np.mean(aucs)),
                         **{f'seed{s}_val_roc_auc': a for s, a in zip(seeds, aucs)}))
    best = max(rows, key=lambda r: r['mean_val_roc_auc'])      # max() keeps the first on ties
    chosen = table[best['config_id']]
    dest = _out('finalists', name)
    dest.mkdir(parents=True, exist_ok=True)
    for r in chosen:
        src = Path(r['checkpoint']).parent
        _atomic_copy(src / 'model.pt', dest / f"seed{r['seed']}.pt")
        _atomic_copy(src / 'val_curves.npz', dest / f"seed{r['seed']}_val_curves.npz")
    metrics = {k: _stats([r['val'][k] for r in chosen]) for k in ('roc_auc', 'pr_auc', 'ap', 'tpr_at_5fpr')}
    sel = dict(experiment=name, row=exp.row, compare=exp.compare, notes=exp.notes,
               config=chosen[0]['config'], config_id=best['config_id'], seeds=list(seeds),
               selection_rule='argmax over grid of mean final-epoch seen_val ROC-AUC across seeds',
               seen_val=metrics, trainable_params=chosen[0]['trainable_params'], spec=chosen[0]['spec'],
               epochs=chosen[0]['epochs'], finalists=str(dest))
    _out('selections').mkdir(parents=True, exist_ok=True)
    write_json(_out('selections', f'{name}.json'), sel)
    tmp = _out('selections', f'{name}_grid.csv.{os.getpid()}.tmp')
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, _out('selections', f'{name}_grid.csv'))
    print(f"{name}: selected {best['config_id']}  seen-val ROC-AUC {metrics['roc_auc']['mean']:.4f}"
          f" +/- {metrics['roc_auc']['std'] or 0:.4f}")
    return sel


def selection(name: str) -> dict | None:
    path = _out('selections', f'{name}.json')
    return read_json(path) if path.exists() else None


def _seed_mean_z(name):
    """Seed-averaged, per-seed z-scored full-horizon rollout scores of a finalist on seen_val."""
    sel = selection(name)
    zs, labels = [], None
    for s in sel['seeds']:
        c = load_curves(_out('finalists', name, f'seed{s}_val_curves.npz'))
        sc = episode_scores(c['curves'])
        zs.append((sc - sc.mean()) / (sc.std() + 1e-12))
        labels = c['labels']
    return np.mean(zs, 0), labels


def select_layers(handoff: str | None = None, k: int = 3) -> dict:
    """Anchor = best single raw-depth probe; complements = greedy score-ensemble gain.

    Uses only seen_val predictions of already-selected probes. A teammate's
    handoff JSON {"layers": [anchor, c1, c2], "source": "..."} overrides it.
    """
    if handoff:
        h = read_json(handoff)
        layers = [int(l) for l in h['layers']]
        if len(set(layers)) != k or not all(0 <= l <= 33 for l in layers):
            raise ValueError(f'handoff needs {k} distinct cache layers, got {layers}')
        out = dict(layers=layers, anchor=layers[0], complements=layers[1:], method='handoff',
                   source=h.get('source', str(handoff)), selection_split='seen_val')
    else:
        missing = [p for p in C.REQUIRED_PROBES if selection(p) is None]
        if missing:
            raise RuntimeError(f'probes not selected yet: {missing}')
        cand = {l: selection(f'P_L{l:02d}') for l in C.RAW_DEPTH if selection(f'P_L{l:02d}')}
        ranking = sorted(cand, key=lambda l: -cand[l]['seen_val']['roc_auc']['mean'])
        anchor = ranking[0]
        z = {l: _seed_mean_z(f'P_L{l:02d}') for l in cand}
        labels = z[anchor][1]
        chosen, trace = [anchor], []
        while len(chosen) < k:
            gains = {}
            for l in cand:
                if l in chosen:
                    continue
                ens = np.mean([z[c][0] for c in chosen + [l]], 0)
                gains[l] = float(fast_auc(labels[None].astype(float), ens[None])[0])
            nxt = max(gains, key=gains.get)
            trace.append(dict(added=nxt, ensemble_auc=gains[nxt], candidates=gains))
            chosen.append(nxt)
        out = dict(layers=chosen, anchor=anchor, complements=chosen[1:], method='greedy_score_ensemble',
                   selection_split='seen_val', candidate_pool=sorted(cand),
                   probe_ranking=[dict(layer=l, mean_val_roc_auc=cand[l]['seen_val']['roc_auc']['mean'])
                                  for l in ranking], greedy_trace=trace,
                   note='L32 (post-norm) excluded from the pool; L33 is the raw final block')
    write_json(_out('layer_selection.json'), out)
    print(f"layers {out['layers']} (anchor {out['anchor']}, method {out['method']})")
    return out


def select_winner(candidates=C.FUSION_CANDIDATES, device='cpu') -> dict:
    sels = {c: selection(c) for c in candidates}
    missing = [c for c, s in sels.items() if s is None]
    if missing:
        raise RuntimeError(f'fusion candidates not selected yet: {missing}')
    best = max(candidates, key=lambda c: sels[c]['seen_val']['roc_auc']['mean'])
    s = sels[best]
    m = load_detector(_out('finalists', best, 'seed0.pt'), device)
    embed_dim = int(m.out.in_features)
    out = dict(experiment=best, model=s['spec']['kind'], layers=s['spec']['layers'], config=s['config'],
               trainable_params=s['trainable_params'], embed_dim=embed_dim,
               seen_val_roc_auc=s['seen_val']['roc_auc'],
               rule='max seed-mean seen_val ROC-AUC among ' + ', '.join(candidates),
               table={c: sels[c]['seen_val']['roc_auc']['mean'] for c in candidates})
    write_json(_out('winner.json'), out)
    print(f"winner {best}: {out['table']}")
    return out


def status() -> list[dict]:
    rows = []
    for name, exp in C.EXPERIMENTS.items():
        sel = selection(name)
        keys = units_of(name)
        done = sum((unit_dir(*k) / 'result.json').exists() for k in keys)
        rows.append(dict(experiment=name, row=exp.row, stage=exp.stage, units_done=done, units=len(keys),
                         selected=sel['config_id'] if sel else '',
                         val_roc_auc=sel['seen_val']['roc_auc']['mean'] if sel else None))
    return rows
