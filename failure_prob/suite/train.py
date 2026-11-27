"""SAFE objectives and the concurrent grid trainer.

All grid points of one (experiment, seed) are trained side by side. Each model
is built right after seeding with the training seed, so all points share the
initialization, and they consume the same shuffled batch order - exactly what
separate SAFE multirun processes with the same seed would do. Optimizers, LRs,
regularization coefficients and layer-dropout streams stay per point.
"""
from __future__ import annotations

import json
import os
import random
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .data import SplitData, class_weights
from .metrics import binary_metrics, episode_scores


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- SAFE scoring / loss
def step_scores(logits, mask, objective):
    p = torch.sigmoid(logits) * mask
    return p.cumsum(1) if objective == 'cumsum' else p


def safe_loss(logits, mask, labels, objective, w_fail, w_success):
    """IndepModel / LstmModel forward_compute_loss with SAFE's aggregate_monitor_loss."""
    m = mask.float()
    fail = labels[:, None]
    if objective == 'cumsum':
        s = step_scores(logits, m, 'cumsum')
        per_step = torch.where(fail == 1, -s, torch.relu(s))
    else:
        per_step = F.binary_cross_entropy_with_logits(logits, fail.expand_as(logits), reduction='none')
    seq = (per_step * m).sum(1) / m.sum(1)
    return (w_fail * (seq * (labels == 1)).sum() + w_success * (seq * (labels == 0)).sum()) / len(labels)


def reg_loss(model, coef):
    """SAFE compute_regularization_loss: coef * sum of squares of all non-bias parameters."""
    return coef * sum((p ** 2).sum() for n, p in model.named_parameters() if 'bias' not in n and p.requires_grad)


# --------------------------------------------------------------------------- batches
class Batch:
    def __init__(self, data: SplitData, episodes: torch.Tensor, sequential: bool):
        rows, self.mask = data.batch(episodes)
        self.labels = data.labels[episodes.to(data.device)]
        x = data.x[rows]
        if sequential:
            B, T = self.mask.shape
            self.x = x.new_zeros(B, T, *x.shape[1:])
            self.x[self.mask] = x
        else:
            self.x = x

    def logits(self, model, keep=None):
        if model.sequential:
            return model.seq_logits(self.x, self.mask)
        out = self.x.new_zeros(self.mask.shape)
        out[self.mask] = model.step_logits(self.x, keep)
        return out


def layer_keep(n, K, p, gen, device):
    """Per-timestep whole-layer dropout; at least one layer always kept."""
    keep = torch.rand(n, K, generator=gen, device=device) >= p
    empty = ~keep.any(1)
    if empty.any():
        pick = torch.randint(0, K, (int(empty.sum()),), generator=gen, device=device)
        keep[empty.nonzero().squeeze(1), pick] = True
    return keep


@torch.no_grad()
def predict(model, data: SplitData, batch_size: int = C.BATCH_SIZE) -> list[np.ndarray]:
    """Per-episode step-score curves (valid steps only)."""
    model.eval()
    out = []
    for start in range(0, data.n_episodes, batch_size):
        eps = torch.arange(start, min(start + batch_size, data.n_episodes))
        b = Batch(data, eps, model.sequential)
        s = step_scores(b.logits(model), b.mask, model.objective).cpu().numpy()
        out.extend(s[i, :n] for i, n in enumerate(b.mask.sum(1).tolist()))
    return out


def evaluate(model, data: SplitData) -> tuple[dict, list[np.ndarray]]:
    curves = predict(model, data)
    return binary_metrics(data.labels.cpu().numpy(), episode_scores(curves)), curves


def retry_io(fn, path: Path, attempts: int = 6, delay: float = 10.0):
    """Shared-filesystem writes occasionally fail transiently (seen on /scratch1: 'Parent directory
    ... does not exist' for an existing directory). Re-create the parent and retry with backoff."""
    for i in range(attempts):
        try:
            return fn()
        except (OSError, RuntimeError) as e:
            if i == attempts - 1:
                raise
            print(f'I/O retry {i + 1}/{attempts - 1} for {path}: {e}', flush=True)
            time.sleep(delay * (i + 1))
            Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_curves(path: Path, curves: list[np.ndarray], data: SplitData):
    retry_io(lambda: np.savez_compressed(
        path, scores=np.concatenate(curves).astype(np.float32), lengths=np.array([len(c) for c in curves]),
        labels=data.labels.cpu().numpy(), task_ids=data.task_ids, trajectory_ids=np.array(data.trajectory_ids)),
        path)


def load_curves(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    curves = np.split(z['scores'], np.cumsum(z['lengths'])[:-1])
    return dict(curves=curves, labels=z['labels'], task_ids=z['task_ids'], trajectory_ids=list(z['trajectory_ids']))


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(f'{path.suffix}.{os.getpid()}.tmp')

    def go():
        torch.save(obj, tmp)
        os.replace(tmp, path)
    retry_io(go, path)


def write_json(path: Path, obj):
    tmp = path.with_suffix(f'{path.suffix}.{os.getpid()}.tmp')
    text = json.dumps(obj, indent=2, default=str)

    def go():
        tmp.write_text(text)
        os.replace(tmp, path)
    retry_io(go, path)


# --------------------------------------------------------------------------- unit trainer
class Unit:
    """One (experiment, seed, grid point). Its init, batch order and layer-dropout stream depend only
    on (seed, config), so results are identical whichever worker / device / grouping trains it."""

    def __init__(self, exp, seed, cfg, directory: Path, build, spec):
        from .config import config_id
        self.exp, self.seed, self.cfg, self.dir, self.spec = exp, seed, cfg, Path(directory), spec
        self.cid = config_id(cfg)
        self.dir.mkdir(parents=True, exist_ok=True)
        seed_all(seed)
        self.model = build(cfg)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg['lr'])
        self.shuffle = torch.Generator().manual_seed(seed)
        dev = next(self.model.parameters()).device
        self.drop = torch.Generator(device=dev).manual_seed(seed * 100003 + zlib.crc32(self.cid.encode()))
        self.history, self.epoch, self.loss, self.t_train = [], 0, float('nan'), 0.0
        ckpt = self.dir / 'ckpt.pt'
        if ckpt.exists():
            st = torch.load(ckpt, map_location=dev, weights_only=False)
            self.model.load_state_dict(st['model'])
            self.opt.load_state_dict(st['opt'])
            self.shuffle.set_state(st['shuffle'].cpu())
            self.drop.set_state(st['drop'].cpu())
            self.history, self.epoch, self.t_train = st['history'], st['epoch'], st.get('t_train', 0.0)

    @property
    def name(self):
        return f'{self.exp}/seed{self.seed}/{self.cid}'

    def save_ckpt(self):
        atomic_save(dict(model=self.model.state_dict(), opt=self.opt.state_dict(), shuffle=self.shuffle.get_state(),
                         drop=self.drop.get_state(), history=self.history, epoch=self.epoch,
                         t_train=self.t_train), self.dir / 'ckpt.pt')

    def train_epoch(self, train: SplitData, w_fail, w_success):
        t = time.time()
        m, p = self.model, self.cfg.get('layer_dropout', 0.0)
        m.train()
        perm = torch.randperm(train.n_episodes, generator=self.shuffle)
        total_loss, n = 0.0, 0
        for start in range(0, train.n_episodes, C.BATCH_SIZE):
            b = Batch(train, perm[start:start + C.BATCH_SIZE], m.sequential)
            keep = layer_keep(b.x.shape[0], b.x.shape[1], p, self.drop, b.x.device) if p > 0 else None
            loss = safe_loss(b.logits(m, keep), b.mask, b.labels, m.objective, w_fail, w_success) \
                + reg_loss(m, self.cfg['reg'])
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            total_loss += float(loss.detach())
            n += 1
        self.epoch += 1
        self.loss = total_loss / n
        self.t_train += time.time() - t

    def finalize(self, train, val, epochs):
        val_m, val_curves = evaluate(self.model, val)
        self.history.append(dict(epoch=epochs, train_loss=self.loss, val_roc_auc=val_m['roc_auc']))
        atomic_save(dict(state_dict={k: v.cpu() for k, v in self.model.state_dict().items()}, spec=self.spec),
                    self.dir / 'model.pt')
        save_curves(self.dir / 'val_curves.npz', val_curves, val)
        result = dict(experiment=self.exp, seed=self.seed, config=self.cfg, config_id=self.cid, epochs=epochs,
                      trainable_params=self.model.n_params(), val=val_m, history=self.history,
                      spec=self.spec, checkpoint=str(self.dir / 'model.pt'),
                      device=str(train.device), train_seconds=round(self.t_train, 1),
                      worker=os.environ.get('SLURM_JOB_ID', 'local'), finished=time.strftime('%Y-%m-%dT%H:%M:%S'))
        write_json(self.dir / 'result.json', result)      # completion marker: written last
        (self.dir / 'ckpt.pt').unlink(missing_ok=True)
        return result


def train_units(units: list[Unit], train: SplitData, val: SplitData, epochs: int = C.EPOCHS, log=print):
    """Advance every unit one epoch at a time until all reach `epochs` (units may resume at different epochs)."""
    w_fail, w_success = class_weights(train)
    t0 = time.time()
    for u in units:
        if u.epoch:
            log(f'[{u.name}] resumed at epoch {u.epoch}')
    active = list(units)
    results = []
    while active:
        for u in list(active):
            u.train_epoch(train, w_fail, w_success)
            e = u.epoch
            if e == epochs:
                results.append(u.finalize(train, val, epochs))
                active.remove(u)
                log(f'[{u.name}] done: val ROC-AUC {results[-1]["val"]["roc_auc"]:.4f} '
                    f'({u.t_train / 60:.1f} train-min)')
                continue
            if (e - 1) % C.EVAL_EVERY == 0:
                val_m, _ = evaluate(u.model, val)
                u.history.append(dict(epoch=e, train_loss=u.loss, val_roc_auc=val_m['roc_auc']))
                log(f'[{u.name}] epoch {e}/{epochs} loss {u.loss:.4f} val AUC {val_m["roc_auc"]:.4f} '
                    f'({u.t_train / e:.2f}s/epoch, {time.time() - t0:.0f}s)')
            if e % C.CKPT_EVERY == 0:
                u.save_ckpt()
    return results
