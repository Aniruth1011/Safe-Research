"""RunTracker must work on Windows (no fcntl) as well as POSIX."""
import builtins
import importlib
import sys
import types

import failure_prob.engine.tracking as tracking


def _run_tracker(mod, tmp_path):
    tr = mod.RunTracker(output_root=tmp_path / "runs", index_path=tmp_path / "experiments" / "index.csv")
    run_dir, row = tr.create("r1", experiment_id="EXP_X", seed=0)
    tr.finish(row, "COMPLETED", val_roc_auc=0.5)
    text = (tmp_path / "experiments" / "index.csv").read_text()
    assert "r1" in text and "COMPLETED" in text


def test_posix_path(tmp_path):
    _run_tracker(tracking, tmp_path)


def test_windows_path_without_fcntl(tmp_path, monkeypatch):
    calls = []
    fake_msvcrt = types.SimpleNamespace(LK_LOCK=1, LK_UNLCK=0,
                                        locking=lambda fd, mode, n: calls.append((mode, n)))
    real_import = builtins.__import__

    def no_fcntl(name, *a, **k):
        if name == "fcntl":
            raise ImportError("No module named 'fcntl'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_fcntl)
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.delitem(sys.modules, "fcntl", raising=False)
    win = importlib.reload(tracking)
    try:
        assert win.fcntl is None
        _run_tracker(win, tmp_path)
        assert calls and calls[0] == (1, 1) and calls[1] == (0, 1)   # lock then unlock, every update
        assert len(calls) % 2 == 0
    finally:
        monkeypatch.undo()
        importlib.reload(tracking)
