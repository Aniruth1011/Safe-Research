"""The shared CARC cache withholds unseen_test by permissions (features/<pool>/unseen_test is drwx------, owner only).
Dhyan's tools must work on such a copy without touching the locked folders, and the loader must say why it fails."""
import os
import shutil
import stat
import subprocess
import sys

import pytest

from test_fusion_runner import cache, cfg  # noqa: F401  (module-scoped 34-layer cache)
from test_a9_loader_export import ENV
from test_shared_cache import KIT
import load_cache
from failure_prob import fusion_experiments as FX
from failure_prob import fusion_runner as FR

pytestmark = pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0,
                                reason="permission locks do not apply to root")


@pytest.fixture
def locked(cache, tmp_path):  # noqa: F811
    copy = tmp_path / "v1"
    shutil.copytree(cache, copy)
    dirs = sorted((copy / "features").glob("*/unseen_test"))
    assert dirs
    for d in dirs:
        d.chmod(0)
    yield copy
    for d in dirs:
        d.chmod(stat.S_IRWXU)


def test_loader_explains_the_lock(locked):
    c = load_cache.SharedCache(locked)
    with pytest.raises(PermissionError, match="locked"):
        c.records("unseen_test", layers=[33])
    with pytest.raises(PermissionError, match="withheld until the locked final benchmark"):
        c.records("unseen_test", layers=[33], allow_unseen=True)
    assert c.records("seen_val", layers=[33], horizon="task_min_step")


def test_fusion_run_search_and_a9_work_on_a_locked_cache(locked, tmp_path):
    s = FR.run(cfg(tmp_path / "run", cache=str(locked), seeds=[0], bootstrap=20))
    assert s["metrics"]["roc_auc"]["n_seeds_defined"] == 1
    FX.k_ablation(cfg(tmp_path / "k", cache=str(locked), layers=[16, 33], seeds=[0], bootstrap=0), ks=(1, 2))
    r = subprocess.run([sys.executable, str(KIT.parent / "scripts" / "a9_verify_export.py"), "--cache", str(locked),
                        "--out", str(tmp_path / "a9"), "--layers", "33"], capture_output=True, text=True, env=ENV)
    assert r.returncode == 0 and "A9 PASS" in r.stdout, r.stdout + r.stderr
    assert "NOTE train" in r.stdout  # the tiny test cache is not the published v1 cache
