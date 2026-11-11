"""Static checks of the collection kit's environment pins (must-fix 2) and of the install-time half of
must-fix 3 (flash-attn is required, no fallback). CPU only; nothing is run or installed.

collection/constraints.txt is the single source of truth for every version: these tests read it and
never repeat a version. The dry runs of the shell scripts are in test_kit_env_scripts.py.
"""
import re
from pathlib import Path

import pytest

packaging = pytest.importorskip("packaging")
from packaging.requirements import Requirement  # noqa: E402
from packaging.utils import canonicalize_name  # noqa: E402
from packaging.version import Version  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "collection"
CARC = KIT / "carc"
CONSTRAINTS = KIT / "constraints.txt"
KIT_SHELL_FILES = sorted({*KIT.rglob("*.sh"), *KIT.rglob("*.slurm")})

# Package NAMES only (the versions live in constraints.txt). Exact pins come from OpenVLA/LIBERO; the
# capped packages are unpinned transitive dependencies that broke on CARC.
EXACT = {"torch", "torchvision", "torchaudio", "transformers", "tokenizers", "tensorflow", "flash-attn", "robosuite"}
CAPPED = {"numpy", "protobuf", "tensorflow-metadata", "mujoco"}
CU121 = {"torch", "torchvision", "torchaudio"}  # installed from the PyTorch cu121 index
INSTALL_CMD = re.compile(r'(?:\bpip3?|"\$\{PIP\[@\]\}"|\$\{?PIP\b\S*|\buv\s+pip)\s+install\b')


def constraint_lines():
    """(requirement, comment) for every non-comment line of constraints.txt."""
    out = []
    for raw in CONSTRAINTS.read_text().splitlines():
        code, _, comment = raw.partition("#")  # pip and uv treat " #..." as a comment
        if code.strip():
            out.append((code.strip(), comment.strip()))
    return out


def logical_lines(path):
    """Non-comment lines of a shell file, with backslash continuations joined."""
    out, buf = [], ""
    for raw in Path(path).read_text().splitlines():
        if raw.rstrip().endswith("\\"):
            buf += raw.rstrip()[:-1] + " "
            continue
        line, buf = buf + raw, ""
        if line.strip() and not line.strip().startswith("#"):
            out.append(line)
    return out


def test_constraints_cover_every_package_once():
    found = {}
    for code, _ in constraint_lines():
        assert not code.startswith("-"), f"a constraints file must not carry options (pip rejects them): {code}"
        req = Requirement(code)
        assert not req.extras and req.url is None and req.marker is None, code
        name = canonicalize_name(req.name)
        assert name not in found, f"{name} is constrained twice"
        found[name] = req.specifier
    assert set(found) == {canonicalize_name(n) for n in EXACT | CAPPED}
    for name in EXACT:
        (spec,) = found[canonicalize_name(name)]
        assert spec.operator == "==", f"{name} must be an exact pin, got {spec}"
        if name in CU121:  # the local +cu121 build from the index URL still satisfies the public pin (PEP 440)
            assert found[name].contains(Version(spec.version + "+cu121"))
    for name in CAPPED:
        assert all(s.operator != "==" for s in found[canonicalize_name(name)]), f"{name} is a cap, not a pin"


def test_every_pin_has_a_reason_and_ranges_name_the_verified_version():
    for code, comment in constraint_lines():
        assert len(comment) > 10, f"pin without a reason comment: {code!r}"
        req = Requirement(code)
        m = re.search(r"\bverified (\S+?)[;,)]?(?:\s|$)", comment)
        if canonicalize_name(req.name) in {canonicalize_name(n) for n in CAPPED | CU121}:
            assert m, f"{req.name}: say which version was installed ('verified X') in the comment"
        if m:
            assert req.specifier.contains(Version(m.group(1)), prereleases=True), (code, m.group(1))


def test_no_version_is_hard_coded_outside_the_constraints_file():
    names = "|".join(re.escape(n) for n in sorted(EXACT | CAPPED, key=len, reverse=True))
    pinned = re.compile(rf"(?<![\w-])({names})\s*(==|<=|>=|<|>|~=)\s*\d", re.I)
    hits = [f"{p.relative_to(ROOT)}: {ln.strip()}" for p in KIT_SHELL_FILES for ln in logical_lines(p)
            if pinned.search(ln)]
    assert not hits, "versions belong in collection/constraints.txt only:\n" + "\n".join(hits)


def test_every_install_line_uses_the_constraints_file():
    setup = CARC / "00_setup_env.sh"
    assert re.search(r'^CONSTRAINTS="\$KIT_DIR/constraints\.txt"', setup.read_text(), re.M)
    installs = [ln for ln in logical_lines(setup) if INSTALL_CMD.search(ln)]
    assert len(installs) >= 7, installs  # torch, openvla, pandas, LIBERO, libero requirements, build tools, flash-attn
    for path in KIT_SHELL_FILES:  # every install path in the kit, not only the setup script
        for ln in logical_lines(path):
            if INSTALL_CMD.search(ln):
                assert re.search(r'-c\s+"?\$\{?CONSTRAINTS\}?"?|-c\s+\S*constraints\.txt', ln), \
                    f"{path.name}: install without the constraints file: {ln.strip()}"
            assert not re.search(r"\bconda\s+install\b", ln), f"{path.name}: conda install bypasses the constraints"
            assert not re.search(r"\binstall\b.*\s(-U|--upgrade)\b", ln), f"{path.name}: upgrades ignore the pins"
    torch = [ln for ln in installs if re.search(r"\btorch\b", ln) and "flash-attn" not in ln]
    assert len(torch) == 1 and "--index-url https://download.pytorch.org/whl/cu121" in torch[0], torch
    assert "check_pins ||" in setup.read_text() and '"${PIP[@]}" check' in setup.read_text()


def test_flash_attn_install_hard_fails():
    setup = CARC / "00_setup_env.sh"
    text = setup.read_text()
    flash = [ln for ln in logical_lines(setup) if INSTALL_CMD.search(ln) and "flash-attn" in ln]
    assert len(flash) == 1, flash
    assert "--no-build-isolation" in flash[0] and "||" not in flash[0]
    assert flash[0].strip().startswith("if !"), "a failed flash-attn install must reach flash_attn_fail"
    body = text.split("flash_attn_fail() {", 1)[1].split("\n}\n", 1)[0]
    assert "exit 1" in body and "flash_attention_2" in body
    check = text.split("check_flash_attn() {", 1)[1].split("\n}\n", 1)[0]
    assert "import flash_attn, flash_attn_2_cuda" in check and "|| flash_attn_fail" in check
    # CUDA_HOME is checked before anything is installed
    assert text.index('[ -x "$CUDA_HOME/bin/nvcc" ]') < text.index('install -c "$CONSTRAINTS"')
    for path in KIT_SHELL_FILES:
        for ln in path.read_text().splitlines():
            assert not re.search(r"(?i)flash.?attn.*\boptional\b", ln), f"{path.name}: {ln}"
            assert not re.search(r"flash-attn[^\n]*\|\|\s*(echo|true|:)", ln), f"{path.name} swallows a failure: {ln}"


def test_no_file_in_the_kit_mentions_sdpa():
    hits = []
    for path in KIT.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                continue
            hits += [f"{path.relative_to(ROOT)}:{i}" for i, ln in enumerate(text.splitlines(), 1) if "sdpa" in ln.lower()]
    assert not hits, hits
