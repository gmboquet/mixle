"""The publication builder is exact and emits a candidate-bound receipt."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = ROOT / "scripts" / "build_environment_receipt.py"
    spec = importlib.util.spec_from_file_location("_build_environment_receipt", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_lock_is_exact_and_matches_build_system() -> None:
    module = _module()
    lock = ROOT / "release-checklists" / "0.8.0-build-requirements.txt"
    locked = module._locked(lock)
    # The lock is the FULL resolved closure, not just the top-level tools. The equality form of
    # this assertion pinned exactly five names, which is precisely the incompleteness the systems
    # review caught (SYS-1): an offline install from the five-pin lock failed on wheel's transitive
    # requirement packaging>=24, so a lock that omits transitives is not exact. The reviewed tool
    # versions must still be present and pinned; the closure must contain the transitive that
    # failed; and every entry must be an exact pin.
    for name, version in {
        "pip": "26.1.2",
        "build": "1.5.0",
        "setuptools": "83.0.0",
        "wheel": "0.47.0",
        "twine": "7.0.0",
    }.items():
        assert locked.get(name) == version, name
    assert "packaging" in locked, "the transitive that broke the offline install must be pinned"
    assert len(locked) > 15, "the closure must include transitive dependencies"
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"setuptools==83.0.0"' in pyproject
    assert '"wheel==0.47.0"' in pyproject


def test_build_receipt_rejects_tool_drift(tmp_path: Path) -> None:
    module = _module()
    lock = tmp_path / "builder.txt"
    lock.write_text("build==1.5.0\n", encoding="utf-8")
    with mock.patch.object(module.importlib.metadata, "version", return_value="1.4.0"):
        with pytest.raises(ValueError, match="differs from lock"):
            module.build(lock, "a" * 40, 1)
