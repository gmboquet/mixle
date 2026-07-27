"""Publication must compare two independent clean candidate builds."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _verifier():
    path = ROOT / "scripts" / "verify_reproducible_builds.py"
    spec = importlib.util.spec_from_file_location("_verify_reproducible_builds", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_equal_builds_pass_and_one_changed_byte_fails(tmp_path: Path) -> None:
    verifier = _verifier()
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    for root in (left, right):
        (root / "mixle.whl").write_bytes(b"wheel")
        (root / "mixle.tar.gz").write_bytes(b"sdist")
    assert verifier.verify(left, right)["passed"] is True
    (right / "mixle.whl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="not byte-for-byte reproducible"):
        verifier.verify(left, right)


def test_publish_builds_from_two_clean_archives_before_upload() -> None:
    workflow = (ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    assert workflow.count("git archive HEAD") == 2
    compare = workflow.index("verify_reproducible_builds.py")
    upload = workflow.index("gh release upload")
    assert compare < upload
