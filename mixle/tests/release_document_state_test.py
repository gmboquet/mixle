"""Publication rejects stale unreleased markers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = ROOT / "scripts" / "check_release_document_state.py"
    spec = importlib.util.spec_from_file_location("_check_release_document_state", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_current_candidate_is_deliberately_blocked_until_final_document_transition() -> None:
    errors = _module().validate(ROOT, "0.8.0")
    assert "CHANGELOG.md has no dated 0.8.0 release heading" in errors
    assert "0.8.0 migration guide still declares a development draft" in errors


def test_released_document_state_passes(tmp_path: Path) -> None:
    (tmp_path / "docs" / "migrations").mkdir(parents=True)
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n## [0.8.0] — 2026-07-27\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "changelog.rst").write_text(
        "Changelog\n=========\n\nUnreleased\n----------\n\n0.8.0\n-----\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "migrations" / "0.8.0.md").write_text("# Migrating to 0.8.0\n", encoding="utf-8")
    (tmp_path / "docs" / "charter.md").write_text("# Charter\n", encoding="utf-8")
    assert _module().validate(tmp_path, "0.8.0") == []
