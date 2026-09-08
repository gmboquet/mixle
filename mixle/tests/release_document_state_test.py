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


def test_release_documents_have_made_the_final_transition() -> None:
    """The tree's own release documents must describe 0.8.0 as released.

    This assertion used to run the other way: it required the candidate to be BLOCKED, asserting that
    the changelog was undated and the migration guide still said "development draft". That was the
    tripwire that kept anyone from quietly flipping the documents to "released" early, and it fired --
    correctly -- the moment the transition was made (2026-08-22). The transition is a one-way door, so
    the guard now points the other way and catches a regression back to draft wording, which would
    make publish.yml's own `check_release_document_state.py` gate fail at verify-candidate time.
    """
    assert _module().validate(ROOT, "0.8.2") == []


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


def test_a_pre_release_is_judged_by_its_base_release_documents(tmp_path: Path) -> None:
    # The rehearsal cut of D-0216 changes only the version string, so ``0.8.2rc1`` must find the
    # 0.8.2 heading, section and migration guide rather than demand documents of its own.
    module = _module()
    assert module.base_release("0.8.2rc1") == "0.8.2"
    assert module.base_release("0.8.2") == "0.8.2"
    assert module.base_release("1.0.0b2.dev3") == "1.0.0"
    (tmp_path / "docs" / "migrations").mkdir(parents=True)
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n## [0.8.2] — 2026-09-08\n", encoding="utf-8"
    )
    (tmp_path / "docs" / "changelog.rst").write_text("Changelog\n=========\n\n0.8.2\n-----\n", encoding="utf-8")
    (tmp_path / "docs" / "migrations" / "0.8.2.md").write_text("# Migrating to 0.8.2\n", encoding="utf-8")
    (tmp_path / "docs" / "charter.md").write_text("Version 0.8.2 is prepared.\n", encoding="utf-8")
    assert module.validate(tmp_path, "0.8.2rc1") == []
    assert module.validate(tmp_path, "0.8.2.post1") == []
