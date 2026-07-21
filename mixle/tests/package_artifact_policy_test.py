"""Release-artifact contents stay intentional and auditable."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_runtime_wheel_excludes_the_test_tree() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    setuptools = project["tool"]["setuptools"]
    assert setuptools["include-package-data"] is False
    assert "mixle.tests*" in setuptools["packages"]["find"]["exclude"]


def test_sdist_retains_release_manifests_and_changelog() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "include CHANGELOG.md" in manifest
    assert "recursive-include manifests *.json *.md" in manifest
    assert "prune mixle/tests" in manifest
