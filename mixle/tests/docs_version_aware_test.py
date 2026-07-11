"""Worklist X12.5 -- documentation is version-aware.

A user arriving from PyPI must be able to tell which release (and which source
revision) the docs they are reading correspond to. This test pins the machinery that
makes that possible, without needing a full Sphinx build:

  * ``docs/conf.py`` derives ``release`` from ``pyproject.toml`` (not a hardcoded literal
    that can drift from the package);
  * it captures the build commit, preferring a CI-provided SHA, with a safe fallback;
  * it publishes ``release`` / ``version`` / ``commit`` into ``html_context`` so every
    page can display them;
  * the sidebar template actually renders the release and the commit stamp.
"""

from __future__ import annotations

import os
import runpy
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
CONF = ROOT / "docs" / "conf.py"
SWITCHER = ROOT / "docs" / "_templates" / "sidebar" / "version-switcher.html"


def _load_conf(env: dict[str, str] | None = None) -> dict:
    if not CONF.is_file():
        pytest.skip(f"{CONF} not found")
    saved = dict(os.environ)
    try:
        # Clear any ambient CI SHA so precedence tests are deterministic.
        for var in ("MIXLE_DOCS_COMMIT", "GITHUB_SHA", "READTHEDOCS_GIT_COMMIT_HASH"):
            os.environ.pop(var, None)
        if env:
            os.environ.update(env)
        return runpy.run_path(str(CONF))
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_release_is_derived_from_pyproject() -> None:
    ns = _load_conf()
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert ns["release"] == pyproject["project"]["version"], (
        "docs release must track the packaged version, not a hardcoded string"
    )
    # version is the major.minor prefix of release.
    assert ns["version"] == ".".join(ns["release"].split(".")[:2])


def test_commit_is_captured() -> None:
    ns = _load_conf()
    commit = ns["commit"]
    assert isinstance(commit, str) and commit, "commit must be a non-empty string"
    # Either a short hex SHA (local git) or the documented fallback.
    assert commit == "unknown" or all(c in "0123456789abcdef" for c in commit), (
        f"commit {commit!r} is neither a hex SHA nor the 'unknown' fallback"
    )


def test_ci_commit_env_takes_precedence() -> None:
    """A CI-provided SHA must win, so an exported (.git-less) build is still stamped."""
    ns = _load_conf(env={"MIXLE_DOCS_COMMIT": "abcdef1234567890"})
    assert ns["commit"] == "abcdef123456", "MIXLE_DOCS_COMMIT should be used (truncated to 12)"


def test_html_context_publishes_version_and_commit() -> None:
    ns = _load_conf()
    ctx = ns["html_context"]
    assert ctx["mixle_release"] == ns["release"]
    assert ctx["mixle_version"] == ns["version"]
    assert ctx["mixle_commit"] == ns["commit"]


def test_switcher_template_displays_release_and_commit() -> None:
    if not SWITCHER.is_file():
        pytest.skip(f"{SWITCHER} not found")
    tpl = SWITCHER.read_text(encoding="utf-8")
    assert "mixle_release" in tpl, "switcher must show the release from html_context"
    assert "mixle_commit" in tpl, "switcher must show the build commit stamp"
