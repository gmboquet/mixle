"""Contributor, release-target, and hook policy must agree with machine governance."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "manifests" / "development_policy.json"


def _load_renderer():
    path = ROOT / "scripts" / "render_contributing_policy.py"
    spec = importlib.util.spec_from_file_location("_render_contributing_policy", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_contributor_summary_is_generated_from_machine_policy() -> None:
    renderer = _load_renderer()
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    current = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert renderer.replace_block(current, renderer.render(policy)) == current
    assert policy["compatibility"]["stable_deprecation_min_minor_releases"] == 2
    assert "typed-core mypy" in policy["validation"]["blocking"]
    assert policy["validation"]["advisory"] == ["whole-tree mypy"]


def test_dependabot_targets_the_machine_release_branch() -> None:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    dependabot = (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    target = f'target-branch: "{policy["release"]["target_branch"]}"'
    assert dependabot.count(target) == dependabot.count("package-ecosystem:")


def test_precommit_aborts_partial_staging_before_mutation() -> None:
    hook = (ROOT / ".githooks" / "pre-commit").read_text(encoding="utf-8")
    guard = hook.index("partially_staged=$(git diff --name-only")
    formatter = hook.index("$RUFF format")
    restage = hook.index("git add -- $files")
    assert guard < formatter < restage
    guarded_block = hook[guard:formatter]
    assert "exit 1" in guarded_block
    assert "index left unchanged" in guarded_block


def test_support_policy_matches_machine_maturity_contract() -> None:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    support = (ROOT / "docs" / "support-policy.rst").read_text(encoding="utf-8")
    assert "**stable**" in support and "**provisional**" in support and "**experimental**" in support
    count = policy["compatibility"]["stable_deprecation_min_minor_releases"]
    assert f"at least **{count} minor releases**" in support
