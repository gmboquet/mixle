"""The family release manifest keeps the sibling repositories in lockstep with core (docs/family-release.rst).

Offline: the manifest's shape and its agreement with core's own version are checked here; the sync
checker's version arithmetic and its verdicts are exercised against a fixture family built in a temp
directory, one member in step and one drifted, so the release gate cannot pass vacuously. The live
check against GitHub is the receipt's job, not this test's.
"""

import importlib.util
import json
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "manifests" / "family_release.json"
MEMBERS = {"mixle-pde", "mixle-discrete", "mixle-physics", "mixle-sim", "mixle-notebooks", "mixle-agent"}


def _checker():
    spec = importlib.util.spec_from_file_location("_check_family_sync", ROOT / "scripts" / "check_family_sync.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_shape_and_members():
    manifest = _manifest()
    assert manifest["artifact"] == "mixle.family_release/v1"
    assert manifest["policy"]["version_policy"] == "lockstep"
    assert manifest["branch"].startswith("release/")
    assert {m["name"] for m in manifest["members"]} == MEMBERS
    for member in manifest["members"]:
        assert member["version_format"] in {"pep440", "semver", "none"}
        assert (member["version_file"] is None) == (member["version_format"] == "none")
        assert member["changelog_heading"].startswith("## ")
        assert member["release_checklist"] == f"release-checklists/{manifest['release']}.md"
        for pin in member["core_pins"]:
            assert manifest["branch"] in pin["must_contain"] or manifest["version"] in pin["must_contain"]
    assert not MEMBERS & set(manifest["excluded"])


def test_manifest_agrees_with_core_version_and_release():
    manifest = _manifest()
    checker = _checker()
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == manifest["version"]
    # the pre-release rehearses the release the documents describe (the D-0216 rule)
    doc_state = importlib.util.spec_from_file_location(
        "_doc_state", ROOT / "scripts" / "check_release_document_state.py"
    )
    module = importlib.util.module_from_spec(doc_state)
    doc_state.loader.exec_module(module)
    assert module.base_release(manifest["version"]) == manifest["release"]
    # the changelog headings the members must carry spell the family version in each member's format
    for member in manifest["members"]:
        want = checker.expected_version(manifest["version"], member["version_format"])
        if want is not None:
            assert member["changelog_heading"] == f"## {want}", member["name"]
    passed, failed = checker.check_core(manifest, ROOT)
    assert failed == [] and passed


@pytest.mark.parametrize(
    ("pep440", "semver"),
    [
        ("0.8.3", "0.8.3"),
        ("0.8.3rc1", "0.8.3-rc.1"),
        ("0.8.3rc12", "0.8.3-rc.12"),
        ("0.8.3.dev0", "0.8.3-dev.0"),
        ("0.8.3a1", "0.8.3-a.1"),
        ("0.8.3rc1.dev4", "0.8.3-rc.1.dev.4"),
        ("0.8.3.post1", "0.8.3-post.1"),
    ],
)
def test_semver_form(pep440, semver):
    assert _checker().semver_form(pep440) == semver


def _fixture_family(tmp_path: Path, drifted: bool) -> tuple[Path, Path]:
    """A two-member family: a Python package and a Node workspace; ``drifted`` leaves the workspace behind."""
    manifest = {
        "artifact": "mixle.family_release/v1",
        "release": "0.8.3",
        "version": "0.8.3rc1",
        "branch": "release/0.8.3",
        "policy": {"version_policy": "lockstep"},
        "core": {
            "name": "mixle",
            "repository": "x/mixle",
            "version_file": "pyproject.toml",
            "version_format": "pep440",
        },
        "members": [
            {
                "name": "pkg",
                "repository": "x/pkg",
                "version_file": "pyproject.toml",
                "version_format": "pep440",
                "changelog_file": "CHANGELOG.md",
                "changelog_heading": "## 0.8.3rc1",
                "core_pins": [{"file": "ci.yml", "must_contain": "mixle.git@release/0.8.3"}],
            },
            {
                "name": "node",
                "repository": "x/node",
                "version_file": "package.json",
                "version_format": "semver",
                "additional_version_files": ["packages/a/package.json"],
                "changelog_file": "CHANGELOG.md",
                "changelog_heading": "## 0.8.3-rc.1",
                "core_pins": [],
            },
        ],
        "excluded": [],
    }
    root = tmp_path / "family"
    core = tmp_path / "core"
    core.mkdir()
    (core / "pyproject.toml").write_text('[project]\nname = "mixle"\nversion = "0.8.3rc1"\n', encoding="utf-8")
    manifest_path = core / "family_release.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    pkg = root / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "pyproject.toml").write_text('[project]\nname = "pkg"\nversion = "0.8.3rc1"\n', encoding="utf-8")
    (pkg / "CHANGELOG.md").write_text("# Changelog\n\n## 0.8.3rc1 (pre-release)\n", encoding="utf-8")
    (pkg / "ci.yml").write_text(
        "pip install 'mixle @ git+https://github.com/x/mixle.git@release/0.8.3'\n", encoding="utf-8"
    )
    node = root / "node" / "packages" / "a"
    node.mkdir(parents=True)
    workspace_version = "0.8.0-dev.0" if drifted else "0.8.3-rc.1"
    (root / "node" / "package.json").write_text(
        json.dumps({"name": "node", "version": workspace_version}), encoding="utf-8"
    )
    (node / "package.json").write_text(json.dumps({"name": "@node/a", "version": "0.8.3-rc.1"}), encoding="utf-8")
    (root / "node" / "CHANGELOG.md").write_text("# Changelog\n\n## 0.8.3-rc.1 — pre-release\n", encoding="utf-8")
    return manifest_path, root


def test_local_check_passes_for_a_family_in_lockstep(tmp_path, capsys):
    manifest_path, root = _fixture_family(tmp_path, drifted=False)
    code = _checker().main(
        ["--manifest", str(manifest_path), "--local", str(root), "--core-root", str(manifest_path.parent)]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "every member is in lockstep with core" in out


def test_local_check_names_the_drifted_member(tmp_path, capsys):
    manifest_path, root = _fixture_family(tmp_path, drifted=True)
    code = _checker().main(
        ["--manifest", str(manifest_path), "--local", str(root), "--core-root", str(manifest_path.parent)]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "DRIFT node" in captured.out
    assert "package.json: version 0.8.0-dev.0 (want 0.8.3-rc.1)" in captured.out
    assert "out of sync: node" in captured.err


def test_local_check_reports_an_unreadable_member(tmp_path, capsys):
    manifest_path, root = _fixture_family(tmp_path, drifted=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["members"].append(
        {
            "name": "ghost",
            "repository": "x/ghost",
            "version_file": "pyproject.toml",
            "version_format": "pep440",
            "changelog_file": "CHANGELOG.md",
            "changelog_heading": "## 0.8.3rc1",
            "core_pins": [],
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    code = _checker().main(
        ["--manifest", str(manifest_path), "--local", str(root), "--core-root", str(manifest_path.parent)]
    )
    assert code == 2
    assert "could not be read" in capsys.readouterr().err
