"""Release publication binds one signed, approved commit to retained artifacts."""

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "verify_required_checks.py"
_SHA = "a" * 40


def _load():
    spec = importlib.util.spec_from_file_location("_verify_required_checks", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run(name, run_id, *, sha=_SHA, status="completed", conclusion="success"):
    return {
        "name": name,
        "id": run_id,
        "head_sha": sha,
        "status": status,
        "conclusion": conclusion,
        "details_url": f"https://github.com/gmboquet/mixle/actions/runs/{run_id}/job/{run_id}",
    }


def test_required_check_evidence_is_exact_sha_latest_success():
    module = _load()
    required = ("fast", "docs")
    payload = {
        "check_runs": [
            _run("fast", 1, conclusion="failure"),
            _run("fast", 2),
            _run("docs", 3),
            _run("docs", 99, sha="b" * 40),
        ]
    }
    assert module.verify_required_checks(payload, required, _SHA) == {
        "fast": {
            "check_run_id": 2,
            "details_url": "https://github.com/gmboquet/mixle/actions/runs/2/job/2",
        },
        "docs": {
            "check_run_id": 3,
            "details_url": "https://github.com/gmboquet/mixle/actions/runs/3/job/3",
        },
    }

    for bad in (
        {"check_runs": [_run("fast", 1)]},
        {"check_runs": [_run("fast", 1), _run("docs", 2, status="in_progress", conclusion=None)]},
        {"check_runs": [_run("fast", 1), _run("docs", 2), _run("docs", 3, conclusion="failure")]},
    ):
        with pytest.raises(ValueError):
            module.verify_required_checks(bad, required, _SHA)


def test_publish_workflow_has_fail_closed_candidate_binding():
    workflow = (_ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    for required_fragment in (
        'test "$TAG" = "v$VERSION"',
        'test "$(git cat-file -t "refs/tags/$TAG")" = "tag"',
        ".verification.verified",
        'git merge-base --is-ancestor origin/release/0.8.0 "$SHA"',
        'test "$SHA" = "$(git rev-parse origin/release/0.8.0)"',
        'test "$EVENT_SHA" = "$SHA"',
        "verify_required_checks.py",
        "release_metadata.py",
        "sha256sum -c",
        "gh release upload",
        "environment: pypi",
    ):
        assert required_fragment in workflow
    assert workflow.index("verify-candidate:") < workflow.index("build:")
    assert workflow.index("sha256sum -c") < workflow.index("gh-action-pypi-publish")


def test_publication_is_recoverable_two_phase_and_public_transition_is_last():
    workflow = (_ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    assert "release:\n" not in workflow
    assert "options: [prepare, promote]" in workflow
    assert "prepare_run_id:" in workflow
    assert "if: inputs.phase == 'prepare'" in workflow
    assert "if: inputs.phase == 'promote'" in workflow
    assert "gh release create" in workflow and "--draft" in workflow
    assert 'test "$(gh release view "$CANDIDATE_TAG" --json isDraft --jq .isDraft)" = "true"' in workflow
    assert "run-id: ${{ inputs.prepare_run_id }}" in workflow
    assert "skip-existing: true" in workflow
    assert workflow.index("gh-action-pypi-publish") < workflow.index("verify_published_artifacts.py")
    assert workflow.index("verify_published_artifacts.py") < workflow.index(
        'gh release edit "$CANDIDATE_TAG" --draft=false'
    )


def test_release_check_policy_is_nonempty_and_unique():
    module = _load()
    names = module.required_check_names(_ROOT / ".github" / "release-required-checks.txt")
    assert len(names) >= 10
    assert len(names) == len(set(names))


def test_check_evidence_cli_fails_nonzero(tmp_path):
    module = _load()
    required = tmp_path / "required.txt"
    evidence = tmp_path / "checks.json"
    required.write_text("fast\n", encoding="utf-8")
    evidence.write_text(json.dumps({"check_runs": [_run("fast", 1, conclusion="failure")]}), encoding="utf-8")
    assert module.main(["--required", str(required), "--input", str(evidence), "--sha", _SHA]) == 1
