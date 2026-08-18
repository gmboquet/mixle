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


def _payload(*runs):
    """One GitHub check-runs page: GitHub always reports the commit's total_count beside the runs."""
    return {"total_count": len(runs), "check_runs": list(runs)}


def test_required_check_evidence_is_exact_sha_latest_success():
    module = _load()
    required = ("fast", "docs")
    payload = _payload(
        _run("fast", 1, conclusion="failure"),
        _run("fast", 2),
        _run("docs", 3),
        _run("docs", 99, sha="b" * 40),
    )
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
        _payload(_run("fast", 1)),
        _payload(_run("fast", 1), _run("docs", 2, status="in_progress", conclusion=None)),
        _payload(_run("fast", 1), _run("docs", 2), _run("docs", 3, conclusion="failure")),
    ):
        with pytest.raises(ValueError):
            module.verify_required_checks(bad, required, _SHA)


def test_check_run_evidence_must_be_complete_not_a_truncated_first_page():
    """A page that does not add up to GitHub's total_count cannot support "latest run per name".

    verify-candidate fetched ``check-runs?per_page=100`` and the generator never looked at
    ``total_count``; a commit with more runs than one page (a dispatch plus a superseded push run
    is already ~70) could have had its newest run cut off, leaving an older success standing in
    for a newer failure. Paginated evidence (``gh api --paginate --slurp``: a list of pages) is
    accepted when the pages agree and add up; anything else is refused.
    """
    module = _load()
    required = ("fast", "docs")
    complete = _payload(_run("fast", 1), _run("docs", 2))
    assert set(module.verify_required_checks(complete, required, _SHA)) == {"fast", "docs"}

    truncated = dict(complete, total_count=3)
    with pytest.raises(ValueError, match="incomplete"):
        module.verify_required_checks(truncated, required, _SHA)
    for missing_total in ({"check_runs": complete["check_runs"]}, dict(complete, total_count="2")):
        with pytest.raises(ValueError, match="total_count"):
            module.verify_required_checks(missing_total, required, _SHA)

    # the newest run for a name lives on the second page: paginated evidence sees it
    page_one = {"total_count": 3, "check_runs": [_run("fast", 1), _run("docs", 2)]}
    page_two = {"total_count": 3, "check_runs": [_run("docs", 3, conclusion="failure")]}
    with pytest.raises(ValueError, match="incomplete"):
        module.verify_required_checks(page_one, required, _SHA)  # first page alone is refused
    with pytest.raises(ValueError, match="docs: latest run 3"):
        module.verify_required_checks([page_one, page_two], required, _SHA)
    with pytest.raises(ValueError, match="disagree"):
        module.verify_required_checks([page_one, dict(page_two, total_count=4)], required, _SHA)
    with pytest.raises(ValueError, match="empty"):
        module.verify_required_checks([], required, _SHA)


def test_publish_workflow_has_fail_closed_candidate_binding():
    workflow = (_ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    for required_fragment in (
        'test "$TAG" = "v$VERSION"',
        'test "$(git cat-file -t "refs/tags/$TAG")" = "tag"',
        ".verification.verified",
        'gh api "repos/$REPOSITORY/branches/release/0.8.0" --jq .commit.sha',
        'test "$SHA" = "$RELEASE_SHA"',
        'test "$EVENT_SHA" = "$SHA"',
        "verify_required_checks.py",
        'gh api --paginate --slurp "repos/$REPOSITORY/commits/$SHA/check-runs?per_page=100"',
        "release_metadata.py",
        "release_candidate_record.py",
        "--tree \"$(git rev-parse 'HEAD^{tree}')\"",
        "package_docs_archive.py",
        "sha256sum -c",
        "gh release upload",
        "environment: pypi",
    ):
        assert required_fragment in workflow
    assert workflow.index("verify-candidate:") < workflow.index("build:")
    assert workflow.index("sha256sum -c") < workflow.index("gh-action-pypi-publish")
    assert "candidate/docs-dist/" in workflow


def test_publication_is_recoverable_two_phase_and_public_transition_is_last():
    workflow = (_ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    assert "release:\n" not in workflow
    assert "options: [prepare, testpypi, promote]" in workflow
    assert "prepare_run_id:" in workflow
    assert "testpypi_run_id:" in workflow
    assert "if: inputs.phase == 'prepare'" in workflow
    assert "if: inputs.phase == 'testpypi'" in workflow
    assert "if: inputs.phase == 'promote'" in workflow
    assert "gh release create" in workflow and "--draft" in workflow
    assert 'test "$(gh release view "$CANDIDATE_TAG" --json isDraft --jq .isDraft)" = "true"' in workflow
    assert "run-id: ${{ inputs.prepare_run_id }}" in workflow
    assert "run-id: ${{ inputs.testpypi_run_id }}" in workflow
    assert "repository-url: https://test.pypi.org/legacy/" in workflow
    assert "testpypi-rehearsal.json" in workflow
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
    evidence.write_text(json.dumps(_payload(_run("fast", 1, conclusion="failure"))), encoding="utf-8")
    assert module.main(["--required", str(required), "--input", str(evidence), "--sha", _SHA]) == 1


def test_candidate_record_producer_writes_what_the_receipt_resolver_requires():
    """One producer for release-candidate.json, and its output is what the resolver binds against.

    publish.yml used to write the record with an inline one-liner that carried no ``tree``; the
    receipt resolver, repaired against hand-built review candidates that did carry one, requires
    it. The real workflow could therefore not have bound a single receipt (D-0180). The producer
    is now a script the workflow calls and this test exercises end to end.
    """
    import contextlib
    import io
    import tempfile

    spec = importlib.util.spec_from_file_location(
        "_release_candidate_record", _ROOT / "scripts" / "release_candidate_record.py"
    )
    producer = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(producer)
    spec = importlib.util.spec_from_file_location(
        "_run_repro_entry_for_record", _ROOT / "scripts" / "run_repro_entry.py"
    )
    runner = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(runner)
    bundle = json.loads((_ROOT / "release-checklists" / "0.8.0-repro-bundle.json").read_text(encoding="utf-8"))

    tree = "7a" * 20
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "release-candidate.json"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            status = producer.main(
                [
                    "--commit",
                    _SHA,
                    "--tree",
                    tree,
                    "--tag",
                    "v0.8.0",
                    "--version",
                    "0.8.0",
                    "--workflow-run",
                    "123",
                    "--out",
                    str(out),
                ]
            )
        assert status == 0
        record = json.loads(out.read_text(encoding="utf-8"))
        assert record == json.loads(stdout.getvalue())
        assert record == {
            "artifact": "mixle.release_candidate/v1",
            "commit": _SHA,
            "tree": tree,
            "tag": "v0.8.0",
            "version": "0.8.0",
            "workflow_run": "123",
        }
        # the resolver's candidate-identity checks pass on the producer's record and fail on
        # nothing but a genuinely different executing artifact
        root = Path(directory) / "records"
        (root / "metadata").mkdir(parents=True)
        (root / "metadata" / "release-candidate.json").write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        result = runner._resolve_candidate_records(bundle, root, {"source_commit": _SHA, "source_tree": tree})
        identity_problems = [p for p in result["problems"] if "release-candidate.json" in p or "executing" in p]
        assert identity_problems == [], identity_problems
        result = runner._resolve_candidate_records(bundle, root, {"source_commit": _SHA, "source_tree": "8" * 40})
        assert any("is not the candidate tree" in p for p in result["problems"])

    # and the producer refuses to write an identity that is not one
    for bad in (
        {"commit": _SHA[:39], "tree": tree},
        {"commit": _SHA, "tree": tree.upper()},
        {"commit": _SHA, "tree": tree, "tag": "0.8.0"},
        {"commit": _SHA, "tree": tree, "workflow_run": " "},
    ):
        arguments = {"commit": _SHA, "tree": tree, "tag": "v0.8.0", "version": "0.8.0", "workflow_run": "123"}
        arguments.update(bad)
        with pytest.raises(ValueError):
            producer.release_candidate_record(**arguments)
