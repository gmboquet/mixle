"""Release publication binds one signed, approved commit to retained artifacts."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "verify_required_checks.py"
_SHA = "a" * 40
_REPOSITORY = "gmboquet/mixle"


def _load():
    spec = importlib.util.spec_from_file_location("_verify_required_checks", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _url(run_id, *, repository=_REPOSITORY, workflow_run=None):
    """GitHub's details_url for a check run: the job segment IS the check-run id (SYS5-01)."""
    workflow_run = 1000 + run_id if workflow_run is None else workflow_run
    return f"https://github.com/{repository}/actions/runs/{workflow_run}/job/{run_id}"


def _run(name, run_id, *, sha=_SHA, status="completed", conclusion="success", details_url=None, app="github-actions"):
    """A check run shaped like GitHub's, including the reporting app (SYS5-01)."""
    run = {
        "name": name,
        "id": run_id,
        "head_sha": sha,
        "status": status,
        "conclusion": conclusion,
        "details_url": _url(run_id) if details_url is None else details_url,
    }
    if app is not None:
        run["app"] = {"slug": app}
    return run


def _payload(*runs):
    """One GitHub check-runs page: GitHub always reports the commit's total_count beside the runs."""
    return {"total_count": len(runs), "check_runs": list(runs)}


def _verify(module, payload, required, sha=_SHA, *, repository=_REPOSITORY):
    return module.verify_required_checks(payload, required, sha, repository=repository)


def _cli(module, argv):
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        status = module.main(argv)
    return status, stdout.getvalue()


def test_required_check_evidence_is_exact_sha_latest_success():
    module = _load()
    required = ("fast", "docs")
    payload = _payload(
        _run("fast", 1, conclusion="failure"),
        _run("fast", 2),
        _run("docs", 3),
    )
    assert _verify(module, payload, required) == {
        "fast": {"check_run_id": 2, "details_url": "https://github.com/gmboquet/mixle/actions/runs/1002/job/2"},
        "docs": {"check_run_id": 3, "details_url": "https://github.com/gmboquet/mixle/actions/runs/1003/job/3"},
    }
    # a run for another commit is not part of this commit's listing at all: refused, not ignored,
    # because an ignored one pads the count after a dropped failure (SYS5-02's neighbour)
    with pytest.raises(ValueError, match="for commit 'bbbb.*not the candidate"):
        _verify(module, _payload(*payload["check_runs"], _run("docs", 99, sha="b" * 40)), required)

    for bad in (
        _payload(_run("fast", 1)),
        _payload(_run("fast", 1), _run("docs", 2, status="in_progress", conclusion=None)),
        _payload(_run("fast", 1), _run("docs", 2), _run("docs", 3, conclusion="failure")),
    ):
        with pytest.raises(ValueError):
            _verify(module, bad, required)


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
    assert set(_verify(module, complete, required)) == {"fast", "docs"}

    truncated = dict(complete, total_count=3)
    with pytest.raises(ValueError, match="incomplete"):
        _verify(module, truncated, required)
    for missing_total in ({"check_runs": complete["check_runs"]}, dict(complete, total_count="2")):
        with pytest.raises(ValueError, match="total_count"):
            _verify(module, missing_total, required)

    # the newest run for a name lives on the second page: paginated evidence sees it
    page_one = {"total_count": 3, "check_runs": [_run("fast", 1), _run("docs", 2)]}
    page_two = {"total_count": 3, "check_runs": [_run("docs", 3, conclusion="failure")]}
    with pytest.raises(ValueError, match="incomplete"):
        _verify(module, page_one, required)  # first page alone is refused
    with pytest.raises(ValueError, match="docs: latest run 3"):
        _verify(module, [page_one, page_two], required)
    with pytest.raises(ValueError, match="disagree"):
        _verify(module, [page_one, dict(page_two, total_count=4)], required)
    with pytest.raises(ValueError, match="empty"):
        _verify(module, [], required)


def _sixty_nine_run_truth():
    """SYS5-02 reviewer fixture: 68 successful runs plus a NEWER failed run of a required check.

    Mirrors the retained candidate (68 runs on one page) with one more run: ``docs`` re-ran and
    failed as check run 69. The truth must be refused; so must every rearrangement of it that keeps
    ``total_count`` while dropping the failure.
    """
    runs = [_run("fast", 1), _run("docs", 2)]
    runs.extend(_run(f"extra-{index}", index) for index in range(3, 69))
    runs.append(_run("docs", 69, conclusion="failure"))
    return runs


def test_check_run_ids_must_be_unique_not_merely_add_up_to_total_count():
    """SYS5-02: ``len(runs) == total_count`` is not completeness when a run is supplied twice.

    Reviewer fixture: a 69-run truth whose newest ``docs`` run failed is refused. Omitting that
    failure and DUPLICATING the older successful ``docs`` run keeps 69 runs against
    ``total_count == 69`` (only 68 unique ids) and previously produced the same authorized
    evidence. Ids must be unique across every page, checked BEFORE the count comparison so the
    refusal names the forgery rather than a truncation it also happens to be.
    """
    module = _load()
    required = ("fast", "docs")
    truth = _sixty_nine_run_truth()
    assert len(truth) == 69
    with pytest.raises(ValueError, match="docs: latest run 69"):
        _verify(module, _payload(*truth), required)

    # the reviewer's forgery: newest failure omitted, older success duplicated, total preserved
    forged = [run for run in truth if run["id"] != 69] + [dict(truth[1])]
    assert len(forged) == 69 and len({run["id"] for run in forged}) == 68
    with pytest.raises(ValueError, match=r"duplicate.*\b2\b"):
        _verify(module, {"total_count": 69, "check_runs": forged}, required)
    # ...also when the duplicate is spread across pages
    with pytest.raises(ValueError, match=r"duplicate.*\b2\b"):
        _verify(
            module,
            [{"total_count": 69, "check_runs": forged[:40]}, {"total_count": 69, "check_runs": forged[40:]}],
            required,
        )

    # duplicates AND truncation together: the duplicate is named, not just "incomplete"
    both = forged[:-2] + [dict(truth[0])]  # 68 supplied of 69, and ``fast`` run 1 appears twice
    assert len(both) == 68 and len({run["id"] for run in both}) == 67
    with pytest.raises(ValueError, match=r"duplicate.*\b1\b") as excinfo:
        _verify(module, {"total_count": 69, "check_runs": both}, required)
    assert "incomplete" not in str(excinfo.value)

    # a duplicated id anywhere in the evidence is refused, required name or not
    filler_twice = [_run("fast", 1), _run("docs", 2), _run("extra", 3), _run("extra", 3)]
    with pytest.raises(ValueError, match=r"duplicate.*\b3\b"):
        _verify(module, _payload(*filler_twice), required)


def test_every_supplied_check_run_must_carry_a_valid_id():
    """SYS5-02: uniqueness is only meaningful if every run has a real id; a run that is not an
    object, or whose id is missing, boolean, negative, or a string, is refused wherever it sits."""
    module = _load()
    required = ("fast",)
    good = _run("fast", 1)
    for bad_id in (None, True, False, -1, 0, "1", 1.0, "abc"):
        stray = _run("extra", 2)
        stray["id"] = bad_id
        with pytest.raises(ValueError, match="invalid") as excinfo:
            _verify(module, _payload(good, stray), required)
        assert "extra" in str(excinfo.value)
    missing = _run("extra", 2)
    del missing["id"]
    with pytest.raises(ValueError, match="invalid"):
        _verify(module, _payload(good, missing), required)
    with pytest.raises(ValueError, match="not an object"):
        _verify(module, _payload(good, "not-a-run"), required)


def test_selected_runs_must_be_github_actions_jobs_of_the_repository():
    """SYS5-01: the old ``"/actions/runs/" in details_url`` substring check accepted forged URLs.

    GitHub's invariant, verified 68/68 on the retained candidate: a check run's ``details_url``
    is exactly ``https://github.com/{repository}/actions/runs/{run_id}/job/{check_run_id}`` and
    the reporting app is ``github-actions``. Anything else for a SELECTED run is a failure line
    naming the check.
    """
    module = _load()
    required = ("fast", "docs")
    good = _run("fast", 1)
    for label, bad_docs in (
        ("wrong host", _run("docs", 2, details_url="https://example.invalid/actions/runs/1002/job/2")),
        ("wrong repository", _run("docs", 2, details_url=_url(2, repository="someone/else"))),
        ("job segment is not the check run id", _run("docs", 2, details_url=_url(3))),
        ("job segment differs by prefix", _run("docs", 2, details_url=_url(2) + "0")),
        ("http scheme", _run("docs", 2, details_url=_url(2).replace("https://", "http://"))),
        ("run id not positive", _run("docs", 2, details_url=_url(2, workflow_run=0))),
        ("run id not an integer", _run("docs", 2, details_url=_url(2, workflow_run="12x"))),
        ("trailing path", _run("docs", 2, details_url=_url(2) + "/")),
        ("query string", _run("docs", 2, details_url=_url(2) + "?pr=1")),
        ("missing url", dict(_run("docs", 2), details_url=None)),
        ("missing app", _run("docs", 2, app=None)),
        ("wrong app", _run("docs", 2, app="some-other-app")),
        ("app not an object", dict(_run("docs", 2), app="github-actions")),
    ):
        with pytest.raises(ValueError, match="docs: latest run 2") as excinfo:
            _verify(module, _payload(good, bad_docs), required)
        assert "fast" not in str(excinfo.value).split("\n- ", 1)[1], label
    # a superseded (not selected) run may carry anything: only the selected run is the evidence
    superseded = _run("docs", 2, details_url="https://example.invalid/actions/runs/1/job/2", app=None)
    assert set(_verify(module, _payload(good, superseded, _run("docs", 3)), required)) == {"fast", "docs"}
    # the repository the caller names is what the URLs are checked against
    with pytest.raises(ValueError, match="docs: latest run 2"):
        _verify(module, _payload(good, _run("docs", 2)), required, repository="someone/else")
    with pytest.raises(ValueError, match="repository"):
        _verify(module, _payload(good, _run("docs", 2)), required, repository="not-a-repository")


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
        '--repository "$REPOSITORY"',
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
    argv = ["--required", str(required), "--input", str(evidence), "--sha", _SHA, "--repository", _REPOSITORY]
    assert module.main(argv) == 1


def test_check_evidence_cli_requires_a_well_formed_repository(tmp_path):
    """SYS5-01: the record names where its evidence came from, so the caller must say which
    repository's check runs these are; a missing or malformed OWNER/REPO exits 1 before any
    evidence is written."""
    module = _load()
    required = tmp_path / "required.txt"
    evidence = tmp_path / "checks.json"
    required.write_text("fast\n", encoding="utf-8")
    evidence.write_text(json.dumps(_payload(_run("fast", 1))), encoding="utf-8")
    base = ["--required", str(required), "--input", str(evidence), "--sha", _SHA]
    status, out = _cli(module, base + ["--repository", _REPOSITORY])
    assert status == 0 and json.loads(out)["publication_authorized"] is True

    status, out = _cli(module, base)
    assert (status, out) == (1, "")
    for bad in (
        "",
        "mixle",
        "gmboquet/",
        "/mixle",
        "gmboquet/mixle/extra",
        "gm boquet/mixle",
        "gmboquet/mixle\n",
        "a/b/",
    ):
        status, out = _cli(module, base + ["--repository", bad])
        assert (status, out) == (1, ""), bad


def test_check_evidence_record_names_its_generator_and_source(tmp_path):
    """SYS5-01: the record carries its generator and the exact evidence it was derived from --
    repository, endpoint, the sha256 of the input file's EXACT bytes and GitHub's total_count --
    so a consumer can re-fetch and re-derive the selection. Output is deterministic per input."""
    module = _load()
    required = tmp_path / "required.txt"
    evidence = tmp_path / "checks.json"
    required.write_text("fast\ndocs\n", encoding="utf-8")
    # non-canonical bytes on purpose: the digest is over the file, not a re-serialization
    raw = (
        b'{ "total_count" : 3,\n "check_runs": '
        + json.dumps([_run("fast", 1), _run("docs", 2, conclusion="failure"), _run("docs", 3)]).encode("utf-8")
        + b" }\n\n"
    )
    evidence.write_bytes(raw)
    argv = ["--required", str(required), "--input", str(evidence), "--sha", _SHA, "--repository", _REPOSITORY]
    status, out = _cli(module, argv)
    assert status == 0
    record = json.loads(out)
    assert record == {
        "artifact": "mixle.release_check_evidence/v1",
        "generator": "scripts/verify_required_checks.py",
        "commit": _SHA,
        "source": {
            "repository": _REPOSITORY,
            "endpoint": f"repos/{_REPOSITORY}/commits/{_SHA}/check-runs",
            "check_runs_sha256": hashlib.sha256(raw).hexdigest(),
            "check_runs_count": 3,
        },
        "checks": {
            "fast": {"check_run_id": 1, "details_url": _url(1)},
            "docs": {"check_run_id": 3, "details_url": _url(3)},
        },
        "publication_authorized": True,
    }
    assert record["source"]["check_runs_sha256"] != hashlib.sha256(json.dumps(json.loads(raw)).encode()).hexdigest()
    # deterministic: identical input bytes -> identical output bytes
    assert _cli(module, argv) == (0, out)
    # and a byte-different input with the same meaning is a different digest, still authorized
    evidence.write_bytes(raw + b"\n")
    status, again = _cli(module, argv)
    assert status == 0
    assert json.loads(again)["source"]["check_runs_sha256"] == hashlib.sha256(raw + b"\n").hexdigest()
    assert json.loads(again)["checks"] == record["checks"]
    # the record function is the same code path main() prints
    assert module.evidence_record(raw, ("fast", "docs"), _SHA, repository=_REPOSITORY) == record


def test_candidate_record_producer_writes_what_the_receipt_resolver_requires():
    """One producer for release-candidate.json, and its output is what the resolver binds against.

    publish.yml used to write the record with an inline one-liner that carried no ``tree``; the
    receipt resolver, repaired against hand-built review candidates that did carry one, requires
    it. The real workflow could therefore not have bound a single receipt (D-0180). The producer
    is now a script the workflow calls and this test exercises end to end.
    """
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


def test_generator_matches_whole_fields_not_prefixes_and_agrees_with_the_resolver_on_ids():
    """Neighbours found by attacking the SYS5-02 repair: ``re.match`` with ``$`` let a trailing newline
    through (a run id segment or a --repository ending in a newline passed the generator and failed
    the resolver's ``fullmatch``), and id 0 was generator-approved but resolver-refused. Both sides now
    use whole-field matches and require ids > 0."""
    module = _load()
    required = ("fast",)
    newline_url = "https://github.com/gmboquet/mixle/actions/runs/1001\n/job/1"
    with pytest.raises(ValueError, match="details_url"):
        _verify(module, _payload(_run("fast", 1, details_url=newline_url)), required)
    with pytest.raises(ValueError, match="OWNER/REPO"):
        _verify(module, _payload(_run("fast", 1)), required, repository="gmboquet/mixle\n")
    for repository in ("../..", "./.", "a/b/c", "/mixle", "mixle/"):
        with pytest.raises(ValueError, match="OWNER/REPO"):
            module._validated_repository(repository)
    with pytest.raises(ValueError, match="invalid ID 0"):
        _verify(
            module,
            _payload(_run("fast", 0, details_url="https://github.com/gmboquet/mixle/actions/runs/1000/job/0")),
            required,
        )
