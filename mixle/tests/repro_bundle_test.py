"""The reproducibility bundle is a complete, executable, candidate-bound closure."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "release-checklists" / "0.8.0-repro-bundle.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _bundle():
    return json.loads(BUNDLE.read_text(encoding="utf-8"))


def test_tracked_bundle_is_canonical_and_complete():
    builder = _load(ROOT / "scripts" / "build_repro_bundle.py", "_build_repro_bundle")
    runner = _load(ROOT / "scripts" / "run_repro_entry.py", "_run_repro_entry")
    tracked = _bundle()
    assert tracked == builder.build()
    try:
        assert runner.validate_bundle(tracked) is tracked
    except ValueError as exc:
        # validate_bundle enforces the implementer-environment pins (numpy/scipy versions in
        # 0.8.0-repro-environment.json) and fails CLOSED anywhere else. On such hosts the precise
        # version-mismatch refusal IS correct validator behavior, so the test accepts exactly that
        # message and still fails on any other refusal. Same class as the replay skips below;
        # first fired when the sharded full tier ran on hosted runners (numpy 2.5.1).
        assert "is required; found" in str(exc), exc
    assert tracked["candidate_binding"]["required_records"]
    assert tracked["acceptance"]
    assert tracked["code_license"]["spdx"] == "MIT"


@pytest.mark.parametrize(
    "entry_id",
    ["gallery-univariate", "gallery-structured", "production-provenance", "scaling-backend"],
)
def test_every_local_entry_reproduces_exact_expected_output(entry_id):
    runner = _load(ROOT / "scripts" / "run_repro_entry.py", f"_run_repro_entry_{entry_id}")
    try:
        receipt = runner.run_entry(_bundle(), entry_id)
    except ValueError as exc:
        if "is required; found" in str(exc):
            # The bundle pins the implementer environment exactly (numpy/scipy versions recorded in
            # 0.8.0-repro-environment.json), and the runner fails CLOSED on any other -- byte-exact
            # stdout digests are only meaningful under the arithmetic that produced them. On a host
            # with different pins that refusal is correct behavior, not a failed reproduction, so
            # the test records it as a skip. The checklist says the same: the local bundle is a
            # working-bundle check by the implementer, and independent replays are the separate
            # EXTERNAL gate. This first fired when the sharded full tier ran to completion on
            # hosted runners (numpy 2.5.1) on 2026-08-04 -- the monolith was always killed first.
            pytest.skip(f"bundle binds the implementer environment: {exc}")
        raise
    assert receipt["passed"] is True
    assert receipt["entry"] == entry_id


def test_declared_volatile_spans_do_not_weaken_the_stdout_digest():
    """Normalizing a volatile span must exempt only that span, and must fail if it stops matching.

    The provenance entry names the commit it reproduces from, so its digest has to be taken over a
    normalized output. That is the one legitimate exemption; a rule that quietly matched nothing
    would turn the digest into a check on whatever the output happens to be.
    """
    runner = _load(ROOT / "scripts" / "run_repro_entry.py", "_run_repro_entry_volatile")
    stdout = "commit abc1234 built\nresult: 41\n"
    entry = {
        "id": "fixture",
        "expected": {
            "format": "text",
            "volatile": [{"pattern": r"commit [0-9a-f]{7} ", "placeholder": "commit <sha> "}],
            "stdout_sha256": hashlib.sha256(b"commit <sha> built\nresult: 41\n").hexdigest(),
        },
    }
    runner._validate_output(entry, stdout)

    # the exempted span may vary freely -- that is what it is for
    runner._validate_output(entry, "commit fedcba9 built\nresult: 41\n")

    # nothing else may
    with pytest.raises(ValueError, match="digest mismatch"):
        runner._validate_output(entry, "commit abc1234 built\nresult: 42\n")

    # a rule that stops matching must fail loudly rather than quietly widen the digest
    stale = json.loads(json.dumps(entry))
    stale["expected"]["volatile"] = [{"pattern": "never appears anywhere", "placeholder": ""}]
    with pytest.raises(ValueError, match="never matched"):
        runner._validate_output(stale, stdout)

    # the production entry is the one real user of the exemption, and exempts only the commit
    provenance = next(e for e in _bundle()["entries"] if e["id"] == "production-provenance")
    assert [rule["placeholder"] for rule in provenance["expected"]["volatile"]] == ["git / mixle  : <commit> / "]


def test_bundle_rejects_unresolved_license_and_integrity_placeholders():
    serialized = json.dumps(_bundle(), sort_keys=True).upper()
    for marker in ("CONFIRM-AT-PUBLISH", "TODO", "TBD"):
        assert marker not in serialized


def _runner():
    return _load(ROOT / "scripts" / "run_repro_entry.py", "_run_repro_entry")


_CANDIDATE_COMMIT = "e" * 40
_REQUIRED_CHECKS_POLICY = ROOT / ".github" / "release-required-checks.txt"
_CHECK_EVIDENCE_GENERATOR = ROOT / "scripts" / "verify_required_checks.py"


def _github_check_runs(commit: str, names, *, first_id: int = 1000, conclusion: str = "success") -> dict:
    """A GitHub ``check-runs`` payload with one completed run per name on ``commit``."""
    return {
        "total_count": len(names),
        "check_runs": [
            {
                "name": name,
                "id": first_id + index,
                "head_sha": commit,
                "status": "completed",
                "conclusion": conclusion,
                "details_url": f"https://github.com/gmboquet/mixle/actions/runs/{500 + index}/job/{first_id + index}",
            }
            for index, name in enumerate(names)
        ],
    }


def _generated_check_evidence(commit: str = _CANDIDATE_COMMIT) -> str:
    """Produce release-check-evidence.json exactly as publish.yml does: the generator's own ``main``.

    The record's only real producer is scripts/verify_required_checks.py over the check runs of
    the candidate SHA. The first SYS3-05 fixture hand-wrote a shape the generator never emits
    ({"tests": "success", ...}); the resolver was written against that fiction and would have
    rejected every real record. Building the fixture through the generator's ``main`` -- not a
    re-implementation of its output -- is what makes this test a contract between the two.
    """
    import contextlib
    import io
    import tempfile

    generator = _load(_CHECK_EVIDENCE_GENERATOR, "_verify_required_checks_for_bundle")
    names = generator.required_check_names(_REQUIRED_CHECKS_POLICY)
    with tempfile.TemporaryDirectory() as directory:
        payload = Path(directory) / "check-runs.json"
        payload.write_text(json.dumps(_github_check_runs(commit, names)), encoding="utf-8")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            status = generator.main(
                ["--required", str(_REQUIRED_CHECKS_POLICY), "--input", str(payload), "--sha", commit]
            )
    assert status == 0, "generator refused an all-success payload"
    return stdout.getvalue()


def _write_valid_records(
    root: Path, *, commit: str = _CANDIDATE_COMMIT, release: str = "0.8.0", tree: str = "7" * 40
) -> dict:
    """Write a records root whose contents actually assert what their names promise.

    Includes a real (tiny) wheel under ``dist/`` whose bytes hash to the SHA256SUMS entry, and
    passing check evidence -- because the binding now compares the executing installation's RECORD
    hashes against the PINNED WHEEL's RECORD (SYS3-01) and requires the check evidence to be
    terminal-passing (SYS3-05). Returns the wheel's RECORD hashes so a test can present a matching
    or deliberately mismatching installation.
    """
    import base64
    import hashlib
    import zipfile

    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (root / "dist").mkdir(exist_ok=True)
    wheel, sdist = "mixle-0.8.0-py3-none-any.whl", "mixle-0.8.0.tar.gz"

    # a minimal but structurally VALID wheel: every member the archive holds is claimed by RECORD
    # with its real digest, as real wheels do. An earlier fixture left METADATA unclaimed and was
    # only exposed when the RECORD-vs-members check landed (SYS4-01) -- the fixture was the bug.
    def _enc(data: bytes) -> str:
        return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("ascii").rstrip("=")

    module = b"VALUE = 1\n"
    metadata_text = b"Name: mixle\nVersion: 0.8.0\n"
    module_digest = hashlib.sha256(module).hexdigest()
    record = (
        f"mixle/__init__.py,sha256={_enc(module)},{len(module)}\n"
        f"mixle-0.8.0.dist-info/METADATA,sha256={_enc(metadata_text)},{len(metadata_text)}\n"
        f"mixle-0.8.0.dist-info/RECORD,,\n"
    )
    wheel_path = root / "dist" / wheel
    with zipfile.ZipFile(wheel_path, "w") as archive:
        archive.writestr("mixle/__init__.py", module)
        archive.writestr("mixle-0.8.0.dist-info/METADATA", metadata_text)
        archive.writestr("mixle-0.8.0.dist-info/RECORD", record)
    wheel_digest = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    sdist_digest = "b" * 64

    (metadata / "release-candidate.json").write_text(
        json.dumps(
            {
                "artifact": "mixle.release_candidate/v1",
                "commit": commit,
                "tree": tree,
                "version": release,
                "tag": "v0.8.0",
                "workflow_run": "1",
            }
        ),
        encoding="utf-8",
    )
    (metadata / "SHA256SUMS").write_text(f"{wheel_digest}  {wheel}\n{sdist_digest}  {sdist}\n", encoding="utf-8")
    (metadata / f"{wheel}.json").write_text(json.dumps({"filename": wheel, "sha256": wheel_digest}), encoding="utf-8")
    (metadata / "release-check-evidence.json").write_text(_generated_check_evidence(commit), encoding="utf-8")
    (metadata / "reproduction-a.json").write_text(
        json.dumps({"artifact": "mixle.reproduction_entry_receipt/v2"}), encoding="utf-8"
    )
    return {
        "mixle/__init__.py": module_digest,
        "mixle-0.8.0.dist-info/METADATA": hashlib.sha256(metadata_text).hexdigest(),
    }


def _executing(commit: str = _CANDIDATE_COMMIT, tree: str = "7" * 40, record_hashes: dict | None = None) -> dict:
    return {
        "source_commit": commit,
        "source_tree": tree,
        "installed_record_hashes": dict(record_hashes or {}),
    }


def test_required_records_are_opened_not_merely_name_matched():
    """SYS-03 second pass: matching FILENAMES must not authorize a candidate binding.

    The first repair globbed for pathnames and never opened a match, so a directory of files
    containing ``{}`` satisfied it -- and the regression test shipped alongside it asserted exactly
    that as correct behaviour, which is how the defect survived its own fix.
    """
    import tempfile

    runner, bundle = _runner(), _bundle()
    executing = _executing()

    unresolved = runner._resolve_candidate_records(bundle, None, executing)
    assert unresolved["resolved"] is False
    assert unresolved["missing"] == bundle["candidate_binding"]["required_records"]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "metadata").mkdir()
        for pattern in bundle["candidate_binding"]["required_records"]:
            target = root / pattern.replace("*", "x")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}", encoding="utf-8")
        empty = runner._resolve_candidate_records(bundle, root, executing)
        assert empty["resolved"] is False, "files containing {} must not resolve a candidate binding"
        assert empty["problems"]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        hashes = _write_valid_records(root)
        good = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
        assert good["resolved"] is True, good["problems"]
        assert good["candidate_commit"] == _CANDIDATE_COMMIT


def test_an_artifact_from_another_commit_cannot_satisfy_the_binding():
    """The reviewer's sharpest falsifier: an older wheel executing against these records.

    Resolution previously depended only on filenames, so a receipt reported ``verified`` while the
    mixle that actually ran came from a different commit entirely.
    """
    import tempfile

    runner, bundle = _runner(), _bundle()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        hashes = _write_valid_records(root)
        wrong = runner._resolve_candidate_records(bundle, root, _executing(commit="7" * 40, record_hashes=hashes))
        assert wrong["resolved"] is False
        assert any("not the candidate commit" in problem for problem in wrong["problems"])


def test_a_wheel_record_disagreeing_with_sha256sums_is_refused():
    import tempfile

    runner, bundle = _runner(), _bundle()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        hashes = _write_valid_records(root)
        wheel_record = root / "metadata" / "mixle-0.8.0-py3-none-any.whl.json"
        wheel_record.write_text(
            json.dumps({"filename": "mixle-0.8.0-py3-none-any.whl", "sha256": "c" * 64}), encoding="utf-8"
        )
        result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
        assert result["resolved"] is False
        assert any("SHA256SUMS records" in problem for problem in result["problems"])


def test_a_source_shadowed_run_cannot_claim_candidate_binding():
    """SYS-03: executing from a source checkout is a legitimate run but not candidate evidence.

    The receipt separates the two facts. ``execution_status`` says the entry ran and matched its
    pinned output, which a local reviewer working from a checkout can legitimately produce.
    ``claim_status`` is the stronger statement -- that the receipt is evidence about the candidate
    ARTIFACT -- and it now requires the executing mixle to be an installed distribution whose
    content verifies, plus resolved candidate records.
    """
    import subprocess
    import sys
    import tempfile

    runner = _runner()

    # The probe is what decides whether a receipt may claim candidate binding, so exercise it for
    # real rather than inspecting its text. Run from a directory containing a decoy `mixle`
    # package: the probe must report the package it actually imported, which is the decoy, and
    # must not confuse that with the installed distribution.
    with tempfile.TemporaryDirectory() as directory:
        decoy = Path(directory) / "mixle"
        decoy.mkdir()
        (decoy / "__init__.py").write_text("__version__ = '0.0.0-decoy'\n", encoding="utf-8")
        probe_file = Path(directory) / "_probe.py"
        probe_file.write_text(runner._ARTIFACT_PROBE, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(probe_file)], cwd=directory, capture_output=True, text=True, timeout=120
        )
        info = json.loads(completed.stdout.strip().splitlines()[-1])

    assert info["importable"] is True
    assert isinstance(info["installed_distribution"], bool)
    # the shadowing decoy is what got imported, and the probe says so
    assert info["version"] == "0.0.0-decoy"
    assert Path(info["package_root"]).resolve().parent == decoy.resolve()
    # and it must not be credited as the installed distribution, which is the whole point
    assert info["installed_distribution"] is False


def test_receipt_reports_binding_facts_instead_of_a_constant_verified():
    """The receipt must carry the evidence for its own claim_status, not assert it."""
    runner = _runner()
    source = Path(runner.__file__ if hasattr(runner, "__file__") else "").name
    del source
    import inspect

    body = inspect.getsource(runner.run_entry)
    for field in ("executing_artifact", "candidate_binding", "unbound_reasons"):
        assert field in body, f"receipt must record {field}"
    assert '"claim_status": "verified" if not unbound else "unbound"' in body


def test_an_installation_that_is_not_the_pinned_wheel_bytes_is_refused():
    """SYS3-01: binding is to the pinned wheel's BYTES, not merely to a shared commit.

    A wheel rebuilt from the candidate sdist carries the same commit and different content, and
    comparing commit alone let it impersonate the frozen candidate. The executing installation's
    RECORD hashes must equal the pinned wheel's RECORD.
    """
    import tempfile

    runner, bundle = _runner(), _bundle()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        hashes = _write_valid_records(root)
        # same commit, same tree, but the installed module hashes to something else
        impostor = _executing(record_hashes={"mixle/__init__.py": "f" * 64})
        result = runner._resolve_candidate_records(bundle, root, impostor)
        assert result["resolved"] is False
        assert any("is not the pinned wheel" in problem for problem in result["problems"])
        # and an installation reporting NO record hashes at all is a stated problem, not a pass
        bare = runner._resolve_candidate_records(
            bundle, root, {"source_commit": _CANDIDATE_COMMIT, "source_tree": "7" * 40}
        )
        assert bare["resolved"] is False
        assert any("no installed RECORD hashes" in problem for problem in bare["problems"])
        # tree is compared too (the half an earlier docstring promised and never implemented)
        wrong_tree = runner._resolve_candidate_records(bundle, root, _executing(tree="8" * 40, record_hashes=hashes))
        assert wrong_tree["resolved"] is False
        assert any("not the candidate tree" in problem for problem in wrong_tree["problems"])


def test_pinned_wheel_bytes_absent_or_altered_beside_the_records_is_refused():
    import tempfile

    runner, bundle = _runner(), _bundle()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        hashes = _write_valid_records(root)
        wheel_path = root / "dist" / "mixle-0.8.0-py3-none-any.whl"
        wheel_path.write_bytes(wheel_path.read_bytes() + b"\x00")  # altered bytes, digest no longer matches
        result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
        assert result["resolved"] is False
        assert any("not the pinned" in problem for problem in result["problems"])
        wheel_path.unlink()
        result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
        assert any("not present at dist/" in problem for problem in result["problems"])


def test_check_evidence_is_parsed_and_must_be_terminal_passing():
    """SYS3-05: release-check-evidence.json was required but never opened; in-progress checks resolved."""
    import tempfile

    runner, bundle = _runner(), _bundle()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        hashes = _write_valid_records(root)
        # the generator's record, unmodified, resolves
        result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
        assert result["resolved"] is True, result["problems"]
        evidence = root / "metadata" / "release-check-evidence.json"
        payload = json.loads(evidence.read_text(encoding="utf-8"))
        first = bundle["candidate_binding"]["required_checks"][0]
        # a required check that is not a selected (completed/success) run: the generator never
        # writes such a value, so any non-record state means the file was not the generator's
        payload["checks"][first] = "in_progress"
        evidence.write_text(json.dumps(payload), encoding="utf-8")
        result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
        assert result["resolved"] is False
        assert any(f"{first!r} is 'in_progress'" in problem for problem in result["problems"])
        # a required check missing altogether
        del payload["checks"][first]
        evidence.write_text(json.dumps(payload), encoding="utf-8")
        result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
        assert any(f"{first!r} is absent" in problem for problem in result["problems"])
        # bound to another commit is also refused
        payload = json.loads(_generated_check_evidence())
        payload["commit"] = "9" * 40
        evidence.write_text(json.dumps(payload), encoding="utf-8")
        result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
        assert any("different commit" in problem for problem in result["problems"])


def test_check_evidence_must_be_the_generators_record_not_an_invented_shape():
    """The receipt resolver and the record generator must agree on ONE schema.

    The first SYS3-05 repair validated ``checks == {"tests": "success", "docs": ..., ...}`` -- a
    shape invented for the hand-written review-candidate stubs. scripts/verify_required_checks.py,
    which publish.yml runs, emits ``checks == {<check-run name>: {check_run_id, details_url}}`` for
    the 24 policy names and writes nothing unless all are completed/success. Under the real
    workflow every receipt would therefore have been unbound and the manifest builder would have
    refused, so no candidate could publish -- the same failure the resolver's own comment records
    fixing once before. Found the first time the generator was run on a real candidate.
    """
    import tempfile

    runner, bundle = _runner(), _bundle()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        hashes = _write_valid_records(root)
        evidence = root / "metadata" / "release-check-evidence.json"
        # the exact stub the review candidates carried
        evidence.write_text(
            json.dumps(
                {
                    "artifact": "mixle.release_check_evidence/v1",
                    "commit": _CANDIDATE_COMMIT,
                    "checks": {
                        "tests": "success",
                        "docs": "success",
                        "security": "success",
                        "extras_resolver_matrix": "success",
                    },
                }
            ),
            encoding="utf-8",
        )
        result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
        assert result["resolved"] is False
        absent = [problem for problem in result["problems"] if "is absent" in problem]
        assert len(absent) == len(bundle["candidate_binding"]["required_checks"])
        assert any("publication_authorized" in problem for problem in result["problems"])

        # generator-shaped but hollow: one bogus entry, or the same check run cited for every name,
        # or an id/URL that is not a check run's
        payload = json.loads(_generated_check_evidence())
        for corrupt in (
            lambda p: p["checks"].update(
                {name: {"check_run_id": 1, "details_url": "https://x/actions/runs/1/"} for name in p["checks"]}
            ),
            lambda p: p["checks"][next(iter(p["checks"]))].update({"check_run_id": "95335978969"}),
            lambda p: p["checks"][next(iter(p["checks"]))].update({"check_run_id": True}),
            lambda p: p["checks"][next(iter(p["checks"]))].update({"details_url": "https://github.com/gmboquet/mixle"}),
            lambda p: p.pop("publication_authorized"),
            lambda p: p.update({"publication_authorized": "true"}),
        ):
            damaged = json.loads(json.dumps(payload))
            corrupt(damaged)
            evidence.write_text(json.dumps(damaged), encoding="utf-8")
            result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
            assert result["resolved"] is False, damaged

        # and the bundle itself must carry the policy; a bundle without it cannot bind anything
        evidence.write_text(_generated_check_evidence(), encoding="utf-8")
        stripped = json.loads(json.dumps(bundle))
        del stripped["candidate_binding"]["required_checks"]
        result = runner._resolve_candidate_records(stripped, root, _executing(record_hashes=hashes))
        assert result["resolved"] is False
        assert any("must name the required checks" in problem for problem in result["problems"])
        with pytest.raises(ValueError, match="must name the required checks"):
            runner.validate_bundle(stripped)


def test_bundle_required_checks_equal_the_publication_policy():
    """The bundle's embedded check list IS the publication policy, parsed by the generator's parser."""
    generator = _load(_CHECK_EVIDENCE_GENERATOR, "_verify_required_checks_policy")
    policy = list(generator.required_check_names(_REQUIRED_CHECKS_POLICY))
    binding = _bundle()["candidate_binding"]
    assert binding["required_checks"] == policy
    assert binding["required_checks_policy"] == ".github/release-required-checks.txt"
    assert binding["check_evidence_generator"] == "scripts/verify_required_checks.py"
    # both files are content-addressed by the closure so a change to either forces regeneration
    closure = {item["path"] for item in _bundle()["closure"]}
    assert {".github/release-required-checks.txt", "scripts/verify_required_checks.py"} <= closure
    # and the workflow that consumes the record reads the same shape the resolver requires
    workflow = (ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    assert '["checks"]["pip-audit (all runtime features)"]["details_url"]' in workflow
    assert "pip-audit (all runtime features)" in policy


def test_publish_workflow_writes_receipts_outside_the_records_root_then_moves():
    """SYS3-04: redirecting stdout into records_root created an empty file the runner then read as
    a malformed prior receipt, deterministically unbinding every publication receipt."""
    workflow = (ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    loop = workflow[workflow.index("for entry in gallery-univariate") :]
    loop = loop[: loop.index("done")]
    assert '> "$RUNNER_TEMP/reproduction-$entry.json"' in loop, "receipt must be written outside records_root"
    assert 'mv "$RUNNER_TEMP/reproduction-$entry.json" "metadata/reproduction-$entry.json"' in loop
    assert '> "metadata/reproduction-$entry.json"' not in loop


def test_candidate_record_without_a_tree_is_refused():
    """SYS4-03: the tree is part of the identity contract and must be present, not merely compared
    when it happens to be there."""
    import tempfile

    runner, bundle = _runner(), _bundle()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        hashes = _write_valid_records(root)
        record_path = root / "metadata" / "release-candidate.json"
        for label, mutate in {
            "absent": lambda r: r.pop("tree", None),
            "short": lambda r: r.__setitem__("tree", "abc"),
            "uppercase": lambda r: r.__setitem__("tree", "T" * 40),
            "not a string": lambda r: r.__setitem__("tree", 12345),
        }.items():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["tree"] = "7" * 40
            mutate(record)
            record_path.write_text(json.dumps(record), encoding="utf-8")
            result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
            assert result["resolved"] is False, label
            assert any("40-hex tree" in problem for problem in result["problems"]), (label, result["problems"])


def test_a_pinned_wheel_whose_payload_disagrees_with_its_record_is_refused():
    """SYS4-01: the pinned wheel's RECORD is a set of claims about its members; they must be checked
    against the archived bytes before those declared hashes are used as candidate identity.

    The reviewer's fixture: copy the pinned wheel, alter one member's bytes, leave RECORD stale,
    update the outer digest and metadata exactly as a records producer would. Hashing the container
    against SHA256SUMS passes; only recomputing the members catches it.
    """
    import hashlib
    import tempfile
    import zipfile

    runner, bundle = _runner(), _bundle()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        hashes = _write_valid_records(root)
        wheel_path = root / "dist" / "mixle-0.8.0-py3-none-any.whl"
        # alter the member, keep RECORD, re-pin the OUTER digest so SHA256SUMS still matches
        stale = wheel_path.with_suffix(".stale")
        with zipfile.ZipFile(wheel_path) as zi, zipfile.ZipFile(stale, "w") as zo:
            for info in zi.infolist():
                data = zi.read(info.filename)
                if info.filename == "mixle/__init__.py":
                    data = data + b"# altered\n"
                zo.writestr(info, data)
        stale.replace(wheel_path)
        new_digest = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
        sums = root / "metadata" / "SHA256SUMS"
        sums.write_text(
            sums.read_text(encoding="utf-8").replace("\n", "\n", 1).split("\n")[0].split("  ")[0].join(["", ""]) or "",
            encoding="utf-8",
        ) if False else None
        lines = sums.read_text(encoding="utf-8").splitlines()
        lines = [f"{new_digest}  mixle-0.8.0-py3-none-any.whl" if l.endswith(".whl") else l for l in lines]
        sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
        meta = root / "metadata" / "mixle-0.8.0-py3-none-any.whl.json"
        meta.write_text(
            json.dumps({"filename": "mixle-0.8.0-py3-none-any.whl", "sha256": new_digest}), encoding="utf-8"
        )
        # an installation that matches the STALE RECORD (i.e. the original bytes) must not be authorized
        result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
        assert result["resolved"] is False
        assert any("does not match its RECORD hash" in problem for problem in result["problems"]), result["problems"]
