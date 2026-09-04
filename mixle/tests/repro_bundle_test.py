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
    runner = _load(ROOT / "scripts" / "run_repro_entry.py", "_run_repro_entry")

    # hermetic by default: no test here may reach the real gh CLI (or the network behind it); a
    # test that wants to observe the gh command line installs its own fake on this seam
    def _no_gh(command):
        raise FileNotFoundError("gh is not consulted by unit tests")

    runner._run_gh = _no_gh
    # ...and the resolved-executable identity the receipt records is a fixture too
    runner._gh_identity = lambda: {
        "executable": "/fixture/bin/gh",
        "version": "gh version 2.93.0 (fixture)",
        "sha256": "f" * 64,
    }
    return runner


_CANDIDATE_COMMIT = "e" * 40
_REQUIRED_CHECKS_POLICY = ROOT / ".github" / "release-required-checks.txt"
_CHECK_EVIDENCE_GENERATOR = ROOT / "scripts" / "verify_required_checks.py"
_CANDIDATE_RECORD_PRODUCER = ROOT / "scripts" / "release_candidate_record.py"


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
                "app": {"slug": "github-actions"},
                "details_url": f"https://github.com/gmboquet/mixle/actions/runs/{500 + index}/job/{first_id + index}",
            }
            for index, name in enumerate(names)
        ],
    }


def _generated_check_evidence(commit: str = _CANDIDATE_COMMIT) -> str:
    return _generated_check_evidence_and_payload(commit)[0]


def _generated_check_evidence_and_payload(commit: str = _CANDIDATE_COMMIT) -> tuple[str, bytes]:
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
        payload_bytes = payload.read_bytes()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            status = generator.main(
                [
                    "--required",
                    str(_REQUIRED_CHECKS_POLICY),
                    "--input",
                    str(payload),
                    "--sha",
                    commit,
                    "--repository",
                    "gmboquet/mixle",
                ]
            )
    assert status == 0, "generator refused an all-success payload"
    return stdout.getvalue(), payload_bytes


def _attest(runner, root: Path) -> None:
    """Stand in for GitHub: the attestation verifier accepts exactly the record at ``root`` as signed.

    Real verification shells out to ``gh attestation verify`` over a Sigstore bundle that only a
    workflow of this repository can produce; a unit test cannot mint one. So the verifier is replaced
    by one that accepts precisely the digest of the record the fixture wrote and refuses every other,
    which is the property the real one has. The gh command line itself is asserted separately.
    """
    issued = hashlib.sha256((root / "metadata" / "release-check-evidence.json").read_bytes()).hexdigest()

    def verify(record_path, bundle_path, contract, commit):
        digest = hashlib.sha256(Path(record_path).read_bytes()).hexdigest()
        if digest != issued:
            raise ValueError("check-evidence attestation did not verify: no attestation names this record")
        return {
            "attested": True,
            "record_sha256": digest,
            "signer_workflow": f"https://github.com/{contract['repository']}/{contract['signer_workflows'][0]}@refs/heads/x",
            "source_repository": f"https://github.com/{contract['repository']}",
            "source_digest": commit,
            "run_invocation_uri": f"https://github.com/{contract['repository']}/actions/runs/1/attempts/1",
            "verifier": "test double for gh attestation verify",
        }

    runner._verify_check_evidence_attestation = verify


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

    # written by the record's one real producer (scripts/release_candidate_record.py, called by
    # publish.yml), not by hand: the hand-written fixture carried a `tree` the workflow's inline
    # writer never emitted, and the resolver was repaired against the fixture (see D-0180)
    producer = _load(_CANDIDATE_RECORD_PRODUCER, "_release_candidate_record_for_bundle")
    (metadata / "release-candidate.json").write_text(
        producer.render(
            producer.release_candidate_record(
                commit=commit, tree=tree, tag=f"v{release}", version=release, workflow_run="1"
            )
        ),
        encoding="utf-8",
    )
    (metadata / "SHA256SUMS").write_text(f"{wheel_digest}  {wheel}\n{sdist_digest}  {sdist}\n", encoding="utf-8")
    (metadata / f"{wheel}.json").write_text(json.dumps({"filename": wheel, "sha256": wheel_digest}), encoding="utf-8")
    record, payload = _generated_check_evidence_and_payload(commit)
    (metadata / "release-check-evidence.json").write_text(record, encoding="utf-8")
    (metadata / "check-runs.json").write_bytes(payload)
    # the attestation bundle is opaque to the resolver -- gh reads it; the unit tests replace the
    # gh call (see _attest) and assert its argv separately
    (metadata / "release-check-evidence.sigstore.json").write_text('{"fixture": "sigstore bundle"}\n', encoding="utf-8")
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
        _attest(runner, root)
        good = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
        assert good["resolved"] is True, good["problems"]
        assert good["candidate_commit"] == _CANDIDATE_COMMIT
        assert good["check_evidence"]["attested"] is True
        assert (
            good["check_evidence"]["record_sha256"]
            == hashlib.sha256((root / "metadata" / "release-check-evidence.json").read_bytes()).hexdigest()
        )


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
        _attest(runner, root)
        # the generator's record, unmodified and attested, resolves
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
    every policy name and writes nothing unless all are completed/success. Under the real
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


def test_an_unissued_but_well_shaped_check_record_is_refused():
    """SYS5-01: shape is not approval. A record with every required name, distinct integer ids,
    ``publication_authorized: true`` and URLs containing ``/actions/runs/`` -- authored by hand, its cited
    runs never issued -- produced four verified receipts and a complete manifest. Now the record must
    (1) point at THIS repository's Actions job pages whose job segment is the check-run id, (2) commit
    to and re-derive from a retained check-runs payload, and (3) carry a GitHub attestation from one
    of the bundle's signing workflows at the candidate commit. A forger can fake (1) and (2); the
    attestation is what they cannot produce, and it is required.
    """
    import tempfile

    runner, bundle = _runner(), _bundle()
    names = bundle["candidate_binding"]["required_checks"]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        hashes = _write_valid_records(root)
        _attest(runner, root)  # GitHub signed exactly the fixture's record
        evidence = root / "metadata" / "release-check-evidence.json"

        # the reviewer's exact forgery: unique positive ids, invented example.invalid URLs
        forged = {
            "artifact": "mixle.release_check_evidence/v1",
            "commit": _CANDIDATE_COMMIT,
            "checks": {
                name: {
                    "check_run_id": 900000 + index,
                    "details_url": f"https://example.invalid/actions/runs/{700 + index}/job/{900000 + index}",
                }
                for index, name in enumerate(names)
            },
            "publication_authorized": True,
        }
        evidence.write_text(json.dumps(forged), encoding="utf-8")
        result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
        assert result["resolved"] is False
        assert sum("does not point at a gmboquet/mixle Actions job page" in p for p in result["problems"]) == len(names)
        assert any("attestation did not verify" in p for p in result["problems"])
        assert result["check_evidence"]["attested"] is False

        # a more careful forgery: repo-correct URLs whose job segment is the id, a matching payload
        # retained beside it (so it re-derives), the source block filled in -- everything but the
        # signature. Still refused, and only by the attestation.
        payload = _github_check_runs(_CANDIDATE_COMMIT, names, first_id=900000)
        payload_bytes = json.dumps(payload).encode("utf-8")
        (root / "metadata" / "check-runs.json").write_bytes(payload_bytes)
        careful = json.loads(_generated_check_evidence())  # generator shape for the digest layout
        careful["checks"] = {
            name: {
                "check_run_id": 900000 + index,
                "details_url": f"https://github.com/gmboquet/mixle/actions/runs/{500 + index}/job/{900000 + index}",
            }
            for index, name in enumerate(names)
        }
        careful["source"] = dict(careful["source"], check_runs_sha256=hashlib.sha256(payload_bytes).hexdigest())
        evidence.write_text(json.dumps(careful), encoding="utf-8")
        result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
        assert result["resolved"] is False
        assert not any("Actions job page" in p or "re-derive" in p or "hashes to" in p for p in result["problems"]), (
            result["problems"]
        )
        assert [p for p in result["problems"] if "attestation" in p], result["problems"]

        # and the genuine record, restored, resolves again
        record, genuine_payload = _generated_check_evidence_and_payload()
        evidence.write_text(record, encoding="utf-8")
        (root / "metadata" / "check-runs.json").write_bytes(genuine_payload)
        _attest(runner, root)
        result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
        assert result["resolved"] is True, result["problems"]


def test_check_evidence_must_re_derive_from_the_retained_payload():
    """The record commits to the payload it came from; both halves are checked."""
    import tempfile

    runner, bundle = _runner(), _bundle()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        hashes = _write_valid_records(root)
        _attest(runner, root)
        payload_path = root / "metadata" / "check-runs.json"
        evidence = root / "metadata" / "release-check-evidence.json"
        original_payload = payload_path.read_bytes()

        # payload altered -> digest mismatch
        payload_path.write_bytes(original_payload + b"\n")
        result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
        assert any("the record commits to" in p for p in result["problems"])
        payload_path.write_bytes(original_payload)

        # payload absent
        payload_path.unlink()
        result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
        assert any("absent: metadata/check-runs.json" in p for p in result["problems"])
        payload_path.write_bytes(original_payload)

        # payload intact but the record's selection edited (a different, real-looking id): the
        # attestation would refuse it anyway, but re-derivation names the defect precisely
        payload = json.loads(evidence.read_text(encoding="utf-8"))
        first = bundle["candidate_binding"]["required_checks"][0]
        payload["checks"][first]["check_run_id"] += 1
        payload["checks"][first]["details_url"] = payload["checks"][first]["details_url"][:-1] + "1"
        evidence.write_text(json.dumps(payload), encoding="utf-8")
        result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
        assert any("do not re-derive" in p for p in result["problems"]), result["problems"]

        # a payload that does not approve (one required run failed) cannot back a record either
        names = bundle["candidate_binding"]["required_checks"]
        failing = _github_check_runs(_CANDIDATE_COMMIT, names)
        failing["check_runs"][0]["conclusion"] = "failure"
        failing_bytes = json.dumps(failing).encode("utf-8")
        payload_path.write_bytes(failing_bytes)
        record = json.loads(_generated_check_evidence())
        record["source"]["check_runs_sha256"] = hashlib.sha256(failing_bytes).hexdigest()
        evidence.write_text(json.dumps(record), encoding="utf-8")
        result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
        assert any("does not approve this candidate" in p for p in result["problems"]), result["problems"]


def test_attestation_is_verified_offline_with_gh_bound_to_repo_workflow_and_commit():
    """The verifier's exact gh command line is the binding; assert every flag that makes it one."""
    import subprocess
    import tempfile

    runner, bundle = _runner(), _bundle()
    contract = runner._attestation_contract(bundle)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        hashes = _write_valid_records(root)
        record = root / "metadata" / "release-check-evidence.json"
        digest = hashlib.sha256(record.read_bytes()).hexdigest()
        seen: list[list[str]] = []

        def gh_response(**overrides):
            # the shape and fields real gh returned for the fb8c02d6 candidate (retained verify.json)
            certificate = {
                "buildSignerURI": "https://github.com/gmboquet/mixle/.github/workflows/publish.yml@refs/tags/v0.8.0",
                "subjectAlternativeName": "https://github.com/gmboquet/mixle/.github/workflows/publish.yml@refs/tags/v0.8.0",
                "issuer": "https://token.actions.githubusercontent.com",
                "sourceRepositoryURI": "https://github.com/gmboquet/mixle",
                "sourceRepositoryDigest": _CANDIDATE_COMMIT,
                "githubWorkflowRepository": "gmboquet/mixle",
                "githubWorkflowSHA": _CANDIDATE_COMMIT,
                "runnerEnvironment": "github-hosted",
                "runInvocationURI": "https://github.com/gmboquet/mixle/actions/runs/1/attempts/1",
            }
            statement = {
                "predicateType": contract["predicate_type"],
                "predicate": {"commit": _CANDIDATE_COMMIT, "phase": "review-candidate"},
                "subject": [{"name": "release-check-evidence.json", "digest": {"sha256": digest}}],
            }
            certificate.update({k: v for k, v in overrides.items() if k in certificate})
            statement.update({k: v for k, v in overrides.items() if k in statement})
            return [{"verificationResult": {"statement": statement, "signature": {"certificate": certificate}}}]

        def fake_run(command):
            seen.append(list(command))
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(gh_response()), stderr="")

        # the resolver's gh call goes through a module-local seam; replacing the process-wide
        # subprocess.run here once leaked a fake gh into every later test's subprocesses
        runner._run_gh = fake_run
        result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
        assert result["resolved"] is True, result["problems"]
        assert result["check_evidence"]["source_digest"] == _CANDIDATE_COMMIT
        (command,) = seen
        # the resolved executable, not a bare name, is what runs -- and the receipt names it
        assert command[:3] == ["/fixture/bin/gh", "attestation", "verify"]
        assert command[3] == str(record)
        assert result["check_evidence"]["verifier"]["executable"] == "/fixture/bin/gh"
        assert result["check_evidence"]["verifier"]["sha256"] == "f" * 64
        assert "trust_prerequisite" in result["check_evidence"]["verifier"]
        joined = " ".join(command)
        assert f"--bundle {root / 'metadata' / 'release-check-evidence.sigstore.json'}" in joined
        assert "--repo gmboquet/mixle" in joined
        assert f"--source-digest {_CANDIDATE_COMMIT}" in joined
        assert f"--predicate-type {contract['predicate_type']}" in joined
        assert "--cert-oidc-issuer https://token.actions.githubusercontent.com" in joined
        assert "--deny-self-hosted-runners" in joined and "--format json" in joined
        # gh refuses --signer-repo together with --cert-identity-regex (the first repair passed
        # both and could never have verified anything real), and a trusted root read from the
        # caller's records is no trust anchor: neither flag may appear
        assert "--signer-repo" not in joined and "--custom-trusted-root" not in joined
        identity = command[command.index("--cert-identity-regex") + 1]
        assert identity.startswith("^https://github\\.com/gmboquet/mixle/(")
        for workflow in contract["signer_workflows"]:
            assert workflow.replace(".", "\\.") in identity
        assert identity.endswith(")@")
        # the identity regex is exactly the two workflow alternatives -- verified against a real SAN
        import re as _re

        assert _re.match(
            identity, "https://github.com/gmboquet/mixle/.github/workflows/tests.yml@refs/heads/release/0.8.0"
        )
        assert _re.match(identity, "https://github.com/gmboquet/mixle/.github/workflows/publish.yml@refs/tags/v0.8.0")
        assert not _re.match(identity, "https://github.com/gmboquet/mixle/.github/workflows/docs.yml@refs/heads/main")
        assert not _re.match(
            identity, "https://github.com/gmboquet/mixle-fork/.github/workflows/tests.yml@refs/heads/x"
        )

        # gh says the signed subject is a DIFFERENT digest -> refused
        def other_subject(command):
            payload = json.loads(fake_run(command).stdout)
            payload[0]["verificationResult"]["statement"]["subject"][0]["digest"]["sha256"] = "0" * 64
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

        runner._run_gh = other_subject
        result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
        assert any("no verified attestation binds this record" in p for p in result["problems"])

        # nested values that are not objects are gh's-shape violations: each must be a structured
        # refusal (unbound receipt), never an exception out of the resolver (pass 7, SYS7-E02)
        for label, response in (
            ("verificationResult is a string", [{"verificationResult": "ok"}]),
            ("statement is a list", [{"verificationResult": {"statement": ["x"], "signature": {}}}]),
            (
                "signature is an int",
                [
                    {
                        "verificationResult": {
                            "statement": gh_response()[0]["verificationResult"]["statement"],
                            "signature": 7,
                        }
                    }
                ],
            ),
            (
                "subject entries are strings",
                [
                    {
                        "verificationResult": {
                            "statement": {"predicateType": contract["predicate_type"], "subject": ["x"]},
                            "signature": {},
                        }
                    }
                ],
            ),
            ("top level is an object", {"verificationResult": {}}),
            ("results contain a non-object", ["x", 3]),
        ):
            runner._run_gh = lambda command, response=response: subprocess.CompletedProcess(
                command, 0, stdout=json.dumps(response), stderr=""
            )
            result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
            assert result["resolved"] is False, label
            assert result["check_evidence"]["attested"] is False, label
            assert any("attestation" in p for p in result["problems"]), (label, result["problems"])

        # gh's exit code is not the verdict; its RESPONSE must bind every flag's expectation. The
        # reviewer's stand-in -- digest and predicate type only -- was accepted before (pass 6).
        seen.clear()
        stub = [
            {
                "verificationResult": {
                    "statement": {
                        "predicateType": contract["predicate_type"],
                        "subject": [{"digest": {"sha256": digest}}],
                    },
                    "signature": {"certificate": {}},
                }
            }
        ]
        runner._run_gh = lambda command: subprocess.CompletedProcess(command, 0, stdout=json.dumps(stub), stderr="")
        result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
        assert result["resolved"] is False
        assert any("does not bind: predicate.commit, signer identity, OIDC issuer" in p for p in result["problems"]), (
            result["problems"]
        )
        for field, wrong in (
            ("sourceRepositoryDigest", "0" * 40),
            ("githubWorkflowSHA", "0" * 40),
            ("sourceRepositoryURI", "https://github.com/gmboquet/mixle-fork"),
            ("githubWorkflowRepository", "gmboquet/mixle-fork"),
            ("issuer", "https://accounts.google.com"),
            ("runnerEnvironment", "self-hosted"),
            ("runInvocationURI", "https://github.com/someone/else/actions/runs/1/attempts/1"),
            ("buildSignerURI", "https://github.com/gmboquet/mixle/.github/workflows/docs.yml@refs/heads/main"),
            ("predicate", {"commit": "1" * 40}),
        ):
            response = gh_response(**{field: wrong})
            if field == "buildSignerURI":
                response[0]["verificationResult"]["signature"]["certificate"]["subjectAlternativeName"] = wrong
            runner._run_gh = lambda command, response=response: subprocess.CompletedProcess(
                command, 0, stdout=json.dumps(response), stderr=""
            )
            result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
            assert result["resolved"] is False, field
            assert any("does not bind" in p for p in result["problems"]), (field, result["problems"])

        # gh refuses -> refused, with gh's last line
        runner._run_gh = lambda command: subprocess.CompletedProcess(
            command, 1, stdout="", stderr="Error: verifying with issuer\nfailed to verify"
        )
        result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
        assert any("attestation did not verify: failed to verify" in p for p in result["problems"])

        # gh absent -> refused, and says so
        def no_gh(command):
            raise FileNotFoundError("gh")

        runner._run_gh = no_gh
        result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
        assert any("gh CLI is unavailable" in p for p in result["problems"])

        # bundle missing -> named absent, no gh call
        runner._run_gh = fake_run
        seen.clear()
        (root / "metadata" / "release-check-evidence.sigstore.json").unlink()
        result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
        assert any("release-check-evidence.sigstore.json (the attestation bundle)" in p for p in result["problems"])
        assert seen == []


def test_a_pinned_wheel_with_a_truncated_central_directory_is_a_structured_problem():
    """SYS5-03: zipfile.BadZipFile escaped the record-resolution boundary as a traceback (exit 1)
    instead of an unbound receipt with its reason. Every way the archive can be unreadable is now a
    problem string, and the CLI exits 0 with claim_status unbound."""
    import tempfile

    runner, bundle = _runner(), _bundle()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        hashes = _write_valid_records(root)
        _attest(runner, root)
        wheel_path = root / "dist" / "mixle-0.8.0-py3-none-any.whl"
        original = wheel_path.read_bytes()
        truncated = original[: len(original) - 40]  # cuts into the central directory / EOCD
        wheel_path.write_bytes(truncated)
        # keep SHA256SUMS honest about the bytes so the failure is the archive, not the digest
        sums = root / "metadata" / "SHA256SUMS"
        sums.write_text(
            sums.read_text(encoding="utf-8").replace(
                hashlib.sha256(original).hexdigest(), hashlib.sha256(truncated).hexdigest()
            ),
            encoding="utf-8",
        )
        (root / "metadata" / "mixle-0.8.0-py3-none-any.whl.json").write_text(
            json.dumps({"filename": "mixle-0.8.0-py3-none-any.whl", "sha256": hashlib.sha256(truncated).hexdigest()}),
            encoding="utf-8",
        )
        result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
        assert result["resolved"] is False
        assert any("pinned wheel RECORD could not be read: BadZipFile" in p for p in result["problems"]), result[
            "problems"
        ]


def test_check_evidence_workflows_attest_and_verify_and_the_bundle_names_them():
    """The signing workflows named by the bundle exist on disk, attest with a pinned action, retain the
    payload and bundle, and verify with the same flags the resolver uses -- and none of the flags gh
    refuses to combine, nor a trusted root read from the artifact."""
    contract = _bundle()["candidate_binding"]["check_evidence_attestation"]
    assert _bundle()["candidate_binding"]["repository"] == "gmboquet/mixle"
    assert contract["signer_workflows"] == [".github/workflows/publish.yml", ".github/workflows/tests.yml"]
    for workflow in contract["signer_workflows"]:
        text = (ROOT / workflow).read_text(encoding="utf-8")
        assert "uses: actions/attest@1e69f48acb82d1966a394da916b4c1698aa569d6" in text
        assert f"predicate-type: {contract['predicate_type']}" in text
        assert "id-token: write" in text and "attestations: write" in text
        assert '--repository "$REPOSITORY"' in text
        for flag in (
            '--repo "$REPOSITORY"',
            '--source-digest "$SHA"',
            f"--predicate-type {contract['predicate_type']}",
            "--cert-identity-regex",
            "--cert-oidc-issuer https://token.actions.githubusercontent.com",
            "--deny-self-hosted-runners",
        ):
            assert flag in text, (workflow, flag)
        assert "--signer-repo" not in text and "--custom-trusted-root" not in text and "trusted-root" not in text
        for record in ("check_runs_record", "bundle_record"):
            assert Path(contract[record]).name in text, (workflow, record)
    tests = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert "if: github.event_name == 'workflow_dispatch'" in tests.split("release-check-evidence:", 1)[1]
    assert "release-check-evidence-${{ github.sha }}-attempt-${{ github.run_attempt }}" in tests
    publish = (ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    assert "--check-evidence candidate/metadata/release-check-evidence.json" in publish
    assert "--check-evidence-bundle candidate/metadata/release-check-evidence.sigstore.json" in publish
    assert "--live-check-evidence promotion/release-check-evidence.json" in publish
    assert "--live-check-evidence-bundle promotion/release-check-evidence.sigstore.json" in publish
    assert "path: ${{ runner.temp }}/approval/" in publish
    assert not (ROOT / ".github" / "workflows" / "release-check-evidence.yml").exists()


def test_every_archive_read_failure_is_a_structured_problem_not_only_bad_zip_file():
    """Attacking the SYS5-03 repair: the first fix enumerated exception classes and missed what the
    decompressors raise. A pinned wheel whose RECORD member is LZMA-compressed and corrupt raised
    lzma.LZMAError (likewise zlib.error, NotImplementedError for unsupported methods/flags,
    RuntimeError for an encrypted member) straight through the boundary. Every failure to read the
    untrusted archive is now the same structured problem, with the exception type named."""
    import tempfile
    import zipfile

    runner, bundle = _runner(), _bundle()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        hashes = _write_valid_records(root)
        _attest(runner, root)
        wheel_path = root / "dist" / "mixle-0.8.0-py3-none-any.whl"
        # rewrite the fixture wheel with an LZMA-compressed RECORD, then corrupt that member's data
        with zipfile.ZipFile(wheel_path) as archive:
            members = [(info.filename, archive.read(info.filename)) for info in archive.infolist()]
        with zipfile.ZipFile(wheel_path, "w") as archive:
            for name, data in members:
                method = zipfile.ZIP_LZMA if name.endswith("RECORD") else zipfile.ZIP_STORED
                archive.writestr(name, data, compress_type=method)
        with zipfile.ZipFile(wheel_path) as archive:
            record = next(info for info in archive.infolist() if info.filename.endswith("RECORD"))
            data_start = record.header_offset + 30 + len(record.filename.encode()) + len(record.extra)
            assert record.compress_type == zipfile.ZIP_LZMA and record.compress_size > 8
        raw = bytearray(wheel_path.read_bytes())
        for offset in range(data_start + 4, data_start + record.compress_size):
            raw[offset] ^= 0xFF
        wheel_path.write_bytes(bytes(raw))
        digest = hashlib.sha256(bytes(raw)).hexdigest()
        sums = root / "metadata" / "SHA256SUMS"
        sums.write_text(
            "\n".join(
                (f"{digest}  mixle-0.8.0-py3-none-any.whl" if line.endswith("mixle-0.8.0-py3-none-any.whl") else line)
                for line in sums.read_text(encoding="utf-8").splitlines()
            )
            + "\n",
            encoding="utf-8",
        )
        (root / "metadata" / "mixle-0.8.0-py3-none-any.whl.json").write_text(
            json.dumps({"filename": "mixle-0.8.0-py3-none-any.whl", "sha256": digest}), encoding="utf-8"
        )
        result = runner._resolve_candidate_records(bundle, root, _executing(record_hashes=hashes))
        assert result["resolved"] is False
        problems = [p for p in result["problems"] if p.startswith("pinned wheel RECORD could not be read: ")]
        assert problems, result["problems"]
        assert not any("BadZipFile" in p for p in problems), problems  # this one is a decompressor error
        assert any("LZMAError" in p or "zlib" in p or "Error" in p for p in problems), problems
