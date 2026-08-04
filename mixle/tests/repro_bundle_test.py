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
    assert runner.validate_bundle(tracked) is tracked
    assert tracked["candidate_binding"]["required_records"]
    assert tracked["acceptance"]
    assert tracked["code_license"]["spdx"] == "MIT"


@pytest.mark.parametrize(
    "entry_id",
    ["gallery-univariate", "gallery-structured", "production-provenance", "scaling-backend"],
)
def test_every_local_entry_reproduces_exact_expected_output(entry_id):
    runner = _load(ROOT / "scripts" / "run_repro_entry.py", f"_run_repro_entry_{entry_id}")
    receipt = runner.run_entry(_bundle(), entry_id)
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
