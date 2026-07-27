"""The reproducibility bundle is a complete, executable, candidate-bound closure."""

from __future__ import annotations

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


def test_network_entry_is_executed_by_the_hosted_optional_lane():
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert (
        "python scripts/run_repro_entry.py --entry flagship-banking77-cascade"
        in workflow
    )


def test_bundle_rejects_unresolved_license_and_integrity_placeholders():
    serialized = json.dumps(_bundle(), sort_keys=True).upper()
    for marker in ("CONFIRM-AT-PUBLISH", "TODO", "TBD"):
        assert marker not in serialized
