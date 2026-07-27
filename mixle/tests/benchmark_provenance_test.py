"""Worklist B7.3 -- no headline benchmark number may come from a stale artifact.

B7.3 requires that 0.8.0's published measurements are produced on 0.8.0, not carried over
from 0.5.x/0.6.x. The enforcement is provenance: ``scripts/benchmark_provenance.py``
stamps every result with the mixle version/minor/commit that produced it, and this test
proves the stamping + staleness detection works, and scans any benchmark results files in
the repo to ensure none carry a stale (different major.minor) or missing stamp.
"""

from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
PROVENANCE = ROOT / "scripts" / "benchmark_provenance.py"
BENCH_DIR = ROOT / "benchmarks"


def _mod():
    if not PROVENANCE.is_file():
        pytest.skip(f"{PROVENANCE} not found")
    spec = importlib.util.spec_from_file_location("benchmark_provenance", PROVENANCE)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_stamp_marks_the_current_version() -> None:
    prov = _mod()
    digest = "a" * 64
    commit = prov._git_commit()
    stamped = prov.stamp_result(
        {"benchmark": "gmm_fit", "seconds": 0.1},
        commit=commit,
        artifact_sha256=digest,
    )
    with (ROOT / "pyproject.toml").open("rb") as handle:
        expected = tomllib.load(handle)["project"]["version"]
    assert stamped["mixle_version"] == expected, "stamp must record this checkout's release version"
    assert stamped["mixle_minor"], "stamp must record the major.minor line"
    assert prov.is_current(
        stamped,
        current_commit=commit,
        artifact_sha256=digest,
    ), "an exact-candidate stamp must read as current"


def test_stale_and_unstamped_results_are_detected() -> None:
    prov = _mod()
    digest = "a" * 64
    commit = prov._git_commit()
    current = prov.stamp_result({"benchmark": "x"}, commit=commit, artifact_sha256=digest)
    stale = {"benchmark": "x", "mixle_version": "0.6.2", "mixle_minor": "0.6"}
    unstamped = {"benchmark": "x", "seconds": 0.1}
    bad = prov.stale_results(
        [current, stale, unstamped],
        current_commit=commit,
        artifact_sha256=digest,
    )
    assert stale in bad and unstamped in bad, "stale and unstamped results must be flagged"
    assert current not in bad, "the current-version result must not be flagged"


def test_missing_unrelated_or_abbreviated_provenance_is_rejected() -> None:
    prov = _mod()
    commit = prov._git_commit()
    digest = "a" * 64
    base = {
        "mixle_version": "0.8.0",
        "mixle_commit": commit,
        "mixle_artifact_sha256": digest,
    }
    assert prov.is_current(
        base,
        current_version="0.8.0",
        current_commit=commit,
        artifact_sha256=digest,
    )
    for changed in (
        {**base, "mixle_commit": "deadbeef"},
        {**base, "mixle_commit": "b" * 40},
        {**base, "mixle_artifact_sha256": "unknown"},
        {**base, "mixle_artifact_sha256": "b" * 64},
        {**base, "mixle_version": "0.8.0.dev1"},
        {"mixle_version": "0.8.0"},
    ):
        assert not prov.is_current(
            changed,
            current_version="0.8.0",
            current_commit=commit,
            artifact_sha256=digest,
        )


def test_minor_extraction() -> None:
    prov = _mod()
    assert prov.minor_of("0.8.0.dev1") == "0.8"
    assert prov.minor_of("0.6.2") == "0.6"


def test_no_committed_benchmark_result_is_stale() -> None:
    """Any results file in the repo must be stamped with the current release line."""
    prov = _mod()
    if not BENCH_DIR.is_dir():
        pytest.skip("no benchmarks/ directory in this checkout (harness lands separately)")
    result_files = [p for p in BENCH_DIR.rglob("*.json") if "result" in p.name.lower() and "archive" not in p.parts]
    if not result_files:
        return
    artifact_digest = "a" * 64
    commit = prov._git_commit()
    offenders = []
    for path in result_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        records = data if isinstance(data, list) else [data]
        for rec in records:
            if isinstance(rec, dict) and not prov.is_current(
                rec,
                current_commit=commit,
                artifact_sha256=artifact_digest,
            ):
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, (
        "benchmark results carry a stale or missing version stamp (B7.3 -- no headline "
        "number from old artifacts):\n" + "\n".join(offenders)
    )
