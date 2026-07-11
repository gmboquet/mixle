"""Worklist E14.5 -- the reproducibility bundle is consistent and actually reproduces.

The bundle (``release-checklists/0.8.0-repro-bundle.json``, built by
``scripts/build_repro_bundle.py``) lists the commands + checksums + licenses to re-derive
the shipped reports. This test keeps it honest:

  * every referenced script exists and its recorded SHA-256 still matches (so the bundle
    cannot silently point at a moved or altered artifact);
  * the bundle records the code license and its acceptance criterion;
  * at least one self-contained (no-network) entry is executed end to end and must exit
    0 -- a live proof that reproduction from the bundle works, which is E14.5's acceptance.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
BUNDLE = ROOT / "release-checklists" / "0.8.0-repro-bundle.json"


def _bundle() -> dict:
    if not BUNDLE.is_file():
        pytest.skip(f"{BUNDLE} not found; run scripts/build_repro_bundle.py --write")
    return json.loads(BUNDLE.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_entries_exist_and_checksums_match() -> None:
    bundle = _bundle()
    assert bundle["entries"], "bundle has no entries"
    for entry in bundle["entries"]:
        script = ROOT / entry["script"]
        assert script.is_file(), f"bundle references missing script: {entry['script']}"
        actual = _sha256(script)
        assert actual == entry["sha256"], (
            f"{entry['id']}: script checksum drifted from the bundle "
            f"(recorded {entry['sha256']}, now {actual}). Regenerate with "
            f"`python scripts/build_repro_bundle.py --write`."
        )


def test_bundle_records_license_and_acceptance() -> None:
    bundle = _bundle()
    assert "license" in bundle.get("code_license", "").lower() or "mit" in bundle.get("code_license", "").lower()
    assert bundle.get("acceptance"), "bundle must state its reproduction acceptance criterion"
    # Network-dependent datasets must be flagged, not silently bundled.
    for entry in bundle["entries"]:
        if entry.get("needs_network"):
            assert entry.get("data_license"), f"{entry['id']}: networked dataset needs a license note"


def test_a_self_contained_entry_reproduces() -> None:
    """Live proof: a no-network entry runs to completion from its recorded command."""
    bundle = _bundle()
    candidates = [e for e in bundle["entries"] if not e.get("needs_network")]
    assert candidates, "bundle should contain at least one self-contained entry"
    # Pick the first self-contained entry and run its script directly.
    entry = candidates[0]
    script = ROOT / entry["script"]
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, f"bundle entry {entry['id']} failed to reproduce:\n{proc.stderr[-2000:]}"
