"""Statistical claims are inventoried and carry their scope (the pass-15 root-cause gate).

Seven review passes found the same defect in different clothes: a free-text statistical claim --
a docstring, an example header, a report label, a test comment -- stronger than what the
mathematics or the experiment establishes. ``scripts/scan_statistical_claims.py`` enumerates the
user-facing claim surfaces; this test is the ratchet: a NEW claim-bearing file must be audited
into the manifest, an inventoried file must keep its scope tokens (marginal/exchangeability/shift
for coverage claims; uncertainty-or-reduced wording for comparative claims), and stale entries
must be pruned. Regenerate with ``python scripts/scan_statistical_claims.py --write`` after
auditing the change.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCANNER = Path(__file__).resolve().parent.parent.parent / "scripts" / "scan_statistical_claims.py"


def _load_scanner():
    if not _SCANNER.is_file():
        pytest.skip(f"scanner not found at {_SCANNER}")
    spec = importlib.util.spec_from_file_location("scan_statistical_claims", _SCANNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_claim_site_carries_its_scope() -> None:
    scanner = _load_scanner()
    violations = {rel: e["violations"] for rel, e in scanner.scan().items() if e["violations"]}
    assert not violations, (
        "statistical claims without their scope (coverage claims need marginal/exchangeability/"
        "shift statements; comparative claims need uncertainty language or reduced wording):\n"
        + "\n".join(f"  - {rel}: {v}" for rel, v in sorted(violations.items()))
    )


def test_claim_sites_match_the_reviewed_inventory() -> None:
    scanner = _load_scanner()
    current = {rel: sorted(set(e["classes"])) for rel, e in scanner.scan().items()}
    manifest = scanner.load_manifest()["sites"]
    new = {rel: cls for rel, cls in current.items() if rel not in manifest}
    assert not new, (
        "new statistical-claim sites outside the reviewed inventory -- audit their scope, then "
        "regenerate with `python scripts/scan_statistical_claims.py --write`:\n"
        + "\n".join(f"  - {rel}: {cls}" for rel, cls in sorted(new.items()))
    )
    stale = {rel: cls for rel, cls in manifest.items() if rel not in current}
    assert not stale, (
        "the claims manifest lists sites that no longer make claims -- prune with "
        "`python scripts/scan_statistical_claims.py --write`:\n"
        + "\n".join(f"  - {rel}: {cls}" for rel, cls in sorted(stale.items()))
    )
