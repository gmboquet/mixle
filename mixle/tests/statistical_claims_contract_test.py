"""The enforced statistical-claim phrase classes are inventoried and carry their scope.

What this gate is, exactly (STAT-RR16-3): a LEXICAL ratchet over the three phrase classes the
scanner declares (coverage, comparative, competitive) -- not a proof of claim validity, and not
an enumeration of every sentence a reader might take as a claim. Token presence cannot validate
a statistical method: conclusion rules are validated by method-specific tests (the exact
paired-rule tests in ``task_active_test.py`` exist because a Wald gate passed this lexical check
while making an invalid inference at four discordant pairs). What the ratchet does guarantee:
text matching the enforced classes never ships without scope language, new files entering a
class get a human audit recorded in the manifest, and the negative controls below pin exactly
what each class matches. Regenerate with ``python scripts/scan_statistical_claims.py --write``
after auditing the change.
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


def test_detector_negative_and_positive_controls() -> None:
    """Pin what each phrase class does and does not match, so widenings are deliberate."""
    scanner = _load_scanner()
    must_match_coverage = (
        "the set covers the true label with probability >= 1 - alpha",
        "carries a finite-sample coverage guarantee",
        "the model's local errors are conformally bounded",
        "a coverage contract for every record",
    )
    must_not_match_coverage = (
        "coverage_by_hop_count requires at least one hop.",
        "a sample corroborates a claim when it covers at least one clause",
        "code coverage stayed at 94%",
    )
    for text in must_match_coverage:
        assert scanner._COVERAGE_CLAIM.search(text), text
    for text in must_not_match_coverage:
        assert not scanner._COVERAGE_CLAIM.search(text), text
    # negated mentions are DISCLAIMERS, suppressed line-locally (with the previous line's tail in
    # scope for wrapped sentences); a disclaimer must never be forced to carry the scope triad
    negated_coverage = (
        "This is a ranking diagnostic, not a conformal or bootstrap coverage guarantee.",
        "one verifier score cannot establish a coverage guarantee",
        "carries no coverage contract of any kind",
    )
    for text in negated_coverage:
        assert scanner._NEGATED_COVERAGE.search(text), text
    assert not scanner._NEGATED_COVERAGE.search("the set carries a finite-sample coverage guarantee")

    must_match_comparative = (
        "reaches the same student quality as random labeling for far fewer paid calls",
        "the cascade gets cheaper with use",
        "uncertainty sampling beats random",
        "active labeling measurably beats random at the same budget",
        "the winner: model A (runner-up trails by 12 elpd)",
    )
    must_not_match_comparative = (
        "fewer allocations per call in the hot path",
        "the beat frequency of the oscillator",
    )
    for text in must_match_comparative:
        assert scanner._COMPARATIVE_CLAIM.search(text), text
    for text in must_not_match_comparative:
        assert not scanner._COMPARATIVE_CLAIM.search(text), text

    must_match_competitive = (
        "mixle fits every field; the nearest rival can't",
        "pomegranate raises inside its own heterogeneous path",
        "how much of the gap closes with zero gradient steps",
    )
    must_not_match_competitive = (
        "torch is an optional dependency",
        "install scikit-learn for the comparison baselines",
    )
    for text in must_match_competitive:
        assert scanner._COMPETITIVE_CLAIM.search(text), text
    for text in must_not_match_competitive:
        assert not scanner._COMPETITIVE_CLAIM.search(text), text

    must_match_certification = (
        "certify a held-out accepted-error threshold",
        "serve the best under a certified selective-risk gate",
        "no nonempty accepted subset certifies risk <= alpha",
        "a finite-sample (alpha, delta)-PAC guarantee",
        "this is selective classification / risk control",
        "the selective-risk threshold from held-out pairs",
    )
    # ledger-provenance narration describes bookkeeping, never a guarantee offered to a caller
    must_not_match_certification = (
        "reloaded an uncertified threshold as certified",
        "which rows certified the CURRENT conformal threshold",
        "kept serving was serving an uncertified threshold, with nothing in the object saying so",
        "the certificate authority rotated its keys",
    )
    for text in must_match_certification:
        assert scanner._CERTIFICATION_CLAIM.search(text), text
    for text in must_not_match_certification:
        assert not scanner._CERTIFICATION_CLAIM.search(text), text


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
