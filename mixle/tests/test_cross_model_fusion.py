"""L5 DoD -- cross-model adjudication & fusion (notes/exec/workstream-L.md).

``fuse_claims`` is the precision-weighted product-of-experts rule already stated at the top of
``reason/fusion.py`` (``prec_fused = sum(prec_i) + prior_prec``; ``mean = sum(prec_i * value_i) /
prec_fused``), applied to scalar claims from independent external models instead of learned tokens.
Two model stubs (a synthetic "cmipA"/"cmipB" climate-projection pair) exercise both branches:

* strongly disagreeing claims (``|2 - 4| / sqrt(0.1 + 0.1) = 4.47 > 3`` sigma) trip ``disagreement``
  and -- since no claim survives cross-model adjudication when both models are that far apart --
  ``abstained`` too, so a driller-facing number is never quietly averaged out of a real conflict.
* near-agreeing claims (``2.0`` vs ``2.1``) stay well under the 3-sigma flag and fuse normally.
"""

from __future__ import annotations

import math

import pytest

from mixle.reason.fusion import FusedBelief, ModelClaim, fuse_claims


def test_disagreeing_stubs_fused():
    disagreeing = fuse_claims(
        [
            ModelClaim(value=2.0, variance=0.1, model_id="cmipA", version="v1", content_hash="a" * 64),
            ModelClaim(value=4.0, variance=0.1, model_id="cmipB", version="v1", content_hash="b" * 64),
        ]
    )
    assert isinstance(disagreeing, FusedBelief)
    assert disagreeing.mean == pytest.approx(3.0, abs=1e-9)
    assert disagreeing.weights["cmipA"] == pytest.approx(0.5, abs=1e-9)
    assert disagreeing.weights["cmipB"] == pytest.approx(0.5, abs=1e-9)
    z = abs(2.0 - 4.0) / math.sqrt(0.1 + 0.1)
    assert z == pytest.approx(4.4721, abs=1e-3)
    assert disagreeing.disagreement is True
    assert disagreeing.abstained is True

    agreeing = fuse_claims(
        [
            ModelClaim(value=2.0, variance=0.1, model_id="cmipA", version="v1", content_hash="a" * 64),
            ModelClaim(value=2.1, variance=0.1, model_id="cmipB", version="v1", content_hash="b" * 64),
        ]
    )
    assert agreeing.disagreement is False
    assert agreeing.abstained is False


def test_weights_track_reliability_and_provenance_is_attributable():
    fused = fuse_claims(
        [
            ModelClaim(
                value=1.0, variance=0.2, model_id="modelA", version="v1", content_hash="c" * 64, reliability=3.0
            ),
            ModelClaim(
                value=2.0, variance=0.2, model_id="modelB", version="v1", content_hash="d" * 64, reliability=1.0
            ),
        ]
    )
    # higher reliability -> higher precision -> higher weight, in the frozen 3:1 ratio
    assert fused.weights["modelA"] == pytest.approx(0.75, abs=1e-9)
    assert fused.weights["modelB"] == pytest.approx(0.25, abs=1e-9)
    assert sum(fused.weights.values()) == pytest.approx(1.0, abs=1e-9)

    by_id = {entry["model_id"]: entry for entry in fused.provenance["claims"]}
    assert by_id["modelA"]["content_hash"] == "c" * 64
    assert by_id["modelA"]["version"] == "v1"
    assert by_id["modelA"]["weight"] == pytest.approx(0.75, abs=1e-9)
    assert by_id["modelB"]["content_hash"] == "d" * 64


def test_prior_precision_pulls_the_fused_mean_and_shrinks_variance():
    claims = [
        ModelClaim(value=5.0, variance=1.0, model_id="only", version="v1", content_hash="e" * 64),
    ]
    unregularized = fuse_claims(claims)
    regularized = fuse_claims(claims, prior_prec=1.0)
    assert unregularized.mean == pytest.approx(5.0, abs=1e-9)
    assert regularized.mean == pytest.approx(2.5, abs=1e-9)  # (1*5 + 0) / (1 + 1)
    assert regularized.variance < unregularized.variance


def test_verifier_accepting_a_claim_prevents_abstention_on_disagreement():
    class _AlwaysPass:
        def verify(self, claim, context):
            class _Verdict:
                passed = True

            return _Verdict()

    fused = fuse_claims(
        [
            ModelClaim(value=2.0, variance=0.1, model_id="cmipA", version="v1", content_hash="a" * 64),
            ModelClaim(value=4.0, variance=0.1, model_id="cmipB", version="v1", content_hash="b" * 64),
        ],
        verifier=_AlwaysPass(),
    )
    assert fused.disagreement is True
    assert fused.abstained is False  # the verifier vouched for a claim despite the raw disagreement


def test_rejects_non_positive_variance_and_empty_claim_list():
    with pytest.raises(ValueError):
        fuse_claims([])
    with pytest.raises(ValueError):
        fuse_claims([ModelClaim(value=1.0, variance=0.0, model_id="bad", version="v1", content_hash="f" * 64)])


class InvalidScalarsRejectedTest:
    """Regression tests for MXR-080-0283: previously only `variance <= 0` was rejected, so NaN
    (`NaN <= 0` is False in Python!), infinities, non-positive/non-finite reliability, and
    negative/non-finite prior_prec all silently passed into the precision arithmetic -- sometimes
    producing a `FusedBelief` with `mean=nan` or literally `variance < 0`, sometimes crashing with a
    bare `ZeroDivisionError` instead of a clear validation error. Every case here must now raise
    `ValueError` before any arithmetic happens, and must never construct a `FusedBelief`."""

    def _one_claim(self, **overrides):
        defaults = dict(value=5.0, variance=1.0, model_id="A", version="v1", content_hash="a" * 64)
        defaults.update(overrides)
        return ModelClaim(**defaults)

    def test_nan_value_rejected(self):
        with pytest.raises(ValueError):
            fuse_claims([self._one_claim(value=float("nan"))])

    def test_infinite_value_rejected(self):
        with pytest.raises(ValueError):
            fuse_claims([self._one_claim(value=float("inf"))])

    def test_infinite_variance_rejected_not_zerodivisionerror(self):
        # infinite variance -> precision 0 -> if it were the only claim, total_prec=0 used to raise a
        # bare ZeroDivisionError instead of a clean, catchable validation error.
        with pytest.raises(ValueError):
            fuse_claims([self._one_claim(variance=float("inf"))])

    def test_nan_variance_rejected(self):
        # `NaN <= 0` is False, so the old `if c.variance <= 0` check silently let this through.
        with pytest.raises(ValueError):
            fuse_claims([self._one_claim(variance=float("nan"))])

    def test_negative_reliability_rejected(self):
        # Negative reliability flips the sign of that claim's precision; mixed with a positive-precision
        # claim this can drive TOTAL precision negative, so `variance = 1/prec_fused` comes out negative
        # while still being packaged as a `FusedBelief`.
        with pytest.raises(ValueError):
            fuse_claims(
                [
                    self._one_claim(model_id="A", content_hash="a" * 64, reliability=1.0),
                    self._one_claim(value=100.0, model_id="B", content_hash="b" * 64, reliability=-2.0),
                ]
            )

    def test_zero_reliability_rejected(self):
        with pytest.raises(ValueError):
            fuse_claims([self._one_claim(reliability=0.0)])

    def test_non_finite_reliability_rejected(self):
        with pytest.raises(ValueError):
            fuse_claims([self._one_claim(reliability=float("inf"))])
        with pytest.raises(ValueError):
            fuse_claims([self._one_claim(reliability=float("nan"))])

    def test_negative_prior_prec_rejected(self):
        with pytest.raises(ValueError):
            fuse_claims([self._one_claim()], prior_prec=-0.5)

    def test_non_finite_prior_prec_rejected(self):
        with pytest.raises(ValueError):
            fuse_claims([self._one_claim()], prior_prec=float("nan"))
        with pytest.raises(ValueError):
            fuse_claims([self._one_claim()], prior_prec=float("inf"))

    def test_no_invalid_input_ever_yields_a_fused_belief(self):
        """Every rejected case above must raise BEFORE constructing a FusedBelief -- never return one
        carrying a NaN or negative variance."""
        bad_calls = [
            (lambda: fuse_claims([self._one_claim(value=float("nan"))])),
            (lambda: fuse_claims([self._one_claim(variance=float("inf"))])),
            (lambda: fuse_claims([self._one_claim(variance=float("nan"))])),
            (lambda: fuse_claims([self._one_claim(reliability=-1.0)])),
            (lambda: fuse_claims([self._one_claim()], prior_prec=-5.0)),
        ]
        for call in bad_calls:
            with pytest.raises(ValueError):
                result = call()
                assert not isinstance(result, FusedBelief)  # should never get this far


def test_weights_do_not_collapse_when_claims_share_a_model_id():
    """Regression test: `weights` used to be built as `{c.model_id: p / total_prec for p, c in
    zip(...)}`, a dict comprehension keyed by bare model_id -- so when two claims shared a model_id
    (repeated ensemble members of the same model is the realistic case; see
    test_climate_ensemble.py), only the LAST one's weight survived and `weights` summed to well
    under 1.0, contradicting FusedBelief's own "sums to 1" docstring promise.

    variance=1.0 has a true precision-share of 0.9; variance=9.0 has 0.1 (both reliability=1.0, so
    precisions are 1.0 and 1/9, total 10/9)."""
    fused = fuse_claims(
        [
            ModelClaim(value=10.0, variance=1.0, model_id="X", version="v1", content_hash="a" * 64),
            ModelClaim(value=20.0, variance=9.0, model_id="X", version="v2", content_hash="b" * 64),
        ]
    )
    assert len(fused.weights) == 2  # one entry per CLAIM, not collapsed to one per model_id
    assert sum(fused.weights.values()) == pytest.approx(1.0, abs=1e-9)
    # first claim with a given id keeps the bare model_id; the next is disambiguated "#1"
    assert fused.weights["X"] == pytest.approx(0.9, abs=1e-9)
    assert fused.weights["X#1"] == pytest.approx(0.1, abs=1e-9)
    assert fused.mean == pytest.approx(11.0, abs=1e-9)  # unaffected -- mean/variance were already correct

    by_key = {entry["weight_key"]: entry for entry in fused.provenance["claims"]}
    assert by_key["X"]["version"] == "v1"
    assert by_key["X"]["weight"] == pytest.approx(0.9, abs=1e-9)
    assert by_key["X#1"]["version"] == "v2"
    assert by_key["X#1"]["weight"] == pytest.approx(0.1, abs=1e-9)


def test_weights_key_by_bare_model_id_when_there_is_no_collision():
    """No duplicate model_ids -- keys/values must be byte-for-byte what they were before the fix."""
    fused = fuse_claims(
        [
            ModelClaim(value=1.0, variance=0.2, model_id="modelA", version="v1", content_hash="c" * 64),
            ModelClaim(value=2.0, variance=0.2, model_id="modelB", version="v1", content_hash="d" * 64),
        ]
    )
    assert set(fused.weights) == {"modelA", "modelB"}
    assert all(entry["weight_key"] == entry["model_id"] for entry in fused.provenance["claims"])


class AdjudicationIdentityAndInclusionTest:
    """Regression tests for MXR-080-0284 (duplicate model_id corrupts adjudication) and MXR-080-0285
    (one random tail sample / one accepted claim could clear an entire disagreeing set).

    Old mechanism: adjudication stored synthetic posteriors in a `{model_id: draws}` dict (duplicates
    silently overwrote each other) and skipped comparing any two claims that shared a model_id --
    "distinct ensemble realizations" of the same model were therefore NEVER actually compared against
    each other. Acceptance was "did >= 1 of 200 random draws land in a 1-sigma interval", a seeded,
    non-calibrated coin flip; a SINGLE claim clearing that bar set `abstained=False` for the whole set,
    and mean/variance still silently blended in every OTHER claim regardless of whether it was ever
    actually vetted.

    New mechanism: deterministic pairwise significance test (same statistic/bar as the disagreement
    flag itself), connected components identify a corroborated core vs. an outlier, and each claim's
    accept/reject/unresolved status independently gates whether it contributes to mean/variance/weights.
    """

    def test_two_claims_sharing_a_model_id_are_compared_against_each_other_not_skipped(self):
        """P1 and P2 are two DISTINCT realizations of the same underlying model "M" (a real CMIP
        ensemble routinely submits several realizations of one model) that wildly disagree (707 sigma
        apart); P3 is a different model that closely agrees with P1. Under the old model_id-keyed
        adjudication, P1 was NEVER compared against P2 (both "M") -- it only ever got compared against
        P3, trivially cleared, and that ONE clearance both (a) declared the whole set NOT abstained and
        (b) left P2 blended into the fused mean despite never having survived any real comparison.
        """
        p1 = ModelClaim(value=0.0, variance=0.01, model_id="M", version="realization-1", content_hash="1" * 64)
        p2 = ModelClaim(value=100.0, variance=0.01, model_id="M", version="realization-2", content_hash="2" * 64)
        p3 = ModelClaim(value=0.05, variance=0.01, model_id="N", version="v1", content_hash="3" * 64)
        fused = fuse_claims([p1, p2, p3])

        assert fused.disagreement is True
        assert fused.abstained is False  # P1+P3 corroborate each other -- a real resolution exists

        by_key = {e["weight_key"]: e for e in fused.provenance["claims"]}
        assert by_key["M"]["adjudication_status"] == "accepted"  # P1: corroborated by P3
        assert by_key["N"]["adjudication_status"] == "accepted"  # P3: corroborates P1
        assert by_key["M#1"]["adjudication_status"] == "rejected"  # P2: the 707-sigma outlier

        # P2 (the outlier sharing P1's model_id) must be EXCLUDED from the fused mean, not blended in.
        assert fused.weights["M#1"] == pytest.approx(0.0, abs=1e-12)
        assert by_key["M#1"]["included_in_fused_mean"] is False
        assert by_key["M"]["included_in_fused_mean"] is True
        assert by_key["N"]["included_in_fused_mean"] is True
        # only P1 (0.0) and P3 (0.05), equal precision -> simple average, nowhere near P2's 100.0
        assert fused.mean == pytest.approx(0.025, abs=1e-9)
        assert fused.weights["M"] == pytest.approx(0.5, abs=1e-9)
        assert fused.weights["N"] == pytest.approx(0.5, abs=1e-9)

    def test_verifier_accepting_only_one_claim_excludes_the_others_from_the_fused_mean(self):
        """Sharpest test of MXR-080-0285's inclusion-gating fix: a verifier that vouches for exactly ONE
        of three wildly disagreeing claims must leave ONLY that claim in the fused mean/weights -- not
        silently drag the other two (which the verifier never vouched for) along too."""

        class _OnlyAcceptsA:
            def verify(self, claim, context):
                class _Verdict:
                    passed = claim["model_id"] == "A"

                return _Verdict()

        fused = fuse_claims(
            [
                ModelClaim(value=10.0, variance=0.01, model_id="A", version="v1", content_hash="a" * 64),
                ModelClaim(value=500.0, variance=0.01, model_id="B", version="v1", content_hash="b" * 64),
                ModelClaim(value=-500.0, variance=0.01, model_id="C", version="v1", content_hash="c" * 64),
            ],
            verifier=_OnlyAcceptsA(),
        )
        assert fused.disagreement is True
        assert fused.abstained is False
        assert fused.mean == pytest.approx(10.0, abs=1e-6)  # ONLY A, not an average dragged toward B/C
        assert fused.weights == {"A": pytest.approx(1.0), "B": pytest.approx(0.0), "C": pytest.approx(0.0)}
        by_key = {e["weight_key"]: e for e in fused.provenance["claims"]}
        assert by_key["A"]["verifier_passed"] is True
        assert by_key["B"]["verifier_passed"] is False
        assert by_key["C"]["verifier_passed"] is False

    def test_two_mutually_disagreeing_claims_with_no_verifier_are_unresolved_and_abstain(self):
        """With exactly two claims and no third arbiter, neither the graph (no edge, both isolated) nor
        a verifier can prefer one over the other -- both are "unresolved", nothing is "accepted", and
        the set abstains. This pins down that "unresolved" (no basis to pick a winner) is distinct from
        "rejected" (a confirmed outlier against a corroborated core)."""
        fused = fuse_claims(
            [
                ModelClaim(value=2.0, variance=0.1, model_id="cmipA", version="v1", content_hash="a" * 64),
                ModelClaim(value=4.0, variance=0.1, model_id="cmipB", version="v1", content_hash="b" * 64),
            ]
        )
        assert fused.abstained is True
        statuses = {e["weight_key"]: e["adjudication_status"] for e in fused.provenance["claims"]}
        assert statuses == {"cmipA": "unresolved", "cmipB": "unresolved"}
        assert "rejected" not in statuses.values()

    def test_adjudication_is_deterministic_no_seed_dependence(self):
        """The old mechanism accepted a genuinely-disagreeing 3.04-sigma pair in a seed-dependent,
        non-trivial fraction of calls (empirically ~16% over 400 seeds) because acceptance hinged on
        whether >= 1 of 200 random draws landed inside a narrow interval. The new mechanism has no RNG
        anywhere in the adjudication path, so repeated calls on identical input are byte-for-byte
        identical -- this pair (3.04 sigma, just past the disagreement flag) must ALWAYS abstain."""
        claims = [
            ModelClaim(value=0.0, variance=1.0, model_id="A", version="v1", content_hash="a" * 64),
            ModelClaim(value=4.3, variance=1.0, model_id="B", version="v1", content_hash="b" * 64),
        ]
        outcomes = {(fuse_claims(claims).abstained, round(fuse_claims(claims).mean, 12)) for _ in range(25)}
        assert outcomes == {(True, round(2.15, 12))}

    def test_provenance_reports_calibrated_z_and_p_value_per_claim(self):
        """Genuine, quantified uncertainty (not a silent binary pass/fail from one MC draw set): every
        claim's provenance entry carries its nearest-other-claim standardized distance and the exact
        two-sided p-value implied by it under the null of a shared true value."""
        fused = fuse_claims(
            [
                ModelClaim(value=2.0, variance=0.1, model_id="cmipA", version="v1", content_hash="a" * 64),
                ModelClaim(value=4.0, variance=0.1, model_id="cmipB", version="v1", content_hash="b" * 64),
            ]
        )
        z = abs(2.0 - 4.0) / math.sqrt(0.1 + 0.1)
        for entry in fused.provenance["claims"]:
            assert entry["min_pairwise_z"] == pytest.approx(z, abs=1e-6)
            assert 0.0 < entry["min_pairwise_p_value"] < 0.01  # z ~ 4.47 -- a very small two-sided p-value
            assert entry["verifier_passed"] is None  # no verifier was supplied

    def test_no_disagreement_means_no_adjudication_fields_are_populated(self):
        """When claims agree, adjudication never runs at all -- provenance should not claim a status."""
        fused = fuse_claims(
            [
                ModelClaim(value=2.0, variance=0.1, model_id="cmipA", version="v1", content_hash="a" * 64),
                ModelClaim(value=2.1, variance=0.1, model_id="cmipB", version="v1", content_hash="b" * 64),
            ]
        )
        assert fused.disagreement is False
        for entry in fused.provenance["claims"]:
            assert "adjudication_status" not in entry
            assert entry["included_in_fused_mean"] is True
