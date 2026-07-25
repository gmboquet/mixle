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
