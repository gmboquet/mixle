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
