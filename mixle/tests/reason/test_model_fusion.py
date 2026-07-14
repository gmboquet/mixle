"""M2 DoD: numeric cross-model belief fusion.

Two agreeing models should fuse to a tighter belief than either alone, with both model identities
showing up in the attribution weights; two models that disagree by more than ``disagree_sigma``
should set ``abstain`` rather than silently presenting the fused mean as settled.
"""

from __future__ import annotations

import pytest

from mixle.reason import Latent
from mixle.reason.core import LinearGaussianEvidence, reason
from mixle.reason.model_fusion import ModelClaim, ModelFusionResult, fuse_models


def _claim(model_id: str, version: str, *, y: float, r: float, reliability: float = 1.0) -> ModelClaim:
    return ModelClaim(
        evidence=LinearGaussianEvidence(H=[[1.0]], y=[y], R=[[r]], name="ignored"),
        model_id=model_id,
        version=version,
        reliability=reliability,
    )


class FuseModelsAgreementTest:
    def test_two_agreeing_models_fuse_tighter_than_either_alone(self):
        prior = Latent.vector(1, var=10.0)
        claim_a = _claim("A", "1", y=5.0, r=1.0)
        claim_b = _claim("B", "1", y=5.2, r=1.0)

        res = fuse_models(prior, [claim_a, claim_b])

        assert isinstance(res, ModelFusionResult)
        assert set(res.weights) == {"A@1", "B@1"}

        alone_a = reason(prior, [LinearGaussianEvidence(H=[[1.0]], y=[5.0], R=[[1.0]], name="A@1")])
        alone_b = reason(prior, [LinearGaussianEvidence(H=[[1.0]], y=[5.2], R=[[1.0]], name="B@1")])
        assert res.answer.cov()[0, 0] < min(alone_a.cov()[0, 0], alone_b.cov()[0, 0])
        assert res.abstain is False

    def test_weights_sum_to_total_information_gain(self):
        prior = Latent.vector(1, var=10.0)
        res = fuse_models(prior, [_claim("A", "1", y=5.0, r=1.0), _claim("B", "1", y=5.1, r=2.0)])
        assert sum(res.weights.values()) == pytest.approx(res.answer.information_gain(), rel=1e-9)

    def test_reliability_scales_effective_noise(self):
        prior = Latent.vector(1, var=10.0)
        # A flaky model (low reliability) should have its noise inflated and so contribute less
        # than an equally-noisy but fully-trusted model.
        trusted = _claim("A", "1", y=5.0, r=1.0, reliability=1.0)
        flaky = _claim("B", "1", y=-5.0, r=1.0, reliability=0.01)
        res = fuse_models(prior, [trusted, flaky])
        assert res.weights["A@1"] > res.weights["B@1"]
        assert abs(res.answer.mean[0] - 5.0) < 1.0

    def test_single_claim_never_abstains(self):
        prior = Latent.vector(1, var=10.0)
        res = fuse_models(prior, [_claim("A", "1", y=5.0, r=1.0)])
        assert set(res.weights) == {"A@1"}
        assert res.abstain is False
        assert res.disagreement["max_sigma"] == 0.0


class FuseModelsDisagreementTest:
    def test_far_apart_models_set_abstain(self):
        prior = Latent.vector(1, var=100.0)
        claim_a = _claim("A", "1", y=0.0, r=0.01)
        claim_b = _claim("B", "1", y=50.0, r=0.01)

        res = fuse_models(prior, [claim_a, claim_b], disagree_sigma=3.0)

        assert res.abstain is True
        assert res.disagreement["max_sigma"] > 3.0
        assert res.disagreement["flagged_pairs"] == ["A@1|B@1"]

    def test_close_models_do_not_abstain(self):
        prior = Latent.vector(1, var=10.0)
        claim_a = _claim("A", "1", y=5.0, r=1.0)
        claim_b = _claim("B", "1", y=5.1, r=1.0)

        res = fuse_models(prior, [claim_a, claim_b], disagree_sigma=3.0)

        assert res.abstain is False
        assert res.disagreement["flagged_pairs"] == []


class ModelClaimValidationTest:
    def test_reliability_must_be_in_unit_interval(self):
        with pytest.raises(ValueError):
            _claim("A", "1", y=1.0, r=1.0, reliability=0.0)
        with pytest.raises(ValueError):
            _claim("A", "1", y=1.0, r=1.0, reliability=1.5)

    def test_duplicate_model_identity_rejected(self):
        prior = Latent.vector(1, var=10.0)
        with pytest.raises(ValueError):
            fuse_models(prior, [_claim("A", "1", y=1.0, r=1.0), _claim("A", "1", y=2.0, r=1.0)])

    def test_empty_claims_rejected(self):
        prior = Latent.vector(1, var=10.0)
        with pytest.raises(ValueError):
            fuse_models(prior, [])
