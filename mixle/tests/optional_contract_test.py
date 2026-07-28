"""Regression contracts for modeled and marginalized optional observations."""

from __future__ import annotations

import numpy as np
import pytest

from mixle.capability import describe
from mixle.stats import (
    BernoulliDistribution,
    BetaDistribution,
    CategoricalDistribution,
    CompositeDistribution,
    GaussianDistribution,
    MixtureDistribution,
)
from mixle.stats.combinator.optional import (
    NonGenerativeOptionalError,
    OptionalDataEncoder,
    OptionalDistribution,
    OptionalEstimator,
)
from mixle.stats.compute.pdist import ContractError, DensitySemantics


class _StaticEstimator:
    def __init__(self) -> None:
        self.prior = None
        self.counts: list[float] = []

    def set_prior(self, prior) -> None:
        self.prior = prior

    def get_prior(self):
        return self.prior

    def estimate(self, nobs, suff_stat):
        self.counts.append(nobs)
        return GaussianDistribution(0.0, 1.0)


def test_discrete_missing_sentinel_must_be_disjoint_from_child_support() -> None:
    with pytest.raises(ValueError, match="collides"):
        OptionalDistribution(BernoulliDistribution(0.5), p=0.2, missing_value=0)
    with pytest.raises(ValueError, match="collides"):
        OptionalDistribution(CategoricalDistribution({"missing": 1.0}), p=0.2, missing_value="missing")


def test_disjoint_modeled_optional_is_normalized() -> None:
    model = OptionalDistribution(BernoulliDistribution(0.5), p=0.2, missing_value="missing")
    total = sum(model.density(value) for value in ("missing", False, True))
    assert total == pytest.approx(1.0)


def test_marginalized_optional_is_explicitly_non_generative() -> None:
    factor = OptionalDistribution(GaussianDistribution(0.0, 1.0), p=None)
    assert factor.density_semantics() is DensitySemantics.LIKELIHOOD_FACTOR
    assert "sample" not in describe(factor).splitlines()[1]
    assert "non-generative likelihood factor" in describe(factor)
    assert factor.log_density(None) == 0.0
    with pytest.raises(NonGenerativeOptionalError):
        factor.sampler(3)

    # A marginalized law IS admissible as a mixture component. Marginalizing over missing values
    # inside a mixture or an HMM is what mixle.stats.missing exists for, and mixle/tests/missing_data_test.py
    # has exercised exactly that since the package was renamed. What makes this factor non-generative is
    # the unspecified missingness rate, not the law it wraps; the mixture does not hide the consequence,
    # because its semantics join reports LIKELIHOOD_FACTOR.
    marginalized_mixture = MixtureDistribution((factor, factor), (0.5, 0.5))
    assert marginalized_mixture.density_semantics() is DensitySemantics.LIKELIHOOD_FACTOR

    # A leaf that intrinsically cannot generate has no such wrapped law and is still refused.
    with pytest.raises(TypeError, match="likelihood factors"):
        MixtureDistribution(
            (CategoricalDistribution({"a": 0.5}, scoring_only=True),) * 2,
            (0.5, 0.5),
        )

    composite = CompositeDistribution((factor, GaussianDistribution(0.0, 1.0)))
    assert composite.density_semantics() is DensitySemantics.LIKELIHOOD_FACTOR


def test_nan_and_array_sentinels_have_total_stable_equivalence() -> None:
    nan_a = OptionalDataEncoder(GaussianDistribution(0.0, 1.0).dist_to_encoder(), float("nan"))
    nan_b = OptionalDataEncoder(GaussianDistribution(0.0, 1.0).dist_to_encoder(), np.float64("nan"))
    assert nan_a == nan_b
    encoded_nan = nan_a.seq_encode([float("nan"), 2.0, np.float64("nan")])
    np.testing.assert_array_equal(encoded_nan[1], np.array([0, 2]))

    sentinel = np.array([1, 2], dtype=np.int64)
    array_encoder = OptionalDataEncoder(GaussianDistribution(0.0, 1.0).dist_to_encoder(), sentinel)
    same_encoder = OptionalDataEncoder(
        GaussianDistribution(0.0, 1.0).dist_to_encoder(),
        np.array([1, 2], dtype=np.int64),
    )
    assert array_encoder == same_encoder
    encoded_array = array_encoder.seq_encode([np.array([1, 2], dtype=np.int64), 3.0])
    np.testing.assert_array_equal(encoded_array[1], np.array([0]))
    assert array_encoder.row_count(encoded_array) == 2


def test_optional_statistics_do_not_alias_external_count_storage() -> None:
    model = OptionalDistribution(GaussianDistribution(0.0, 1.0), p=0.25)
    accumulator = model.estimator().accumulator_factory().make()
    accumulator.update(None, 2.0, model)
    accumulator.update(1.0, 3.0, model)
    value = accumulator.value()
    assert value[0] == (2.0, 3.0)
    assert isinstance(value[0], tuple)

    accumulator.update(None, 1.0, model)
    assert value[0] == (2.0, 3.0)

    supplied_counts = [4.0, 5.0]
    restored = model.estimator().accumulator_factory().make()
    restored.from_value((supplied_counts, value[1]))
    supplied_counts[0] = 99.0
    assert restored.value()[0] == (4.0, 5.0)


@pytest.mark.parametrize(
    ("counts", "error"),
    [
        ((-1.0, 2.0), ValueError),
        ((float("nan"), 2.0), ValueError),
        ((1.0,), ContractError),
    ],
)
def test_optional_statistics_reject_invalid_counts(counts, error) -> None:
    estimator = OptionalEstimator(_StaticEstimator(), est_prob=True)
    with pytest.raises(error):
        estimator.estimate(None, (counts, None))


@pytest.mark.parametrize(
    ("alpha", "beta", "expected"),
    [
        (0.5, 2.0, 0.0),
        (2.0, 0.5, 1.0),
        (0.5, 0.5, 0.5),
        (2.0, 2.0, 0.5),
    ],
)
def test_beta_posterior_point_handles_boundary_and_interior_modes(alpha, beta, expected) -> None:
    child = _StaticEstimator()
    estimator = OptionalEstimator(
        child,
        est_prob=True,
        prior=(BetaDistribution(alpha, beta), None),
    )
    fitted = estimator.estimate(None, ((0.0, 0.0), None))
    assert fitted.p == pytest.approx(expected)
    assert child.counts == [0.0]
