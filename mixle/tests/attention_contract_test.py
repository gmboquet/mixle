"""Probability, evidence, weighting, sampling, and state contracts for attention leaves."""

import json

import numpy as np
import pytest

from mixle.stats.latent.chained_attention import (
    ChainedAttentionDistribution,
    ChainedAttentionEstimator,
)
from mixle.stats.latent.responsibility_attention import (
    ResponsibilityAttentionDistribution,
    ResponsibilityAttentionEstimator,
)
from mixle.stats.latent.variational_embedding_attention import (
    VariationalEmbeddingAttentionDistribution,
    VariationalEmbeddingAttentionEstimator,
)
from mixle.stats.latent.variational_multihop_attention import (
    VariationalMultiHopAttentionDistribution,
    VariationalMultiHopAttentionEstimator,
)
from mixle.utils.vector import ImpossibleEvidenceError


def _embedding_distribution(log_var=-1.0):
    return VariationalEmbeddingAttentionDistribution(
        mean=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        log_var=np.full((3, 2), log_var),
        emission=np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]),
        position_prior=np.array([0.25, 0.75]),
        sigma2=0.5,
    )


def _multihop_distribution(log_var=-1.0):
    return VariationalMultiHopAttentionDistribution(
        mean=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        log_var=np.full((3, 2), log_var),
        emission=np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]),
        sigma2=0.5,
    )


@pytest.mark.parametrize(
    "constructor,args",
    [
        (
            ResponsibilityAttentionDistribution,
            (np.zeros((2, 1)), np.array([[0.2, 0.2], [0.5, 0.5]]), np.ones(2) / 2),
        ),
        (
            ChainedAttentionDistribution,
            (np.zeros((2, 2, 2)), np.array([[1.1, -0.1], [0.5, 0.5]])),
        ),
        (
            VariationalEmbeddingAttentionDistribution,
            (
                np.zeros((2, 1)),
                np.zeros((2, 1)),
                np.array([[0.2, 0.2], [0.5, 0.5]]),
                np.ones(2) / 2,
            ),
        ),
        (
            VariationalMultiHopAttentionDistribution,
            (np.zeros((2, 1)), np.zeros((2, 1)), np.array([[0.2, 0.2], [0.5, 0.5]])),
        ),
    ],
)
def test_distributions_reject_non_simplex_emissions(constructor, args):
    with pytest.raises(ValueError):
        constructor(*args)


@pytest.mark.parametrize(
    "distribution",
    [
        ResponsibilityAttentionDistribution(np.zeros((2, 1)), np.ones((2, 2)) / 2, np.ones(2) / 2),
        ChainedAttentionDistribution(np.zeros((2, 2, 2)), np.ones((2, 2)) / 2),
        _embedding_distribution(),
        _multihop_distribution(),
    ],
)
def test_encoders_reject_fractional_negative_and_out_of_range_ids(distribution):
    encoder = distribution.dist_to_encoder()
    if isinstance(distribution, ResponsibilityAttentionDistribution):
        invalid = [
            ([0.5, 1], [0.0], 0),
            ([0, -1], [0.0], 0),
            ([0, 2], [0.0], 0),
        ]
    elif isinstance(distribution, VariationalEmbeddingAttentionDistribution):
        invalid = [([0.5, 1], 0, 0), ([0, -1], 0, 0), ([0, 3], 0, 0)]
    else:
        invalid = [
            ([0.5, 1], [0, 1], 0, 0),
            ([0, -1], [0, 1], 0, 0),
            ([0, 2 if distribution.num_symbols == 2 else 3], [0, 1], 0, 0),
        ]
    for observation in invalid:
        with pytest.raises((TypeError, ValueError)):
            encoder.seq_encode([observation])


def test_structural_position_zero_is_not_revived():
    distribution = ResponsibilityAttentionDistribution(
        key_means=np.zeros((2, 1)),
        emission=np.array([[0.0, 1.0], [1.0, 0.0]]),
        position_prior=np.array([0.0, 1.0]),
    )
    observation = (np.array([0, 1]), np.array([0.0]), 1)
    assert distribution.log_density(observation) == -np.inf
    estimator = ResponsibilityAttentionEstimator(2, 2, 1, 2)
    accumulator = estimator.accumulator_factory().make()
    before = accumulator.value()
    with pytest.raises(ImpossibleEvidenceError):
        accumulator.update(observation, 1.0, distribution)
    for old, new in zip(before, accumulator.value(), strict=True):
        np.testing.assert_array_equal(old, new)


@pytest.mark.parametrize("family", ["chained", "embedding", "multihop"])
def test_impossible_targets_fail_before_accumulator_mutation(family):
    emission = np.array([[1.0, 0.0], [1.0, 0.0]])
    if family == "chained":
        distribution = ChainedAttentionDistribution(np.zeros((2, 2, 2)), emission)
        estimator = ChainedAttentionEstimator(2, 2, 2)
        observation = (np.array([0, 1]), np.array([0, 1]), 0, 1)
    elif family == "embedding":
        distribution = VariationalEmbeddingAttentionDistribution(
            np.zeros((2, 1)), np.zeros((2, 1)), emission, np.ones(2) / 2
        )
        estimator = VariationalEmbeddingAttentionEstimator(2, 2, 1, 2, mc=1)
        observation = (np.array([0, 1]), 0, 1)
    else:
        distribution = VariationalMultiHopAttentionDistribution(np.zeros((2, 1)), np.zeros((2, 1)), emission)
        estimator = VariationalMultiHopAttentionEstimator(2, 1, 2, mc=1)
        observation = (np.array([0, 1]), np.array([0, 1]), 0, 1)
    assert distribution.log_density(observation) == -np.inf
    accumulator = estimator.accumulator_factory().make()
    before = accumulator.value()
    with pytest.raises(ImpossibleEvidenceError):
        accumulator.update(observation, 1.0, distribution)
    after = accumulator.value()
    compared_before = before if family == "chained" else before[:-1]
    compared_after = after if family == "chained" else after[:-1]
    for old, new in zip(compared_before, compared_after, strict=True):
        np.testing.assert_array_equal(old, new)
    if family != "chained":
        assert after[-1] is None


@pytest.mark.parametrize("family", ["embedding", "multihop"])
def test_variational_gradients_honor_per_observation_weights(family):
    if family == "embedding":
        distribution = _embedding_distribution()
        estimator = VariationalEmbeddingAttentionEstimator(3, 2, 2, 2, mc=2, seed=7)
        data = [
            (np.array([0, 1]), 0, 1),
            (np.array([1, 2]), 2, 0),
        ]
    else:
        distribution = _multihop_distribution()
        estimator = VariationalMultiHopAttentionEstimator(3, 2, 2, mc=2, seed=7)
        data = [
            (np.array([0, 1]), np.array([1, 2]), 0, 1),
            (np.array([1, 2]), np.array([2, 0]), 2, 0),
        ]
    encoder = distribution.dist_to_encoder()
    weighted = estimator.accumulator_factory().make()
    weighted.seq_update(encoder.seq_encode(data), np.array([0.0, 1.0]), distribution)
    single = estimator.accumulator_factory().make()
    single.seq_update(encoder.seq_encode(data[1:]), np.ones(1), distribution)
    for left, right in zip(weighted.value()[:-1], single.value()[:-1], strict=True):
        np.testing.assert_allclose(left, right)


def _assert_observations_equal(left, right):
    assert len(left) == len(right)
    for l_obs, r_obs in zip(left, right, strict=True):
        assert len(l_obs) == len(r_obs)
        for l_value, r_value in zip(l_obs, r_obs, strict=True):
            np.testing.assert_array_equal(l_value, r_value)


@pytest.mark.parametrize("factory", [_embedding_distribution, _multihop_distribution])
def test_default_sampler_matches_plugin_law_and_posterior_draws_are_iid(factory):
    low_variance = factory(-8.0)
    high_variance = factory(1.0)
    _assert_observations_equal(
        low_variance.sampler(11).sample(5),
        high_variance.sampler(11).sample(5),
    )
    batched_sampler = high_variance.posterior_predictive_sampler(19)
    sequential_sampler = high_variance.posterior_predictive_sampler(19)
    _assert_observations_equal(
        batched_sampler.sample(4),
        [sequential_sampler.sample() for _ in range(4)],
    )


@pytest.mark.parametrize("family", ["embedding", "multihop"])
def test_optimizer_state_is_serializable_restartable_and_estimator_independent(family):
    if family == "embedding":
        model = _embedding_distribution()
        estimator = VariationalEmbeddingAttentionEstimator(3, 2, 2, 2, mc=1, seed=23)
        data = [(np.array([0, 1]), 0, 1), (np.array([1, 2]), 2, 0)]
    else:
        model = _multihop_distribution()
        estimator = VariationalMultiHopAttentionEstimator(3, 2, 2, mc=1, seed=23)
        data = [
            (np.array([0, 1]), np.array([1, 2]), 0, 1),
            (np.array([1, 2]), np.array([2, 0]), 2, 0),
        ]
    accumulator = estimator.accumulator_factory().make()
    accumulator.seq_update(model.dist_to_encoder().seq_encode(data), np.ones(2), model)
    value = accumulator.value()
    json.dumps(value[-1])
    restored = estimator.accumulator_factory().make().from_value(value)
    assert restored.value()[-1] == value[-1]

    first = estimator.estimate(2.0, value)
    second = estimator.estimate(2.0, value)
    fresh_estimator = model.estimator()
    restarted = fresh_estimator.estimate(2.0, value)
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.log_var, second.log_var)
    np.testing.assert_array_equal(first.mean, restarted.mean)
    np.testing.assert_array_equal(first.log_var, restarted.log_var)
    assert first.optimizer_state.iteration == model.optimizer_state.iteration + 1
    assert not hasattr(estimator, "_t")
    assert not hasattr(estimator, "mean")
