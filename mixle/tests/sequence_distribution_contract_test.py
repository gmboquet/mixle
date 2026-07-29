"""Regression contracts for sequence scoring, length laws, and effective counts."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mixle.stats import (
    BernoulliDistribution,
    CategoricalDistribution,
    GaussianEstimator,
    MixtureDistribution,
    NonGenerativeSequenceError,
    PointMassDistribution,
    PoissonDistribution,
    PoissonEstimator,
    SequenceDistribution,
    SequenceEstimator,
    SequenceSampler,
    SequenceStatistics,
)
from mixle.stats.compute.pdist import DensitySemantics


class _RecordingGaussianEstimator(GaussianEstimator):
    def __init__(self) -> None:
        super().__init__()
        self.nobs_seen = []

    def estimate(self, nobs, suff_stat):
        self.nobs_seen.append(nobs)
        return super().estimate(nobs, suff_stat)


class _RecordingPoissonEstimator(PoissonEstimator):
    def __init__(self) -> None:
        super().__init__()
        self.nobs_seen = []

    def estimate(self, nobs, suff_stat):
        self.nobs_seen.append(nobs)
        return super().estimate(nobs, suff_stat)


def test_length_normalized_score_has_one_formula_and_is_not_generative() -> None:
    model = SequenceDistribution(
        BernoulliDistribution(0.5),
        len_dist=CategoricalDistribution({2: 0.5, 3: 0.5}),
        len_normalized=True,
    )
    value = [False, True]
    expected = math.sqrt(0.5 * 0.5) * 0.5
    assert model.density(value) == pytest.approx(expected)
    assert model.density(value) == pytest.approx(math.exp(model.log_density(value)))
    assert model.density_semantics() is DensitySemantics.LIKELIHOOD_FACTOR
    with pytest.raises(NonGenerativeSequenceError):
        model.sampler(3)
    with pytest.raises(TypeError, match="likelihood factors"):
        MixtureDistribution((model, model), (0.5, 0.5))


def test_sequence_without_length_law_is_an_explicit_non_generative_factor() -> None:
    model = SequenceDistribution(BernoulliDistribution(0.5))
    assert model.density_semantics() is DensitySemantics.LIKELIHOOD_FACTOR
    assert model.log_density([False, True]) == pytest.approx(2.0 * math.log(0.5))
    with pytest.raises(NonGenerativeSequenceError):
        model.sampler()


def test_length_support_is_proved_at_construction() -> None:
    with pytest.raises(ValueError, match="exact finite non-negative integer"):
        SequenceDistribution(BernoulliDistribution(0.5), PointMassDistribution(1.9))
    with pytest.raises(ValueError, match="exact finite non-negative integer"):
        SequenceDistribution(BernoulliDistribution(0.5), PointMassDistribution(-1))
    with pytest.raises(ValueError, match="exact finite non-negative integer"):
        SequenceDistribution(
            BernoulliDistribution(0.5),
            CategoricalDistribution({0: 0.5, 1.5: 0.5}),
        )
    valid = SequenceDistribution(
        BernoulliDistribution(0.5),
        CategoricalDistribution({0.0: 0.5, 2.0: 0.5, 1.5: 0.0}),
    )
    assert valid.density_semantics() is DensitySemantics.EXACT
    mixture_length = MixtureDistribution(
        (PoissonDistribution(1.0), PoissonDistribution(4.0)),
        (0.25, 0.75),
    )
    assert (
        SequenceDistribution(
            BernoulliDistribution(0.5),
            mixture_length,
        ).density_semantics()
        is DensitySemantics.EXACT
    )


def test_sampler_rejects_invalid_lengths_and_sample_sizes_before_allocation() -> None:
    for value in (1.9, -1, float("nan"), True, "2", 2 + 1j):
        sampler = SequenceSampler(BernoulliDistribution(0.5), PointMassDistribution(value), seed=4)
        with pytest.raises((TypeError, ValueError), match="sequence length"):
            sampler.sample()
        with pytest.raises((TypeError, ValueError), match="sequence length"):
            sampler.sample(size=2)

    sampler = SequenceSampler(BernoulliDistribution(0.5), PointMassDistribution(2), seed=4)
    for size in (True, 1.5):
        with pytest.raises(TypeError):
            sampler.sample(size=size)
    with pytest.raises(ValueError):
        sampler.sample(size=-1)


def test_sequence_encoder_reports_and_validates_outer_row_geometry() -> None:
    encoder = SequenceDistribution(
        BernoulliDistribution(0.5),
        CategoricalDistribution({0: 0.5, 2: 0.5}),
    ).dist_to_encoder()
    encoded = encoder.seq_encode(([False, True], [], [True, False]))
    assert encoder.row_count(encoded) == 3
    malformed = (
        encoded[0],
        np.asarray([0.25, 0.0, 0.5]),
        encoded[2],
        encoded[3],
        encoded[4],
    )
    with pytest.raises(ValueError, match="inverse lengths"):
        encoder.row_count(malformed)


@pytest.mark.parametrize(
    ("len_normalized", "expected_element_nobs"),
    ((False, 16.0), (True, 6.0)),
)
def test_sequence_statistics_carry_child_effective_counts(
    len_normalized: bool,
    expected_element_nobs: float,
) -> None:
    entry = _RecordingGaussianEstimator()
    length = _RecordingPoissonEstimator()
    estimator = SequenceEstimator(
        entry,
        len_estimator=length,
        len_normalized=len_normalized,
    )
    data = ([1.0, 2.0], [], [3.0, 4.0, 5.0])
    weights = np.asarray([2.0, 3.0, 4.0])
    accumulator = estimator.accumulator_factory().make()
    for value, weight in zip(data, weights):
        accumulator.update(value, weight, None)

    statistics = accumulator.value()
    assert isinstance(statistics, SequenceStatistics)
    assert statistics.schema_version == 1
    assert statistics.element_nobs == pytest.approx(expected_element_nobs)
    assert statistics.length_nobs == pytest.approx(9.0)

    vectorized = estimator.accumulator_factory().make()
    encoded = vectorized.acc_to_encoder().seq_encode(data)
    vectorized.seq_update(encoded, weights, None)
    vectorized_statistics = vectorized.value()
    assert vectorized_statistics.element_nobs == pytest.approx(expected_element_nobs)
    assert vectorized_statistics.length_nobs == pytest.approx(9.0)
    assert vectorized_statistics.elements == pytest.approx(statistics.elements)
    assert vectorized_statistics.lengths == pytest.approx(statistics.lengths)

    estimator.estimate(999.0, statistics)
    assert entry.nobs_seen == [pytest.approx(expected_element_nobs)]
    assert length.nobs_seen == [pytest.approx(9.0)]


def test_sequence_statistics_reject_legacy_or_invalid_count_envelopes() -> None:
    estimator = SequenceEstimator(
        GaussianEstimator(),
        len_estimator=PoissonEstimator(),
    )
    accumulator = estimator.accumulator_factory().make()
    accumulator.update([1.0], 1.0, None)
    statistics = accumulator.value()

    malformed = (
        (statistics.elements, statistics.lengths),
        SequenceStatistics(
            2,
            statistics.element_nobs,
            statistics.length_nobs,
            statistics.elements,
            statistics.lengths,
        ),
        SequenceStatistics(
            1,
            -1.0,
            statistics.length_nobs,
            statistics.elements,
            statistics.lengths,
        ),
        SequenceStatistics(
            1,
            statistics.element_nobs,
            np.nan,
            statistics.elements,
            statistics.lengths,
        ),
    )
    for value in malformed:
        with pytest.raises(ValueError):
            estimator.estimate(None, value)
        with pytest.raises(ValueError):
            accumulator.combine(value)
        with pytest.raises(ValueError):
            accumulator.from_value(value)
