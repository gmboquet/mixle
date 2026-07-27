"""Regression contracts for fixed-arity composite distributions."""

from __future__ import annotations

import numpy as np
import pytest

from mixle.stats.combinator.composite import (
    CompositeAccumulator,
    CompositeAccumulatorFactory,
    CompositeDataEncoder,
    CompositeDistribution,
    CompositeEstimator,
)
from mixle.stats.compute.pdist import DataSequenceEncoder
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution


class _ArrayEncoder(DataSequenceEncoder):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ArrayEncoder)

    def seq_encode(self, x):
        return np.asarray(x)


class _RngAccumulator:
    def __init__(self) -> None:
        self.draws: list[int] = []
        self.restores = 0

    def initialize(self, x, weight, rng) -> None:
        self.draws.append(int(rng.randint(2**31)))

    def seq_initialize(self, x, weights, rng) -> None:
        for _ in weights:
            self.draws.append(int(rng.randint(2**31)))

    def update(self, x, weight, estimate) -> None:
        return None

    def seq_update(self, x, weights, estimate) -> None:
        return None

    def combine(self, value):
        return self

    def value(self):
        return tuple(self.draws)

    def from_value(self, value):
        self.restores += 1
        self.draws = list(value)
        return self

    def acc_to_encoder(self):
        return _ArrayEncoder()


class _Factory:
    def make(self):
        return _RngAccumulator()


class _Estimator:
    def accumulator_factory(self):
        return _Factory()

    def estimate(self, nobs, suff_stat):
        return GaussianDistribution(0.0, 1.0)


def _distribution() -> CompositeDistribution:
    return CompositeDistribution(
        (GaussianDistribution(0.0, 1.0), GaussianDistribution(1.0, 2.0))
    )


def test_empty_composite_structures_are_rejected_consistently() -> None:
    constructors = (
        lambda: CompositeDistribution(()),
        lambda: CompositeDataEncoder(()),
        lambda: CompositeAccumulator(()),
        lambda: CompositeAccumulatorFactory(()),
        lambda: CompositeEstimator(()),
    )
    for constructor in constructors:
        with pytest.raises(ValueError, match="at least one component"):
            constructor()


def test_distribution_copies_children_to_an_immutable_tuple() -> None:
    children = [GaussianDistribution(0.0, 1.0)]
    composite = CompositeDistribution(children)
    children.append(GaussianDistribution(1.0, 1.0))
    assert isinstance(composite.dists, tuple)
    assert composite.count == 1
    assert len(composite.dists) == 1


def test_marginal_preserves_requested_order_and_rejects_ambiguous_indices() -> None:
    composite = _distribution()
    reversed_model = composite.marginal([1, 0])
    assert reversed_model.dists == (composite.dists[1], composite.dists[0])
    for indices in ([], [0, 0], [-1], [2]):
        with pytest.raises(ValueError):
            composite.marginal(indices)
    with pytest.raises(TypeError):
        composite.marginal([True])


def test_condition_validates_indices_values_and_nonempty_remainder() -> None:
    composite = _distribution()
    assert composite.condition({1: 2.0}).dists == (composite.dists[0],)
    with pytest.raises(ValueError):
        composite.condition({2: 0.0})
    with pytest.raises(ValueError):
        composite.condition({0: float("nan")})
    with pytest.raises(ValueError, match="at least one unobserved"):
        composite.condition({0: 0.0, 1: 1.0})


def test_scoring_requires_exact_encoded_arity_and_common_row_count() -> None:
    composite = _distribution()
    encoded = composite.dist_to_encoder().seq_encode([(0.0, 1.0), (2.0, 3.0)])
    with pytest.raises(ValueError, match="exactly 2"):
        composite.seq_log_density(encoded[:1])
    with pytest.raises(ValueError, match="inconsistent row counts"):
        composite.seq_log_density((encoded[0], encoded[1][:-1]))


def test_accumulation_validates_observations_encodings_weights_and_estimates() -> None:
    composite = _distribution()
    accumulator = composite.estimator().accumulator_factory().make()
    encoded = composite.dist_to_encoder().seq_encode([(0.0, 1.0), (2.0, 3.0)])
    with pytest.raises(ValueError, match="exactly 2 fields"):
        accumulator.update((0.0,), 1.0, composite)
    with pytest.raises(ValueError, match="exactly 2"):
        accumulator.seq_update(encoded[:1], np.ones(2), composite)
    with pytest.raises(ValueError, match="inconsistent row counts"):
        accumulator.seq_update((encoded[0], encoded[1][:-1]), np.ones(2), composite)
    with pytest.raises(ValueError, match="length 2"):
        accumulator.seq_update(encoded, np.ones(1), composite)
    with pytest.raises(ValueError, match="does not match"):
        accumulator.seq_update(
            encoded,
            np.ones(2),
            CompositeDistribution((GaussianDistribution(0.0, 1.0),)),
        )


def test_restore_and_combine_reject_wrong_arity_before_child_mutation() -> None:
    children = [_RngAccumulator(), _RngAccumulator()]
    accumulator = CompositeAccumulator(children)
    with pytest.raises(ValueError, match="exactly 2"):
        accumulator.from_value(((),))
    assert [child.restores for child in children] == [0, 0]
    with pytest.raises(ValueError, match="exactly 2"):
        accumulator.combine(((),))
    assert [child.restores for child in children] == [0, 0]


def test_scalar_and_batch_initialization_use_the_same_persistent_child_streams() -> None:
    scalar_children = [_RngAccumulator(), _RngAccumulator()]
    scalar = CompositeAccumulator(scalar_children)
    scalar.initialize((1.0, 2.0), 1.0, np.random.RandomState(91))
    scalar.initialize((3.0, 4.0), 1.0, np.random.RandomState(91))

    batch_children = [_RngAccumulator(), _RngAccumulator()]
    batch = CompositeAccumulator(batch_children)
    batch.seq_initialize(
        (np.asarray([1.0, 3.0]), np.asarray([2.0, 4.0])),
        np.ones(2),
        np.random.RandomState(91),
    )
    assert [child.draws for child in scalar_children] == [
        child.draws for child in batch_children
    ]


def test_encoder_and_estimator_copy_their_structural_children() -> None:
    encoders = [_ArrayEncoder()]
    encoder = CompositeDataEncoder(encoders)
    encoders.append(_ArrayEncoder())
    assert len(encoder.encoders) == 1

    estimators = [_Estimator()]
    estimator = CompositeEstimator(estimators)
    estimators.append(_Estimator())
    assert estimator.count == 1
