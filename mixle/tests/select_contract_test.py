"""Regression contracts for conditional and generative select routing."""

from __future__ import annotations

import numpy as np
import pytest

from mixle.stats import (
    BernoulliDistribution,
    CategoricalDistribution,
    GaussianDistribution,
    MixtureDistribution,
    PoissonDistribution,
)
from mixle.stats.combinator.select import (
    NonGenerativeSelectError,
    SelectDataEncoder,
    SelectDistribution,
    SelectEstimator,
    SelectStatistics,
    certify_select_routing,
)
from mixle.stats.compute.pdist import DensitySemantics, EnumerationError


def _letter_route(value):
    return 0 if value == "a" else 1


def _always_zero(value):
    return 0


def _negative_route(value):
    return -1


def _float_route(value):
    return 0.0


def _first_route(value):
    return 0 if value < 0 else 1


def _opposite_route(value):
    return 1 if value < 0 else 0


def test_weightless_select_is_a_non_generative_conditional_likelihood() -> None:
    factor = SelectDistribution(
        (GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)),
        _first_route,
    )
    assert factor.density_semantics() is DensitySemantics.LIKELIHOOD_FACTOR
    with pytest.raises(NonGenerativeSelectError):
        factor.sampler(2)
    with pytest.raises(EnumerationError):
        factor.enumerator()
    with pytest.raises(TypeError, match="likelihood factors"):
        MixtureDistribution((factor, factor), (0.5, 0.5))


def test_finite_support_routing_is_proved_exhaustively() -> None:
    with pytest.raises(ValueError, match="routing contract violated"):
        SelectDistribution(
            (BernoulliDistribution(0.5), BernoulliDistribution(0.5)),
            _always_zero,
            weights=(0.5, 0.5),
        )


def test_infinite_support_requires_a_certificate_and_checks_every_draw() -> None:
    with pytest.raises(ValueError, match="cannot prove routing"):
        SelectDistribution(
            (PoissonDistribution(1.0), PoissonDistribution(2.0)),
            _always_zero,
            weights=(0.5, 0.5),
        )
    certify_select_routing(_always_zero, "deliberately-broken/test")
    broken = SelectDistribution(
        (PoissonDistribution(1.0), PoissonDistribution(2.0)),
        _always_zero,
        weights=(0.0, 1.0),
    )
    with pytest.raises(ValueError, match="violated at sampling"):
        broken.sampler(1).sample()


def test_weights_and_routes_are_strict() -> None:
    children = (CategoricalDistribution({"a": 1.0}), CategoricalDistribution({"b": 1.0}))
    for weights in ((float("nan"), 1.0), (float("inf"), 1.0), (-1.0, 2.0)):
        with pytest.raises(ValueError):
            SelectDistribution(children, _letter_route, weights=weights)
    for router, error in ((_negative_route, ValueError), (_float_route, TypeError)):
        model = SelectDistribution(children, router)
        with pytest.raises(error):
            model.log_density("a")
        with pytest.raises(error):
            model.dist_to_encoder().seq_encode(["a"])
        accumulator = SelectEstimator(
            tuple(child.estimator() for child in children),
            router,
        ).accumulator_factory().make()
        with pytest.raises(error):
            accumulator.update("a", 1.0, None)


def test_router_identity_and_child_layout_are_part_of_encoder_identity() -> None:
    encoders = [
        GaussianDistribution(0.0, 1.0).dist_to_encoder(),
        GaussianDistribution(0.0, 1.0).dist_to_encoder(),
    ]
    first = SelectDataEncoder(encoders, _first_route)
    opposite = SelectDataEncoder(encoders, _opposite_route)
    assert first != opposite
    encoders.pop()
    assert len(first.encoders) == 2

    children = [GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)]
    model = SelectDistribution(children, _first_route)
    children.pop()
    assert isinstance(model.dists, tuple)
    assert model.count == 2


def test_select_statistics_are_versioned_exact_and_non_aliasing() -> None:
    estimator = SelectEstimator(
        (GaussianDistribution(-1.0, 1.0).estimator(), GaussianDistribution(1.0, 1.0).estimator()),
        _first_route,
    )
    accumulator = estimator.accumulator_factory().make()
    accumulator.update(-2.0, 1.5, None)
    accumulator.update(2.0, 2.5, None)
    value = accumulator.value()
    assert isinstance(value, SelectStatistics)
    assert value.schema_version == 1
    assert tuple(branch[0] for branch in value.branches) == (1.5, 2.5)

    malformed = (
        SelectStatistics(2, value.branches),
        SelectStatistics(1, value.branches[:1]),
        SelectStatistics(1, ((-1.0, value.branches[0][1]), value.branches[1])),
        tuple(value),
    )
    for item in malformed:
        with pytest.raises(ValueError):
            estimator.estimate(None, item)
        with pytest.raises(ValueError):
            accumulator.from_value(item)
        with pytest.raises(ValueError):
            accumulator.combine(item)


def test_zero_total_weight_cannot_invent_a_dispatch_mixture() -> None:
    estimator = SelectEstimator(
        (GaussianDistribution(-1.0, 1.0).estimator(), GaussianDistribution(1.0, 1.0).estimator()),
        _first_route,
        estimate_weights=True,
    )
    empty = estimator.accumulator_factory().make().value()
    with pytest.raises(ValueError, match="zero total"):
        estimator.estimate(None, empty)


def test_grouped_encoding_validates_routes_positions_and_child_rows() -> None:
    model = SelectDistribution(
        (GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)),
        _first_route,
    )
    encoded = model.dist_to_encoder().seq_encode([-2.0, 2.0])
    assert model.dist_to_encoder().row_count(encoded) == 2
    bad_positions = ((np.array([0]), np.array([0])), encoded[1], encoded[2])
    with pytest.raises(ValueError, match="partition"):
        model.seq_log_density(bad_positions)
