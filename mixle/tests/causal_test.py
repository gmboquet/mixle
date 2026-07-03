"""do(): graph-surgery interventions on the heterogeneous Bayesian network."""

import numpy as np

from mixle.inference import average_causal_effect, do
from mixle.inference.bayesian_network import (
    HeterogeneousBayesianNetwork,
    _LinearGaussianFactor,
    _MarginalFactor,
)
from mixle.stats import GaussianDistribution


def _chain():
    """X -> Y with Y = 2X + eps(0.5); X ~ N(0,1). Hand-built so the causal direction is fixed."""
    fx = _MarginalFactor(0, GaussianDistribution(0.0, 1.0))
    fy = _LinearGaussianFactor(1, [0], {}, np.array([2.0, 0.0]), 0.5)
    return HeterogeneousBayesianNetwork([fx, fy])


def test_do_on_the_cause_moves_the_effect_exactly():
    net = _chain()
    world = do(net, {0: 2.0})
    assert abs(world.expectation(1, n=6000, seed=0) - 4.0) < 0.05  # E[Y | do(X=2)] = 2*2


def test_do_on_the_effect_leaves_the_cause_at_its_marginal():
    net = _chain()
    # THE do-vs-conditioning signature: setting Y tells us nothing about X under intervention,
    # whereas OBSERVING Y=6 would have pulled E[X | Y=6] far above 0.
    world = do(net, {1: 6.0})
    assert abs(world.expectation(0, n=6000, seed=1) - 0.0) < 0.05
    # and Y is exactly clamped
    ys = {row[1] for row in world.sample(50, seed=2)}
    assert ys == {6.0}


def test_average_causal_effect_matches_the_structural_slope():
    net = _chain()
    ace = average_causal_effect(net, treatment=0, a=1.0, b=0.0, outcome=1, n=6000, seed=3)
    assert abs(ace - 2.0) < 0.06  # the structural coefficient
