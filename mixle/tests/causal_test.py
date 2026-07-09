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


def test_package_level_do_reduces_to_bn_do_for_a_bayesian_network():
    """mixle.inference.do is now M0's generic condition()/do() engine, not the BN-only causal.do
    directly -- but for a HeterogeneousBayesianNetwork with flat (non-nested) evidence it dispatches
    straight through to bn_do (mixle.inference.causal.do, still reachable under that name), so every
    existing BN caller keeps working unmodified. Confirm both paths agree exactly, not just similarly."""
    from mixle.inference import bn_do

    net = _chain()
    generic_world = do(net, {0: 2.0})
    bn_world = bn_do(net, {0: 2.0})
    assert type(generic_world) is type(bn_world)
    assert abs(generic_world.expectation(1, n=6000, seed=7) - bn_world.expectation(1, n=6000, seed=7)) < 1e-10


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


# --- counterfactuals: abduction-action-prediction with the honest discrete boundary ---------------------
# The DAG is constructed EXPLICITLY (kind -> x0 -> x1): counterfactual() answers relative to the given
# graph, and purely observational learning cannot orient Markov-equivalent edges — that caveat lives in
# the docstring, and the unit under test here is the abduction, not structure discovery.


def _cf_network(seed=0):
    from mixle.inference.bayesian_network import (
        HeterogeneousBayesianNetwork,
        _columns,
        _LinearGaussianFactor,
        _MarginalFactor,
    )
    from mixle.inference.estimation import optimize
    from mixle.stats import CategoricalEstimator

    rng = np.random.RandomState(seed)
    kinds = ["a", "b"]
    rows = []
    for _ in range(1500):
        k = kinds[rng.randint(0, 2)]
        x0 = float(rng.normal(0.0, 1.0))
        x1 = float(2.0 * x0 + (1.0 if k == "b" else 0.0) + rng.normal(0.0, 0.5))
        rows.append((k, x0, x1))
    cols = _columns(rows)
    kind_dist = optimize(cols[0], CategoricalEstimator(), max_its=5, out=None)
    from mixle.stats import GaussianEstimator

    x0_dist = optimize(cols[1], GaussianEstimator(), max_its=5, out=None)
    f2 = _LinearGaussianFactor.fit(2, [0, 1], cols, {0: sorted(set(cols[0]))})
    net = HeterogeneousBayesianNetwork([_MarginalFactor(0, kind_dist), _MarginalFactor(1, x0_dist), f2])
    return net, rows


def test_counterfactual_replays_the_abducted_residual_exactly():
    from mixle.inference.causal import counterfactual

    net, rows = _cf_network()
    f = {g.child: g for g in net.factors}[2]

    obs = rows[10]
    cf = counterfactual(net, obs, {1: obs[1] + 1.0})
    mu_obs = float(f._row([obs[p] for p in f.parents]) @ f.coef)
    want = list(obs)
    want[1] = obs[1] + 1.0
    mu_cf = float(f._row([want[p] for p in f.parents]) @ f.coef)
    assert abs((cf[2] - obs[2]) - (mu_cf - mu_obs)) < 1e-10  # the SAME noise, replayed
    assert cf[0] == obs[0]  # untouched discrete field keeps its observed value
    assert abs((cf[2] - obs[2]) - 2.0) < 0.15  # the learned effect matches the true slope


def test_counterfactual_downstream_intervention_leaves_ancestors_alone():
    from mixle.inference.causal import counterfactual

    net, rows = _cf_network()
    obs = rows[3]
    cf = counterfactual(net, obs, {2: 99.0})
    assert cf[2] == 99.0
    assert cf[1] == obs[1]  # in THIS dag x1 is upstream: intervening on the effect leaves the cause
    assert cf[0] == obs[0]


def test_counterfactual_discrete_with_changed_parents_raises_honestly():
    import pytest

    from mixle.inference.bayesian_network import (
        HeterogeneousBayesianNetwork,
        _columns,
        _DiscreteConditionalFactor,
        _MarginalFactor,
    )
    from mixle.inference.causal import counterfactual
    from mixle.inference.estimation import optimize
    from mixle.stats import GaussianEstimator

    rng = np.random.RandomState(1)
    rows = []
    for _ in range(1200):
        x0 = float(rng.normal(0.0, 1.0))
        k = "hi" if x0 + rng.normal(0.0, 0.3) > 0 else "lo"  # discrete CHILD of x0
        rows.append((x0, k))
    from mixle.stats import CategoricalEstimator

    cols = _columns(rows)
    x0_dist = optimize(cols[0], GaussianEstimator(), max_its=5, out=None)
    f1 = _DiscreteConditionalFactor.fit(1, [0], cols, template=CategoricalEstimator(), max_its=5)
    net = HeterogeneousBayesianNetwork([_MarginalFactor(0, x0_dist), f1])
    with pytest.raises(ValueError):
        counterfactual(net, rows[0], {0: rows[0][0] + 3.0})
