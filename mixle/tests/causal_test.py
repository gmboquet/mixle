"""do(): graph-surgery interventions on the heterogeneous Bayesian network."""

import numpy as np

from mixle.inference import CausalIdentification, average_causal_effect, do
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


def _identification(*, counterfactuals=False):
    return CausalIdentification.domain_asserted(
        "test fixture DAG was specified independently of the observations",
        structural_counterfactuals=counterfactuals,
    )


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
    ace = average_causal_effect(
        net,
        treatment=0,
        a=1.0,
        b=0.0,
        outcome=1,
        identification=_identification(),
        n=6000,
        seed=3,
    )
    assert abs(ace - 2.0) < 0.06  # the structural coefficient


# --- interventions dict key validation: a mistyped key must be REJECTED, never silently dropped -------
# InterventionalNetwork.sample() looks fields up as `i in self.interventions` with a real int `i` taken
# from net.order. A string key like "0" is never equal (nor hash-equal) to the int 0, so if the
# constructor accepted it uncritically, the "intervention" would just never be consulted -- the run
# would look exactly like an unconditioned baseline instead of failing loudly.
#
# These target `bn_do` (mixle.inference.causal.do / InterventionalNetwork) directly rather than the
# package-level `do`: the latter is M0's generic condition()/do() engine (mixle.inference.condition),
# which does its own key normalization ahead of dispatch and so wouldn't exercise causal.py's own
# validation at all -- see test_package_level_do_reduces_to_bn_do_for_a_bayesian_network above for how
# the two relate.


def test_do_rejects_a_mistyped_string_key_instead_of_silently_dropping_it():
    import pytest

    from mixle.inference import bn_do

    net = _chain()
    with pytest.raises(ValueError, match="not a valid node id"):
        bn_do(net, {"0": 2.0})  # "0" LOOKS like the valid int node id 0, but is not one


def test_do_rejects_an_out_of_range_int_field():
    import pytest

    from mixle.inference import bn_do

    net = _chain()
    with pytest.raises(ValueError, match="not a valid node id"):
        bn_do(net, {99: 1.0})  # correctly typed, but the network only has fields 0 and 1


def test_do_with_a_valid_int_key_applies_exactly_the_intended_intervention():
    """Regression guard: a correctly-typed int key is stored verbatim and still clamps the field --
    unchanged behavior from before the stricter key validation was added."""
    from mixle.inference import bn_do

    net = _chain()
    world = bn_do(net, {0: 2.0})
    assert world.interventions == {0: 2.0}
    xs = {row[0] for row in world.sample(50, seed=9)}
    assert xs == {2.0}  # X is exactly clamped
    assert abs(world.expectation(1, n=6000, seed=0) - 4.0) < 0.05  # E[Y | do(X=2)] = 2*2


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
    cf = counterfactual(net, obs, {1: obs[1] + 1.0}, identification=_identification(counterfactuals=True))
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
    cf = counterfactual(net, obs, {2: 99.0}, identification=_identification(counterfactuals=True))
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
        counterfactual(
            net,
            rows[0],
            {0: rows[0][0] + 3.0},
            identification=_identification(counterfactuals=True),
        )
