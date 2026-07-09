"""simulate()/Simulator (M2): on-the-fly conditional simulators for special scenarios.

Acceptance receipts, one per test:
  (a) BN fixture, clamp + intervention: do()-then-condition() rollout matches the closed-form
      interventional-conditional Gaussian; the wrong order (condition then do) differs by a stated
      margin.
  (b) plausibility receipt separates in-support from off-support scenarios by a stated margin.
  (c) HMM horizon rollout matches an independently forward-filtered projection past the clamped
      window.
  (d) seed-determinism of both the BN (SIR) and HMM (forward-rollout) paths.
"""

import numpy as np

from mixle.inference.bayesian_network import HeterogeneousBayesianNetwork, _LinearGaussianFactor, _MarginalFactor
from mixle.inference.condition import condition as m0_condition
from mixle.inference.scenario import Scenario, simulate
from mixle.stats import GaussianDistribution, HiddenMarkovModelDistribution
from mixle.stats.combinator.composite import CompositeDistribution
from mixle.stats.compute.posterior import MarkovChainLatentPosterior

# --------------------------------------------------------------------------------------------- #
# (a) BN fixture: Z -> X -> Y, Z -> Y direct.  do(X=x0) THEN condition(Y=y0) vs the wrong order.
# --------------------------------------------------------------------------------------------- #


def _mediator_confounder_bn():
    # Z ~ N(0, 1); X = a*Z + eps_x; Y = c*X + d*Z + eps_y.
    a, c, d, sigma_x, sigma_y = 1.5, 0.8, 1.2, 0.6, 0.5
    f_z = _MarginalFactor(0, GaussianDistribution(0.0, 1.0))
    f_x = _LinearGaussianFactor(1, [0], {}, np.array([a, 0.0]), sigma_x)
    f_y = _LinearGaussianFactor(2, [0, 1], {}, np.array([d, c, 0.0]), sigma_y)
    net = HeterogeneousBayesianNetwork([f_z, f_x, f_y])
    return net, a, c, d, sigma_x, sigma_y


def test_bn_do_then_condition_matches_closed_form_and_order_matters():
    net, a, c, d, sigma_x, sigma_y = _mediator_confounder_bn()
    # x0 chosen large so the do()-intervened mean offset (c*x0) clearly separates the two orders --
    # the wrong order's closed form does not depend on x0 at all (it ignores the intervention).
    x0, y0 = 5.0, 3.0

    scenario = Scenario(interventions={1: x0}, evidence={2: y0})
    sim = simulate(scenario, base=net, seed=0)
    rollout = sim.rollout(60000)
    z_vals = np.array([row[0] for row in rollout], dtype=np.float64)
    got_mean, got_var = float(np.mean(z_vals)), float(np.var(z_vals))

    # Closed form for Z | do(X=x0), Y=y0: after do(X), Y = c*x0 + d*Z + eps_y (the a*Z->X->c edge is
    # severed), so (Z, Y) are jointly Gaussian with Cov(Z,Y) = d, Var(Y) = d^2 + sigma_y^2.
    var_y_do = d**2 + sigma_y**2
    mean_y_do = c * x0
    want_mean = (d / var_y_do) * (y0 - mean_y_do)
    want_var = 1.0 - d**2 / var_y_do

    assert abs(got_mean - want_mean) < 0.05
    assert abs(got_var - want_var) < 0.05
    assert sim.receipt.composition_order == "do-then-condition"

    # The WRONG order: condition on Y in the ORIGINAL (non-intervened) network first, then "do" X.
    # Z | Y=y0 in the original model: Y = (c*a+d)*Z + c*eps_x + eps_y (X marginalized out).
    cov_zy = c * a + d
    var_y = cov_zy**2 + c**2 * sigma_x**2 + sigma_y**2
    wrong_mean = (cov_zy / var_y) * y0
    wrong_var = 1.0 - cov_zy**2 / var_y

    # Independently verify the wrong-order number too, via condition() directly on the original net.
    wrong_post = m0_condition(net, {2: y0}, method="sir", n_particles=60000, seed=1)
    wrong_got_mean = wrong_post.mean(0)
    assert abs(wrong_got_mean - wrong_mean) < 0.1
    assert abs(wrong_var - want_var) > 0.05  # the two orders give a meaningfully different variance too
    assert abs(wrong_mean - want_mean) > 1.0  # and a meaningfully different mean -- order matters


# --------------------------------------------------------------------------------------------- #
# (b) plausibility receipt separates in-support from off-support scenarios
# --------------------------------------------------------------------------------------------- #


def _fitted_gaussian_composite():
    leaves = [GaussianDistribution(0.0, 1.0), GaussianDistribution(5.0, 2.0)]
    return CompositeDistribution(leaves), leaves


def test_plausibility_receipt_separates_in_support_from_off_support():
    model, leaves = _fitted_gaussian_composite()

    in_support = Scenario(evidence={0: 0.0, 1: 5.0})
    off_support = Scenario(evidence={0: 20.0, 1: -30.0})

    sim_in = simulate(in_support, base=model, seed=0)
    sim_off = simulate(off_support, base=model, seed=0)

    assert sim_in.receipt.plausibility_method == "exact"
    assert sim_off.receipt.plausibility_method == "exact"

    want_in = sum(leaf.log_density(v) for leaf, v in zip(leaves, [0.0, 5.0]))
    want_off = sum(leaf.log_density(v) for leaf, v in zip(leaves, [20.0, -30.0]))
    assert abs(sim_in.receipt.plausibility - want_in) < 1e-8
    assert abs(sim_off.receipt.plausibility - want_off) < 1e-8

    assert sim_in.receipt.plausibility - sim_off.receipt.plausibility > 50.0  # a real, stated margin


# --------------------------------------------------------------------------------------------- #
# (c) HMM horizon rollout past the clamped evidence window
# --------------------------------------------------------------------------------------------- #


def _hmm_fixture():
    return HiddenMarkovModelDistribution(
        [GaussianDistribution(-2.0, 1.0), GaussianDistribution(2.0, 1.0), GaussianDistribution(6.0, 1.0)],
        [0.5, 0.3, 0.2],
        [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]],
    )


def test_hmm_horizon_rollout_matches_forward_filtered_projection():
    hmm = _hmm_fixture()
    v0, v1 = -1.8, 2.1
    horizon = 5
    scenario = Scenario(evidence={0: v0, 1: v1}, horizon=horizon)
    sim = simulate(scenario, base=hmm, seed=3)
    rows = sim.rollout(40000)

    # (i) the clamped positions reproduce the evidence exactly, on every draw.
    assert all(row[0] == v0 and row[1] == v1 for row in rows)

    # (ii) t=4 (beyond the evidenced window [0,1]) matches an independently forward-filtered
    # projection: smoothed state marginal at t=1, propagated forward through the transition matrix
    # (horizon - 1 - t_max) times, then the state-weighted topic mean.
    log_b = np.zeros((2, 3))
    for t, val in {0: v0, 1: v1}.items():
        for k, topic in enumerate(hmm.topics):
            log_b[t, k] = topic.log_density(val)
    q = MarkovChainLatentPosterior(hmm.log_w, hmm.log_transitions, log_b)
    marginals = q.marginals()
    trans = np.exp(np.asarray(hmm.log_transitions))
    state_at_4 = marginals[1] @ np.linalg.matrix_power(trans, 3)
    means = np.array([-2.0, 2.0, 6.0])
    want_mean_t4 = float(state_at_4 @ means)

    got_mean_t4 = float(np.mean([row[4] for row in rows]))
    assert abs(got_mean_t4 - want_mean_t4) < 0.15


# --------------------------------------------------------------------------------------------- #
# (d) determinism
# --------------------------------------------------------------------------------------------- #


def test_bn_rollout_is_deterministic_given_seed():
    net, *_ = _mediator_confounder_bn()
    scenario = Scenario(interventions={1: 2.0}, evidence={2: 3.0})
    sim1 = simulate(scenario, base=net, seed=11)
    sim2 = simulate(scenario, base=net, seed=11)
    r1 = sim1.rollout(50)
    r2 = sim2.rollout(50)
    assert r1 == r2
    assert sim1.receipt.plausibility == sim2.receipt.plausibility


def test_hmm_rollout_is_deterministic_given_seed():
    hmm = _hmm_fixture()
    scenario = Scenario(evidence={0: -1.8}, horizon=4)
    sim1 = simulate(scenario, base=hmm, seed=21)
    sim2 = simulate(scenario, base=hmm, seed=21)
    r1 = sim1.rollout(30)
    r2 = sim2.rollout(30)
    assert r1 == r2
