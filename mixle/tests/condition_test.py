"""condition()/do(): the generic conditioning + intervention engine (M0).

Acceptance receipts, one per test:
  (a) linear-Gaussian composite: conditional mean/cov match the analytic Schur-complement formula
      (computed independently of MultivariateGaussianDistribution.condition) to 1e-8.
  (b) mixture: component reweighting matches Bayes computed by hand on a 2-component fixture.
  (c) HMM: clamped-emission posterior matches forward-backward computed independently.
  (d) SIR fallback matches a brute-force grid posterior within MC error, and its ESS receipt drops
      under extreme evidence.
  (e) do() vs condition() differ per the backdoor formula on a 3-node confounded BN fixture.
  (f) determinism given seed.
"""

import numpy as np
from scipy.stats import norm

from mixle.inference.bayesian_network import HeterogeneousBayesianNetwork, _LinearGaussianFactor, _MarginalFactor
from mixle.inference.condition import ConditionReceipt, condition, do
from mixle.inference.structure import DependencyTreeDistribution
from mixle.stats import (
    CategoricalDistribution,
    ConditionalDistribution,
    GaussianDistribution,
    HiddenMarkovModelDistribution,
    MixtureDistribution,
    MultivariateGaussianDistribution,
)
from mixle.stats.compute.posterior import MarkovChainLatentPosterior

# --------------------------------------------------------------------------------------------- #
# (a) linear-Gaussian composite -- analytic Schur complement to 1e-8
# --------------------------------------------------------------------------------------------- #


def test_gaussian_conditional_matches_schur_complement_analytically():
    rng = np.random.RandomState(42)
    ell = rng.normal(size=(3, 3))
    cov = ell @ ell.T + 3.0 * np.eye(3)  # PD by construction
    mu = np.array([1.0, -2.0, 0.5])
    model = MultivariateGaussianDistribution(mu, cov)

    observed = {0: 2.0, 2: -0.3}
    post = condition(model, observed, method="exact")
    assert post.receipt.method == "exact"

    # Independent re-derivation of the Schur complement (not calling model.condition()).
    obs_idx = [0, 2]
    unobs_idx = [1]
    x_o = np.array([2.0, -0.3])
    s_oo = cov[np.ix_(obs_idx, obs_idx)]
    s_uo = cov[np.ix_(unobs_idx, obs_idx)]
    s_uu = cov[np.ix_(unobs_idx, unobs_idx)]
    mu_cond = mu[unobs_idx] + s_uo @ np.linalg.solve(s_oo, x_o - mu[obs_idx])
    cov_cond = s_uu - s_uo @ np.linalg.solve(s_oo, s_uo.T)

    assert abs(post.mean(1) - mu_cond[0]) < 1e-8
    assert abs(float(post.model.covar[0, 0]) - cov_cond[0, 0]) < 1e-8


# --------------------------------------------------------------------------------------------- #
# (b) mixture reweighting -- Bayes by hand on a 2-component fixture
# --------------------------------------------------------------------------------------------- #


def test_mixture_conditional_reweighting_matches_bayes_by_hand():
    cov1 = np.array([[1.0, 0.5], [0.5, 1.0]])
    cov2 = np.array([[1.0, -0.3], [-0.3, 1.0]])
    mu1 = np.array([0.0, 0.0])
    mu2 = np.array([3.0, 3.0])
    comp1 = MultivariateGaussianDistribution(mu1, cov1)
    comp2 = MultivariateGaussianDistribution(mu2, cov2)
    mix = MixtureDistribution([comp1, comp2], [0.4, 0.6])

    x_o = 1.0
    post = condition(mix, {0: x_o}, method="exact")
    assert post.receipt.method == "exact"

    # Bayes by hand: w'_k propto w_k * N(x_o; mu_k[0], cov_k[0,0]).
    log_lik = np.array(
        [
            norm.logpdf(x_o, loc=mu1[0], scale=np.sqrt(cov1[0, 0])),
            norm.logpdf(x_o, loc=mu2[0], scale=np.sqrt(cov2[0, 0])),
        ]
    )
    log_w = np.log([0.4, 0.6]) + log_lik
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    w /= w.sum()
    np.testing.assert_allclose(post.model.w, w, atol=1e-10)

    # Per-component conditional mean/var of dim 1 given dim 0 = x_o, standard 1-D Gaussian conditional.
    for k, (mu_k, cov_k) in enumerate([(mu1, cov1), (mu2, cov2)]):
        rho = cov_k[0, 1] / cov_k[0, 0]
        mean_want = mu_k[1] + rho * (x_o - mu_k[0])
        var_want = cov_k[1, 1] - cov_k[0, 1] ** 2 / cov_k[0, 0]
        assert abs(float(post.model.components[k].mu[0]) - mean_want) < 1e-8
        assert abs(float(post.model.components[k].covar[0, 0]) - var_want) < 1e-8

    mean_want_total = sum(
        w[k] * (mu[1] + cov[0, 1] / cov[0, 0] * (x_o - mu[0])) for k, (mu, cov) in enumerate([(mu1, cov1), (mu2, cov2)])
    )
    assert abs(post.mean(1) - mean_want_total) < 1e-8


# --------------------------------------------------------------------------------------------- #
# (c) HMM clamped-emission posterior -- forward-backward, independently re-derived
# --------------------------------------------------------------------------------------------- #


def _hmm_fixture():
    return HiddenMarkovModelDistribution(
        [GaussianDistribution(-2.0, 1.0), GaussianDistribution(2.0, 1.0), GaussianDistribution(6.0, 1.0)],
        [0.5, 0.3, 0.2],
        [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]],
    )


def test_hmm_clamped_emission_posterior_matches_forward_backward():
    hmm = _hmm_fixture()
    evidence = {0: -1.8, 3: 5.9}  # positions 1, 2 are unobserved
    post = condition(hmm, evidence, method="exact")
    assert post.receipt.method == "exact"

    # Independent re-derivation: build the partial log_b matrix and forward-backward it directly.
    log_b = np.zeros((4, 3))
    for t, val in evidence.items():
        for k, topic in enumerate(hmm.topics):
            log_b[t, k] = topic.log_density(val)
    q = MarkovChainLatentPosterior(hmm.log_w, hmm.log_transitions, log_b)
    want_marginals = q.marginals()

    np.testing.assert_allclose(post.state_marginals, want_marginals, atol=1e-10)

    means = np.array([-2.0, 2.0, 6.0])
    for t in (1, 2):
        want_mean = float(np.sum(want_marginals[t] * means))
        assert abs(post.mean(t) - want_mean) < 1e-8


# --------------------------------------------------------------------------------------------- #
# (d) SIR fallback vs brute-force grid posterior + ESS receipt drop
# --------------------------------------------------------------------------------------------- #


def _dependency_tree_fixture():
    z_probs = {0: 0.1, 1: 0.2, 2: 0.4, 3: 0.2, 4: 0.1}
    z_dist = CategoricalDistribution(z_probs)
    x_given_z = ConditionalDistribution({k: GaussianDistribution(float(k), 0.5) for k in z_probs})
    tree = DependencyTreeDistribution([None, 0], [z_dist, x_given_z])
    return tree, z_probs


def _grid_posterior_mean(z_probs, obs, sigma=0.5):
    log_w = np.array([np.log(z_probs[k]) + norm.logpdf(obs, loc=float(k), scale=sigma) for k in sorted(z_probs)])
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    w /= w.sum()
    return float(np.sum(w * np.array(sorted(z_probs))))


def test_sir_fallback_matches_brute_force_grid_posterior():
    tree, z_probs = _dependency_tree_fixture()
    obs = 2.0  # near the prior mode -- well-behaved evidence
    post = condition(tree, {1: obs}, method="sir", n_particles=20000, seed=0)
    assert post.receipt.method == "sir"

    want_mean = _grid_posterior_mean(z_probs, obs)
    assert abs(post.mean(0) - want_mean) < 0.05  # within Monte-Carlo error at this ESS


def test_sir_ess_receipt_drops_under_extreme_evidence():
    # The 5-category tree fixture plateaus at ESS ratio ~= the rarest category's prior (~0.1) no
    # matter how far out the evidence sits (the discrete support caps how "surprised" the weights can
    # get); the continuous confounded-BN fixture below has no such floor -- evidence far in the tail
    # of X's marginal drives the weights (and hence the ESS) arbitrarily low, which is the realistic
    # "near-impossible evidence" case the receipt exists to flag.
    tree, _ = _dependency_tree_fixture()
    typical = condition(tree, {1: 2.0}, method="sir", n_particles=20000, seed=1)
    mild_extreme = condition(tree, {1: 10.0}, method="sir", n_particles=20000, seed=1)
    assert isinstance(typical.receipt, ConditionReceipt)
    assert mild_extreme.receipt.ess_ratio < typical.receipt.ess_ratio
    assert mild_extreme.receipt.ess_ratio < 0.5 * typical.receipt.ess_ratio  # a real, not marginal, drop

    net, *_ = _confounded_bn()
    typical_bn = condition(net, {1: 2.0}, method="sir", n_particles=20000, seed=1)
    extreme_bn = condition(net, {1: 60.0}, method="sir", n_particles=20000, seed=1)
    assert extreme_bn.receipt.ess_ratio < typical_bn.receipt.ess_ratio
    assert any("ESS ratio" in w for w in extreme_bn.receipt.warnings)
    assert not any("ESS ratio" in w for w in typical_bn.receipt.warnings)


# --------------------------------------------------------------------------------------------- #
# (e) do() vs condition() -- backdoor formula on a 3-node confounded BN
# --------------------------------------------------------------------------------------------- #


def _confounded_bn():
    # Z -> X, Z -> Y  (Z confounds X and Y; no direct X -> Y edge).
    a, b, sigma_x, sigma_y = 2.0, 3.0, 0.5, 0.5
    f_z = _MarginalFactor(0, GaussianDistribution(0.0, 1.0))
    f_x = _LinearGaussianFactor(1, [0], {}, np.array([a, 0.0]), sigma_x)
    f_y = _LinearGaussianFactor(2, [0], {}, np.array([b, 0.0]), sigma_y)
    return HeterogeneousBayesianNetwork([f_z, f_x, f_y]), a, b, sigma_x


def test_do_vs_condition_differ_per_backdoor_formula():
    net, a, b, sigma_x = _confounded_bn()
    x0 = 2.0

    cond_post = condition(net, {1: x0}, method="sir", n_particles=30000, seed=0)
    e_y_given_x = cond_post.mean(2)

    world = do(net, {1: x0})
    e_y_do_x = world.expectation(2, n=30000, seed=0)

    # Closed form: E[Y | X=x] = a*b*x / (a^2 + sigma_x^2)  (jointly Gaussian (Z,X,Y)).
    want_condition = a * b * x0 / (a**2 + sigma_x**2)
    # E[Y | do(X=x)] = b * E[Z] = 0, since there is no X -> Y edge (backdoor-adjustment closed form:
    # sum_z P(z) E[Y | X=x, Z=z] = integral P(z) * b*z dz = b*E[Z]).
    want_do = 0.0

    assert abs(e_y_given_x - want_condition) < 0.15
    assert abs(e_y_do_x - want_do) < 0.15
    assert abs(e_y_given_x - e_y_do_x) > 1.5  # the backdoor path makes these clearly different


# --------------------------------------------------------------------------------------------- #
# (f) determinism given seed
# --------------------------------------------------------------------------------------------- #


def test_sir_posterior_is_deterministic_given_seed():
    tree, _ = _dependency_tree_fixture()
    p1 = condition(tree, {1: 2.0}, method="sir", n_particles=2000, seed=7)
    p2 = condition(tree, {1: 2.0}, method="sir", n_particles=2000, seed=7)

    assert p1.receipt.ess == p2.receipt.ess
    s1 = p1.sample(10, seed=3)
    s2 = p2.sample(10, seed=3)
    assert s1 == s2


def test_do_sampling_is_deterministic_given_seed():
    net, *_ = _confounded_bn()
    world = do(net, {1: 2.0})
    r1 = world.sample(20, seed=5)
    r2 = world.sample(20, seed=5)
    assert r1 == r2
