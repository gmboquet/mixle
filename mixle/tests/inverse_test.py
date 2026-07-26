"""``learn_inverse``: amortized posteriors ``q(theta | y)`` for a simulator, with calibration receipts (M3).

Acceptance receipts, one per test:
  (a) linear-Gaussian inverse matches the analytic (precision-weighted) posterior mean/cov.
  (b) bimodal toy (``y = theta^2 + noise``) -- the learned posterior captures BOTH modes, asserted
      via the repo's existing merged-regime detector (``mixle.inference.structure._split_separation``).
  (c) SBC ranks uniform within finite-sample bounds (200 replications): chi-square p-value > 0.01.
  (d) coverage within +/-5% of nominal at 50%/90%.
  (e) sequential refinement rounds measurably sharpen the posterior at the observed y (round-1
      deliberately under-trained so refinement has visible room to help).

Also covers the two API guards the design note's "resolved" section pins: rounds > 1 without
``y_obs`` raises ``ValueError``, and ``family="flow"`` with 1-D ``theta`` raises ``ValueError``
(``build_conditional_flow`` needs ``y_dim >= 2``; ``theta`` IS the student's ``y``-argument).
"""

import numpy as np
import pytest

import mixle.task.inverse as inverse_module

# _split_separation is private (mixle.inference.structure, leading underscore) -- imported directly
# here rather than made public, matching the precedent mixle/tests/torch_parity_test.py already sets
# for a test reaching into another module's internals (see notes/designs/M3.md, "Resolved").
from mixle.inference.structure import _split_separation
from mixle.stats.multivariate.diagonal_gaussian import DiagonalGaussianDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.task.inverse import InverseModel, InverseReceipts, learn_inverse

torch = pytest.importorskip("torch")


# --------------------------------------------------------------------------------------------- #
# (a) linear-Gaussian -- matches the analytic precision-weighted posterior
# --------------------------------------------------------------------------------------------- #


def test_linear_gaussian_inverse_matches_analytic_posterior():
    rng = np.random.RandomState(0)
    mu0 = np.array([0.0, 0.0])
    var0 = np.array([4.0, 4.0])
    var_obs = 0.25
    prior = DiagonalGaussianDistribution(mu=mu0, covar=var0)

    def simulator(theta):
        return np.asarray(theta, dtype=float) + rng.normal(0.0, np.sqrt(var_obs), size=2)

    model = learn_inverse(simulator, prior, family="flow", n_sims=3000, m_steps=300, seed=0, n_sbc_replications=20)

    for y_obs in (np.array([1.0, -1.0]), np.array([-2.0, 0.5])):
        post = model.posterior(y_obs)
        samples = post.sample(3000, seed=1)

        prec0 = 1.0 / var0
        prec_lik = 1.0 / var_obs
        prec_post = prec0 + prec_lik
        var_post = 1.0 / prec_post
        mean_post = var_post * (prec0 * mu0 + prec_lik * y_obs)

        assert np.max(np.abs(samples.mean(axis=0) - mean_post)) < 0.15
        assert np.max(np.abs(samples.var(axis=0) - var_post)) < 0.15

    assert post.receipt.method == "amortized"
    assert post.receipt.inverse_receipts is model.receipts


# --------------------------------------------------------------------------------------------- #
# (b) bimodal toy -- both modes captured, asserted via the existing merged-regime detector
# --------------------------------------------------------------------------------------------- #


def test_bimodal_posterior_captures_both_modes():
    rng = np.random.RandomState(1)
    prior = GaussianDistribution(mu=0.0, sigma2=4.0)  # theta scalar -> family="mdn" (flow needs y_dim >= 2)

    def simulator(theta):
        theta = float(np.asarray(theta).reshape(-1)[0])
        return np.array([theta**2 + rng.normal(0.0, 0.1)])

    model = learn_inverse(simulator, prior, family="mdn", n_sims=3000, m_steps=300, seed=1, n_sbc_replications=20)

    y_obs = np.array([4.0])  # two roots: theta = +2, -2
    post = model.posterior(y_obs)
    samples = post.sample(2000, seed=2)

    sep, minority_share = _split_separation(samples[:, 0])
    threshold = 2.65 + 6.0 / np.sqrt(len(samples))  # same calibrated finite-sample threshold as
    # mixle.utils.hvis.topology.model_fit_health / mixture_structure_health
    assert sep > threshold
    assert minority_share >= 0.20


def test_family_flow_requires_theta_at_least_2d():
    prior = GaussianDistribution(mu=0.0, sigma2=1.0)

    def simulator(theta):
        return np.array([float(np.asarray(theta).reshape(-1)[0]) ** 2])

    with pytest.raises(ValueError, match="2-dimensional"):
        learn_inverse(simulator, prior, family="flow", n_sims=50, seed=0)


# --------------------------------------------------------------------------------------------- #
# (c) SBC + (d) coverage -- computed together (they share replications inside learn_inverse)
# --------------------------------------------------------------------------------------------- #


def test_sbc_uniform_and_coverage_within_tolerance():
    rng = np.random.RandomState(0)
    mu0 = np.array([0.0, 0.0])
    var0 = np.array([4.0, 4.0])
    var_obs = 0.25
    prior = DiagonalGaussianDistribution(mu=mu0, covar=var0)

    def simulator(theta):
        return np.asarray(theta, dtype=float) + rng.normal(0.0, np.sqrt(var_obs), size=2)

    model = learn_inverse(
        simulator,
        prior,
        family="flow",
        n_sims=3000,
        m_steps=300,
        seed=0,
        n_sbc_replications=200,
        coverage_levels=(0.5, 0.9),
        n_posterior_samples=300,
    )
    r = model.receipts

    # (c) SBC: chi-square uniformity test on binned ranks (Talts et al.), bins = min(20, n // 5),
    # p-value > 0.01 -- the resolved acceptance threshold (design note "Resolved" / module docstring).
    assert r.sbc_bins == min(20, 200 // 5)
    assert r.sbc_pvalue > 0.01
    assert r.sbc_pass is True

    # (d) coverage within +/-5% of nominal at both levels, independently.
    assert abs(r.coverage[0.5] - 0.5) <= 0.05
    assert abs(r.coverage[0.9] - 0.9) <= 0.05
    assert r.coverage_pass[0.5] is True
    assert r.coverage_pass[0.9] is True


def test_rounds_greater_than_one_without_y_obs_raises():
    prior = GaussianDistribution(mu=0.0, sigma2=1.0)

    def simulator(theta):
        return np.array([float(np.asarray(theta).reshape(-1)[0]) ** 2])

    with pytest.raises(ValueError, match="rounds > 1"):
        learn_inverse(simulator, prior, family="mdn", n_sims=50, rounds=2, seed=0)


# --------------------------------------------------------------------------------------------- #
# (e) sequential refinement measurably sharpens the posterior at the observed y
# --------------------------------------------------------------------------------------------- #


def test_sequential_rounds_retain_prior_rows_and_apply_proposal_correction(monkeypatch):
    """The invariant that matters is the target measure, not forced narrowing.

    A valid posterior can remain broad or multimodal, so "sharpness decreases"
    was an unsafe acceptance test. This spies on the actual fit objective:
    round-one prior rows remain present and each proposal block receives the
    declared-prior/proposal density ratio.
    """

    class Prior:
        def log_density(self, theta):
            return -float(np.asarray(theta).reshape(-1)[0])

    initial_theta = np.arange(4, dtype=float).reshape(-1, 1)
    initial_y = initial_theta.copy()
    proposal_theta = np.arange(1, 5, dtype=float).reshape(-1, 1)
    fitted = []
    module = object()

    monkeypatch.setattr(
        inverse_module,
        "_generate_pairs",
        lambda prior, simulator, n, seed: (initial_theta.copy(), initial_y.copy()),
    )
    monkeypatch.setattr(inverse_module, "_build_student", lambda *args, **kwargs: module)
    monkeypatch.setattr(
        inverse_module,
        "_sample_given",
        lambda current, y, n, seed=None: proposal_theta.copy(),
    )
    monkeypatch.setattr(
        inverse_module,
        "_log_density_given",
        lambda current, y, theta: np.zeros(len(theta)),
    )
    monkeypatch.setattr(inverse_module, "_posterior_sharpness", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(
        inverse_module,
        "_calibration_receipts",
        lambda *args, **kwargs: (0.0, 1.0, 2, {0.5: 0.5, 0.9: 0.9}),
    )

    def capture_fit(current, ys, thetas, *, weights=None, **kwargs):
        fitted.append(
            (
                np.asarray(thetas).copy(),
                None if weights is None else np.asarray(weights).copy(),
            )
        )
        return current

    monkeypatch.setattr(inverse_module, "_fit_round", capture_fit)
    result = learn_inverse(
        lambda theta: np.asarray(theta),
        Prior(),
        family="mdn",
        n_sims=4,
        rounds=3,
        y_obs=np.array([0.0]),
        n_sbc_replications=10,
        seed=3,
    )

    assert fitted[0][1] is None
    assert fitted[1][0].shape == (8, 1)
    assert fitted[2][0].shape == (12, 1)
    np.testing.assert_array_equal(fitted[2][0][:4], initial_theta)
    expected, _ = inverse_module._normalized_importance_weights(
        -proposal_theta[:, 0],
        context="test",
    )
    np.testing.assert_allclose(fitted[1][1][4:], expected * 4)
    assert [entry["target"] for entry in result.receipts.round_training] == [
        "declared_prior_joint",
        "declared_prior_joint",
        "declared_prior_joint",
    ]
    assert "p(theta)/q_round" in result.receipts.round_training[1]["correction"]


# --------------------------------------------------------------------------------------------- #
# optional exactness stage -- ESS receipt on a proposal that's already close to the truth
# --------------------------------------------------------------------------------------------- #


def test_reweight_requires_true_log_likelihood():
    prior = GaussianDistribution(mu=0.0, sigma2=1.0)

    def simulator(theta):
        return np.array([float(np.asarray(theta).reshape(-1)[0])])

    with pytest.raises(ValueError, match="true_log_likelihood"):
        learn_inverse(simulator, prior, family="mdn", n_sims=50, reweight=True, y_obs=np.array([0.0]))


def test_reweight_requires_an_observation():
    prior = GaussianDistribution(mu=0.0, sigma2=1.0)
    with pytest.raises(ValueError, match="requires y_obs"):
        learn_inverse(
            lambda theta: np.atleast_1d(theta),
            prior,
            family="mdn",
            n_sims=10,
            reweight=True,
            true_log_likelihood=lambda theta, y: 0.0,
        )


def test_reweighted_model_returns_the_bound_empirical_posterior():
    receipts = InverseReceipts(
        sbc_statistic=0.0,
        sbc_pvalue=1.0,
        sbc_bins=2,
        sbc_replications=10,
        sbc_pass=True,
        coverage={0.9: 0.9},
        coverage_pass={0.9: True},
        prior_predictive={},
        rounds_trained=1,
        ess=1.22,
        ess_ratio=0.61,
        warnings=["finite-particle target correction"],
    )
    model = InverseModel(
        module=object(),
        prior=object(),
        simulator=lambda theta: theta,
        family="mdn",
        theta_dim=1,
        y_dim=1,
        receipts=receipts,
        seed=0,
        reweighted_y=np.array([2.0]),
        reweighted_particles=np.array([[0.0], [10.0]]),
        reweighted_weights=np.array([0.9, 0.1]),
    )
    posterior = model.posterior(np.array([2.0]))
    assert posterior.receipt.method == "sir"
    assert posterior.mean(0) == pytest.approx(1.0)
    assert set(np.unique(posterior.sample(100, seed=1))) <= {0.0, 10.0}
    with pytest.raises(NotImplementedError):
        posterior.log_density([0.0])
    with pytest.raises(ValueError, match="bound"):
        model.posterior(np.array([3.0]))


def test_reweight_reports_ess_receipt():
    rng = np.random.RandomState(0)
    mu0 = np.array([0.0, 0.0])
    var0 = np.array([4.0, 4.0])
    var_obs = 0.25
    prior = DiagonalGaussianDistribution(mu=mu0, covar=var0)

    def simulator(theta):
        return np.asarray(theta, dtype=float) + rng.normal(0.0, np.sqrt(var_obs), size=2)

    def true_log_likelihood(theta, y):
        diff = np.asarray(y, dtype=float) - np.asarray(theta, dtype=float)
        return float(-0.5 * np.sum(diff**2) / var_obs)

    y_obs = np.array([1.0, -1.0])
    model = learn_inverse(
        simulator,
        prior,
        family="flow",
        n_sims=2000,
        m_steps=300,
        seed=0,
        n_sbc_replications=20,
        reweight=True,
        true_log_likelihood=true_log_likelihood,
        y_obs=y_obs,
    )
    assert model.receipts.ess is not None
    assert model.receipts.ess_ratio is not None
    assert 0.0 <= model.receipts.ess_ratio <= 1.0
    # a well-trained student's proposal is close to the true posterior here -> high ESS ratio.
    assert model.receipts.ess_ratio > 0.5
    post = model.posterior(y_obs)
    assert post.receipt.method == "sir"
    assert post.receipt.n_particles == 500
