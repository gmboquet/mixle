import unittest

import numpy as np

from mixle.inference import optimize
from mixle.inference.mcmc import (
    AdaptiveCovarianceProposal,
    AdaptiveRandomWalkProposal,
    BlockProposal,
    IndependentProposal,
    LangevinProposal,
    MCMCResult,
    MixtureProposal,
    RandomWalkProposal,
    build_parameter_bridge,
    hamiltonian_monte_carlo,
    metropolis_hastings,
    metropolis_within_gibbs,
    nuts,
    posterior_predictive,
    sample_conjugate_posterior,
    sample_distribution,
    sample_parameter_posterior,
)
from mixle.stats import (
    BernoulliDistribution,
    BinomialDistribution,
    CategoricalDistribution,
    DiagonalGaussianDistribution,
    DirichletDistribution,
    GammaDistribution,
    GaussianDistribution,
    GeometricDistribution,
    GumbelDistribution,
    HalfNormalDistribution,
    IntegerCategoricalDistribution,
    LaplaceDistribution,
    NegativeBinomialDistribution,
    PoissonDistribution,
    StudentTDistribution,
    UniformDistribution,
    WeibullDistribution,
)


class MCMCTestCase(unittest.TestCase):
    def test_random_walk_samples_gaussian_target(self):
        dist = GaussianDistribution(1.0, 2.0)
        result = sample_distribution(
            dist,
            initial=0.0,
            proposal=RandomWalkProposal(scale=1.2),
            num_samples=2500,
            burn_in=500,
            thin=2,
            rng=np.random.RandomState(7),
        )
        samples = np.asarray(result.samples, dtype=float)

        self.assertEqual(len(samples), 2500)
        self.assertEqual(len(result.log_probs), 2500)
        self.assertEqual(len(result.accepted), 5500)
        self.assertLess(abs(float(samples.mean()) - 1.0), 0.15)
        self.assertLess(abs(float(samples.var()) - 2.0), 0.35)
        self.assertGreater(result.acceptance_rate, 0.45)
        self.assertLess(result.acceptance_rate, 0.9)
        self.assertGreater(result.effective_sample_size(max_lag=100), 50.0)
        summary = result.summary(max_lag=100)
        self.assertEqual(summary["num_samples"], 2500)
        self.assertLess(abs(summary["mean"] - 1.0), 0.15)
        self.assertGreater(summary["ess"], 50.0)
        self.assertGreater(summary["mcse"], 0.0)

    def test_effective_sample_size_handles_mixed_constant_vector_coordinates(self):
        samples = [np.asarray([3.0, float(i)], dtype=float) for i in range(40)]
        result = MCMCResult(
            samples=samples,
            log_probs=np.zeros(len(samples), dtype=float),
            accepted=np.ones(len(samples), dtype=bool),
        )

        ess = result.effective_sample_size(max_lag=10)
        self.assertEqual(ess.shape, (2,))
        self.assertEqual(float(ess[0]), float(len(samples)))
        self.assertGreater(float(ess[1]), 0.0)
        self.assertLessEqual(float(ess[1]), float(len(samples)))

        summary = result.summary(max_lag=10)
        self.assertEqual(float(np.asarray(summary["mcse"])[0]), 0.0)
        self.assertGreater(float(np.asarray(summary["mcse"])[1]), 0.0)

    def test_mixture_proposal_samples_with_exact_proposal_density(self):
        proposal = MixtureProposal(
            [RandomWalkProposal(scale=0.3), RandomWalkProposal(scale=2.0)],
            weights=[0.7, 0.3],
        )
        result = metropolis_hastings(
            log_target=lambda x: -0.5 * float(x) * float(x),
            initial=0.0,
            proposal=proposal,
            num_samples=1500,
            burn_in=300,
            rng=np.random.RandomState(10),
        )
        samples = np.asarray(result.samples, dtype=float)

        self.assertTrue(np.isfinite(proposal.log_density(1.0, 0.0)))
        self.assertLess(abs(float(samples.mean())), 0.2)
        self.assertLess(abs(float(samples.var()) - 1.0), 0.25)
        self.assertGreater(result.acceptance_rate, 0.5)
        self.assertLess(result.acceptance_rate, 0.95)

    def test_langevin_proposal_uses_gradient_information(self):
        result = metropolis_hastings(
            log_target=lambda x: -0.5 * float(x) * float(x),
            initial=0.0,
            proposal=LangevinProposal(step_size=0.9, grad_log_target=lambda x: -float(x)),
            num_samples=2000,
            burn_in=300,
            rng=np.random.RandomState(5),
        )
        samples = np.asarray(result.samples, dtype=float)

        self.assertLess(abs(float(samples.mean())), 0.15)
        self.assertLess(abs(float(samples.var()) - 1.0), 0.25)
        self.assertGreater(result.acceptance_rate, 0.7)
        self.assertGreater(result.effective_sample_size(max_lag=100), 100.0)

    def test_hamiltonian_monte_carlo_samples_scalar_target(self):
        result = hamiltonian_monte_carlo(
            log_target=lambda x: -0.5 * (float(x) - 1.0) * (float(x) - 1.0) / 2.0,
            grad_log_target=lambda x: -(float(x) - 1.0) / 2.0,
            initial=0.0,
            num_samples=2000,
            burn_in=300,
            step_size=0.25,
            num_steps=8,
            rng=np.random.RandomState(21),
        )
        samples = np.asarray(result.samples, dtype=float)

        self.assertEqual(len(samples), 2000)
        self.assertEqual(set(result.acceptance_rate_by_label), {"hmc"})
        self.assertLess(abs(float(samples.mean()) - 1.0), 0.15)
        self.assertLess(abs(float(samples.var()) - 2.0), 0.35)
        self.assertGreater(result.acceptance_rate, 0.8)
        self.assertGreater(result.effective_sample_size(max_lag=100), 300.0)

    def test_hamiltonian_monte_carlo_samples_vector_target(self):
        mean = np.asarray([1.0, -1.0])
        var = np.asarray([1.0, 4.0])

        def log_target(x):
            xx = np.asarray(x, dtype=float)
            return float(-0.5 * np.sum((xx - mean) * (xx - mean) / var))

        def grad_log_target(x):
            xx = np.asarray(x, dtype=float)
            return -(xx - mean) / var

        result = hamiltonian_monte_carlo(
            log_target=log_target,
            grad_log_target=grad_log_target,
            initial=np.asarray([0.0, 0.0]),
            num_samples=1500,
            burn_in=300,
            step_size=0.2,
            num_steps=10,
            mass=np.asarray([1.0, 2.0]),
            rng=np.random.RandomState(22),
        )
        samples = np.asarray(result.samples, dtype=float)

        self.assertEqual(samples.shape, (1500, 2))
        self.assertLess(abs(float(samples[:, 0].mean()) - 1.0), 0.15)
        self.assertLess(abs(float(samples[:, 1].mean()) + 1.0), 0.25)
        self.assertLess(abs(float(samples[:, 0].var()) - 1.0), 0.25)
        self.assertLess(abs(float(samples[:, 1].var()) - 4.0), 0.6)
        self.assertGreater(result.acceptance_rate, 0.8)

    def test_nuts_samples_correlated_gaussian(self):
        mean = np.asarray([2.0, -3.0])
        cov = np.asarray([[1.0, 0.8], [0.8, 1.0]])
        prec = np.linalg.inv(cov)

        def log_target(x):
            xx = np.asarray(x, dtype=float) - mean
            return float(-0.5 * xx @ prec @ xx)

        def grad_log_target(x):
            return -prec @ (np.asarray(x, dtype=float) - mean)

        result = nuts(
            log_target,
            grad_log_target,
            np.zeros(2),
            num_samples=2000,
            warmup=500,
            rng=np.random.RandomState(0),
        )
        s = np.asarray(result.samples, dtype=float)
        self.assertEqual(s.shape, (2000, 2))
        self.assertTrue(np.allclose(s.mean(0), mean, atol=0.15))
        self.assertLess(abs(np.cov(s.T)[0, 1] - 0.8), 0.2)  # recovers the correlation
        self.assertGreater(result.step_size, 0.0)  # adapted

    def test_posterior_predictive_uses_retained_states(self):
        result = hamiltonian_monte_carlo(
            log_target=lambda x: -0.5 * float(x) * float(x),
            grad_log_target=lambda x: -float(x),
            initial=0.0,
            num_samples=1000,
            burn_in=100,
            step_size=0.25,
            num_steps=6,
            rng=np.random.RandomState(25),
        )
        draws = posterior_predictive(
            result,
            sampler=lambda state, rng: float(rng.normal(loc=state, scale=0.5)),
            rng=np.random.RandomState(26),
        )
        sized_draws = posterior_predictive(
            [1.0, 2.0],
            sampler=lambda state, rng, size: rng.normal(loc=state, scale=1.0, size=size),
            rng=np.random.RandomState(27),
            size=3,
        )

        self.assertEqual(len(draws), 1000)
        self.assertLess(abs(float(np.mean(draws))), 0.2)
        self.assertLess(abs(float(np.var(draws)) - 1.25), 0.35)
        self.assertEqual([len(x) for x in sized_draws], [3, 3])

    def test_adaptive_random_walk_adapts_scale_during_burn_in(self):
        proposal = AdaptiveRandomWalkProposal(scale=0.05, adaptation_rate=1.0)
        result = metropolis_hastings(
            log_target=lambda x: -0.5 * float(x) * float(x),
            initial=0.0,
            proposal=proposal,
            num_samples=500,
            burn_in=500,
            rng=np.random.RandomState(6),
        )

        self.assertGreater(float(proposal.scale), 1.0)
        self.assertGreater(result.acceptance_rate, 0.2)
        self.assertLess(result.acceptance_rate, 0.7)

    def test_adaptive_covariance_proposal_learns_vector_scale(self):
        target_cov = np.asarray([[1.0, 0.8], [0.8, 4.0]])
        target_inv = np.linalg.inv(target_cov)

        def log_target(x):
            xx = np.asarray(x, dtype=float)
            return float(-0.5 * np.dot(xx, np.dot(target_inv, xx)))

        proposal = AdaptiveCovarianceProposal(
            initial_covariance=0.05,
            regularization=1.0e-5,
            adapt_after=20,
        )
        result = metropolis_hastings(
            log_target=log_target,
            initial=np.asarray([0.0, 0.0]),
            proposal=proposal,
            num_samples=3000,
            burn_in=1500,
            rng=np.random.RandomState(31),
        )
        samples = np.asarray(result.samples, dtype=float)
        empirical_cov = np.cov(samples.T)
        summary = result.summary(max_lag=100)

        self.assertEqual(samples.shape, (3000, 2))
        self.assertLess(abs(float(samples[:, 0].mean())), 0.15)
        self.assertLess(abs(float(samples[:, 1].mean())), 0.2)
        self.assertLess(abs(float(empirical_cov[0, 0]) - 1.0), 0.25)
        self.assertLess(abs(float(empirical_cov[1, 1]) - 4.0), 0.7)
        self.assertGreater(float(proposal.covariance[1, 1]), float(proposal.covariance[0, 0]))
        self.assertGreater(float(proposal.covariance[0, 1]), 0.3)
        self.assertGreater(result.acceptance_rate, 0.2)
        self.assertLess(result.acceptance_rate, 0.6)
        self.assertEqual(summary["num_samples"], 3000)
        self.assertEqual(summary["mean"].shape, (2,))
        self.assertEqual(summary["mcse"].shape, (2,))

    def test_independent_proposal_uses_hastings_correction(self):
        target = GaussianDistribution(1.0, 2.0)

        def proposal_log_density(x):
            xx = float(x)
            return -0.5 * np.log(2.0 * np.pi * 4.0) - 0.5 * (xx + 3.0) * (xx + 3.0) / 4.0

        proposal = IndependentProposal(
            sampler=lambda rng: float(rng.normal(loc=-3.0, scale=2.0)),
            log_density=proposal_log_density,
        )
        result = sample_distribution(
            target,
            initial=1.0,
            proposal=proposal,
            num_samples=1500,
            burn_in=200,
            rng=np.random.RandomState(3),
        )
        samples = np.asarray(result.samples, dtype=float)

        self.assertEqual(len(samples), 1500)
        self.assertTrue(np.all(np.isfinite(result.log_probs)))
        self.assertGreater(result.acceptance_rate, 0.02)
        self.assertLess(result.acceptance_rate, 0.3)
        self.assertLess(abs(float(samples.mean()) - 1.0), 0.35)

    def test_blocked_metropolis_within_gibbs_updates_dict_state(self):
        def log_target(state):
            x = float(state["x"])
            y = float(state["y"])
            return -0.5 * x * x - 0.5 * (y - 2.0) * (y - 2.0) / 0.5

        result = metropolis_within_gibbs(
            log_target=log_target,
            initial={"x": 0.0, "y": 0.0},
            proposals={
                "x": BlockProposal("x", RandomWalkProposal(scale=0.8)),
                "y": BlockProposal("y", RandomWalkProposal(scale=0.7)),
            },
            num_samples=2500,
            burn_in=500,
            rng=np.random.RandomState(9),
        )
        xs = np.asarray([sample["x"] for sample in result.samples], dtype=float)
        ys = np.asarray([sample["y"] for sample in result.samples], dtype=float)
        by_label = result.acceptance_rate_by_label

        self.assertEqual(set(by_label), {"x", "y"})
        self.assertEqual(len(result.accepted), 6000)
        self.assertLess(abs(float(xs.mean())), 0.15)
        self.assertLess(abs(float(xs.var()) - 1.0), 0.35)
        self.assertLess(abs(float(ys.mean()) - 2.0), 0.15)
        self.assertLess(abs(float(ys.var()) - 0.5), 0.2)
        self.assertGreater(by_label["x"], 0.4)
        self.assertGreater(by_label["y"], 0.4)

    def test_evidence_updates_distribution_target(self):
        prior = GaussianDistribution(0.0, 10.0)
        observation = 4.0

        def evidence(x):
            xx = float(x)
            return -0.5 * (xx - observation) * (xx - observation)

        result = sample_distribution(
            prior,
            initial=0.0,
            proposal=RandomWalkProposal(scale=1.0),
            num_samples=2500,
            burn_in=500,
            thin=2,
            rng=np.random.RandomState(11),
            evidence=evidence,
        )
        samples = np.asarray(result.samples, dtype=float)

        # N(0, 10) prior plus unit-variance Gaussian evidence at 4.0.
        self.assertLess(abs(float(samples.mean()) - (40.0 / 11.0)), 0.15)
        self.assertLess(abs(float(samples.var()) - (10.0 / 11.0)), 0.2)

    def test_metropolis_hastings_validates_inputs(self):
        target = GaussianDistribution(0.0, 1.0)
        proposal = RandomWalkProposal(scale=1.0)

        with self.assertRaisesRegex(ValueError, "num_samples"):
            sample_distribution(target, 0.0, proposal, num_samples=-1)
        with self.assertRaisesRegex(ValueError, "burn_in"):
            sample_distribution(target, 0.0, proposal, num_samples=1, burn_in=-1)
        with self.assertRaisesRegex(ValueError, "thin"):
            sample_distribution(target, 0.0, proposal, num_samples=1, thin=0)
        with self.assertRaisesRegex(ValueError, "initial state"):
            sample_distribution(GammaDistribution(2.0, 1.0), -1.0, proposal, num_samples=1)
        with self.assertRaisesRegex(ValueError, "step_size"):
            hamiltonian_monte_carlo(
                lambda x: -float(x) * float(x),
                lambda x: -2.0 * float(x),
                0.0,
                num_samples=1,
                step_size=0.0,
                num_steps=1,
            )
        with self.assertRaisesRegex(ValueError, "num_steps"):
            hamiltonian_monte_carlo(
                lambda x: -float(x) * float(x),
                lambda x: -2.0 * float(x),
                0.0,
                num_samples=1,
                step_size=0.1,
                num_steps=0,
            )
        with self.assertRaisesRegex(ValueError, "grad_log_target shape"):
            hamiltonian_monte_carlo(
                lambda x: -0.5 * float(x) * float(x),
                lambda x: [0.0, 0.0],
                0.0,
                num_samples=1,
                step_size=0.1,
                num_steps=1,
            )
        with self.assertRaisesRegex(ValueError, "adapt_after"):
            AdaptiveCovarianceProposal(adapt_after=1)
        with self.assertRaisesRegex(ValueError, "initial_covariance"):
            AdaptiveCovarianceProposal(initial_covariance=-1.0).sample(np.asarray([0.0]), np.random.RandomState(1))

    def test_metropolis_hastings_accepts_unnormalized_log_targets(self):
        result = metropolis_hastings(
            log_target=lambda x: -0.5 * float(x) * float(x),
            initial=0.0,
            proposal=RandomWalkProposal(scale=0.8),
            num_samples=100,
            burn_in=20,
            rng=np.random.RandomState(19),
        )

        self.assertEqual(len(result.samples), 100)
        self.assertTrue(np.all(np.isfinite(result.log_probs)))
        self.assertGreater(result.acceptance_rate, 0.0)


class ParameterPosteriorTestCase(unittest.TestCase):
    def test_gaussian_mean_posterior_matches_conjugate(self):
        # Known-variance normal model with a normal prior on the mean has a
        # closed-form normal posterior. We sample the full (mu, sigma2) posterior
        # under a flat prior and compare the marginal mu posterior mean/variance
        # to the analytic Normal-known-variance result evaluated at the MLE
        # variance, which is the right reference up to O(1/n).
        rng = np.random.RandomState(101)
        true_mu, true_sigma2 = 2.0, 1.5
        data = rng.normal(true_mu, np.sqrt(true_sigma2), size=400)
        n = len(data)
        sample_mean = float(data.mean())
        sample_var = float(data.var())

        result = sample_parameter_posterior(
            GaussianDistribution(0.0, 1.0),
            data,
            prior=None,
            sampler="mh",
            steps=4000,
            burn_in=2000,
            seed=7,
            proposal=RandomWalkProposal(scale=[0.08, 0.08]),
        )

        mus = np.asarray([t[0] for t in result.samples], dtype=float)
        s2s = np.asarray([t[1] for t in result.samples], dtype=float)

        # Posterior mean of mu concentrates at the sample mean; posterior
        # variance of mu is approximately sigma2 / n.
        post_mu_mean = float(mus.mean())
        post_mu_var = float(mus.var())
        analytic_mu_var = sample_var / n

        self.assertLess(abs(post_mu_mean - sample_mean), 5.0 * np.sqrt(analytic_mu_var))
        self.assertLess(abs(post_mu_var - analytic_mu_var) / analytic_mu_var, 0.4)
        # Variance posterior centers near the sample variance.
        self.assertLess(abs(float(s2s.mean()) - sample_var) / sample_var, 0.2)
        self.assertGreater(result.acceptance_rate, 0.1)
        self.assertLess(result.acceptance_rate, 0.95)

    def test_gaussian_posterior_hmc_agrees_with_mh(self):
        rng = np.random.RandomState(202)
        data = rng.normal(-1.0, 2.0, size=300)

        mh = sample_parameter_posterior(
            GaussianDistribution(0.0, 1.0),
            data,
            sampler="mh",
            steps=3000,
            burn_in=1500,
            seed=11,
            proposal=RandomWalkProposal(scale=[0.1, 0.1]),
        )
        hmc = sample_parameter_posterior(
            GaussianDistribution(0.0, 1.0),
            data,
            sampler="hmc",
            steps=1500,
            burn_in=800,
            seed=12,
            step_size=0.02,
            num_steps=15,
        )

        mh_mu = float(np.mean([t[0] for t in mh.samples]))
        hmc_mu = float(np.mean([t[0] for t in hmc.samples]))
        self.assertLess(abs(mh_mu - hmc_mu), 0.1)
        self.assertGreater(hmc.acceptance_rate, 0.5)

    def test_poisson_rate_posterior_matches_gamma_conjugate(self):
        # Flat improper prior on log(lam) gives posterior proportional to the
        # likelihood; with the reparameterization Jacobian this equals the
        # Gamma(sum+1, 1/n) posterior in lam-space (a Gamma posterior for an
        # improper Gamma(1, inf) prior). Compare the MH posterior mean of lam to
        # that analytic mean within a few posterior standard errors.
        rng = np.random.RandomState(303)
        data = rng.poisson(5.0, size=250)
        total = float(data.sum())
        n = len(data)

        result = sample_parameter_posterior(
            PoissonDistribution(1.0),
            data,
            prior=None,
            sampler="mh",
            steps=4000,
            burn_in=2000,
            seed=13,
            proposal=RandomWalkProposal(scale=0.05),
        )

        lams = np.asarray(result.samples, dtype=float)
        # Posterior over lam under flat prior on log-lam: Gamma(total, scale=1/n)
        # => mean total/n, variance total/n^2.
        analytic_mean = total / n
        analytic_var = total / (n * n)
        self.assertLess(abs(float(lams.mean()) - analytic_mean), 5.0 * np.sqrt(analytic_var))
        self.assertLess(abs(float(lams.var()) - analytic_var) / analytic_var, 0.4)
        self.assertTrue(np.all(lams > 0.0))

    def test_bernoulli_probability_posterior_stays_in_unit_interval(self):
        rng = np.random.RandomState(404)
        data = (rng.rand(300) < 0.35).astype(int)
        ones = int(data.sum())
        n = len(data)

        result = sample_parameter_posterior(
            BernoulliDistribution(0.5),
            data,
            prior=None,
            sampler="mh",
            steps=4000,
            burn_in=2000,
            seed=14,
            proposal=RandomWalkProposal(scale=0.1),
        )

        ps = np.asarray(result.samples, dtype=float)
        self.assertTrue(np.all(ps > 0.0))
        self.assertTrue(np.all(ps < 1.0))
        # Flat prior on logit p -> Beta(ones, zeros) posterior; mean ones/n.
        self.assertLess(abs(float(ps.mean()) - ones / n), 0.05)

    def test_categorical_posterior_samples_are_valid_simplices(self):
        rng = np.random.RandomState(505)
        labels = ["a", "b", "c", "d"]
        true_p = [0.1, 0.4, 0.3, 0.2]
        data = rng.choice(labels, p=true_p, size=600).tolist()
        proto = CategoricalDistribution({k: 0.25 for k in labels})

        result = sample_parameter_posterior(proto, data, prior=None, sampler="mh", steps=3000, burn_in=1500, seed=15)

        for prob_map in result.samples:
            self.assertEqual(set(prob_map), set(labels))
            vals = np.asarray(list(prob_map.values()), dtype=float)
            self.assertTrue(np.all(vals > 0.0))
            self.assertTrue(np.all(vals < 1.0))
            self.assertAlmostEqual(float(vals.sum()), 1.0, places=9)

        # Posterior means recover the empirical frequencies.
        counts = {k: data.count(k) for k in labels}
        n = len(data)
        for k in labels:
            post_mean = float(np.mean([pm[k] for pm in result.samples]))
            self.assertLess(abs(post_mean - counts[k] / n), 0.05)

    def test_return_distributions_rebuilds_models(self):
        rng = np.random.RandomState(606)
        data = rng.poisson(3.0, size=100)
        result = sample_parameter_posterior(
            PoissonDistribution(1.0),
            data,
            sampler="mh",
            steps=50,
            burn_in=50,
            seed=16,
            return_distributions=True,
            proposal=RandomWalkProposal(scale=0.05),
        )
        self.assertEqual(len(result.samples), 50)
        for dist in result.samples:
            self.assertIsInstance(dist, PoissonDistribution)
            self.assertGreater(dist.lam, 0.0)

    def test_prior_callable_shifts_posterior(self):
        # A strong prior pulling lam toward a small value should lower the
        # posterior mean relative to the flat-prior posterior.
        rng = np.random.RandomState(707)
        data = rng.poisson(8.0, size=40)

        flat = sample_parameter_posterior(
            PoissonDistribution(1.0),
            data,
            prior=None,
            sampler="mh",
            steps=3000,
            burn_in=1500,
            seed=17,
            proposal=RandomWalkProposal(scale=0.08),
        )
        # log Gamma(k=2, scale=1) density on lam (drops constant terms).
        shaped = sample_parameter_posterior(
            PoissonDistribution(1.0),
            data,
            prior=lambda lam: (2.0 - 1.0) * np.log(lam) - lam / 1.0,
            sampler="mh",
            steps=3000,
            burn_in=1500,
            seed=18,
            proposal=RandomWalkProposal(scale=0.08),
        )

        flat_mean = float(np.mean(flat.samples))
        shaped_mean = float(np.mean(shaped.samples))
        self.assertLess(shaped_mean, flat_mean)
        self.assertGreater(shaped_mean, 0.0)

    def test_unsupported_family_raises_not_implemented(self):
        with self.assertRaisesRegex(NotImplementedError, "does not support"):
            build_parameter_bridge(GammaDistribution(2.0, 1.0).estimator())

    def test_bridge_round_trips_parameters(self):
        for proto in (
            GaussianDistribution(1.3, 2.7),
            PoissonDistribution(4.2),
            BernoulliDistribution(0.37),
            CategoricalDistribution({"a": 0.2, "b": 0.5, "c": 0.3}),
        ):
            bridge = build_parameter_bridge(proto)
            phi = bridge.to_unconstrained(bridge.initial_theta)
            theta = bridge.from_unconstrained(phi)
            if isinstance(theta, dict):
                for k in theta:
                    self.assertAlmostEqual(theta[k], bridge.initial_theta[k], places=8)
            elif isinstance(theta, tuple):
                for a, b in zip(theta, bridge.initial_theta):
                    self.assertAlmostEqual(a, b, places=8)
            else:
                self.assertAlmostEqual(theta, bridge.initial_theta, places=8)


class GenericParameterBridgeTestCase(unittest.TestCase):
    """Real posterior-sampling scenarios for families reachable only through the generic
    declaration-driven bridge (``_generic_declared_bridge`` in ``parameter_bridge.py``), not one
    of the 7 hand-tuned branches in ``build_parameter_bridge``. Each test fits an independent
    reference -- the generalized-EM ``optimize()`` MLE, or (for Binomial, where ``optimize()``
    also estimates the trial count ``n``) a closed-form formula -- and checks the MCMC posterior
    mean agrees with it within a comfortable multiple of the posterior's own standard error, plus
    domain-validity checks (positivity, unit interval, simplex). This is meant to catch a wrong
    Jacobian sign or a wrong reparameterization, not just confirm dispatch doesn't raise."""

    def test_weibull_shape_scale_posterior_matches_mle(self):
        truth = WeibullDistribution(2.0, 3.0)
        data = truth.sampler(1).sample(800)
        fitted = optimize(data, truth.estimator(), max_its=100)

        result = sample_parameter_posterior(
            WeibullDistribution(1.0, 1.0),
            data,
            prior=None,
            sampler="mh",
            steps=5000,
            burn_in=2000,
            seed=27,
            proposal=RandomWalkProposal(scale=[0.05, 0.05]),
        )
        shapes = np.asarray([t["shape"] for t in result.samples], dtype=float)
        scales = np.asarray([t["scale"] for t in result.samples], dtype=float)
        self.assertTrue(np.all(shapes > 0.0))
        self.assertTrue(np.all(scales > 0.0))
        self.assertLess(abs(float(shapes.mean()) - fitted.shape), 6.0 * float(shapes.std()))
        self.assertLess(abs(float(scales.mean()) - fitted.scale), 6.0 * float(scales.std()))

    def test_student_t_df_loc_scale_posterior_matches_mle(self):
        truth = StudentTDistribution(6.0, 1.0, 2.0)
        data = truth.sampler(2).sample(1500)
        fitted = optimize(data, truth.estimator(), max_its=100)

        result = sample_parameter_posterior(
            StudentTDistribution(3.0, 0.0, 1.0),
            data,
            prior=None,
            sampler="mh",
            steps=5000,
            burn_in=2000,
            seed=22,
            proposal=RandomWalkProposal(scale=[0.05, 0.05, 0.05]),
        )
        dfs = np.asarray([t["df"] for t in result.samples], dtype=float)
        locs = np.asarray([t["loc"] for t in result.samples], dtype=float)
        scales = np.asarray([t["scale"] for t in result.samples], dtype=float)
        self.assertTrue(np.all(dfs > 0.0))
        self.assertTrue(np.all(scales > 0.0))
        self.assertLess(abs(float(dfs.mean()) - fitted.df), 6.0 * float(dfs.std()))
        self.assertLess(abs(float(locs.mean()) - fitted.loc), 6.0 * float(locs.std()))
        self.assertLess(abs(float(scales.mean()) - fitted.scale), 6.0 * float(scales.std()))

    def test_laplace_posterior_matches_mle(self):
        truth = LaplaceDistribution(1.0, 2.0)
        data = truth.sampler(3).sample(800)
        fitted = optimize(data, truth.estimator(), max_its=100)

        result = sample_parameter_posterior(
            LaplaceDistribution(0.0, 1.0), data, prior=None, sampler="mh", steps=4000, burn_in=2000, seed=29
        )
        mus = np.asarray([t["mu"] for t in result.samples], dtype=float)
        bs = np.asarray([t["b"] for t in result.samples], dtype=float)
        self.assertTrue(np.all(bs > 0.0))
        self.assertLess(abs(float(mus.mean()) - fitted.mu), 6.0 * float(mus.std()))
        self.assertLess(abs(float(bs.mean()) - fitted.b), 6.0 * float(bs.std()))

    def test_gumbel_posterior_matches_mle(self):
        truth = GumbelDistribution(0.0, 1.5)
        data = truth.sampler(4).sample(800)
        fitted = optimize(data, truth.estimator(), max_its=100)

        result = sample_parameter_posterior(
            GumbelDistribution(1.0, 1.0), data, prior=None, sampler="mh", steps=4000, burn_in=2000, seed=28
        )
        locs = np.asarray([t["loc"] for t in result.samples], dtype=float)
        scales = np.asarray([t["scale"] for t in result.samples], dtype=float)
        self.assertTrue(np.all(scales > 0.0))
        self.assertLess(abs(float(locs.mean()) - fitted.loc), 6.0 * float(locs.std()))
        self.assertLess(abs(float(scales.mean()) - fitted.scale), 6.0 * float(scales.std()))

    def test_geometric_posterior_matches_mle(self):
        truth = GeometricDistribution(0.3)
        data = truth.sampler(5).sample(800)
        fitted = optimize(data, truth.estimator(), max_its=100)

        result = sample_parameter_posterior(
            GeometricDistribution(0.5), data, prior=None, sampler="mh", steps=4000, burn_in=2000, seed=30
        )
        ps = np.asarray([t["p"] for t in result.samples], dtype=float)
        self.assertTrue(np.all(ps > 0.0))
        self.assertTrue(np.all(ps < 1.0))
        self.assertLess(abs(float(ps.mean()) - fitted.p), 6.0 * float(ps.std()))

    def test_negative_binomial_posterior_matches_mle(self):
        truth = NegativeBinomialDistribution(4.0, 0.4)
        data = truth.sampler(6).sample(1500)
        fitted = optimize(data, truth.estimator(), max_its=100)

        result = sample_parameter_posterior(
            NegativeBinomialDistribution(2.0, 0.5),
            data,
            prior=None,
            sampler="mh",
            steps=5000,
            burn_in=2000,
            seed=31,
            proposal=RandomWalkProposal(scale=[0.05, 0.05]),
        )
        rs = np.asarray([t["r"] for t in result.samples], dtype=float)
        ps = np.asarray([t["p"] for t in result.samples], dtype=float)
        self.assertTrue(np.all(rs > 0.0))
        self.assertTrue(np.all((ps > 0.0) & (ps < 1.0)))
        self.assertLess(abs(float(rs.mean()) - fitted.r), 6.0 * float(rs.std()))
        self.assertLess(abs(float(ps.mean()) - fitted.p), 6.0 * float(ps.std()))

    def test_binomial_posterior_matches_analytic_with_fixed_trial_count(self):
        # Binomial's n is a non-differentiable (fixed) declared parameter: the bridge must hold
        # it at the prototype's value rather than sample it, so every rebuilt distribution keeps
        # n == 20 and the posterior mean of p should track the closed-form flat-prior-on-logit(p)
        # reference sum(data) / (n * count). optimize() is not used as the reference here because
        # Binomial's estimator also fits n from data (a much noisier joint problem).
        truth = BinomialDistribution(0.35, 20)
        data = truth.sampler(7).sample(800)
        reference_p = float(np.sum(data)) / (20.0 * len(data))

        result = sample_parameter_posterior(
            BinomialDistribution(0.5, 20),
            data,
            prior=None,
            sampler="mh",
            steps=4000,
            burn_in=2000,
            seed=26,
            return_distributions=True,
        )
        ps = np.asarray([d.p for d in result.samples], dtype=float)
        ns = {int(d.n) for d in result.samples}
        self.assertEqual(ns, {20})  # fixed nuisance parameter never moves
        self.assertTrue(np.all(ps > 0.0))
        self.assertTrue(np.all(ps < 1.0))
        self.assertLess(abs(float(ps.mean()) - reference_p), 0.03)

    def test_dirichlet_posterior_matches_mle(self):
        truth = DirichletDistribution([2.0, 3.0, 4.0])
        data = truth.sampler(8).sample(1000)
        fitted = optimize(data, truth.estimator(), max_its=100)

        result = sample_parameter_posterior(
            DirichletDistribution([1.0, 1.0, 1.0]),
            data,
            prior=None,
            sampler="mh",
            steps=5000,
            burn_in=2000,
            seed=23,
            proposal=RandomWalkProposal(scale=[0.04, 0.04, 0.04]),
        )
        alphas = np.stack([np.asarray(t["alpha"], dtype=float) for t in result.samples])
        self.assertTrue(np.all(alphas > 0.0))
        post_mean = alphas.mean(axis=0)
        post_std = alphas.std(axis=0)
        np.testing.assert_array_less(np.abs(post_mean - np.asarray(fitted.alpha)), 6.0 * post_std)

    def test_diagonal_gaussian_posterior_matches_mle(self):
        truth = DiagonalGaussianDistribution([1.0, -1.0], [1.0, 2.0])
        data = truth.sampler(9).sample(1000)
        fitted = optimize(data, truth.estimator(), max_its=100)

        result = sample_parameter_posterior(
            DiagonalGaussianDistribution([0.0, 0.0], [1.0, 1.0]),
            data,
            prior=None,
            sampler="mh",
            steps=5000,
            burn_in=2000,
            seed=24,
            proposal=RandomWalkProposal(scale=[0.05, 0.05, 0.05, 0.05]),
        )
        mus = np.stack([np.asarray(t["mu"], dtype=float) for t in result.samples])
        covars = np.stack([np.asarray(t["covar"], dtype=float) for t in result.samples])
        self.assertTrue(np.all(covars > 0.0))
        np.testing.assert_array_less(np.abs(mus.mean(axis=0) - np.asarray(fitted.mu)), 6.0 * mus.std(axis=0))
        np.testing.assert_array_less(np.abs(covars.mean(axis=0) - np.asarray(fitted.covar)), 6.0 * covars.std(axis=0))

    def test_integer_categorical_posterior_matches_mle(self):
        truth = IntegerCategoricalDistribution(0, [0.2, 0.3, 0.5])
        data = truth.sampler(10).sample(1000)
        fitted = optimize(data, truth.estimator(), max_its=100)

        result = sample_parameter_posterior(
            IntegerCategoricalDistribution(0, [1.0 / 3, 1.0 / 3, 1.0 / 3]),
            data,
            prior=None,
            sampler="mh",
            steps=4000,
            burn_in=2000,
            seed=25,
        )
        p_vecs = np.stack([np.asarray(t["p_vec"], dtype=float) for t in result.samples])
        self.assertTrue(np.all(p_vecs > 0.0))
        np.testing.assert_allclose(p_vecs.sum(axis=1), 1.0, atol=1.0e-9)
        post_mean = p_vecs.mean(axis=0)
        post_std = p_vecs.std(axis=0)
        np.testing.assert_array_less(np.abs(post_mean - np.asarray(fitted.p_vec)), 6.0 * post_std)

    def test_half_normal_posterior_matches_mle(self):
        truth = HalfNormalDistribution(2.0)
        data = truth.sampler(11).sample(800)
        fitted = optimize(data, truth.estimator(), max_its=100)

        result = sample_parameter_posterior(
            HalfNormalDistribution(1.0), data, prior=None, sampler="mh", steps=4000, burn_in=2000, seed=32
        )
        sigmas = np.asarray([t["sigma"] for t in result.samples], dtype=float)
        self.assertTrue(np.all(sigmas > 0.0))
        self.assertLess(abs(float(sigmas.mean()) - fitted.sigma), 6.0 * float(sigmas.std()))

    def test_return_distributions_rebuilds_generic_family(self):
        truth = WeibullDistribution(2.0, 3.0)
        data = truth.sampler(35).sample(100)
        result = sample_parameter_posterior(
            WeibullDistribution(1.0, 1.0),
            data,
            sampler="mh",
            steps=50,
            burn_in=50,
            seed=35,
            return_distributions=True,
            proposal=RandomWalkProposal(scale=[0.05, 0.05]),
        )
        self.assertEqual(len(result.samples), 50)
        for dist in result.samples:
            self.assertIsInstance(dist, WeibullDistribution)
            self.assertGreater(dist.shape, 0.0)
            self.assertGreater(dist.scale, 0.0)

    def test_declared_family_with_coupled_bound_still_raises(self):
        # UniformDistribution declares a coupled `greater_than:low` bound on `high`; the generic
        # bridge deliberately excludes coupled bounds (see build_parameter_bridge's docstring),
        # so this must still raise -- a regression guard against silently mis-dispatching it.
        with self.assertRaisesRegex(NotImplementedError, "does not support"):
            build_parameter_bridge(UniformDistribution(0.0, 1.0))


class ConjugatePosteriorTestCase(unittest.TestCase):
    def _stats(self):
        import mixle.stats as stats
        from mixle.stats.univariate.continuous.beta import BetaDistribution as BBeta
        from mixle.stats.univariate.continuous.gamma import GammaDistribution as BGamma

        return stats, BBeta, BGamma

    def test_poisson_gamma_conjugate_matches_analytic(self):
        stats, _, BGamma = self._stats()
        rng = np.random.RandomState(11)
        data = rng.poisson(4.0, size=200).tolist()
        k0, theta0 = 2.0, 1.0
        proto = stats.PoissonDistribution(1.0, prior=BGamma(k0, theta0))

        result = sample_conjugate_posterior(proto, data, draws=40000, seed=21)
        samples = np.asarray(result.samples, dtype=float)

        total = float(sum(data))
        n = len(data)
        post_k = k0 + total
        post_theta = theta0 / (n * theta0 + 1.0)
        analytic_mean = post_k * post_theta
        analytic_var = post_k * post_theta * post_theta

        self.assertTrue(np.all(result.accepted))
        self.assertLess(abs(float(samples.mean()) - analytic_mean), 4.0 * np.sqrt(analytic_var / len(samples)) + 0.01)
        self.assertLess(abs(float(samples.var()) - analytic_var) / analytic_var, 0.1)

    def test_bernoulli_beta_conjugate_matches_analytic(self):
        stats, BBeta, _ = self._stats()
        rng = np.random.RandomState(12)
        data = (rng.rand(200) < 0.3).astype(int).tolist()
        a0, b0 = 1.0, 1.0
        proto = stats.BernoulliDistribution(0.5, prior=BBeta(a0, b0))

        result = sample_conjugate_posterior(proto, data, draws=40000, seed=22)
        samples = np.asarray(result.samples, dtype=float)

        ones = int(sum(data))
        zeros = len(data) - ones
        pa, pb = a0 + ones, b0 + zeros
        analytic_mean = pa / (pa + pb)
        analytic_var = pa * pb / ((pa + pb) ** 2 * (pa + pb + 1.0))

        self.assertTrue(np.all((samples > 0.0) & (samples < 1.0)))
        self.assertLess(abs(float(samples.mean()) - analytic_mean), 0.01)
        self.assertLess(abs(float(samples.var()) - analytic_var) / analytic_var, 0.1)

    def test_exponential_gamma_conjugate_matches_analytic(self):
        stats, _, BGamma = self._stats()
        rng = np.random.RandomState(14)
        data = rng.exponential(scale=2.0, size=200).tolist()
        k0, theta0 = 2.0, 1.0
        proto = stats.ExponentialDistribution(1.0, prior=BGamma(k0, theta0))

        result = sample_conjugate_posterior(proto, data, draws=40000, seed=24)
        scales = np.asarray(result.samples, dtype=float)
        rates = 1.0 / scales

        post_k = k0 + len(data)
        post_theta = 1.0 / (1.0 / theta0 + float(sum(data)))
        analytic_mean = post_k * post_theta
        analytic_var = post_k * post_theta * post_theta

        self.assertTrue(np.all(scales > 0.0))
        self.assertLess(abs(float(rates.mean()) - analytic_mean), 4.0 * np.sqrt(analytic_var / len(rates)) + 0.01)
        self.assertLess(abs(float(rates.var()) - analytic_var) / analytic_var, 0.1)

    def test_binomial_beta_conjugate_matches_analytic(self):
        stats, BBeta, _ = self._stats()
        rng = np.random.RandomState(15)
        trials = 5
        data = rng.binomial(n=trials, p=0.4, size=250).tolist()
        a0, b0 = 2.0, 3.0
        proto = stats.BinomialDistribution(0.5, trials, prior=BBeta(a0, b0))

        result = sample_conjugate_posterior(proto, data, draws=40000, seed=25)
        samples = np.asarray(result.samples, dtype=float)

        successes = float(sum(data))
        failures = float(len(data) * trials - successes)
        pa, pb = a0 + successes, b0 + failures
        analytic_mean = pa / (pa + pb)
        analytic_var = pa * pb / ((pa + pb) ** 2 * (pa + pb + 1.0))

        self.assertTrue(np.all((samples > 0.0) & (samples < 1.0)))
        self.assertLess(abs(float(samples.mean()) - analytic_mean), 0.01)
        self.assertLess(abs(float(samples.var()) - analytic_var) / analytic_var, 0.1)

        rebuilt = sample_conjugate_posterior(proto, data, draws=5, seed=26, return_distributions=True)
        for dist in rebuilt.samples:
            self.assertIsInstance(dist, stats.BinomialDistribution)
            self.assertEqual(dist.n, trials)

    def test_geometric_beta_conjugate_matches_analytic(self):
        stats, BBeta, _ = self._stats()
        rng = np.random.RandomState(16)
        data = rng.geometric(p=0.35, size=250).tolist()
        a0, b0 = 2.0, 3.0
        proto = stats.GeometricDistribution(0.5, prior=BBeta(a0, b0))

        result = sample_conjugate_posterior(proto, data, draws=40000, seed=27)
        samples = np.asarray(result.samples, dtype=float)

        pa = a0 + len(data)
        pb = b0 + float(sum(data) - len(data))
        analytic_mean = pa / (pa + pb)
        analytic_var = pa * pb / ((pa + pb) ** 2 * (pa + pb + 1.0))

        self.assertTrue(np.all((samples > 0.0) & (samples < 1.0)))
        self.assertLess(abs(float(samples.mean()) - analytic_mean), 0.01)
        self.assertLess(abs(float(samples.var()) - analytic_var) / analytic_var, 0.1)

    def test_gaussian_normalgamma_conjugate_centers_on_data(self):
        stats, _, _ = self._stats()
        rng = np.random.RandomState(13)
        data = rng.normal(2.0, 1.5, size=300).tolist()
        proto = stats.GaussianDistribution(0.0, 1.0)

        result = sample_conjugate_posterior(proto, data, draws=40000, seed=23)
        mus = np.asarray([t[0] for t in result.samples], dtype=float)
        s2s = np.asarray([t[1] for t in result.samples], dtype=float)

        self.assertTrue(np.all(s2s > 0.0))
        self.assertLess(abs(float(mus.mean()) - float(np.mean(data))), 0.05)
        self.assertLess(abs(float(s2s.mean()) - float(np.var(data))) / float(np.var(data)), 0.15)

    def test_conjugate_unsupported_leaf_raises(self):
        stats, _, _ = self._stats()
        with self.assertRaisesRegex(NotImplementedError, "Gaussian, Poisson, Exponential, Bernoulli"):
            sample_conjugate_posterior(stats.CategoricalDistribution({"a": 0.5, "b": 0.5}), ["a", "b"], draws=5)


if __name__ == "__main__":
    unittest.main()
