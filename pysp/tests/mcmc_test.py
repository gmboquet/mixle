import unittest

import numpy as np

from pysp.stats import GammaDistribution, GaussianDistribution
from pysp.utils.mcmc import (
    AdaptiveCovarianceProposal,
    AdaptiveRandomWalkProposal,
    BlockProposal,
    IndependentProposal,
    LangevinProposal,
    MixtureProposal,
    RandomWalkProposal,
    hamiltonian_monte_carlo,
    metropolis_hastings,
    metropolis_within_gibbs,
    posterior_predictive,
    sample_distribution,
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
        self.assertEqual(summary['num_samples'], 2500)
        self.assertLess(abs(summary['mean'] - 1.0), 0.15)
        self.assertGreater(summary['ess'], 50.0)
        self.assertGreater(summary['mcse'], 0.0)

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
        self.assertEqual(set(result.acceptance_rate_by_label), {'hmc'})
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
        self.assertEqual(summary['num_samples'], 3000)
        self.assertEqual(summary['mean'].shape, (2,))
        self.assertEqual(summary['mcse'].shape, (2,))

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
            x = float(state['x'])
            y = float(state['y'])
            return -0.5 * x * x - 0.5 * (y - 2.0) * (y - 2.0) / 0.5

        result = metropolis_within_gibbs(
            log_target=log_target,
            initial={'x': 0.0, 'y': 0.0},
            proposals={
                'x': BlockProposal('x', RandomWalkProposal(scale=0.8)),
                'y': BlockProposal('y', RandomWalkProposal(scale=0.7)),
            },
            num_samples=2500,
            burn_in=500,
            rng=np.random.RandomState(9),
        )
        xs = np.asarray([sample['x'] for sample in result.samples], dtype=float)
        ys = np.asarray([sample['y'] for sample in result.samples], dtype=float)
        by_label = result.acceptance_rate_by_label

        self.assertEqual(set(by_label), {'x', 'y'})
        self.assertEqual(len(result.accepted), 6000)
        self.assertLess(abs(float(xs.mean())), 0.15)
        self.assertLess(abs(float(xs.var()) - 1.0), 0.35)
        self.assertLess(abs(float(ys.mean()) - 2.0), 0.15)
        self.assertLess(abs(float(ys.var()) - 0.5), 0.2)
        self.assertGreater(by_label['x'], 0.4)
        self.assertGreater(by_label['y'], 0.4)

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

        with self.assertRaisesRegex(ValueError, 'num_samples'):
            sample_distribution(target, 0.0, proposal, num_samples=-1)
        with self.assertRaisesRegex(ValueError, 'burn_in'):
            sample_distribution(target, 0.0, proposal, num_samples=1, burn_in=-1)
        with self.assertRaisesRegex(ValueError, 'thin'):
            sample_distribution(target, 0.0, proposal, num_samples=1, thin=0)
        with self.assertRaisesRegex(ValueError, 'initial state'):
            sample_distribution(GammaDistribution(2.0, 1.0), -1.0, proposal, num_samples=1)
        with self.assertRaisesRegex(ValueError, 'step_size'):
            hamiltonian_monte_carlo(lambda x: -float(x) * float(x), lambda x: -2.0 * float(x),
                                    0.0, num_samples=1, step_size=0.0, num_steps=1)
        with self.assertRaisesRegex(ValueError, 'num_steps'):
            hamiltonian_monte_carlo(lambda x: -float(x) * float(x), lambda x: -2.0 * float(x),
                                    0.0, num_samples=1, step_size=0.1, num_steps=0)
        with self.assertRaisesRegex(ValueError, 'grad_log_target shape'):
            hamiltonian_monte_carlo(lambda x: -0.5 * float(x) * float(x), lambda x: [0.0, 0.0],
                                    0.0, num_samples=1, step_size=0.1, num_steps=1)
        with self.assertRaisesRegex(ValueError, 'adapt_after'):
            AdaptiveCovarianceProposal(adapt_after=1)
        with self.assertRaisesRegex(ValueError, 'initial_covariance'):
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


if __name__ == '__main__':
    unittest.main()
