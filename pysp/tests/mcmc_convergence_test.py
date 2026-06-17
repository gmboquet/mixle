"""Tests for multi-chain MCMC convergence diagnostics (Gelman-Rubin R-hat + run_chains)."""

import unittest

import numpy as np

from pysp.utils.mcmc import (
    MCMCResult,
    RandomWalkProposal,
    gelman_rubin,
    metropolis_hastings,
    run_chains,
)


def _gaussian_log_target(mu=0.0, sigma=1.0):
    return lambda x: -0.5 * ((float(np.asarray(x).reshape(-1)[0]) - mu) / sigma) ** 2


class GelmanRubinTestCase(unittest.TestCase):
    def test_well_mixed_chains_have_rhat_near_one(self):
        # Independent chains from the same standard-normal target should converge: R-hat ~ 1.
        log_target = _gaussian_log_target()
        proposal = RandomWalkProposal(scale=1.5)
        results = []
        for seed in range(4):
            rng = np.random.RandomState(seed)
            results.append(
                metropolis_hastings(
                    log_target, initial=np.array([0.0]), proposal=proposal, num_samples=4000, burn_in=1000, rng=rng
                )
            )
        rhat = gelman_rubin(results)
        self.assertIsInstance(rhat, float)
        self.assertLess(rhat, 1.05)

    def test_stuck_chains_flag_nonconvergence(self):
        # Chains pinned to two well-separated modes never mix: R-hat must be well above 1.
        a = MCMCResult(
            samples=list(np.random.RandomState(0).normal(-10.0, 0.1, size=500)),
            log_probs=np.zeros(500),
            accepted=np.ones(500, dtype=bool),
        )
        b = MCMCResult(
            samples=list(np.random.RandomState(1).normal(10.0, 0.1, size=500)),
            log_probs=np.zeros(500),
            accepted=np.ones(500, dtype=bool),
        )
        rhat = gelman_rubin([a, b])
        self.assertGreater(rhat, 1.5)

    def test_requires_two_chains(self):
        a = MCMCResult(samples=[0.0, 1.0], log_probs=np.zeros(2), accepted=np.ones(2, dtype=bool))
        with self.assertRaises(ValueError):
            gelman_rubin([a])

    def test_unequal_length_chains_truncate(self):
        rng = np.random.RandomState(3)
        a = MCMCResult(samples=list(rng.normal(0, 1, 300)), log_probs=np.zeros(300), accepted=np.ones(300, bool))
        b = MCMCResult(samples=list(rng.normal(0, 1, 500)), log_probs=np.zeros(500), accepted=np.ones(500, bool))
        rhat = gelman_rubin([a, b])  # should truncate to 300, not raise
        self.assertLess(rhat, 1.2)

    def test_vector_parameter_rhat_is_per_dimension(self):
        rng = np.random.RandomState(7)
        chains = [
            MCMCResult(
                samples=list(rng.normal(0, 1, size=(400, 3))), log_probs=np.zeros(400), accepted=np.ones(400, bool)
            )
            for _ in range(3)
        ]
        rhat = gelman_rubin(chains)
        self.assertEqual(np.asarray(rhat).shape, (3,))
        self.assertTrue(np.all(np.asarray(rhat) < 1.1))


class RunChainsTestCase(unittest.TestCase):
    def test_run_chains_returns_results_and_rhat(self):
        log_target = _gaussian_log_target()
        proposal = RandomWalkProposal(scale=1.5)
        rng = np.random.RandomState(0)
        results, rhat = run_chains(
            metropolis_hastings,
            num_chains=4,
            initials=lambda r: np.array([r.uniform(-5.0, 5.0)]),  # overdispersed starts
            rng=rng,
            log_target=log_target,
            proposal=proposal,
            num_samples=3000,
            burn_in=1000,
        )
        self.assertEqual(len(results), 4)
        self.assertTrue(all(isinstance(r, MCMCResult) for r in results))
        self.assertLess(rhat, 1.1)

    def test_run_chains_is_reproducible(self):
        log_target = _gaussian_log_target()
        proposal = RandomWalkProposal(scale=1.0)
        kw = dict(
            num_chains=2,
            initials=[np.array([0.0]), np.array([1.0])],
            log_target=log_target,
            proposal=proposal,
            num_samples=500,
            burn_in=100,
        )
        r1, h1 = run_chains(metropolis_hastings, rng=np.random.RandomState(42), **kw)
        r2, h2 = run_chains(metropolis_hastings, rng=np.random.RandomState(42), **kw)
        np.testing.assert_array_equal(r1[0].sample_array(), r2[0].sample_array())
        self.assertEqual(h1, h2)

    def test_run_chains_requires_two_chains(self):
        with self.assertRaises(ValueError):
            run_chains(
                metropolis_hastings,
                num_chains=1,
                initials=[np.array([0.0])],
                log_target=_gaussian_log_target(),
                proposal=RandomWalkProposal(scale=1.0),
                num_samples=10,
            )


if __name__ == "__main__":
    unittest.main()
