"""Low-rank assignment / first-order Fourier permutation model: normalization, fit, marginals."""

import itertools
import math
import unittest

import numpy as np

from mixle.stats import LowRankPermutationDistribution
from mixle.stats.rankings._permutation_kernels import ryser_log_permanent, sinkhorn_bethe


class KernelTest(unittest.TestCase):
    def test_ryser_permanent_of_ones_is_factorial(self):
        for n in (3, 5, 7):
            self.assertAlmostEqual(math.exp(ryser_log_permanent(np.ones((n, n)))), math.factorial(n), places=3)

    def test_sinkhorn_returns_doubly_stochastic(self):
        rng = np.random.RandomState(0)
        s = rng.randn(8, 8)
        p, _ = sinkhorn_bethe(np.ascontiguousarray(s), 300)
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(p.sum(axis=0), 1.0, atol=1e-6)


class LowRankPermutationTest(unittest.TestCase):
    def test_density_sums_to_one(self):
        rng = np.random.RandomState(1)
        d = LowRankPermutationDistribution(rng.randn(6, 2) * 0.7, rng.randn(6, 2) * 0.7)
        self.assertAlmostEqual(sum(d.density(list(p)) for p in itertools.permutations(range(6))), 1.0, places=9)

    def test_seq_matches_scalar(self):
        rng = np.random.RandomState(2)
        d = LowRankPermutationDistribution(rng.randn(5, 2) * 0.6, rng.randn(5, 2) * 0.6)
        perms = np.array(list(itertools.permutations(range(5))))
        np.testing.assert_allclose(d.seq_log_density(perms), [d.log_density(p) for p in perms], atol=1e-12)

    def test_fit_recovers_marginals_and_beats_uniform(self):
        rng = np.random.RandomState(3)
        true = LowRankPermutationDistribution(rng.randn(6, 2) * 0.8, rng.randn(6, 2) * 0.8)
        enc = true.dist_to_encoder().seq_encode(true.sampler(seed=1).sample(6000))
        acc = true.estimator().accumulator_factory().make()
        acc.seq_update(enc, np.ones(enc.shape[0]), None)
        fit = true.estimator().estimate(enc.shape[0], acc.value())
        m_emp = acc.counts / acc.count
        self.assertLess(float(np.abs(fit.marginals() - m_emp).sum()), 0.3)  # fitted marginals match data
        test = true.sampler(seed=99).sample(400)
        ll_fit = np.mean([fit.log_density(t) for t in test])
        self.assertGreater(ll_fit, -math.log(math.factorial(6)))  # beats the uniform distribution

    def test_validation(self):
        with self.assertRaises(ValueError):
            LowRankPermutationDistribution(np.zeros((3, 2)), np.zeros((4, 2)))  # shape mismatch
        with self.assertRaises(ValueError):
            LowRankPermutationDistribution(np.zeros((1, 2)), np.zeros((1, 2)))  # n < 2


if __name__ == "__main__":
    unittest.main()
