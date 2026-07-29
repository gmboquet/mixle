"""Low-rank assignment / first-order Fourier permutation model: normalization, fit, marginals."""

import itertools
import math
import unittest

import numpy as np

from mixle.stats import LowRankPermutationDistribution
from mixle.stats.rankings._permutation_kernels import ryser_log_permanent, sinkhorn_bethe
from mixle.stats.rankings.low_rank_permutation import (
    LowRankPermutationDataEncoder,
    LowRankPermutationEstimator,
)


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
        self.assertTrue(fit.fit_diagnostics.converged)
        self.assertEqual(fit.fit_diagnostics.marginal_algorithm, "exact_log_permanent")
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

    def test_bethe_branch_is_not_exposed_as_a_probability_distribution(self):
        with self.assertRaisesRegex(ValueError, "cannot normalize a probability distribution"):
            LowRankPermutationDistribution(np.ones((4, 1)), np.ones((4, 1)), max_exact=3)

    def test_exact_marginals_are_distinct_from_sinkhorn_relaxation(self):
        dist = LowRankPermutationDistribution(np.asarray([[2.0], [0.0]]), np.asarray([[1.0], [0.0]]))
        exact = dist.marginals()
        relaxation = dist.sinkhorn_relaxation()
        self.assertAlmostEqual(exact[0, 0], math.exp(2.0) / (math.exp(2.0) + 1.0), places=12)
        self.assertGreater(abs(exact[0, 0] - relaxation[0, 0]), 0.1)
        self.assertTrue(dist.computation_diagnostics.normalizer_exact)
        self.assertTrue(dist.computation_diagnostics.marginals_exact)
        self.assertTrue(dist.computation_diagnostics.sampler_exact)
        self.assertIn("transport", dist.computation_diagnostics.optional_relaxation)

    def test_distribution_owns_finite_positive_rank_factors(self):
        u = np.asarray([[1.0], [0.0]])
        v = np.asarray([[1.0], [0.0]])
        dist = LowRankPermutationDistribution(u, v)
        u[0, 0] = 99.0
        v[0, 0] = 99.0
        self.assertEqual(dist.u[0, 0], 1.0)
        self.assertEqual(dist.v[0, 0], 1.0)
        with self.assertRaises(ValueError):
            dist.u[0, 0] = 0.0
        with self.assertRaises(ValueError):
            LowRankPermutationDistribution(np.zeros((3, 0)), np.zeros((3, 0)))
        with self.assertRaises(ValueError):
            LowRankPermutationDistribution(np.full((2, 1), np.nan), np.zeros((2, 1)))
        with self.assertRaises(ValueError):
            LowRankPermutationDistribution(np.ones((2, 3)), np.ones((2, 3)))

    def test_all_boundaries_validate_support_and_dimension(self):
        dist = LowRankPermutationDistribution(np.ones((3, 1)), np.ones((3, 1)))
        malformed = ([0, 0, 1], [0, 1], [0, 1, 4], [0.5, 1.0, 2.0])
        for value in malformed:
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                dist.log_density(value)
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                dist.seq_log_density(np.asarray([value]))
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                dist.dist_to_encoder().seq_encode([value])
        self.assertEqual(LowRankPermutationDataEncoder(3), LowRankPermutationDataEncoder(3))
        self.assertNotEqual(LowRankPermutationDataEncoder(3), LowRankPermutationDataEncoder(4))
        self.assertIn("dim=3", str(LowRankPermutationDataEncoder(3)))

    def test_accumulator_validates_evidence_and_copies_state(self):
        acc = LowRankPermutationEstimator(3, rank=1).accumulator_factory().make()
        with self.assertRaises(ValueError):
            acc.update([0, 0, 1], 1.0, None)
        with self.assertRaises(ValueError):
            acc.seq_update(np.asarray([[0, 1, 2]]), np.asarray([-1.0]), None)
        with self.assertRaises(ValueError):
            acc.seq_update(np.asarray([[0, 1, 2]]), np.asarray([1.0, 2.0]), None)
        acc.update([0, 1, 2], 1.0, None)
        receipt = acc.value()
        receipt[1][:] = 0.0
        self.assertEqual(acc.value()[1].sum(), 3.0)
        malformed = np.zeros((3, 3))
        malformed[0, 0] = 3.0
        with self.assertRaises(ValueError):
            acc.from_value((1.0, malformed))

    def test_estimator_controls_pseudo_count_and_sample_size(self):
        invalid = (
            {"dim": 1},
            {"dim": 3.5},
            {"dim": 3, "rank": 0},
            {"dim": 3, "rank": 4},
            {"dim": 3, "max_exact": 2},
            {"dim": 3, "sinkhorn_iter": 0},
            {"dim": 3, "max_iter": 0},
            {"dim": 3, "lr": 0.0},
            {"dim": 3, "tol": 0.0},
            {"dim": 3, "pseudo_count": -1.0},
            {"dim": 3, "require_convergence": "false"},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=repr(kwargs)), self.assertRaises((TypeError, ValueError)):
                LowRankPermutationEstimator(**kwargs)

        uniform = LowRankPermutationDistribution(np.zeros((3, 1)), np.zeros((3, 1)))
        fitted = uniform.estimator(pseudo_count=2.0).estimate(None, (0.0, np.zeros((3, 3))))
        self.assertTrue(fitted.fit_diagnostics.regularized)
        np.testing.assert_allclose(fitted.marginals(), np.full((3, 3), 1.0 / 3.0))
        with self.assertRaises(ValueError):
            uniform.sampler().sample(-1)
        with self.assertRaises(ValueError):
            uniform.sampler().sample(1.5)


if __name__ == "__main__":
    unittest.main()
