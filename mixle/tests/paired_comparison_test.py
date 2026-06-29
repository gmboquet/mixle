"""Paired-comparison models: Thurstone-Mosteller (Gaussian) and Bradley-Terry with ties (Davidson, Rao-Kupper)."""

import unittest

import numpy as np

from mixle.stats import DavidsonDistribution, RaoKupperDistribution, ThurstoneMostellerDistribution


class ThurstoneMostellerTest(unittest.TestCase):
    def test_density_sums_to_one(self):
        d = ThurstoneMostellerDistribution([1.5, 0.5, -0.5, -1.5])
        tot = sum(d.density((w, ell)) for w in range(4) for ell in range(4) if w != ell)
        self.assertAlmostEqual(tot, 1.0, places=9)

    def test_probit_win_probability(self):
        from scipy.special import ndtr

        d = ThurstoneMostellerDistribution([1.0, -1.0])
        # P(0 beats 1) = Phi((mu0-mu1)/sqrt2) = Phi(2/sqrt2)
        self.assertAlmostEqual(np.exp(d.log_density((0, 1)) + d.log_pairs), ndtr(2.0 / np.sqrt(2.0)), places=9)

    def test_mu_recovery(self):
        true = ThurstoneMostellerDistribution([2.0, 1.0, 0.0, -1.0, -2.0])
        samp = true.sampler(seed=1).sample(30000)
        acc = true.estimator().accumulator_factory().make()
        acc.seq_update(true.dist_to_encoder().seq_encode(samp), np.ones(len(samp)), None)
        fit = true.estimator().estimate(len(samp), acc.value())
        np.testing.assert_allclose(fit.mu, true.mu, atol=0.2)
        self.assertEqual(list(np.argsort(-fit.mu)), list(np.argsort(-true.mu)))


class DavidsonTest(unittest.TestCase):
    def test_density_sums_to_one(self):
        d = DavidsonDistribution([1.0, 0.0, -1.0], nu=1.5)
        tot = sum(d.density((i, j, o)) for i in range(3) for j in range(i + 1, 3) for o in range(3))
        self.assertAlmostEqual(tot, 1.0, places=9)

    def test_canonicalization_flips_outcome(self):
        d = DavidsonDistribution([1.0, 0.0, -1.0], nu=1.0)
        self.assertAlmostEqual(d.log_density((1, 0, 0)), d.log_density((0, 1, 1)))  # i-wins(1,0) == j-wins(0,1)
        self.assertAlmostEqual(d.log_density((1, 0, 2)), d.log_density((0, 1, 2)))  # ties symmetric

    def test_recovers_worths_and_tie_parameter(self):
        true = DavidsonDistribution([2.0, 0.7, -0.7, -2.0], nu=1.2)
        samp = true.sampler(seed=2).sample(40000)
        acc = true.estimator().accumulator_factory().make()
        acc.seq_update(true.dist_to_encoder().seq_encode(samp), np.ones(len(samp)), None)
        fit = true.estimator().estimate(len(samp), acc.value())
        np.testing.assert_allclose(fit.log_w, true.log_w, atol=0.2)
        self.assertAlmostEqual(fit.nu, 1.2, delta=0.2)

    def test_validation(self):
        with self.assertRaises(ValueError):
            DavidsonDistribution([0.0, 0.0], nu=-0.5)


class RaoKupperTest(unittest.TestCase):
    def test_density_sums_to_one(self):
        d = RaoKupperDistribution([1.0, 0.0, -1.0], nu=1.8)
        tot = sum(d.density((i, j, o)) for i in range(3) for j in range(i + 1, 3) for o in range(3))
        self.assertAlmostEqual(tot, 1.0, places=9)

    def test_recovers_worths_and_threshold(self):
        true = RaoKupperDistribution([2.0, 0.7, -0.7, -2.0], nu=1.6)
        samp = true.sampler(seed=3).sample(40000)
        acc = true.estimator().accumulator_factory().make()
        acc.seq_update(true.dist_to_encoder().seq_encode(samp), np.ones(len(samp)), None)
        fit = true.estimator().estimate(len(samp), acc.value())
        np.testing.assert_allclose(fit.log_w, true.log_w, atol=0.2)
        self.assertAlmostEqual(fit.nu, 1.6, delta=0.25)

    def test_validation(self):
        with self.assertRaises(ValueError):
            RaoKupperDistribution([0.0, 0.0], nu=0.5)  # nu must be >= 1


if __name__ == "__main__":
    unittest.main()
