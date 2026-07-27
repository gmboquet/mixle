"""Hurdle count model: a zero hurdle + zero-truncated base, fit in closed form (no latent EM)."""

import unittest

import numpy as np

from mixle.inference import estimate
from mixle.stats import (
    GaussianDistribution,
    GeometricDistribution,
    HurdleDistribution,
    HurdleEstimator,
    PointMassDistribution,
    PoissonDistribution,
    PoissonEstimator,
)


class HurdleDistributionTest(unittest.TestCase):
    def setUp(self):
        self.d = HurdleDistribution(PoissonDistribution(3.0), 0.4)

    def test_density_normalizes_and_hurdle_at_zero(self):
        mass = np.array([self.d.density(int(k)) for k in range(300)])
        self.assertAlmostEqual(mass.sum(), 1.0, places=9)
        self.assertAlmostEqual(self.d.density(0), 0.4, places=12)

    def test_seq_matches_scalar(self):
        xs = [0, 1, 2, 5, 0, 3]
        enc = self.d.dist_to_encoder().seq_encode(xs)
        np.testing.assert_allclose(self.d.seq_log_density(enc), [self.d.log_density(v) for v in xs])

    def test_sampler_zero_rate_and_positivity(self):
        s = np.array(self.d.sampler(seed=0).sample(20000))
        self.assertAlmostEqual(np.mean(s == 0), 0.4, delta=0.02)
        self.assertTrue(np.all(s[s != 0] > 0))

    def test_truncated_mle_recovers_true_base_not_positive_mean(self):
        # the count part is the zero-truncated MLE: it must recover lambda=3.0, NOT the positives'
        # mean (~3.16) that a naive "fit base to positives" would give.
        data = list(self.d.sampler(seed=1).sample(50000))
        est = estimate(data, self.d.estimator())
        self.assertAlmostEqual(est.pi, 0.4, delta=0.02)
        self.assertAlmostEqual(est.base.lam, 3.0, delta=0.06)
        # and the fitted model matches the data
        for k in range(5):
            self.assertAlmostEqual(est.density(k), float(np.mean(np.array(data) == k)), delta=0.01)

    def test_base_without_zero_mass_has_no_truncation(self):
        d = HurdleDistribution(GeometricDistribution(0.3), 0.0)  # geometric on {1,2,...}: P(0)=0
        self.assertAlmostEqual(d._log_renorm, 0.0)
        mass = np.array([d.density(int(k)) for k in range(1, 300)])
        self.assertAlmostEqual(mass.sum(), 1.0, places=9)

    def test_requires_declared_atomic_nonnegative_count_support(self):
        with self.assertRaises(TypeError):
            HurdleDistribution(GaussianDistribution(0.0, 1.0), 0.5)
        with self.assertRaises(TypeError):
            HurdleDistribution(PointMassDistribution(-1), 0.5)
        with self.assertRaises(ValueError):
            HurdleDistribution(PointMassDistribution(0), 0.5)

    def test_probability_boundaries_are_exact(self):
        all_zero = HurdleDistribution(PoissonDistribution(2.0), 1.0)
        self.assertEqual(all_zero.log_density(0), 0.0)
        self.assertEqual(all_zero.log_density(1), -np.inf)
        self.assertEqual(all_zero.sampler(seed=3).sample(20), [0] * 20)
        with self.assertRaises(ValueError):
            HurdleDistribution(PoissonDistribution(2.0), np.nan)

    def test_estimator_validates_controls_and_statistics(self):
        with self.assertRaises(ValueError):
            HurdleEstimator(PoissonEstimator(), trunc_max_iter=0)
        with self.assertRaises(ValueError):
            HurdleEstimator(PoissonEstimator(), trunc_threshold=0.0)
        with self.assertRaises(ValueError):
            HurdleEstimator(PoissonEstimator(), pseudo_count=-1.0)
        estimator = HurdleEstimator(PoissonEstimator())
        for stats in [
            ((0.0, 0.0), -1.0, 1.0),
            ((0.0, 0.0), 2.0, 1.0),
            ((0.0, 0.0), np.nan, 1.0),
        ]:
            with self.assertRaises(ValueError):
                estimator.estimate(None, stats)

    def test_estimator_preserves_deterministic_empirical_boundaries(self):
        estimator = HurdleEstimator(PoissonEstimator(), pseudo_count=10.0)
        all_zero = estimator.estimate(None, ((0.0, 0.0), 4.0, 4.0))
        no_zero = estimator.estimate(None, ((4.0, 8.0), 0.0, 4.0))
        self.assertEqual(all_zero.pi, 1.0)
        self.assertEqual(no_zero.pi, 0.0)

    def test_accumulator_rejects_invalid_evidence_before_mutation(self):
        acc = self.d.estimator().accumulator_factory().make()
        before = acc.value()
        with self.assertRaises(ValueError):
            acc.update(-1, 1.0, self.d)
        with self.assertRaises(ValueError):
            acc.update(1, np.inf, self.d)
        self.assertEqual(acc.value(), before)


if __name__ == "__main__":
    unittest.main()
