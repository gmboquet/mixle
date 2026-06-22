"""Hurdle count model: a zero hurdle + zero-truncated base, fit in closed form (no latent EM)."""

import unittest

import numpy as np

from pysp.inference import estimate
from pysp.stats import (
    GeometricDistribution,
    HurdleDistribution,
    PoissonDistribution,
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


if __name__ == "__main__":
    unittest.main()
