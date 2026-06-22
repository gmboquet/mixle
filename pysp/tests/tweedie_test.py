"""Tests for the Tweedie (compound Poisson-Gamma, 1<p<2) distribution.

No external reference is available, so correctness is checked by self-consistency of the series
density (numerical normalization + moments matching mu / phi*mu**p) and against the *exact*
compound-Poisson-Gamma sampler.
"""

import math
import unittest

import numpy as np
from scipy import integrate

from pysp.inference.estimation import optimize
from pysp.stats import TweedieDistribution, TweedieEstimator


class TweedieDistributionTest(unittest.TestCase):
    def test_density_normalizes_and_matches_moments(self):
        for mu, phi, p in [(2.0, 1.0, 1.5), (5.0, 0.5, 1.3), (1.0, 2.0, 1.7)]:
            with self.subTest(mu=mu, phi=phi, p=p):
                d = TweedieDistribution(mu, phi, p)
                p0 = math.exp(-d.lam)
                hi = mu * 60.0
                mass, _ = integrate.quad(lambda y, d=d: math.exp(d.log_density(y)), 0.0, hi, limit=400)
                mean_int, _ = integrate.quad(lambda y, d=d: y * math.exp(d.log_density(y)), 0.0, hi, limit=400)
                m2_int, _ = integrate.quad(lambda y, d=d: y * y * math.exp(d.log_density(y)), 0.0, hi, limit=400)
                self.assertAlmostEqual(p0 + mass, 1.0, places=3)  # total probability
                self.assertAlmostEqual(mean_int, mu, places=2)  # E[Y] = mu
                self.assertAlmostEqual(m2_int - mu * mu, phi * mu**p, places=2)  # Var = phi*mu**p

    def test_point_mass_at_zero(self):
        d = TweedieDistribution(3.0, 1.0, 1.5)
        self.assertAlmostEqual(d.log_density(0.0), -d.lam, places=12)
        self.assertEqual(d.log_density(-1.0), -np.inf)

    def test_seq_log_density_matches_scalar(self):
        d = TweedieDistribution(2.5, 0.8, 1.4)
        xs = np.array([0.0, 0.3, 1.0, 2.5, 7.0])
        enc = d.dist_to_encoder().seq_encode(list(xs))
        seq = np.asarray(d.seq_log_density(enc))
        scalar = np.array([d.log_density(float(x)) for x in xs])
        np.testing.assert_allclose(seq, scalar, atol=1e-9)

    def test_sampler_moments_and_zero_fraction(self):
        d = TweedieDistribution(2.0, 1.0, 1.5)
        y = d.sampler(0).sample(40000)
        self.assertGreaterEqual(float(np.min(y)), 0.0)
        self.assertAlmostEqual(float(np.mean(y)), 2.0, delta=0.1)  # E[Y] = mu
        self.assertAlmostEqual(float(np.var(y)), 1.0 * 2.0**1.5, delta=0.4)  # Var = phi*mu**p
        self.assertAlmostEqual(float(np.mean(y == 0.0)), math.exp(-d.lam), delta=0.02)  # P(Y=0)

    def test_estimator_recovers_parameters(self):
        true = TweedieDistribution(4.0, 0.7, 1.5)
        data = list(true.sampler(1).sample(30000))
        fit = optimize(data, TweedieEstimator(p=1.5), max_its=1, rng=np.random.RandomState(0), out=None)
        self.assertAlmostEqual(fit.mu, 4.0, delta=0.15)
        self.assertAlmostEqual(fit.phi, 0.7, delta=0.2)


if __name__ == "__main__":
    unittest.main()
