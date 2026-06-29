"""WS-2: RicianDistribution (Rice fading/MRI envelope family), cross-checked vs scipy."""

import unittest

import numpy as np
import scipy.stats as ss

import mixle
from mixle.capability import HasCDF, HasMoments
from mixle.stats import RicianDistribution as R


class RicianTest(unittest.TestCase):
    def test_density_cdf_moments_match_scipy(self):
        for nu, sigma in [(2.0, 1.5), (0.0, 1.0), (5.0, 0.8), (1.0, 2.0)]:
            d, fr = R(nu, sigma), ss.rice(nu / sigma, scale=sigma)
            xs = np.array([0.4, 1.2, 2.5, 5.0])
            with self.subTest(nu=nu, sigma=sigma):
                self.assertTrue(np.allclose([d.log_density(x) for x in xs], fr.logpdf(xs)))
                self.assertTrue(np.allclose(d.seq_log_density(xs), fr.logpdf(xs)))
                self.assertTrue(np.allclose([d.cdf(x) for x in xs], fr.cdf(xs)))
                self.assertAlmostEqual(d.mean(), float(fr.mean()), places=7)
                self.assertAlmostEqual(d.variance(), float(fr.var()), places=7)

    def test_rayleigh_special_case(self):
        xs = np.array([0.3, 1.0, 2.0])
        self.assertTrue(np.allclose([R(0.0, 1.0).log_density(x) for x in xs], ss.rayleigh(scale=1.0).logpdf(xs)))

    def test_quantile_inverts_cdf(self):
        d = R(2.0, 1.5)
        for q in (0.05, 0.5, 0.95):
            self.assertAlmostEqual(d.cdf(d.quantile(q)), q, places=6)

    def test_support_guard(self):
        self.assertEqual(R(2.0, 1.5).log_density(-1.0), -np.inf)
        self.assertEqual(R(2.0, 1.5).cdf(-1.0), 0.0)

    def test_mom_recovers_params(self):
        true = R(2.0, 1.5)
        data = np.asarray(true.sampler(seed=0).sample(400_000))
        est = true.estimator()
        acc = est.accumulator_factory().make()
        acc.seq_update(true.dist_to_encoder().seq_encode(data), np.ones(len(data)), None)
        m = est.estimate(len(data), acc.value())
        self.assertAlmostEqual(m.nu, 2.0, delta=0.1)
        self.assertAlmostEqual(m.sigma, 1.5, delta=0.1)

    def test_capabilities(self):
        d = R(2.0, 1.5)
        self.assertTrue(mixle.supports(d, HasCDF))
        self.assertTrue(mixle.supports(d, HasMoments))


if __name__ == "__main__":
    unittest.main()
