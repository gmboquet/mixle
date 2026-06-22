"""WS-2: NakagamiDistribution (fading envelope family), cross-checked vs scipy."""

import unittest

import numpy as np
import scipy.stats as ss

import pysp
from pysp.capability import HasCDF, HasMoments
from pysp.stats import NakagamiDistribution as N


class NakagamiTest(unittest.TestCase):
    def test_density_cdf_moments_match_scipy(self):
        for m, omega in [(2.5, 3.0), (1.0, 2.0), (0.5, 1.0), (4.0, 0.5)]:
            d, fr = N(m, omega), ss.nakagami(m, scale=np.sqrt(omega))
            xs = np.array([0.2, 0.8, 1.5, 3.0])
            with self.subTest(m=m, omega=omega):
                self.assertTrue(np.allclose([d.log_density(x) for x in xs], fr.logpdf(xs)))
                self.assertTrue(np.allclose(d.seq_log_density(xs), fr.logpdf(xs)))
                self.assertTrue(np.allclose([d.cdf(x) for x in xs], fr.cdf(xs)))
                self.assertAlmostEqual(d.mean(), float(fr.mean()), places=7)
                self.assertAlmostEqual(d.variance(), float(fr.var()), places=7)

    def test_quantile_inverts_cdf(self):
        d = N(2.5, 3.0)
        for q in (0.05, 0.25, 0.5, 0.75, 0.95):
            self.assertAlmostEqual(d.cdf(d.quantile(q)), q, places=7)

    def test_rayleigh_special_case(self):
        xs = np.array([0.3, 1.0, 2.0])
        self.assertTrue(np.allclose([N(1.0, 2.0).log_density(x) for x in xs], ss.rayleigh(scale=1.0).logpdf(xs)))

    def test_support_guard(self):
        d = N(2.0, 1.0)
        self.assertEqual(d.log_density(-1.0), -np.inf)
        self.assertEqual(d.cdf(-1.0), 0.0)

    def test_mom_recovers_params(self):
        true = N(2.5, 3.0)
        data = np.asarray(true.sampler(seed=0).sample(300_000))
        est = true.estimator()
        acc = est.accumulator_factory().make()
        acc.seq_update(true.dist_to_encoder().seq_encode(data), np.ones(len(data)), None)
        mm = est.estimate(len(data), acc.value())
        self.assertAlmostEqual(mm.m, 2.5, delta=0.1)
        self.assertAlmostEqual(mm.omega, 3.0, delta=0.1)

    def test_capabilities(self):
        d = N(2.5, 3.0)
        self.assertTrue(pysp.supports(d, HasCDF))
        self.assertTrue(pysp.supports(d, HasMoments))


if __name__ == "__main__":
    unittest.main()
