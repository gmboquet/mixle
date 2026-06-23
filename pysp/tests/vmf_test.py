"""Tests for the von Mises-Fisher distribution (pysp.stats.directional.von_mises_fisher).

Normalization is checked against exact references: arbitrary-precision Bessel
functions (mpmath) for log I_v, the d=3 closed form c = k/(4 pi sinh k), and
direct quadrature on the circle. The sampler is checked against the exact
mean resultant length A_d(kappa) and the estimator against parameter recovery.
"""

import unittest

import mpmath
import numpy as np

from pysp.stats import VonMisesFisherDistribution, VonMisesFisherEstimator
from pysp.stats.directional.von_mises_fisher import lniv


class BesselTestCase(unittest.TestCase):
    def test_lniv_matches_mpmath(self):
        # spans the scaled-Bessel regime and the large-order underflow regime
        cases = [
            (0.5, 1.0),
            (0.5, 50.0),
            (1.0, 1e-6),
            (4.5, 2.0),
            (4.5, 800.0),
            (49.0, 1.0),
            (49.0, 3000.0),
            (200.0, 10.0),
            (500.0, 1.0),
            (1000.0, 10.0),
            (300.0, 0.01),
        ]
        for v, z in cases:
            exact = float(mpmath.log(mpmath.besseli(v, z)))
            got = float(lniv(v, np.log(z)))
            rel = abs(got - exact) / max(1.0, abs(exact))
            self.assertLess(rel, 1.0e-6, "v=%g z=%g exact=%g got=%g" % (v, z, exact, got))


class NormalizationTestCase(unittest.TestCase):
    def test_d3_closed_form(self):
        # c_3(k) = k / (4 pi sinh k)
        for k in (1e-6, 0.5, 5.0, 50.0, 700.0):
            d = VonMisesFisherDistribution([0.0, 0.0, 1.0], k)
            exact = float(mpmath.log(k / (4 * mpmath.pi * mpmath.sinh(k))))
            self.assertAlmostEqual(d.log_const, exact, places=8, msg="kappa=%g" % k)

    def test_general_dim_against_mpmath(self):
        for dim, k in [(2, 3.0), (5, 0.7), (10, 25.0), (50, 4.0)]:
            mu = np.zeros(dim)
            mu[0] = 1.0
            d = VonMisesFisherDistribution(list(mu), k)
            v = dim / 2.0 - 1.0
            exact = float(
                v * mpmath.log(k) - (dim / 2.0) * mpmath.log(2 * mpmath.pi) - mpmath.log(mpmath.besseli(v, k))
            )
            self.assertAlmostEqual(d.log_const, exact, places=8, msg="dim=%d kappa=%g" % (dim, k))

    def test_circle_quadrature(self):
        th = np.linspace(0, 2 * np.pi, 100001)[:-1]
        pts = np.stack([np.cos(th), np.sin(th)], axis=1)
        for k in (0.5, 5.0):
            d = VonMisesFisherDistribution([1.0, 0.0], k)
            integral = np.exp(d.seq_log_density(pts)).mean() * 2 * np.pi
            self.assertAlmostEqual(integral, 1.0, places=6)

    def test_kappa_zero_is_uniform_on_sphere(self):
        for dim in (2, 3, 7):
            mu = np.ones(dim) / np.sqrt(dim)
            d = VonMisesFisherDistribution(list(mu), 0.0)
            # uniform density = Gamma(d/2) / (2 pi^{d/2}) = 1 / area(S^{d-1})
            from scipy.special import gammaln

            exact = gammaln(dim / 2.0) - np.log(2.0) - (dim / 2.0) * np.log(np.pi)
            self.assertAlmostEqual(d.log_const, exact, places=12)
            self.assertAlmostEqual(d.log_density(mu), exact, places=12)

    def test_seq_scalar_parity(self):
        d = VonMisesFisherDistribution([0.6, 0.8], 3.0)
        x = d.sampler(seed=1).sample(size=50)
        ll_seq = d.seq_log_density(d.dist_to_encoder().seq_encode(x))
        ll_scalar = np.asarray([d.log_density(u) for u in x])
        self.assertTrue(np.allclose(ll_seq, ll_scalar, atol=1.0e-12))


class SamplerTestCase(unittest.TestCase):
    def test_unit_norm_and_moments(self):
        for dim, k in [(3, 5.0), (8, 10.0), (3, 0.5)]:
            mu = np.zeros(dim)
            mu[0] = 1.0
            x = VonMisesFisherDistribution(list(mu), k).sampler(seed=1).sample(size=6000)
            self.assertTrue(np.allclose((x**2).sum(axis=1), 1.0, atol=1.0e-9))
            # E[mu . x] = A_d(kappa) = I_{d/2}(k) / I_{d/2-1}(k)
            a_exact = float(mpmath.besseli(dim / 2.0, k) / mpmath.besseli(dim / 2.0 - 1.0, k))
            self.assertLess(abs((x @ mu).mean() - a_exact), 0.02, "dim=%d kappa=%g" % (dim, k))


class EstimatorTestCase(unittest.TestCase):
    def test_parameter_recovery(self):
        for dim, k in [(3, 5.0), (8, 10.0), (2, 2.0)]:
            mu = np.ones(dim) / np.sqrt(dim)
            data = VonMisesFisherDistribution(list(mu), k).sampler(seed=2).sample(size=6000)
            est = VonMisesFisherEstimator()
            acc = est.accumulator_factory().make()
            acc.seq_update(np.asarray(data), np.ones(len(data)), None)
            f = est.estimate(None, acc.value())
            self.assertLess(abs(f.kappa - k) / k, 0.05)
            self.assertLess(np.linalg.norm(f.mu - mu), 0.05)

    def test_degenerate_resultant(self):
        # all observations identical -> rhat == 1 must yield a finite large kappa
        x = np.tile([0.0, 1.0], (20, 1))
        est = VonMisesFisherEstimator()
        acc = est.accumulator_factory().make()
        acc.seq_update(x, np.ones(20), None)
        f = est.estimate(None, acc.value())
        self.assertTrue(np.isfinite(f.kappa))
        self.assertGreater(f.kappa, 100.0)


if __name__ == "__main__":
    unittest.main()
