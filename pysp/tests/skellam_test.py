"""Tests for the Skellam distribution (difference of two Poissons).

Validates the log-mass against ``scipy.stats.skellam`` (the reference), scalar/sequence
consistency, normalization, the closed-form method-of-moments estimator's parameter recovery,
sampler moments (``E[K]=mu1-mu2``, ``Var[K]=mu1+mu2``), and integer-support validation.
"""

import unittest

import numpy as np
from scipy.stats import skellam as _ref

from pysp.stats import SkellamDistribution, SkellamEstimator
from pysp.utils.estimation import optimize


class SkellamTest(unittest.TestCase):
    def setUp(self):
        self.cases = [(2.0, 1.0), (5.0, 5.0), (0.5, 3.0), (8.0, 0.25)]
        self.ks = np.arange(-25, 26)

    def test_log_density_matches_scipy(self):
        """Scalar log_density equals scipy.stats.skellam.logpmf across signs and rate regimes."""
        for mu1, mu2 in self.cases:
            d = SkellamDistribution(mu1, mu2)
            for k in self.ks:
                with self.subTest(mu1=mu1, mu2=mu2, k=int(k)):
                    self.assertAlmostEqual(d.log_density(int(k)), float(_ref.logpmf(int(k), mu1, mu2)), places=9)

    def test_seq_log_density_matches_scalar_and_scipy(self):
        """Vectorized seq_log_density agrees with the scalar path and scipy."""
        for mu1, mu2 in self.cases:
            d = SkellamDistribution(mu1, mu2)
            enc = d.dist_to_encoder().seq_encode(list(self.ks))
            seq = np.asarray(d.seq_log_density(enc), dtype=np.float64)
            scalar = np.array([d.log_density(int(k)) for k in self.ks])
            ref = _ref.logpmf(self.ks, mu1, mu2)
            np.testing.assert_allclose(seq, scalar, atol=1e-9)
            np.testing.assert_allclose(seq, ref, atol=1e-9)

    def test_normalization(self):
        """The mass over a wide integer window integrates to ~1."""
        for mu1, mu2 in self.cases:
            d = SkellamDistribution(mu1, mu2)
            ks = np.arange(-200, 201)
            total = np.exp(np.asarray(d.seq_log_density(d.dist_to_encoder().seq_encode(list(ks))))).sum()
            self.assertAlmostEqual(total, 1.0, places=6)

    def test_method_of_moments_recovers_parameters(self):
        """The closed-form MoM estimator recovers (mu1, mu2) from a large sample."""
        true = SkellamDistribution(6.0, 2.5)
        data = list(true.sampler(7).sample(200000))
        model = optimize(data, SkellamEstimator(), max_its=1, rng=np.random.RandomState(0), out=None)
        self.assertAlmostEqual(model.mu1, 6.0, delta=0.15)
        self.assertAlmostEqual(model.mu2, 2.5, delta=0.15)

    def test_sampler_moments(self):
        """Sampled mean/variance match E[K]=mu1-mu2 and Var[K]=mu1+mu2."""
        mu1, mu2 = 4.0, 1.5
        draws = np.asarray(SkellamDistribution(mu1, mu2).sampler(11).sample(300000), dtype=np.float64)
        self.assertAlmostEqual(draws.mean(), mu1 - mu2, delta=0.05)
        self.assertAlmostEqual(draws.var(), mu1 + mu2, delta=0.10)

    def test_integer_support_validation(self):
        """Non-integer observations are rejected by the encoder."""
        enc = SkellamDistribution(2.0, 2.0).dist_to_encoder()
        with self.assertRaises(ValueError):
            enc.seq_encode([0, 1, 2.5])

    def test_invalid_rates_raise(self):
        with self.assertRaises(ValueError):
            SkellamDistribution(0.0, 1.0)
        with self.assertRaises(ValueError):
            SkellamDistribution(1.0, -2.0)


if __name__ == "__main__":
    unittest.main()
