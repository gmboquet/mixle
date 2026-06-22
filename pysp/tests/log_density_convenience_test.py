"""Tests for the top-level log_density()/density() convenience functions.

These wrap the seq_encode / seq_log_density / np.concatenate boilerplate into a single raw-data call. They must
agree with the boilerplate exactly, stay aligned to the input order, and compose with the scalar per-observation
methods and seq_log_density_sum.
"""

import unittest

import numpy as np

from pysp.inference.estimation import optimize
from pysp.stats import (
    MixtureEstimator,
    MultivariateGaussianEstimator,
    density,
    log_density,
    seq_encode,
    seq_log_density_sum,
)
from pysp.stats.base.gaussian import GaussianDistribution


class LogDensityConvenienceTestCase(unittest.TestCase):
    def setUp(self):
        self.model = GaussianDistribution(1.3, 4.0)
        self.xs = [0.1, -2.0, 5.0, 1.3, 3.7, -1.1, 0.0]

    def test_matches_seq_boilerplate(self):
        ld = log_density(self.xs, self.model)
        enc = seq_encode(self.xs, model=self.model)
        old = np.concatenate([self.model.seq_log_density(e) for _, e in enc])
        self.assertEqual(ld.shape, (len(self.xs),))
        self.assertTrue(np.allclose(ld, old))

    def test_aligned_to_scalar_method(self):
        ld = log_density(self.xs, self.model)
        scalar = np.array([self.model.log_density(x) for x in self.xs])
        self.assertTrue(np.allclose(ld, scalar))

    def test_density_is_exp_log_density(self):
        self.assertTrue(np.allclose(density(self.xs, self.model), np.exp(log_density(self.xs, self.model))))

    def test_sum_matches_seq_log_density_sum(self):
        ld = log_density(self.xs, self.model)
        enc = seq_encode(self.xs, model=self.model)
        self.assertTrue(np.allclose(ld.sum(), seq_log_density_sum(enc, self.model)[1]))

    def test_compound_mixture_model(self):
        rng = np.random.RandomState(0)
        X = np.vstack([rng.randn(30, 2), rng.randn(30, 2) + 5]).tolist()
        # explicit rng: optimize()'s default rng is a shared mutable default, so seed locally for reproducibility
        mix = optimize(
            X,
            MixtureEstimator([MultivariateGaussianEstimator(dim=2) for _ in range(2)]),
            max_its=20,
            rng=np.random.RandomState(0),
        )
        ld = log_density(X, mix)
        old = np.concatenate([mix.seq_log_density(e) for _, e in seq_encode(X, model=mix)])
        self.assertEqual(ld.shape, (len(X),))
        self.assertTrue(np.allclose(ld, old))
        self.assertTrue(np.isfinite(ld).all())
        self.assertTrue((density(X, mix) >= 0).all())


if __name__ == "__main__":
    unittest.main()
