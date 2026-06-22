"""Ledoit-Wolf covariance shrinkage: exactness vs the closed form, the estimator contract, conditioning."""

import unittest

import numpy as np

from pysp.inference import estimate
from pysp.stats import LedoitWolfEstimator, MultivariateGaussianDistribution


def _lw_reference(R):
    """Direct Ledoit-Wolf (2004) shrinkage to a scaled-identity target, on the raw data matrix."""
    T, n = R.shape
    X = R - R.mean(0)
    S = X.T @ X / T
    mu = np.trace(S) / n
    F = mu * np.eye(n)
    d2 = np.sum((S - F) ** 2)
    b2 = sum(np.sum((np.outer(X[t], X[t]) - S) ** 2) for t in range(T)) / T**2
    delta = float(np.clip(b2 / d2, 0, 1))
    return (1 - delta) * S + delta * F, delta


class LedoitWolfTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.R = rng.randn(120, 6) @ rng.randn(6, 6) + 0.4

    def test_matches_closed_form_exactly(self):
        fit = estimate([r for r in self.R], LedoitWolfEstimator(dim=6))
        cov_ref, delta_ref = _lw_reference(self.R)
        self.assertIsInstance(fit, MultivariateGaussianDistribution)
        np.testing.assert_allclose(fit.mu, self.R.mean(0), atol=1e-9)
        np.testing.assert_allclose(np.asarray(fit.covar), cov_ref, atol=1e-9)
        self.assertAlmostEqual(fit.shrinkage, delta_ref, places=10)

    def test_distributed_combine_equals_batch(self):
        est = LedoitWolfEstimator(dim=6)

        def shard(rows):
            a = est.accumulator_factory().make()
            a.seq_update(a.acc_to_encoder().seq_encode(list(rows)), np.ones(len(rows)), None)
            return a

        a = shard(self.R[:70])
        a.combine(shard(self.R[70:]).value())
        d_split = est.estimate(None, a.value())
        d_full = estimate([r for r in self.R], LedoitWolfEstimator(dim=6))
        np.testing.assert_allclose(np.asarray(d_split.covar), np.asarray(d_full.covar), atol=1e-9)

    def test_shrinkage_improves_conditioning(self):
        rng = np.random.RandomState(1)
        R = rng.randn(20, 12)  # fewer samples than 2x dim -> ill-conditioned sample covariance
        fit = estimate([r for r in R], LedoitWolfEstimator(dim=12))
        self.assertGreater(fit.shrinkage, 0.0)
        self.assertLess(np.linalg.cond(np.asarray(fit.covar)), np.linalg.cond(np.cov(R.T)))

    def test_little_shrinkage_when_already_well_estimated(self):
        rng = np.random.RandomState(2)
        # many samples + anisotropic truth: the sample covariance is well-estimated and clearly differs
        # from the scaled-identity target, so the data-driven shrinkage intensity is small
        R = rng.randn(20000, 4) * np.array([1.0, 2.0, 3.0, 4.0])
        fit = estimate([r for r in R], LedoitWolfEstimator(dim=4))
        self.assertLess(fit.shrinkage, 0.05)


if __name__ == "__main__":
    unittest.main()
