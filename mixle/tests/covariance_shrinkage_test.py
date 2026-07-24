"""Ledoit-Wolf covariance shrinkage: exactness vs the closed form, the estimator contract, conditioning."""

import unittest

import numpy as np

from mixle.analysis import LedoitWolfEstimator, LedoitWolfInsufficientData
from mixle.inference import estimate
from mixle.stats import MultivariateGaussianDistribution


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


class LedoitWolfValidationTest(unittest.TestCase):
    """MXR-080-0076: accumulator updates must reject invalid sufficient statistics instead of silently
    accepting them, and ``estimate`` must return a typed insufficient-data result -- never a Gaussian
    built from NaN/Inf or from a numerically-healed-but-meaningless covariance -- when the accumulated
    weight cannot support a well-posed estimate."""

    # --- update() / seq_update() reject invalid statistics at the update boundary -----------------

    def test_update_rejects_negative_weight(self):
        acc = LedoitWolfEstimator(dim=3).accumulator_factory().make()
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            acc.update(np.array([1.0, 2.0, 3.0]), -1.0, None)

    def test_update_accepts_zero_weight(self):
        # zero is finite and nonnegative -- a legitimate no-op contribution, not an error.
        acc = LedoitWolfEstimator(dim=3).accumulator_factory().make()
        acc.update(np.array([1.0, 2.0, 3.0]), 0.0, None)
        self.assertEqual(acc.count, 0.0)

    def test_update_rejects_non_finite_weight(self):
        acc = LedoitWolfEstimator(dim=3).accumulator_factory().make()
        for bad_weight in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaisesRegex(ValueError, "finite"):
                acc.update(np.array([1.0, 2.0, 3.0]), bad_weight, None)

    def test_update_rejects_non_finite_observation(self):
        acc = LedoitWolfEstimator(dim=3).accumulator_factory().make()
        with self.assertRaisesRegex(ValueError, "finite"):
            acc.update(np.array([1.0, float("nan"), 3.0]), 1.0, None)
        with self.assertRaisesRegex(ValueError, "finite"):
            acc.update(np.array([1.0, float("inf"), 3.0]), 1.0, None)

    def test_update_rejects_mismatched_dimension_fixed_at_construction(self):
        # dim declared explicitly on the estimator: every observation must match it, from the first.
        acc = LedoitWolfEstimator(dim=3).accumulator_factory().make()
        with self.assertRaisesRegex(ValueError, r"\b3\b.*\b2\b"):
            acc.update(np.array([1.0, 2.0]), 1.0, None)

    def test_update_rejects_mismatched_dimension_inferred_from_first_observation(self):
        # no dim declared: it is inferred from (and locked by) the first observation.
        acc = LedoitWolfEstimator().accumulator_factory().make()
        acc.update(np.array([1.0, 2.0, 3.0]), 1.0, None)
        with self.assertRaisesRegex(ValueError, r"\b3\b.*\b2\b"):
            acc.update(np.array([1.0, 2.0]), 1.0, None)

    def test_seq_update_rejects_negative_or_non_finite_weights(self):
        acc = LedoitWolfEstimator(dim=3).accumulator_factory().make()
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            acc.seq_update(x, np.array([1.0, -1.0]), None)
        with self.assertRaisesRegex(ValueError, "finite"):
            acc.seq_update(x, np.array([1.0, float("nan")]), None)

    def test_seq_update_rejects_non_finite_observations(self):
        acc = LedoitWolfEstimator(dim=3).accumulator_factory().make()
        x = np.array([[1.0, 2.0, 3.0], [4.0, float("inf"), 6.0]])
        with self.assertRaisesRegex(ValueError, "finite"):
            acc.seq_update(x, np.array([1.0, 1.0]), None)

    def test_seq_update_rejects_mismatched_dimension(self):
        acc = LedoitWolfEstimator(dim=3).accumulator_factory().make()
        x = np.array([[1.0, 2.0], [4.0, 5.0]])
        with self.assertRaisesRegex(ValueError, r"\b3\b.*\b2\b"):
            acc.seq_update(x, np.array([1.0, 1.0]), None)

    def test_combine_rejects_mismatched_dimension(self):
        est = LedoitWolfEstimator()
        a = est.accumulator_factory().make()
        a.update(np.array([1.0, 2.0, 3.0]), 1.0, None)
        b = est.accumulator_factory().make()
        b.update(np.array([1.0, 2.0]), 1.0, None)
        with self.assertRaisesRegex(ValueError, r"\b3\b.*\b2\b"):
            a.combine(b.value())

    # --- estimate() returns a typed insufficient-data result instead of a fabricated Gaussian -------

    def test_estimate_on_empty_accumulator_returns_insufficient_data(self):
        est = LedoitWolfEstimator(dim=4)
        acc = est.accumulator_factory().make()
        result = est.estimate(None, acc.value())
        self.assertIsInstance(result, LedoitWolfInsufficientData)
        self.assertNotIsInstance(result, MultivariateGaussianDistribution)
        self.assertEqual(result.effective_count, 0.0)
        self.assertEqual(result.dim, 4)

    def test_estimate_on_single_observation_returns_insufficient_data(self):
        # one effective observation's raw sample covariance is exactly zero -- structurally
        # uninformative, not just noisy -- so this must not silently become a "valid" Gaussian.
        est = LedoitWolfEstimator(dim=3)
        acc = est.accumulator_factory().make()
        acc.update(np.array([5.0, -2.0, 7.0]), 1.0, None)
        result = est.estimate(None, acc.value())
        self.assertIsInstance(result, LedoitWolfInsufficientData)
        self.assertEqual(result.effective_count, 1.0)

    def test_estimate_rejects_fractional_effective_count_below_threshold(self):
        est = LedoitWolfEstimator(dim=3)
        acc = est.accumulator_factory().make()
        acc.update(np.array([1.0, 2.0, 3.0]), 1.5, None)
        result = est.estimate(None, acc.value())
        self.assertIsInstance(result, LedoitWolfInsufficientData)
        self.assertEqual(result.effective_count, 1.5)

    def test_estimate_at_minimum_effective_count_is_well_posed(self):
        # boundary: exactly 2 effective observations, with dim > n -- Ledoit-Wolf's whole purpose is
        # to stay well-conditioned in exactly this regime, so this must NOT be treated as insufficient.
        rng = np.random.RandomState(3)
        est = LedoitWolfEstimator(dim=8)
        acc = est.accumulator_factory().make()
        acc.update(rng.randn(8), 1.0, None)
        acc.update(rng.randn(8), 1.0, None)
        result = est.estimate(None, acc.value())
        self.assertIsInstance(result, MultivariateGaussianDistribution)
        covar = np.asarray(result.covar)
        self.assertTrue(np.all(np.isfinite(covar)))
        eigvals = np.linalg.eigvalsh(covar)
        self.assertTrue(np.all(eigvals >= -1e-8), eigvals)

    def test_estimate_negative_control_well_conditioned_data_is_finite_psd(self):
        # negative control: normal, adequately-sized, well-conditioned data must still flow through
        # the same validation path to a legitimate, finite, positive-semidefinite covariance.
        rng = np.random.RandomState(0)
        R = rng.randn(120, 6) @ rng.randn(6, 6) + 0.4
        fit = estimate([r for r in R], LedoitWolfEstimator(dim=6))
        self.assertIsInstance(fit, MultivariateGaussianDistribution)
        covar = np.asarray(fit.covar)
        self.assertTrue(np.all(np.isfinite(covar)))
        eigvals = np.linalg.eigvalsh(covar)
        self.assertTrue(np.all(eigvals >= -1e-8), eigvals)


if __name__ == "__main__":
    unittest.main()
