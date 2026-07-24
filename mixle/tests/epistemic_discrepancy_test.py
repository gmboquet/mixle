"""mixle.epistemic.discrepancy: KL/JS/Wasserstein/MMD between distributions or samples (Card E1)."""

import unittest

import numpy as np

from mixle.epistemic.discrepancy import (
    _sample,
    discrepancy_report,
    js_divergence,
    kl_divergence,
    mmd,
    wasserstein_distance,
)
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution


class SampleDispatchTest(unittest.TestCase):
    """_sample's dist.sample(n) / per-draw dist.sample() fallback dispatch."""

    def test_a_bug_inside_sample_is_not_masked_by_a_retry_looping_n_single_draws(self):
        # a sample(n) that accepts n must be called with it exactly once; a TypeError from inside
        # its own body must propagate, not be swallowed and silently retried as n separate sample()
        # calls -- which would be both wrong (n calls instead of 1) and far more expensive. n=None
        # has a default specifically so a naive try/except TypeError fallback's sample() retries
        # are themselves syntactically valid and reach the body (appending to calls) -- otherwise
        # those retries would fail at argument-binding before ever calling in, and this test could
        # not tell a single correct call apart from a silent duplicate retry loop.
        calls = []

        class BuggyDist:
            def sample(self, n=None):
                calls.append(n)
                return None + (n or 0)  # an internal bug unrelated to whether n is accepted

        with self.assertRaises(TypeError):
            _sample(BuggyDist(), 5, np.random.RandomState(0))
        self.assertEqual(calls, [5])  # called once, with n -- never retried as 5 separate calls

    def test_legacy_single_draw_sample_falls_back_correctly(self):
        class LegacySingleDrawDist:
            def __init__(self):
                self.i = 0

            def sample(self):
                self.i += 1
                return float(self.i)

        out = _sample(LegacySingleDrawDist(), 4, np.random.RandomState(0))
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0, 4.0])


class KLDivergenceTest(unittest.TestCase):
    def test_kl_of_identical_gaussians_is_zero(self):
        p = GaussianDistribution(0.0, 1.0)
        q = GaussianDistribution(0.0, 1.0)
        self.assertAlmostEqual(kl_divergence(p, q), 0.0, places=8)

    def test_kl_matches_closed_form_gaussian_formula(self):
        p = GaussianDistribution(0.0, 1.0)
        q = GaussianDistribution(5.0, 1.0)
        mu_p, var_p, mu_q, var_q = 0.0, 1.0, 5.0, 1.0
        expected = 0.5 * (var_p / var_q + (mu_q - mu_p) ** 2 / var_q - 1.0 + np.log(var_q / var_p))
        self.assertAlmostEqual(kl_divergence(p, q), expected, places=8)


class JSDivergenceTest(unittest.TestCase):
    def test_symmetric_within_mc_tolerance(self):
        p = GaussianDistribution(0.0, 1.0)
        q = GaussianDistribution(3.0, 1.0)
        a = js_divergence(p, q, n=20_000, seed=0)
        b = js_divergence(q, p, seed=0, n=20_000)
        self.assertAlmostEqual(a, b, delta=0.02)


class WassersteinDistanceTest(unittest.TestCase):
    def test_matches_known_toy_case(self):
        class PointMass:
            def __init__(self, value):
                self.value = value

            def sample(self, n):
                return np.full(n, self.value, dtype=np.float64)

        p = PointMass(0.0)
        q = PointMass(3.0)
        self.assertAlmostEqual(wasserstein_distance(p, q, n=100), 3.0, places=8)

    def test_multivariate_raises_not_implemented(self):
        class TwoD:
            def sample(self, n):
                return np.zeros((n, 2), dtype=np.float64)

        with self.assertRaises(NotImplementedError):
            wasserstein_distance(TwoD(), TwoD(), n=10)

    def test_column_vector_samples_are_sorted_before_matching(self):
        # np.sort defaults to the last axis; on a (n, 1) column vector -- what a distribution's
        # .sample(n) returns when each draw is its own length-1 row -- that axis has one element,
        # so sorting it in place is a silent no-op and the draws stay in their original order.
        # p=[100, 0] vs q=[49, 51]: the correct order statistics are p=[0, 100], q=[49, 51], giving
        # mean(|0-49|, |100-51|) = 49.0. Pairing the *unsorted* draws instead gives the wrong 51.0.
        class ColumnVectorDist:
            def __init__(self, values):
                self._values = np.asarray(values, dtype=np.float64).reshape(-1, 1)

            def sample(self, n):
                return self._values

        p = ColumnVectorDist([100.0, 0.0])
        q = ColumnVectorDist([49.0, 51.0])
        self.assertAlmostEqual(wasserstein_distance(p, q, n=2), 49.0, places=8)

    def test_column_vector_and_flat_samples_agree(self):
        # Negative control: a (n, 1) column vector and its (n,) flattened equivalent must give the
        # identical answer -- the extra trailing length-1 axis alone must not change the result now
        # that both shapes are squeezed to 1D before sorting.
        class FlatDist:
            def __init__(self, values):
                self._values = np.asarray(values, dtype=np.float64)

            def sample(self, n):
                return self._values

        class ColumnVectorDist:
            def __init__(self, values):
                self._values = np.asarray(values, dtype=np.float64).reshape(-1, 1)

            def sample(self, n):
                return self._values

        flat = wasserstein_distance(FlatDist([100.0, 0.0, 37.5]), FlatDist([49.0, 51.0, 2.5]), n=3)
        column = wasserstein_distance(ColumnVectorDist([100.0, 0.0, 37.5]), ColumnVectorDist([49.0, 51.0, 2.5]), n=3)
        self.assertAlmostEqual(flat, column, places=8)


class MMDTest(unittest.TestCase):
    def test_same_distribution_is_near_zero(self):
        rng = np.random.RandomState(0)
        xs = rng.normal(size=1000)
        value = mmd(xs[:500], xs[500:])
        self.assertLess(abs(value), 0.05)

    def test_different_distributions_is_clearly_positive(self):
        rng = np.random.RandomState(0)
        xs = rng.normal(size=1000)
        ys = rng.normal(loc=5.0, size=1000)
        self.assertGreater(mmd(xs, ys), 0.5)

    def test_unknown_kernel_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            mmd(np.zeros(5), np.zeros(5), kernel="polynomial")

    def test_empty_samples_raises_instead_of_nan(self):
        with self.assertRaises(ValueError):
            mmd(np.array([]), np.array([1.0, 2.0, 3.0]))
        with self.assertRaises(ValueError):
            mmd(np.array([1.0, 2.0, 3.0]), np.array([]))

    def test_nonpositive_bandwidth_raises_instead_of_silently_using_gamma_one(self):
        for bad_bandwidth in (0.0, -1.0):
            with self.subTest(bandwidth=bad_bandwidth), self.assertRaises(ValueError):
                mmd(np.zeros(5), np.ones(5), bandwidth=bad_bandwidth)


class DiscrepancyReportTest(unittest.TestCase):
    def test_not_degraded_for_registered_closed_form_pair(self):
        p = GaussianDistribution(0.0, 1.0)
        q = GaussianDistribution(1.0, 1.0)
        result = discrepancy_report(p, q)
        self.assertEqual(result.metric, "kl_divergence")
        self.assertFalse(result.degraded)

    def test_degraded_for_a_pair_without_a_closed_form(self):
        class SampleOnly:
            def __init__(self, loc):
                self.loc = loc

            def log_density(self, x):
                return float(-0.5 * (x - self.loc) ** 2)

            def sample(self, n):
                return np.random.RandomState(0).normal(loc=self.loc, size=n)

        result = discrepancy_report(SampleOnly(0.0), SampleOnly(1.0))
        self.assertEqual(result.metric, "kl_divergence")
        self.assertTrue(result.degraded)

    def test_explicit_metric_is_honored(self):
        result = discrepancy_report(np.zeros(50), np.ones(50), metric="mmd")
        self.assertEqual(result.metric, "mmd")
        self.assertTrue(result.degraded)

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            discrepancy_report(np.zeros(5), np.zeros(5), metric="not_a_real_metric")


if __name__ == "__main__":
    unittest.main()
