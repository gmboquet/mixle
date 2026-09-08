"""Univariate laws: bounded-memory entropy, disclosed clamps, refused data, and quantile endpoints.

The ten adversarial reviews of the 0.8.1 candidate found one shape of defect repeated across the
univariate families (pass 01, findings P01-F02..F13): a guard that silently dropped the observations
it did not like, a clamp that bound without saying so, a closed form that raised at a parameter the
same law samples and scores happily, and a series summed into one array as wide as the support.
These tests pin each repair.
"""

from __future__ import annotations

import math
import time
import tracemalloc
import unittest

import numpy as np
import scipy.stats as ss

from mixle.inference import estimate
from mixle.stats import (
    BetaDistribution,
    BinomialDistribution,
    ExponentialEstimator,
    GaussianEstimator,
    GeometricDistribution,
    GeometricEstimator,
    GumbelDistribution,
    LogSeriesDistribution,
    LogSeriesEstimator,
    NegativeBinomialDistribution,
    PoissonDistribution,
    PoissonEstimator,
    SkewNormalDistribution,
    UniformEstimator,
    WeibullDistribution,
    WeibullEstimator,
)


class BoundedEntropyTest(unittest.TestCase):
    """P01-F02: the count-family entropies allocated one array as wide as the effective support."""

    def test_poisson_and_negative_binomial_agree_with_scipy_at_ordinary_parameters(self):
        for lam in (0.5, 1.0, 5.0, 10.0, 100.0):
            self.assertAlmostEqual(PoissonDistribution(lam).entropy(), float(ss.poisson(lam).entropy()), places=12)
        for r, p in ((1.0, 0.5), (5.0, 0.3), (50.0, 0.5), (500.0, 0.7)):
            self.assertAlmostEqual(
                NegativeBinomialDistribution(r, p).entropy(), float(ss.nbinom(r, p).entropy()), places=11
            )

    def test_large_parameters_are_bounded_in_memory_and_time(self):
        # 3.2 GB at r = 1e8 and a 7 PiB MemoryError at lam = 1e15 before the repair.
        for law in (PoissonDistribution(1.0e8), NegativeBinomialDistribution(1.0e8, 0.5), PoissonDistribution(1.0e15)):
            tracemalloc.start()
            try:
                start = time.perf_counter()
                value = law.entropy()
                elapsed = time.perf_counter() - start
                peak = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
            self.assertLess(peak, 32_000_000, law)
            self.assertLess(elapsed, 5.0, law)
            self.assertTrue(math.isfinite(value), law)

    def test_large_parameter_values_match_their_asymptotic_limits(self):
        for lam in (1.0e7, 1.0e10, 1.0e15):
            # H = 0.5 log(2 pi e lam) - 1/(12 lam) + O(lam^-2)
            expected = 0.5 * math.log(2.0 * math.pi * math.e * lam) - 1.0 / (12.0 * lam)
            self.assertAlmostEqual(PoissonDistribution(lam).entropy(), expected, places=9)
        for r in (1.0e7, 1.0e10):
            variance = r * 0.5 / 0.25
            self.assertAlmostEqual(
                NegativeBinomialDistribution(r, 0.5).entropy(),
                0.5 * math.log(2.0 * math.pi * math.e * variance),
                places=6,
            )

    def test_a_heavy_geometric_tail_is_summed_rather_than_cut_at_a_gaussian_width(self):
        # r = 0.5, p = 1e-9 puts the effective support past 1e10 counts with most of the mass in a
        # geometric tail; the answer must still be the Gaussian-scale magnitude, in bounded time.
        start = time.perf_counter()
        value = NegativeBinomialDistribution(0.5, 1.0e-9).entropy()
        self.assertLess(time.perf_counter() - start, 5.0)
        self.assertTrue(math.isfinite(value))
        self.assertGreater(value, 15.0)
        self.assertLess(value, 25.0)


class DisclosedClampTest(unittest.TestCase):
    """P01-F03, F06, F07, F08, F09: every hard wall an estimator hits is now in numerical_repairs()."""

    def test_weibull_reports_its_shape_ceiling_and_scale_floor(self):
        fit = estimate([2.0, 2.0, 2.0], WeibullEstimator())
        self.assertTrue(any("shape-unresolvable" in note for note in fit.numerical_repairs()), fit.numerical_repairs())
        floored = estimate([1.0e-300] * 5, WeibullEstimator())
        self.assertTrue(any("floored" in note for note in floored.numerical_repairs()), floored.numerical_repairs())

    def test_an_ordinary_weibull_sample_reports_nothing(self):
        self.assertEqual(estimate([1.0, 2.0, 3.0, 4.0, 5.0], WeibullEstimator()).numerical_repairs(), ())

    def test_poisson_geometric_uniform_and_logseries_report_their_floors(self):
        self.assertIn("poisson-rate-floored", estimate([0, 0, 0], PoissonEstimator()).numerical_repairs()[0])
        self.assertIn("geometric-p-clamped", estimate([1, 1, 1], GeometricEstimator()).numerical_repairs()[0])
        self.assertIn("width-floored", estimate([5.0, 5.0, 5.0], UniformEstimator()).numerical_repairs()[0])
        self.assertIn("logseries-p-floored", estimate([1, 1, 1], LogSeriesEstimator()).numerical_repairs()[0])


class RefusedObservationTest(unittest.TestCase):
    """P01-F05, F06, F07, F08, F09: silently dropped observations are refused by name."""

    CASES = (
        (ExponentialEstimator, [1.0, 2.0, 3.0, -100.0], "ExponentialDistribution"),
        (ExponentialEstimator, [1.0, 2.0, 3.0, float("nan")], "ExponentialDistribution"),
        (ExponentialEstimator, [1.0, 2.0, 3.0, float("-inf")], "ExponentialDistribution"),
        (PoissonEstimator, [1, 2, 3, -5], "PoissonDistribution"),
        (PoissonEstimator, [1, 2, 2.5], "PoissonDistribution"),
        (GeometricEstimator, [1, 2, 3, -7], "GeometricDistribution"),
        (GeometricEstimator, [0, 0, 0], "GeometricDistribution"),
        (GeometricEstimator, [1.0, 2.0, 3.0, float("inf")], "GeometricDistribution"),
        (UniformEstimator, [0.0, 1.0, float("nan")], "UniformDistribution"),
        (LogSeriesEstimator, [1.0, 2.0, float("nan")], "LogSeriesDistribution"),
        (LogSeriesEstimator, [1.0, 2.0, float("inf")], "LogSeriesDistribution"),
    )

    def test_every_family_names_itself_and_its_support(self):
        for estimator, data, family in self.CASES:
            with self.subTest(family=family, data=str(data)):
                with self.assertRaises(ValueError) as caught:
                    estimate(data, estimator())
                self.assertIn(family, str(caught.exception))

    def test_clean_data_still_fits(self):
        self.assertAlmostEqual(estimate([1.0, 2.0, 3.0], ExponentialEstimator()).beta, 2.0)
        self.assertAlmostEqual(estimate([1, 2, 3], PoissonEstimator()).lam, 2.0)
        self.assertAlmostEqual(estimate([1, 2, 3], GeometricEstimator()).p, 0.5)


class ExtremeParameterMomentTest(unittest.TestCase):
    """P01-F10: closed forms raised OverflowError/ZeroDivisionError where the limit exists."""

    def test_weibull_moments_diverge_to_infinity_rather_than_raising(self):
        self.assertEqual(WeibullDistribution(1.0e-3, 1.0).mean(), math.inf)
        self.assertEqual(WeibullDistribution(1.0e-3, 1.0).variance(), math.inf)

    def test_ordinary_weibull_moments_are_unchanged(self):
        self.assertAlmostEqual(WeibullDistribution(2.0, 3.0).mean(), float(ss.weibull_min(2.0, scale=3.0).mean()))
        self.assertAlmostEqual(WeibullDistribution(2.0, 3.0).variance(), float(ss.weibull_min(2.0, scale=3.0).var()))

    def test_beta_variance_at_the_float_floor_is_the_fair_coin(self):
        self.assertAlmostEqual(BetaDistribution(1.0e-300, 1.0e-300).variance(), 0.25, places=12)
        self.assertAlmostEqual(BetaDistribution(2.0, 5.0).variance(), float(ss.beta(2.0, 5.0).var()))

    def test_log_series_moments_take_their_limit_as_p_goes_to_zero(self):
        self.assertAlmostEqual(LogSeriesDistribution(1.0e-300).mean(), 1.0)
        self.assertEqual(LogSeriesDistribution(1.0e-300).variance(), 0.5e-300)
        self.assertAlmostEqual(LogSeriesDistribution(0.5).variance(), float(ss.logser(0.5).var()), places=12)
        self.assertAlmostEqual(LogSeriesDistribution(1.0e-5).variance(), float(ss.logser(1.0e-5).var()), places=15)

    def test_skew_normal_entropy_reaches_its_half_normal_limit(self):
        self.assertAlmostEqual(SkewNormalDistribution(0.0, 1.0, 1.0e300).entropy(), float(ss.halfnorm().entropy()))
        self.assertAlmostEqual(SkewNormalDistribution(0.0, 1.0, 2.0).entropy(), float(ss.skewnorm(2.0).entropy()))


class QuantileEndpointTest(unittest.TestCase):
    """P01-F11, F12: one endpoint convention, and a finite quantile where scipy's ppf gives NaN."""

    def test_discrete_endpoints_are_the_support_bounds(self):
        self.assertEqual(PoissonDistribution(4.0).quantile(0.0), 0.0)
        self.assertEqual(PoissonDistribution(4.0).quantile(1.0), math.inf)
        self.assertEqual(BinomialDistribution(0.4, 10).quantile(0.0), 0.0)
        self.assertEqual(BinomialDistribution(0.4, 10).quantile(1.0), 10.0)
        self.assertEqual(NegativeBinomialDistribution(4.0, 0.4).quantile(0.0), 0.0)
        self.assertEqual(NegativeBinomialDistribution(4.0, 0.4).quantile(1.0), math.inf)
        self.assertEqual(GeometricDistribution(0.3).quantile(0.0), 1.0)
        self.assertEqual(GeometricDistribution(0.3).quantile(1.0), math.inf)
        self.assertEqual(LogSeriesDistribution(0.5).quantile(0.0), 1.0)

    def test_gumbel_returns_the_infinities_every_other_continuous_family_returns(self):
        self.assertEqual(GumbelDistribution(0.0, 1.0).quantile(0.0), -math.inf)
        self.assertEqual(GumbelDistribution(0.0, 1.0).quantile(1.0), math.inf)
        for bad in (-0.1, 1.5, math.nan):
            with self.assertRaises(ValueError):
                GumbelDistribution(0.0, 1.0).quantile(bad)

    def test_interior_quantiles_keep_scipy_parity(self):
        for lam in (0.5, 4.0, 100.0):
            for q in (0.01, 0.25, 0.5, 0.9, 0.999):
                self.assertEqual(PoissonDistribution(lam).quantile(q), float(ss.poisson.ppf(q, lam)), (lam, q))

    def test_a_huge_rate_gets_a_finite_median_where_scipy_returns_nan(self):
        # scipy's behaviour here is the PREMISE, not the claim, and it is version-dependent: the
        # generic discrete `ppf` returns NaN on scipy 1.17.1 and a finite count on 1.16.0. Asserting
        # the NaN unconditionally failed the minimum-versions tier on the scipy that does not have
        # the defect -- a mixle test failing because a dependency got better. What this pins is
        # mixle's own answer, which must be finite and correct either way.
        if not math.isnan(float(ss.poisson.ppf(0.5, 1.0e12))):
            self.assertTrue(math.isfinite(float(ss.poisson.ppf(0.5, 1.0e12))))  # no third state
        for lam in (1.0e12, 1.0e15):
            law = PoissonDistribution(lam)
            median = law.quantile(0.5)
            self.assertTrue(math.isfinite(median), lam)
            self.assertGreaterEqual(law.cdf(median), 0.5)
            self.assertLess(law.cdf(median - 1.0), 0.5)


class ColumnArrayTest(unittest.TestCase):
    """P01-F13: an (n, 1) column array produced one opaque numpy TypeError for all thirty families."""

    def test_the_error_names_the_estimator_and_the_shape(self):
        column = np.random.RandomState(0).rand(20, 1) + 1.0
        for estimator in (GaussianEstimator(), ExponentialEstimator(), WeibullEstimator()):
            with self.subTest(estimator=type(estimator).__name__):
                with self.assertRaises(ValueError) as caught:
                    estimate(column, estimator)
                message = str(caught.exception)
                self.assertIn(type(estimator).__name__, message)
                self.assertIn("(1,)", message)
                self.assertIn("ravel", message)

    def test_a_flat_array_still_fits(self):
        flat = np.random.RandomState(0).rand(20) + 1.0
        self.assertTrue(math.isfinite(estimate(flat, GaussianEstimator()).mu))


if __name__ == "__main__":
    unittest.main()
