"""The log-series CDF, quantile, and entropy are finite computations in bounded memory.

``quantile`` used to go through scipy's generic discrete ``ppf``, which brackets the answer by
doubling an upper bound until ``cdf(bound) >= q``. The summed CDF can saturate one ULP short of
a ``q`` within double rounding of one (numpy's SIMD ``exp``/``log`` round the last bit differently
per CPU), and on AVX-512 hosted CI runners the bound then doubled without limit while the
vectorized CDF sums grew until the 16 GB runner was OOM-killed -- reached through ``entropy()``,
which calls ``quantile(1 - 1e-16)``. ``entropy()`` itself then built one ``arange`` up to that
quantile, a 226 GiB allocation at ``p = 1 - 1e-9``. These tests pin the bounded replacements.
"""

from __future__ import annotations

import math
import time
import unittest

import numpy as np
from scipy.stats import logser

from mixle.stats import LogSeriesDistribution


class QuantileAndCdfTest(unittest.TestCase):
    def test_exact_agreement_with_scipy_away_from_the_boundary(self):
        for p in (0.3, 0.9, 0.99, 0.999, 0.99999):
            d = LogSeriesDistribution(p)
            for q in (0.01, 0.5, 0.9, 0.999, 1.0 - 1.0e-8):
                self.assertEqual(d.quantile(q), float(logser.ppf(q, p)), (p, q))
            for k in (1, 2, 5, 50, 500, 5000, 50000):
                self.assertAlmostEqual(d.cdf(k), float(logser.cdf(k, p)), places=13, msg=(p, k))

    def test_boundary_quantile_is_bounded_monotone_and_leaves_rounding_level_mass(self):
        # In this regime "the" quantile is not defined at double precision: the crossing point
        # of a sum that lands within an ULP of q depends on the rounding of every term. What is
        # required is a finite answer, monotone in q, beyond which the tail mass is at rounding
        # level -- never an unbounded search.
        for p in (0.3, 0.9, 0.99, 0.999, 0.99999):
            d = LogSeriesDistribution(p)
            previous = d.quantile(1.0 - 1.0e-8)
            for q in (1.0 - 1.0e-12, 1.0 - 1.0e-16):
                start = time.perf_counter()
                k = d.quantile(q)
                self.assertLess(time.perf_counter() - start, 2.0, (p, q))
                self.assertGreaterEqual(k, previous, (p, q))
                self.assertLessEqual(d._tail_upper_bound(int(k)), 4.0 * (1.0 - q) + 1.0e-300, (p, q))
                previous = k

    def test_endpoints_and_validation(self):
        d = LogSeriesDistribution(0.7)
        self.assertEqual(d.quantile(0.0), 1.0)
        self.assertEqual(d.quantile(1.0), math.inf)
        self.assertEqual(d.cdf(0.5), 0.0)
        for bad in (-0.1, 1.5, math.nan):
            with self.assertRaises(ValueError):
                d.quantile(bad)

    def test_p_within_a_billionth_of_one_is_bounded(self):
        d = LogSeriesDistribution(1.0 - 1.0e-9)
        start = time.perf_counter()
        median = d.quantile(0.5)
        far = d.quantile(1.0 - 1.0e-16)
        self.assertLess(time.perf_counter() - start, 10.0)
        self.assertGreater(far, median)
        self.assertGreater(far, 1.0e9)  # the true support really is that long; the answer is just bounded
        self.assertLessEqual(d.cdf(1.0e12), 1.0)


class EntropyTest(unittest.TestCase):
    def test_tail_method_matches_the_full_sum_where_the_full_sum_is_cheap(self):
        cap = LogSeriesDistribution._SERIES_CAP
        try:
            for p, small_cap in ((0.999, 4096), (0.99999, 65_536)):
                d = LogSeriesDistribution(p)
                LogSeriesDistribution._SERIES_CAP = cap
                full = d.entropy()
                LogSeriesDistribution._SERIES_CAP = small_cap
                capped = d.entropy()
                self.assertAlmostEqual(full, capped, places=9, msg=p)
        finally:
            LogSeriesDistribution._SERIES_CAP = cap

    def test_ordinary_values_match_the_direct_series(self):
        for p in (0.3, 0.9, 0.99):
            d = LogSeriesDistribution(p)
            k = np.arange(1, 20_001, dtype=np.float64)
            lp = k * math.log(p) - np.log(k) - math.log(-math.log1p(-p))
            self.assertAlmostEqual(d.entropy(), float(-np.sum(np.exp(lp) * lp)), places=10, msg=p)

    def test_p_within_a_billionth_of_one_completes_in_bounded_memory_and_time(self):
        d = LogSeriesDistribution(1.0 - 1.0e-9)
        start = time.perf_counter()
        value = d.entropy()
        self.assertLess(time.perf_counter() - start, 30.0)
        self.assertTrue(math.isfinite(value))
        self.assertGreater(value, LogSeriesDistribution(0.99999).entropy())  # entropy grows with p


if __name__ == "__main__":
    unittest.main()
