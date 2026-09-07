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
                self.assertLessEqual(d._tail_mass(int(k)), 4.0 * (1.0 - q) + 1.0e-300, (p, q))
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


def _tail_mass_by_quadrature(p: float, k: int) -> float:
    """``P(X > k)`` by an independent route: ``sum_{j>k} p^j / j = int_0^p t^k / (1-t) dt``, with
    ``t = p e^{-s}`` so the integrand is ``exp(a (log p - s)) / (1 - p e^{-s})`` over ``s >= 0``."""
    from scipy.integrate import quad

    a = k + 1

    def integrand(s: float) -> float:
        return math.exp(a * (math.log(p) - s)) / (1.0 - p * math.exp(-s))

    scale = 1.0 / a
    edges = [0.0] + [scale * m for m in (1, 2, 4, 8, 16, 32, 64, 128)]
    value = sum(quad(integrand, lo, hi, limit=200)[0] for lo, hi in zip(edges, edges[1:]))
    return value / (-math.log1p(-p))


class TailPastTheCapTest(unittest.TestCase):
    """0.8.1 adversarial review P01-F01: past ``_SERIES_CAP`` the CDF used one minus a geometric
    tail *bound*, loose by about ``1 / (k (1-p))`` and unclamped, so at ``p`` within ``1e-8`` of one
    the CDF fell from 0.88 to 0.24 between ``k = 5e7`` and ``6e7`` and reached -360 at ``p = 1 -
    1e-12``; the quantile inherited the overshoot. The tail is now the Euler-Maclaurin sum."""

    def test_cdf_past_the_cap_matches_an_independent_quadrature(self):
        d = LogSeriesDistribution(1.0 - 1.0e-9)
        for k in (5 * 10**7, 6 * 10**7, 10**8, 10**9, 10**11):
            expected = 1.0 - _tail_mass_by_quadrature(d.p, k)
            self.assertAlmostEqual(d.cdf(k), expected, places=8, msg=k)
        self.assertAlmostEqual(d.cdf(6 * 10**7), 0.889240, places=5)
        self.assertAlmostEqual(d.cdf(10**8), 0.912035, places=5)
        self.assertAlmostEqual(LogSeriesDistribution(1.0 - 1.0e-12).cdf(10**8), 0.687553, places=5)
        self.assertAlmostEqual(LogSeriesDistribution(1.0 - 1.0e-10).cdf(10**8), 0.824635, places=5)

    def test_cdf_is_within_the_unit_interval_and_monotone_across_the_cap(self):
        for p in (1.0 - 1.0e-8, 1.0 - 1.0e-9, 1.0 - 1.0e-10, 1.0 - 1.0e-12):
            d = LogSeriesDistribution(p)
            ks = np.unique(np.logspace(0, 13, 120).astype(np.int64))
            values = [d.cdf(int(k)) for k in ks]
            self.assertTrue(all(0.0 <= v <= 1.0 for v in values), p)
            self.assertTrue(all(b >= a for a, b in zip(values, values[1:])), p)

    def test_quantile_past_the_cap_is_the_smallest_k_with_cdf_at_least_q(self):
        d = LogSeriesDistribution(1.0 - 1.0e-9)
        for q in (0.9, 0.99):
            k = int(d.quantile(q))
            self.assertGreater(k, LogSeriesDistribution._SERIES_CAP)
            self.assertGreaterEqual(1.0 - _tail_mass_by_quadrature(d.p, k), q - 1.0e-9)
            self.assertLess(1.0 - _tail_mass_by_quadrature(d.p, k - 1), q + 1.0e-9)
        quantiles = [d.quantile(q) for q in (0.5, 0.9, 0.99, 1.0 - 1.0e-4, 1.0 - 1.0e-8, 1.0 - 1.0e-12, 1.0 - 1.0e-16)]
        self.assertTrue(all(b >= a for a, b in zip(quantiles, quantiles[1:])), quantiles)

    def test_capped_branch_agrees_with_scipy_where_scipy_can_sum(self):
        cap = LogSeriesDistribution._SERIES_CAP
        try:
            LogSeriesDistribution._SERIES_CAP = 4096
            d = LogSeriesDistribution(0.999)
            for k in (4097, 5000, 10_000, 50_000, 100_000):
                self.assertAlmostEqual(d.cdf(k), float(logser.cdf(k, 0.999)), places=13, msg=k)
            for q in (0.9, 0.99, 0.999, 1.0 - 1.0e-8):
                self.assertEqual(d.quantile(q), float(logser.ppf(q, 0.999)), q)
        finally:
            LogSeriesDistribution._SERIES_CAP = cap

    def test_tail_mass_matches_quadrature_over_the_regime_it_serves(self):
        for p in (0.999, 1.0 - 1.0e-6, 1.0 - 1.0e-9):
            d = LogSeriesDistribution(p)
            for k in (10**3, 10**5, 10**7, 10**9):
                expected = _tail_mass_by_quadrature(p, k)
                if expected < 1.0e-250:
                    continue
                self.assertAlmostEqual(d._tail_mass(k) / expected, 1.0, places=8, msg=(p, k))


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
