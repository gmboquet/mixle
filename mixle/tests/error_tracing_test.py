"""Sound error tracing (mixle.engines.error_tracing): interval enclosures + precision allocation.

mpmath is the slow correctness oracle: the certified bounds must actually contain the exact result.
"""

import unittest

import numpy as np
import pytest

from mixle.engines.error_tracing import (
    Interval,
    float64_sum_is_accurate,
    sum_enclosure,
    sum_error_bound,
)
from mixle.engines.extended import dd_sum

mpmath = pytest.importorskip("mpmath")


class IntervalSoundnessTest(unittest.TestCase):
    def test_mul_encloses_true_product(self):
        rng = np.random.RandomState(0)
        a = rng.randn(2000) * 1e8
        b = rng.randn(2000) * 1e8
        iv = Interval.exact(a) * Interval.exact(b)
        with mpmath.workprec(200):
            for i in range(a.size):
                true = mpmath.mpf(float(a[i])) * mpmath.mpf(float(b[i]))
                self.assertLessEqual(mpmath.mpf(float(iv.lo[i])), true)
                self.assertGreaterEqual(mpmath.mpf(float(iv.hi[i])), true)

    def test_add_sub_enclose_true_result(self):
        rng = np.random.RandomState(1)
        a = rng.randn(2000) * 1e10
        b = rng.randn(2000)  # tiny next to a -> float64 add loses bits, interval must still enclose
        s = Interval.exact(a) + Interval.exact(b)
        d = Interval.exact(a) - Interval.exact(b)
        with mpmath.workprec(200):
            for i in range(a.size):
                ts = mpmath.mpf(float(a[i])) + mpmath.mpf(float(b[i]))
                td = mpmath.mpf(float(a[i])) - mpmath.mpf(float(b[i]))
                self.assertTrue(mpmath.mpf(float(s.lo[i])) <= ts <= mpmath.mpf(float(s.hi[i])))
                self.assertTrue(mpmath.mpf(float(d.lo[i])) <= td <= mpmath.mpf(float(d.hi[i])))

    def test_from_quantized_encloses_original(self):
        from mixle.engines.formats import CodebookFormat, FloatFormat

        rng = np.random.RandomState(2)
        x = rng.randn(3000)
        self.assertTrue(np.all(Interval.from_quantized(x, FloatFormat.fp(16)).contains(x)))
        self.assertTrue(np.all(Interval.from_quantized(x, FloatFormat.fp(8)).contains(x)))
        # codebook has no analytic relative bound -> uses the measured absolute error, still sound
        cb = CodebookFormat.fit(x, 64)
        self.assertTrue(np.all(Interval.from_quantized(x, cb).contains(x)))


class SumErrorTracingTest(unittest.TestCase):
    def _true(self, x):
        with mpmath.workprec(400):
            return mpmath.fsum(mpmath.mpf(float(v)) for v in x)

    def test_bound_actually_bounds_the_float64_error(self):
        rng = np.random.RandomState(3)
        for _ in range(5):
            x = rng.randn(20000) * 10.0 ** rng.randint(-6, 6, 20000)
            true = self._true(x)
            fl = mpmath.mpf(float(np.sum(x)))
            bound = sum_error_bound(x)
            self.assertLessEqual(float(abs(fl - true)), bound)  # certified: real error <= bound

    def test_enclosure_contains_true_sum(self):
        rng = np.random.RandomState(4)
        x = rng.randn(20000) * 10.0 ** rng.randint(-6, 6, 20000)
        true = float(self._true(x))
        iv = sum_enclosure(x)
        self.assertTrue(bool(iv.contains(true)))

    def test_precision_allocation_flags_when_float64_is_enough_vs_not(self):
        rng = np.random.RandomState(5)
        # well-conditioned (all positive, no cancellation): float64 already accurate -> no extra compute
        good = rng.rand(5000) + 1.0
        self.assertTrue(float64_sum_is_accurate(good, 1e-10))
        # catastrophic cancellation: float64 NOT accurate -> the logic says use double-double
        bad = np.tile(np.array([1e16, 1.0, -1e16, -1.0]), 5000)
        rng.shuffle(bad)
        self.assertFalse(float64_sum_is_accurate(bad, 1e-10))
        # ...and dd_sum then recovers the true value float64 missed
        self.assertLess(abs(float(dd_sum(bad).to_float())), 1e-6)  # true sum is 0


if __name__ == "__main__":
    unittest.main()
