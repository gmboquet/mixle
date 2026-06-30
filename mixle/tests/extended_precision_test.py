"""Double-double extended precision (mixle.engines.extended): exact EFTs + accurate reductions.

mpmath is used ONLY as a slow correctness oracle here -- never in the library hot path. These tests
prove the error-free transformations are bit-exact and that the vectorized double-double reductions beat
float64 on cancellation while running far faster than mpmath at the same precision.
"""

import time
import unittest

import numpy as np
import pytest

from mixle.engines.extended import DoubleDouble, dd_dot, dd_sum, two_prod, two_sum

mpmath = pytest.importorskip("mpmath")


def _mpf_pair_sum(a, b):
    return mpmath.mpf(float(a)) + mpmath.mpf(float(b))


class ErrorFreeTransformTest(unittest.TestCase):
    def test_two_sum_is_bit_exact(self):
        rng = np.random.RandomState(0)
        a = rng.randn(500) * 10.0 ** rng.randint(-30, 30, 500)
        b = rng.randn(500) * 10.0 ** rng.randint(-30, 30, 500)
        s, e = two_sum(a, b)
        with mpmath.workprec(200):
            for i in range(a.size):
                # a + b == s + e must hold EXACTLY (the defining property of the transform)
                self.assertEqual(_mpf_pair_sum(a[i], b[i]), _mpf_pair_sum(s[i], e[i]))

    def test_two_prod_is_bit_exact(self):
        rng = np.random.RandomState(1)
        a = rng.randn(500) * 10.0 ** rng.randint(-20, 20, 500)
        b = rng.randn(500) * 10.0 ** rng.randint(-20, 20, 500)
        p, e = two_prod(a, b)
        with mpmath.workprec(250):
            for i in range(a.size):
                lhs = mpmath.mpf(float(a[i])) * mpmath.mpf(float(b[i]))
                rhs = mpmath.mpf(float(p[i])) + mpmath.mpf(float(e[i]))
                self.assertEqual(lhs, rhs)  # a*b == p + e exactly


class DoubleDoubleArithmeticTest(unittest.TestCase):
    def test_dd_mul_matches_mpmath_to_full_precision(self):
        rng = np.random.RandomState(2)
        a = rng.randn(200)
        b = rng.randn(200)
        prod = DoubleDouble.from_float(a) * DoubleDouble.from_float(b)
        with mpmath.workprec(160):
            for i in range(a.size):
                exact = mpmath.mpf(float(a[i])) * mpmath.mpf(float(b[i]))
                got = mpmath.mpf(float(prod.hi[i])) + mpmath.mpf(float(prod.lo[i]))
                rel = abs((got - exact) / exact) if exact != 0 else abs(got - exact)
                self.assertLess(float(rel), 2.0**-100)  # ~106-bit accuracy, far beyond float64's 2^-53

    def test_dd_add_matches_mpmath_to_full_precision(self):
        rng = np.random.RandomState(3)
        a = rng.randn(200) * 1e8
        b = rng.randn(200)  # very different magnitudes -> float64 would drop b's low bits
        ssum = DoubleDouble.from_float(a) + DoubleDouble.from_float(b)
        with mpmath.workprec(160):
            for i in range(a.size):
                exact = mpmath.mpf(float(a[i])) + mpmath.mpf(float(b[i]))
                got = mpmath.mpf(float(ssum.hi[i])) + mpmath.mpf(float(ssum.lo[i]))
                rel = abs((got - exact) / exact) if exact != 0 else abs(got - exact)
                self.assertLess(float(rel), 2.0**-100)


class AccurateReductionTest(unittest.TestCase):
    def _exact_sum(self, x):
        with mpmath.workprec(400):
            return mpmath.fsum(mpmath.mpf(float(v)) for v in x)

    def test_dd_sum_recovers_catastrophic_cancellation(self):
        # equal counts of +-1e16 and +-1 -> true sum is exactly 0, but a shuffled float64 sum loses the
        # unit terms in the 1e16 magnitude.
        rng = np.random.RandomState(4)
        x = np.tile(np.array([1e16, 1.0, -1e16, -1.0]), 25000)
        rng.shuffle(x)
        true = float(self._exact_sum(x))  # 0.0
        f64 = float(np.sum(x))
        dd = float(dd_sum(x).to_float())
        self.assertEqual(true, 0.0)
        self.assertLess(abs(dd - true), 1e-6)  # double-double nails it
        self.assertLessEqual(abs(dd - true), abs(f64 - true) + 1e-12)  # never worse than float64

    def test_dd_sum_matches_oracle_on_wide_dynamic_range(self):
        rng = np.random.RandomState(5)
        x = rng.randn(50000) * 10.0 ** rng.randint(-12, 12, 50000)
        r = dd_sum(x)
        # The comparison MUST run at high precision: hi+lo differ by ~16 orders, so adding them at
        # mpmath's default 53-bit precision would silently drop lo and look only float64-accurate.
        with mpmath.workprec(400):
            true = mpmath.fsum(mpmath.mpf(float(v)) for v in x)
            dd = mpmath.mpf(float(r.hi)) + mpmath.mpf(float(r.lo))
            denom = abs(true) if true != 0 else mpmath.mpf(1)
            rel = float(abs((dd - true) / denom))
        self.assertLess(rel, 1e-25)  # ~double-double relative accuracy (vs ~1e-16 for float64)

    def test_dd_dot_beats_float64(self):
        rng = np.random.RandomState(6)
        a = rng.randn(20000) * 10.0 ** rng.randint(-8, 8, 20000)
        b = rng.randn(20000) * 10.0 ** rng.randint(-8, 8, 20000)
        with mpmath.workprec(400):
            true = mpmath.fsum(mpmath.mpf(float(a[i])) * mpmath.mpf(float(b[i])) for i in range(a.size))
        dd = float(dd_dot(a, b).to_float())
        f64 = float(a @ b)
        self.assertLessEqual(abs(dd - float(true)), abs(f64 - float(true)) + 1e-9)


class SpeedVsOracleTest(unittest.TestCase):
    def test_dd_sum_is_far_faster_than_mpmath(self):
        # The whole point: vectorized double-double gives ~106-bit accuracy without mpmath's per-object cost.
        rng = np.random.RandomState(7)
        x = rng.randn(20000)

        t0 = time.perf_counter()
        for _ in range(3):
            dd_sum(x)
        t_dd = (time.perf_counter() - t0) / 3

        t0 = time.perf_counter()
        with mpmath.workprec(106):
            mpmath.fsum(mpmath.mpf(float(v)) for v in x)
        t_mp = time.perf_counter() - t0

        self.assertLess(t_dd, t_mp)  # double-double must be faster than the mpmath oracle at equal precision


if __name__ == "__main__":
    unittest.main()
