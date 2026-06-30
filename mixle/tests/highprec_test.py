"""Arbitrary-precision tail (mixle.engines.highprec): fp512/fp1024 via MPFR, verified vs the oracle."""

import unittest

import numpy as np
import pytest

from mixle.engines.highprec import (
    HighPrecisionFormat,
    available,
    hp_sum,
)

if not available():  # pragma: no cover
    raise unittest.SkipTest("no arbitrary-precision backend (gmpy2/mpmath)")

mpmath = pytest.importorskip("mpmath")


class ArbitraryPrecisionTest(unittest.TestCase):
    def test_precision_is_configurable_to_fp1024_and_beyond(self):
        for bits in (128, 256, 512, 1024, 4096):
            fmt = HighPrecisionFormat(bits)
            self.assertEqual(fmt.bits, bits)
            self.assertEqual(fmt.max_rel_error, 2.0 ** -(bits + 1))

    def test_round_trip_of_float64_is_lossless(self):
        rng = np.random.RandomState(0)
        x = rng.randn(500) * 10.0 ** rng.randint(-8, 8, 500)
        self.assertTrue(np.array_equal(HighPrecisionFormat(512).round_trip(x), x))

    def test_hp_sum_matches_exact_oracle(self):
        rng = np.random.RandomState(1)
        x = rng.randn(20000) * 10.0 ** rng.randint(-10, 10, 20000)
        with mpmath.workprec(600):
            true = mpmath.fsum(mpmath.mpf(float(v)) for v in x)
        got = hp_sum(x, 512)
        with mpmath.workprec(600):
            denom = abs(true) if true != 0 else mpmath.mpf(1)
            rel = float(abs((mpmath.mpf(got) - true) / denom))
        self.assertLess(rel, 1e-13)  # float64-rounded result of an exact high-precision sum

    def test_recovers_dynamic_range_float64_loses(self):
        # 1e30 dwarfs 1e-30 by ~200 bits; float64 (53) drops the tiny terms, MPFR at 256 bits keeps them.
        x = np.concatenate([[1e30], np.full(1000, 1e-30), [-1e30]])
        true = 1000.0 * 1e-30  # the exact sum of these float64 values
        f64 = float(np.sum(x))
        hp = hp_sum(x, 256)
        self.assertLess(abs(hp - true), abs(f64 - true))  # MPFR recovers what float64's range lost
        self.assertAlmostEqual(hp / true, 1.0, places=3)

    def test_fp1024_holds_far_more_than_float64_bits(self):
        # accumulate 1/7 a thousand times at 1024-bit precision: result is 1000/7 to ~300 digits,
        # vastly more than float64 could carry through the accumulation.
        x = np.full(1000, 1.0 / 7.0)
        got = hp_sum(x, 1024)
        with mpmath.workprec(1024):
            true = mpmath.mpf(1000) * mpmath.mpf(float(1.0 / 7.0))
            rel = float(abs((mpmath.mpf(got) - true) / true))
        self.assertLess(rel, 1e-14)


if __name__ == "__main__":
    unittest.main()
