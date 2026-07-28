"""Arbitrary-precision tail (mixle.engines.highprec): fp512/fp1024 via MPFR, verified vs the oracle."""

import unittest
from decimal import Decimal

import numpy as np
import pytest

from mixle.engines.highprec import (
    HighPrecisionFormat,
    available,
    hp_array,
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
            self.assertEqual(fmt.name, "fp%d" % bits)
            self.assertEqual(fmt.max_rel_error, 2.0**-fmt.precision_bits)

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

    # -- MXR-080-0135: hp_array/hp_sum used to coerce every input through float64 before it ever
    # reached the arbitrary-precision backend, so a Decimal/string/int carrying more precision than
    # float64 offers was silently rounded down to float64's ~15-17 significant digits first, no matter
    # how many ``bits`` were then requested. The tests below cover each input type the fix now converts
    # directly (no float64 stop), plus the float64 policy and the hp_sum summation path.

    @staticmethod
    def _scalar(arr):
        """Pull the single element out of a (possibly 0-d) hp_array result."""
        return arr[()] if arr.shape == () else arr.reshape(-1)[0]

    @staticmethod
    def _full_precision_str(v, bits):
        """Full-precision decimal string for a backend-native scalar, whichever backend is active.

        ``gmpy2.mpfr.__str__`` is already precision-complete and independent of any ambient context.
        ``mpmath.mpf.__str__`` instead prints the *shortest* round-trippable form and, outside a
        matching ``workprec`` block, silently truncates to the ambient default precision -- so it needs
        ``nstr`` with an explicit digit count to show everything the value actually holds.
        """
        try:
            import gmpy2

            if isinstance(v, gmpy2.mpfr):
                return str(v)
        except ImportError:  # pragma: no cover - mpmath fallback environment
            pass
        ndigits = int(bits * 0.31) + 15
        return mpmath.nstr(v, ndigits)

    def test_hp_array_preserves_decimal_precision_beyond_float64(self):
        # The audit's exact reproduction: a Decimal with far more significant digits than float64's
        # ~15-17 used to be coerced through float64 first inside hp_array, silently collapsing to 1.0.
        d = Decimal("1.0000000000000000000000000000000001")  # 37 significant digits
        self.assertEqual(float(d), 1.0)  # sanity: genuinely beyond float64's own resolution
        got = self._scalar(hp_array(d, 200))
        with mpmath.workprec(220):
            got_val = mpmath.mpf(self._full_precision_str(got, 200))
            true = mpmath.mpf(str(d))
            self.assertNotEqual(got_val, mpmath.mpf(1))  # the bug: float64 detour collapses this to 1
            rel = float(abs((got_val - true) / true))
        self.assertLess(rel, 2.0**-190)  # comfortably inside the 200-bit budget

    def test_hp_array_preserves_string_precision_beyond_float64(self):
        s = "3.0000000000000000000000000000000007"  # 38 significant digits, also beyond float64
        self.assertEqual(float(s), 3.0)  # sanity: also collapses under a naive float64 cast
        got = self._scalar(hp_array(s, 200))
        with mpmath.workprec(220):
            got_val = mpmath.mpf(self._full_precision_str(got, 200))
            true = mpmath.mpf(s)
            self.assertNotEqual(got_val, mpmath.mpf(3))
            rel = float(abs((got_val - true) / true))
        self.assertLess(rel, 2.0**-190)

    def test_hp_array_preserves_large_integer_beyond_float64_exact_range(self):
        # float64 only represents integers exactly up to 2**53; both of these lose their last digit(s)
        # if routed through float() first (2**53+1 rounds down to exactly 2**53).
        for big in (2**53 + 1, 10**60 + 1):
            with self.subTest(big=big):
                got = self._scalar(hp_array(big, 256))
                self.assertEqual(int(got), big)  # exact -- integers have no fractional precision to lose

    def test_hp_array_preserves_backend_native_precision(self):
        # Feeding an already-backend-native high-precision value back into hp_array must re-round it
        # directly (widen/narrow in place), not degrade it through float64 first.
        native = self._scalar(hp_array(Decimal("1.0000000000000000000000000000000001"), 300))
        got = self._scalar(hp_array(native, 300))
        with mpmath.workprec(320):
            native_val = mpmath.mpf(self._full_precision_str(native, 300))
            got_val = mpmath.mpf(self._full_precision_str(got, 300))
            self.assertNotEqual(got_val, mpmath.mpf(1))  # the bug: passthrough via float() collapses this
            rel = float(abs((got_val - native_val) / native_val))
        self.assertLess(rel, 2.0**-290)  # same bits in, same bits out

    def test_hp_array_float64_input_preserves_exact_binary_value(self):
        # Explicit float64 policy: a plain float has no more than float64 precision to begin with, so
        # converting it must reproduce its *exact* dyadic value -- not "fix it up" to a tidier decimal.
        # A naive ``Decimal(str(x))`` round-trip would do exactly that (wrongly) instead of ``Decimal(x)``.
        f = 0.1  # not exactly 0.1 in binary
        exact = Decimal(f)  # the float's true value
        self.assertNotEqual(str(exact), "0.1")  # sanity: confirms this float's exact value isn't tidy
        got = self._scalar(hp_array(f, 200))
        with mpmath.workprec(220):
            got_val = mpmath.mpf(self._full_precision_str(got, 200))
            true = mpmath.mpf(str(exact))
            self.assertEqual(got_val, true)  # bit-for-bit exact, not rounded to a "nicer" decimal

    def test_hp_sum_avoids_catastrophic_cancellation_on_decimal_input(self):
        # Two ~1e29-magnitude Decimals differing only in their 31st significant digit: naive float64
        # arithmetic (and the old hp_sum bug, which cast every element to float64 before summing) can't
        # tell them apart, so their difference collapses to 0.0 instead of the true 0.2 residual.
        a = Decimal("100000000000000000000000000000.3")
        b = Decimal("-100000000000000000000000000000.1")
        naive = float(a) + float(b)
        self.assertEqual(naive, 0.0)  # sanity: this really is a catastrophic-cancellation setup
        got = hp_sum([a, b], 256)
        self.assertEqual(got, 0.2)

    def test_hp_sum_ordinary_float64_inputs_unaffected(self):
        # Negative control: ordinary well-behaved float64 inputs still sum to the plain expected answer.
        x = np.array([1.5, 2.25, 3.125, -0.5])
        self.assertEqual(hp_sum(x, 128), 6.375)


if __name__ == "__main__":
    unittest.main()
