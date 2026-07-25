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

    # -- from_quantized's analytic path trusts a format's static max_rel_error as a bound for ANY
    # input it might quantize, but that attribute is a duck-typed, self-reported claim -- nothing
    # enforces it actually holds. mixle's own FloatFormat is safe (mantissa-only, unbounded exponent,
    # see numeric_formats_test.py's MXR-080-0126 comment for exactly this reasoning), but a format
    # that flushes underflowing values to exactly zero (the ordinary behavior a REAL bounded float
    # format's underflow handling would have) would report the same fixed max_rel_error while its
    # true error at those magnitudes no longer scales with it. Before from_quantized cross-checked
    # the analytic pad against the error measured on the actual data, this produced an
    # (effectively) zero-width enclosure around 0 that did NOT contain the nonzero original -- a
    # silently unsound "certificate", exactly what this module exists to prevent. These two tests
    # fail against the pre-fix code and pass once the cross-check is in place.

    def test_from_quantized_cross_checks_analytic_bound_against_measured_error(self):
        class FlushToZeroMockFormat:
            """Deliberately unsound mock (not FloatFormat, which is correctly unbounded): exposes a
            static max_rel_error like a real bounded fpN codec would, but round_trip flushes
            sub-threshold magnitudes to exactly 0.0 -- the behavior a real bounded format would have.
            """

            max_rel_error = 0.0625  # a plausible-looking analytic bound (~fp8-ish), same for every input

            def __init__(self, flush_threshold):
                self.flush_threshold = flush_threshold

            def round_trip(self, x):
                x = np.asarray(x, dtype=np.float64)
                out = x.copy()
                out[np.abs(out) < self.flush_threshold] = 0.0
                return out

            def measured_max_abs_error(self, x):
                x = np.asarray(x, dtype=np.float64)
                return float(np.max(np.abs(self.round_trip(x) - x))) if x.size else 0.0

        # the exact seed/data numeric_formats_test.py's MXR-080-0126 comment measures: 40/3000 values
        # fall below a real fp8's smallest normal magnitude (2**-6).
        rng = np.random.RandomState(2)
        x = rng.randn(3000)
        fmt = FlushToZeroMockFormat(flush_threshold=2.0**-6)
        flushed = int(np.sum(np.abs(x) < fmt.flush_threshold))
        self.assertEqual(flushed, 40)  # sanity: the vulnerable case is actually exercised here
        self.assertTrue(np.all(Interval.from_quantized(x, fmt).contains(x)))

        # a single, easy-to-reason-about scalar case in the same direction: q rounds to exactly 0,
        # so the analytic pad (proportional to |q|) is also exactly 0 -- only the measured error
        # (|original - q|) can supply a pad wide enough to actually enclose the nonzero original.
        tiny = np.array([1e-10])
        iv_tiny = Interval.from_quantized(tiny, fmt)
        self.assertEqual(float(fmt.round_trip(tiny)[0]), 0.0)  # confirms this hits the q==0 case
        self.assertTrue(bool(iv_tiny.contains(tiny)[0]))
        self.assertGreater(float(iv_tiny.width()[0]), 0.0)  # not a degenerate point at a nonzero value

    def test_from_quantized_cross_check_also_covers_saturating_overflow(self):
        # Same class of gap in the opposite direction: a format that saturates on overflow instead
        # of flushing on underflow. Not exercised by the fixed-seed data above, but structurally
        # identical -- a fixed max_rel_error stops bounding the true error once round_trip clips.
        class SaturatingMockFormat:
            max_rel_error = 0.0625

            def __init__(self, sat_max):
                self.sat_max = sat_max

            def round_trip(self, x):
                return np.clip(np.asarray(x, dtype=np.float64), -self.sat_max, self.sat_max)

            def measured_max_abs_error(self, x):
                x = np.asarray(x, dtype=np.float64)
                return float(np.max(np.abs(self.round_trip(x) - x))) if x.size else 0.0

        fmt = SaturatingMockFormat(sat_max=448.0)  # a real e4m3 fp8's max representable magnitude
        big = np.array([1.0e6])
        self.assertEqual(float(fmt.round_trip(big)[0]), fmt.sat_max)  # confirms this hits saturation
        self.assertTrue(bool(Interval.from_quantized(big, fmt).contains(big)[0]))

    # -- MXR-080-0125: [0,0] * [-inf,inf] used to come back [nan,nan] (IEEE 0*inf), and the
    # constructor accepted reversed bounds and NaN endpoints without complaint. --

    def test_mul_zero_times_unbounded_is_exact_zero(self):
        # the audit's exact repro, plus the sign variants it calls out by name
        cases = [
            (Interval(0.0, 0.0), Interval(-np.inf, np.inf)),
            (Interval(0.0, 0.0), Interval(0.0, np.inf)),
            (Interval(0.0, 0.0), Interval(-np.inf, 0.0)),
            (Interval(-0.0, 0.0), Interval(-np.inf, np.inf)),  # negative-zero lower bound
        ]
        for zero, unbounded in cases:
            for r in (zero * unbounded, unbounded * zero):  # both operand orders
                self.assertFalse(bool(np.any(np.isnan(r.lo))) or bool(np.any(np.isnan(r.hi))), r)
                self.assertEqual(float(r.lo), 0.0)
                self.assertEqual(float(r.hi), 0.0)

    def test_mul_finite_matches_hand_computed_corners(self):
        # negative control: ordinary finite multiplication never touches the 0*inf special case and
        # must be unaffected by it
        cases = [
            ((2.0, 3.0), (4.0, 5.0), (8.0, 15.0)),
            ((-3.0, 2.0), (-1.0, 4.0), (-12.0, 8.0)),
            ((-5.0, -2.0), (-3.0, -1.0), (2.0, 15.0)),
        ]
        for (a_lo, a_hi), (b_lo, b_hi), (exp_lo, exp_hi) in cases:
            iv = Interval(a_lo, a_hi) * Interval(b_lo, b_hi)
            self.assertLessEqual(float(iv.lo), exp_lo)
            self.assertGreaterEqual(float(iv.hi), exp_hi)
            self.assertAlmostEqual(float(iv.lo), exp_lo, places=9)  # sound, and not pathologically loose
            self.assertAlmostEqual(float(iv.hi), exp_hi, places=9)

    def test_mul_property_finite_samples_stay_enclosed_across_zero_and_infinite_bounds(self):
        # The actual definition of a sound enclosure: every finite value that could legitimately be
        # drawn from within each factor interval, multiplied together, lands inside the computed
        # product -- not just the specific named edge cases above. Bounds are drawn from a mix of
        # finite, zero, and infinite endpoints so both the 0*inf corner case and ordinary rounding
        # are exercised together.
        rng = np.random.RandomState(7)
        bound_choices = [-np.inf, -1e6, -100.0, -1.0, -1e-3, 0.0, 1e-3, 1.0, 100.0, 1e6, np.inf]

        def is_degenerate_infinite_point(lo, hi):
            # [inf, inf] / [-inf, -inf]: a degenerate point exactly at +/-inf has no finite elements
            # to sample from at all (unlike [-inf, hi] or [lo, inf], where the infinite endpoint just
            # means "no bound" and every finite value up to the finite side is a legitimate sample) --
            # out of scope for this specific "sample finite values, check enclosure" methodology; the
            # degenerate-infinity behavior itself is covered directly by the width/midpoint and
            # add/sub tests above.
            return bool(lo == hi and np.isinf(lo))

        def sample_within(lo, hi):
            big = 1e10  # finite stand-in for a one-sided infinite endpoint
            lo_s = -big if np.isinf(lo) else lo
            hi_s = big if np.isinf(hi) else hi
            return lo_s if lo_s == hi_s else rng.uniform(lo_s, hi_s)

        with mpmath.workprec(200):
            checked = 0
            while checked < 300:
                lo_a, hi_a = sorted(rng.choice(bound_choices, size=2))
                lo_b, hi_b = sorted(rng.choice(bound_choices, size=2))
                if is_degenerate_infinite_point(lo_a, hi_a) or is_degenerate_infinite_point(lo_b, hi_b):
                    continue
                checked += 1
                iv = Interval(float(lo_a), float(hi_a)) * Interval(float(lo_b), float(hi_b))
                for _ in range(5):
                    a = sample_within(lo_a, hi_a)
                    b = sample_within(lo_b, hi_b)
                    true = mpmath.mpf(float(a)) * mpmath.mpf(float(b))
                    self.assertLessEqual(mpmath.mpf(float(iv.lo)), true)
                    self.assertGreaterEqual(mpmath.mpf(float(iv.hi)), true)

    def test_construction_rejects_reversed_bounds(self):
        with self.assertRaises(ValueError):
            Interval(5.0, 2.0)
        with self.assertRaises(ValueError):
            Interval(np.array([0.0, 3.0]), np.array([1.0, 2.0]))  # reversed only at index 1

    def test_construction_rejects_nan_endpoints(self):
        with self.assertRaises(ValueError):
            Interval(np.nan, 1.0)
        with self.assertRaises(ValueError):
            Interval(0.0, np.nan)
        with self.assertRaises(ValueError):
            Interval.exact(np.nan)

    def test_construction_allows_infinite_bounds(self):
        # unbounded intervals are a normal, useful part of interval arithmetic -- only NaN and
        # reversed order are rejected, not +/-inf (including the degenerate point [inf, inf], since
        # lo == hi there is not a reversed bound)
        Interval(-np.inf, np.inf)
        Interval(0.0, np.inf)
        Interval(-np.inf, 0.0)
        Interval(np.inf, np.inf)
        Interval(-np.inf, -np.inf)

    def test_add_sub_degenerate_infinite_operand_raises_instead_of_returning_nan(self):
        # inf + -inf (unlike 0*inf) has no single correct finite answer -- refusing via the
        # constructor's NaN rejection is the sound response to an unrepresentable result, not
        # fabricating a bound. Only reachable via a *degenerate* point exactly at +/-inf; see
        # test_add_ordinary_unbounded_intervals_do_not_raise for the (unaffected) normal case.
        with self.assertRaises(ValueError):
            Interval.exact(np.inf) + Interval(-np.inf, 3.0)
        with self.assertRaises(ValueError):
            Interval.exact(np.inf) - Interval.exact(np.inf)
        with self.assertRaises(ValueError):
            Interval.exact(-np.inf) - Interval.exact(-np.inf)

    def test_add_ordinary_unbounded_intervals_do_not_raise(self):
        # negative control: an ordinary unbounded (non-degenerate) interval has a finite lo or hi,
        # so it never hits the inf + -inf gap above
        r = Interval(5.0, np.inf) + Interval(-np.inf, 3.0)
        self.assertEqual(float(r.lo), -np.inf)
        self.assertEqual(float(r.hi), np.inf)

    def test_width_and_midpoint_of_degenerate_infinite_point(self):
        self.assertEqual(float(Interval.exact(np.inf).width()), 0.0)
        self.assertEqual(float(Interval.exact(-np.inf).width()), 0.0)
        self.assertEqual(float(Interval.exact(np.inf).midpoint()), np.inf)
        self.assertEqual(float(Interval.exact(-np.inf).midpoint()), -np.inf)
        # negative control: an ordinary non-degenerate unbounded interval is unaffected
        self.assertEqual(float(Interval(0.0, np.inf).width()), np.inf)
        self.assertEqual(float(Interval(0.0, np.inf).midpoint()), np.inf)


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
