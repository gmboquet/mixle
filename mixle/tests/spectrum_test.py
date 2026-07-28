"""The precision-spectrum front door (mixle.engines.spectrum): auto-route to the cheapest accurate backend."""

import math
import time
import unittest
from decimal import Decimal
from unittest import mock

import numpy as np
import pytest

from mixle.engines.extended import DoubleDouble
from mixle.engines.spectrum import accurate_sum, cast, sum_certificate

mpmath = pytest.importorskip("mpmath")


class AccurateSumRoutingTest(unittest.TestCase):
    def _true(self, x):
        with mpmath.workprec(600):
            return float(mpmath.fsum(mpmath.mpf(float(v)) for v in x))

    def test_well_conditioned_stays_in_float64(self):
        x = np.random.RandomState(0).rand(5000) + 1.0  # all positive, no cancellation
        r = accurate_sum(x, 1e-12)
        self.assertEqual(r.backend, "float64")
        self.assertEqual(r.status, "ok")
        self.assertTrue(r.met_target)
        self.assertLessEqual(r.rel_error_bound, 1e-12)  # the certificate itself confirms the target was met
        self.assertAlmostEqual(r.value, self._true(x), places=6)

    def test_moderate_cancellation_escalates_to_double_double(self):
        rng = np.random.RandomState(1)
        x = np.tile(np.array([1e16, 1.0, -1e16]), 1000)  # true sum 1000, cond ~ 2^54
        rng.shuffle(x)
        r = accurate_sum(x, 1e-12)
        self.assertEqual(r.backend, "dd")
        self.assertEqual(r.status, "ok")
        self.assertAlmostEqual(r.value, self._true(x), delta=1e-6)

    def test_catastrophic_cancellation_escalates_to_mpfr(self):
        x = np.array([1e40, 1.0, -1e40])  # true sum 1; needs > double-double's ~106 bits
        r = accurate_sum(x, 1e-12)
        self.assertTrue(r.backend.startswith("mpfr"), r.backend)
        self.assertEqual(r.status, "ok")
        self.assertAlmostEqual(r.value, 1.0, places=6)

    def test_result_always_meets_target_vs_oracle(self):
        rng = np.random.RandomState(2)
        for _ in range(8):
            x = rng.randn(8000) * 10.0 ** rng.randint(-8, 8, 8000)
            r = accurate_sum(x, 1e-12)
            self.assertEqual(r.status, "ok")  # a target this loose against 8000 well-scaled terms is
            true = self._true(x)  # always achievable within the escalation cap
            rel = abs(r.value - true) / max(abs(true), 1e-300)
            self.assertLess(rel, 1e-9)  # comfortably within target across regimes

    def test_certificate_reports_condition_and_bound(self):
        cert = sum_certificate(np.tile(np.array([1e16, 1.0, -1e16]), 500))
        self.assertGreater(cert["condition_number"], 1e10)  # ill-conditioned
        self.assertIn("rel_error_bound", cert)


class AccurateSumFailClosedTest(unittest.TestCase):
    def test_finite_overflow_is_not_certified_as_ok(self):
        x = np.array([1e308, 1e308])
        r = accurate_sum(x, 1e-12)
        self.assertEqual(r.status, "overflow")
        self.assertFalse(r.met_target)
        self.assertTrue(np.isinf(r.value))
        cert = sum_certificate(x)
        self.assertEqual(cert["status"], "overflow_or_unbounded")
        self.assertFalse(cert["certified"])
        self.assertTrue(np.isinf(cert["rel_error_bound"]))

    def test_decimal_input_is_preserved_until_high_precision_accumulation(self):
        x = [Decimal("1.0000000000000000000000000000000001"), Decimal("-1")]
        r = accurate_sum(x, 1e-25)
        self.assertTrue(r.backend.startswith("mpfr"))
        self.assertEqual(r.status, "ok")
        self.assertAlmostEqual(r.value / 1e-34, 1.0, places=12)

    """MXR-080-0136: accurate_sum used to fail open (silently return an unverified value) instead of
    reporting that the requested target was not actually met, and a specific exact-cancellation shape
    crashed with an unrelated-looking OverflowError instead of either outcome."""

    def _true(self, x):
        with mpmath.workprec(600):
            return float(mpmath.fsum(mpmath.mpf(float(v)) for v in x))

    def test_exact_cancellation_of_six_unit_magnitude_terms_does_not_crash(self):
        """The audit's exact repro. dd's result is exactly 0 (the true sum here is exactly 0 and exactly
        representable), so cond = abs_sum / s_dd used to divide by ~0 and overflow to float('inf'); then
        math.ceil(inf) -- converting that infinite estimate to a bit count -- raised OverflowError. Now
        the saturated condition number routes into ordinary mpfr escalation and succeeds for real."""
        x = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
        r = accurate_sum(x, 1e-12)  # must not raise
        self.assertEqual(r.value, 0.0)
        self.assertEqual(r.status, "ok")
        self.assertTrue(r.met_target)
        self.assertLessEqual(r.rel_error_bound, 1e-12)

    def test_exact_cancellation_fails_closed_when_mpfr_unavailable(self):
        """Same exact-cancellation input, but with arbitrary precision unavailable: double-double's own
        condition-based bound is astronomically larger than the target, so this must come back flagged
        insufficient rather than crash or silently claim success."""
        x = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
        with mock.patch("mixle.engines.highprec.available", return_value=False):
            r = accurate_sum(x, 1e-12)  # must not raise
        self.assertEqual(r.backend, "dd")
        self.assertEqual(r.status, "insufficient")
        self.assertFalse(r.met_target)

    def test_fails_closed_instead_of_silently_returning_dd_when_mpfr_unavailable(self):
        """Before the fix, this returned (value, "dd") unconditionally whenever arbitrary precision was
        unavailable -- even though double-double had already been proven insufficient for the target.
        Here that "best effort" is actually 62% off from the true value against a 1e-12 target: the
        contract was being silently violated, not just theoretically at risk."""
        rng = np.random.RandomState(7)
        x = np.tile(np.array([1e40, 1.0, -1e40]), 50)  # true sum 50
        rng.shuffle(x)
        true = self._true(x)
        with mock.patch("mixle.engines.highprec.available", return_value=False):
            r = accurate_sum(x, 1e-12)  # must not raise
        self.assertEqual(r.backend, "dd")
        self.assertEqual(r.status, "insufficient")
        self.assertFalse(r.met_target)
        actual_rel_error = abs(r.value - true) / abs(true)
        self.assertGreater(actual_rel_error, 1e-3)  # "insufficient" is honest here, not overcautious
        self.assertGreater(r.rel_error_bound, 1e-12)  # the certificate itself also flags the miss
        # available (the default in this environment): the same input actually succeeds via mpfr.
        r_available = accurate_sum(x, 1e-12)
        self.assertEqual(r_available.status, "ok")
        self.assertAlmostEqual(r_available.value, true, delta=1e-6)

    def test_fails_closed_when_even_max_precision_is_insufficient(self):
        """Arbitrary precision available but capped: a target the cap genuinely cannot reach must be
        reported insufficient rather than silently returned as the capped best effort. The default cap
        (4096 bits) is never actually reachable by any representable finite (cond, target) pair, so this
        exercises the cap directly by lowering it rather than hunting for an unreachable input."""
        x = np.array([1e40, 1.0, -1e40])
        with mock.patch("mixle.engines.spectrum._MAX_MPFR_BITS", 64):
            r = accurate_sum(x, 1e-12)  # must not raise
        self.assertEqual(r.backend, "mpfr64")
        self.assertEqual(r.status, "insufficient")
        self.assertFalse(r.met_target)
        self.assertGreater(r.rel_error_bound, 1e-12)
        # negative control: uncapped, the identical input succeeds -- the cap was the limiting factor,
        # not the input itself.
        r_uncapped = accurate_sum(x, 1e-12)
        self.assertEqual(r_uncapped.status, "ok")

    def test_certificate_bound_is_accurate_when_target_is_met(self):
        """rel_error_bound is a genuine certified bound, not a placeholder: whenever status == "ok" it
        must itself be <= target_rel_error, across all three backends."""
        cases = {
            "float64": np.random.RandomState(0).rand(5000) + 1.0,
            "dd": np.tile(np.array([1e16, 1.0, -1e16]), 1000),
            "mpfr": np.array([1e40, 1.0, -1e40]),
        }
        for expected_backend, x in cases.items():
            r = accurate_sum(x, 1e-12)
            self.assertTrue(r.backend.startswith(expected_backend), r.backend)
            self.assertEqual(r.status, "ok")
            self.assertLessEqual(r.rel_error_bound, 1e-12)
            self.assertEqual(r.target_rel_error, 1e-12)

    def test_invalid_target_rejected(self):
        for bad in (0.0, -1e-9, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                accurate_sum(np.array([1.0, 2.0]), bad)


class AccurateSumCostModelTest(unittest.TestCase):
    """accurate_sum's routing decision must cost one dtype inspection, not one per element.

    The module's opening docstring rejects mpmath for being "~1000x slower than float64" -- so a
    per-element Python-object scan (box every value, re-derive its type and width one at a time) put
    accurate_sum's own *fast* path in exactly that regime: ~1620x np.sum and ~13.7x math.fsum on 200k
    float64, with ~79% of the runtime in the scan alone. A native numeric dtype answers "is this
    float64-exact?" for the whole array at once; only a genuine ``object`` array has per-element types
    to inspect.
    """

    def test_native_dtypes_never_take_the_per_element_object_route(self):
        def _boom(raw, target_rel_error):
            raise AssertionError("native dtype %r must not be boxed per element" % (raw.dtype,))

        native = [
            np.arange(1000, dtype=np.float64) + 1.0,
            np.arange(1000, dtype=np.float32) + 1.0,
            np.arange(1000, dtype=np.float16) + 1.0,
            np.arange(1000, dtype=np.int64),
            np.arange(1000, dtype=np.uint16),
            (np.arange(1000) % 2).astype(bool),
        ]
        with mock.patch("mixle.engines.spectrum._preserved_evidence_sum", _boom):
            for x in native:
                self.assertEqual(accurate_sum(x, 1e-12).status, "ok")

    def test_non_native_evidence_still_takes_the_object_route(self):
        # Negative control for the early-out: everything whose per-element identity actually matters must
        # still reach the evidence-preserving accumulator, not get flattened through float64 first.
        for x in (
            [Decimal("1.0000000000000000000000000000000001"), Decimal("-1")],
            np.array([2**54, 1], dtype=np.int64),  # past float64's exact integer range
            [2**70, -(2**70), 1],  # too large for any native dtype at all
            ["1.5", "2.5"],
        ):
            r = accurate_sum(x, 1e-25)
            self.assertTrue(r.backend.startswith("mpfr"), r.backend)

    def test_native_float64_path_beats_math_fsum(self):
        # math.fsum is the obvious per-element correctly-rounded alternative; a vectorized backend that
        # is *slower* than it has no reason to exist. Best-of-5 on the vectorized side (a scheduler
        # preemption under the parallel runner can inflate any single rep); fsum single-shot, since load
        # only ever slows it, which cannot flip the assertion.
        x = np.random.RandomState(0).randn(200_000)
        t_spectrum = float("inf")
        for _ in range(5):
            t0 = time.perf_counter()
            accurate_sum(x, 1e-12)
            t_spectrum = min(t_spectrum, time.perf_counter() - t0)
        t0 = time.perf_counter()
        math.fsum(x)
        t_fsum = time.perf_counter() - t0
        self.assertLess(t_spectrum, t_fsum)


class CastTest(unittest.TestCase):
    def test_cast_routes_to_backends(self):
        x = np.array([1.0, 2.0, 3.0])
        self.assertIsInstance(cast(x, "dd"), DoubleDouble)
        self.assertIsInstance(cast(x, "fp128"), DoubleDouble)
        self.assertEqual(cast(x, "fp32").dtype, np.float32)
        self.assertEqual(cast(x, 16).dtype, np.float16)
        hp = cast(x, "fp512")  # arbitrary precision -> object array of mpfr/mpf
        self.assertEqual(hp.dtype, object)
        self.assertTrue(np.allclose([float(v) for v in hp], x))


class CastPrecisionSpellingTest(unittest.TestCase):
    """MXR-080-0137: a string spelling ("fp96") and the equivalent integer spelling (96) used to select
    different representation types (MPFR vs. DoubleDouble) for what is supposed to be the identical
    request, and nonpositive integer widths reached malformed native-format construction instead of
    being rejected."""

    def _values(self, cast_result):
        if isinstance(cast_result, DoubleDouble):
            return np.asarray(cast_result.to_float(), dtype=np.float64)
        if isinstance(cast_result, np.ndarray) and cast_result.dtype == object:
            return np.array([float(v) for v in cast_result], dtype=np.float64)
        return np.asarray(cast_result, dtype=np.float64)

    def test_fp96_string_and_96_int_now_select_the_same_representation(self):
        """The audit's exact repro: 'fp96' used to select an MPFR object array while 96 selected a
        DoubleDouble -- incompatible types, costs, and semantics for the identical 96-bit request."""
        x = np.array([1.0, 2.0, 3.0])
        by_string = cast(x, "fp96")
        by_int = cast(x, 96)
        self.assertIs(type(by_string), type(by_int))
        self.assertIsInstance(by_string, DoubleDouble)  # <= 128 total bits: DoubleDouble's own tier
        self.assertTrue(np.allclose(self._values(by_string), self._values(by_int)))

    def test_equivalent_spellings_agree_on_type_and_value_across_every_tier(self):
        """Negative control, swept across the full tiering: native dtype, simulated low-bit float,
        DoubleDouble, and MPFR all agree between the 'fp<bits>' and bare-int spellings of the same
        width, not just at the one width the audit happened to name."""
        x = np.array([1.0, 2.0, 3.0])
        widths = (8, 16, 32, 48, 64, 65, 96, 100, 128, 129, 200, 256, 512, 1024)
        for bits in widths:
            with self.subTest(bits=bits):
                by_string = cast(x, "fp%d" % bits)
                by_int = cast(x, bits)
                self.assertIs(type(by_string), type(by_int), msg="type mismatch at %d bits" % bits)
                self.assertEqual(
                    getattr(by_string, "dtype", None),
                    getattr(by_int, "dtype", None),
                    msg="dtype mismatch at %d bits" % bits,
                )
                self.assertTrue(np.allclose(self._values(by_string), self._values(by_int)))

    def test_dd_fp128_and_128_are_the_same_request(self):
        x = np.array([1.0, 2.0, 3.0])
        by_dd = cast(x, "dd")
        by_fp128 = cast(x, "fp128")
        by_int = cast(x, 128)
        self.assertIs(type(by_dd), type(by_fp128))
        self.assertIs(type(by_fp128), type(by_int))
        self.assertTrue(np.allclose(self._values(by_dd), self._values(by_fp128)))
        self.assertTrue(np.allclose(self._values(by_fp128), self._values(by_int)))

    def test_nonpositive_integer_width_rejected(self):
        """Before the fix, -5 and 0 reached FloatFormat.fp's exponent-bits formula and crashed with an
        opaque 'math domain error' (math.log2 of a nonpositive number) instead of a clear rejection."""
        x = np.array([1.0, 2.0, 3.0])
        for bad in (-5, 0, -1):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError) as ctx:
                    cast(x, bad)
                self.assertNotIn("math domain error", str(ctx.exception))
                self.assertIn("positive", str(ctx.exception))

    def test_width_rejects_boolean_fractional_and_accepts_numpy_integer(self):
        x = np.array([1.0])
        for bad in (True, np.bool_(False), 1.9):
            with self.assertRaises(ValueError):
                cast(x, bad)
        self.assertIsInstance(cast(x, np.int64(96)), DoubleDouble)

    def test_high_precision_name_matches_requested_total_width(self):
        from mixle.engines.highprec import HighPrecisionFormat

        fmt = HighPrecisionFormat(256)
        self.assertEqual(fmt.name, "fp256")
        self.assertEqual(fmt.total_bits, 256)
        with self.assertRaises(ValueError):
            HighPrecisionFormat(True)
        with self.assertRaises(ValueError):
            HighPrecisionFormat(1.9)

    def test_nonpositive_fp_string_width_rejected(self):
        x = np.array([1.0, 2.0, 3.0])
        for bad in ("fp-5", "fp0", "fp-1"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError) as ctx:
                    cast(x, bad)
                self.assertNotIn("math domain error", str(ctx.exception))
                self.assertIn("positive", str(ctx.exception))

    def test_malformed_fp_string_rejected_cleanly(self):
        x = np.array([1.0, 2.0, 3.0])
        for bad in ("fp", "fpabc", "fp9.5"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    cast(x, bad)


class ExactInfiniteSumTest(unittest.TestCase):
    """A one-signed infinity determines the sum exactly, so it is not an overflow to be escalated.

    The log density of a zero-probability event is -inf. Summing log densities that include one is
    exactly -inf, which no additional precision can improve; reporting status="overflow" for it made
    a false claim about a numerical failure and paid for an arbitrary-precision accumulator.
    """

    def test_one_signed_infinity_is_exact_and_stays_on_float64(self):
        for values, expected in (
            ([-1.5, -2.25, -math.inf, -0.5], -math.inf),
            ([-math.inf, -math.inf], -math.inf),
            ([1.0, math.inf], math.inf),
        ):
            with self.subTest(values=values):
                result = accurate_sum(values)
                self.assertEqual(result.value, expected)
                self.assertEqual(result.status, "ok")
                self.assertEqual(result.backend, "float64")
                self.assertEqual(result.rel_error_bound, 0.0)

    def test_indeterminate_and_invalid_inputs_are_not_resolved(self):
        mixed = accurate_sum([math.inf, -math.inf])
        self.assertNotEqual(mixed.status, "ok")  # +inf + -inf has no value to certify
        with self.assertRaises(ValueError):
            accurate_sum([1.0, math.nan])

    def test_finite_inputs_are_unaffected(self):
        result = accurate_sum([-1.5, -2.25, -0.5])
        self.assertAlmostEqual(result.value, -4.25, places=12)
        self.assertEqual(result.status, "ok")


if __name__ == "__main__":
    unittest.main()
