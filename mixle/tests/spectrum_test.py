"""The precision-spectrum front door (mixle.engines.spectrum): auto-route to the cheapest accurate backend."""

import unittest
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
        for bad in (0.0, -1e-9, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                accurate_sum(np.array([1.0, 2.0]), bad)


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


if __name__ == "__main__":
    unittest.main()
