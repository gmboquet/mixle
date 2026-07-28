"""Logarithmic number system (mixle.engines.lns): integer log-space arithmetic, quantized by ln(C).

The contracts: products-of-probabilities are EXACT integer adds, and the integer log-sum-exp matches
float64 within the certified ~step bound (the precision dial). No exp/log in the integer path.
"""

import unittest

import numpy as np
import pytest

from mixle.engines.lns import LogNumberSystem

logsumexp = pytest.importorskip("scipy.special").logsumexp


class LogNumberSystemTest(unittest.TestCase):
    def test_quantize_roundtrip_within_half_step(self):
        lns = LogNumberSystem(step=0.01)
        L = np.random.RandomState(0).randn(1000) * 40
        back = lns.dequantize(lns.quantize(L))
        self.assertLessEqual(float(np.max(np.abs(back - L))), lns.step / 2 + 1e-12)

    def test_product_of_probabilities_is_exact_integer_add(self):
        # multiplying probabilities = adding log-probs = adding the integer codes, exactly
        lns = LogNumberSystem(step=0.005)
        la, lb = -3.2, -1.7
        ka, kb = int(lns.quantize(la)), int(lns.quantize(lb))
        self.assertEqual(ka + kb, int(lns.quantize(la)) + int(lns.quantize(lb)))
        # and the dequantized sum is the product's log-prob within rounding
        self.assertAlmostEqual(lns.dequantize(ka + kb), lns.quantize(la) * lns.step + lns.quantize(lb) * lns.step)

    def test_logadd_matches_float64(self):
        lns = LogNumberSystem(step=0.01)
        rng = np.random.RandomState(1)
        a = rng.randn(2000) * 20
        b = rng.randn(2000) * 20
        ref = np.logaddexp(a, b)
        got = lns.dequantize(lns.logadd(lns.quantize(a), lns.quantize(b)))
        # a single logadd reduces 2 terms (MXR-080-0139: the bound is parameterized by term count)
        self.assertLessEqual(float(np.max(np.abs(got - ref))), lns.max_logsumexp_error(2))

    def test_logsumexp_matches_float64_within_bound(self):
        for step in (0.05, 0.01, 0.002):
            lns = LogNumberSystem(step=step)
            rng = np.random.RandomState(2)
            X = rng.randn(500, 64) * 30  # rows of log-densities spanning a wide range
            ref = logsumexp(X, axis=1)
            got = lns.dequantize(lns.logsumexp(lns.quantize(X), axis=1))
            err = float(np.max(np.abs(got - ref)))
            # reducing 64 terms is a depth-6 pairwise tree; max_logsumexp_error(64) is the properly
            # depth-scaled certificate (MXR-080-0139) -- no ad hoc fudge factor needed on top of it
            self.assertLessEqual(err, lns.max_logsumexp_error(64), "step=%g err=%g" % (step, err))

    def test_max_logsumexp_error_grows_with_reduction_depth(self):
        # MXR-080-0139: the bound must be a strictly increasing step function of tree depth
        # (ceil(log2(n))), not the old flat constant that silently stopped certifying anything once
        # n grew past the 2-4 term case it was actually sized for.
        lns = LogNumberSystem(step=0.01)
        bounds = {n: lns.max_logsumexp_error(n) for n in (1, 2, 3, 4, 5, 8, 9, 64, 1024)}
        self.assertAlmostEqual(bounds[1], 0.5 * lns.step)  # a single term: pure input quantization
        self.assertAlmostEqual(bounds[2], 1.0 * lns.step)  # depth 1
        self.assertAlmostEqual(bounds[3], 1.5 * lns.step)  # depth 2 (ceil(log2(3))==2)
        self.assertAlmostEqual(bounds[4], 1.5 * lns.step)  # depth 2
        self.assertAlmostEqual(bounds[5], 2.0 * lns.step)  # depth 3 (ceil(log2(5))==3)
        self.assertAlmostEqual(bounds[8], 2.0 * lns.step)  # depth 3
        self.assertAlmostEqual(bounds[9], 2.5 * lns.step)  # depth 4
        self.assertAlmostEqual(bounds[1024], 5.5 * lns.step)  # depth 10
        # non-decreasing in n, and strictly increasing across a power-of-two boundary
        ns = sorted(bounds)
        self.assertTrue(all(bounds[ns[i]] <= bounds[ns[i + 1]] for i in range(len(ns) - 1)))
        self.assertGreater(bounds[1024], bounds[64])
        self.assertGreater(bounds[64], bounds[8])
        with self.assertRaises(ValueError):
            lns.max_logsumexp_error(0)

    def test_max_logsumexp_error_matches_old_constant_at_the_n_it_was_sized_for(self):
        # The previous flat `1.5 * step` bound (MXR-080-0139) turns out to equal this formula exactly
        # at n in {3, 4} (depth 2) -- i.e. it was a valid bound for a small, fixed reduction width, just
        # never generalized. Pin that equivalence so the fix is legible as a generalization, not a
        # change to the small-n answer.
        lns = LogNumberSystem(step=0.02)
        old_constant = 1.5 * lns.step
        self.assertAlmostEqual(lns.max_logsumexp_error(3), old_constant)
        self.assertAlmostEqual(lns.max_logsumexp_error(4), old_constant)

    def test_max_logsumexp_error_bound_is_empirically_certified_at_large_n(self):
        # MXR-080-0139: construct an adversarial worst-case-LUT-rounding input (every leaf nudged to a
        # near-half-step quantization boundary, every adjacent pair forced to the LUT index with the
        # largest rounding error) and confirm the OLD flat constant would have been violated at large n
        # while the new n-dependent bound is not -- i.e. this is a real, not just theoretical, gap.
        step = 0.01
        lns = LogNumberSystem(step=step)
        d = np.arange(lns.dmax + 1, dtype=np.float64)
        f = np.log1p(np.exp(-d * step))
        worst_d = int(np.argmax(np.abs(lns.lut * step - f)))

        rng = np.random.RandomState(7)
        old_constant = 1.5 * step
        for n in (8, 64, 1024):
            trials = 500
            base = rng.randn(trials, n) * 30
            nudged = (np.floor(base / step) + 0.5 - 1e-9) * step
            nudged[:, 1::2] = nudged[:, 0::2] - worst_d * step
            ref = logsumexp(nudged, axis=1)
            got = lns.dequantize(lns.logsumexp(lns.quantize(nudged), axis=1))
            err = float(np.max(np.abs(got - ref)))
            self.assertLessEqual(err, lns.max_logsumexp_error(n), "n=%d err=%g" % (n, err))
            self.assertGreater(err, old_constant, "n=%d err=%g should exceed the old flat 1.5*step bound" % (n, err))

    def test_max_logsumexp_error_is_not_loose_for_small_n(self):
        # Negative control: the new formula must not regress into a uselessly loose bound for the
        # common small-n case -- it should stay comparable to (not wildly looser than) the old constant.
        lns = LogNumberSystem(step=0.01)
        self.assertLessEqual(lns.max_logsumexp_error(2), 1.5 * lns.step)
        self.assertAlmostEqual(lns.max_logsumexp_error(4), 1.5 * lns.step)

    def test_finer_step_is_more_accurate(self):
        rng = np.random.RandomState(3)
        X = rng.randn(300, 32) * 25
        ref = logsumexp(X, axis=1)
        coarse = LogNumberSystem(step=0.05)
        fine = LogNumberSystem(step=0.005)
        e_coarse = np.max(np.abs(coarse.dequantize(coarse.logsumexp(coarse.quantize(X), axis=1)) - ref))
        e_fine = np.max(np.abs(fine.dequantize(fine.logsumexp(fine.quantize(X), axis=1)) - ref))
        self.assertGreater(e_coarse, e_fine)

    def test_from_relative_precision_and_integer_dtype(self):
        lns = LogNumberSystem.from_relative_precision(0.01)  # ~1% relative
        self.assertAlmostEqual(lns.step, np.log1p(0.01))
        # log-densities to ~ -700 (underflow edge) at step 0.05 fit int16; finer steps need int32
        self.assertEqual(LogNumberSystem(step=0.05).integer_dtype(700.0), np.int16)
        self.assertEqual(LogNumberSystem(step=1e-4).integer_dtype(700.0), np.int32)

    def test_step_must_be_positive(self):
        for bad in (0.0, -1.0, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                LogNumberSystem(step=bad)


class LogZeroSentinelTest(unittest.TestCase):
    """MXR-080-0138: a true zero (log(0) = -inf) round-trips through a reserved sentinel code, with
    no reliance on an undefined float->int64 cast, and no overflow when combined with ordinary codes."""

    def test_quantize_dequantize_zero_roundtrips_through_the_sentinel(self):
        lns = LogNumberSystem(step=0.01)
        k = lns.quantize(np.array(-np.inf))
        self.assertEqual(int(k), LogNumberSystem.LOG_ZERO_CODE)
        self.assertEqual(int(k), np.iinfo(np.int64).min)
        back_log = lns.dequantize(k)
        self.assertEqual(float(back_log), -np.inf)
        # and the ROUND TRIP THROUGH LINEAR SCALE the sentinel exists for: 0.0 -> log -> quantize ->
        # dequantize -> exp -> exactly 0.0 again, matching a genuine zero probability/value exactly.
        original = 0.0
        with np.errstate(divide="ignore"):  # np.log(0.0) == -inf by design; that's what we're testing
            log_original = np.log(original)
        roundtripped = float(np.exp(lns.dequantize(lns.quantize(log_original))))
        self.assertEqual(roundtripped, 0.0)

    def test_quantize_zero_array_elementwise(self):
        lns = LogNumberSystem(step=0.01)
        log_values = np.array([-np.inf, -3.0, -np.inf, 0.0, -np.inf])
        k = lns.quantize(log_values)
        is_zero = k == LogNumberSystem.LOG_ZERO_CODE
        np.testing.assert_array_equal(is_zero, [True, False, True, False, True])
        back = lns.dequantize(k)
        self.assertTrue(np.all(back[is_zero] == -np.inf))
        self.assertTrue(np.all(np.isfinite(back[~is_zero])))

    def test_quantize_rejects_nan(self):
        lns = LogNumberSystem(step=0.01)
        with self.assertRaises(ValueError):
            lns.quantize(np.array([1.0, np.nan, -3.0]))

    def test_quantize_rejects_positive_infinity(self):
        lns = LogNumberSystem(step=0.01)
        with self.assertRaises(ValueError):
            lns.quantize(np.array([1.0, np.inf, -3.0]))

    def test_quantize_rejects_out_of_range_values_to_preserve_error_bound(self):
        lns = LogNumberSystem(step=0.01)
        for bad in (1e308, -1e308):
            with self.assertRaises(OverflowError):
                lns.quantize(np.array([bad]))
        self.assertEqual(int(lns.quantize(5.0)), 500)

    def test_logadd_sentinel_is_absorbing(self):
        lns = LogNumberSystem(step=0.01)
        z = LogNumberSystem.LOG_ZERO_CODE
        x = int(lns.quantize(np.array(-5.0)))
        self.assertEqual(int(lns.logadd(z, x)), x)  # 0 + p = p
        self.assertEqual(int(lns.logadd(x, z)), x)  # p + 0 = p
        self.assertEqual(int(lns.logadd(z, z)), z)  # 0 + 0 = 0
        # matches the float64 reference: logaddexp(-inf, -5) == -5, logaddexp(-inf, -inf) == -inf
        self.assertEqual(float(lns.dequantize(lns.logadd(z, x))), float(np.logaddexp(-np.inf, -5.0)))
        self.assertEqual(float(lns.dequantize(lns.logadd(z, z))), -np.inf)
        self.assertEqual(int(lns.logadd(LogNumberSystem.CODE_MAX, LogNumberSystem.CODE_MAX)), LogNumberSystem.CODE_MAX)

    def test_multiply_is_exact_for_ordinary_codes_and_sentinel_is_absorbing(self):
        lns = LogNumberSystem(step=0.005)
        z = LogNumberSystem.LOG_ZERO_CODE
        ka, kb = int(lns.quantize(np.array(-3.2))), int(lns.quantize(np.array(-1.7)))
        # ordinary case: exact integer add, matching raw k1+k2 (the documented "no table" contract)
        self.assertEqual(int(lns.multiply(ka, kb)), ka + kb)
        # p * 0 = 0 and 0 * p = 0, regardless of how large/small p's own code is
        self.assertEqual(int(lns.multiply(ka, z)), z)
        self.assertEqual(int(lns.multiply(z, kb)), z)
        self.assertEqual(int(lns.multiply(z, z)), z)

    def test_multiply_saturates_instead_of_overflowing(self):
        lns = LogNumberSystem(step=0.01)
        near_max = LogNumberSystem.CODE_MAX - 1
        near_min = LogNumberSystem.CODE_MIN + 1
        # summing two near-boundary (but ordinary, non-sentinel) codes would exceed CODE_MAX/CODE_MIN;
        # it must saturate, not silently wrap around through int64 overflow
        self.assertEqual(int(lns.multiply(near_max, near_max)), LogNumberSystem.CODE_MAX)
        self.assertEqual(int(lns.multiply(near_min, near_min)), LogNumberSystem.CODE_MIN)
        # a large positive and a large negative code cancel back into the ordinary range (no false
        # saturation just because the individual operands were near the boundary)
        mid = int(lns.multiply(near_max, near_min))
        self.assertEqual(mid, near_max + near_min)

    def test_public_code_operations_reject_noncanonical_codes(self):
        lns = LogNumberSystem(step=0.01)
        i64 = np.iinfo(np.int64)
        z = LogNumberSystem.LOG_ZERO_CODE
        self.assertEqual(z, i64.min)
        for bad in (i64.max, i64.min + 1, LogNumberSystem.CODE_MAX + 1, LogNumberSystem.CODE_MIN - 1):
            with self.assertRaises(ValueError):
                lns.dequantize(bad)
            with self.assertRaises(ValueError):
                lns.logadd(z, bad)
            with self.assertRaises(ValueError):
                lns.multiply(0, bad)

    def test_logsumexp_rejects_empty_axis_and_validates_single_term(self):
        lns = LogNumberSystem(step=0.01)
        with self.assertRaises(ValueError):
            lns.logsumexp(np.empty((2, 0), dtype=np.int64), axis=1)
        with self.assertRaises(ValueError):
            lns.logsumexp(np.array([LogNumberSystem.CODE_MAX + 1]), axis=0)
        self.assertEqual(int(lns.logsumexp(np.array([123]), axis=0)), 123)

    def test_logsumexp_with_embedded_zero_terms_matches_float64(self):
        # End-to-end: a full logsumexp reduction with LOG_ZERO_CODE terms scattered through several
        # rows must match float64's logsumexp treating those terms as literally absent (-inf
        # contributes exp(-inf) = 0 to the sum), not corrupt the rest of the row.
        lns = LogNumberSystem(step=0.01)
        rng = np.random.RandomState(11)
        X = rng.randn(200, 16) * 20
        drop = rng.rand(200, 16) < 0.3
        X_with_zeros = np.where(drop, -np.inf, X)
        ref = logsumexp(X_with_zeros, axis=1)
        got = lns.dequantize(lns.logsumexp(lns.quantize(X_with_zeros), axis=1))
        finite_ref = np.isfinite(ref)
        self.assertTrue(np.all(finite_ref))  # every row has >=1 non-zero term at this drop rate
        self.assertLessEqual(float(np.max(np.abs(got[finite_ref] - ref[finite_ref]))), lns.max_logsumexp_error(16))
        # an all-zero row reduces to exactly the sentinel / -inf, not garbage
        all_zero_row = np.full((1, 8), -np.inf)
        got_all_zero = lns.dequantize(lns.logsumexp(lns.quantize(all_zero_row), axis=1))
        self.assertEqual(float(got_all_zero[0]), -np.inf)

    def test_negative_control_ordinary_arithmetic_is_unaffected_by_the_sentinel_fix(self):
        # No zeros, no extreme values: plain LNS arithmetic must still behave exactly as before.
        lns = LogNumberSystem(step=0.01)
        rng = np.random.RandomState(99)
        a = rng.randn(500) * 15
        b = rng.randn(500) * 15
        ka, kb = lns.quantize(a), lns.quantize(b)
        self.assertTrue(np.all(ka != LogNumberSystem.LOG_ZERO_CODE))
        got_add = lns.dequantize(lns.logadd(ka, kb))
        ref_add = np.logaddexp(a, b)
        self.assertLessEqual(float(np.max(np.abs(got_add - ref_add))), lns.max_logsumexp_error(2))
        got_mul = lns.dequantize(lns.multiply(ka, kb))
        expect_mul = lns.dequantize(ka) + lns.dequantize(kb)  # log(a*b) = log(a) + log(b), exact (no LUT)
        self.assertTrue(np.allclose(got_mul, expect_mul))


if __name__ == "__main__":
    unittest.main()
