"""Double-double extended precision (mixle.engines.extended): exact EFTs + accurate reductions.

mpmath is used ONLY as a slow correctness oracle here -- never in the library hot path. These tests
prove the error-free transformations are bit-exact and that the vectorized double-double reductions beat
float64 on cancellation while running far faster than mpmath at the same precision.
"""

import time
import unittest
from unittest import mock

import numpy as np
import pytest

from mixle.engines.extended import HAS_DD_KERNELS, DoubleDouble, dd_dot, dd_sum, two_prod, two_sum

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

    def test_two_prod_scales_large_finite_operands(self):
        p, e = two_prod(np.array([1e308]), np.array([1.0]))
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertTrue(np.all(np.isfinite(e)))
        self.assertEqual(float(p[0]), 1e308)


class DoubleDoubleArithmeticTest(unittest.TestCase):
    def test_constructor_normalizes_owns_and_freezes_components(self):
        hi = np.array([0.0])
        lo = np.array([1e308])
        value = DoubleDouble(hi, lo)
        hi[0] = 7.0
        lo[0] = 9.0
        self.assertEqual(float(value.hi[0]), 1e308)
        self.assertEqual(float(value.lo[0]), 0.0)
        self.assertFalse(value.hi.flags.writeable)
        self.assertFalse(value.lo.flags.writeable)
        ulp = np.spacing(np.abs(value.hi))
        self.assertTrue(np.all(np.abs(value.lo) <= 0.5 * ulp))

    def test_constructor_rejects_nan_and_unrepresentable_components(self):
        with self.assertRaises(ValueError):
            DoubleDouble(float("nan"))
        with self.assertRaises(ValueError):  # +inf and -inf together is indeterminate, not an infinity
            DoubleDouble(float("inf"), float("-inf"))
        with self.assertRaises(OverflowError):  # finite operands whose *sum* leaves float64's range
            DoubleDouble(1e308, 1e308)

    def test_constructor_carries_infinities_exactly(self):
        # An infinity that was already in the input is an exactly representable float64 value with no
        # rounding error to carry -- and -inf specifically is the log-density of a zero-probability
        # event, which the EM reductions in this module have to be able to represent.
        for value in (float("inf"), float("-inf")):
            dd = DoubleDouble(value)
            self.assertEqual(float(dd.to_float()), value)
            self.assertEqual(float(dd.lo), 0.0)
        mixed = DoubleDouble(np.array([1.0, -np.inf, 3.0]), np.array([1e-20, 0.0, 1e-20]))
        np.testing.assert_array_equal(mixed.hi, np.array([1.0, -np.inf, 3.0]))
        self.assertEqual(float(mixed.lo[1]), 0.0)  # the infinite entry carries a zero residual...
        self.assertEqual(float(mixed.lo[0]), 1e-20)  # ...without disturbing its finite neighbours

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
    def test_large_finite_dot_remains_finite(self):
        result = dd_dot([1e308], [1.0])
        self.assertTrue(np.all(np.isfinite(result.hi)))
        self.assertTrue(np.all(np.isfinite(result.lo)))
        self.assertEqual(float(result.to_float()), 1e308)

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


class DdDotValidationTest(unittest.TestCase):
    """MXR-080-0132: dd_dot must reject mismatched-length vectors on every dispatch path.

    The compiled kernel is only ever reached when sizes already match, but the pure-numpy fallback used
    to skip straight to ``two_prod``/``dd_sum`` with no length check at all -- numpy's elementwise ops
    then broadcast the length-1 side, silently computing a *different* operation (not a dot product) and
    returning a plausible-looking number instead of raising.
    """

    def test_mismatched_length_raises_on_fallback_path(self):
        # Force the pure-numpy fallback regardless of what happens to be built in this environment.
        with mock.patch("mixle.engines.extended.HAS_DD_KERNELS", False):
            with self.assertRaises(ValueError):
                # Audit's exact repro: a 1-element vector dotted with a 3-element one used to broadcast
                # and silently return 24.0 (4 * (1+2+3)) instead of rejecting the mismatched shapes.
                dd_dot(np.array([4.0]), np.array([1.0, 2.0, 3.0]))

    def test_mismatched_length_raises_before_reaching_compiled_kernel(self):
        # Force the "compiled available" branch with a stand-in kernel that fails the test if it is ever
        # actually called -- proves the length check runs once, before dispatch, not separately inside
        # each arm (so the two paths can never diverge on it again).
        def _boom(a, b):
            raise AssertionError("compiled kernel must not be reached for mismatched-length inputs")

        with mock.patch("mixle.engines.extended.HAS_DD_KERNELS", True):
            with mock.patch("mixle.engines.extended._dd_dot_c", _boom, create=True):
                with self.assertRaises(ValueError):
                    dd_dot(np.array([4.0]), np.array([1.0, 2.0, 3.0]))

    @unittest.skipUnless(
        HAS_DD_KERNELS, "compiled _dd_kernels not built (run mixle.engines.build_kernels.compile_dd_kernels)"
    )
    def test_mismatched_length_raises_with_the_real_compiled_kernel(self):
        # Belt-and-suspenders: when the accelerator is actually built in this environment, the real
        # dispatch must reject too, not just the monkeypatched stand-in above.
        with self.assertRaises(ValueError):
            dd_dot(np.array([4.0]), np.array([1.0, 2.0, 3.0]))

    def test_equal_length_still_dot_products_correctly_on_fallback_path(self):
        # Negative control: the new check must not disturb a legitimate equal-length dot. Hand-verifiable
        # exact value (no rounding at these magnitudes): 1*4 + 2*5 + 3*6 = 32.
        with mock.patch("mixle.engines.extended.HAS_DD_KERNELS", False):
            r = dd_dot(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
        self.assertEqual(float(r.to_float()), 32.0)
        self.assertEqual(float(r.hi), 32.0)
        self.assertEqual(float(r.lo), 0.0)

    def test_equal_length_dispatches_to_the_compiled_kernel_when_available(self):
        # Wiring check, not a numerics check (the real kernel's accuracy is dd_kernels_test.py's job):
        # confirm dd_dot both calls the compiled kernel and correctly plumbs its (hi, lo) back out.
        stub = mock.Mock(return_value=(32.0, 0.0))
        with mock.patch("mixle.engines.extended.HAS_DD_KERNELS", True):
            with mock.patch("mixle.engines.extended._dd_dot_c", stub, create=True):
                r = dd_dot(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
        stub.assert_called_once()
        self.assertEqual(float(r.to_float()), 32.0)


class ReductionInputDomainTest(unittest.TestCase):
    """The reductions' accepted input domain must match the one mixle's EM hot paths actually produce.

    The module docstring says these exist for "log-sum-exp" in EM -- which is exactly where ``-inf``
    (the log-density of a zero-probability event) and products that underflow to zero show up. Guards
    that reject those reject the workload, and guards that live in only one of the two dd_dot dispatch
    arms let the compiled and pure-numpy paths accept different inputs.
    """

    # Always exercise the pure-numpy fallback; add the compiled arm when one is actually built.
    _DISPATCH = (False,) + ((True,) if HAS_DD_KERNELS else ())

    def test_dd_sum_carries_a_zero_probability_log_density(self):
        self.assertEqual(float(dd_sum([-np.inf, 1.0, 2.0]).to_float()), -np.inf)
        self.assertEqual(float(dd_sum(np.full(4, -np.inf)).to_float()), -np.inf)
        self.assertEqual(float(dd_sum([np.inf, 1.0]).to_float()), np.inf)

    def test_dd_sum_still_fails_closed_where_there_is_no_sum(self):
        with self.assertRaises(ValueError):  # NaN has no sum
            dd_sum([np.nan, 1.0])
        with self.assertRaises(ValueError):  # +inf - inf is indeterminate, not an infinity
            dd_sum([np.inf, -np.inf])

    def test_finite_dd_sum_is_untouched_by_the_infinite_short_circuit(self):
        # Negative control: the short-circuit must not perturb the ordinary path's accuracy.
        rng = np.random.RandomState(11)
        x = rng.randn(4000) * 10.0 ** rng.randint(-12, 12, 4000)
        with mpmath.workprec(400):
            exact = mpmath.fsum(mpmath.mpf(float(v)) for v in x)
        self.assertLess(abs(float(dd_sum(x).to_float()) - float(exact)), abs(float(exact)) * 1e-15 + 1e-300)

    def test_one_underflowing_element_does_not_kill_the_dot_product(self):
        # The audit's exact repro. 1e-200 * 1e-200 rounds to zero -- it is worth less than one smallest
        # subnormal against the other term -- so the dot is 1.0, on both dispatch paths.
        for available in self._DISPATCH:
            with self.subTest(compiled=repr(available)):
                with mock.patch("mixle.engines.extended.HAS_DD_KERNELS", available):
                    self.assertEqual(float(dd_dot([1e-200, 1.0], [1e-200, 1.0]).to_float()), 1.0)

    def test_two_prod_alone_still_fails_closed_on_underflow(self):
        # A lone product has nothing to be negligible against: (0, 0) does not reproduce a*b, so the
        # error-free transform must still refuse. Only the *reduction* gets to call it negligible.
        with self.assertRaises(ArithmeticError):
            two_prod(np.array([1e-200]), np.array([1e-200]))

    def test_both_dispatch_paths_reject_the_same_inputs(self):
        cases = {
            "mismatched length": ((np.array([4.0]), np.array([1.0, 2.0, 3.0])), ValueError),
            "non-finite operand": ((np.array([np.inf, 1.0]), np.array([1.0, 1.0])), ValueError),
            "overflowing element": ((np.array([1e300, -1e300]), np.array([1e300, 1e300])), OverflowError),
        }
        for label, (args, expected) in cases.items():
            for available in self._DISPATCH:
                with self.subTest(case=repr(label), compiled=repr(available)):
                    with mock.patch("mixle.engines.extended.HAS_DD_KERNELS", available):
                        with self.assertRaises(expected):
                            dd_dot(*args)


class DoubleDoubleCostModelTest(unittest.TestCase):
    """The constructor is on the hot path -- every ``__add__`` and ``__mul__`` builds one -- so its
    validation must not walk the data more times than it needs to. It used to make four full finiteness
    passes (both inputs, then both normalized outputs) plus two defensive copies of arrays ``two_sum``
    had just allocated. One pass over the normalized leading component subsumes all four: ``hi + lo`` is
    non-finite whenever either input is, or the sum overflows.
    """

    def test_hot_path_makes_a_single_finiteness_pass(self):
        hi = np.random.RandomState(0).randn(1000)
        lo = np.zeros_like(hi)
        with mock.patch("numpy.isfinite", wraps=np.isfinite) as spy:
            DoubleDouble(hi, lo)
        self.assertEqual(spy.call_count, 1)

    def test_components_stay_owned_and_frozen_without_defensive_copies(self):
        # Dropping the copies must not reintroduce aliasing: two_sum allocates its own results, so the
        # value cannot be perturbed by a later write to the arrays it was built from.
        hi = np.array([1.0, 2.0])
        lo = np.array([1e-18, 2e-18])
        value = DoubleDouble(hi, lo)
        hi[:] = 99.0
        lo[:] = 99.0
        np.testing.assert_array_equal(value.hi, np.array([1.0, 2.0]))
        self.assertFalse(value.hi.flags.writeable)
        self.assertFalse(value.lo.flags.writeable)


class SpeedVsOracleTest(unittest.TestCase):
    def test_dd_sum_is_far_faster_than_mpmath(self):
        # The whole point: vectorized double-double gives ~106-bit accuracy without mpmath's per-object cost.
        rng = np.random.RandomState(7)
        x = rng.randn(20000)

        # best-of-5: under the loaded parallel runner a scheduler preemption can inflate any single
        # rep by orders of magnitude; the minimum is the honest speed of the vectorized path. The
        # mpmath side stays single-shot -- load only ever slows it, which cannot flip the assertion.
        t_dd = float("inf")
        for _ in range(5):
            t0 = time.perf_counter()
            dd_sum(x)
            t_dd = min(t_dd, time.perf_counter() - t0)

        t0 = time.perf_counter()
        with mpmath.workprec(106):
            mpmath.fsum(mpmath.mpf(float(v)) for v in x)
        t_mp = time.perf_counter() - t0

        self.assertLess(t_dd, t_mp)  # double-double must be faster than the mpmath oracle at equal precision


if __name__ == "__main__":
    unittest.main()
