"""Affine error tracing (mixle.engines.affine): tighter than intervals, ULP-injection precision dial."""

import unittest

import numpy as np

from mixle.engines.affine import AffineForm, allocate_precision, unit_roundoff
from mixle.engines.error_tracing import Interval


class UnitRoundoffTest(unittest.TestCase):
    def test_lookup(self):
        self.assertEqual(unit_roundoff("float32"), 2.0**-24)
        self.assertEqual(unit_roundoff("float64"), 2.0**-53)
        self.assertEqual(unit_roundoff("dd"), 2.0**-106)
        self.assertEqual(unit_roundoff(np.float32), 2.0**-24)
        with self.assertRaises(ValueError):
            unit_roundoff("int8")


class AffineSoundnessTest(unittest.TestCase):
    def test_constructor_rejects_invalid_state_and_owns_immutable_evidence(self):
        for center in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                AffineForm.constant(center)
        with self.assertRaises(ValueError):
            AffineForm.uncertain(1.0, float("nan"))
        with self.assertRaises(ValueError):
            AffineForm([1.0, 2.0], {1: np.ones(3)})

        center = np.array([1.0])
        coefficient = np.array([0.1])
        terms = {1: coefficient}
        form = AffineForm(center, terms)
        center[0] = 99.0
        coefficient[0] = 7.0
        terms[1] = np.array([8.0])
        self.assertEqual(float(form.center[0]), 1.0)
        self.assertAlmostEqual(form.max_radius(), 0.1)
        self.assertFalse(form.center.flags.writeable)
        self.assertFalse(form.terms[1].flags.writeable)
        with self.assertRaises(TypeError):
            form.terms[2] = np.array([1.0])

    def test_form_contains_actual_values_incl_nonlinear(self):
        rng = np.random.RandomState(0)
        a0, ra = 5.0, 0.1
        b0, rb = 3.0, 0.2
        a = AffineForm.uncertain(a0, ra)
        b = AffineForm.uncertain(b0, rb)
        f = a * b + a  # has a quadratic (nonlinear) term -> lumped symbol must keep it sound
        for _ in range(200):
            ea, eb = rng.uniform(-1, 1, 2)
            true = (a0 + ra * ea) * (b0 + rb * eb) + (a0 + ra * ea)
            self.assertTrue(bool(f.contains(true)), "not contained: %r" % true)

    def test_array_valued_forms(self):
        rng = np.random.RandomState(1)
        x = rng.randn(1000)
        a = AffineForm.uncertain(x, 0.05)
        f = a * a  # x^2 with uncertainty
        for _ in range(50):
            e = rng.uniform(-1, 1)
            true = (x + 0.05 * e) ** 2
            self.assertTrue(np.all(f.contains(true)))


class AffineTighterThanIntervalTest(unittest.TestCase):
    def test_cancellation_recovers_tight_bound(self):
        # (a + b) - a : a's noise symbol cancels in affine; an interval would double a's width.
        a = AffineForm.uncertain(5.0, 0.1)
        b = AffineForm.uncertain(3.0, 0.01)
        expr = (a + b) - a
        affine_r = expr.max_radius()

        ia, ib = Interval(4.9, 5.1), Interval(2.99, 3.01)
        interval_expr = (ia + ib) - ia
        interval_r = 0.5 * float(interval_expr.width())

        self.assertLess(affine_r, 0.02)  # ~ b's uncertainty only
        self.assertGreater(interval_r, 0.15)  # interval ~ 2*a + b
        self.assertLess(affine_r, interval_r / 5.0)  # decisively tighter
        # and still sound: the true value 3 +- 0.01 is enclosed
        self.assertTrue(bool(expr.contains(3.0)))


class PrecisionDialTest(unittest.TestCase):
    def test_inject_roundoff_adds_expected_radius(self):
        f = AffineForm.constant(np.array([100.0, -8.0]))
        injected = f.inject_roundoff("float32")
        # radius ~ u(f32) * |center|
        self.assertTrue(np.allclose(injected.radius(), 2.0**-24 * np.abs(f.center), rtol=1e-6))

    def test_allocate_precision_picks_cheapest_adequate(self):
        # 1000 ops on magnitude ~1.0, tolerate 1e-4 abs error
        r = allocate_precision(1.0, 1000, 1e-4)
        self.assertEqual(r.dtype, "float32")  # 1000*2^-24 ~ 6e-5 < 1e-4
        self.assertEqual(r.status, "heuristic_fit")
        self.assertTrue(r.heuristic_met_target)
        self.assertFalse(r.met_target)
        self.assertLessEqual(r.estimated_abs_error, 1e-4)
        self.assertEqual(r.target_abs_error, 1e-4)
        # same ops but a tiny tolerance -> needs float64
        self.assertEqual(allocate_precision(1.0, 1000, 1e-10).dtype, "float64")
        # huge magnitude + tight tol -> escalate to double-double
        self.assertEqual(allocate_precision(1e6, 100000, 1e-6).dtype, "dd")
        # float16 suffices for a loose budget
        self.assertEqual(allocate_precision(1.0, 10, 1e-1).dtype, "float16")


class AllocatePrecisionFailClosedTest(unittest.TestCase):
    """MXR-080-0154: allocate_precision used to accept nonsensical error models -- a negative op_count
    flipped the estimated error negative and misselected float16, a negative target_abs_error fell
    through to silently returning qd as if it certified an impossible bound, NaN/Inf controls rode
    comparison-with-NaN-is-False into an unprincipled fallthrough, and even a legitimately-selected qd
    never indicated when its own estimated error still missed the target."""

    def test_negative_op_count_no_longer_misselects_float16(self):
        # Audit's exact repro shape: before the fix, a negative op_count flipped the estimated error
        # negative, which spuriously satisfies float16's budget check first no matter how large the
        # magnitude or how tight the target -- allocate_precision(1e6, -100000, 1e-30) used to return
        # "float16" outright. Now it must be rejected instead.
        with self.assertRaises(ValueError):
            allocate_precision(1e6, -100000, 1e-30)

    def test_negative_target_abs_error_rejected_instead_of_silently_returning_qd(self):
        # Before the fix, every dtype's comparison failed against a negative target and the loop fell
        # through to returning "qd" unconditionally -- as if qd had certified an impossible bound. No
        # dtype can ever satisfy a negative error target, so this must now raise, not return anything.
        with self.assertRaises(ValueError):
            allocate_precision(1.0, 1000, -1e-10)

    def test_zero_target_abs_error_rejected(self):
        # A target of exactly zero is unmeetable by any finite-precision format; treated as invalid
        # input (ValueError), matching plan_heterogeneous's and accurate_sum's target_error > 0 contract.
        with self.assertRaises(ValueError):
            allocate_precision(1.0, 1000, 0.0)

    def test_nonfinite_target_abs_error_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(target_abs_error=bad):
                with self.assertRaises(ValueError):
                    allocate_precision(1.0, 1000, bad)

    def test_nonfinite_center_magnitude_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(center_magnitude=bad):
                with self.assertRaises(ValueError):
                    allocate_precision(bad, 1000, 1e-6)

    def test_negative_center_magnitude_rejected(self):
        with self.assertRaises(ValueError):
            allocate_precision(-5.0, 1000, 1e-6)

    def test_nonfinite_op_count_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(op_count=bad):
                with self.assertRaises(ValueError):
                    allocate_precision(1.0, bad, 1e-6)

    def test_qd_insufficient_is_signaled_not_silently_returned_as_certified(self):
        # A target far tighter than qd's own roundoff can ever certify: qd's estimated error here is
        # ~1.5e-54 (10**10 ops * 2**-212 * magnitude 1.0), nowhere near the 1e-70 ask. Before the fix
        # this silently returned "qd" with no indication the target was actually missed.
        r = allocate_precision(1.0, 10**10, 1e-70)
        self.assertEqual(r.dtype, "qd")
        self.assertEqual(r.status, "heuristic_insufficient")
        self.assertFalse(r.met_target)
        self.assertGreater(r.estimated_abs_error, r.target_abs_error)
        self.assertEqual(r.target_abs_error, 1e-70)

    def test_achievable_endpoint_estimate_is_explicitly_not_a_certificate(self):
        r = allocate_precision(1.0, 1000, 1e-4)
        self.assertEqual(r.dtype, "float32")
        self.assertEqual(r.status, "heuristic_fit")
        self.assertTrue(r.heuristic_met_target)
        self.assertFalse(r.met_target)
        self.assertLessEqual(r.estimated_abs_error, r.target_abs_error)

    def test_zero_endpoint_cannot_create_a_false_zero_error_certificate(self):
        r = allocate_precision(0.0, 1e12, 1e-300)
        self.assertEqual(r.estimated_abs_error, 0.0)
        self.assertTrue(r.heuristic_met_target)
        self.assertFalse(r.met_target)

    def test_zero_op_count_and_zero_magnitude_are_legitimate_nonnegative_inputs(self):
        # Zero is a valid (not rejected) op_count/magnitude -- no operations or a zero-sized quantity
        # legitimately accumulates zero error, satisfying even the cheapest dtype.
        r = allocate_precision(0.0, 0, 1e-6)
        self.assertEqual(r.dtype, "float16")
        self.assertEqual(r.status, "heuristic_fit")
        self.assertEqual(r.estimated_abs_error, 0.0)


if __name__ == "__main__":
    unittest.main()
