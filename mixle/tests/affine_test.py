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
        self.assertEqual(allocate_precision(1.0, 1000, 1e-4), "float32")  # 1000*2^-24 ~ 6e-5 < 1e-4
        # same ops but a tiny tolerance -> needs float64
        self.assertEqual(allocate_precision(1.0, 1000, 1e-10), "float64")
        # huge magnitude + tight tol -> escalate to double-double
        self.assertEqual(allocate_precision(1e6, 100000, 1e-6), "dd")
        # float16 suffices for a loose budget
        self.assertEqual(allocate_precision(1.0, 10, 1e-1), "float16")


if __name__ == "__main__":
    unittest.main()
