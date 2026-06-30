"""The precision-spectrum front door (mixle.engines.spectrum): auto-route to the cheapest accurate backend."""

import unittest

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
        val, backend = accurate_sum(x, 1e-12)
        self.assertEqual(backend, "float64")
        self.assertAlmostEqual(val, self._true(x), places=6)

    def test_moderate_cancellation_escalates_to_double_double(self):
        rng = np.random.RandomState(1)
        x = np.tile(np.array([1e16, 1.0, -1e16]), 1000)  # true sum 1000, cond ~ 2^54
        rng.shuffle(x)
        val, backend = accurate_sum(x, 1e-12)
        self.assertEqual(backend, "dd")
        self.assertAlmostEqual(val, self._true(x), delta=1e-6)

    def test_catastrophic_cancellation_escalates_to_mpfr(self):
        x = np.array([1e40, 1.0, -1e40])  # true sum 1; needs > double-double's ~106 bits
        val, backend = accurate_sum(x, 1e-12)
        self.assertTrue(backend.startswith("mpfr"), backend)
        self.assertAlmostEqual(val, 1.0, places=6)

    def test_result_always_meets_target_vs_oracle(self):
        rng = np.random.RandomState(2)
        for _ in range(8):
            x = rng.randn(8000) * 10.0 ** rng.randint(-8, 8, 8000)
            val, _ = accurate_sum(x, 1e-12)
            true = self._true(x)
            rel = abs(val - true) / max(abs(true), 1e-300)
            self.assertLess(rel, 1e-9)  # comfortably within target across regimes

    def test_certificate_reports_condition_and_bound(self):
        cert = sum_certificate(np.tile(np.array([1e16, 1.0, -1e16]), 500))
        self.assertGreater(cert["condition_number"], 1e10)  # ill-conditioned
        self.assertIn("rel_error_bound", cert)


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
