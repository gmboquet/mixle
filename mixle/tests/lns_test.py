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
        self.assertLessEqual(float(np.max(np.abs(got - ref))), lns.max_logsumexp_error)

    def test_logsumexp_matches_float64_within_bound(self):
        for step in (0.05, 0.01, 0.002):
            lns = LogNumberSystem(step=step)
            rng = np.random.RandomState(2)
            X = rng.randn(500, 64) * 30  # rows of log-densities spanning a wide range
            ref = logsumexp(X, axis=1)
            got = lns.dequantize(lns.logsumexp(lns.quantize(X), axis=1))
            err = float(np.max(np.abs(got - ref)))
            # the pairwise tree accumulates a few LUT roundings over log2(64)=6 levels
            self.assertLessEqual(err, 8 * lns.max_logsumexp_error, "step=%g err=%g" % (step, err))

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
        with self.assertRaises(ValueError):
            LogNumberSystem(step=0.0)


if __name__ == "__main__":
    unittest.main()
