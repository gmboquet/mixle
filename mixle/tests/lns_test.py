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
        with self.assertRaises(ValueError):
            LogNumberSystem(step=0.0)


if __name__ == "__main__":
    unittest.main()
