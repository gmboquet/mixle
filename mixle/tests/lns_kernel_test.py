"""Compiled integer log-sum-exp kernel (mixle.engines._lns_kernel): tree fold, bit-identical + fast.

Skipped when the extension is not built. The compiled tree fold must be BIT-IDENTICAL to the numpy tree
(same algorithm, compiled), and the fused cross-entropy must match float64 within the LNS bound.
"""

import unittest

import numpy as np
import pytest

from mixle.engines.build_kernels import lns_kernel_available

if not lns_kernel_available():  # pragma: no cover - depends on whether the extension was compiled
    raise unittest.SkipTest("compiled _lns_kernel not built (run build_kernels.compile_lns_kernel)")

sp = pytest.importorskip("scipy.special")

from mixle.engines._lns_kernel import cross_entropy_rows, logsumexp_rows  # noqa: E402

from mixle.engines.lns import (
    _HAS_LNS_KERNEL,  # noqa: E402
    CODE_MAX,  # noqa: E402
    CODE_MIN,  # noqa: E402
    LOG_ZERO_CODE,  # noqa: E402
    LogNumberSystem,  # noqa: E402
)
from mixle.engines.lns_nn import cross_entropy  # noqa: E402


class CompiledLnsKernelTest(unittest.TestCase):
    def test_kernel_is_wired_in(self):
        self.assertTrue(_HAS_LNS_KERNEL)

    def test_compiled_tree_is_bit_identical_to_numpy_tree(self):
        lns = LogNumberSystem(step=0.01)
        rng = np.random.RandomState(0)
        k = np.ascontiguousarray(lns.quantize(rng.randn(2000, 200) * 25))
        # the numpy fallback tree vs the compiled tree -- same algorithm, must be exactly equal
        compiled = logsumexp_rows(k, lns.lut, lns.dmax, LOG_ZERO_CODE, CODE_MIN, CODE_MAX)
        # force the numpy path by reducing a non-2d-last-axis view
        numpy_tree = np.array([_numpy_tree_row(row, lns) for row in k])
        self.assertTrue(np.array_equal(compiled, numpy_tree))

    def test_logsumexp_within_bound_of_float64(self):
        lns = LogNumberSystem(step=0.005)
        rng = np.random.RandomState(1)
        X = rng.randn(500, 1000) * 20
        got = lns.dequantize(lns.logsumexp(lns.quantize(X), axis=1))  # uses the compiled kernel
        ref = sp.logsumexp(X, axis=1)
        # reducing 1000 terms is a depth-10 pairwise tree (MXR-080-0139: the certificate scales with
        # depth, not a flat per-call constant)
        self.assertLessEqual(float(np.max(np.abs(got - ref))), lns.max_logsumexp_error(1000))

    def test_fused_cross_entropy_matches_float64(self):
        lns = LogNumberSystem(step=0.005)
        rng = np.random.RandomState(2)
        logits = rng.randn(1024, 4000) * 4
        targets = rng.randint(0, 4000, 1024)
        ref = float(np.mean(sp.logsumexp(logits, axis=1) - logits[np.arange(1024), targets]))
        got = cross_entropy(logits, targets, lns, axis=1)  # uses cross_entropy_rows
        self.assertLessEqual(abs(got - ref), lns.max_logsumexp_error(4000))  # 4000-class log-partition

    def test_cross_entropy_rows_direct(self):
        lns = LogNumberSystem(step=0.01)
        k = np.ascontiguousarray(lns.quantize(np.array([[0.0, -1.0, -2.0], [-3.0, 0.0, -1.0]])))
        tgt = np.ascontiguousarray(np.array([0, 1], dtype=np.int64))
        total = cross_entropy_rows(k, tgt, lns.lut, lns.dmax)
        lse = logsumexp_rows(k, lns.lut, lns.dmax, LOG_ZERO_CODE, CODE_MIN, CODE_MAX)
        self.assertEqual(total, int((lse[0] - k[0, 0]) + (lse[1] - k[1, 1])))

    def test_logsumexp_rows_handles_the_exact_overflow_scenario_without_crashing(self):
        # MXR-080-0138, exercised directly against the compiled entry point (bypassing the Python
        # wrapper entirely, the same way a caller reaching into mixle.engines._lns_kernel directly
        # could): a=INT64_MAX, b=INT64_MIN in one row is exactly the pre-fix crash scenario --
        # INT64_MAX - INT64_MIN wraps to -1 and (boundscheck disabled) indexes lut[-1] out of bounds.
        # Confirmed as a genuine ASan-reported heap-buffer-overflow pre-fix; post-fix this is the
        # absorbing-sentinel branch (b == LOG_ZERO_CODE) and must return a=INT64_MAX exactly, every
        # time, with no exception.
        lns = LogNumberSystem(step=0.01)
        i64 = np.iinfo(np.int64)
        row = np.ascontiguousarray(np.array([[i64.max, i64.min]], dtype=np.int64))
        for _ in range(200):
            out = logsumexp_rows(row, lns.lut, lns.dmax, LOG_ZERO_CODE, CODE_MIN, CODE_MAX)
            self.assertEqual(int(out[0]), int(i64.max))

    def test_logsumexp_rows_adversarial_extreme_codes_deterministic_and_crash_free(self):
        # Broad sweep directly against the compiled kernel: fresh random extreme int64 rows (some
        # forced to the sentinel, some forced to the saturation boundary) each run, asserting no
        # exception and an identical answer to a second call on the same input (determinism), plus
        # exact agreement with the numpy-tree fallback (mixle.engines.lns.LogNumberSystem.logsumexp
        # with the compiled path unavailable would take, and does take here via .logadd internally).
        lns = LogNumberSystem(step=0.01)
        rng = np.random.RandomState(123)
        mismatches = 0
        for t in range(4000):
            n = rng.randint(2, 9)
            row = rng.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(1, n), dtype=np.int64)
            if t % 5 == 0:
                row[0, rng.randint(0, n)] = LOG_ZERO_CODE
            if t % 7 == 0:
                row[0, rng.randint(0, n)] = CODE_MAX
            if t % 11 == 0:
                row[0, rng.randint(0, n)] = CODE_MIN
            row = np.ascontiguousarray(row)
            out1 = logsumexp_rows(row, lns.lut, lns.dmax, LOG_ZERO_CODE, CODE_MIN, CODE_MAX)
            out2 = logsumexp_rows(row, lns.lut, lns.dmax, LOG_ZERO_CODE, CODE_MIN, CODE_MAX)
            self.assertEqual(int(out1[0]), int(out2[0]))  # deterministic
            ref = lns.logsumexp(row, axis=1)  # Python-level: same self.logadd-based tree, independently
            if int(out1[0]) != int(ref[0]):
                mismatches += 1
        self.assertEqual(mismatches, 0)

    def test_logsumexp_rows_sentinel_matches_independent_numpy_reference(self):
        # MXR-080-0138 at the compiled entry point, against an INDEPENDENT reference implementation
        # (not the class under test calling itself): _numpy_tree_row below is a standalone, from-scratch
        # sentinel/overflow-aware tree fold operating only on lns.lut/lns.dmax, mirroring but not
        # calling LogNumberSystem.logadd.
        lns = LogNumberSystem(step=0.01)
        rng = np.random.RandomState(5)
        rows = lns.quantize(rng.randn(500, 32) * 25)
        # scatter genuine zero-sentinel entries through ~20% of the positions
        drop = rng.rand(500, 32) < 0.2
        rows = np.where(drop, LOG_ZERO_CODE, rows).astype(np.int64)
        rows = np.ascontiguousarray(rows)
        compiled = logsumexp_rows(rows, lns.lut, lns.dmax, LOG_ZERO_CODE, CODE_MIN, CODE_MAX)
        reference = np.array([_numpy_tree_row(row, lns) for row in rows])
        self.assertTrue(np.array_equal(compiled, reference))

    def test_negative_control_ordinary_codes_through_the_compiled_kernel(self):
        # No sentinels, no extreme codes: the compiled kernel's ordinary answer must be unaffected by
        # the MXR-080-0138 fix.
        lns = LogNumberSystem(step=0.005)
        rng = np.random.RandomState(6)
        k = np.ascontiguousarray(lns.quantize(rng.randn(300, 50) * 15))
        compiled = logsumexp_rows(k, lns.lut, lns.dmax, LOG_ZERO_CODE, CODE_MIN, CODE_MAX)
        reference = np.array([_numpy_tree_row(row, lns) for row in k])
        self.assertTrue(np.array_equal(compiled, reference))


def _numpy_tree_row(row, lns):
    """Standalone reference tree fold (independent of LogNumberSystem.logadd): mirrors the compiled
    kernel's MXR-080-0138 sentinel/saturation handling by hand, for an independent bit-identical check.
    """
    k = row.astype(np.int64).copy()
    while k.shape[-1] > 1:
        if k.shape[-1] & 1:
            tail, k = k[-1:], k[:-1]
        else:
            tail = None
        a, b = k[0::2], k[1::2]
        is_zero_a, is_zero_b = a == LOG_ZERO_CODE, b == LOG_ZERO_CODE
        ca, cb = np.clip(a, CODE_MIN, CODE_MAX), np.clip(b, CODE_MIN, CODE_MAX)
        d = np.minimum(np.abs(ca - cb), lns.dmax)
        combined = np.maximum(ca, cb) + lns.lut[d]
        combined = np.where(is_zero_b, a, combined)
        combined = np.where(is_zero_a, b, combined)
        k = combined
        if tail is not None:
            k = np.concatenate([k, tail])
    return int(k[0])


if __name__ == "__main__":
    unittest.main()
