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
        compiled = logsumexp_rows(k, lns.lut, lns.dmax)
        # force the numpy path by reducing a non-2d-last-axis view
        numpy_tree = np.array([_numpy_tree_row(row, lns) for row in k])
        self.assertTrue(np.array_equal(compiled, numpy_tree))

    def test_logsumexp_within_bound_of_float64(self):
        lns = LogNumberSystem(step=0.005)
        rng = np.random.RandomState(1)
        X = rng.randn(500, 1000) * 20
        got = lns.dequantize(lns.logsumexp(lns.quantize(X), axis=1))  # uses the compiled kernel
        ref = sp.logsumexp(X, axis=1)
        self.assertLessEqual(float(np.max(np.abs(got - ref))), 8 * lns.max_logsumexp_error)

    def test_fused_cross_entropy_matches_float64(self):
        lns = LogNumberSystem(step=0.005)
        rng = np.random.RandomState(2)
        logits = rng.randn(1024, 4000) * 4
        targets = rng.randint(0, 4000, 1024)
        ref = float(np.mean(sp.logsumexp(logits, axis=1) - logits[np.arange(1024), targets]))
        got = cross_entropy(logits, targets, lns, axis=1)  # uses cross_entropy_rows
        self.assertLessEqual(abs(got - ref), 8 * lns.max_logsumexp_error)

    def test_cross_entropy_rows_direct(self):
        lns = LogNumberSystem(step=0.01)
        k = np.ascontiguousarray(lns.quantize(np.array([[0.0, -1.0, -2.0], [-3.0, 0.0, -1.0]])))
        tgt = np.ascontiguousarray(np.array([0, 1], dtype=np.int64))
        total = cross_entropy_rows(k, tgt, lns.lut, lns.dmax)
        lse = logsumexp_rows(k, lns.lut, lns.dmax)
        self.assertEqual(total, int((lse[0] - k[0, 0]) + (lse[1] - k[1, 1])))


def _numpy_tree_row(row, lns):
    k = row.astype(np.int64).copy()
    while k.shape[-1] > 1:
        if k.shape[-1] & 1:
            tail, k = k[-1:], k[:-1]
        else:
            tail = None
        a, b = k[0::2], k[1::2]
        d = np.minimum(np.abs(a - b), lns.dmax)
        k = np.maximum(a, b) + lns.lut[d]
        if tail is not None:
            k = np.concatenate([k, tail])
    return int(k[0])


if __name__ == "__main__":
    unittest.main()
