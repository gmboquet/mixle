"""Compiled FMA double-double kernels (mixle.engines._dd_kernels): correctness + integration.

Skipped when the optional extension is not compiled (the package works without it). When present, the C
kernel must be bit-for-bit identical to the pure-numpy path and full-precision accurate vs the oracle.
"""

import unittest

import numpy as np
import pytest

from mixle.engines.build_kernels import dd_kernels_available

if not dd_kernels_available():  # pragma: no cover - depends on whether the extension was compiled
    raise unittest.SkipTest("compiled _dd_kernels not built (run mixle.engines.build_kernels.compile_dd_kernels)")

mpmath = pytest.importorskip("mpmath")

from mixle.engines._dd_kernels import dd_dot_c, dd_sum_c  # noqa: E402

from mixle.engines.extended import HAS_DD_KERNELS, dd_dot, dd_sum  # noqa: E402


class CompiledKernelTest(unittest.TestCase):
    def test_extended_uses_the_compiled_kernel(self):
        self.assertTrue(HAS_DD_KERNELS)

    def test_c_dot_matches_numpy_and_oracle(self):
        rng = np.random.RandomState(0)
        a = np.ascontiguousarray(rng.randn(5000) * 10.0 ** rng.randint(-6, 6, 5000))
        b = np.ascontiguousarray(rng.randn(5000) * 10.0 ** rng.randint(-6, 6, 5000))
        # exact vs mpmath
        with mpmath.workprec(400):
            true = mpmath.fsum(mpmath.mpf(float(a[i])) * mpmath.mpf(float(b[i])) for i in range(a.size))
            hi, lo = dd_dot_c(a, b)
            got = mpmath.mpf(float(hi)) + mpmath.mpf(float(lo))
            denom = abs(true) if true != 0 else mpmath.mpf(1)
            self.assertLess(float(abs((got - true) / denom)), 1e-25)
        # identical to the numpy double-double path
        self.assertTrue(np.isclose(dd_dot(a, b).to_float(), float(np.float64(hi) + np.float64(lo)), rtol=1e-14))

    def test_c_sum_is_full_precision(self):
        rng = np.random.RandomState(1)
        x = np.ascontiguousarray(np.tile(np.array([1e16, 1.0, -1e16, -1.0]), 5000))
        rng.shuffle(x)
        hi, lo = dd_sum_c(x)
        self.assertLess(abs(float(np.float64(hi) + np.float64(lo))), 1e-6)  # true sum 0
        self.assertTrue(np.isclose(float(hi), dd_sum(x).to_float(), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
