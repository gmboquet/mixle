"""Packed binary/ternary compute (mixle.engines.bitpacked): exact popcount dot products.

Correctness is the contract (the popcount dot is EXACT integer arithmetic). The numpy fallback runs when
the compiled extension is absent; the compiled path is exercised when present. No speed assertion -- whether
popcount beats fp32 is hardware-dependent (it is a storage win on Apple-AMX, a compute win on non-AMX paths).
"""

import unittest

import numpy as np

from mixle.engines.bitpacked import HAS_BITPACKED, binary_dot, binary_gemm, pack_pm1


class BinaryPackedTest(unittest.TestCase):
    def test_binary_dot_is_exact(self):
        rng = np.random.RandomState(0)
        a = rng.choice([-1, 1], size=(200, 256)).astype(np.int8)
        b = rng.choice([-1, 1], size=(64, 256)).astype(np.int8)
        ref = a.astype(np.int32) @ b.T.astype(np.int32)
        got = binary_dot(a, b)
        self.assertTrue(np.array_equal(ref, got))

    def test_handles_non_multiple_of_64_dim(self):
        rng = np.random.RandomState(1)
        a = rng.choice([-1, 1], size=(30, 100)).astype(np.int8)  # D=100 -> padded to 128
        b = rng.choice([-1, 1], size=(10, 100)).astype(np.int8)
        ref = a.astype(np.int32) @ b.T.astype(np.int32)
        self.assertTrue(np.array_equal(ref, binary_dot(a, b)))

    def test_zeros_encode_as_minus_one_consistently(self):
        # pack_pm1 treats >0 as +1 and everything else as -1, so {0,1} data is a valid binary code too
        a = np.array([[1, 0, 1, 0, 1, 1, 0, 0]])
        b = np.array([[1, 0, 1, 0, 1, 1, 0, 0]])
        # identical codes -> full agreement -> dot == D
        self.assertEqual(int(binary_dot(a, b)[0, 0]), 8)

    def test_storage_is_32x_smaller_than_float64(self):
        x = np.random.RandomState(2).choice([-1, 1], size=(1000, 512)).astype(np.int8)
        packed = pack_pm1(x)
        self.assertEqual(packed.nbytes, 1000 * (512 // 64) * 8)  # 8 bytes/word, 8 words/row
        self.assertEqual(x.astype(np.float64).nbytes / packed.nbytes, 64.0)  # 1 bit vs 64-bit float


@unittest.skipUnless(HAS_BITPACKED, "compiled _bitpacked extension not built")
class CompiledKernelTest(unittest.TestCase):
    def test_compiled_binary_gemm_matches_int_dot(self):
        rng = np.random.RandomState(3)
        a = rng.choice([-1, 1], size=(500, 1024)).astype(np.int8)
        w = rng.choice([-1, 1], size=(1024, 128)).astype(np.int8)
        ref = a.astype(np.int32) @ w.astype(np.int32)
        got = binary_gemm(pack_pm1(a), pack_pm1(w.T), 1024)
        self.assertTrue(np.array_equal(ref, got))

    def test_ternary_gemm_is_exact(self):
        from mixle.engines.bitpacked import pack_ternary, ternary_gemm

        rng = np.random.RandomState(4)
        a = rng.choice([-1, 0, 1], size=(120, 256)).astype(np.int8)
        b = rng.choice([-1, 0, 1], size=(48, 256)).astype(np.int8)
        ref = a.astype(np.int32) @ b.T.astype(np.int32)
        asgn, anz = pack_ternary(a)
        bsgn, bnz = pack_ternary(b)
        self.assertTrue(np.array_equal(ref, ternary_gemm(asgn, anz, bsgn, bnz)))


if __name__ == "__main__":
    unittest.main()
