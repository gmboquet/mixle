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


class AlphabetAndDimensionValidationTest(unittest.TestCase):
    """MXR-080-0130: packing silently mapped every positive value to +1 and everything else to -1 (or a
    ternary sign/mask), so out-of-alphabet inputs (2, -9, 0.2) were reinterpreted instead of rejected, and
    binary_dot did not require its operands' trailing dimensions to match."""

    def test_out_of_alphabet_values_rejected(self):
        # The audit's own examples: none of these belong to {-1,0,+1}.
        for bad in (2, -9, 0.2):
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(ValueError):
                    pack_pm1(np.array([bad]))

    def test_out_of_alphabet_rejected_when_mixed_with_valid_values(self):
        with self.assertRaises(ValueError):
            pack_pm1(np.array([1, -1, 2, 1]))
        # MXR-080-1537: these values are in the union, but not wholly in either binary alphabet.
        with self.assertRaises(ValueError):
            pack_pm1(np.array([-1, 0, 1]))
        with self.assertRaises(ValueError):
            binary_dot(np.array([[-1, 0, 1]]), np.array([[-1, 0, 1]]))

    def test_pack_ternary_out_of_alphabet_rejected(self):
        from mixle.engines.bitpacked import pack_ternary

        for bad in (2, -9, 0.2):
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(ValueError):
                    pack_ternary(np.array([bad]))

    def test_mismatched_trailing_dimensions_rejected(self):
        # Exact audit repro: binary_dot([1,1], [1,1,1]) used to return 0 instead of raising, because
        # both operands happened to pack to the same (1-word) packed shape despite true dims 2 vs 3.
        with self.assertRaises(ValueError):
            binary_dot(np.array([1, 1]), np.array([1, 1, 1]))

    def test_mismatched_dims_rejected_even_when_word_counts_would_coincide(self):
        # dims 5 and 70 both round up to different word counts (1 vs 2) -- a more obviously-mismatched
        # case than the audit's exact repro, kept as a second data point.
        with self.assertRaises(ValueError):
            binary_dot(np.ones(5), np.ones(70))

    def test_legitimate_matching_dim_vectors_still_dot_correctly(self):
        # Negative control: validation must not disturb correct computation, and the numeric answer
        # (not just absence of a crash) is checked.
        self.assertEqual(int(binary_dot(np.array([1, 1]), np.array([1, 1]))[0, 0]), 2)
        self.assertEqual(int(binary_dot(np.array([1, -1]), np.array([1, 1]))[0, 0]), 0)
        # {0,1} alphabet remains valid per pack_pm1's documented contract.
        self.assertEqual(int(binary_dot(np.array([1, 0, 1]), np.array([1, 0, 1]))[0, 0]), 3)
        rng = np.random.RandomState(11)
        a = rng.choice([-1, 1], size=(30, 130)).astype(np.int8)
        b = rng.choice([-1, 1], size=(12, 130)).astype(np.int8)
        ref = a.astype(np.int32) @ b.T.astype(np.int32)
        self.assertTrue(np.array_equal(ref, binary_dot(a, b)))

    def test_public_rank_contract_accepts_vectors_and_matrices_only(self):
        for bad in (np.array(1), np.ones((1, 1, 1))):
            with self.assertRaises(ValueError):
                pack_pm1(bad)
            with self.assertRaises(ValueError):
                binary_dot(bad, np.ones(1))

    def test_binary_gemm_rejects_noncanonical_padding(self):
        # MXR-080-1538 exact adversarial shape: only one bit is data, every other lane is padding.
        with self.assertRaises(ValueError):
            binary_gemm(np.array([[0]], dtype=np.uint64), np.array([[2**64 - 1]], dtype=np.uint64), 1)

    def test_canonical_padding_keeps_short_products_in_range(self):
        for dim in (1, 7, 9, 63, 65):
            a = np.ones((1, dim), dtype=np.int8)
            b = -np.ones((1, dim), dtype=np.int8)
            self.assertEqual(int(binary_dot(a, b)[0, 0]), -dim)


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


@unittest.skipUnless(HAS_BITPACKED, "compiled _bitpacked extension not built")
class CompiledKernelBoundsSafetyRegressionTest(unittest.TestCase):
    """MXR-080-0131 (Critical): the compiled kernels derive their word-count loop bound from ONE input
    array and, with boundscheck disabled, previously read past the end of a shorter b_packed / sign /
    nonzero-mask array inside a nogil loop instead of raising -- a genuine out-of-bounds memory read
    verified pre-fix with AddressSanitizer (heap-buffer-overflow, READ of size 8, in both binary_gemm and
    ternary_gemm). Post-fix, both the Python wrapper and the compiled kernel itself reject the mismatch
    with a clean ValueError; this is checked at both layers and repeatedly to rule out flakiness."""

    def _make_binary_mismatch(self, seed):
        rng = np.random.RandomState(seed)
        a_full = rng.choice([-1, 1], size=(3, 128)).astype(np.int8)  # 2 packed words
        b_short = rng.choice([-1, 1], size=(3, 64)).astype(np.int8)  # 1 packed word -- SHORTER
        return pack_pm1(a_full), pack_pm1(b_short)

    def test_shorter_b_packed_rejected_by_python_wrapper(self):
        a_packed, b_packed = self._make_binary_mismatch(0)
        with self.assertRaises(ValueError):
            binary_gemm(a_packed, b_packed, 128)

    def test_shorter_b_packed_rejected_by_compiled_kernel_directly(self):
        # Bypass mixle.engines.bitpacked entirely and call the raw compiled extension, proving the
        # Cython-level defensive check (not just the Python wrapper) rejects the mismatch.
        from mixle.engines._bitpacked import binary_gemm as binary_gemm_c

        a_packed, b_packed = self._make_binary_mismatch(1)
        with self.assertRaises(ValueError):
            binary_gemm_c(a_packed, b_packed, 128)

    def test_shorter_nz_plane_rejected_by_python_wrapper(self):
        from mixle.engines.bitpacked import pack_ternary, ternary_gemm

        rng = np.random.RandomState(2)
        a = rng.choice([-1, 0, 1], size=(3, 128)).astype(np.int8)
        b = rng.choice([-1, 0, 1], size=(3, 128)).astype(np.int8)
        a_sign, a_nz = pack_ternary(a)
        b_sign, b_nz = pack_ternary(b)
        a_nz_short = np.ascontiguousarray(a_nz[:, :1])  # SHORTER than a_sign
        with self.assertRaises(ValueError):
            ternary_gemm(a_sign, a_nz_short, b_sign, b_nz)

    def test_shorter_nz_plane_rejected_by_compiled_kernel_directly(self):
        from mixle.engines._bitpacked import ternary_gemm as ternary_gemm_c

        from mixle.engines.bitpacked import pack_ternary

        rng = np.random.RandomState(3)
        a = rng.choice([-1, 0, 1], size=(3, 128)).astype(np.int8)
        b = rng.choice([-1, 0, 1], size=(3, 128)).astype(np.int8)
        a_sign, a_nz = pack_ternary(a)
        b_sign, b_nz = pack_ternary(b)
        b_sign_short = np.ascontiguousarray(b_sign[:, :1])  # SHORTER than a_sign
        b_nz_short = np.ascontiguousarray(b_nz[:, :1])
        with self.assertRaises(ValueError):
            ternary_gemm_c(a_sign, a_nz, b_sign_short, b_nz_short)

    def test_shorter_arrays_deterministically_rejected_no_crash_across_many_trials(self):
        # Fresh allocations each trial (not a fixed pair) to catch any residual nondeterminism -- a real
        # OOB read would occasionally return different garbage rather than always raising cleanly.
        from mixle.engines.bitpacked import pack_ternary, ternary_gemm

        for trial in range(100):
            a_packed, b_packed = self._make_binary_mismatch(trial)
            with self.assertRaises(ValueError):
                binary_gemm(a_packed, b_packed, 128)

            rng = np.random.RandomState(trial + 1000)
            a = rng.choice([-1, 0, 1], size=(3, 128)).astype(np.int8)
            b = rng.choice([-1, 0, 1], size=(3, 128)).astype(np.int8)
            a_sign, a_nz = pack_ternary(a)
            b_sign, b_nz = pack_ternary(b)
            a_nz_short = np.ascontiguousarray(a_nz[:, :1])
            with self.assertRaises(ValueError):
                ternary_gemm(a_sign, a_nz_short, b_sign, b_nz)

    def test_correctly_shaped_arrays_still_compute_correctly_through_compiled_path(self):
        # Negative control: the added validation must not disturb the correct, in-bounds compiled path.
        from mixle.engines.bitpacked import pack_ternary, ternary_gemm

        rng = np.random.RandomState(42)
        a = rng.choice([-1, 1], size=(64, 192)).astype(np.int8)
        b = rng.choice([-1, 1], size=(32, 192)).astype(np.int8)
        ref = a.astype(np.int32) @ b.T.astype(np.int32)
        self.assertTrue(np.array_equal(ref, binary_gemm(pack_pm1(a), pack_pm1(b), 192)))

        at = rng.choice([-1, 0, 1], size=(40, 192)).astype(np.int8)
        bt = rng.choice([-1, 0, 1], size=(20, 192)).astype(np.int8)
        ref_t = at.astype(np.int32) @ bt.T.astype(np.int32)
        asgn, anz = pack_ternary(at)
        bsgn, bnz = pack_ternary(bt)
        self.assertTrue(np.array_equal(ref_t, ternary_gemm(asgn, anz, bsgn, bnz)))


if __name__ == "__main__":
    unittest.main()
