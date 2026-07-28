"""Sub-byte bit-packing (mixle.engines.packing) + codebook compression to bytes."""

import unittest

import numpy as np

from mixle.engines.formats import CodebookFormat
from mixle.engines.packing import pack_bits, packed_nbytes, unpack_bits


class PackBitsTest(unittest.TestCase):
    def test_round_trip_all_supported_widths(self):
        rng = np.random.RandomState(0)
        for bits in (1, 2, 4, 8):
            for n in (1, 7, 8, 9, 1000):
                codes = rng.randint(0, 1 << bits, size=n).astype(np.uint64)
                packed = pack_bits(codes, bits)
                back = unpack_bits(packed, bits, n)
                self.assertTrue(np.array_equal(back, codes), "width=%d n=%d" % (bits, n))

    def test_packed_byte_count_matches(self):
        self.assertEqual(packed_nbytes(100, 4), 50)  # 2 codes / byte
        self.assertEqual(packed_nbytes(100, 2), 25)  # 4 codes / byte
        self.assertEqual(packed_nbytes(100, 1), 13)  # 8 codes / byte, rounded up
        self.assertEqual(packed_nbytes(100, 8), 100)
        for bits in (1, 2, 4, 8):
            self.assertEqual(pack_bits(np.zeros(100, np.uint64), bits).size, packed_nbytes(100, bits))

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            pack_bits([0, 1], 3)  # not a power-of-two byte-tiling width
        with self.assertRaises(ValueError):
            pack_bits([0, 16], 4)  # 16 does not fit in 4 bits
        for operation in (pack_bits,):
            with self.assertRaises((TypeError, ValueError)):
                operation([0, 1], True)
        with self.assertRaises((TypeError, ValueError)):
            unpack_bits([0], True, 1)
        with self.assertRaises((TypeError, ValueError)):
            packed_nbytes(1, True)

    def test_unpack_rejects_lossy_or_out_of_range_payload_bytes(self):
        for bad in ([1.9], [-1], [256], [True]):
            with self.assertRaises(ValueError):
                unpack_bits(bad, 1, 1)


class PackBitsCorruptionRegressionTest(unittest.TestCase):
    """MXR-080-0129: codes were cast to uint64 (and count sliced) before validation, so a fractional code
    silently truncated, a negative code silently wrapped, and a negative/oversized unpack count silently
    fell through to ordinary Python slicing instead of raising."""

    def test_fractional_code_rejected(self):
        # The audit's own repro: 1.9 used to silently pack as code one.
        with self.assertRaises(ValueError):
            pack_bits([1.9], 4)

    def test_fractional_code_rejected_in_larger_batch(self):
        with self.assertRaises(ValueError):
            pack_bits([1, 2, 3.5, 4], 4)

    def test_negative_code_rejected(self):
        with self.assertRaises(ValueError):
            pack_bits([-1], 4)

    def test_nan_code_rejected(self):
        with self.assertRaises(ValueError):
            pack_bits([float("nan")], 4)

    def test_nan_in_existing_float_array_rejected(self):
        # A NaN in a *fresh Python list* happens to already trip numpy's own uint64 cast (ValueError:
        # cannot convert float NaN to integer). But an EXISTING float64 ndarray takes a different numpy
        # cast path that does NOT raise -- it silently produces a garbage value with only a RuntimeWarning
        # (the same input-shape-dependent gap the MXR-080-0060 graph_source.py fix hit). Both input shapes
        # must be rejected explicitly rather than relying on whichever behavior numpy's cast happens to have.
        arr = np.array([1.0, float("nan"), 3.0], dtype=np.float64)
        with self.assertRaises(ValueError):
            pack_bits(arr, 4)

    def test_inf_code_rejected(self):
        with self.assertRaises(ValueError):
            pack_bits([float("inf")], 4)

    def test_negative_unpack_count_rejected(self):
        # Previously fell through to Python's negative-slice semantics (arr[:-3]) instead of raising.
        packed = pack_bits(np.arange(10, dtype=np.uint64) % 4, 2)
        with self.assertRaises(ValueError):
            unpack_bits(packed, 2, -3)

    def test_fractional_unpack_count_rejected(self):
        packed = pack_bits(np.arange(10, dtype=np.uint64) % 4, 2)
        with self.assertRaises(ValueError):
            unpack_bits(packed, 2, 1.5)

    def test_oversized_unpack_count_rejected(self):
        # Previously silently returned fewer values than requested instead of raising.
        codes = np.arange(10, dtype=np.uint64) % 4
        packed = pack_bits(codes, 2)
        with self.assertRaises(ValueError):
            unpack_bits(packed, 2, 10_000)

    def test_unpack_count_at_exact_capacity_is_not_an_error(self):
        # Boundary negative control: count == capacity (not > capacity) must still succeed.
        codes = np.arange(10, dtype=np.uint64) % 4
        packed = pack_bits(codes, 2)
        capacity = packed.size * (8 // 2)
        out = unpack_bits(packed, 2, capacity)
        self.assertEqual(len(out), capacity)

    def test_packed_nbytes_negative_count_rejected(self):
        with self.assertRaises(ValueError):
            packed_nbytes(-1, 4)

    def test_packed_nbytes_fractional_count_rejected(self):
        with self.assertRaises(ValueError):
            packed_nbytes(2.5, 4)

    def test_legitimate_exact_codes_and_counts_still_round_trip(self):
        # Negative control: whole-number floats and ordinary nonnegative int codes/counts are unaffected.
        rng = np.random.RandomState(3)
        codes = rng.randint(0, 16, size=50).astype(np.uint64)
        packed = pack_bits(codes, 4)
        self.assertTrue(np.array_equal(unpack_bits(packed, 4, 50), codes))
        # whole-number floats (e.g. 2.0) are exact integers and must still be accepted
        float_codes = [0.0, 15.0, 7.0, 1.0]
        packed_f = pack_bits(float_codes, 4)
        self.assertEqual(unpack_bits(packed_f, 4, 4).tolist(), [0, 15, 7, 1])
        self.assertEqual(packed_nbytes(50, 4), 25)


class CodebookCompressionTest(unittest.TestCase):
    def test_compress_round_trips_and_shrinks_bytes(self):
        rng = np.random.RandomState(1)
        data = np.concatenate([rng.normal(-2, 0.5, 5000), rng.normal(3, 0.5, 5000)])
        fmt = CodebookFormat.fit(data, 16)  # 4-bit indices -> 2 values/byte
        packed, n = fmt.compress(data)
        self.assertEqual(n, data.size)
        # genuine compression: float64 is 8 bytes/value; 16-code packing is 0.5 bytes/value -> 16x
        self.assertEqual(packed.nbytes, packed_nbytes(data.size, 4))
        self.assertLess(packed.nbytes, data.nbytes // 15)
        # lossless vs the (lossy) quantization it represents
        self.assertTrue(np.array_equal(fmt.decompress(packed, n), fmt.round_trip(data)))

    def test_compress_decompress_is_a_reasonable_approximation(self):
        rng = np.random.RandomState(2)
        data = rng.normal(0.0, 1.0, 8000)
        fmt = CodebookFormat.fit(data, 256)  # 8-bit indices
        packed, n = fmt.compress(data)
        rt = fmt.decompress(packed, n)
        self.assertLess(float(np.sqrt(np.mean((rt - data) ** 2))), 0.05)  # small RMSE at 256 codes


if __name__ == "__main__":
    unittest.main()
