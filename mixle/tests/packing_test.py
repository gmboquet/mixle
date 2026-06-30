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
