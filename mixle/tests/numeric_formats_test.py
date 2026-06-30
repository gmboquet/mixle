"""Numeric format codecs (mixle.engines.formats): low-bit float, fixed-point, codebook compression."""

import math
import unittest

import numpy as np

from mixle.engines.formats import (
    CodebookFormat,
    FixedPointFormat,
    FloatFormat,
    min_float_mantissa_bits,
)


class FloatFormatTest(unittest.TestCase):
    def test_round_trip_within_relative_error_bound(self):
        rng = np.random.RandomState(0)
        x = rng.randn(5000) * 10.0 ** rng.randint(-6, 6, 5000)
        for nbits in (8, 16, 32):
            fmt = FloatFormat.fp(nbits)
            rt = fmt.round_trip(x)
            rel = np.abs(rt - x) / np.maximum(np.abs(x), 1e-300)
            self.assertLessEqual(float(rel.max()), fmt.max_rel_error * 1.01)

    def test_precision_increases_monotonically_with_bits(self):
        rng = np.random.RandomState(1)
        x = rng.randn(4000)
        errs = [FloatFormat.fp(n).measured_max_abs_error(x) for n in (8, 16, 32)]
        self.assertGreater(errs[0], errs[1])
        self.assertGreater(errs[1], errs[2])

    def test_fp64_is_lossless(self):
        rng = np.random.RandomState(2)
        x = rng.randn(1000)
        self.assertTrue(np.array_equal(FloatFormat.fp(64).round_trip(x), x))

    def test_extreme_low_bit_formats_do_not_crash(self):
        x = np.array([0.3, -2.7, 11.0, 0.0])
        for n in (1, 2, 3, 4):
            FloatFormat.fp(n).round_trip(x)  # extreme quantization, must not raise


class FixedPointFormatTest(unittest.TestCase):
    def test_round_trip_within_absolute_bound_and_compresses(self):
        rng = np.random.RandomState(3)
        x = rng.uniform(-100, 100, 5000)
        fmt = FixedPointFormat(frac_bits=12, int_bits=10)  # 23 bits -> int32 storage
        q = fmt.quantize(x)
        self.assertEqual(q.dtype, np.int32)
        self.assertLessEqual(fmt.measured_max_abs_error(x), fmt.max_abs_error * 1.01)
        self.assertGreater(fmt.compression_ratio(), 2.0)  # 64 / 23 bits

    def test_out_of_range_clamps(self):
        fmt = FixedPointFormat(frac_bits=4, int_bits=3)  # range ~[-8, 8)
        rt = fmt.round_trip(np.array([1000.0, -1000.0]))
        self.assertTrue(np.all(np.abs(rt) <= 8.0))


class CodebookFormatTest(unittest.TestCase):
    def test_fit_round_trip_error_shrinks_with_more_codes(self):
        rng = np.random.RandomState(4)
        data = np.concatenate([rng.normal(-3, 0.4, 4000), rng.normal(2, 0.6, 4000)])  # bimodal
        e16 = CodebookFormat.fit(data, 16).measured_max_abs_error(data)
        e256 = CodebookFormat.fit(data, 256).measured_max_abs_error(data)
        self.assertGreater(e16, e256)

    def test_indices_are_compact_and_dequantize_gathers_codes(self):
        rng = np.random.RandomState(5)
        data = rng.randn(3000)
        fmt = CodebookFormat.fit(data, 64)
        idx = fmt.quantize(data)
        self.assertEqual(idx.dtype, np.uint8)  # 64 codes -> 1 byte/value
        self.assertEqual(fmt.bits_per_value, 6.0)  # ceil(log2(64))
        # dequantize returns codebook entries
        rt = fmt.dequantize(idx)
        self.assertTrue(np.all(np.isin(rt, fmt.codebook)))

    def test_compression_ratio(self):
        fmt = CodebookFormat.fit(np.random.RandomState(6).randn(2000), 256)
        self.assertEqual(fmt.bits_per_value, 8.0)
        self.assertEqual(fmt.compression_ratio(), 8.0)  # 64-bit float -> 8-bit index


class ErrorTracingTest(unittest.TestCase):
    def test_min_float_mantissa_bits_meets_target(self):
        for target in (1e-2, 1e-3, 1e-5, 1e-7):
            bits = min_float_mantissa_bits(target)
            self.assertLessEqual(2.0 ** -(bits + 1), target)  # the chosen precision meets the budget
            if bits > 0:  # and it is minimal: one fewer bit would violate it
                self.assertGreater(2.0**-bits, target)

    def test_min_float_mantissa_bits_rejects_nonpositive(self):
        with self.assertRaises(ValueError):
            min_float_mantissa_bits(0.0)

    def test_allocation_picks_smallest_adequate_format(self):
        # use the bound to choose the cheapest float that keeps relative error under 1e-4
        bits = min_float_mantissa_bits(1e-4)
        fmt = FloatFormat(mantissa_bits=bits)
        x = np.random.RandomState(7).randn(2000)
        rel = np.abs(fmt.round_trip(x) - x) / np.maximum(np.abs(x), 1e-300)
        self.assertLessEqual(float(rel.max()), 1e-4)
        self.assertLess(bits, 52)  # genuinely cheaper than full float64
        self.assertEqual(bits, math.ceil(-math.log2(1e-4) - 1))


if __name__ == "__main__":
    unittest.main()
