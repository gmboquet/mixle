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
    def test_format_widths_reject_bool_fractional_negative_and_nonfinite_values(self):
        for bad in (True, -1, 2.5, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                FloatFormat(bad)
            with self.assertRaises(ValueError):
                FloatFormat.fp(bad)

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

    # -- MXR-080-0126: exp_bits used to only change the NAME/advertised bit count while quantize
    # rounded the mantissa with a completely unbounded exponent -- no overflow, underflow, subnormal,
    # or special-value handling. The nominal fp8 codec round-tripped 1e100 to ~9.84e99, which no real
    # fp8 format can represent. Chose the documented-fallback fix here rather than implementing bounded
    # exponent semantics: this module's own error_tracing_test.py::test_from_quantized_encloses_original
    # calls Interval.from_quantized(randn(3000, seed=2), FloatFormat.fp(8)/.fp(16)) and trusts
    # max_rel_error as a universal analytic bound to build a *sound* enclosure. Measured directly: with
    # that exact seed, 40/3000 values have |x| below a real fp8's smallest normal magnitude (2**-6) and
    # 2/3000 fall below fp16's (2**-14) -- so adding real underflow-flush-to-zero would make quantize
    # return exactly 0 for a nonzero original, and Interval.from_quantized's analytic-bound path would
    # then produce a zero-width [0, 0] enclosure that provably does NOT contain the true value,
    # deterministically breaking that test's soundness assertion (a file this session must not edit).
    # Real bounded-exponent saturation has the same defect in the overflow direction (not exercised by
    # that test's data, but the identical class of unsoundness). Rather than silently trade "claims fpN
    # semantics it doesn't implement" for "claims a sound error bound it can no longer guarantee",
    # this keeps quantize's numeric behavior byte-identical (a universal, always-true relative bound)
    # and instead makes the NAME and docstrings honest about what it actually does: mantissa-only
    # rounding with a nominal, non-enforced exponent width.
    def test_mantissa_only_naming_does_not_claim_bounded_fpn(self):
        fmt = FloatFormat.fp(8)
        self.assertNotRegex(fmt.name, r"^fp\d+$")  # no longer an unqualified "fp8"-style label
        self.assertEqual(fmt.mantissa_bits, 3)  # the e4m3-style split is still computed and exposed
        self.assertEqual(fmt.exp_bits, 4)

    def test_extreme_magnitude_rounds_mantissa_without_saturating_or_crashing(self):
        # The audit's exact repro. Since exp_bits is documented as non-enforced (mantissa-only
        # rounding), the correct current contract is: no crash, no silent collapse to a small/zero/inf
        # value, and the RELATIVE error still respects max_rel_error at this magnitude.
        fmt = FloatFormat.fp(8)
        rt = fmt.round_trip(np.array([1e100]))
        self.assertTrue(np.all(np.isfinite(rt)))
        self.assertGreater(float(rt[0]), 1e99)  # same astronomical magnitude, not collapsed
        rel_err = abs(float(rt[0]) - 1e100) / 1e100
        self.assertLessEqual(rel_err, fmt.max_rel_error * 1.01)

    def test_round_trip_relative_error_holds_across_full_magnitude_sweep(self):
        # The honest contract (class docstring): max_rel_error is a UNIVERSAL bound, true at every
        # finite magnitude -- not an "in representable range" bound the way a real bounded fpN would
        # need. Sweep tiny, small, unit, large, and astronomical magnitudes, both signs.
        rng = np.random.RandomState(9)
        magnitudes = (1e-300, 1e-30, 1e-3, 1.0, 1e3, 1e30, 1e100, 1e300)
        for nbits in (8, 16, 32):
            fmt = FloatFormat.fp(nbits)
            x = np.array(magnitudes) * rng.choice([-1.0, 1.0], size=len(magnitudes))
            rt = fmt.round_trip(x)
            rel = np.abs(rt - x) / np.abs(x)
            self.assertLessEqual(float(rel.max()), fmt.max_rel_error * 1.01, "nbits=%d" % nbits)

    def test_negative_control_small_value_matches_hand_computed_mantissa_rounding(self):
        # Negative control: confirm rounding actually happens as documented (not a no-op) by
        # hand-computing the expected mantissa-rounded value for a simple input at low precision.
        fmt = FloatFormat(mantissa_bits=2)  # 2 explicit mantissa bits -> scale = 2**3 = 8
        m, e = math.frexp(1.3)  # 1.3 == 0.65 * 2**1
        expected = math.ldexp(round(m * 8) / 8, e)
        got = float(fmt.round_trip(np.array([1.3]))[0])
        self.assertEqual(got, expected)
        self.assertNotEqual(got, 1.3)  # genuinely lossy at this precision, not silently exact


class FixedPointFormatTest(unittest.TestCase):
    def test_round_trip_within_absolute_bound_and_compresses(self):
        rng = np.random.RandomState(3)
        x = rng.uniform(-100, 100, 5000)
        fmt = FixedPointFormat(frac_bits=12, int_bits=10)  # 23 bits -> int32 storage
        q = fmt.quantize(x)
        self.assertEqual(q.dtype, np.int32)
        self.assertLessEqual(fmt.measured_max_abs_error(x), fmt.in_range_max_abs_error * 1.01)
        self.assertTrue(math.isinf(fmt.max_abs_error))
        self.assertGreater(fmt.compression_ratio(), 2.0)  # 64 / 23 bits

    def test_out_of_range_clamps(self):
        fmt = FixedPointFormat(frac_bits=4, int_bits=3)  # range ~[-8, 8)
        rt = fmt.round_trip(np.array([1000.0, -1000.0]))
        self.assertTrue(np.all(np.abs(rt) <= 8.0))
        self.assertGreater(fmt.measured_max_abs_error([1000.0]), fmt.in_range_max_abs_error)

    def test_constructor_and_nonfinite_input_contracts(self):
        for bad in (True, -1, 1.5, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                FixedPointFormat(frac_bits=bad)
            with self.assertRaises(ValueError):
                FixedPointFormat(frac_bits=1, int_bits=bad)
        fmt = FixedPointFormat(frac_bits=4, int_bits=3)
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                fmt.round_trip([bad])

    # -- MXR-080-0127: every format wider than 32 bits used int64 storage regardless of the DECLARED
    # width, so a 65-bit fixed(i32.f32) silently overflowed the int64 cast during quantize and decoded
    # near 2.147e9 (~2**31) instead of its documented ~4.295e9 (~2**32) upper range -- falsifying
    # max_abs_error by billions. Fixed by rejecting widths that exceed int64 storage at construction
    # (arbitrary-width packed storage was judged disproportionate: this codec's quantize/dequantize are
    # vectorized numpy int32/int64 ops with no wider native dtype to fall back to).
    def test_wide_format_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            FixedPointFormat(frac_bits=32, int_bits=32)  # exactly the audit's fixed(i32.f32): 65 bits

    def test_boundary_64_bit_format_still_works_and_decodes_correctly(self):
        # Negative control: the widest format that DOES fit (total == 64 bits, exactly int64's range)
        # must still construct and decode correctly near its documented upper range.
        fmt = FixedPointFormat(frac_bits=32, int_bits=31)  # 1 + 31 + 32 = 64 bits -> exactly int64
        self.assertEqual(fmt.bits_per_value, 64.0)
        near_max = 2.0**31 - 1.0  # just under the documented int_bits=31 upper range
        rt = fmt.round_trip(np.array([near_max]))
        self.assertLessEqual(abs(float(rt[0]) - near_max), fmt.in_range_max_abs_error * 1.01)

    def test_widths_up_to_64_bits_are_not_rejected(self):
        # Only widths that exceed int64 storage should raise; ordinary widths -- including <=32-bit
        # int32-backed and 33..64-bit int64-backed formats -- must keep working.
        for frac_bits, int_bits in [(4, 3), (12, 10), (20, 20), (31, 32)]:  # totals: 8, 23, 41, 64
            fmt = FixedPointFormat(frac_bits=frac_bits, int_bits=int_bits)
            self.assertLessEqual(fmt.bits_per_value, 64.0)
            rt = fmt.round_trip(np.array([0.0, 1.5, -1.5]))
            self.assertTrue(np.all(np.isfinite(rt)))


class CodebookFormatTest(unittest.TestCase):
    def test_constructor_copies_and_freezes_one_dimensional_codebook(self):
        original = np.array([3.0, 1.0, 2.0])
        cb = CodebookFormat(original)
        self.assertTrue(np.array_equal(original, [3.0, 1.0, 2.0]))
        original[0] = 99.0
        self.assertTrue(np.array_equal(cb.codebook, [1.0, 2.0, 3.0]))
        self.assertFalse(cb.codebook.flags.writeable)
        with self.assertRaises(ValueError):
            cb.codebook[0] = 0.0

    def test_constructor_rejects_scalar_and_multidimensional_codebooks(self):
        for bad in (1.0, np.ones((2, 2))):
            with self.assertRaises(ValueError):
                CodebookFormat(bad)

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

    # -- MXR-080-0128: codebooks above 256 entries need 16/32-bit indices (see _idx_dtype), but
    # _pack_bits capped the packing width at 8 regardless, so compress() raised for any code >= 256
    # even though quantize() produced it successfully (a 300-code format's code 299 needs 9 bits).
    # Fixed by making _pack_bits follow the full declared index width (16 or 32 bits, byte-aligned
    # little-endian) instead of capping at 8; also added construction-time validation for the codebook
    # itself (finite, nonempty) and for fit()'s n_codes/iters (positive exact integers), none of which
    # had a coherent contract before.
    def test_large_codebook_compress_succeeds_for_high_codes(self):
        cb = CodebookFormat(np.linspace(-10.0, 10.0, 300))
        x = np.array([10.0, -10.0, 0.0, 3.3])
        idx = cb.quantize(x)
        self.assertEqual(int(idx.max()), 299)  # code 299 -- exactly the audit's previously-failing case
        self.assertEqual(idx.dtype, np.uint16)
        packed, n = cb.compress(x)  # used to raise ValueError: "a code does not fit in 8 bits"
        self.assertEqual(n, x.size)
        self.assertEqual(packed.nbytes, 2 * n)  # 16-bit index per value, not capped at 8
        back = cb.decompress(packed, n)
        self.assertTrue(np.array_equal(back, cb.round_trip(x)))

    def test_very_large_codebook_uses_32_bit_index_path(self):
        # Beyond 65536 codes, quantize's indices step up to uint32; compress must follow with a
        # matching 32-bit-per-code packed width rather than staying capped at 8 or 16 bits.
        cb = CodebookFormat(np.linspace(-1.0, 1.0, 70000))
        x = np.linspace(-1.0, 1.0, 50)
        self.assertEqual(cb.quantize(x).dtype, np.uint32)
        packed, n = cb.compress(x)
        self.assertEqual(packed.nbytes, 4 * n)  # 32-bit index per value, byte-aligned (not sub-byte)
        back = cb.decompress(packed, n)
        self.assertTrue(np.array_equal(back, cb.round_trip(x)))

    def test_empty_codebook_rejected(self):
        with self.assertRaises(ValueError):
            CodebookFormat(np.array([]))

    def test_non_finite_codebook_rejected(self):
        for bad in (np.array([1.0, np.nan, 3.0]), np.array([1.0, np.inf, 3.0]), np.array([1.0, -np.inf, 3.0])):
            with self.assertRaises(ValueError):
                CodebookFormat(bad)

    def test_dequantize_rejects_invalid_indices(self):
        cb = CodebookFormat([1.0, 2.0, 3.0])
        for bad in ([-1], [3], [1.5], [True]):
            with self.assertRaises(ValueError):
                cb.dequantize(bad)

    def test_wide_decompress_validates_payload_and_count(self):
        cb = CodebookFormat(np.linspace(-1.0, 1.0, 300))
        packed, count = cb.compress([0.0, 0.5])
        for bad in ([1.5, 0], [-1, 0], [256, 0], [[0, 0]]):
            with self.assertRaises(ValueError):
                cb.decompress(bad, 1)
        with self.assertRaises(ValueError):
            cb.decompress(packed[:-1], count)
        with self.assertRaises(ValueError):
            cb.decompress(packed, count + 1)

    def test_fit_rejects_bool_counts_and_nonfinite_data(self):
        with self.assertRaises(ValueError):
            CodebookFormat.fit([1.0, 2.0], True)
        with self.assertRaises(ValueError):
            CodebookFormat.fit([1.0, float("nan")], 2)

    def test_duplicate_codebook_entries_are_allowed_and_round_trip_correctly(self):
        # Documented design decision (class docstring): duplicates cost a wasted index but do not break
        # nearest-code assignment or decoding -- both duplicate entries decode to the same, correct
        # value, and fit()'s Lloyd iteration can legitimately produce them on degenerate data.
        cb = CodebookFormat(np.array([1.0, 1.0, 3.0, 3.0, 5.0]))
        rt = cb.round_trip(np.array([1.0, 3.0, 5.0, 0.5, 6.0]))
        self.assertTrue(np.array_equal(rt, np.array([1.0, 3.0, 5.0, 1.0, 5.0])))

    def test_fit_rejects_nonpositive_or_non_integer_n_codes(self):
        data = np.random.RandomState(10).randn(200)
        for bad in (0, -3, 2.5):
            with self.assertRaises(ValueError):
                CodebookFormat.fit(data, bad)

    def test_fit_rejects_nonpositive_or_non_integer_iters(self):
        data = np.random.RandomState(11).randn(200)
        for bad in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                CodebookFormat.fit(data, 8, iters=bad)

    def test_negative_control_well_formed_codebook_still_compresses_round_trip(self):
        # Negative control: an ordinary, valid codebook must keep working exactly as before.
        rng = np.random.RandomState(12)
        data = np.concatenate([rng.normal(-2, 0.5, 3000), rng.normal(2, 0.5, 3000)])
        fmt = CodebookFormat.fit(data, 32)
        packed, n = fmt.compress(data)
        back = fmt.decompress(packed, n)
        self.assertTrue(np.array_equal(back, fmt.round_trip(data)))
        self.assertLess(fmt.measured_max_abs_error(data), 1.0)


class ErrorTracingTest(unittest.TestCase):
    def test_min_float_mantissa_bits_meets_target(self):
        for target in (1e-2, 1e-3, 1e-5, 1e-7):
            bits = min_float_mantissa_bits(target)
            self.assertLessEqual(2.0 ** -(bits + 1), target)  # the chosen precision meets the budget
            if bits > 0:  # and it is minimal: one fewer bit would violate it
                self.assertGreater(2.0**-bits, target)

    def test_min_float_mantissa_bits_rejects_nonpositive(self):
        for bad in (0.0, -1.0, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                min_float_mantissa_bits(bad)

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
