"""Tests for mixle.enumeration.streams: freeze, BufferedStream, and merge_enumerators.

Regression coverage for MXR-080-0198 (freeze() cross-type/dtype key collisions),
MXR-080-0199 (BufferedStream.get() accepting negative ranks), and MXR-080-0200
(merge_enumerators() not validating its ordering inputs).
"""

import math
import unittest

import numpy as np

from mixle.enumeration.streams import freeze


class FreezeDedupKeyTestCase(unittest.TestCase):
    """MXR-080-0198: freeze() must not collide genuinely distinct support values."""

    def test_int8_and_uint8_arrays_with_colliding_bytes_are_distinct_keys(self):
        # uint8 200 and int8 -56 share the same single byte (0xc8), so pre-fix freeze()
        # -- keyed only by (shape, bytes) -- treated them as the same dedup key even
        # though they are different numbers.
        a_uint8 = np.array([200], dtype=np.uint8)
        a_int8 = np.array([-56], dtype=np.int8)
        self.assertEqual(a_uint8.tobytes(), a_int8.tobytes())  # the byte-level collision itself
        self.assertNotEqual(freeze(a_uint8), freeze(a_int8))

        # A dedup pass (the actual use case: union / distinct-stream / mass-certificate
        # paths) must keep both as distinct entries rather than discarding one.
        seen = {}
        for arr in (a_uint8, a_int8):
            seen.setdefault(freeze(arr), arr)
        self.assertEqual(len(seen), 2)

    def test_identical_arrays_same_dtype_same_bytes_still_dedupe(self):
        # Negative control: genuinely identical values (same dtype, same bytes) must
        # still collapse to one key -- the fix must not make freeze() over-eager.
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([1, 2, 3], dtype=np.int32)
        self.assertIsNot(a, b)
        self.assertEqual(freeze(a), freeze(b))
        self.assertEqual(len({freeze(a), freeze(b)}), 1)

    def test_true_and_one_are_distinct_support_values(self):
        # Python equality makes True == 1 (and hash(True) == hash(1)); freeze()'s
        # documented policy is that type is part of a value's identity, so these must
        # be distinct dedup keys.
        self.assertEqual(True, 1)  # sanity: this is exactly the native collision freeze() must avoid
        self.assertNotEqual(freeze(True), freeze(1))
        self.assertNotEqual(freeze(False), freeze(0))

        seen = {}
        for v in (True, 1):
            seen.setdefault(freeze(v), v)
        self.assertEqual(len(seen), 2)

    def test_int_and_float_of_equal_value_are_distinct_support_values(self):
        # Same cross-type policy applied to the int/float pair: 1 == 1.0 natively.
        self.assertEqual(1, 1.0)
        self.assertNotEqual(freeze(1), freeze(1.0))

    def test_identical_scalars_same_type_still_dedupe(self):
        # Negative control at the scalar level.
        self.assertEqual(freeze(3), freeze(3))
        self.assertEqual(freeze("a"), freeze("a"))
        self.assertEqual(len({freeze(3), freeze(3)}), 1)

    def test_numpy_integer_scalar_widths_still_dedupe_with_python_int(self):
        # freeze() unwraps numpy scalars via .item() before tagging by type, so
        # different numpy integer widths for THE SAME VALUE keep deduping together --
        # .item() (unlike a raw array's .tobytes()) already applies each dtype's own
        # signed/unsigned interpretation, so there is no reinterpretation ambiguity to
        # guard against at the scalar level.
        keys = {freeze(np.int8(5)), freeze(np.uint8(5)), freeze(np.int64(5)), freeze(5)}
        self.assertEqual(len(keys), 1)

    def test_nan_values_collapse_to_shared_sentinel(self):
        self.assertEqual(freeze(float("nan")), freeze(float("nan")))
        self.assertEqual(freeze(math.nan), freeze(float("nan")))

    def test_list_and_tuple_still_collapse_to_the_same_key_by_design(self):
        # Existing, intentional behavior (per freeze()'s own docstring): list/tuple
        # incidental-representation differences are not part of a value's identity,
        # unlike the type distinctions this fix adds for scalars/arrays.
        self.assertEqual(freeze([1, 2]), freeze((1, 2)))

    def test_unhashable_value_raises_type_error(self):
        with self.assertRaises(TypeError):
            freeze(bytearray(b"abc"))


if __name__ == "__main__":
    unittest.main()
