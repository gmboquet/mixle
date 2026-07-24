"""Tests for mixle.enumeration.streams: freeze, BufferedStream, and merge_enumerators.

Regression coverage for MXR-080-0198 (freeze() cross-type/dtype key collisions),
MXR-080-0199 (BufferedStream.get() accepting negative ranks), and MXR-080-0200
(merge_enumerators() not validating its ordering inputs).
"""

import math
import unittest

import numpy as np

from mixle.enumeration.streams import BufferedStream, freeze, merge_enumerators


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


class BufferedStreamRankValidationTestCase(unittest.TestCase):
    """MXR-080-0199: BufferedStream.get() must reject negative/non-integer ranks
    uniformly, regardless of how much of the stream has already been buffered."""

    @staticmethod
    def _stream():
        return iter([("a", -0.1), ("b", -0.5), ("c", -1.0)])

    def test_negative_rank_rejected_with_empty_buffer(self):
        buf = BufferedStream(self._stream())
        with self.assertRaises(ValueError):
            buf.get(-1)

    def test_negative_rank_rejected_after_buffering(self):
        # Pre-fix, get(-1) after at least one item is buffered returns that buffered
        # item via ordinary Python negative indexing instead of raising -- this is
        # exactly the history-dependent behavior the fix closes off. Proving both this
        # case and the empty-buffer case above raise the same way is the key regression
        # coverage: rank validation must not depend on prior access history.
        buf = BufferedStream(self._stream())
        self.assertEqual(buf.get(0), ("a", -0.1))
        with self.assertRaises(ValueError):
            buf.get(-1)

    def test_negative_rank_rejected_consistently_at_various_buffer_depths(self):
        for depth in range(4):
            buf = BufferedStream(self._stream())
            for r in range(depth):
                buf.get(r)
            with self.assertRaises(ValueError):
                buf.get(-1)
            with self.assertRaises(ValueError):
                buf.get(-2)

    def test_non_integer_rank_rejected(self):
        buf = BufferedStream(self._stream())
        with self.assertRaises(TypeError):
            buf.get(2.5)
        with self.assertRaises(TypeError):
            buf.get(2.0)  # whole-valued float is still not an exact integer
        with self.assertRaises(TypeError):
            buf.get("0")
        with self.assertRaises(TypeError):
            buf.get(None)

    def test_bool_rank_rejected_despite_being_an_int_subclass(self):
        buf = BufferedStream(self._stream())
        with self.assertRaises(TypeError):
            buf.get(True)
        with self.assertRaises(TypeError):
            buf.get(False)

    def test_valid_nonnegative_integer_ranks_still_retrieve_correctly(self):
        # Negative control: legitimate ranks are unaffected by the validation.
        buf = BufferedStream(self._stream())
        self.assertEqual(buf.get(0), ("a", -0.1))
        self.assertEqual(buf.get(2), ("c", -1.0))
        self.assertEqual(buf.get(1), ("b", -0.5))  # already buffered by the get(2) above
        self.assertIsNone(buf.get(3))  # past the end of a length-3 stream

    def test_numpy_integer_rank_accepted(self):
        buf = BufferedStream(self._stream())
        self.assertEqual(buf.get(np.int64(1)), ("b", -0.5))


class MergeEnumeratorsTestCase(unittest.TestCase):
    """MXR-080-0200: merge_enumerators() must validate its ordering inputs."""

    def test_arity_mismatch_too_few_offsets_rejected(self):
        streams = [iter([("a", -0.1)]), iter([("b", -0.2)])]
        with self.assertRaises(ValueError):
            merge_enumerators(streams, [0.0])  # one offset for two streams

    def test_arity_mismatch_too_many_offsets_rejected_eagerly(self):
        # merge_enumerators() is not itself a generator: a malformed call must raise at
        # call time, not only once the caller starts pulling from the returned iterator
        # (this is what makes the eager offsets[k] IndexError pre-fix possible at all).
        streams = [iter([("a", -0.1)]), iter([("b", -0.2)])]
        with self.assertRaises(ValueError):
            merge_enumerators(streams, [0.0, 0.0, 0.0])  # three offsets for two streams

    def test_nan_offset_rejected(self):
        streams = [iter([("a", -0.1)]), iter([("b", -0.2)])]
        with self.assertRaises(ValueError):
            merge_enumerators(streams, [0.0, float("nan")])

    def test_positive_infinite_offset_rejected(self):
        streams = [iter([("a", -0.1)]), iter([("b", -0.2)])]
        with self.assertRaises(ValueError):
            merge_enumerators(streams, [0.0, math.inf])

    def test_negative_infinite_offset_is_the_documented_exclude_stream_sentinel(self):
        # Negative control: -inf is NOT rejected -- it deliberately means "this stream
        # contributes nothing," and that stream's iterator must never even be opened.
        def _poison():
            raise AssertionError("excluded stream must never be iterated")
            yield  # pragma: no cover - unreachable; only here to make this a generator fn

        streams = [iter([("a", -0.1), ("b", -0.3)]), _poison()]
        out = list(merge_enumerators(streams, [0.0, -math.inf]))
        self.assertEqual([v for v, _ in out], ["a", "b"])

    def test_non_finite_score_rejected_nan(self):
        with self.assertRaises(ValueError):
            list(merge_enumerators([iter([("a", float("nan"))])], [0.0]))

    def test_non_finite_score_rejected_positive_infinity(self):
        with self.assertRaises(ValueError):
            list(merge_enumerators([iter([("a", math.inf)])], [0.0]))

    def test_non_finite_score_rejected_negative_infinity(self):
        # A properly-formed stream should already exclude impossible (-inf) items --
        # matching the convention used elsewhere in this package -- so one leaking
        # through here is treated as malformed, the same as NaN or +inf.
        with self.assertRaises(ValueError):
            list(merge_enumerators([iter([("a", -math.inf)])], [0.0]))

    def test_non_descending_stream_is_detected(self):
        # Deliberately increasing (wrong-order) scores within a single stream: a k-way
        # merge's correctness depends entirely on each input already being sorted
        # descending, so this must be caught rather than silently producing a
        # wrong-order output.
        bad_stream = iter([("a", -3.0), ("b", -1.0)])  # -1.0 > -3.0: not descending
        with self.assertRaises(ValueError):
            list(merge_enumerators([bad_stream], [0.0]))

    def test_well_formed_descending_streams_merge_into_correct_global_order(self):
        # Negative control: verify the actual output ordering, not just that it runs.
        s1 = iter([("a", -0.1), ("c", -0.5), ("e", -2.0)])
        s2 = iter([("b", -0.2), ("d", -1.0)])
        out = list(merge_enumerators([s1, s2], [0.0, 0.0]))
        values = [v for v, _ in out]
        lps = [lp for _, lp in out]
        self.assertEqual(values, ["a", "b", "c", "d", "e"])
        for i in range(len(lps) - 1):
            self.assertGreaterEqual(lps[i], lps[i + 1])

    def test_offsets_are_correctly_folded_into_the_merged_order(self):
        s1 = iter([("a", -0.1), ("c", -0.5)])
        s2 = iter([("b", -0.2)])
        out = list(merge_enumerators([s1, s2], [0.0, 1.0]))  # shift stream 2 so it dominates
        self.assertEqual([v for v, _ in out], ["b", "a", "c"])
        self.assertAlmostEqual(out[0][1], 0.8)  # -0.2 + 1.0

    def test_empty_streams_and_offsets_yield_nothing(self):
        self.assertEqual(list(merge_enumerators([], [])), [])


if __name__ == "__main__":
    unittest.main()
