"""Tests for the non-buffering streaming token source (mixle.data.stream_token_source)."""

import unittest

import numpy as np

from mixle.data.stream_token_source import stream_token_source


class BasicContractTestCase(unittest.TestCase):
    """Baseline shape/dtype/determinism contract stream_token_source must keep regardless of the
    MXR-080-0062/0063 validation work: ``(context (b, block) int64, next_token (b,) int64)`` batches, built
    lazily on the fly, and reproducible given the same seed."""

    def test_shapes_and_dtypes(self):
        ids = np.arange(30)
        batches = list(stream_token_source(ids, block=4, batch_size=6, epochs=1, shuffle=False, seed=0))
        self.assertGreater(len(batches), 0)
        for ctx, nxt in batches:
            self.assertEqual(ctx.ndim, 2)
            self.assertEqual(ctx.shape[1], 4)
            self.assertEqual(nxt.ndim, 1)
            self.assertEqual(ctx.shape[0], nxt.shape[0])
            self.assertEqual(ctx.dtype, np.int64)
            self.assertEqual(nxt.dtype, np.int64)

    def test_unshuffled_windows_and_targets_match_sliding_window_by_hand(self):
        ids = np.arange(10)
        batches = list(stream_token_source(ids, block=3, batch_size=100, epochs=1, shuffle=False, seed=0))
        self.assertEqual(len(batches), 1)
        ctx, nxt = batches[0]
        expected_ctx = np.stack([ids[i : i + 3] for i in range(len(ids) - 3)])
        expected_nxt = ids[3:]
        np.testing.assert_array_equal(ctx, expected_ctx)
        np.testing.assert_array_equal(nxt, expected_nxt)

    def test_same_seed_reproduces_bitwise_identical_batches(self):
        ids = np.arange(50)
        a = list(stream_token_source(ids, block=4, batch_size=7, epochs=2, shuffle=True, seed=11))
        b = list(stream_token_source(ids, block=4, batch_size=7, epochs=2, shuffle=True, seed=11))
        self.assertEqual(len(a), len(b))
        for (ctx_a, nxt_a), (ctx_b, nxt_b) in zip(a, b):
            np.testing.assert_array_equal(ctx_a, ctx_b)
            np.testing.assert_array_equal(nxt_a, nxt_b)

    def test_corpus_shorter_than_block_yields_nothing(self):
        # A legitimate degenerate case (distinct from any of the MXR-080 findings): not enough tokens to
        # form even one window is a clean no-op, not an error.
        ids = np.arange(3)
        batches = list(stream_token_source(ids, block=10, batch_size=4, epochs=1))
        self.assertEqual(batches, [])

    def test_multiple_epochs_repeats_the_full_sweep_each_epoch(self):
        ids = np.arange(20)
        one_epoch = list(stream_token_source(ids, block=4, batch_size=100, epochs=1, shuffle=False, seed=0))
        three_epochs = list(stream_token_source(ids, block=4, batch_size=100, epochs=3, shuffle=False, seed=0))
        self.assertEqual(len(three_epochs), 3 * len(one_epoch))


class TokenValidationTestCase(unittest.TestCase):
    """MXR-080-0062: token ids are validated -- finite, exact-integer, in a lossless int64 range -- instead
    of silently cast, and stay int64 (never downcast to float32) all the way through to yielded batches."""

    def test_fractional_token_ids_are_rejected(self):
        # The exact MXR-080-0062 repro: stream_token_source([1.2, 2.8, 3.9], ...) used to silently return
        # truncated next-token ids [2, 3] instead of raising.
        with self.assertRaises(ValueError):
            list(stream_token_source([1.2, 2.8, 3.9], block=1, batch_size=8))

    def test_nan_and_inf_token_ids_are_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                list(stream_token_source([1.0, 2.0, bad, 4.0], block=1, batch_size=8))

    def test_non_1d_token_array_is_rejected(self):
        ids = np.arange(12).reshape(3, 4)
        with self.assertRaises(ValueError):
            list(stream_token_source(ids, block=1, batch_size=8))

    def test_out_of_int64_range_token_ids_are_rejected(self):
        ids = np.array([1.0, 2.0**64])
        with self.assertRaises(ValueError):
            list(stream_token_source(ids, block=1, batch_size=8))

    def test_exact_integer_valued_floats_are_accepted(self):
        # 5.0 is an exact integer even though it happens to be stored as a float -- must NOT be rejected.
        ids = np.array([3.0, 4.0, 5.0, 6.0])
        ctx, nxt = next(iter(stream_token_source(ids, block=2, batch_size=8, shuffle=False)))
        self.assertEqual(ctx.dtype, np.int64)
        np.testing.assert_array_equal(ctx[0], [3, 4])
        self.assertEqual(nxt[0], 5)

    def test_context_stays_int64_not_float32(self):
        ids = np.arange(30)
        for ctx, nxt in stream_token_source(ids, block=4, batch_size=6, shuffle=False):
            self.assertEqual(ctx.dtype, np.int64)
            self.assertEqual(nxt.dtype, np.int64)

    def test_token_identity_preserved_above_2_pow_24(self):
        # A float32 context would collide 2**24 and 2**24 + 1 (both round to the same float32 value);
        # int64 must not.
        big = 2**24
        ids = np.array([big, big + 1, big + 2, big + 3, big + 4])
        ctx, nxt = next(iter(stream_token_source(ids, block=4, batch_size=8, shuffle=False)))
        np.testing.assert_array_equal(ctx[0], [big, big + 1, big + 2, big + 3])
        self.assertEqual(len(set(ctx[0].tolist())), 4)  # all four values distinguishable, no collision
        self.assertEqual(int(nxt[0]), big + 4)

    def test_integer_dtype_input_is_accepted_without_precision_loss(self):
        ids = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        ctx, _nxt = next(iter(stream_token_source(ids, block=2, batch_size=8, shuffle=False)))
        self.assertEqual(ctx.dtype, np.int64)


class ControlValidationTestCase(unittest.TestCase):
    """MXR-080-0063: block/batch_size/epochs/seed are validated eagerly -- when stream_token_source is
    CALLED, before the returned iterator yields anything -- rather than lazily on first iteration, and are
    never silently cast/truncated or silently turned into a zero-batch no-op (``epochs=0`` is the one
    legitimate, explicit zero-work case)."""

    def test_zero_batch_size_raises_eagerly_at_call_time_not_on_first_iteration(self):
        # Previously this reached a bare `range(..., step=0)` deep inside the (lazy) generator body, so the
        # call itself succeeded and only the FIRST next() raised a cryptic ValueError. The call itself must
        # now raise immediately -- note there is no list()/next() here, just the call.
        with self.assertRaises(ValueError):
            stream_token_source(np.arange(20), block=2, batch_size=0, epochs=1)

    def test_negative_batch_size_rejected_not_silently_empty(self):
        # Previously batch_size=-3 constructed a working generator that silently yielded ZERO batches
        # forever (range(0, n, -3) never advances past a positive n) -- no error at all.
        with self.assertRaises(ValueError):
            stream_token_source(np.arange(20), block=2, batch_size=-3, epochs=1)

    def test_negative_epochs_rejected_not_silently_empty(self):
        with self.assertRaises(ValueError):
            stream_token_source(np.arange(20), block=2, batch_size=4, epochs=-2)

    def test_zero_epochs_is_a_clean_no_op_not_an_error(self):
        out = list(stream_token_source(np.arange(20), block=2, batch_size=4, epochs=0))
        self.assertEqual(out, [])

    def test_zero_and_negative_block_rejected(self):
        for bad_block in (0, -1):
            with self.assertRaises(ValueError):
                stream_token_source(np.arange(20), block=bad_block, batch_size=4, epochs=1)

    def test_fractional_block_rejected_not_silently_truncated(self):
        # Previously int(2.7) == 2 silently drove a block=2 sliding window with no error at all.
        with self.assertRaises(ValueError):
            stream_token_source(np.arange(20), block=2.7, batch_size=4, epochs=1)

    def test_fractional_batch_size_rejected(self):
        with self.assertRaises(ValueError):
            stream_token_source(np.arange(20), block=2, batch_size=3.5, epochs=1)

    def test_fractional_epochs_rejected(self):
        with self.assertRaises(ValueError):
            stream_token_source(np.arange(20), block=2, batch_size=4, epochs=1.5)

    def test_invalid_seed_rejected_eagerly(self):
        with self.assertRaises((ValueError, TypeError)):
            stream_token_source(np.arange(20), block=2, batch_size=4, epochs=1, seed=-1)

    def test_bool_block_rejected(self):
        # bool is an int subclass in Python -- block=True must not silently behave like block=1.
        with self.assertRaises(TypeError):
            stream_token_source(np.arange(20), block=True, batch_size=4, epochs=1)


if __name__ == "__main__":
    unittest.main()
