"""Tests for the F3 streaming tokenized-data pipeline (mixle.data.streaming_corpus)."""

import unittest

import numpy as np

from mixle.data.streaming_corpus import (
    StreamingCorpus,
    global_document_order,
    pack_documents,
    shard_documents_for_rank,
)


def _synthetic_corpus(rng, n_docs=200, min_len=1, max_len=40):
    """Documents with a mix of short and long lengths relative to a small block, like a real corpus."""
    return [rng.randint(1, 5000, size=int(rng.randint(min_len, max_len + 1))) for _ in range(n_docs)]


class ShardingTestCase(unittest.TestCase):
    """Per-rank sharding correctness -- simulate several ranks in-process, mirroring how MPEncodedData's
    own tests exercise its worker split without a real cluster."""

    def test_disjoint_and_complete_coverage(self):
        order = np.arange(37)  # unshuffled order is enough to test the sharding contract itself
        world_size = 4
        shards = [shard_documents_for_rank(order, rank, world_size) for rank in range(world_size)]

        seen = np.concatenate(shards)
        self.assertEqual(sorted(seen.tolist()), list(range(37)))  # complete coverage, no duplicates

        for a in range(world_size):
            for b in range(a + 1, world_size):
                self.assertEqual(set(shards[a].tolist()) & set(shards[b].tolist()), set())  # disjoint

    def test_matches_round_robin_contract_like_mpencoded_data(self):
        # MPEncodedData shards raw data as `data[j] for j in range(i, n, num_workers)` (see
        # mixle/utils/parallel/multiprocessing.py). shard_documents_for_rank applies the identical
        # round-robin split, just to a (possibly shuffled) `order` array instead of raw positions.
        order = np.arange(23)
        world_size = 5
        for rank in range(world_size):
            expected = order[rank::world_size]
            actual = shard_documents_for_rank(order, rank, world_size)
            np.testing.assert_array_equal(actual, expected)

    def test_shards_of_shuffled_order_still_disjoint_and_complete(self):
        order = global_document_order(50, seed=7, epoch=0)
        world_size = 3
        shards = [shard_documents_for_rank(order, rank, world_size) for rank in range(world_size)]
        seen = np.concatenate(shards)
        self.assertEqual(sorted(seen.tolist()), list(range(50)))
        for a in range(world_size):
            for b in range(a + 1, world_size):
                self.assertEqual(set(shards[a].tolist()) & set(shards[b].tolist()), set())

    def test_invalid_rank_rejected(self):
        order = np.arange(10)
        with self.assertRaises(ValueError):
            shard_documents_for_rank(order, rank=3, world_size=3)
        with self.assertRaises(ValueError):
            shard_documents_for_rank(order, rank=-1, world_size=3)
        with self.assertRaises(ValueError):
            shard_documents_for_rank(order, rank=0, world_size=0)


class DeterminismTestCase(unittest.TestCase):
    """Same (seed, epoch) -> bitwise-identical global order & per-rank batches; different epoch -> different,
    still-deterministic order."""

    def _documents(self):
        rng = np.random.RandomState(0)
        return _synthetic_corpus(rng, n_docs=64, min_len=1, max_len=20)

    def test_same_seed_epoch_reproduces_global_order(self):
        order_a = global_document_order(64, seed=42, epoch=3)
        order_b = global_document_order(64, seed=42, epoch=3)
        np.testing.assert_array_equal(order_a, order_b)

    def test_different_epoch_same_seed_gives_different_order(self):
        order_a = global_document_order(64, seed=42, epoch=0)
        order_b = global_document_order(64, seed=42, epoch=1)
        self.assertFalse(np.array_equal(order_a, order_b))
        # both individually reproducible
        np.testing.assert_array_equal(order_a, global_document_order(64, seed=42, epoch=0))
        np.testing.assert_array_equal(order_b, global_document_order(64, seed=42, epoch=1))

    def test_different_seed_same_epoch_gives_different_order(self):
        order_a = global_document_order(64, seed=1, epoch=0)
        order_b = global_document_order(64, seed=2, epoch=0)
        self.assertFalse(np.array_equal(order_a, order_b))

    def test_per_rank_batches_bitwise_identical_across_independent_runs(self):
        documents = self._documents()

        def run():
            corpus = StreamingCorpus(documents, rank=1, world_size=3, block=8, batch_size=4, seed=123)
            return list(corpus.epoch_batches(epoch=5))

        run_a = run()
        run_b = run()
        self.assertEqual(len(run_a), len(run_b))
        self.assertGreater(len(run_a), 0)
        for (ctx_a, tgt_a, mask_a), (ctx_b, tgt_b, mask_b) in zip(run_a, run_b):
            np.testing.assert_array_equal(ctx_a, ctx_b)
            np.testing.assert_array_equal(tgt_a, tgt_b)
            np.testing.assert_array_equal(mask_a, mask_b)

    def test_different_epoch_changes_batches_deterministically(self):
        documents = self._documents()
        corpus = StreamingCorpus(documents, rank=0, world_size=2, block=8, batch_size=4, seed=123)
        epoch0 = list(corpus.epoch_batches(epoch=0))
        epoch1 = list(corpus.epoch_batches(epoch=1))

        flat0 = np.concatenate([c.reshape(-1) for c, _, _ in epoch0]) if epoch0 else np.array([])
        flat1 = np.concatenate([c.reshape(-1) for c, _, _ in epoch1]) if epoch1 else np.array([])
        self.assertFalse(np.array_equal(flat0, flat1))

        # re-running epoch 0 alone still reproduces epoch 0 exactly
        epoch0_again = list(corpus.epoch_batches(epoch=0))
        for (ctx_a, tgt_a, mask_a), (ctx_b, tgt_b, mask_b) in zip(epoch0, epoch0_again):
            np.testing.assert_array_equal(ctx_a, ctx_b)
            np.testing.assert_array_equal(tgt_a, tgt_b)
            np.testing.assert_array_equal(mask_a, mask_b)

    def test_global_batches_identical_across_full_rank_set_two_runs(self):
        """The determinism receipt at the corpus level: replaying the whole run (all ranks, one epoch)
        twice yields the identical set of per-rank batch sequences."""
        documents = self._documents()
        world_size = 4

        def run_all_ranks():
            out = []
            for rank in range(world_size):
                corpus = StreamingCorpus(documents, rank=rank, world_size=world_size, block=8, batch_size=4, seed=99)
                out.append([(c.copy(), t.copy(), m.copy()) for c, t, m in corpus.epoch_batches(epoch=2)])
            return out

        run_a = run_all_ranks()
        run_b = run_all_ranks()
        for rank in range(world_size):
            for (ctx_a, tgt_a, mask_a), (ctx_b, tgt_b, mask_b) in zip(run_a[rank], run_b[rank]):
                np.testing.assert_array_equal(ctx_a, ctx_b)
                np.testing.assert_array_equal(tgt_a, tgt_b)
                np.testing.assert_array_equal(mask_a, mask_b)


class PackingEfficiencyTestCase(unittest.TestCase):
    """Packing efficiency (real-token fraction) on a realistic short/long document mix clears a measured
    floor. Packing waste is bounded by one row (`block + 1` tokens), so efficiency should be high for any
    corpus that's not tiny relative to `block` -- verified concretely here, not asserted blindly."""

    def test_packing_efficiency_floor(self):
        rng = np.random.RandomState(11)
        documents = _synthetic_corpus(rng, n_docs=500, min_len=1, max_len=200)
        indices = np.arange(len(documents))
        packed = pack_documents(documents, indices, block=64, boundary_id=0)

        self.assertGreater(len(packed), 0)
        self.assertGreater(packed.packing_efficiency, 0.90)  # measured well above this in practice, see below
        self.assertLessEqual(packed.packing_efficiency, 1.0)

        # waste is bounded by a single row regardless of corpus size/shape
        max_possible_waste = 64 + 1
        self.assertLessEqual(packed.total_tokens - packed.real_tokens, max_possible_waste)

    def test_packing_efficiency_improves_with_corpus_size(self):
        # same length distribution, more documents -> waste (bounded, fixed) shrinks as a fraction of total
        rng = np.random.RandomState(3)
        small = _synthetic_corpus(np.random.RandomState(3), n_docs=5, min_len=1, max_len=50)
        large = _synthetic_corpus(np.random.RandomState(3), n_docs=2000, min_len=1, max_len=50)
        small_packed = pack_documents(small, np.arange(len(small)), block=32)
        large_packed = pack_documents(large, np.arange(len(large)), block=32)
        self.assertGreaterEqual(large_packed.packing_efficiency, small_packed.packing_efficiency)

    def test_empty_input_reports_full_efficiency_and_no_rows(self):
        packed = pack_documents([], [], block=16)
        self.assertEqual(len(packed), 0)
        self.assertEqual(packed.packing_efficiency, 1.0)


class CurriculumHookTestCase(unittest.TestCase):
    """A pluggable sequence_selector (the E7 extension point) actually influences which documents are
    sampled -- this module only exposes the hook, not any curriculum policy."""

    def test_custom_selector_filters_by_length_bucket(self):
        rng = np.random.RandomState(5)
        documents = [rng.randint(0, 100, size=n) for n in [2, 50, 3, 60, 4, 70, 1, 90]]
        short_ids = {i for i, d in enumerate(documents) if len(d) < 10}

        def only_short(order, seed, epoch):
            return np.asarray([i for i in order if len(documents[i]) < 10])

        order = global_document_order(len(documents), seed=1, epoch=0, sequence_selector=only_short)
        self.assertTrue(set(order.tolist()).issubset(short_ids))
        self.assertGreater(len(order), 0)

        baseline = global_document_order(len(documents), seed=1, epoch=0)
        self.assertLess(len(order), len(baseline))  # selector actually changed the sampled set

    def test_selector_wired_through_streaming_corpus_end_to_end(self):
        rng = np.random.RandomState(9)
        documents = [rng.randint(0, 100, size=n) for n in [2, 50, 3, 60, 4, 70, 1, 90, 5, 95]]

        def only_short(order, seed, epoch):
            return np.asarray([i for i in order if len(documents[i]) < 10])

        corpus = StreamingCorpus(
            documents, rank=0, world_size=1, block=4, batch_size=2, seed=0, sequence_selector=only_short
        )
        indices = corpus.rank_document_indices(epoch=0)
        for i in indices:
            self.assertLess(len(documents[i]), 10)

        without_selector = StreamingCorpus(documents, rank=0, world_size=1, block=4, batch_size=2, seed=0)
        self.assertLess(len(indices), len(without_selector.rank_document_indices(epoch=0)))


class LossMaskTestCase(unittest.TestCase):
    """MXR-080-0061: the final packed row's fabricated pad tokens must never be presented as real
    next-token training targets -- `pack_documents`/`epoch_batches` return a loss mask the caller can (and
    must) apply before averaging next-token loss."""

    def test_short_document_pad_targets_are_masked_false(self):
        # The exact MXR-080-0061 repro shape: a document shorter than `block` forces the sole packed row to
        # be padded, and naive targets look exactly like real observed labels at every padded position.
        doc = np.array([11, 22])
        packed = pack_documents([doc], [0], block=4)
        np.testing.assert_array_equal(packed.rows, [[11, 22, 0, 0, 0]])
        # row[1:] = [22, 0, 0, 0]: position 0 (value 22) is the real next token after 11; positions 1-3 are
        # fabricated pad_id, never observed, and must be masked out.
        np.testing.assert_array_equal(packed.loss_mask, [[True, False, False, False]])

    def test_single_real_token_row_has_zero_real_targets(self):
        # A one-token document has no successor at all -- its own target position is entirely pad, so the
        # WHOLE target row must be masked out (remainder=1 -> valid_targets = max(1-1, 0) = 0).
        doc = np.array([7])
        packed = pack_documents([doc], [0], block=3)
        np.testing.assert_array_equal(packed.rows, [[7, 0, 0, 0]])
        np.testing.assert_array_equal(packed.loss_mask, [[False, False, False]])

    def test_exactly_full_row_has_no_padding_and_mask_is_all_true(self):
        # A corpus that divides evenly into `unit = block + 1` tokens needs no padding at all.
        doc = np.arange(1, 6)  # 5 tokens; block=4 -> unit=5 -> exactly one full row, remainder=0
        packed = pack_documents([doc], [0], block=4)
        self.assertEqual(packed.real_tokens, packed.total_tokens)
        self.assertTrue(bool(np.all(packed.loss_mask)))

    def test_only_the_final_row_can_ever_carry_a_false_mask_entry(self):
        # Many documents -> several full rows plus one padded remainder row. Only the LAST row may contain
        # any False; every earlier row must be entirely real regardless of corpus size (packing waste is
        # bounded by a single row -- see pack_documents' docstring).
        rng = np.random.RandomState(0)
        documents = [rng.randint(1, 100, size=int(n)) for n in rng.randint(1, 30, size=50)]
        packed = pack_documents(documents, np.arange(len(documents)), block=8)
        self.assertGreater(len(packed), 1)  # more than one row, so "only the last row" is a real claim here
        self.assertTrue(bool(np.all(packed.loss_mask[:-1])))
        # any False in the last row is confined to a single trailing run (True*, then False*) -- never
        # interleaved back to True.
        last = packed.loss_mask[-1]
        false_after_true = np.sum(last[:-1] & ~last[1:])
        self.assertLessEqual(false_after_true, 1)

    def test_boundary_id_is_real_signal_not_masked(self):
        # boundary_id counts as a real (non-pad) token per pack_documents' contract -- it must never be
        # masked out just because it sits at a document edge.
        docs = [np.array([1, 2]), np.array([3, 4])]
        packed = pack_documents(docs, [0, 1], block=4, boundary_id=99)
        self.assertEqual(packed.real_tokens, 6)  # 4 doc tokens + 2 boundary tokens (one per document), all real
        rows = packed.rows
        real_run = packed.real_tokens  # positions [0, real_run) of the flattened stream are real tokens
        flat_pos = 0
        for r in range(rows.shape[0]):
            for j in range(rows.shape[1]):
                if rows[r, j] == 99 and flat_pos < real_run and j > 0:
                    self.assertTrue(bool(packed.loss_mask[r, j - 1]), f"boundary token at ({r},{j}) was masked out")
                flat_pos += 1

    def test_masked_mean_loss_ignores_fabricated_pad_positions(self):
        # The actual failure mode, end to end: a naive mean over ALL target positions (including fabricated
        # pad) silently absorbs whatever the pad positions' loss happens to be; only the mask-weighted mean
        # reflects just the real supervision.
        doc = np.array([11, 22])
        packed = pack_documents([doc], [0], block=4, pad_id=0)
        per_position_loss = np.array([[1.0, 5.0, 5.0, 5.0]])  # pretend pad positions carry huge loss
        naive_mean = per_position_loss.mean()
        masked_mean = per_position_loss[packed.loss_mask].mean()
        self.assertEqual(masked_mean, 1.0)  # only the single real target position (loss=1.0) counts
        self.assertNotEqual(naive_mean, masked_mean)

    def test_epoch_batches_yields_mask_aligned_with_targets(self):
        # 17 documents of 3 tokens each = 51 tokens; unit = block + 1 = 5, and 51 % 5 == 1, so this
        # deterministically produces one padded remainder row (unlike a random doc-size draw, which could
        # divide evenly and mask nothing).
        documents = [np.arange(i, i + 3) for i in range(17)]
        corpus = StreamingCorpus(documents, rank=0, world_size=1, block=4, batch_size=3, seed=1)
        saw_any_false = False
        for ctx, tgt, mask in corpus.epoch_batches(epoch=0):
            self.assertEqual(ctx.shape, tgt.shape)
            self.assertEqual(mask.shape, tgt.shape)
            self.assertEqual(mask.dtype, np.dtype(bool))
            saw_any_false = saw_any_false or bool(np.any(~mask))
        self.assertTrue(saw_any_false)  # this corpus/block combination does produce a padded remainder row


class TokenValidationTestCase(unittest.TestCase):
    """MXR-080-0062: token ids entering pack_documents/StreamingCorpus are validated -- finite,
    exact-integer, in a lossless int64 range -- instead of silently cast, and stay int64 (never downcast to
    float32) all the way through epoch_batches."""

    def test_fractional_document_tokens_are_rejected(self):
        doc = np.array([1.2, 2.8, 3.9])
        with self.assertRaises(ValueError):
            pack_documents([doc], [0], block=4)

    def test_nan_and_inf_document_tokens_are_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            doc = np.array([1.0, 2.0, bad])
            with self.assertRaises(ValueError):
                pack_documents([doc], [0], block=4)

    def test_non_1d_document_is_rejected_not_silently_flattened(self):
        doc = np.arange(6).reshape(2, 3)
        with self.assertRaises(ValueError):
            pack_documents([doc], [0], block=4)

    def test_out_of_int64_range_document_tokens_are_rejected(self):
        doc = np.array([1.0, 2.0**64])
        with self.assertRaises(ValueError):
            pack_documents([doc], [0], block=4)

    def test_exact_integer_valued_floats_are_accepted(self):
        # 3.0 is an exact integer even though it happens to be stored as a float -- must NOT be rejected.
        doc = np.array([3.0, 4.0, 5.0])
        packed = pack_documents([doc], [0], block=2)
        self.assertEqual(packed.rows.dtype, np.int64)
        np.testing.assert_array_equal(packed.rows[0][:3], [3, 4, 5])

    def test_fractional_pad_id_is_rejected(self):
        doc = np.array([1, 2])
        with self.assertRaises(ValueError):
            pack_documents([doc], [0], block=4, pad_id=0.5)

    def test_fractional_boundary_id_is_rejected(self):
        docs = [np.array([1, 2]), np.array([3, 4])]
        with self.assertRaises(ValueError):
            pack_documents(docs, [0, 1], block=4, boundary_id=1.5)

    def test_epoch_batches_context_and_targets_stay_int64_not_float32(self):
        doc = np.array([11, 22])
        corpus = StreamingCorpus([doc], rank=0, world_size=1, block=4, batch_size=8, seed=0)
        for ctx, tgt, _mask in corpus.epoch_batches(epoch=0):
            self.assertEqual(ctx.dtype, np.int64)
            self.assertEqual(tgt.dtype, np.int64)

    def test_token_identity_preserved_above_2_pow_24(self):
        # A float32 context would collide 2**24 and 2**24+1 (both round to the same float32 value); int64
        # must not.
        big = 2**24
        doc = np.array([big, big + 1, big + 2, big + 3, big + 4])
        corpus = StreamingCorpus([doc], rank=0, world_size=1, block=4, batch_size=8, seed=0)
        ctx, _tgt, _mask = next(iter(corpus.epoch_batches(epoch=0)))
        np.testing.assert_array_equal(ctx[0], [big, big + 1, big + 2, big + 3])
        self.assertEqual(len(set(ctx[0].tolist())), 4)  # all four values distinguishable, no collision

    def test_context_and_target_arrays_do_not_alias_rows_or_each_other(self):
        # context/targets are column-shifted, overlapping slices of the same packed rows array -- epoch_batches
        # must copy, not view, or mutating one in place would corrupt the other.
        doc = np.arange(1, 10)
        corpus = StreamingCorpus([doc], rank=0, world_size=1, block=4, batch_size=8, seed=0)
        ctx, tgt, _mask = next(iter(corpus.epoch_batches(epoch=0)))
        tgt_before = tgt.copy()
        ctx[:] = -1
        np.testing.assert_array_equal(tgt, tgt_before)  # unaffected by mutating ctx in place


if __name__ == "__main__":
    unittest.main()
