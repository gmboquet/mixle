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
        for (ctx_a, tgt_a), (ctx_b, tgt_b) in zip(run_a, run_b):
            np.testing.assert_array_equal(ctx_a, ctx_b)
            np.testing.assert_array_equal(tgt_a, tgt_b)

    def test_different_epoch_changes_batches_deterministically(self):
        documents = self._documents()
        corpus = StreamingCorpus(documents, rank=0, world_size=2, block=8, batch_size=4, seed=123)
        epoch0 = list(corpus.epoch_batches(epoch=0))
        epoch1 = list(corpus.epoch_batches(epoch=1))

        flat0 = np.concatenate([c.reshape(-1) for c, _ in epoch0]) if epoch0 else np.array([])
        flat1 = np.concatenate([c.reshape(-1) for c, _ in epoch1]) if epoch1 else np.array([])
        self.assertFalse(np.array_equal(flat0, flat1))

        # re-running epoch 0 alone still reproduces epoch 0 exactly
        epoch0_again = list(corpus.epoch_batches(epoch=0))
        for (ctx_a, tgt_a), (ctx_b, tgt_b) in zip(epoch0, epoch0_again):
            np.testing.assert_array_equal(ctx_a, ctx_b)
            np.testing.assert_array_equal(tgt_a, tgt_b)

    def test_global_batches_identical_across_full_rank_set_two_runs(self):
        """The determinism receipt at the corpus level: replaying the whole run (all ranks, one epoch)
        twice yields the identical set of per-rank batch sequences."""
        documents = self._documents()
        world_size = 4

        def run_all_ranks():
            out = []
            for rank in range(world_size):
                corpus = StreamingCorpus(documents, rank=rank, world_size=world_size, block=8, batch_size=4, seed=99)
                out.append([(c.copy(), t.copy()) for c, t in corpus.epoch_batches(epoch=2)])
            return out

        run_a = run_all_ranks()
        run_b = run_all_ranks()
        for rank in range(world_size):
            for (ctx_a, tgt_a), (ctx_b, tgt_b) in zip(run_a[rank], run_b[rank]):
                np.testing.assert_array_equal(ctx_a, ctx_b)
                np.testing.assert_array_equal(tgt_a, tgt_b)


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


if __name__ == "__main__":
    unittest.main()
