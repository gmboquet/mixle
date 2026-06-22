"""Tests for CategoricalAccumulator.seq_update() across multiple batches.

Accumulating encoded batches whose category values were not seen in a
previous batch must add the new keys to count_map rather than raising
KeyError. This is the path taken by incremental update() implementations
that delegate to seq_update batch-by-batch (e.g. lookback HMM length
accumulators).
"""

import unittest

import numpy as np

from pysp.stats.base.categorical import CategoricalDataEncoder, CategoricalEstimator


class CategoricalSeqUpdateTestCase(unittest.TestCase):
    def test_seq_update_disjoint_batches(self):
        enc = CategoricalDataEncoder()
        acc = CategoricalEstimator().accumulator_factory().make()
        acc.seq_update(enc.seq_encode([5]), np.ones(1), None)
        acc.seq_update(enc.seq_encode([6]), np.ones(1), None)
        self.assertEqual(acc.count_map, {5: 1.0, 6: 1.0})

    def test_seq_update_overlapping_batches(self):
        enc = CategoricalDataEncoder()
        acc = CategoricalEstimator().accumulator_factory().make()
        acc.seq_update(enc.seq_encode(["a", "b", "a"]), np.ones(3), None)
        acc.seq_update(enc.seq_encode(["b", "c"]), np.asarray([2.0, 0.5]), None)
        self.assertEqual(acc.count_map, {"a": 2.0, "b": 3.0, "c": 0.5})

    def test_seq_update_matches_update(self):
        rng = np.random.RandomState(3)
        batch1 = [int(v) for v in rng.randint(0, 4, size=10)]
        batch2 = [int(v) for v in rng.randint(2, 6, size=10)]
        enc = CategoricalDataEncoder()

        seq_acc = CategoricalEstimator().accumulator_factory().make()
        seq_acc.seq_update(enc.seq_encode(batch1), np.ones(len(batch1)), None)
        seq_acc.seq_update(enc.seq_encode(batch2), np.ones(len(batch2)), None)

        one_acc = CategoricalEstimator().accumulator_factory().make()
        for v in batch1 + batch2:
            one_acc.update(v, 1.0, None)

        self.assertEqual(set(seq_acc.count_map), set(one_acc.count_map))
        for k, v in one_acc.count_map.items():
            self.assertAlmostEqual(seq_acc.count_map[k], v, places=12)


if __name__ == "__main__":
    unittest.main()
