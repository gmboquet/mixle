"""Tests for the SparseMarkovAssociation model.

Covers the seq_initialize fast-path fix (uniform allocation of initial
transition mass instead of reading the all-zero trans_count), agreement
between the fast and low-memory initialization paths, and scalar vs
vectorized update parity.
"""

import unittest
import warnings

import numpy as np

from mixle.stats.combinator.null_dist import NullDataEncoder
from mixle.stats.sequences.sparse_markov_transform import (
    SparseMarkovAssociationAccumulator,
    SparseMarkovAssociationDataEncoder,
    SparseMarkovAssociationEstimator,
)

NUM_VALS = 6

DATA = [
    ([(0, 2.0), (1, 1.0)], [(2, 1.0), (3, 2.0)]),
    ([(1, 1.0)], [(4, 1.0)]),
    ([(2, 2.0), (0, 1.0)], [(5, 1.0), (2, 1.0)]),
]

WEIGHTS = np.asarray([1.0, 0.5, 2.0])


def observed_pairs(data):
    pairs = set()
    for s1, s2 in data:
        for u, _ in s1:
            for v, _ in s2:
                pairs.add((u, v))
    return pairs


def weighted_target_total(data, weights):
    return sum(w * sum(c for _, c in s2) for (s1, s2), w in zip(data, weights))


def make_accumulator(low_memory):
    return SparseMarkovAssociationAccumulator(NUM_VALS, low_memory=low_memory)


def encode(data, low_memory):
    encoder = SparseMarkovAssociationDataEncoder(len_encoder=NullDataEncoder(), low_memory=low_memory)
    return encoder.seq_encode(data)


class SparseMarkovAssociationSeqInitializeTestCase(unittest.TestCase):
    def test_seq_initialize_fast_path_is_finite_and_warning_free(self):
        acc = make_accumulator(low_memory=False)
        enc = encode(DATA, low_memory=False)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            acc.seq_initialize(enc, WEIGHTS.copy(), np.random.RandomState(1))

        trans = acc.trans_count.toarray()

        self.assertTrue(np.all(np.isfinite(trans)))
        self.assertTrue(np.all(trans >= 0.0))
        self.assertAlmostEqual(trans.sum(), weighted_target_total(DATA, WEIGHTS))

        self.assertTrue(np.all(np.isfinite(acc.init_count)))
        self.assertTrue(np.all(acc.init_count >= 0.0))

    def test_fast_and_low_memory_paths_agree(self):
        acc_fast = make_accumulator(low_memory=False)
        acc_slow = make_accumulator(low_memory=True)

        acc_fast.seq_initialize(encode(DATA, low_memory=False), WEIGHTS.copy(), np.random.RandomState(1))
        acc_slow.seq_initialize(encode(DATA, low_memory=True), WEIGHTS.copy(), np.random.RandomState(1))

        trans_fast = acc_fast.trans_count.toarray()
        trans_slow = acc_slow.trans_count.toarray()

        self.assertAlmostEqual(trans_fast.sum(), trans_slow.sum())

        pairs = observed_pairs(DATA)
        support_fast = set(zip(*np.nonzero(trans_fast)))
        support_slow = set(zip(*np.nonzero(trans_slow)))

        self.assertEqual(support_fast, pairs)
        self.assertEqual(support_slow, pairs)

        est = SparseMarkovAssociationEstimator(num_vals=NUM_VALS, alpha=0.3, low_memory=False)

        for acc in (acc_fast, acc_slow):
            model = est.estimate(None, acc.value())
            self.assertTrue(np.all(np.isfinite(model.init_prob_vec)))
            self.assertTrue(np.all(np.isfinite(model.cond_prob_mat.toarray())))

    def test_scalar_update_matches_seq_update(self):
        est = SparseMarkovAssociationEstimator(num_vals=NUM_VALS, alpha=0.3, low_memory=False)

        acc_init = make_accumulator(low_memory=False)
        acc_init.seq_initialize(encode(DATA, low_memory=False), WEIGHTS.copy(), np.random.RandomState(1))
        model = est.estimate(None, acc_init.value())

        acc_scalar = make_accumulator(low_memory=False)
        for xx, ww in zip(DATA, WEIGHTS):
            acc_scalar.update(xx, ww, model)

        acc_seq = make_accumulator(low_memory=False)
        acc_seq.seq_update(encode(DATA, low_memory=False), WEIGHTS.copy(), model)

        self.assertTrue(np.allclose(acc_scalar.init_count, acc_seq.init_count))
        self.assertTrue(np.allclose(acc_scalar.trans_count.toarray(), acc_seq.trans_count.toarray()))


if __name__ == "__main__":
    unittest.main()
