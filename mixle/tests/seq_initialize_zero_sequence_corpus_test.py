"""T4-02: seq_initialize() raised a raw IndexError on a truly-empty corpus (zero sequences, not zero-
length sequences). Its "nothing was selected" fallback always seeded from the FIRST chunk it saw,
regardless of whether that chunk had any rows; seq_encode([], ...) still returns a single chunk with
a declared row count of 0, so ``seed_mask = np.zeros(0); seed_mask[0] = 1.0`` indexed an empty array.

This is distinct from the D-0203/D-0204 all-empty-corpus fixes, which cover a corpus of individually
EMPTY SEQUENCES (data=[[], [], []] -- three rows, each of length zero); this covers a corpus of ZERO
sequences (data=[] -- zero rows). The high-level optimize() guards the zero-rows case for its own
`data` argument, but that guard was never reached by the documented low-level
seq_encode/seq_initialize/seq_estimate pipeline this repo's own tests use directly.
"""

from __future__ import annotations

import unittest

from numpy.random import RandomState

import mixle.stats as st
from mixle.stats import seq_encode
from mixle.stats.combinator.null_dist import NullEstimator
from mixle.stats.combinator.sequence import SequenceEstimator
from mixle.stats.compute.sequence import seq_initialize
from mixle.stats.latent.hidden_markov import HiddenMarkovEstimator
from mixle.stats.latent.lookback_hidden_markov_model import LookbackHiddenMarkovModelEstimator
from mixle.stats.univariate.discrete.categorical import CategoricalEstimator
from mixle.stats.univariate.discrete.integer_categorical import IntegerCategoricalEstimator


def _hmm_estimator():
    return HiddenMarkovEstimator([st.GaussianEstimator(), st.GaussianEstimator()])


def _lookback_estimator():
    topic_est = SequenceEstimator(
        IntegerCategoricalEstimator(min_val=0, max_val=2, pseudo_count=0.1),
        len_estimator=CategoricalEstimator(pseudo_count=0.1),
    )
    return LookbackHiddenMarkovModelEstimator(
        [topic_est] * 2,
        lag=0,
        init_estimators=[NullEstimator()] * 2,
        len_estimator=CategoricalEstimator(pseudo_count=0.1),
        pseudo_count=(1.0, 1.0),
    )


class SeqInitializeZeroSequenceCorpusTest(unittest.TestCase):
    def test_hmm_zero_sequence_corpus_initializes_without_raising(self):
        est = _hmm_estimator()
        enc_data = seq_encode([], estimator=est)
        model = seq_initialize(enc_data, est, RandomState(1), p=1.0)
        self.assertEqual(model.n_states, 2)

    def test_lookback_hmm_zero_sequence_corpus_initializes_without_raising(self):
        est = _lookback_estimator()
        enc_data = seq_encode([], estimator=est)
        # Previously: IndexError: index 0 is out of bounds for axis 0 with size 0
        # (mixle/stats/compute/sequence.py, seq_initialize's nothing-selected fallback).
        seq_initialize(enc_data, est, RandomState(1), p=1.0)

    def test_zero_sequence_corpus_is_distinct_from_all_empty_sequences_corpus(self):
        # D-0203/D-0204 already cover data=[[], [], []] (3 rows, each length 0); confirm that case
        # still works too, so this test file's scope does not overlap silently with that fix.
        est = _hmm_estimator()
        enc_data = seq_encode([[], [], []], estimator=est)
        model = seq_initialize(enc_data, est, RandomState(1), p=1.0)
        self.assertEqual(model.n_states, 2)

    def test_ordinary_small_corpus_fallback_path_is_unchanged(self):
        # Regression guard: p=0.1 on this 2-sequence corpus selects nothing under this seed, which
        # exercises the SAME "nothing selected, seed from the first real row" fallback this fix
        # touched -- but with a first chunk that genuinely has rows (unlike the zero-sequence-corpus
        # tests above). The fix must not change this ordinary, non-degenerate outcome: confirmed by
        # running both the pre-fix and post-fix code on this exact call and diffing the repr, which
        # matched exactly. mu==2.0 (the first sequence's own mean) on both emission topics is that
        # pinned value, not a re-derivation -- it would move if the fallback ever picked a different
        # seed row.
        est = _hmm_estimator()
        data = [[1.0, 2.0, 3.0], [4.0, 5.0]]
        enc_data = seq_encode(data, estimator=est)
        model = seq_initialize(enc_data, est, RandomState(1), p=0.1)
        self.assertEqual(model.n_states, 2)
        means = [d.mu for d in model.topics]
        self.assertAlmostEqual(means[0], 2.0, places=9)
        self.assertAlmostEqual(means[1], 2.0, places=9)


if __name__ == "__main__":
    unittest.main()
