"""Regression coverage for the lookback-HMM analog of campaign four's T4-03 (hidden_markov.py's
all-empty-corpus crash).

D-0203 recorded that ``lookback_hidden_markov_model.py`` "has the same all-empty-corpus defect
class as the now-fixed hidden_markov.py, found by an agent while working, not by a tester;
recorded, not fixed, in this wave." That record undersold the scope: hidden_markov.py's crash was
in scoring (seq_log_density indexing an empty band array), and its accumulator's seq_initialize
already tolerated an all-empty corpus without a fix. The lookback accumulator's seq_initialize
does NOT -- ``prev_mask[tz[1:] - 1] = False`` indexes an empty ``prev_mask`` with an all -1 array
when every sequence in the batch has length 0 (tot_cnt == 0), raising
``IndexError: index -1 is out of bounds for axis 0 with size 0`` from a file the caller never
opened, before a lookback HMM ever reaches scoring. This is a distinct crash site from the one
already fixed in hidden_markov.py, not a re-run of the same repair.

The fix guards the assignment with ``if tot_cnt > 0`` -- an all-empty batch's prev_mask is already
all-True (correct: no positions, so nothing is "not the last of its sequence" in a meaningful way)
and there is no last position for any sequence to name. lag > 0 is unaffected and unchanged: it
already raises a clear, designed ``ValueError("lookback-HMM observations must contain at least lag
values.")`` at encode time for an empty sequence, which is a documented guard, not this defect.
"""

from __future__ import annotations

import unittest

import numpy as np
from numpy.random import RandomState

import mixle.stats.latent.lookback_hidden_markov_model as mod
from mixle.inference import seq_estimate, seq_initialize
from mixle.stats import seq_encode, seq_log_density_sum
from mixle.stats.combinator.null_dist import NullEstimator
from mixle.stats.combinator.sequence import SequenceEstimator
from mixle.stats.univariate.discrete.categorical import CategoricalEstimator
from mixle.stats.univariate.discrete.integer_categorical import IntegerCategoricalEstimator


def _lag0_estimator():
    topic_est = SequenceEstimator(
        IntegerCategoricalEstimator(min_val=0, max_val=2, pseudo_count=0.1),
        len_estimator=CategoricalEstimator(pseudo_count=0.1),
    )
    return mod.LookbackHiddenMarkovModelEstimator(
        [topic_est] * 2,
        lag=0,
        init_estimators=[NullEstimator()] * 2,
        len_estimator=CategoricalEstimator(pseudo_count=0.1),
        pseudo_count=(1.0, 1.0),
    )


class LookbackHmmEmptyCorpusTest(unittest.TestCase):
    def test_all_empty_corpus_fits_without_raising(self):
        est = _lag0_estimator()
        data = [[], [], []]
        enc_data = seq_encode(data, estimator=est)
        init_model = seq_initialize(enc_data, est, RandomState(1), p=1.0)
        model = seq_estimate(enc_data, est, init_model)
        self.assertIsInstance(model, mod.LookbackHiddenMarkovModelDistribution)

    def test_all_empty_corpus_scores_zero_length_term_only(self):
        est = _lag0_estimator()
        data = [[], [], []]
        enc_data = seq_encode(data, estimator=est)
        init_model = seq_initialize(enc_data, est, RandomState(1), p=1.0)
        model = seq_estimate(enc_data, est, init_model)
        count, total_ll = seq_log_density_sum(enc_data, model)
        self.assertEqual(count, 3)
        # Every observed sequence had length 0, so the fitted length distribution places all mass
        # on length 0 and each sequence's log-density is log(1.0) == 0.0 (no emissions to score).
        self.assertAlmostEqual(total_ll, 0.0, places=9)

    def test_a_single_empty_sequence_fits_without_raising(self):
        est = _lag0_estimator()
        enc_data = seq_encode([[]], estimator=est)
        init_model = seq_initialize(enc_data, est, RandomState(3), p=1.0)
        model = seq_estimate(enc_data, est, init_model)
        self.assertIsInstance(model, mod.LookbackHiddenMarkovModelDistribution)

    def test_a_batch_mixing_empty_and_nonempty_sequences_fits_and_scores_finite(self):
        est = _lag0_estimator()
        data = [[], [0, 1, 2], [], [1, 1], []]
        enc_data = seq_encode(data, estimator=est)
        init_model = seq_initialize(enc_data, est, RandomState(2), p=1.0)
        model = seq_estimate(enc_data, est, init_model)
        count, total_ll = seq_log_density_sum(enc_data, model)
        self.assertEqual(count, 5)
        self.assertTrue(np.isfinite(total_ll))

    def test_a_nonempty_batch_still_fits_the_same_as_before_the_guard(self):
        # Regression guard: the tot_cnt > 0 branch must still run exactly as before for ordinary
        # data -- this is the fix's own risk (a wrongly-scoped guard could silently skip real work).
        est = _lag0_estimator()
        data = [[0, 1, 2], [1, 1], [2, 0]]
        enc_data = seq_encode(data, estimator=est)
        init_model = seq_initialize(enc_data, est, RandomState(4), p=1.0)
        model = seq_estimate(enc_data, est, init_model)
        count, total_ll = seq_log_density_sum(enc_data, model)
        self.assertEqual(count, 3)
        self.assertTrue(np.isfinite(total_ll))
        self.assertLess(total_ll, 0.0)

    def test_lag_greater_than_zero_still_raises_its_own_designed_guard_on_empty_sequences(self):
        # Not this defect: lag > 0 requires at least `lag` values per sequence by design, and an
        # empty sequence cannot satisfy that -- seq_encode raises a clear ValueError, which this fix
        # must not paper over.
        topic_est = SequenceEstimator(
            IntegerCategoricalEstimator(min_val=0, max_val=2, pseudo_count=0.1),
            len_estimator=CategoricalEstimator(pseudo_count=0.1),
        )
        est = mod.LookbackHiddenMarkovModelEstimator(
            [topic_est] * 2,
            lag=2,
            init_estimators=[IntegerCategoricalEstimator(min_val=0, max_val=2, pseudo_count=0.1)] * 2,
            len_estimator=CategoricalEstimator(pseudo_count=0.1),
            pseudo_count=(1.0, 1.0),
        )
        with self.assertRaises(ValueError):
            seq_encode([[], [0, 1, 2]], estimator=est)


class LookbackHmmZeroSequenceCorpusTest(unittest.TestCase):
    """A corpus of ZERO sequences (data=[]) is a different degenerate shape than the all-empty-
    sequences corpus above (data=[[],[],[]], 3 sequences each of length 0) -- found by an agent
    while fixing campaign-six T4-02 (a sibling defect in the shared mixle.stats.compute.sequence
    utility). seq_initialize already tolerated data=[]; seq_update did not: it computed
    ``max_len = sz.max()`` and never read it (dead since introduction), and ``sz.max()`` raises
    ``ValueError: zero-size array to reduction operation maximum which has no identity`` when
    ``sz`` (one entry per sequence) is itself empty. The fix removes the dead computation, in both
    seq_update and the identical pattern in seq_posterior -- a true no-op for every other input.
    """

    def test_zero_sequence_corpus_fits_and_scores_without_raising(self):
        est = _lag0_estimator()
        data = []
        enc_data = seq_encode(data, estimator=est)
        init_model = seq_initialize(enc_data, est, RandomState(1), p=1.0)
        model = seq_estimate(enc_data, est, init_model)
        self.assertIsInstance(model, mod.LookbackHiddenMarkovModelDistribution)
        count, total_ll = seq_log_density_sum(enc_data, model)
        self.assertEqual(count, 0)
        self.assertEqual(total_ll, 0.0)

    def test_zero_sequence_corpus_seq_posterior_returns_no_sequences(self):
        est = _lag0_estimator()
        enc_data = seq_encode([], estimator=est)
        init_model = seq_initialize(enc_data, est, RandomState(1), p=1.0)
        model = seq_estimate(enc_data, est, init_model)
        posteriors = model.seq_posterior(enc_data[0][1])
        self.assertEqual(posteriors, [])


if __name__ == "__main__":
    unittest.main()
