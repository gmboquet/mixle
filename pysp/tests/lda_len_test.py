"""Tests for document-length estimation wired through the LDA model.

LDADistribution accepts a 'len_dist' over the total token count of a document. These tests
check that (1) supplying 'len_estimator' to LDAEstimator yields a fitted (non-Null) length
distribution whose contribution is added to the document log-density, (2) the default
(no len_estimator) behavior is unchanged from the same fitted topics and alpha without
an explicit length model, and (3) LDADistribution.estimator() propagates the length
model class.
"""

import unittest

import numpy as np

from pysp.inference import seq_estimate, seq_initialize
from pysp.stats import (
    CategoricalEstimator,
    LDADistribution,
    LDAEstimator,
    NullDistribution,
    NullEstimator,
    PoissonDistribution,
    PoissonEstimator,
    seq_encode,
)
from pysp.stats.latent.lda import seq_posterior
from pysp.stats.univariate.discrete.categorical import CategoricalDistribution
from pysp.utils.optsutil import count_by_value

DOCS = [
    ["a", "a", "b"],
    ["b", "c", "c", "c"],
    ["a", "d", "d"],
    ["c", "c", "d", "d", "d"],
    ["a", "b", "c", "d"],
    ["a", "a", "a", "b", "b", "c"],
    ["d", "d"],
    ["b", "b", "c", "d"],
]


def fit_lda(data, len_estimator=None, n_iter=10):
    """Fit a 2-topic LDA on data with a fixed seed; optionally with a length estimator."""
    topic_est = CategoricalEstimator(pseudo_count=0.001, suff_stat={w: 0.25 for w in "abcd"})
    est = LDAEstimator([topic_est] * 2, len_estimator=len_estimator, gamma_threshold=1.0e-8)

    enc_data = seq_encode(data, estimator=est)
    model = seq_initialize(enc_data, est, np.random.RandomState(1), p=1.0)
    for _ in range(n_iter):
        model = seq_estimate(enc_data, est, prev_estimate=model)

    return model


class LDALenTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.data = [sorted(count_by_value(u).items()) for u in DOCS]
        self.doc_lens = [sum(cnt for _, cnt in doc) for doc in self.data]

    def test_default_len_dist_is_null_and_scores_unchanged(self):
        model = fit_lda(self.data)

        self.assertIsInstance(model.len_dist, NullDistribution)

        enc = model.dist_to_encoder().seq_encode(self.data)
        ld = model.seq_log_density(enc)
        ref_model = LDADistribution(model.topics, model.alpha, gamma_threshold=model.gamma_threshold)
        np.testing.assert_allclose(ld, ref_model.seq_log_density(enc), rtol=1.0e-12, atol=0.0)

    def test_len_estimator_is_fitted_and_scored(self):
        model = fit_lda(self.data, len_estimator=PoissonEstimator())

        self.assertIsInstance(model.len_dist, PoissonDistribution)
        self.assertAlmostEqual(model.len_dist.lam, np.mean(self.doc_lens), places=12)

        enc = model.dist_to_encoder().seq_encode(self.data)
        ld = model.seq_log_density(enc)

        # The same model with a Null length distribution differs exactly by the length log-density.
        null_model = LDADistribution(model.topics, model.alpha, gamma_threshold=model.gamma_threshold)
        ld_null = null_model.seq_log_density(enc)

        expected = np.asarray([model.len_dist.log_density(n) for n in self.doc_lens])
        np.testing.assert_allclose(ld - ld_null, expected, rtol=1.0e-12, atol=1.0e-12)

        # log_density() agrees with the vectorized path for a single document.
        self.assertAlmostEqual(model.log_density(self.data[0]), ld[0], places=12)

    def test_estimator_round_trip_preserves_length_model(self):
        model = fit_lda(self.data, len_estimator=PoissonEstimator())

        est = model.estimator()
        self.assertIsInstance(est.len_estimator, PoissonEstimator)

        enc_data = seq_encode(self.data, estimator=est)
        refit = seq_estimate(enc_data, est, prev_estimate=model)
        self.assertIsInstance(refit.len_dist, PoissonDistribution)

        # Null length models round trip to NullEstimator/NullDistribution.
        null_model = fit_lda(self.data)
        null_est = null_model.estimator()
        self.assertIsInstance(null_est.len_estimator, NullEstimator)

        enc_data = seq_encode(self.data, estimator=null_est)
        null_refit = seq_estimate(enc_data, null_est, prev_estimate=null_model)
        self.assertIsInstance(null_refit.len_dist, NullDistribution)


class LDASufficientStatisticWeightingTestCase(unittest.TestCase):
    def test_seq_update_uses_weighted_expected_counts_once(self):
        topics = [
            CategoricalDistribution({"a": 0.7, "b": 0.2, "c": 0.1}),
            CategoricalDistribution({"a": 0.1, "b": 0.3, "c": 0.6}),
        ]
        model = LDADistribution(topics, np.asarray([1.4, 0.8]), gamma_threshold=1.0e-10)
        est = LDAEstimator([CategoricalEstimator(), CategoricalEstimator()], gamma_threshold=1.0e-10)

        docs = [
            [("a", 3.0), ("b", 1.0)],
            [("b", 2.0), ("c", 4.0)],
            [("a", 1.0), ("c", 2.0)],
        ]
        weights = np.asarray([1.0, 0.25, 2.0])
        flat_words = [word for doc in docs for word, _ in doc]

        enc = model.dist_to_encoder().seq_encode(docs)
        acc = est.accumulator_factory().make()
        acc.seq_update(enc, weights, model)

        responsibilities, _, _ = seq_posterior(model, enc)
        expected = responsibilities * np.reshape(weights[enc[1]], (-1, 1))
        _, _, _, topic_counts, topic_suff_stats, _ = acc.value()

        np.testing.assert_allclose(topic_counts, expected.sum(axis=0), rtol=1.0e-10, atol=1.0e-10)

        for topic_idx, suff_stat in enumerate(topic_suff_stats):
            for word in sorted(set(flat_words)):
                expected_word_count = sum(expected[i, topic_idx] for i, w in enumerate(flat_words) if w == word)
                self.assertAlmostEqual(suff_stat.get(word, 0.0), expected_word_count, places=10)
            self.assertAlmostEqual(sum(suff_stat.values()), topic_counts[topic_idx], places=10)


if __name__ == "__main__":
    unittest.main()
