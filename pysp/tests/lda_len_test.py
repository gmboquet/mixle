"""Tests for document-length estimation wired through the LDA model.

LDADistribution accepts a 'len_dist' over the total token count of a document. These tests
check that (1) supplying 'len_estimator' to LDAEstimator yields a fitted (non-Null) length
distribution whose contribution is added to the document log-density, (2) the default
(no len_estimator) behavior is unchanged from before length support was added (scores are
compared against hard-coded pre-change values), and (3) LDADistribution.estimator()
propagates the length model class.
"""
import unittest

import numpy as np

from pysp.stats import (LDADistribution, LDAEstimator, CategoricalEstimator, PoissonEstimator,
                        PoissonDistribution, NullDistribution, NullEstimator, seq_encode,
                        seq_initialize, seq_estimate)
from pysp.utils.optsutil import count_by_value

DOCS = [
    ['a', 'a', 'b'],
    ['b', 'c', 'c', 'c'],
    ['a', 'd', 'd'],
    ['c', 'c', 'd', 'd', 'd'],
    ['a', 'b', 'c', 'd'],
    ['a', 'a', 'a', 'b', 'b', 'c'],
    ['d', 'd'],
    ['b', 'b', 'c', 'd'],
]

# Per-document seq_log_density values for the fit below with len_estimator unset, captured
# from the implementation before length estimation was wired through LDA (regression bar).
PRE_CHANGE_LD = [
    -3.7686329374451484,
    -5.9878169518559865,
    -4.109357682600127,
    -5.08946881464783,
    -6.108322841068291,
    -8.142439860250882,
    -2.1682033134303764,
    -6.425785093555406,
]


def fit_lda(data, len_estimator=None, n_iter=10):
    """Fit a 2-topic LDA on data with a fixed seed; optionally with a length estimator."""
    topic_est = CategoricalEstimator(pseudo_count=0.001, suff_stat={w: 0.25 for w in 'abcd'})
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
        np.testing.assert_allclose(ld, PRE_CHANGE_LD, rtol=1.0e-12, atol=0.0)

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


if __name__ == '__main__':
    unittest.main()
