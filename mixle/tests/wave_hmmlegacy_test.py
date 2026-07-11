"""Tests for the modernized legacy modules mixle.stats.latent.labeled_lda.

Covers import, DataSequenceEncoder round-trips, scalar vs vectorized agreement, and short
seq_estimate smoke runs on tiny synthetic data with fixed seeds.
"""

import unittest

import numpy as np
from numpy.random import RandomState

from mixle.inference import seq_estimate, seq_initialize
from mixle.stats.latent.labeled_lda import (
    LabeledLDADataEncoder,
    LabeledLDADistribution,
    LabeledLDAEstimator,
    LabeledLDAEstimatorAccumulator,
    LabeledLDAEstimatorAccumulatorFactory,
)
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution, CategoricalEstimator


def make_llda_model():
    topics = [
        CategoricalDistribution({"w0": 0.4, "w1": 0.4, "w2": 0.1, "w3": 0.1}),
        CategoricalDistribution({"w0": 0.1, "w1": 0.1, "w2": 0.4, "w3": 0.4}),
    ]
    alphas = np.asarray([[2.0, 0.5], [0.5, 2.0]])
    return LabeledLDADistribution(topics, alphas)


def make_llda_data(n=30, seed=5):
    rng = RandomState(seed)
    vocab = ["w0", "w1", "w2", "w3"]
    pmats = [[0.4, 0.4, 0.1, 0.1], [0.1, 0.1, 0.4, 0.4]]
    data = []
    for i in range(n):
        label = i % 2
        words = rng.choice(4, size=6, p=pmats[label])
        cnts = {}
        for wd in words:
            cnts[vocab[wd]] = cnts.get(vocab[wd], 0) + 1
        doc = sorted(cnts.items())
        data.append((doc, [label]))
    return data


def make_llda_estimator():
    return LabeledLDAEstimator([CategoricalEstimator(), CategoricalEstimator()], num_alphas=2)


class LabeledLDATestCase(unittest.TestCase):
    def setUp(self):
        self.model = make_llda_model()
        self.data = make_llda_data(n=30, seed=5)

    def test_encoder_equality_and_str(self):
        enc1 = self.model.dist_to_encoder()
        enc2 = self.model.dist_to_encoder()
        self.assertIsInstance(enc1, LabeledLDADataEncoder)
        self.assertEqual(enc1, enc2)
        self.assertIn("LabeledLDADataEncoder", str(enc1))

        est = make_llda_estimator()
        acc = est.accumulator_factory().make()
        self.assertIsInstance(acc, LabeledLDAEstimatorAccumulator)
        self.assertEqual(acc.acc_to_encoder(), enc1)

    def test_encoding_round_trip(self):
        enc = self.model.dist_to_encoder().seq_encode(self.data)
        num_documents, idx, counts, gammas, enc_data, nbx, nbcnt, nbidx = enc

        self.assertEqual(num_documents, len(self.data))
        self.assertIsNone(gammas)
        self.assertEqual(np.sum(counts), 6 * len(self.data))
        np.testing.assert_array_equal(np.bincount(idx), [len(d[0]) for d in self.data])
        np.testing.assert_array_equal(nbcnt, [1] * len(self.data))
        np.testing.assert_array_equal(nbx, [i % 2 for i in range(len(self.data))])
        np.testing.assert_array_equal(nbidx, np.arange(len(self.data)))

        # legacy seq_encode on the distribution delegates to the encoder (and warns: it is deprecated)
        with self.assertWarns(DeprecationWarning):
            legacy_enc = self.model.seq_encode(self.data)
        np.testing.assert_array_equal(legacy_enc[1], idx)
        np.testing.assert_array_equal(legacy_enc[5], nbx)

    def test_seq_log_density_finite_and_matches_scalar(self):
        enc = self.model.dist_to_encoder().seq_encode(self.data)
        seq_ll = self.model.seq_log_density(enc)
        self.assertEqual(np.shape(seq_ll), (len(self.data),))
        self.assertTrue(np.all(np.isfinite(seq_ll)))

        for i in (0, 1, 7):
            self.assertAlmostEqual(self.model.log_density(self.data[i]), seq_ll[i], places=6)

    def test_update_matches_seq_update(self):
        est = make_llda_estimator()
        sub = self.data[:8]

        acc_a = est.accumulator_factory().make()
        for xx in sub:
            acc_a.update(xx, 1.0, self.model)

        acc_b = est.accumulator_factory().make()
        enc = self.model.dist_to_encoder().seq_encode(sub)
        acc_b.seq_update(enc, np.ones(len(sub)), self.model)

        pa_a, sol_a, dc_a, tc_a, topic_ss_a = acc_a.value()
        pa_b, sol_b, dc_b, tc_b, topic_ss_b = acc_b.value()

        self.assertTrue(np.allclose(pa_a, pa_b))
        self.assertTrue(np.allclose(sol_a, sol_b, rtol=1.0e-6, atol=1.0e-8))
        self.assertTrue(np.allclose(dc_a, dc_b, rtol=1.0e-6, atol=1.0e-8))
        self.assertTrue(np.allclose(tc_a, tc_b, rtol=1.0e-6, atol=1.0e-8))

        for ss_a, ss_b in zip(topic_ss_a, topic_ss_b):
            self.assertEqual(set(ss_a.keys()), set(ss_b.keys()))
            for k in ss_a:
                self.assertAlmostEqual(ss_a[k], ss_b[k], places=6)

    def test_seq_estimate_smoke(self):
        est = make_llda_estimator()
        encoder = est.accumulator_factory().make().acc_to_encoder()
        enc_data = [(len(self.data), encoder.seq_encode(self.data))]

        model = seq_initialize(enc_data, est, RandomState(11), p=1.0)
        self.assertIsInstance(model, LabeledLDADistribution)

        for _ in range(3):
            model = seq_estimate(enc_data, est, model)

        self.assertIsInstance(model, LabeledLDADistribution)
        self.assertEqual(np.shape(model.alphas), (2, 2))
        self.assertTrue(np.all(np.isfinite(model.alphas)))
        self.assertTrue(np.all(model.alphas > 0))

        ll = model.seq_log_density(encoder.seq_encode(self.data))
        self.assertTrue(np.all(np.isfinite(ll)))

    def test_accumulator_factory_alias(self):
        est = make_llda_estimator()
        f1 = est.accumulator_factory()
        with self.assertWarns(DeprecationWarning):  # camelCase alias is deprecated
            f2 = est.accumulatorFactory()
        self.assertIsInstance(f1, LabeledLDAEstimatorAccumulatorFactory)
        self.assertIsInstance(f2, LabeledLDAEstimatorAccumulatorFactory)


if __name__ == "__main__":
    unittest.main()
