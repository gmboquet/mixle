"""Tests for the modernized legacy modules pysp.stats.hidden_markov_ind_pi and pysp.stats.llda.

Covers import, DataSequenceEncoder round-trips, scalar vs vectorized agreement, and short
seq_estimate smoke runs on tiny synthetic data with fixed seeds.
"""

import unittest

import numpy as np
from numpy.random import RandomState

from pysp.stats import seq_estimate, seq_initialize
from pysp.stats.categorical import CategoricalDistribution, CategoricalEstimator
from pysp.stats.hidden_markov_ind_pi import (
    IndPiHiddenMarkovDataEncoder,
    IndPiHiddenMarkovEstimator,
    IndPiHiddenMarkovEstimatorAccumulator,
    IndPiHiddenMarkovEstimatorAccumulatorFactory,
    IndPiHiddenMarkovModelDistribution,
    IndPiHiddenMarkovSampler,
)
from pysp.stats.llda import (
    LLDADataEncoder,
    LLDADistribution,
    LLDAEstimator,
    LLDAEstimatorAccumulator,
    LLDAEstimatorAccumulatorFactory,
)


def make_ind_pi_dist(use_numba=True, n_rows=2):
    topics = [
        CategoricalDistribution({"a": 0.7, "b": 0.2, "c": 0.1}),
        CategoricalDistribution({"a": 0.1, "b": 0.2, "c": 0.7}),
    ]
    if n_rows == 2:
        w = [[0.8, 0.2], [0.3, 0.7]]
    else:
        w = [[0.55, 0.45]] * n_rows
    transitions = [[0.9, 0.1], [0.2, 0.8]]
    len_dist = CategoricalDistribution({3: 0.5, 4: 0.5})
    return IndPiHiddenMarkovModelDistribution(topics, w, transitions, None, len_dist=len_dist, use_numba=use_numba)


def make_ind_pi_estimator():
    return IndPiHiddenMarkovEstimator(
        [CategoricalEstimator(), CategoricalEstimator()], len_estimator=CategoricalEstimator(), pseudo_count=(1.0, 1.0)
    )


class IndPiHiddenMarkovTestCase(unittest.TestCase):
    def setUp(self):
        self.dist = make_ind_pi_dist()
        self.data = self.dist.sampler(seed=1).sample(30)

    def test_sampler_output(self):
        sampler = self.dist.sampler(seed=1)
        self.assertIsInstance(sampler, IndPiHiddenMarkovSampler)
        self.assertEqual(len(self.data), 30)
        for seq in self.data:
            self.assertIn(len(seq), (3, 4))
            for v in seq:
                self.assertIn(v, ("a", "b", "c"))

    def test_encoder_equality_and_str(self):
        enc1 = self.dist.dist_to_encoder()
        enc2 = self.dist.dist_to_encoder()
        self.assertIsInstance(enc1, IndPiHiddenMarkovDataEncoder)
        self.assertEqual(enc1, enc2)
        self.assertIn("IndPiHiddenMarkovDataEncoder", str(enc1))

        est = make_ind_pi_estimator()
        acc = est.accumulator_factory().make()
        self.assertIsInstance(acc, IndPiHiddenMarkovEstimatorAccumulator)
        self.assertEqual(acc.acc_to_encoder(), enc1)

    def test_numba_encoding_round_trip(self):
        enc = self.dist.dist_to_encoder().seq_encode(self.data)
        x0, x1 = enc
        self.assertIsNone(x0)
        (idx, sz, xs), len_enc = x1

        np.testing.assert_array_equal(sz, np.asarray([len(u) for u in self.data], dtype=np.int32))
        self.assertEqual(len(idx), sum(len(u) for u in self.data))
        np.testing.assert_array_equal(np.bincount(idx), sz)
        self.assertIsNotNone(len_enc)

        # legacy seq_encode on the distribution delegates to the encoder
        legacy_enc = self.dist.seq_encode(self.data)
        np.testing.assert_array_equal(legacy_enc[1][0][0], idx)
        np.testing.assert_array_equal(legacy_enc[1][0][1], sz)

    def test_seq_log_density_matches_scalar(self):
        # numba path with per-sequence rows averaged to logW
        enc = self.dist.dist_to_encoder().seq_encode(self.data)
        seq_ll = self.dist.seq_log_density(enc)
        scalar_ll = np.asarray([self.dist.log_density(u) for u in self.data])
        self.assertTrue(np.all(np.isfinite(seq_ll)))
        self.assertTrue(np.allclose(seq_ll, scalar_ll, rtol=1.0e-8, atol=1.0e-8))

        # non-numba (numpy) path requires one w row per sequence; use identical rows equal to the mean
        dist_np = make_ind_pi_dist(use_numba=False, n_rows=len(self.data))
        enc_np = dist_np.dist_to_encoder().seq_encode(self.data)
        self.assertIsNone(enc_np[1])
        seq_ll_np = dist_np.seq_log_density(enc_np)
        self.assertTrue(np.allclose(seq_ll_np, scalar_ll, rtol=1.0e-8, atol=1.0e-8))

    def test_scalar_update_runs(self):
        est = make_ind_pi_estimator()
        acc = est.accumulator_factory().make()
        acc.update(self.data[0], 1.0, self.dist)
        self.assertAlmostEqual(np.sum(acc.state_counts), float(len(self.data[0])), places=8)
        self.assertEqual(np.shape(acc.init_counts), (1, 2))

    def test_seq_estimate_smoke(self):
        est = make_ind_pi_estimator()
        encoder = est.accumulator_factory().make().acc_to_encoder()
        enc_data = [(len(self.data), encoder.seq_encode(self.data))]

        model = seq_initialize(enc_data, est, RandomState(7), p=1.0)
        self.assertIsInstance(model, IndPiHiddenMarkovModelDistribution)
        self.assertEqual(np.shape(model.w), (len(self.data), 2))

        for _ in range(3):
            model = seq_estimate(enc_data, est, model)

        self.assertIsInstance(model, IndPiHiddenMarkovModelDistribution)
        self.assertEqual(model.nStates, 2)
        self.assertEqual(np.shape(model.w), (len(self.data), 2))
        self.assertTrue(np.allclose(model.w.sum(axis=1), 1.0))
        self.assertEqual(np.shape(model.transitions), (2, 2))
        self.assertTrue(np.allclose(model.transitions.sum(axis=1), 1.0))

        ll = model.seq_log_density(encoder.seq_encode(self.data))
        self.assertTrue(np.all(np.isfinite(ll)))

    def test_accumulator_factory_alias(self):
        est = make_ind_pi_estimator()
        f1 = est.accumulator_factory()
        f2 = est.accumulatorFactory()
        self.assertIsInstance(f1, IndPiHiddenMarkovEstimatorAccumulatorFactory)
        self.assertIsInstance(f2, IndPiHiddenMarkovEstimatorAccumulatorFactory)


def make_llda_model():
    topics = [
        CategoricalDistribution({"w0": 0.4, "w1": 0.4, "w2": 0.1, "w3": 0.1}),
        CategoricalDistribution({"w0": 0.1, "w1": 0.1, "w2": 0.4, "w3": 0.4}),
    ]
    alphas = np.asarray([[2.0, 0.5], [0.5, 2.0]])
    return LLDADistribution(topics, alphas)


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
    return LLDAEstimator([CategoricalEstimator(), CategoricalEstimator()], num_alphas=2)


class LLDATestCase(unittest.TestCase):
    def setUp(self):
        self.model = make_llda_model()
        self.data = make_llda_data(n=30, seed=5)

    def test_encoder_equality_and_str(self):
        enc1 = self.model.dist_to_encoder()
        enc2 = self.model.dist_to_encoder()
        self.assertIsInstance(enc1, LLDADataEncoder)
        self.assertEqual(enc1, enc2)
        self.assertIn("LLDADataEncoder", str(enc1))

        est = make_llda_estimator()
        acc = est.accumulator_factory().make()
        self.assertIsInstance(acc, LLDAEstimatorAccumulator)
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

        # legacy seq_encode on the distribution delegates to the encoder
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
        self.assertIsInstance(model, LLDADistribution)

        for _ in range(3):
            model = seq_estimate(enc_data, est, model)

        self.assertIsInstance(model, LLDADistribution)
        self.assertEqual(np.shape(model.alphas), (2, 2))
        self.assertTrue(np.all(np.isfinite(model.alphas)))
        self.assertTrue(np.all(model.alphas > 0))

        ll = model.seq_log_density(encoder.seq_encode(self.data))
        self.assertTrue(np.all(np.isfinite(ll)))

    def test_accumulator_factory_alias(self):
        est = make_llda_estimator()
        f1 = est.accumulator_factory()
        f2 = est.accumulatorFactory()
        self.assertIsInstance(f1, LLDAEstimatorAccumulatorFactory)
        self.assertIsInstance(f2, LLDAEstimatorAccumulatorFactory)


if __name__ == "__main__":
    unittest.main()
