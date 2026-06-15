"""Tests for the coupled multi-label alpha update in pysp.stats.latent.llda.

Covers the single-label fast path (agreement with the per-row fixed-point update recomputed from the
same per-label statistics), the coupled multi-label update (objective increase, stationarity in beta
space, positivity), the label-set sufficient-statistic plumbing (combine/value/from_value), and an
estimate -> distribution -> seq_posterior round trip.
"""

import unittest

import numpy as np
from numpy.random import RandomState
from scipy.special import digamma

from pysp.stats.latent.llda import (
    LLDADistribution,
    LLDAEstimator,
    LLDALabelSetStats,
    coupled_alpha_objective,
    seq_posterior,
    update_alpha,
    update_alpha_coupled,
)
from pysp.stats.leaf.categorical import CategoricalDistribution, CategoricalEstimator

VOCAB = ["w0", "w1", "w2", "w3"]
PMATS = [[0.4, 0.4, 0.1, 0.1], [0.1, 0.1, 0.4, 0.4]]


def make_model(alphas):
    topics = [
        CategoricalDistribution({"w0": 0.4, "w1": 0.4, "w2": 0.1, "w3": 0.1}),
        CategoricalDistribution({"w0": 0.1, "w1": 0.1, "w2": 0.4, "w3": 0.4}),
    ]
    return LLDADistribution(topics, np.asarray(alphas, dtype=float))


def make_data(label_sets, n, seed):
    rng = RandomState(seed)
    data = []
    for i in range(n):
        labels = list(label_sets[i % len(label_sets)])
        p = np.asarray(PMATS[labels[0] % 2]) * 0.7 + np.asarray(PMATS[(labels[-1] + 1) % 2]) * 0.3
        words = rng.choice(4, size=8, p=p / p.sum())
        cnts = {}
        for wd in words:
            cnts[VOCAB[wd]] = cnts.get(VOCAB[wd], 0) + 1
        data.append((sorted(cnts.items()), labels))
    return data


def make_estimator(num_alphas, pseudo_count=None):
    return LLDAEstimator(
        [CategoricalEstimator(), CategoricalEstimator()], num_alphas=num_alphas, pseudo_count=pseudo_count
    )


def accumulate(model, est, data):
    enc = model.dist_to_encoder().seq_encode(data)
    acc = est.accumulator_factory().make()
    acc.seq_update(enc, np.ones(len(data)), model)
    return enc, acc.value()


class LLDASingleLabelAlphaTestCase(unittest.TestCase):
    def setUp(self):
        self.model = make_model([[2.0, 0.5], [0.5, 2.0]])
        self.data = make_data([(0,), (1,)], n=30, seed=5)
        self.est = make_estimator(num_alphas=2)
        self.enc, self.ss = accumulate(self.model, self.est, self.data)

    def test_matches_independent_row_update(self):
        # Recompute the pre-fix per-label statistics from the same posterior quantities and apply the
        # old independent per-row fixed-point update.
        _, final_gammas, _, _ = seq_posterior(self.model, self.enc)
        mlpf = digamma(final_gammas) - digamma(np.sum(final_gammas, axis=1, keepdims=True))

        labels = np.asarray([d[1][0] for d in self.data], dtype=int)
        sum_of_logs = np.zeros((2, 2))
        doc_counts = np.zeros(2)
        for i in range(2):
            sum_of_logs[:, i] = np.bincount(labels, weights=mlpf[:, i], minlength=2)
        doc_counts = np.bincount(labels, weights=np.ones(len(labels)), minlength=2)

        old_alpha = update_alpha(
            self.model.alphas, sum_of_logs / np.reshape(doc_counts, (-1, 1)), self.est.alpha_threshold
        )

        fitted = self.est.estimate(None, self.ss)
        self.assertTrue(np.allclose(fitted.alphas, old_alpha, rtol=1.0e-8, atol=1.0e-10))

    def test_general_optimizer_agrees_on_single_label(self):
        # The coupled optimizer over singleton label sets must converge to the decoupled fixed point.
        label_sets, set_n, set_m = self.ss[1].arrays()
        self.assertTrue(all(len(u) == 1 for u in label_sets))

        mean_logs = set_m / np.reshape(set_n, (-1, 1))
        rows = np.asarray([u[0] for u in label_sets], dtype=int)

        a_fp = update_alpha(self.model.alphas[rows, :], mean_logs, 1.0e-10)
        a_gd = update_alpha_coupled(self.model.alphas, label_sets, set_n, mean_logs, 1.0e-10, max_its=20000)

        self.assertTrue(np.allclose(a_gd[rows, :], a_fp, rtol=1.0e-6, atol=1.0e-6))


class LLDAMultiLabelAlphaTestCase(unittest.TestCase):
    def setUp(self):
        self.model = make_model([[2.0, 0.5], [0.5, 2.0], [1.0, 1.0]])
        self.data = make_data([(0,), (0, 1), (1, 2), (2,), (0, 2), (1,)], n=48, seed=7)
        self.est = make_estimator(num_alphas=3)
        self.enc, self.ss = accumulate(self.model, self.est, self.data)
        self.fitted = self.est.estimate(None, self.ss)

        label_sets, set_n, set_m = self.ss[1].arrays()
        self.label_sets = label_sets
        self.set_n = set_n
        self.mean_logs = set_m / np.reshape(set_n, (-1, 1))

    def objective(self, alpha):
        return coupled_alpha_objective(alpha, self.label_sets, self.set_n, self.mean_logs)

    def test_label_set_grouping(self):
        self.assertEqual(self.label_sets, [(0,), (0, 1), (0, 2), (1,), (1, 2), (2,)])
        self.assertAlmostEqual(np.sum(self.set_n), float(len(self.data)), places=10)

    def test_objective_increases(self):
        f_warm = self.objective(self.model.alphas)
        f_new = self.objective(self.fitted.alphas)
        self.assertTrue(np.isfinite(f_new))
        self.assertGreater(f_new, f_warm)

    def test_stationary_in_beta_space(self):
        # Numerical gradient of F with respect to beta = log(alpha) at the returned alpha is ~0.
        alpha = self.fitted.alphas
        beta = np.log(alpha)
        f_scale = 1.0 + abs(self.objective(alpha))
        h = 1.0e-5

        g_num = np.zeros(beta.shape)
        for l in range(beta.shape[0]):
            for k in range(beta.shape[1]):
                bp = beta.copy()
                bm = beta.copy()
                bp[l, k] += h
                bm[l, k] -= h
                g_num[l, k] = (self.objective(np.exp(bp)) - self.objective(np.exp(bm))) / (2.0 * h)

        self.assertLess(np.max(np.abs(g_num)) / f_scale, 1.0e-4)

    def test_alpha_positive_and_finite(self):
        self.assertEqual(np.shape(self.fitted.alphas), (3, 2))
        self.assertTrue(np.all(np.isfinite(self.fitted.alphas)))
        self.assertTrue(np.all(self.fitted.alphas > 0))

    def test_pseudo_count_smoothing(self):
        est_pc = make_estimator(num_alphas=3, pseudo_count=(1.0, 0.5))
        fitted_pc = est_pc.estimate(None, self.ss)
        self.assertTrue(np.all(np.isfinite(fitted_pc.alphas)))
        self.assertTrue(np.all(fitted_pc.alphas > 0))

        # The smoothed coupled objective is also increased relative to the warm start.
        label_sets, set_n, set_m = self.ss[1].arrays()
        n_eff = set_n + 1.0
        mbar = (set_m + np.log(0.5)) / np.reshape(n_eff, (-1, 1))
        f_warm = coupled_alpha_objective(self.model.alphas, label_sets, n_eff, mbar)
        f_new = coupled_alpha_objective(fitted_pc.alphas, label_sets, n_eff, mbar)
        self.assertGreater(f_new, f_warm)

    def test_round_trip_posterior(self):
        post = self.fitted.seq_posterior(self.enc)
        self.assertEqual(np.shape(post), (len(self.data), 2))
        self.assertTrue(np.all(np.isfinite(post)))
        self.assertTrue(np.allclose(post.sum(axis=1), 1.0))

        ll = self.fitted.seq_log_density(self.enc)
        self.assertTrue(np.all(np.isfinite(ll)))

    def test_suff_stat_plumbing_roundtrip(self):
        # Split-update + combine must match a single batch update, and from_value must round trip.
        half = len(self.data) // 2
        enc_a = self.model.dist_to_encoder().seq_encode(self.data[:half])
        enc_b = self.model.dist_to_encoder().seq_encode(self.data[half:])

        acc_a = self.est.accumulator_factory().make()
        acc_b = self.est.accumulator_factory().make()
        acc_a.seq_update(enc_a, np.ones(half), self.model)
        acc_b.seq_update(enc_b, np.ones(len(self.data) - half), self.model)
        acc_a.combine(acc_b.value())

        ss_split = acc_a.value()
        self.assertTrue(np.allclose(ss_split[1], self.ss[1], rtol=1.0e-8, atol=1.0e-10))
        self.assertTrue(np.allclose(ss_split[2], self.ss[2], rtol=1.0e-8, atol=1.0e-10))
        self.assertTrue(np.allclose(ss_split[3], self.ss[3], rtol=1.0e-8, atol=1.0e-10))

        acc_rt = self.est.accumulator_factory().make().from_value(ss_split)
        self.assertIsInstance(acc_rt.set_stats, LLDALabelSetStats)
        self.assertTrue(np.allclose(acc_rt.value()[1], self.ss[1], rtol=1.0e-8, atol=1.0e-10))

        fitted_split = self.est.estimate(None, ss_split)
        self.assertTrue(np.allclose(fitted_split.alphas, self.fitted.alphas, rtol=1.0e-6, atol=1.0e-8))

    def test_topic_suff_stats_use_weighted_expected_counts_once(self):
        weights = np.linspace(0.25, 1.75, len(self.data))
        acc = self.est.accumulator_factory().make()
        acc.seq_update(self.enc, weights, self.model)

        responsibilities, _, _, _ = seq_posterior(self.model, self.enc)
        expected = responsibilities * np.reshape(weights[self.enc[1]], (-1, 1))
        flat_words = [word for doc, _ in self.data for word, _ in doc]

        _, _, _, topic_counts, topic_suff_stats = acc.value()
        np.testing.assert_allclose(topic_counts.sum(axis=0), expected.sum(axis=0), rtol=1.0e-10, atol=1.0e-10)

        for topic_idx, suff_stat in enumerate(topic_suff_stats):
            for word in sorted(set(flat_words)):
                expected_word_count = sum(expected[i, topic_idx] for i, w in enumerate(flat_words) if w == word)
                self.assertAlmostEqual(suff_stat.get(word, 0.0), expected_word_count, places=10)
            self.assertAlmostEqual(sum(suff_stat.values()), expected[:, topic_idx].sum(), places=10)


if __name__ == "__main__":
    unittest.main()
