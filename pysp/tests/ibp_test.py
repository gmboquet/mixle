"""Tests for the finite-truncated Indian buffet process implementation."""

import unittest

import numpy as np

from pysp.stats import (
    IndianBuffetProcessDistribution,
    IndianBuffetProcessEstimator,
    seq_encode,
    seq_initialize,
)


class IndianBuffetProcessTestCase(unittest.TestCase):
    def test_vb_posterior_update_dense(self):
        data = [
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
        ]
        est = IndianBuffetProcessEstimator(3, alpha=1.5, estimate_alpha=False, data_format="dense")

        enc = seq_encode(data, estimator=est)
        model = seq_initialize(enc, est, np.random.RandomState(1), p=1.0)

        counts = np.asarray([2.0, 2.0, 2.0])
        expected_a = 0.5 + counts
        expected_b = 1.0 + (3.0 - counts)

        np.testing.assert_allclose(model.beta_params[:, 0], expected_a)
        np.testing.assert_allclose(model.beta_params[:, 1], expected_b)
        np.testing.assert_allclose(model.feature_probs, expected_a / (expected_a + expected_b))
        self.assertAlmostEqual(model.alpha, 1.5)

    def test_pseudocount_without_suff_stat_uses_ibp_prior_mean(self):
        data = [
            [1, 0, 0],
            [0, 0, 0],
        ]
        est = IndianBuffetProcessEstimator(3, alpha=3.0, pseudo_count=2.0, estimate_alpha=False, data_format="dense")
        enc = seq_encode(data, estimator=est)
        model = seq_initialize(enc, est, np.random.RandomState(2), p=1.0)

        expected_a = np.asarray([3.0, 2.0, 2.0])
        expected_b = np.asarray([3.0, 4.0, 4.0])

        np.testing.assert_allclose(model.beta_params[:, 0], expected_a)
        np.testing.assert_allclose(model.beta_params[:, 1], expected_b)

    def test_pseudocount_with_suff_stat_uses_supplied_probabilities(self):
        est = IndianBuffetProcessEstimator(
            3, alpha=3.0, pseudo_count=5.0, suff_stat=[0.8, 0.1, 0.3], estimate_alpha=False, data_format="dense"
        )

        model = est.estimate(None, (np.zeros(3), 0.0, 3.0))

        expected_a = 1.0 + 5.0 * np.asarray([0.8, 0.1, 0.3])
        expected_b = 1.0 + 5.0 * (1.0 - np.asarray([0.8, 0.1, 0.3]))

        np.testing.assert_allclose(model.beta_params[:, 0], expected_a)
        np.testing.assert_allclose(model.beta_params[:, 1], expected_b)

    def test_scalar_and_vectorized_scoring_match_mixed_inputs(self):
        dist = IndianBuffetProcessDistribution(
            4, alpha=2.0, beta_params=[[2.0, 3.0], [4.0, 2.0], [1.5, 5.0], [3.0, 1.0]]
        )
        data = [
            [1, 0, 0, 1],
            [1, 3],
            set([0, 2]),
            [],
        ]

        enc = dist.dist_to_encoder().seq_encode(data)
        np.testing.assert_allclose(dist.seq_log_density(enc), [dist.log_density(x) for x in data])
        np.testing.assert_allclose(dist.seq_expected_log_density(enc), [dist.expected_log_density(x) for x in data])

    def test_sparse_accumulator_matches_vectorized_path(self):
        dist = IndianBuffetProcessDistribution(3, alpha=2.0, data_format="sparse")
        est = dist.estimator()
        factory = est.accumulator_factory()
        data = [[0, 2], [1], [], [0, 1, 2]]
        weights = np.asarray([1.0, 0.5, 2.0, 1.5])

        scalar = factory.make()
        for x, w in zip(data, weights):
            scalar.update(x, w, dist)

        vector = factory.make()
        enc = dist.dist_to_encoder().seq_encode(data)
        vector.seq_update(enc, weights, dist)

        sv = scalar.value()
        vv = vector.value()
        np.testing.assert_allclose(sv[0], vv[0])
        self.assertAlmostEqual(sv[1], vv[1])
        self.assertAlmostEqual(sv[2], vv[2])

    def test_sampler_repeat_and_eval_round_trip(self):
        dist = IndianBuffetProcessDistribution(
            3, alpha=1.0, beta_params=[[2.0, 3.0], [4.0, 1.0], [1.0, 5.0]], data_format="sparse"
        )

        self.assertEqual(dist.sampler(seed=7).sample(size=10), dist.sampler(seed=7).sample(size=10))

        copy = eval(str(dist))
        self.assertEqual(str(copy), str(dist))

    def test_model_log_density_is_finite(self):
        dist = IndianBuffetProcessDistribution(3, alpha=1.2, beta_params=[[2.0, 3.0], [1.5, 4.0], [3.0, 2.0]])
        est = dist.estimator()
        self.assertTrue(np.isfinite(est.model_log_density(dist)))


if __name__ == "__main__":
    unittest.main()
