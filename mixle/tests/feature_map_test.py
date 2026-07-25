"""FeatureMapDensity/Estimator: a frozen, deterministic feature map composed with any inner density.

The "frozen encoder + structured head" leaf used by the modality-routing upgrade (workstream A1) for
image-shaped fields. Uses a plain (torch-free) inner distribution here so this file runs in the base suite;
mixle/tests/automatic_modality_routing_test.py exercises the torch-backed hybrid-density callers.
"""

import unittest

import numpy as np

from mixle.inference import optimize
from mixle.models.feature_map import (
    FeatureMapDensity,
    FeatureMapEstimator,
    register_feature_fn,
)
from mixle.stats import MultivariateGaussianEstimator
from mixle.utils.serialization import from_json, to_json


def _pair_sums(x):
    return np.array([float(np.sum(x[:2])), float(np.sum(x[2:]))])


register_feature_fn("test_pair_sums", _pair_sums, version="1", feature_dim=2)


def _data(n=200, seed=0):
    return np.random.RandomState(seed).normal(size=(n, 4)).tolist()


class FeatureMapTest(unittest.TestCase):
    def test_fits_the_inner_distribution_on_featurized_data(self):
        est = FeatureMapEstimator("test_pair_sums", MultivariateGaussianEstimator(dim=2))
        fitted = optimize(_data(), est, max_its=3, out=None)
        self.assertIsInstance(fitted, FeatureMapDensity)
        self.assertEqual(fitted.inner.dim, 2)

    def test_log_density_matches_inner_on_featurized_point(self):
        est = FeatureMapEstimator("test_pair_sums", MultivariateGaussianEstimator(dim=2))
        fitted = optimize(_data(), est, max_its=3, out=None)
        x = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(fitted.log_density(x), fitted.inner.log_density(_pair_sums(np.asarray(x))), places=8)

    def test_seq_log_density_is_finite_and_matches_single_calls(self):
        data = _data(n=40)
        fitted = optimize(
            data, FeatureMapEstimator("test_pair_sums", MultivariateGaussianEstimator(dim=2)), max_its=3, out=None
        )
        enc = fitted.dist_to_encoder().seq_encode(data)
        batch_ll = fitted.seq_log_density(enc)
        single_ll = np.array([fitted.log_density(row) for row in data])
        self.assertTrue(np.isfinite(batch_ll).all())
        self.assertTrue(np.allclose(batch_ll, single_ll, atol=1e-8))

    def test_json_round_trip_preserves_type_and_scores(self):
        data = _data(n=40)
        fitted = optimize(
            data, FeatureMapEstimator("test_pair_sums", MultivariateGaussianEstimator(dim=2)), max_its=3, out=None
        )
        back = from_json(to_json(fitted))
        self.assertIsInstance(back, FeatureMapDensity)
        self.assertEqual(back.feature_name, "test_pair_sums")
        self.assertEqual(back.feature_version, "1")
        self.assertEqual(back.feature_dim, 2)
        self.assertEqual(back.feature_digest, fitted.feature_digest)
        enc = fitted.dist_to_encoder().seq_encode(data)
        self.assertTrue(np.allclose(fitted.seq_log_density(enc), back.seq_log_density(enc)))

    def test_raw_sampler_is_unavailable_and_feature_sampler_is_explicit(self):
        fitted = optimize(
            _data(), FeatureMapEstimator("test_pair_sums", MultivariateGaussianEstimator(dim=2)), max_its=3, out=None
        )
        with self.assertRaisesRegex(NotImplementedError, "cannot sample raw items"):
            fitted.sampler(0)
        draws = fitted.feature_sampler(0).sample(5)
        self.assertEqual(np.asarray(draws).shape, (5, 2))  # feature space is 2-D, not the raw 4-D input

    def test_unregistered_feature_name_raises(self):
        with self.assertRaises(KeyError):
            FeatureMapEstimator("does_not_exist", MultivariateGaussianEstimator(dim=2))

    def test_duplicate_registration_cannot_change_an_existing_identity(self):
        def incompatible(x):
            return np.asarray(x[:2], dtype=float)

        with self.assertRaisesRegex(ValueError, "already registered"):
            register_feature_fn("test_pair_sums", incompatible, version="1", feature_dim=2)

    def test_feature_shape_and_finiteness_are_checked_before_stacking(self):
        register_feature_fn("test_bad_shape", lambda x: np.asarray(x), version="1", feature_dim=2)
        encoder = FeatureMapEstimator(
            "test_bad_shape",
            MultivariateGaussianEstimator(dim=2),
        ).accumulator_factory().make().acc_to_encoder()
        with self.assertRaisesRegex(ValueError, "must return shape"):
            encoder.seq_encode([[1.0], [1.0, 2.0]])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            encoder.seq_encode([[1.0, np.nan]])


if __name__ == "__main__":
    unittest.main()
