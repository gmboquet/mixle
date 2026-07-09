"""Modality-fingerprint routing (workstream A1): a fixed-length numeric vector or 2-D numeric array is
not always low-dimensional tabular numeric. Above EMBEDDING_MIN_DIM, or when the field is a homogeneous
2-D numeric array, ``mixle.utils.automatic`` routes to a hybrid neural density instead of a bare
multivariate Gaussian / per-row sequence model -- with the routing reasoning recorded in
``StructureProfile.warnings``. Below the threshold, nothing changes (GMM/MVN stays the default for plain
low-dim tabular numeric).
"""

import unittest
from unittest.mock import patch

import numpy as np

from mixle.stats import MultivariateGaussianEstimator
from mixle.utils.automatic import analyze_structure, get_estimator


def _vectors(dim, n=60, seed=0):
    return np.random.RandomState(seed).normal(size=(n, dim)).tolist()


def _images(h=8, w=8, n=40, seed=0):
    rng = np.random.RandomState(seed)
    return [rng.rand(h, w).tolist() for _ in range(n)]


class LowDimUnchangedTest(unittest.TestCase):
    """The existing, unaffected path: plain low-dimensional tabular numeric stays a bare MVN."""

    def test_low_dim_vector_still_recommends_mvn(self):
        est = get_estimator(_vectors(3))
        self.assertIsInstance(est, MultivariateGaussianEstimator)
        self.assertEqual(est.dim, 3)

    def test_low_dim_vector_carries_no_modality_warning(self):
        profile = analyze_structure(_vectors(3), pairwise=False, validate_marginals=False)
        self.assertFalse(any("modality fingerprint" in w for w in profile.warnings))


class TorchAbsentFallbackTest(unittest.TestCase):
    """Graceful degradation: if torch is unavailable, the existing family is kept and the gap is recorded."""

    def test_embedding_falls_back_to_mvn_without_torch(self):
        with patch("mixle.utils.automatic.profiling._has_torch", return_value=False):
            est = get_estimator(_vectors(20))
        self.assertIsInstance(est, MultivariateGaussianEstimator)
        self.assertEqual(est.dim, 20)

    def test_embedding_fallback_is_recorded(self):
        with patch("mixle.utils.automatic.profiling._has_torch", return_value=False):
            profile = analyze_structure(_vectors(20), pairwise=False, validate_marginals=False)
        self.assertTrue(any("modality fingerprint: embedding" in w and "fell back" in w for w in profile.warnings))

    def test_image_fallback_is_recorded(self):
        with patch("mixle.utils.automatic.profiling._has_torch", return_value=False):
            profile = analyze_structure(_images(), pairwise=False, validate_marginals=False)
        self.assertTrue(any("modality fingerprint: image" in w and "fell back" in w for w in profile.warnings))


class HybridRoutingTest(unittest.TestCase):
    """Torch present: embedding/image-shaped fields route to a hybrid neural density, reasoning recorded."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("hybrid routing needs torch")

    def test_embedding_dim_routes_to_neural_density(self):
        est = get_estimator(_vectors(20))
        self.assertEqual(type(est).__name__, "NeuralDensityEstimator")

    def test_embedding_routing_is_recorded(self):
        profile = analyze_structure(_vectors(20), pairwise=False, validate_marginals=False)
        self.assertTrue(
            any("modality fingerprint: embedding" in w and "hybrid neural density" in w for w in profile.warnings)
        )

    def test_image_shape_routes_to_feature_map(self):
        est = get_estimator(_images())
        self.assertEqual(type(est).__name__, "FeatureMapEstimator")

    def test_image_routing_is_recorded(self):
        profile = analyze_structure(_images(), pairwise=False, validate_marginals=False)
        self.assertTrue(
            any("modality fingerprint: image" in w and "hybrid neural density" in w for w in profile.warnings)
        )

    def test_embedding_field_fits_and_scores_finite(self):
        from mixle.inference import optimize

        data = _vectors(20, n=80)
        est = get_estimator(data)
        fitted = optimize(data, est, max_its=2, out=None)
        enc = fitted.dist_to_encoder().seq_encode(data)
        ll = fitted.seq_log_density(enc)
        self.assertTrue(np.isfinite(ll).all())

    def test_image_field_fits_and_scores_finite(self):
        from mixle.inference import optimize

        data = _images(n=50)
        est = get_estimator(data)
        fitted = optimize(data, est, max_its=2, out=None)
        enc = fitted.dist_to_encoder().seq_encode(data)
        ll = fitted.seq_log_density(enc)
        self.assertTrue(np.isfinite(ll).all())

    def test_recommend_model_reports_modality_reasoning(self):
        from mixle.task.recommend import recommend_model

        rec = recommend_model(_vectors(20), pairwise=False, validate_marginals=False)
        self.assertTrue(any("modality fingerprint" in line for line in rec.explain()))


if __name__ == "__main__":
    unittest.main()
