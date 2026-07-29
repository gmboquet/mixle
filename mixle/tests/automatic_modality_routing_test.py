"""Explicit modality routing: shape alone never asserts embedding/image semantics.

Callers with provenance can request ``modality="embedding"`` or ``"image"``;
ordinary vectors and matrices retain mathematical-array semantics.
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

    def test_wide_vector_is_not_assumed_to_be_an_embedding(self):
        est = get_estimator(_vectors(20))
        self.assertIsInstance(est, MultivariateGaussianEstimator)
        profile = analyze_structure(_vectors(20), pairwise=False, validate_marginals=False)
        self.assertTrue(any("retained mathematical-vector semantics" in warning for warning in profile.warnings))


class TorchAbsentFallbackTest(unittest.TestCase):
    """Graceful degradation: if torch is unavailable, the existing family is kept and the gap is recorded."""

    def test_embedding_falls_back_to_mvn_without_torch(self):
        with patch("mixle.utils.automatic.profiling._has_torch", return_value=False):
            est = get_estimator(_vectors(20), modality="embedding")
        self.assertIsInstance(est, MultivariateGaussianEstimator)
        self.assertEqual(est.dim, 20)

    def test_embedding_fallback_is_recorded(self):
        with patch("mixle.utils.automatic.profiling._has_torch", return_value=False):
            profile = analyze_structure(_vectors(20), pairwise=False, validate_marginals=False, modality="embedding")
        self.assertTrue(any("explicit modality: embedding" in w and "fell back" in w for w in profile.warnings))

    def test_image_fallback_is_recorded(self):
        with patch("mixle.utils.automatic.profiling._has_torch", return_value=False):
            profile = analyze_structure(_images(), pairwise=False, validate_marginals=False, modality="image")
        self.assertTrue(any("explicit modality: image" in w and "fell back" in w for w in profile.warnings))


class HybridRoutingTest(unittest.TestCase):
    """Torch present: embedding/image-shaped fields route to a hybrid neural density, reasoning recorded."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("hybrid routing needs torch")

    def test_embedding_dim_routes_to_neural_density(self):
        est = get_estimator(_vectors(20), modality="embedding")
        self.assertEqual(type(est).__name__, "NeuralDensityEstimator")

    def test_embedding_routing_is_recorded(self):
        profile = analyze_structure(_vectors(20), pairwise=False, validate_marginals=False, modality="embedding")
        self.assertTrue(
            any("explicit modality: embedding" in w and "hybrid neural density" in w for w in profile.warnings)
        )

    def test_image_shape_routes_to_feature_map(self):
        est = get_estimator(_images(), modality="image")
        self.assertEqual(type(est).__name__, "FeatureMapEstimator")

    def test_image_routing_is_recorded(self):
        profile = analyze_structure(_images(), pairwise=False, validate_marginals=False, modality="image")
        self.assertTrue(any("explicit modality: image" in w and "hybrid neural density" in w for w in profile.warnings))

    def test_embedding_field_fits_and_scores_finite(self):
        from mixle.inference import optimize

        data = _vectors(20, n=80)
        est = get_estimator(data, modality="embedding")
        fitted = optimize(data, est, max_its=2, out=None)
        enc = fitted.dist_to_encoder().seq_encode(data)
        ll = fitted.seq_log_density(enc)
        self.assertTrue(np.isfinite(ll).all())

    def test_image_field_fits_and_scores_finite(self):
        from mixle.inference import optimize

        data = _images(n=50)
        est = get_estimator(data, modality="image")
        fitted = optimize(data, est, max_its=2, out=None)
        enc = fitted.dist_to_encoder().seq_encode(data)
        ll = fitted.seq_log_density(enc)
        self.assertTrue(np.isfinite(ll).all())

    def test_recommend_model_reports_modality_reasoning(self):
        from mixle.task.recommend import recommend_model

        rec = recommend_model(_vectors(20), pairwise=False, validate_marginals=False, modality="embedding")
        self.assertTrue(any("explicit modality" in line for line in rec.explain()))

    def test_invalid_explicit_modality_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            get_estimator([1.0, 2.0, 3.0], modality="embedding")
        with self.assertRaisesRegex(ValueError, "requires"):
            analyze_structure(_vectors(3), pairwise=False, modality="image")


def _missing_value_note(warnings):
    return [w for w in warnings if "modality fingerprint" in w and "missing" in w]


class MissingDataFallbackTest(unittest.TestCase):
    """Missing/non-finite entries in an otherwise-eligible numeric-vector field disqualify the
    multivariate-Gaussian joint/dependency route (the distribution requires a fully-observed vector; no
    missing-data-aware joint model exists yet), so the automatic pipeline falls back to the independent
    per-field composite. The fallback must be RECORDED, the same "modality fingerprint" convention used
    for the embedding/image routes above -- not just a silently smaller/different candidate."""

    def test_complete_vector_still_recommends_mvn_with_no_missing_value_note(self):
        # baseline: unaffected when nothing is missing (mirrors LowDimUnchangedTest).
        est = get_estimator(_vectors(4, n=60))
        self.assertIsInstance(est, MultivariateGaussianEstimator)
        profile = analyze_structure(_vectors(4, n=60), pairwise=False, validate_marginals=False)
        self.assertEqual(_missing_value_note(profile.warnings), [])

    def test_missing_values_fall_back_to_independent_composite(self):
        data = _vectors(4, n=60)
        data[0][0] = float("nan")
        est = get_estimator(data)
        self.assertEqual(type(est).__name__, "CompositeEstimator")

    def test_missing_value_fallback_is_recorded(self):
        data = _vectors(4, n=60)
        data[0][0] = float("nan")
        profile = analyze_structure(data, pairwise=False, validate_marginals=False)
        notes = _missing_value_note(profile.warnings)
        self.assertEqual(len(notes), 1, f"expected exactly one missing-value modality note, got: {profile.warnings}")
        self.assertIn("1 of 4", notes[0])  # names how many of the fields carried the missing/non-finite values

    def test_none_entries_also_count_as_missing(self):
        data = _vectors(4, n=60)
        data[0][0] = None
        profile = analyze_structure(data, pairwise=False, validate_marginals=False)
        self.assertEqual(len(_missing_value_note(profile.warnings)), 1)

    def test_tuple_rows_are_unaffected_by_the_new_note(self):
        # Tuple-typed rows already route to Composite regardless of missing values -- a separate,
        # pre-existing "fixed-arity tuples are records" rule (see get_estimator's first structured
        # branch). Missing data is never what decided the route here, so the note must not fire.
        rows = [tuple(row) for row in _vectors(4, n=60)]
        rows[0] = (float("nan"),) + rows[0][1:]
        profile = analyze_structure(rows, pairwise=False, validate_marginals=False)
        self.assertEqual(type(profile.estimator).__name__, "CompositeEstimator")
        self.assertEqual(_missing_value_note(profile.warnings), [])

    def test_single_field_vector_is_unaffected_by_the_new_note(self):
        # dim<=1 never qualified for the joint/dependency route in the first place (nothing to lose) --
        # the note is specifically about a *joint* route becoming unavailable, so it must not fire.
        data = [[x] for x in np.random.RandomState(0).normal(size=60)]
        data[0][0] = float("nan")
        profile = analyze_structure(data, pairwise=False, validate_marginals=False)
        self.assertEqual(_missing_value_note(profile.warnings), [])

    def test_propose_surfaces_the_fallback_reason_in_notes(self):
        # The end-to-end entry point an external caller actually sees: mixle.propose(data). The
        # frontier legitimately collapses to one candidate (recommended == independent baseline, per
        # test_propose_skips_independent_baseline_when_structurally_identical in lifecycle_test.py) --
        # but that collapse must be explained, not silent.
        import mixle

        data = _vectors(4, n=200)
        for i in range(0, len(data), 10):
            data[i][0] = float("nan")
        m = mixle.propose(data, fit=True)
        names = [f["name"] for f in m.frontier]
        self.assertEqual(names.count("independent"), 0)
        self.assertTrue(
            any("missing" in n and "composite" in n for n in m.notes),
            f"no explanation of the missing-value dependency-route fallback in notes: {m.notes}",
        )


if __name__ == "__main__":
    unittest.main()
