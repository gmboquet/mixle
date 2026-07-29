"""Evidence and fallback contracts for model-derived visualization geometry."""

import unittest
from unittest.mock import patch

import numpy as np

from mixle.stats import GaussianDistribution, MixtureDistribution, OptionalDistribution
from mixle.utils.hvis import (
    AffinityCapabilityUnavailableError,
    _posteriors_and_loglikes,
    _resolve_affinity,
    conditional_pmat,
    fisher_factors,
    local_factors,
    model_knn,
    model_log_affinity,
    sparse_model_distances,
)
from mixle.utils.hvis.neighbors import approx_sparse_model_distances
from mixle.utils.vector import ImpossibleEvidenceError


class _ScoredModel:
    def __init__(self, log_likelihoods, log_weights=(0.0, 0.0)):
        self.log_likelihoods = np.asarray(log_likelihoods, dtype=np.float64)
        self.log_w = np.asarray(log_weights, dtype=np.float64)

    def seq_component_log_density(self, _encoded):
        return self.log_likelihoods


class AutomaticAffinityFallbackTest(unittest.TestCase):
    def setUp(self):
        self.model = MixtureDistribution(
            [GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)],
            [0.5, 0.5],
        )

    def test_auto_falls_back_only_for_typed_capability_absence(self):
        with patch(
            "mixle.utils.hvis.affinity.local_factors",
            side_effect=AffinityCapabilityUnavailableError("not representable"),
        ):
            self.assertEqual(_resolve_affinity("auto", self.model, [0.0, 1.0], None), "bhattacharyya")

    def test_auto_surfaces_invalid_input_and_implementation_failures(self):
        for error in (ValueError("invalid field weights"), RuntimeError("broken statistics")):
            with (
                self.subTest(error=type(error).__name__),
                patch("mixle.utils.hvis.affinity.local_factors", side_effect=error),
                self.assertRaises(type(error)),
            ):
                _resolve_affinity("auto", self.model, [0.0, 1.0], None)

    def test_auto_surfaces_nonfinite_field_weights(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            _resolve_affinity("auto", self.model, [0.0, 1.0], [float("nan")])


class PosteriorEvidenceContractTest(unittest.TestCase):
    def test_all_impossible_evidence_preserves_typed_failure(self):
        model = _ScoredModel([[-np.inf, -np.inf], [0.0, -1.0]])
        with self.assertRaises(ImpossibleEvidenceError):
            _posteriors_and_loglikes(model, enc_data=object())

    def test_invalid_likelihoods_and_weights_are_not_normalized(self):
        with self.assertRaisesRegex(ValueError, "finite values or -inf"):
            _posteriors_and_loglikes(_ScoredModel([[np.nan, 0.0]]), enc_data=object())
        with self.assertRaisesRegex(ValueError, "log_w"):
            _posteriors_and_loglikes(
                _ScoredModel([[0.0, 0.0]], log_weights=(float("nan"), 0.0)),
                enc_data=object(),
            )

    def test_valid_evidence_produces_finite_row_stochastic_posteriors(self):
        posterior, likelihood = _posteriors_and_loglikes(
            _ScoredModel([[0.0, -1.0], [-np.inf, 2.0]]),
            enc_data=object(),
        )
        self.assertEqual(posterior.shape, likelihood.shape)
        self.assertTrue(np.all(np.isfinite(posterior)))
        self.assertTrue(np.all(posterior >= 0.0))
        np.testing.assert_allclose(posterior.sum(axis=1), 1.0)


class AffinityAxiomTest(unittest.TestCase):
    def test_public_posterior_affinities_require_probability_rows(self):
        invalid = (
            np.asarray([[0.4, 0.4]]),
            np.asarray([[-0.1, 1.1]]),
            np.asarray([[np.nan, 1.0]]),
        )
        for posterior in invalid:
            with self.subTest(posterior=repr(posterior)), self.assertRaises(ValueError):
                model_log_affinity(posterior)

    def test_negative_or_nonfinite_prebuilt_similarities_are_rejected(self):
        negative = [(np.asarray([[1.0], [-1.0]]), np.ones((2, 1)))]
        with self.assertRaisesRegex(ValueError, "non-negative"):
            model_log_affinity(None, None, affinity=negative)

        nonfinite = [(np.asarray([[1.0], [np.nan]]), np.ones((2, 1)))]
        with self.assertRaisesRegex(ValueError, "finite"):
            model_log_affinity(None, None, affinity=nonfinite)

    def test_zero_similarity_remains_impossible_instead_of_becoming_epsilon(self):
        posterior = np.eye(2)
        result = model_log_affinity(posterior, affinity="coassign")
        self.assertTrue(np.all(np.isneginf(result)))

    def test_factor_weights_and_evidence_cap_must_be_finite_nonnegative(self):
        factor = (np.ones((2, 1)), np.ones((2, 1)), float("nan"))
        with self.assertRaisesRegex(ValueError, "finite"):
            model_log_affinity(None, None, affinity=[factor])
        with self.assertRaisesRegex(ValueError, "evidence_cap"):
            model_log_affinity(np.eye(2), evidence_cap=-1.0)


class OptionalAndFisherGeometryContractTest(unittest.TestCase):
    def test_optional_inner_field_retains_values_and_presence_state(self):
        model = MixtureDistribution(
            [
                OptionalDistribution(GaussianDistribution(-2.0, 1.0), p=0.3),
                OptionalDistribution(GaussianDistribution(2.0, 1.0), p=0.3),
            ],
            [0.5, 0.5],
        )
        data = [None, -2.0, -1.0, None, 1.0, 2.0]
        factors = local_factors(model, data)
        local = [factor for factor in factors if isinstance(factor, dict) and factor.get("kind") == "local"]
        self.assertEqual(len(local), 2)
        inner = max(local, key=lambda factor: factor["x"].shape[1])
        self.assertEqual(inner["x"].shape, (len(data), 2))
        np.testing.assert_array_equal(inner["x"][:, -1], [0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        self.assertNotEqual(inner["x"][1, 0], inner["x"][2, 0])

    def test_fisher_controls_must_be_finite_and_valid(self):
        model = GaussianDistribution(0.0, 1.0)
        for kwargs in (
            {"weight": float("nan")},
            {"weight": float("inf")},
            {"ridge": float("nan")},
            {"ridge": 0.0},
            {"metric": "unknown"},
        ):
            with self.subTest(kwargs=repr(kwargs)), self.assertRaises(ValueError):
                fisher_factors(model, data=[-1.0, 1.0], **kwargs)

    def test_nonfinite_fisher_statistics_are_rejected_not_zero_filled(self):
        class BadView:
            def expected_statistics_matrix(self, *, data):
                return np.asarray([[0.0], [np.nan]])

        class BadModel:
            def to_fisher(self):
                return BadView()

        with self.assertRaisesRegex(ValueError, "statistics"):
            fisher_factors(BadModel(), data=[0.0, 1.0])

    def test_absent_fisher_capability_has_a_typed_result(self):
        with self.assertRaises(AffinityCapabilityUnavailableError):
            fisher_factors(object(), data=[0.0, 1.0])


class ConditionalAffinityContractTest(unittest.TestCase):
    def test_diagonal_is_unconditionally_excluded(self):
        log_affinity = np.asarray([[1000.0, -1.0, -2.0], [-1.0, 1000.0, -3.0], [-2.0, -3.0, 1000.0]])
        probability = conditional_pmat(log_affinity)
        np.testing.assert_array_equal(np.diag(probability), 0.0)
        np.testing.assert_allclose(probability.sum(axis=1), 1.0)

    def test_matrix_requires_finite_off_diagonal_affinities(self):
        invalid = (
            np.asarray([[0.0, np.nan], [-1.0, 0.0]]),
            np.asarray([[0.0, np.inf], [-1.0, 0.0]]),
            np.asarray([[0.0, -np.inf], [-1.0, 0.0]]),
            np.ones((2, 3)),
        )
        for matrix in invalid:
            with self.subTest(matrix=repr(matrix)), self.assertRaises(ValueError):
                conditional_pmat(matrix)

    def test_perplexity_must_be_reachable_without_self_mass(self):
        matrix = np.zeros((4, 4), dtype=np.float64)
        for perplexity in (0.5, 3.1, float("nan"), float("inf")):
            with self.subTest(perplexity=repr(perplexity)), self.assertRaises(ValueError):
                conditional_pmat(matrix, perplexity=perplexity)

        probability = conditional_pmat(matrix, perplexity=3.0)
        np.testing.assert_array_equal(np.diag(probability), 0.0)
        np.testing.assert_allclose(probability.sum(axis=1), 1.0)


class SparseNeighborEvidenceContractTest(unittest.TestCase):
    def setUp(self):
        self.features = np.asarray([[1.0, 0.1], [0.8, 0.2], [0.1, 1.0]], dtype=np.float64)

    def test_zero_total_factor_weight_is_explicitly_no_evidence(self):
        factors = [(self.features, self.features, 0.0)]
        for fn in (sparse_model_distances, model_knn, approx_sparse_model_distances):
            with self.subTest(fn=repr(fn.__name__)), self.assertRaises(ImpossibleEvidenceError):
                fn(None, None, k=1, affinity=factors)

    def test_impossible_similarity_rows_do_not_become_epsilon_neighbors(self):
        factors = [(np.eye(3), np.eye(3))]
        for fn in (sparse_model_distances, model_knn, approx_sparse_model_distances):
            with self.subTest(fn=repr(fn.__name__)), self.assertRaises(ImpossibleEvidenceError):
                fn(None, None, k=1, affinity=factors)

    def test_neighbor_controls_are_not_silently_clamped_or_truncated(self):
        factors = [(self.features, self.features)]
        for kwargs in ({"k": 3}, {"k": 1.5}, {"block_size": 0}, {"evidence_cap": 0.0}):
            with self.subTest(kwargs=repr(kwargs)), self.assertRaises((TypeError, ValueError)):
                sparse_model_distances(None, None, affinity=factors, **kwargs)
        for kwargs in (
            {"n_trees": 0},
            {"n_trees": 1.5},
            {"leaf_size": 4},
            {"candidate_multiplier": 0},
        ):
            with self.subTest(kwargs=repr(kwargs)), self.assertRaises((TypeError, ValueError)):
                approx_sparse_model_distances(None, None, k=1, affinity=factors, **kwargs)

    def test_factor_row_counts_must_match(self):
        factors = [(self.features, self.features), (np.ones((2, 1)), np.ones((2, 1)))]
        with self.assertRaises(ValueError):
            model_knn(None, None, k=1, affinity=factors)


if __name__ == "__main__":
    unittest.main()
