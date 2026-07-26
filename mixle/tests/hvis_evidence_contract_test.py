"""Evidence and fallback contracts for model-derived visualization geometry."""

import unittest
from unittest.mock import patch

import numpy as np

from mixle.stats import GaussianDistribution, MixtureDistribution, OptionalDistribution
from mixle.utils.hvis import (
    AffinityCapabilityUnavailableError,
    _posteriors_and_loglikes,
    _resolve_affinity,
    fisher_factors,
    local_factors,
    model_log_affinity,
)
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
            with self.subTest(error=type(error).__name__), patch(
                "mixle.utils.hvis.affinity.local_factors", side_effect=error
            ), self.assertRaises(type(error)):
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
            with self.subTest(posterior=posterior), self.assertRaises(ValueError):
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
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
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


if __name__ == "__main__":
    unittest.main()
