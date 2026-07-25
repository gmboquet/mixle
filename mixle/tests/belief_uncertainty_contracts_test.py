"""Contracts for shared-prior fusion and predictive uncertainty mass."""

import unittest

import numpy as np

from mixle.inference.belief import CategoricalBelief, GaussianBelief
from mixle.inference.uncertainty import (
    _entropy_last,
    cluster_samples,
    decompose_entropy,
    decompose_variance,
    marginalize_meaning,
    posterior_ensemble,
    predictive_distribution,
)


class SharedPriorFusionContractsTest(unittest.TestCase):
    def test_ambiguous_posterior_product_is_rejected(self):
        first = GaussianBelief([0.0], [[1.0]])
        second = GaussianBelief([1.0], [[2.0]])
        with self.assertRaises(ValueError):
            first.fuse(second)

    def test_shared_prior_is_subtracted_exactly_once(self):
        prior = GaussianBelief([0.0], [[4.0]])
        first = prior.update([[1.0]], [2.0], [[1.0]])
        second = prior.update([[1.0]], [6.0], [[3.0]])
        fused = first.fuse(second, shared_prior=prior)
        sequential = prior.update([[1.0]], [2.0], [[1.0]]).update([[1.0]], [6.0], [[3.0]])
        np.testing.assert_allclose(fused.mean(), sequential.mean(), atol=1e-12)
        np.testing.assert_allclose(fused.cov(), sequential.cov(), atol=1e-12)

    def test_likelihood_message_requires_explicit_declaration(self):
        posterior = GaussianBelief([2.0], [[1.0]])
        likelihood = GaussianBelief([6.0], [[3.0]])
        fused = posterior.fuse(likelihood, other_is_likelihood=True)
        precision = 1.0 + 1.0 / 3.0
        self.assertAlmostEqual(fused.var()[0], 1.0 / precision)

    def test_density_fusion_rejects_singular_inputs(self):
        singular = GaussianBelief([0.0, 0.0], [[1.0, 1.0], [1.0, 1.0]])
        proper = GaussianBelief([0.0, 0.0], np.eye(2))
        with self.assertRaises(ValueError):
            proper.fuse(proper, shared_prior=singular)

    def test_impossible_categorical_evidence_is_not_certainty(self):
        belief = CategoricalBelief([0.5, 0.5])
        with self.assertRaises(ValueError):
            belief.update([-np.inf, -np.inf])
        with self.assertRaises(ValueError):
            belief.update([np.nan, 0.0])
        with self.assertRaises(ValueError):
            belief.sample(0)


class PredictiveMassContractsTest(unittest.TestCase):
    def test_zero_sum_and_nonfinite_predictive_rows_are_rejected(self):
        invalid = [
            np.array([[0.0, 0.0], [0.5, 0.5]]),
            np.array([[np.nan, 1.0], [0.5, 0.5]]),
            np.array([[-1.0, 2.0], [0.5, 0.5]]),
        ]
        for probabilities in invalid:
            with self.subTest(), self.assertRaises(ValueError):
                decompose_entropy(probabilities)
        with self.assertRaises(ValueError):
            _entropy_last(np.array([0.0, 0.0]))

    def test_predictive_distribution_rejects_impossible_support(self):
        class Impossible:
            def log_density(self, _value):
                return -np.inf

        with self.assertRaises(ValueError):
            predictive_distribution([Impossible(), Impossible()], ["a", "b"])

    def test_variance_decomposition_rejects_nonfinite_values(self):
        with self.assertRaises(ValueError):
            decompose_variance([1.0, np.nan])
        with self.assertRaises(ValueError):
            decompose_variance([1.0, 2.0], [0.1, np.inf])

    def test_single_linkage_uses_transitive_closure(self):
        clustering = cluster_samples([0, 2, 1], equivalent=lambda left, right: abs(left - right) <= 1)
        self.assertEqual(len(clustering.representatives), 1)
        np.testing.assert_array_equal(clustering.labels, [0, 0, 0])

    def test_invalid_meaning_mass_is_rejected(self):
        with self.assertRaises(ValueError):
            marginalize_meaning(["a", "b"], log_probs=[-np.inf, -np.inf])
        with self.assertRaises(ValueError):
            marginalize_meaning(["a", "b"], weights=[1.0, np.nan])
        with self.assertRaises(ValueError):
            marginalize_meaning(["a", "b"], log_probs=[-1.0, -2.0], weights=[1.0, 1.0])

    def test_posterior_ensemble_requires_enough_members(self):
        class Posterior:
            def sample(self, _rng):
                return 1.0

        with self.assertRaises(ValueError):
            posterior_ensemble(Posterior(), lambda value: value, n=1)


if __name__ == "__main__":
    unittest.main()
