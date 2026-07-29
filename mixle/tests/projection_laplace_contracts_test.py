"""Contracts for Gaussian projection, Fisher merging, and Laplace curvature."""

import unittest
from unittest import mock

import numpy as np

import mixle.stats as st
from mixle.inference import optimize
from mixle.inference.blackbox import _flatten, laplace_posterior
from mixle.inference.project import collapse_mixture, fisher_merge, gaussian_kl


class _GaussianLike:
    def __init__(self, mean, covariance):
        self.mu = np.asarray(mean, dtype=float)
        self.covar = np.asarray(covariance, dtype=float)


class _MixtureLike:
    def __init__(self, weights, components):
        self.w = np.asarray(weights, dtype=float)
        self.components = components


class ProjectionContractsTest(unittest.TestCase):
    def test_mixture_weights_and_covariances_are_validated(self):
        components = [_GaussianLike([0.0], [[1.0]]), _GaussianLike([1.0], [[1.0]])]
        for weights in ([0.0, 0.0], [1.0, -1.0], [1.0, np.nan]):
            with self.subTest(weights=repr(weights)), self.assertRaises(ValueError):
                collapse_mixture(_MixtureLike(weights, components))
        with self.assertRaises(ValueError):
            collapse_mixture(_MixtureLike([0.5, 0.5], [components[0], _GaussianLike([1.0], [[-1.0]])]))

    def test_unnormalized_positive_weight_mass_is_explicitly_normalized(self):
        mixture = _MixtureLike(
            [2.0, 1.0],
            [_GaussianLike([0.0], [[1.0]]), _GaussianLike([3.0], [[1.0]])],
        )
        collapsed = collapse_mixture(mixture)
        self.assertAlmostEqual(collapsed.mu[0], 1.0)

    def test_gaussian_kl_rejects_non_spd_or_mismatched_geometry(self):
        good = _GaussianLike([0.0, 0.0], np.eye(2))
        with self.assertRaises(ValueError):
            gaussian_kl(good, _GaussianLike([0.0, 0.0], np.diag([1.0, -1.0])))
        with self.assertRaises(ValueError):
            gaussian_kl(good, _GaussianLike([0.0], [[1.0]]))

    def test_fisher_merge_rejects_indefinite_and_unidentified_information(self):
        estimates = [np.array([0.0, 1.0]), np.array([1.0, 0.0])]
        invalid = [
            [np.diag([1.0, -1.0]), np.eye(2)],
            [np.array([[1.0, 2.0], [0.0, 1.0]]), np.eye(2)],
            [np.array([1.0, 0.0]), np.array([1.0, 0.0])],
        ]
        for information in invalid:
            with self.subTest(), self.assertRaises(ValueError):
                fisher_merge(estimates, information)


class _CurvatureModel:
    def __init__(self, coordinates, sign):
        self.coordinates = np.asarray(coordinates, dtype=float)
        self.sign = float(sign)

    def dist_to_encoder(self):
        return self

    def seq_encode(self, data):
        return tuple(data)

    def seq_log_density(self, encoded):
        value = self.sign * 0.5 * self.coordinates[0] ** 2
        return np.full(len(encoded), value)


def _curvature_flatten(sign, dimension=1):
    def flatten(_model):
        mode = np.zeros(dimension)

        def rebuild(coordinates):
            return _CurvatureModel(coordinates, sign), np.array([])

        return mode, rebuild

    return flatten


class LaplaceContractsTest(unittest.TestCase):
    def _fit(self):
        data = tuple(np.random.RandomState(1).normal(2.0, 1.0, 100))
        return optimize(data, st.GaussianEstimator(), out=None), data

    def test_generator_data_are_reused_for_every_finite_difference(self):
        model, data = self._fit()
        from_list = laplace_posterior(model, data)
        from_generator = laplace_posterior(model, (value for value in data))
        np.testing.assert_allclose(from_generator.cov, from_list.cov)

    def test_likelihood_only_approximation_is_not_called_a_posterior(self):
        model, data = self._fit()
        approximation = laplace_posterior(model, data)
        summary = approximation.summary()
        self.assertFalse(approximation.is_posterior)
        self.assertEqual(summary["target"], "likelihood")
        self.assertFalse(summary["prior_included"])
        self.assertIn("curvature_rank", summary)
        self.assertIn("regularization", summary)

    def test_explicit_prior_produces_disclosed_posterior_target(self):
        model, data = self._fit()
        mode, _ = _flatten(model)
        approximation = laplace_posterior(
            model,
            data,
            log_prior=lambda coordinates: -0.5 * float(np.sum((coordinates - mode) ** 2)) / 100.0,
        )
        self.assertTrue(approximation.is_posterior)
        self.assertEqual(approximation.summary()["target"], "posterior")

    def test_nonstationary_model_is_rejected(self):
        data = tuple(np.random.RandomState(2).normal(5.0, 1.0, 100))
        with self.assertRaises(ValueError):
            laplace_posterior(st.GaussianDistribution(0.0, 1.0), data, mode_tol=1e-8)

    def test_saddle_curvature_is_rejected_instead_of_clipped(self):
        with mock.patch("mixle.inference.blackbox._flatten", _curvature_flatten(+1.0)):
            with self.assertRaises(ValueError):
                laplace_posterior(_CurvatureModel([0.0], +1.0), [0.0])

    def test_rank_deficiency_requires_and_discloses_regularization(self):
        with mock.patch("mixle.inference.blackbox._flatten", _curvature_flatten(-1.0, dimension=2)):
            model = _CurvatureModel([0.0, 0.0], -1.0)
            with self.assertRaises(ValueError):
                laplace_posterior(model, [0.0], ridge=0)
            approximation = laplace_posterior(model, [0.0], ridge=1e-4)
            self.assertEqual(approximation.metadata["curvature_rank"], 1)
            self.assertEqual(approximation.metadata["approximation_status"], "regularized_rank_deficient_local_mode")

    def test_sampling_controls_and_covariance_are_validated(self):
        model, data = self._fit()
        approximation = laplace_posterior(model, data)
        with self.assertRaises(ValueError):
            approximation.sample(0)
        with self.assertRaises(ValueError):
            laplace_posterior(model, data, eps=0)
        with self.assertRaises(ValueError):
            laplace_posterior(model, data, ridge=-1)


if __name__ == "__main__":
    unittest.main()
