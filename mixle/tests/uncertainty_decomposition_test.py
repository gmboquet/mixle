"""Tests for epistemic/aleatoric uncertainty decomposition (mixle.inference.uncertainty)."""

import unittest

import numpy as np

from mixle.inference.uncertainty import (
    decompose_entropy,
    decompose_uncertainty,
    decompose_variance,
    posterior_ensemble,
    predictive_distribution,
)


class EntropySplitTest(unittest.TestCase):
    def test_agreeing_members_have_zero_epistemic(self):
        # All members identical -> no disagreement -> epistemic (mutual information) == 0,
        # and aleatoric == total (all uncertainty is genuine ambiguity).
        p = np.array([[0.2, 0.3, 0.5]] * 4)
        d = decompose_entropy(p)
        self.assertEqual(d.kind, "entropy")
        self.assertAlmostEqual(d.epistemic, 0.0, places=12)
        self.assertAlmostEqual(d.aleatoric, d.total, places=12)
        self.assertGreater(d.total, 0.0)

    def test_confident_but_disagreeing_members_are_all_epistemic(self):
        # Each member is a confident one-hot but on different classes -> per-member entropy ~ 0
        # (aleatoric ~ 0), total entropy is high, so essentially all uncertainty is epistemic.
        eps = 1e-6
        a = [1 - 2 * eps, eps, eps]
        b = [eps, 1 - 2 * eps, eps]
        d = decompose_entropy(np.array([a, b]))
        self.assertLess(d.aleatoric, 1e-4)
        self.assertGreater(d.epistemic, 0.6)  # ~ln(2) split between two confident-but-opposed members
        self.assertAlmostEqual(d.total, d.aleatoric + d.epistemic, places=10)

    def test_total_equals_entropy_of_mean(self):
        rng = np.random.RandomState(0)
        p = rng.dirichlet(np.ones(5), size=6)  # (6, 5)
        d = decompose_entropy(p)
        mean = p.mean(axis=0)
        h_mean = float(-np.sum(mean * np.log(mean)))
        self.assertAlmostEqual(d.total, h_mean, places=10)
        # mutual information is nonnegative
        self.assertGreaterEqual(d.epistemic, 0.0)

    def test_batched_query_points(self):
        rng = np.random.RandomState(1)
        p = rng.dirichlet(np.ones(4), size=(3, 7))  # (M=3, N=7, K=4)
        d = decompose_entropy(p)
        self.assertEqual(np.shape(d.total), (7,))
        self.assertEqual(np.shape(d.epistemic), (7,))
        np.testing.assert_allclose(d.total, d.aleatoric + d.epistemic, atol=1e-10)
        self.assertTrue(np.all(d.epistemic >= -1e-12))

    def test_unnormalized_rows_are_renormalized(self):
        p_norm = np.array([[0.1, 0.4, 0.5], [0.6, 0.2, 0.2]])
        p_scaled = p_norm * np.array([[3.0], [10.0]])  # same distributions, different scale
        d1 = decompose_entropy(p_norm)
        d2 = decompose_entropy(p_scaled)
        self.assertAlmostEqual(d1.total, d2.total, places=12)
        self.assertAlmostEqual(d1.epistemic, d2.epistemic, places=12)

    def test_requires_two_members(self):
        with self.assertRaises(ValueError):
            decompose_entropy(np.array([[0.5, 0.5]]))


class VarianceSplitTest(unittest.TestCase):
    def test_law_of_total_variance(self):
        # members: means differ (epistemic), each carries its own noise variance (aleatoric).
        means = np.array([1.0, 3.0, 5.0])
        varis = np.array([0.5, 0.5, 0.5])
        d = decompose_variance(means, varis)
        self.assertEqual(d.kind, "variance")
        self.assertAlmostEqual(d.aleatoric, 0.5, places=12)  # mean of per-member variances
        self.assertAlmostEqual(d.epistemic, np.var(means), places=12)  # spread of the means
        self.assertAlmostEqual(d.total, d.aleatoric + d.epistemic, places=12)

    def test_point_predictors_have_zero_aleatoric(self):
        means = np.array([2.0, 2.0, 8.0, 8.0])
        d = decompose_variance(means)  # no variances given
        self.assertAlmostEqual(d.aleatoric, 0.0, places=12)
        self.assertAlmostEqual(d.epistemic, np.var(means), places=12)

    def test_agreeing_means_zero_epistemic(self):
        means = np.array([4.0, 4.0, 4.0])
        d = decompose_variance(means, np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(d.epistemic, 0.0, places=12)
        self.assertAlmostEqual(d.aleatoric, 2.0, places=12)

    def test_batched(self):
        rng = np.random.RandomState(2)
        means = rng.normal(size=(5, 8))
        varis = rng.random((5, 8)) + 0.1
        d = decompose_variance(means, varis)
        self.assertEqual(np.shape(d.total), (8,))
        np.testing.assert_allclose(d.total, means.var(axis=0) + varis.mean(axis=0), atol=1e-12)

    def test_mismatched_shapes_raise(self):
        with self.assertRaises(ValueError):
            decompose_variance(np.zeros((3, 2)), np.zeros((3, 4)))


class FrontDoorAndBuildersTest(unittest.TestCase):
    def test_front_door_dispatch(self):
        p = np.array([[0.2, 0.8], [0.6, 0.4]])
        self.assertEqual(decompose_uncertainty(probs=p).kind, "entropy")
        self.assertEqual(decompose_uncertainty(means=np.array([1.0, 2.0])).kind, "variance")
        with self.assertRaises(ValueError):
            decompose_uncertainty(probs=p, means=np.array([1.0, 2.0]))
        with self.assertRaises(ValueError):
            decompose_uncertainty()

    def test_predictive_distribution_from_fitted_models(self):
        # Two Poisson-ish members represented by objects with a log_density; predictive_distribution
        # should turn them into normalized categorical rows over a discrete support.
        from mixle.stats import PoissonDistribution

        members = [PoissonDistribution(2.0), PoissonDistribution(6.0)]
        support = list(range(15))
        probs = predictive_distribution(members, support)
        self.assertEqual(probs.shape, (2, len(support)))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)
        d = decompose_entropy(probs)
        # Two clearly different rates disagree -> positive epistemic uncertainty.
        self.assertGreater(d.epistemic, 0.0)

    def test_posterior_ensemble_integrates_parameter_uncertainty(self):
        # A conjugate parameter posterior over a Poisson rate -> ensemble of Poisson models ->
        # discrete predictive split. More data should shrink epistemic uncertainty.
        from mixle.inference import posterior
        from mixle.stats import PoissonDistribution

        def fit_and_split(n):
            rng = np.random.RandomState(0)
            data = rng.poisson(4.0, size=n).tolist()
            pp = posterior(PoissonDistribution(1.0), data, over="params", method="conjugate")
            members = posterior_ensemble(pp, lambda th: PoissonDistribution(_rate(th)), n=60, rng=1)
            probs = predictive_distribution(members, list(range(20)))
            return decompose_entropy(probs).epistemic

        small = fit_and_split(15)
        large = fit_and_split(2000)
        self.assertGreater(small, 0.0)
        self.assertLess(large, small)  # epistemic uncertainty shrinks with more data


def _rate(theta):
    """Pull a scalar Poisson rate out of whatever the parameter posterior hands back."""
    if isinstance(theta, dict):
        for key in ("lam", "lambda", "rate", "mean"):
            if key in theta:
                return float(np.asarray(theta[key]).reshape(-1)[0])
        return float(np.asarray(next(iter(theta.values()))).reshape(-1)[0])
    return float(np.asarray(theta).reshape(-1)[0])


if __name__ == "__main__":
    unittest.main()
