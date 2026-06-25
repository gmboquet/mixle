"""SpatialMixture: a Potts-coupled mixture over a grid with pluggable pysp emission distributions."""

import itertools
import unittest

import numpy as np

from pysp.analysis.spatial_mixture import SpatialMixture
from pysp.stats import GaussianEstimator, MultivariateGaussianDistribution, MultivariateGaussianEstimator


def _layered_field(seed=0):
    rng = np.random.RandomState(seed)
    nx, ny = 40, 40
    true = np.zeros((nx, ny), dtype=int)
    true[14:27] = 1
    true[27:] = 2
    yy, xx = np.mgrid[0:nx, 0:ny]
    true[((xx - 30) ** 2 + (yy - 8) ** 2) < 25] = 2
    means = np.array([[2.0, 2.0], [3.0, 2.6], [4.0, 2.4]])
    obs = [rng.multivariate_normal(means[true[i, j]], 0.45 * np.eye(2)) for i in range(nx) for j in range(ny)]
    return (nx, ny), true, obs


def _best_accuracy(pred, truth, k=3):
    return max(
        np.mean(np.vectorize(dict(zip(range(k), perm)).get)(pred) == truth) for perm in itertools.permutations(range(k))
    )


class SpatialMixtureTest(unittest.TestCase):
    def setUp(self):
        self.shape, self.true, self.obs = _layered_field()

    def test_potts_coupling_beats_an_ordinary_mixture(self):
        spatial = SpatialMixture(self.shape, 3, MultivariateGaussianEstimator(), beta=2.0).fit(self.obs, seed=1)
        plain = SpatialMixture(self.shape, 3, MultivariateGaussianEstimator(), beta=0.0).fit(self.obs, seed=1)
        self.assertGreater(_best_accuracy(spatial.labels(), self.true), 0.9)
        self.assertGreater(_best_accuracy(spatial.labels(), self.true), _best_accuracy(plain.labels(), self.true))

    def test_components_are_pysp_distributions(self):
        sm = SpatialMixture(self.shape, 3, MultivariateGaussianEstimator(), beta=2.0).fit(self.obs, seed=1)
        self.assertIsInstance(sm.component(0), MultivariateGaussianDistribution)
        np.testing.assert_allclose(sorted(sm.component(j).mu[0] for j in range(3)), [2.0, 3.0, 4.0], atol=0.3)

    def test_responsibilities_and_labels(self):
        sm = SpatialMixture(self.shape, 3, MultivariateGaussianEstimator(), beta=2.0).fit(self.obs, seed=1)
        q = sm.responsibilities()
        self.assertEqual(q.shape, (self.shape[0] * self.shape[1], 3))
        np.testing.assert_allclose(q.sum(axis=1), 1.0, atol=1e-8)
        self.assertEqual(sm.labels().shape, self.shape)
        self.assertEqual(sm.entropy().shape, self.shape)

    def test_uncertainty_peaks_at_boundaries(self):
        sm = SpatialMixture(self.shape, 3, MultivariateGaussianEstimator(), beta=2.0).fit(self.obs, seed=1)
        ent = sm.entropy()
        self.assertLess(ent[2:8].mean(), ent[12:16].mean())  # confident interior, uncertain layer boundary

    def test_composes_with_a_different_emission_family(self):
        rng = np.random.RandomState(3)
        obs1 = [
            float(rng.normal([0.0, 5.0, 10.0][self.true.ravel()[i]], 0.6)) for i in range(self.shape[0] * self.shape[1])
        ]
        sm = SpatialMixture(self.shape, 3, GaussianEstimator(), beta=1.5).fit(obs1, seed=0)
        self.assertGreater(_best_accuracy(sm.labels(), self.true), 0.85)
        self.assertEqual(type(sm.component(0)).__name__, "GaussianDistribution")

    def test_three_dimensional_grid(self):
        rng = np.random.RandomState(2)
        true = np.zeros((8, 8, 8), dtype=int)
        true[4:] = 1
        obs = [np.array([rng.normal([0.0, 5.0][true.ravel()[i]], 0.4)]) for i in range(8**3)]
        sm = SpatialMixture((8, 8, 8), 2, MultivariateGaussianEstimator(), beta=1.5).fit(obs, seed=0)
        self.assertEqual(sm.labels().shape, (8, 8, 8))
        self.assertGreater(_best_accuracy(sm.labels().ravel(), true.ravel(), k=2), 0.9)


if __name__ == "__main__":
    unittest.main()
