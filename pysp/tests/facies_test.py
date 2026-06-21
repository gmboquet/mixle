"""Latent facies field (spatial HMRF): subsurface composition/structure from multimodal properties."""

import itertools
import unittest

import numpy as np

from pysp.stats.facies import LatentFaciesField


def _synthetic_subsurface(seed=0):
    rng = np.random.RandomState(seed)
    nx, ny = 40, 40
    true = np.zeros((nx, ny), dtype=int)
    true[14:27] = 1
    true[27:] = 2
    yy, xx = np.mgrid[0:nx, 0:ny]
    true[((xx - 30) ** 2 + (yy - 8) ** 2) < 25] = 2  # a reservoir lens in the top layer
    fmeans = np.array([[2.0, 2.0], [3.0, 2.6], [4.0, 2.4]])  # [Vp, density] per facies (overlapping)
    obs = np.array(
        [rng.multivariate_normal(fmeans[true[i, j]], 0.45 * np.eye(2)) for i in range(nx) for j in range(ny)]
    )
    return (nx, ny), true, obs


def _best_accuracy(pred, truth, k=3):
    return max(
        np.mean(np.vectorize(dict(zip(range(k), perm)).get)(pred) == truth) for perm in itertools.permutations(range(k))
    )


class LatentFaciesFieldTest(unittest.TestCase):
    def setUp(self):
        self.shape, self.true, self.obs = _synthetic_subsurface()

    def test_recovers_the_structure(self):
        m = LatentFaciesField(self.shape, 3, beta=2.0).fit(self.obs, seed=1)
        self.assertGreater(_best_accuracy(m.map_facies(), self.true), 0.85)

    def test_spatial_coherence_beats_independent_classification(self):
        gmm = LatentFaciesField(self.shape, 3, beta=0.0).fit(self.obs, seed=1)
        hmrf = LatentFaciesField(self.shape, 3, beta=2.0).fit(self.obs, seed=1)
        self.assertGreater(_best_accuracy(hmrf.map_facies(), self.true), _best_accuracy(gmm.map_facies(), self.true))

    def test_recovers_facies_properties(self):
        m = LatentFaciesField(self.shape, 3, beta=2.0).fit(self.obs, seed=1)
        np.testing.assert_allclose(np.sort(m.means[:, 0]), [2.0, 3.0, 4.0], atol=0.3)

    def test_no_dead_facies(self):
        m = LatentFaciesField(self.shape, 3, beta=2.0).fit(self.obs, seed=1)
        self.assertTrue(np.all(m.weights > 0.05))

    def test_posterior_is_a_composition_distribution(self):
        m = LatentFaciesField(self.shape, 3, beta=2.0).fit(self.obs, seed=1)
        q = m.posterior()
        self.assertEqual(q.shape, (self.shape[0] * self.shape[1], 3))
        np.testing.assert_allclose(q.sum(axis=1), 1.0, atol=1e-8)

    def test_uncertainty_peaks_at_boundaries(self):
        m = LatentFaciesField(self.shape, 3, beta=2.0).fit(self.obs, seed=1)
        ent = m.entropy()
        self.assertLess(ent[2:8].mean(), ent[12:16].mean())  # interior is confident, layer boundary is not

    def test_facies_distribution_is_a_pysp_mvn(self):
        from pysp.stats import MultivariateGaussianDistribution

        m = LatentFaciesField(self.shape, 3, beta=2.0).fit(self.obs, seed=1)
        self.assertIsInstance(m.facies_distribution(0), MultivariateGaussianDistribution)

    def test_three_dimensional_grid(self):
        rng = np.random.RandomState(2)
        true = np.zeros((8, 8, 8), dtype=int)
        true[4:] = 1
        obs = np.array([rng.normal([0.0, 5.0][true.ravel()[i]], 0.4, 2) for i in range(8**3)])
        m = LatentFaciesField((8, 8, 8), 2, beta=1.5).fit(obs, seed=0)
        self.assertEqual(m.map_facies().shape, (8, 8, 8))
        self.assertGreater(_best_accuracy(m.map_facies().ravel(), true.ravel(), k=2), 0.9)


if __name__ == "__main__":
    unittest.main()
