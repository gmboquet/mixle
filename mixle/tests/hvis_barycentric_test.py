"""The (posterior, remainder) decomposition made first-class (mixle.utils.hvis):
mixture_coordinates exposes both levels of the observation description, and Y='barycentric'
initializes the embedding from the barycentric reading of the posterior -- the layout's global
arrangement comes from the model's own component geometry instead of the random seed.
"""

import io
import unittest

import numpy as np

from mixle.stats import GaussianDistribution, MixtureDistribution
from mixle.utils.hvis import barycentric_init, component_map, htsne, mixture_coordinates

# three regimes on a line: 0 and 4 are confusable, 20 is far -- the overlap geometry is unambiguous
_MODEL3 = MixtureDistribution(
    [GaussianDistribution(0.0, 1.0), GaussianDistribution(4.0, 1.0), GaussianDistribution(20.0, 1.0)],
    [1.0 / 3.0] * 3,
)


def _data3(n_per=25, seed=0):
    rng = np.random.RandomState(seed)
    xs = np.concatenate([rng.normal(0.0, 1.0, n_per), rng.normal(4.0, 1.0, n_per), rng.normal(20.0, 1.0, n_per)])
    labels = np.repeat([0, 1, 2], n_per)
    return [float(v) for v in xs], labels


class MixtureCoordinatesTest(unittest.TestCase):
    def test_decomposition_exposes_both_levels(self):
        data, _ = _data3()
        decomp = mixture_coordinates(_MODEL3, data)
        z = decomp["posterior"]
        self.assertEqual(z.shape, (75, 3))
        np.testing.assert_allclose(z.sum(axis=1), 1.0, atol=1.0e-9)  # barycentric: on the simplex
        self.assertEqual(len(decomp["fields"]), 1)  # one scalar Gaussian leaf
        field = decomp["fields"][0]
        self.assertEqual(field["log_density"].shape, (75, 3))
        self.assertTrue(field["native"])  # Gaussian leaf: native value coordinates
        self.assertEqual(field["coords"].shape, (75, 1))

    def test_field_weights_length_is_validated(self):
        data, _ = _data3(n_per=5)
        with self.assertRaises(ValueError):
            mixture_coordinates(_MODEL3, data, field_weights=[1.0, 1.0])


class ComponentMapTest(unittest.TestCase):
    def test_vertices_reflect_overlap_geometry(self):
        data, _ = _data3()
        z = mixture_coordinates(_MODEL3, data)["posterior"]
        vertices = component_map(z)
        self.assertEqual(vertices.shape, (3, 2))
        d01 = float(np.linalg.norm(vertices[0] - vertices[1]))
        d02 = float(np.linalg.norm(vertices[0] - vertices[2]))
        d12 = float(np.linalg.norm(vertices[1] - vertices[2]))
        self.assertLess(d01, d02)  # confusable regimes (0, 4) sit closer than either does to 20
        self.assertLess(d01, d12)

    def test_single_component_degenerates_to_origin(self):
        z = np.ones((10, 1))
        np.testing.assert_array_equal(component_map(z), np.zeros((1, 2)))


class BarycentricInitTest(unittest.TestCase):
    def test_init_is_scaled_and_deterministic(self):
        data, _ = _data3()
        z = mixture_coordinates(_MODEL3, data)["posterior"]
        y1 = barycentric_init(z, seed=0)
        y2 = barycentric_init(z, seed=0)
        np.testing.assert_array_equal(y1, y2)
        self.assertAlmostEqual(float(y1.std()), 1.0e-4, delta=3.0e-5)  # optimizer-conventional scale

    def test_global_arrangement_is_seed_independent(self):
        # the payoff of the barycentric reading: the random seed no longer decides which clusters
        # sit near which. Two runs with different seeds must produce the SAME centroid geometry.
        data, labels = _data3()

        def centroid_gaps(seed):
            y = htsne(
                data, mix_model=_MODEL3, Y="barycentric", method="exact", max_its=300, seed=seed, out=io.StringIO()
            )
            cents = np.stack([y[labels == c].mean(axis=0) for c in (0, 1, 2)])
            d01 = np.linalg.norm(cents[0] - cents[1])
            return d01 / np.linalg.norm(cents[0] - cents[2]), d01 / np.linalg.norm(cents[1] - cents[2])

        r_a = centroid_gaps(seed=1)
        r_b = centroid_gaps(seed=2)
        for a, b in zip(r_a, r_b):
            self.assertLess(a, 0.9)  # confusable pair closer than either is to the far regime...
            self.assertLess(abs(a - b), 0.05)  # ...and the arrangement is reproducible across seeds

    def test_mixed_membership_point_lands_between_its_clusters(self):
        data, labels = _data3()
        data = data + [2.0]  # posterior ~(0.5, 0.5, 0) between regimes 0 and 4
        y = htsne(data, mix_model=_MODEL3, Y="barycentric", method="exact", max_its=300, seed=3, out=io.StringIO())
        cents = np.stack([y[:75][labels == c].mean(axis=0) for c in (0, 1, 2)])
        point = y[75]
        d0, d1, d2 = (float(np.linalg.norm(point - cents[c])) for c in (0, 1, 2))
        self.assertLess(max(d0, d1), d2)  # closer to both parent regimes than to the far one
        midpoint = (cents[0] + cents[1]) / 2.0
        self.assertLess(float(np.linalg.norm(point - midpoint)), float(np.linalg.norm(cents[0] - cents[1])))

    def test_barnes_hut_engine_accepts_barycentric(self):
        data, labels = _data3()
        y = htsne(data, mix_model=_MODEL3, Y="barycentric", method="barnes_hut", max_its=300, seed=4, out=io.StringIO())
        self.assertEqual(y.shape, (75, 2))
        d2 = np.square(y[:, None, :] - y[None, :, :]).sum(axis=2)
        np.fill_diagonal(d2, np.inf)
        self.assertGreater(float(np.mean(labels[d2.argmin(axis=1)] == labels)), 0.9)

    def test_unknown_string_and_prebuilt_affinity_raise(self):
        data, _ = _data3(n_per=8)
        with self.assertRaises(ValueError):
            htsne(data, mix_model=_MODEL3, Y="not-a-mode", method="exact", out=io.StringIO())
        from mixle.utils.hvis import local_factors

        factors = local_factors(_MODEL3, data)
        with self.assertRaises(ValueError):
            htsne(data, affinity=factors, Y="barycentric", method="exact", out=io.StringIO())


if __name__ == "__main__":
    unittest.main()
