"""Distributed layout construction (mixle.utils.hvis.distributed).

The contract is EQUALITY, not resemblance: distributed_model_map over shards reproduces
model_map over the concatenation to floating-point summation order, for any chunking, chart,
field mixture, and mapper -- because every learned quantity is an additive sufficient statistic
(the one order statistic, the occlusion radius, is gathered exactly). If direct.py's math drifts,
these tests fail loudly rather than letting the two paths diverge quietly.
"""

import unittest

import numpy as np

from mixle.stats import (
    CategoricalDistribution,
    CompositeDistribution,
    DiagonalGaussianDistribution,
    GaussianDistribution,
    MixtureDistribution,
)
from mixle.utils.hvis import distributed_model_map, fiber_stats, fuzzy_nerve_from_stats, model_map
from mixle.utils.hvis.topology import fuzzy_nerve

_MODEL3 = MixtureDistribution(
    [GaussianDistribution(0.0, 1.0), GaussianDistribution(4.0, 1.0), GaussianDistribution(20.0, 1.0)],
    [1.0 / 3.0] * 3,
)


def _data3(n_per=30, seed=0):
    rng = np.random.RandomState(seed)
    xs = np.concatenate([rng.normal(0.0, 1.0, n_per), rng.normal(4.0, 1.0, n_per), rng.normal(20.0, 1.0, n_per)])
    return [float(v) for v in xs]


def _chunks(data, m):
    edges = np.linspace(0, len(data), m + 1).astype(int)
    return [data[a:b] for a, b in zip(edges[:-1], edges[1:])]


def _assert_maps_equal(test, a, b, atol=1.0e-8):
    np.testing.assert_allclose(a.vertices, b.vertices, atol=atol)
    for pa, pb in zip(a.loadings, b.loadings):
        np.testing.assert_allclose(pa, pb, atol=atol)
    for fa, fb in zip(a.frames, b.frames):
        np.testing.assert_allclose(fa, fb, atol=atol)
    np.testing.assert_allclose(a.chart_residuals, b.chart_residuals, atol=atol)
    test.assertEqual(a.coord_labels, b.coord_labels)
    np.testing.assert_allclose(a.coords, b.coords, atol=atol)
    np.testing.assert_allclose(a.responsibilities, b.responsibilities, atol=atol)


class EqualityTest(unittest.TestCase):
    def test_linear_map_matches_single_machine(self):
        data = _data3()
        local = model_map(data, mix_model=_MODEL3)
        dist = distributed_model_map(_chunks(data, 3), _MODEL3)
        _assert_maps_equal(self, dist, local)

    def test_chunking_is_irrelevant(self):
        data = _data3(seed=1)
        two = distributed_model_map(_chunks(data, 2), _MODEL3)
        five = distributed_model_map(_chunks(data, 5), _MODEL3)
        _assert_maps_equal(self, two, five)

    def test_quadratic_chart_matches(self):
        data = _data3(seed=2)
        local = model_map(data, mix_model=_MODEL3, chart="quadratic")
        dist = distributed_model_map(_chunks(data, 4), _MODEL3, chart="quadratic")
        self.assertEqual(dist.chart, "quadratic")
        _assert_maps_equal(self, dist, local)

    def test_wide_quadratic_lift_needs_and_gets_the_extra_pass(self):
        # 10-D fibers exceed the lift cap, so the distributed path must fit the per-component
        # pre-PCA from pass-2 moments and re-run the pass for the lifted moments.
        rng = np.random.RandomState(3)
        centers = np.zeros((2, 10))
        centers[1, 0] = 8.0
        model = MixtureDistribution([DiagonalGaussianDistribution(list(c), [1.0] * 10) for c in centers], [0.5, 0.5])
        data = [list(centers[i % 2] + rng.normal(0, 1, 10)) for i in range(80)]
        local = model_map(data, mix_model=model, chart="quadratic")
        dist = distributed_model_map(_chunks(data, 3), model, chart="quadratic")
        self.assertTrue(dist.coord_labels[0].startswith("pre_pc0"))
        _assert_maps_equal(self, dist, local)

    def test_mixed_continuous_discrete_fields_match(self):
        # composite (Gaussian x Categorical): exercises typicality coordinates, per-field
        # whitening from moments, and multi-field concatenation.
        rng = np.random.RandomState(4)
        model = MixtureDistribution(
            [
                CompositeDistribution(
                    [GaussianDistribution(0.0, 1.0), CategoricalDistribution({"a": 0.8, "b": 0.1, "c": 0.1})]
                ),
                CompositeDistribution(
                    [GaussianDistribution(6.0, 1.0), CategoricalDistribution({"a": 0.1, "b": 0.1, "c": 0.8})]
                ),
            ],
            [0.5, 0.5],
        )
        data = [(float(rng.normal(0.0 if i % 2 == 0 else 6.0, 1.0)), rng.choice(["a", "b", "c"])) for i in range(90)]
        local = model_map(data, mix_model=model)
        dist = distributed_model_map(_chunks(data, 3), model)
        _assert_maps_equal(self, dist, local)

    def test_occlusion_resolution_matches_on_the_adversarial_fixture(self):
        # spread=1.2 makes fiber clouds larger than the nerve's rendering gap: the percentile
        # radii and the deterministic push-apart must reproduce exactly from gathered norms.
        data = _data3()
        local = model_map(data, mix_model=_MODEL3, spread=1.2, occlusion=True)
        dist = distributed_model_map(_chunks(data, 4), _MODEL3, spread=1.2, occlusion=True)
        _assert_maps_equal(self, dist, local)


class MapperAndGeometryOnlyTest(unittest.TestCase):
    def test_pool_mapper_matches_serial(self):
        from multiprocessing.dummy import Pool

        data = _data3(seed=5)
        serial = distributed_model_map(_chunks(data, 4), _MODEL3)
        with Pool(4) as pool:
            pooled = distributed_model_map(_chunks(data, 4), _MODEL3, mapper=pool.map)
        _assert_maps_equal(self, pooled, serial)

    def test_geometry_only_then_place_per_shard(self):
        data = _data3(seed=6)
        full = distributed_model_map(_chunks(data, 3), _MODEL3)
        geo = distributed_model_map(_chunks(data, 3), _MODEL3, with_points=False)
        self.assertEqual(geo.coords.shape[0], 0)  # nothing gathered
        placed = np.vstack([geo.place(c) for c in _chunks(data, 3)])
        np.testing.assert_allclose(placed, full.coords, atol=1.0e-10)

    def test_requires_a_fitted_model(self):
        with self.assertRaises(ValueError):
            distributed_model_map(_chunks(_data3(), 2), None)

    def test_rejects_all_empty_shards(self):
        with self.assertRaises(ValueError):
            distributed_model_map([[], []], _MODEL3)


class DistributedNerveTest(unittest.TestCase):
    def test_nerve_from_combined_stats_matches_fuzzy_nerve(self):
        data = _data3(seed=7)
        z_model = _MODEL3
        from mixle.utils.hvis import _posteriors_and_loglikes

        z, _ = _posteriors_and_loglikes(z_model, data=data)
        want = fuzzy_nerve(z)

        parts = [fiber_stats(z_model, c, nerve_triple=True) for c in _chunks(data, 3)]
        stats = parts[0] + parts[1] + parts[2]
        got = fuzzy_nerve_from_stats(stats)

        np.testing.assert_allclose(got["masses"], want["masses"], atol=1.0e-9)
        self.assertEqual(set(got["edges"]), set(want["edges"]))
        for e, w in want["edges"].items():
            self.assertAlmostEqual(got["edges"][e], w, places=9)
        self.assertEqual(set(got["triangles"]), set(want["triangles"]))
        for t, w in want["triangles"].items():
            self.assertAlmostEqual(got["triangles"][t], w, places=9)

    def test_triangles_without_the_tensor_raise_rather_than_undercount(self):
        stats = fiber_stats(_MODEL3, _data3(seed=8))  # no nerve_triple
        with self.assertRaises(ValueError):
            fuzzy_nerve_from_stats(stats)


if __name__ == "__main__":
    unittest.main()
