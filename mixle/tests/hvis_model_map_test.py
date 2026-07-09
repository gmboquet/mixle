"""The direct compositional layout (mixle.utils.hvis.direct): read the map off the model.

The claims that distinguish it from the neighbor-optimizer path, each pinned: bit-determinism with
NO seed; closed-form out-of-sample placement that reproduces the training coordinates exactly;
nameable within-regime axes (the fiber loadings recover the generative field); model-decided global
arrangement (confusable regimes adjacent); and the complaint fixtures (HMM sequences) rendering as
structured clouds. refine=True only polishes -- the global arrangement stays the skeleton's.
"""

import unittest

import numpy as np

from mixle.stats import (
    CategoricalDistribution,
    CompositeDistribution,
    GaussianDistribution,
    HiddenMarkovModelDistribution,
    IntegerCategoricalDistribution,
    MixtureDistribution,
)
from mixle.utils.hvis import model_map

_MODEL3 = MixtureDistribution(
    [GaussianDistribution(0.0, 1.0), GaussianDistribution(4.0, 1.0), GaussianDistribution(20.0, 1.0)],
    [1.0 / 3.0] * 3,
)


def _data3(n_per=25, seed=0):
    rng = np.random.RandomState(seed)
    xs = np.concatenate([rng.normal(0.0, 1.0, n_per), rng.normal(4.0, 1.0, n_per), rng.normal(20.0, 1.0, n_per)])
    return [float(v) for v in xs], np.repeat([0, 1, 2], n_per)


def _purity(coords, labels):
    d2 = np.square(coords[:, None, :] - coords[None, :, :]).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    return float(np.mean(labels[d2.argmin(axis=1)] == labels))


class DeterminismAndPlacementTest(unittest.TestCase):
    def test_bit_deterministic_with_no_seed_at_all(self):
        data, _ = _data3()
        a = model_map(data, mix_model=_MODEL3)
        b = model_map(data, mix_model=_MODEL3)
        np.testing.assert_array_equal(a.coords, b.coords)
        np.testing.assert_array_equal(a.vertices, b.vertices)

    def test_place_reproduces_training_coordinates_exactly(self):
        data, _ = _data3()
        fitted = model_map(data, mix_model=_MODEL3)
        np.testing.assert_allclose(fitted.place(data), fitted.coords, atol=1.0e-10)

    def test_place_streams_new_points_to_the_right_regime(self):
        data, labels = _data3()
        fitted = model_map(data, mix_model=_MODEL3)
        rng = np.random.RandomState(7)
        new = [float(v) for v in np.concatenate([rng.normal(0, 1, 10), rng.normal(20, 1, 10)])]
        placed = fitted.place(new)
        d2 = np.square(placed[:, None, :] - fitted.coords[None, :, :]).sum(axis=2)
        nearest_labels = labels[d2.argmin(axis=1)]
        self.assertGreater(float(np.mean(nearest_labels == np.repeat([0, 2], 10))), 0.9)

    def test_place_rejects_a_different_field_shape(self):
        data, _ = _data3()
        fitted = model_map(data, mix_model=_MODEL3)
        comp_model = MixtureDistribution(
            [CompositeDistribution((GaussianDistribution(0.0, 1.0), CategoricalDistribution({"a": 1.0})))], [1.0]
        )
        fitted._model = comp_model  # simulate a model/data mismatch at placement time
        with self.assertRaises(ValueError):
            fitted.place([(0.0, "a"), (1.0, "a")])


class GeometryAndInterpretabilityTest(unittest.TestCase):
    def test_confusable_regimes_sit_adjacent_and_clusters_separate(self):
        data, labels = _data3()
        fitted = model_map(data, mix_model=_MODEL3)
        v = fitted.vertices
        d01 = float(np.linalg.norm(v[0] - v[1]))
        self.assertLess(d01, float(np.linalg.norm(v[0] - v[2])))
        self.assertLess(d01, float(np.linalg.norm(v[1] - v[2])))
        self.assertGreater(_purity(fitted.coords, labels), 0.9)

    def test_mixed_membership_point_sits_between_its_regimes(self):
        data, labels = _data3()
        fitted = model_map(data, mix_model=_MODEL3)
        point = fitted.place([2.0])[0]  # posterior ~(0.5, 0.5, 0)
        cents = np.stack([fitted.coords[labels == c].mean(axis=0) for c in (0, 1, 2)])
        d0, d1, d2 = (float(np.linalg.norm(point - cents[c])) for c in (0, 1, 2))
        self.assertLess(max(d0, d1), d2)

    def test_fiber_axes_are_nameable_and_recover_the_generative_field(self):
        # two regimes over (Gaussian value, sharp categorical): within a regime, the leading fiber
        # axis should BE the Gaussian value -- the "axes mean something" claim, checked.
        cat = {"p": 0.5, "q": 0.5}
        comps = [
            CompositeDistribution((GaussianDistribution(-4.0, 1.0), CategoricalDistribution(cat))),
            CompositeDistribution((GaussianDistribution(4.0, 1.0), CategoricalDistribution(cat))),
        ]
        model = MixtureDistribution(comps, [0.5, 0.5])
        data = []
        for k, comp in enumerate(comps):
            data.extend(comp.sampler(seed=k).sample(size=30))
        fitted = model_map(data, mix_model=model)
        self.assertEqual(len(fitted.coord_labels), fitted.loadings[0].shape[0])
        values = np.array([x[0] for x in data[:30]])
        offsets = fitted.coords[:30] - fitted.vertices[0][None, :]  # regime-0 chart offsets
        major = offsets @ fitted.frames[0][0]  # read them in the chart's own frame (row 0 = major axis)
        r = float(np.corrcoef(major, values)[0, 1])
        self.assertGreater(abs(r), 0.9)


class ComplaintFixtureTest(unittest.TestCase):
    def test_hmm_sequences_render_as_structured_clouds(self):
        len_dist = IntegerCategoricalDistribution(18, [1.0 / 9.0] * 9)

        def hmm(trans, emit_a):
            return HiddenMarkovModelDistribution(
                [
                    CategoricalDistribution({"a": emit_a, "b": 1 - emit_a}),
                    CategoricalDistribution({"a": 0.5, "b": 0.5}),
                ],
                [0.5, 0.5],
                trans,
                len_dist=len_dist,
            )

        comps = [hmm([[0.9, 0.1], [0.1, 0.9]], 0.9), hmm([[0.2, 0.8], [0.8, 0.2]], 0.1)]
        model = MixtureDistribution(comps, [0.5, 0.5])
        data, labels = [], []
        for k, comp in enumerate(comps):
            data.extend(comp.sampler(seed=k).sample(size=35))
            labels.extend([k] * 35)
        labels = np.asarray(labels)

        fitted = model_map(data, mix_model=model)
        self.assertTrue(np.all(np.isfinite(fitted.coords)))
        self.assertGreater(_purity(fitted.coords, labels), 0.85)
        cents = np.stack([fitted.coords[labels == c].mean(axis=0) for c in (0, 1)])
        within = np.mean(
            [np.linalg.norm(fitted.coords[labels == c] - cents[i], axis=1).mean() for i, c in enumerate((0, 1))]
        )
        self.assertGreater(float(within), 0.10 * float(np.linalg.norm(cents[0] - cents[1])))


class RefineTest(unittest.TestCase):
    def test_refine_polishes_without_losing_the_global_arrangement(self):
        data, labels = _data3()
        skeleton = model_map(data, mix_model=_MODEL3)
        refined = model_map(data, mix_model=_MODEL3, refine=True, seed=0)

        def pair_dists(coords):
            cents = np.stack([coords[labels == c].mean(axis=0) for c in (0, 1, 2)])
            return {p: float(np.linalg.norm(cents[p[0]] - cents[p[1]])) for p in ((0, 1), (0, 2), (1, 2))}

        self.assertGreaterEqual(_purity(refined.coords, labels), _purity(skeleton.coords, labels) - 0.05)
        # the invariant refine must keep is the ORDERING: the confusable pair stays the closest.
        # A tight ratio pin would be wrong -- the skeleton's disconnected-piece gap is an explicit
        # rendering choice that the affinity-driven polish legitimately renegotiates.
        for dists in (pair_dists(skeleton.coords), pair_dists(refined.coords)):
            self.assertLess(dists[(0, 1)], dists[(0, 2)])
            self.assertLess(dists[(0, 1)], dists[(1, 2)])


if __name__ == "__main__":
    unittest.main()
