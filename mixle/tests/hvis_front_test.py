"""The front door (mixle.utils.hvis.front): one call -> coordinates + every receipt.

Also covers the roadmap items it composes: model_fit_health (merged / shattered / clean), the
occlusion invariant (no model overlap => no screen overlap), quadratic charts (mechanics + exact
placement), the merge tree, and zoom with measured alignment.
"""

import unittest

import numpy as np
import pytest

import mixle.utils.hvis as hvis
from mixle.stats import GaussianDistribution, MixtureDistribution
from mixle.utils.hvis import component_tree, fuzzy_nerve, hvis_map, model_fit_health, model_map

_MODEL3 = MixtureDistribution(
    [GaussianDistribution(0.0, 1.0), GaussianDistribution(4.0, 1.0), GaussianDistribution(20.0, 1.0)],
    [1.0 / 3.0] * 3,
)


def _data3(n_per=25, seed=0):
    rng = np.random.RandomState(seed)
    xs = np.concatenate([rng.normal(0.0, 1.0, n_per), rng.normal(4.0, 1.0, n_per), rng.normal(20.0, 1.0, n_per)])
    return [float(v) for v in xs], np.repeat([0, 1, 2], n_per)


class FrontDoorTest(unittest.TestCase):
    def test_one_call_returns_coords_and_all_receipts(self):
        data, _ = _data3()
        m = hvis_map(data, _MODEL3)
        self.assertEqual(m.coords.shape, (75, 2))
        self.assertEqual(m.posterior_entropy.shape, (75,))
        self.assertEqual(m.typicality.shape, (75,))
        self.assertIn("trustworthiness", m.render_health)
        self.assertIn("components", m.fit_health)
        self.assertIn("holes", m.nerve_health)
        self.assertIsInstance(m.summary(), str)
        self.assertIn("findings", m.summary())

    def test_hvis_map_alias_is_the_promised_name(self):
        self.assertIs(hvis.map, hvis_map)

    def test_uncertainty_channels_mean_what_they_claim(self):
        data, _ = _data3()
        data = data + [2.0, 60.0]  # a mixed-membership point and an outlier
        m = hvis_map(data, _MODEL3, health=False)
        self.assertGreater(m.posterior_entropy[75], np.median(m.posterior_entropy[:75]))  # mixed = high entropy
        self.assertLess(m.typicality[76], 0.05)  # the outlier sits in the lowest typicality percentiles

    def test_goals_imply_refine(self):
        from mixle.utils.hvis import Anchor

        data, _ = _data3()
        pins = np.array([[-5.0, 0.0]])
        m = hvis_map(data, _MODEL3, goals=[Anchor([0], pins)], health=False, seed=0, refine_kwargs={"max_its": 150})
        np.testing.assert_array_equal(m.coords[0], pins[0])  # the hard anchor proves the optimizer ran


class FitHealthTest(unittest.TestCase):
    def test_merged_regimes_are_flagged_on_the_right_component(self):
        rng = np.random.RandomState(0)
        data = [float(v) for v in np.concatenate([rng.normal(0, 1, 40), rng.normal(4, 1, 40), rng.normal(20, 1, 40)])]
        underfit = MixtureDistribution(
            [GaussianDistribution(2.0, 5.0), GaussianDistribution(20.0, 1.0)], [2.0 / 3.0, 1.0 / 3.0]
        )  # one wide component covering what the data treats as TWO regimes
        report = model_fit_health(underfit, data)
        self.assertTrue(any("component 0" in d and "merged" in d for d in report["diagnosis"]))
        self.assertGreater(report["components"][0]["merged_separation"], 3.4)

    def test_shattered_duplicates_are_flagged(self):
        rng = np.random.RandomState(1)
        data = [float(v) for v in rng.normal(0, 1, 80)]
        shattered = MixtureDistribution([GaussianDistribution(0.0, 1.0), GaussianDistribution(0.0, 1.0)], [0.5, 0.5])
        report = model_fit_health(shattered, data)
        self.assertTrue(any("near-duplicates" in d for d in report["diagnosis"]))
        self.assertEqual(report["shattered_pairs"][0][0], [0, 1])

    def test_well_specified_model_is_clean(self):
        data, _ = _data3(n_per=40)
        report = model_fit_health(_MODEL3, data)
        self.assertEqual(report["diagnosis"], [])

    def test_holdout_drop_is_reported(self):
        data, _ = _data3(n_per=40)
        rng = np.random.RandomState(5)
        shifted_holdout = [float(v) for v in rng.normal(40.0, 1.0, 30)]  # nothing like the training data
        report = model_fit_health(_MODEL3, data, holdout=shifted_holdout)
        self.assertGreater(report["holdout_drop_nats"], 1.0)
        self.assertTrue(any("held-out" in d for d in report["diagnosis"]))


class OcclusionTest(unittest.TestCase):
    def _radii_and_dists(self, fitted):
        z = fitted.responsibilities
        dominant = z.argmax(axis=1)
        radii = []
        for k in range(z.shape[1]):
            mine = fitted.coords[dominant == k]
            radii.append(
                float(np.percentile(np.linalg.norm(mine - fitted.vertices[k], axis=1), 90)) if len(mine) else 0.0
            )
        return np.asarray(radii)

    def test_no_model_overlap_means_no_screen_overlap(self):
        # big fibers (spread=1.2) + a disconnected far regime: without the occlusion pass, the
        # rendering gap between nerve pieces is smaller than the fiber clouds it separates.
        data, _ = _data3(n_per=30)
        loose = model_map(data, mix_model=_MODEL3, spread=1.2, occlusion=False)
        tight = model_map(data, mix_model=_MODEL3, spread=1.2, occlusion=True)

        def violation(fitted, a, b):
            radii = self._radii_and_dists(fitted)
            dist = float(np.linalg.norm(fitted.vertices[a] - fitted.vertices[b]))
            return dist - 1.05 * (radii[a] + radii[b])

        # component 2 overlaps neither 0 nor 1 in the model: the invariant applies to both pairs
        self.assertLess(min(violation(loose, 0, 2), violation(loose, 1, 2)), 0.0)  # fixture is adversarial
        self.assertGreaterEqual(violation(tight, 0, 2), -1.0e-9)
        self.assertGreaterEqual(violation(tight, 1, 2), -1.0e-9)

    def test_occlusion_preserves_determinism_and_placement(self):
        data, _ = _data3()
        a = model_map(data, mix_model=_MODEL3, spread=1.2)
        b = model_map(data, mix_model=_MODEL3, spread=1.2)
        np.testing.assert_array_equal(a.coords, b.coords)
        np.testing.assert_allclose(a.place(data), a.coords, atol=1.0e-10)


class QuadraticChartTest(unittest.TestCase):
    def test_quadratic_chart_mechanics(self):
        data, labels = _data3()
        fitted = model_map(data, mix_model=_MODEL3, chart="quadratic")
        self.assertEqual(fitted.chart, "quadratic")
        self.assertEqual(fitted.coords.shape, (75, 2))
        self.assertEqual(len(fitted.coord_labels), fitted.loadings[0].shape[0])  # labels track the lift
        np.testing.assert_allclose(fitted.place(data), fitted.coords, atol=1.0e-9)  # still closed-form
        d2 = np.square(fitted.coords[:, None, :] - fitted.coords[None, :, :]).sum(axis=2)
        np.fill_diagonal(d2, np.inf)
        self.assertGreater(float(np.mean(labels[d2.argmin(axis=1)] == labels)), 0.85)

    def test_chart_residuals_are_reported_and_bounded(self):
        data, _ = _data3()
        fitted = model_map(data, mix_model=_MODEL3)
        self.assertEqual(fitted.chart_residuals.shape, (3,))
        self.assertTrue(np.all((fitted.chart_residuals >= 0.0) & (fitted.chart_residuals <= 1.0)))
        # 1-D fibers in a 2-D chart leave nothing behind
        np.testing.assert_allclose(fitted.chart_residuals, 0.0, atol=1.0e-9)

    def test_bad_chart_name_raises(self):
        data, _ = _data3(n_per=5)
        with self.assertRaises(ValueError):
            model_map(data, mix_model=_MODEL3, chart="cubic")


class HierarchyTest(unittest.TestCase):
    _MODEL4 = MixtureDistribution(
        [
            GaussianDistribution(0.0, 1.0),
            GaussianDistribution(3.0, 1.0),
            GaussianDistribution(40.0, 1.0),
            GaussianDistribution(43.0, 1.0),
        ],
        [0.25] * 4,
    )

    def _data4(self, n_per=30, seed=0):
        rng = np.random.RandomState(seed)
        xs = np.concatenate([rng.normal(mu, 1.0, n_per) for mu in (0.0, 3.0, 40.0, 43.0)])
        return [float(v) for v in xs], np.repeat([0, 1, 2, 3], n_per)

    def test_merge_tree_joins_the_adjacent_pairs_first(self):
        data, _ = self._data4()
        from mixle.utils.hvis import _posteriors_and_loglikes

        z, _ = _posteriors_and_loglikes(self._MODEL4, data=data)
        merges = component_tree(fuzzy_nerve(z))
        first_two = [m["merged"] for m in merges[:2]]
        self.assertIn(frozenset({0, 1}), first_two)
        self.assertIn(frozenset({2, 3}), first_two)

    def test_zoom_recharts_a_group_and_measures_alignment(self):
        data, labels = self._data4()
        parent = hvis_map(data, self._MODEL4, health=False)
        child = parent.zoom([0, 1])
        self.assertEqual(child.coords.shape[0], int(np.sum(labels <= 1)))
        self.assertIsNotNone(child.zoom_alignment_rms)
        parent_spread = float(parent.coords[labels <= 1].std())
        self.assertLess(child.zoom_alignment_rms, parent_spread)  # continuity measured, not assumed
        # the child still separates its two regimes
        child_labels = labels[labels <= 1]
        d2 = np.square(child.coords[:, None, :] - child.coords[None, :, :]).sum(axis=2)
        np.fill_diagonal(d2, np.inf)
        self.assertGreaterEqual(float(np.mean(child_labels[d2.argmin(axis=1)] == child_labels)), 0.85)

    def test_zoom_on_too_few_points_raises(self):
        data, _ = _data3(n_per=2)
        parent = hvis_map(data, _MODEL3, health=False)
        with self.assertRaises(ValueError):
            parent.zoom([2])


@pytest.mark.slow
class ScalingSmokeTest(unittest.TestCase):
    def test_twenty_thousand_points_complete(self):
        import time

        rng = np.random.RandomState(0)
        data = [float(v) for v in np.concatenate([rng.normal(0, 1, 10000), rng.normal(8, 1, 10000)])]
        model = MixtureDistribution([GaussianDistribution(0.0, 1.0), GaussianDistribution(8.0, 1.0)], [0.5, 0.5])
        start = time.time()
        m = hvis_map(data, model, health=True)
        elapsed = time.time() - start
        self.assertEqual(m.coords.shape, (20000, 2))
        self.assertLess(elapsed, 300.0)


if __name__ == "__main__":
    unittest.main()
