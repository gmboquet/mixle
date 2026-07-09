"""The fuzzy nerve + map-fidelity receipts (mixle.utils.hvis.topology).

The topology framing made checkable: a ring of overlapping components IS a loop in the data's
topology and the nerve detects it from posteriors alone; a chain has none; a mutually-overlapping
triple is a FILLED cycle, not a hole; a disconnected cover is reported. embedding_health separates
"the layout misrepresents the model" from "the map is fine" -- and says exactly which it audits.
"""

import unittest

import numpy as np

from mixle.stats import DiagonalGaussianDistribution, GaussianDistribution, MixtureDistribution
from mixle.utils.hvis import (
    _posteriors_and_loglikes,
    embedding_health,
    fuzzy_nerve,
    model_map,
    nerve_report,
)


def _ring_fixture(k=8, radius=5.0, n_per=30, seed=0):
    angles = 2.0 * np.pi * np.arange(k) / k
    comps = [DiagonalGaussianDistribution([radius * np.cos(a), radius * np.sin(a)], [1.0, 1.0]) for a in angles]
    model = MixtureDistribution(comps, [1.0 / k] * k)
    rng = np.random.RandomState(seed)
    data = [
        list(np.array([radius * np.cos(a), radius * np.sin(a)]) + rng.normal(0, 1, 2))
        for a in angles
        for _ in range(n_per)
    ]
    z, _ = _posteriors_and_loglikes(model, data=data)
    return z


class FuzzyNerveTest(unittest.TestCase):
    def test_ring_cover_reports_an_unfilled_cycle(self):
        z = _ring_fixture()
        nerve = fuzzy_nerve(z)
        report = nerve_report(nerve)
        self.assertEqual(report["n_components"], 1)
        self.assertEqual(report["n_strong_edges"], 8)  # exactly the ring adjacencies
        self.assertEqual(len(report["holes"]), 1)  # one independent loop, unfilled
        self.assertEqual(sorted(report["holes"][0]), list(range(8)))
        self.assertTrue(any("loop" in d for d in report["diagnosis"]))

    def test_chain_cover_has_no_cycles(self):
        model = MixtureDistribution(
            [GaussianDistribution(0.0, 1.0), GaussianDistribution(3.0, 1.0), GaussianDistribution(6.0, 1.0)],
            [1.0 / 3.0] * 3,
        )
        rng = np.random.RandomState(1)
        data = [float(v) for mu in (0.0, 3.0, 6.0) for v in rng.normal(mu, 1.0, 40)]
        z, _ = _posteriors_and_loglikes(model, data=data)
        report = nerve_report(fuzzy_nerve(z))
        self.assertEqual(report["n_components"], 1)
        self.assertEqual(report["cycles"], [])
        self.assertEqual(report["holes"], [])
        self.assertEqual(report["diagnosis"], [])

    def test_mutually_overlapping_triple_is_a_filled_cycle_not_a_hole(self):
        angles = 2.0 * np.pi * np.arange(3) / 3
        comps = [DiagonalGaussianDistribution([np.cos(a), np.sin(a)], [1.5, 1.5]) for a in angles]
        model = MixtureDistribution(comps, [1.0 / 3.0] * 3)
        rng = np.random.RandomState(2)
        data = [list(rng.normal(0, 1.4, 2)) for _ in range(120)]
        z, _ = _posteriors_and_loglikes(model, data=data)
        report = nerve_report(fuzzy_nerve(z))
        self.assertEqual(len(report["cycles"]), 1)  # the triangle is a cycle of the 1-skeleton...
        self.assertEqual(report["holes"], [])  # ...but its strong 2-simplex fills it: no hole
        self.assertEqual(report["diagnosis"], [])

    def test_disconnected_cover_is_reported(self):
        model = MixtureDistribution(
            [GaussianDistribution(0.0, 1.0), GaussianDistribution(3.0, 1.0), GaussianDistribution(100.0, 1.0)],
            [1.0 / 3.0] * 3,
        )
        rng = np.random.RandomState(3)
        data = [float(v) for mu in (0.0, 3.0, 100.0) for v in rng.normal(mu, 1.0, 40)]
        z, _ = _posteriors_and_loglikes(model, data=data)
        report = nerve_report(fuzzy_nerve(z))
        self.assertEqual(report["n_components"], 2)
        self.assertTrue(any("disconnected" in d for d in report["diagnosis"]))

    def test_ring_cover_renders_as_a_ring_not_a_line(self):
        # the R3 acceptance: the geodesic nerve layout must realize the loop geometrically --
        # vertices near-equidistant from their centroid, every vertex's nearest a true ring
        # neighbor. Bare MDS on the clipped dense distance matrix cannot promise this.
        from mixle.utils.hvis import component_map

        z = _ring_fixture()
        v = component_map(z)
        radial = np.linalg.norm(v - v.mean(axis=0, keepdims=True), axis=1)
        self.assertLess(float(radial.std() / radial.mean()), 0.2)  # a ring, not a smear
        for k in range(8):
            d = np.linalg.norm(v - v[k], axis=1)
            d[k] = np.inf
            self.assertIn(int(np.argmin(d)), ((k - 1) % 8, (k + 1) % 8))  # adjacency survives

    def test_nerve_is_deterministic(self):
        z = _ring_fixture()
        a, b = fuzzy_nerve(z), fuzzy_nerve(z)
        self.assertEqual(a["edges"], b["edges"])
        self.assertEqual(a["triangles"], b["triangles"])


class EmbeddingHealthTest(unittest.TestCase):
    _MODEL3 = MixtureDistribution(
        [GaussianDistribution(0.0, 1.0), GaussianDistribution(4.0, 1.0), GaussianDistribution(20.0, 1.0)],
        [1.0 / 3.0] * 3,
    )

    def _data(self, n_per=25, seed=0):
        rng = np.random.RandomState(seed)
        xs = np.concatenate([rng.normal(0, 1, n_per), rng.normal(4, 1, n_per), rng.normal(20, 1, n_per)])
        return [float(v) for v in xs]

    def test_faithful_map_scores_high_with_empty_diagnosis(self):
        data = self._data()
        fitted = model_map(data, mix_model=self._MODEL3)
        report = embedding_health(fitted.coords, self._MODEL3, data)
        self.assertGreater(report["trustworthiness"], 0.85)
        self.assertGreater(report["continuity"], 0.85)
        self.assertEqual(report["diagnosis"], [])

    def test_scrambled_map_is_flagged(self):
        data = self._data()
        fitted = model_map(data, mix_model=self._MODEL3)
        rng = np.random.RandomState(9)
        scrambled = fitted.coords[rng.permutation(len(data))]
        report = embedding_health(scrambled, self._MODEL3, data)
        self.assertLess(report["trustworthiness"], 0.7)
        self.assertTrue(report["diagnosis"])

    def test_subsampling_caps_cost(self):
        data = self._data(n_per=200, seed=1)
        fitted = model_map(data, mix_model=self._MODEL3)
        report = embedding_health(fitted.coords, self._MODEL3, data, max_rows=100)
        self.assertEqual(report["n_sampled"], 100)
        self.assertEqual(len(report["per_point_trust_penalty"]), 100)


if __name__ == "__main__":
    unittest.main()
