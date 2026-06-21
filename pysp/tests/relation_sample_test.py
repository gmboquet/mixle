"""Relation.sample: Gibbs-weighted sampling over enumerated relation members (Phase B)."""

import unittest
from collections import Counter

import numpy as np

from pysp.relations import Assignment


class RelationSampleTest(unittest.TestCase):
    def setUp(self):
        self.cost = np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])
        self.rel = Assignment(self.cost)  # sense="min"
        self.keys = [tuple(s.value) for s in self.rel.enumerator()]

    def test_frequencies_match_gibbs_weights(self):
        """Empirical frequencies must match exp(-objective/T) normalized over all members."""
        objs = np.array([s.objective for s in self.rel.enumerator()])
        t = 1.5
        w = np.exp(-objs / t)
        w /= w.sum()
        draws = self.rel.sample(60000, rng=0, temperature=t)
        cnt = Counter(tuple(d) for d in draws)
        emp = np.array([cnt[k] / 60000 for k in self.keys])
        np.testing.assert_allclose(emp, w, atol=0.01)

    def test_zero_temperature_is_the_optimum(self):
        self.assertEqual(tuple(self.rel.sample(temperature=0.0)), tuple(self.rel.solve().value))

    def test_uniform_ignores_objective(self):
        cnt = Counter(tuple(d) for d in self.rel.sample(60000, rng=1, uniform=True))
        for k in self.keys:
            self.assertAlmostEqual(cnt[k] / 60000, 1 / len(self.keys), delta=0.01)

    def test_truncation_restricts_to_top_k(self):
        used = {tuple(d) for d in self.rel.sample(2000, rng=0, k=2, temperature=5.0)}
        self.assertTrue(used.issubset(set(self.keys[:2])))

    def test_seed_is_deterministic(self):
        a = [tuple(x) for x in self.rel.sample(6, rng=3, temperature=1.5)]
        b = [tuple(x) for x in self.rel.sample(6, rng=3, temperature=1.5)]
        self.assertEqual(a, b)

    def test_single_draw_returns_a_value_not_a_list(self):
        v = self.rel.sample(rng=0)
        self.assertEqual(tuple(v), tuple(v))  # indexable member value
        self.assertEqual(len(v), 3)

    def test_infeasible_relation_raises(self):
        with self.assertRaises(ValueError):
            Assignment(np.full((2, 2), np.inf)).sample()


if __name__ == "__main__":
    unittest.main()
