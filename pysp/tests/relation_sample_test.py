"""RelationSampler: Gibbs-weighted sampling over a relation's enumerated members (Phase B)."""

import unittest
from collections import Counter

import numpy as np

from pysp.relations import Assignment, RelationSampler


class RelationSamplerTest(unittest.TestCase):
    def setUp(self):
        self.cost = np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])
        self.rel = Assignment(self.cost)  # sense="min"
        self.keys = [tuple(s.value) for s in self.rel.enumerator()]

    def test_sampler_returns_a_sampler_object(self):
        s = self.rel.sampler(seed=0, temperature=1.5)
        self.assertIsInstance(s, RelationSampler)
        self.assertTrue(hasattr(s, "rng"))  # the sampler owns the RNG, not the relation

    def test_frequencies_match_gibbs_weights(self):
        """Empirical frequencies must match exp(-objective/T) normalized over all members."""
        objs = np.array([s.objective for s in self.rel.enumerator()])
        t = 1.5
        w = np.exp(-objs / t)
        w /= w.sum()
        draws = self.rel.sampler(seed=0, temperature=t).sample(60000)
        cnt = Counter(tuple(d) for d in draws)
        emp = np.array([cnt[k] / 60000 for k in self.keys])
        np.testing.assert_allclose(emp, w, atol=0.01)

    def test_zero_temperature_is_the_optimum(self):
        self.assertEqual(tuple(self.rel.sampler(temperature=0.0).sample()), tuple(self.rel.solve().value))

    def test_uniform_ignores_objective(self):
        cnt = Counter(tuple(d) for d in self.rel.sampler(seed=1, uniform=True).sample(60000))
        for k in self.keys:
            self.assertAlmostEqual(cnt[k] / 60000, 1 / len(self.keys), delta=0.01)

    def test_truncation_restricts_to_top_k(self):
        used = {tuple(d) for d in self.rel.sampler(seed=0, k=2, temperature=5.0).sample(2000)}
        self.assertTrue(used.issubset(set(self.keys[:2])))

    def test_seed_is_deterministic(self):
        a = [tuple(x) for x in self.rel.sampler(seed=3, temperature=1.5).sample(6)]
        b = [tuple(x) for x in self.rel.sampler(seed=3, temperature=1.5).sample(6)]
        self.assertEqual(a, b)

    def test_shared_rng_is_threaded_through(self):
        rng = np.random.RandomState(7)
        s = self.rel.sampler(rng=rng, temperature=1.5)
        self.assertIs(s.rng, rng)

    def test_repeated_draws_reuse_one_enumeration(self):
        s = self.rel.sampler(seed=0, temperature=1.5)
        s.sample(1)
        cached = s._values
        s.sample(1)
        self.assertIs(s._values, cached)  # enumerated once, cached for subsequent draws

    def test_single_draw_returns_a_value_not_a_list(self):
        v = self.rel.sampler(seed=0).sample()
        self.assertEqual(len(v), 3)  # a single member value (the column assignment), not a list of draws

    def test_infeasible_relation_raises(self):
        with self.assertRaises(ValueError):
            Assignment(np.full((2, 2), np.inf)).sampler().sample()


if __name__ == "__main__":
    unittest.main()
