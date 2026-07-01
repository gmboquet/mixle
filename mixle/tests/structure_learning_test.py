"""Automatic dependency-structure learning (mixle.inference.structure): the tagline, actually delivered.

A CompositeDistribution models heterogeneous fields as independent; on data where they depend, that is badly
wrong. learn_structure must discover the dependency and fit a joint model that beats the independent composite
by a wide margin on held-out data -- and must NOT invent edges where the fields are truly independent.
"""

import unittest

import numpy as np

import mixle.stats as st
from mixle.inference import fit
from mixle.inference.structure import (
    DependencyTreeDistribution,
    dependency_gain,
    learn_structure,
)


def _dependent(seed, n=600):
    """(category -> shifts a real's mean) + an independent count field."""
    r = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        c = "hi" if r.rand() < 0.5 else "lo"
        x = (5.0 if c == "hi" else -5.0) + r.randn()
        k = int(r.poisson(3))
        out.append((c, float(x), k))
    return out


def _independent(seed, n=600):
    r = np.random.RandomState(seed)
    return [(str(r.randint(0, 3)), float(r.randn()), int(r.poisson(3))) for _ in range(n)]


def _ll(model, data):
    return float(np.sum(model.seq_log_density(model.dist_to_encoder().seq_encode(data))))


def _composite(data):
    est = st.CompositeEstimator((st.CategoricalEstimator(), st.GaussianEstimator(), st.PoissonEstimator()))
    return fit(data, est, max_its=30, out=None)


class DependencyGainTest(unittest.TestCase):
    def test_positive_when_dependent(self):
        data = _dependent(0)
        cat = [d[0] for d in data]
        real = [d[1] for d in data]
        gain = dependency_gain(cat, real, st.GaussianEstimator())
        self.assertGreater(gain, 100.0)  # a strong category->real link is worth many nats

    def test_near_zero_when_independent(self):
        data = _independent(1)
        cat = [d[0] for d in data]
        real = [d[1] for d in data]
        gain = dependency_gain(cat, real, st.GaussianEstimator())
        self.assertLess(gain, 20.0)  # BIC penalty keeps a spurious edge from paying off


class LearnStructureTest(unittest.TestCase):
    def test_finds_the_edge_and_beats_composite(self):
        train, test = _dependent(1), _dependent(2)
        model = learn_structure(train)
        self.assertIsInstance(model, DependencyTreeDistribution)
        self.assertIn((0, 1), model.edges())  # category(0) -> real(1)
        # the independent count field is not spuriously attached to anything
        self.assertNotIn(2, [c for _p, c in model.edges()])
        gain = _ll(model, test) - _ll(_composite(train), test)
        self.assertGreater(gain, 300.0)  # dramatically better held-out likelihood

    def test_no_edges_when_independent(self):
        train = _independent(3)
        model = learn_structure(train)
        self.assertEqual(model.edges(), [])  # nothing to model -> falls back to independent marginals

    def test_scores_and_samples(self):
        model = learn_structure(_dependent(4))
        s = model.sampler(0).sample(20)
        self.assertEqual(len(s), 20)
        self.assertEqual(len(s[0]), 3)
        # a drawn record scores finite under the model
        self.assertTrue(np.isfinite(model.log_density(s[0])))
        # the dependency is respected: 'hi' rows sample high reals, 'lo' rows low
        big = model.sampler(1).sample(400)
        hi = np.mean([r[1] for r in big if r[0] == "hi"])
        lo = np.mean([r[1] for r in big if r[0] == "lo"])
        self.assertGreater(hi, lo + 3.0)

    def test_log_density_matches_seq(self):
        model = learn_structure(_dependent(5))
        rows = _dependent(6)[:50]
        seq = model.seq_log_density(model.dist_to_encoder().seq_encode(rows))
        for i, r in enumerate(rows):
            self.assertAlmostEqual(model.log_density(r), float(seq[i]), places=6)


if __name__ == "__main__":
    unittest.main()
