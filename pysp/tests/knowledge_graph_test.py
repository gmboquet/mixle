"""Tests for the stats-native knowledge-graph embedding distribution (DistMult)."""

import unittest

import numpy as np

from pysp.stats import KnowledgeGraphDistribution, KnowledgeGraphEstimator
from pysp.utils.estimation import optimize


def _kg_truth(nE=60, nR=4, d=12, seed=0):
    rng = np.random.RandomState(seed)
    return rng.normal(0, 1, (nE, d)), rng.normal(0, 1, (nR, d))


def _sample(ent, rel, n, seed):
    rng = np.random.RandomState(seed)
    nE, nR = ent.shape[0], rel.shape[0]
    hs, rs = rng.randint(nE, size=n), rng.randint(nR, size=n)
    return [(int(h), int(r), int(np.argmax(ent @ (ent[h] * rel[r])))) for h, r in zip(hs, rs)]


class KnowledgeGraphTestCase(unittest.TestCase):
    def test_fits_via_optimize_and_recovers_tails(self):
        ent, rel = _kg_truth()
        nE, nR, d = ent.shape[0], rel.shape[0], ent.shape[1]
        train = _sample(ent, rel, 4000, seed=1)
        test = _sample(ent, rel, 1500, seed=2)  # same ground-truth KG, fresh queries
        m = optimize(
            train,
            KnowledgeGraphEstimator(nE, nR, dim=d, seed=1),
            max_its=1,
            rng=np.random.RandomState(0),
            print_iter=99999,
        )
        self.assertIsInstance(m, KnowledgeGraphDistribution)
        acc = np.mean([m.tail_log_posterior(h, r).argmax() == t for h, r, t in test])
        self.assertGreater(acc, 0.4)  # far above the 1/60 chance rate
        ll = m.seq_log_density(m.dist_to_encoder().seq_encode(test)).mean()
        self.assertGreater(ll, -np.log(nE))  # beats the uniform tail distribution

    def test_tail_posterior_normalized(self):
        ent, rel = _kg_truth(seed=1)
        nE, nR, d = ent.shape[0], rel.shape[0], ent.shape[1]
        m = optimize(
            _sample(ent, rel, 1500, seed=3),
            KnowledgeGraphEstimator(nE, nR, dim=d, epochs=20, seed=2),
            max_its=1,
            rng=np.random.RandomState(0),
            print_iter=99999,
        )
        p = np.exp(m.tail_log_posterior(3, 1))
        self.assertAlmostEqual(float(p.sum()), 1.0, places=6)

    def test_sampler_and_encoder(self):
        rng = np.random.RandomState(0)
        m = KnowledgeGraphDistribution(rng.normal(0, 0.3, (10, 4)), rng.normal(0, 0.3, (3, 4)))
        triples = m.sampler(seed=1).sample(5)
        self.assertEqual(len(triples), 5)
        enc = m.dist_to_encoder().seq_encode(triples)
        self.assertEqual(enc.shape, (5, 3))
        self.assertEqual(m.seq_log_density(enc).shape, (5,))


if __name__ == "__main__":
    unittest.main()
