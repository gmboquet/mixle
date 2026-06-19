"""Tests for the stats-native knowledge-graph embedding distribution (DistMult)."""

import unittest

import numpy as np

from pysp.stats import KnowledgeGraphDistribution, KnowledgeGraphEstimator, fit_knowledge_graph_ensemble
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


class KnowledgeGraphRecommendTestCase(unittest.TestCase):
    def setUp(self):
        self.ent, self.rel = _kg_truth()
        self.nE, self.nR, self.d = self.ent.shape[0], self.rel.shape[0], self.ent.shape[1]
        self.train = _sample(self.ent, self.rel, 4000, seed=1)
        self.m = optimize(
            self.train,
            KnowledgeGraphEstimator(self.nE, self.nR, dim=self.d, seed=1),
            max_its=1,
            rng=np.random.RandomState(0),
            print_iter=10**9,
        )

    def test_any_slot_completion_recovers_truth(self):
        sample = self.train[:400]
        tail_acc = np.mean([self.m.complete(h=h, r=r).argmax() == t for h, r, t in sample])
        head_acc = np.mean([self.m.complete(r=r, t=t).argmax() == h for h, r, t in sample])
        self.assertGreater(tail_acc, 0.4)  # both directions recover the truth well above chance
        self.assertGreater(head_acc, 0.3)  # head completion works too (genuinely harder than tails here)
        self.assertEqual(len(self.m.complete(h=self.train[0][0], t=self.train[0][2])), self.nR)
        with self.assertRaises(ValueError):
            self.m.complete(h=self.train[0][0])  # two slots missing

    def test_recommend_excludes_known_and_ranks(self):
        recs = self.m.recommend(self.train, top_n=8)
        known = {(int(a), int(b), int(c)) for a, b, c in self.train}
        self.assertTrue(all((a, b, c) not in known for a, b, c, _ in recs))
        scores = [s for *_, s in recs]
        self.assertEqual(scores, sorted(scores, reverse=True))  # ranked best first

    def test_subgraph_edges_touch_node(self):
        sg = self.m.recommend_subgraph(0, self.train, top_n=5)
        self.assertTrue(all(0 in (a, c) for a, b, c, _ in sg))

    def test_ensemble_epistemic_uncertainty(self):
        ens = fit_knowledge_graph_ensemble(
            self.train, self.nE, self.nR, dim=self.d, members=3, epochs=30, rng=np.random.RandomState(0)
        )
        u = [ens.epistemic_tail_uncertainty(h, r) for h, r, _ in self.train[:100]]
        self.assertGreaterEqual(min(u), 0.0)  # mutual information is nonnegative
        self.assertGreater(max(u), 0.0)  # members disagree somewhere
        self.assertEqual(len(ens.mean_tail_posterior(0, 0)), self.nE)


if __name__ == "__main__":
    unittest.main()
