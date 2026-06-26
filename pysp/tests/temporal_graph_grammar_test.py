"""Temporal graph grammar: contract (sample/score/fit/combine) + motif-distribution recovery."""

import unittest

import numpy as np

import pysp.stats as stats


def _seed_graph(rng, n=30, p=0.45):
    a = (rng.rand(n, n) < p).astype(float)
    a = np.triu(a, 1)
    return a + a.T


class TemporalGraphGrammarTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(1)
        self.true_w = [0.15, 0.35, 0.30, 0.20]
        self.gt = stats.TemporalGraphGrammarDistribution(self.true_w, edge_rate=3.0, node_rate=0.5)
        self.seqs = [
            self.gt.sampler(seed=s).sample_one(num_steps=8, seed_graph=_seed_graph(self.rng)) for s in range(120)
        ]

    def test_sample_shape_and_growth(self):
        seq = self.seqs[0]
        self.assertEqual(len(seq), 9)  # seed + 8 steps
        sizes = [s.shape[0] for s in seq]
        self.assertTrue(all(b >= a for a, b in zip(sizes, sizes[1:])))  # nodes only grow
        # edges only added (each snapshot a supergraph of the previous on shared nodes)
        for a, b in zip(seq, seq[1:]):
            n = a.shape[0]
            self.assertTrue(np.all(b[:n, :n] >= a))

    def test_scores_finite_and_growth_only(self):
        ll = self.gt.seq_log_density(self.gt.dist_to_encoder().seq_encode(self.seqs))
        self.assertTrue(np.all(np.isfinite(ll)))
        # a removal step is impossible under the growth grammar -> -inf
        a = _seed_graph(self.rng, n=10)
        b = a.copy()
        b[np.where(np.triu(a, 1))[0][0], np.where(np.triu(a, 1))[1][0]] = 0.0  # delete one edge
        b = np.triu(b, 1)
        b = b + b.T
        self.assertEqual(self.gt.log_density([a, b]), float("-inf"))

    def test_recovers_motif_distribution(self):
        est = stats.TemporalGraphGrammarEstimator(stats.CommonNeighbourMotif(), pseudo_count=0.5)
        acc = est.accumulator_factory().make()
        acc.seq_update(self.seqs, np.ones(len(self.seqs)), None)
        fit = est.estimate(float(len(self.seqs)), acc.value())
        self.assertLess(float(np.max(np.abs(fit.motif_weights - self.true_w))), 0.07)  # relative weights recover
        self.assertAlmostEqual(fit.node_rate, 0.5, delta=0.15)
        # the fitted grammar scores the data better than a wrong (uniform) motif grammar
        uni = stats.TemporalGraphGrammarDistribution([1, 1, 1, 1], fit.edge_rate, fit.node_rate)
        self.assertGreater(float(fit.seq_log_density(self.seqs).sum()), float(uni.seq_log_density(self.seqs).sum()))

    def test_combine_matches_single_pass(self):
        est = stats.TemporalGraphGrammarEstimator(pseudo_count=0.5)
        full = est.accumulator_factory().make()
        full.seq_update(self.seqs, np.ones(len(self.seqs)), None)
        a1 = est.accumulator_factory().make()
        a1.seq_update(self.seqs[:60], np.ones(60), None)
        a2 = est.accumulator_factory().make()
        a2.seq_update(self.seqs[60:], np.ones(60), None)
        a1.combine(a2.value())
        self.assertTrue(
            np.allclose(est.estimate(120.0, a1.value()).motif_weights, est.estimate(120.0, full.value()).motif_weights)
        )

    def test_custom_motif_partition(self):
        # a coarser {bridge, closes-a-triangle} partition still round-trips. Uses a sparse seed so the
        # bridge motif (cn=0) keeps plentiful anchors as the graph fills (a dense seed starves it -> capping).
        motif = stats.CommonNeighbourMotif(bins=(0, 1))
        gt = stats.TemporalGraphGrammarDistribution([0.3, 0.7], edge_rate=3.0, motif=motif)
        seqs = [
            gt.sampler(seed=s).sample_one(num_steps=6, seed_graph=_seed_graph(self.rng, n=40, p=0.15))
            for s in range(120)
        ]
        self.assertTrue(np.all(np.isfinite(gt.seq_log_density(seqs))))
        est = stats.TemporalGraphGrammarEstimator(motif, pseudo_count=0.5)
        acc = est.accumulator_factory().make()
        acc.seq_update(seqs, np.ones(len(seqs)), None)
        self.assertLess(float(np.max(np.abs(est.estimate(120.0, acc.value()).motif_weights - [0.3, 0.7]))), 0.08)


if __name__ == "__main__":
    unittest.main()
