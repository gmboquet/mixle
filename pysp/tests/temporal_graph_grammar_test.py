"""Temporal graph grammar: contract (sample/score/fit/combine) + motif-distribution recovery."""

import unittest

import numpy as np
import scipy.sparse as sp

import pysp.stats as stats
from pysp.stats.graphs.temporal_graph_grammar import (
    LabeledTemporalGraphGrammarDistribution,
)


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

    def test_add_and_remove_grammars(self):
        # a realistic full-edit grammar: growth favours triadic closure, decay favours bridges
        add_w, rem_w = [0.15, 0.4, 0.3, 0.15], [0.5, 0.25, 0.15, 0.1]
        gt = stats.TemporalGraphGrammarDistribution(
            add_w, edge_rate=4.0, node_rate=0.5, remove_weights=rem_w, edge_remove_rate=2.5
        )
        seqs = [
            gt.sampler(seed=s).sample_one(num_steps=8, seed_graph=_seed_graph(self.rng, n=40, p=0.25))
            for s in range(150)
        ]
        # both grammars fire: there are removed edges somewhere (the chain is not monotone growth)
        any_removed = any(
            np.any((a[: b.shape[0], : b.shape[0]] > 0) & (b == 0))
            if b.shape[0] <= a.shape[0]
            else np.any((a > 0) & (b[: a.shape[0], : a.shape[0]] == 0))
            for seq in seqs
            for a, b in zip(seq, seq[1:])
        )
        self.assertTrue(any_removed)
        self.assertTrue(np.all(np.isfinite(gt.seq_log_density(seqs))))
        est = stats.TemporalGraphGrammarEstimator(pseudo_count=0.5)
        acc = est.accumulator_factory().make()
        acc.seq_update(seqs, np.ones(len(seqs)), None)
        fit = est.estimate(float(len(seqs)), acc.value())
        self.assertLess(float(np.max(np.abs(fit.motif_weights - add_w))), 0.05)  # ADD grammar recovered
        self.assertLess(float(np.max(np.abs(fit.remove_weights - rem_w))), 0.05)  # REMOVE grammar recovered
        self.assertAlmostEqual(fit.edge_remove_rate, 2.5, delta=0.3)

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


class SparseTemporalGraphGrammarTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(2)
        self.gt = stats.TemporalGraphGrammarDistribution(
            [0.2, 0.4, 0.25, 0.15], edge_rate=4.0, remove_weights=[0.5, 0.25, 0.15, 0.1], edge_remove_rate=2.0
        )
        self.dense = [
            self.gt.sampler(seed=s).sample_one(num_steps=6, seed_graph=_seed_graph(self.rng, n=60, p=0.3))
            for s in range(40)
        ]
        self.sparse = [[sp.csr_array(a) for a in seq] for seq in self.dense]

    def test_sparse_scoring_matches_dense(self):
        self.assertTrue(np.allclose(self.gt.seq_log_density(self.dense), self.gt.seq_log_density(self.sparse)))

    def test_sparse_fit_matches_dense(self):
        est = stats.TemporalGraphGrammarEstimator(pseudo_count=0.5)
        ad = est.accumulator_factory().make()
        ad.seq_update(self.dense, np.ones(len(self.dense)), None)
        asp = est.accumulator_factory().make()
        asp.seq_update(self.sparse, np.ones(len(self.sparse)), None)
        fd, fs = est.estimate(40.0, ad.value()), est.estimate(40.0, asp.value())
        self.assertTrue(np.allclose(fd.motif_weights, fs.motif_weights))
        self.assertTrue(np.allclose(fd.remove_weights, fs.remove_weights))

    def test_scales_past_dense(self):
        # a 50k-node sparse graph the dense path (20 GB) can't hold; one transition scores in well under a second
        n, deg = 50_000, 10
        nnz = n * deg // 2
        ii, jj = self.rng.randint(0, n, nnz), self.rng.randint(0, n, nnz)
        a = sp.csr_array((np.ones(len(ii)), (ii, jj)), shape=(n, n))
        a = sp.triu(a, 1)
        a = a + a.T
        a.data[:] = 1.0
        b = a.tolil()
        for x, y in zip(self.rng.randint(0, n, 300), self.rng.randint(0, n, 300)):
            if x != y:
                b[x, y] = b[y, x] = 1
        self.assertTrue(np.isfinite(self.gt.log_density([a, b.tocsr()])))


class LabeledTemporalGraphGrammarTest(unittest.TestCase):
    def test_recovers_structure_and_node_and_edge_attributes(self):
        rng = np.random.RandomState(0)
        struct = stats.TemporalGraphGrammarDistribution([0.2, 0.4, 0.25, 0.15], edge_rate=4.0, node_rate=1.0)
        node_dist = stats.CompositeDistribution(
            (stats.GaussianDistribution(40.0, 9.0), stats.CategoricalDistribution({"NYC": 0.5, "LA": 0.3, "SF": 0.2}))
        )
        edge_dist = stats.PoissonDistribution(6.0)
        gt = LabeledTemporalGraphGrammarDistribution(struct, node_dist, edge_dist)
        obs = [gt.sampler(seed=s).sample_one(num_steps=8, seed_graph=_seed_graph(rng, n=25, p=0.3)) for s in range(120)]
        self.assertTrue(np.all(np.isfinite(gt.seq_log_density(obs))))
        est = gt.estimator(pseudo_count=0.5)
        acc = est.accumulator_factory().make()
        acc.seq_update(obs, np.ones(len(obs)), gt)
        fit = est.estimate(float(len(obs)), acc.value())
        self.assertLess(float(np.max(np.abs(fit.structure.motif_weights - [0.2, 0.4, 0.25, 0.15]))), 0.05)
        self.assertAlmostEqual(fit.node_dist.dists[0].mu, 40.0, delta=0.7)  # node age
        self.assertAlmostEqual(fit.node_dist.dists[1].pmap["NYC"], 0.5, delta=0.05)  # node location
        self.assertAlmostEqual(fit.edge_dist.lam, 6.0, delta=0.3)  # edge communication count


class HomophilyTemporalGraphGrammarTest(unittest.TestCase):
    def test_recovers_homophily_and_types(self):
        rng = np.random.RandomState(0)
        M, K = 4, 3
        base = np.array([[3.0, 0.7, 0.7], [0.7, 3.0, 0.7], [0.7, 0.7, 3.0]])  # same-type ~4x cross-type
        rate = np.stack([base * w for w in (0.2, 0.4, 0.25, 0.15)])
        gt = stats.HomophilyTemporalGraphGrammarDistribution(rate, [0.4, 0.35, 0.25], node_rate=1.0)
        obs = [gt.sampler(seed=s).sample_one(num_steps=8, seed_graph=_seed_graph(rng, n=24, p=0.3)) for s in range(150)]
        self.assertTrue(np.all(np.isfinite(gt.seq_log_density(obs))))
        est = stats.HomophilyTemporalGraphGrammarEstimator(M, K, stats.CommonNeighbourMotif(), pseudo_count=0.5)
        acc = est.accumulator_factory().make()
        acc.seq_update(obs, np.ones(len(obs)), None)
        fit = est.estimate(float(len(obs)), acc.value())
        same = np.mean([fit.rate[:, a, a].sum() for a in range(K)])
        cross = np.mean([fit.rate[:, a, b].sum() for a in range(K) for b in range(K) if a != b])
        self.assertGreater(same, 2.5 * cross)  # homophily recovered (same-type edges form much faster)
        self.assertLess(float(np.max(np.abs(fit.type_weights - [0.4, 0.35, 0.25]))), 0.05)
        # the fitted homophily grammar out-scores a homophily-blind (flat-affinity) one
        flat = np.broadcast_to(fit.rate.mean(axis=(1, 2), keepdims=True), fit.rate.shape).copy()
        blind = stats.HomophilyTemporalGraphGrammarDistribution(flat, fit.type_weights, fit.node_rate)
        self.assertGreater(float(fit.seq_log_density(obs).sum()), float(blind.seq_log_density(obs).sum()))


if __name__ == "__main__":
    unittest.main()
