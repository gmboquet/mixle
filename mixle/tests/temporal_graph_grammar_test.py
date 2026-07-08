"""Temporal graph grammar: contract (sample/score/fit/combine) + motif-distribution recovery."""

import unittest

import numpy as np
import scipy.sparse as sp

import mixle.stats as stats
from mixle.stats.graphs.temporal_graph_grammar import (
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


class DirectedTemporalGraphGrammarTest(unittest.TestCase):
    def _dseed(self, rng, n=40, p=0.2):
        a = (rng.rand(n, n) < p).astype(float)
        np.fill_diagonal(a, 0.0)
        return a  # asymmetric -> a genuine directed graph

    def test_directed_round_trip_and_sparse_parity(self):
        rng = np.random.RandomState(3)
        gt = stats.TemporalGraphGrammarDistribution(
            [0.2, 0.4, 0.25, 0.15],
            edge_rate=4.0,
            node_rate=0.5,
            remove_weights=[0.5, 0.25, 0.15, 0.1],
            edge_remove_rate=2.0,
            directed=True,
        )
        self.assertTrue(gt.directed and gt.motif.directed)
        seqs = [gt.sampler(seed=s).sample_one(num_steps=6, seed_graph=self._dseed(rng)) for s in range(60)]
        g = seqs[0][-1]
        self.assertFalse(np.array_equal(g, g.T))  # genuinely directed (A != A.T)
        self.assertTrue(np.all(np.isfinite(gt.seq_log_density(seqs))))
        sparse = [[sp.csr_array(a) for a in seq] for seq in seqs]
        self.assertTrue(np.allclose(gt.seq_log_density(seqs), gt.seq_log_density(sparse)))  # directed sparse parity
        est = stats.TemporalGraphGrammarEstimator(stats.CommonNeighbourMotif(directed=True), pseudo_count=0.5)
        acc = est.accumulator_factory().make()
        acc.seq_update(seqs, np.ones(len(seqs)), None)
        fit = est.estimate(float(len(seqs)), acc.value())
        self.assertTrue(fit.directed)
        self.assertLess(float(np.max(np.abs(fit.motif_weights - [0.2, 0.4, 0.25, 0.15]))), 0.06)  # ADD recovered
        self.assertLess(float(np.max(np.abs(fit.remove_weights - [0.5, 0.25, 0.15, 0.1]))), 0.06)  # REMOVE recovered

    def test_directed_labeled_composes_with_weighted_edges(self):
        # directed structure + a weighted-edge (Poisson volume) attribute = a directed, weighted, labeled graph
        rng = np.random.RandomState(1)
        struct = stats.TemporalGraphGrammarDistribution([0.3, 0.3, 0.2, 0.2], edge_rate=3.0, directed=True)
        gt = LabeledTemporalGraphGrammarDistribution(struct, edge_dist=stats.PoissonDistribution(5.0))
        obs = [gt.sampler(seed=s).sample_one(num_steps=5, seed_graph=self._dseed(rng, n=30)) for s in range(80)]
        self.assertTrue(np.all(np.isfinite(gt.seq_log_density(obs))))
        est = gt.estimator()
        acc = est.accumulator_factory().make()
        acc.seq_update(obs, np.ones(len(obs)), gt)
        fit = est.estimate(float(len(obs)), acc.value())
        self.assertTrue(fit.structure.directed)
        self.assertAlmostEqual(fit.edge_dist.lam, 5.0, delta=0.4)  # weighted-edge volume recovered


class ScalableSamplerTest(unittest.TestCase):
    def _seed_edges(self, rng, n=60, p=0.12):
        a = np.triu((rng.rand(n, n) < p), 1)
        ii, jj = np.where(a)
        return list(zip(ii.tolist(), jj.tolist()))

    def test_scalable_sampler_is_consistent_with_scorer(self):
        rng = np.random.RandomState(7)
        gt = stats.TemporalGraphGrammarDistribution(
            [0.3, 0.35, 0.2, 0.15],
            edge_rate=6.0,
            node_rate=1.0,
            remove_weights=[0.4, 0.3, 0.2, 0.1],
            edge_remove_rate=2.0,
        )
        seqs = [
            gt.sampler(seed=s).sample_one_scalable(num_steps=6, seed_edges=self._seed_edges(rng, n=40))
            for s in range(60)
        ]
        self.assertTrue(all(sp.issparse(a) for seq in seqs for a in seq))  # never densified
        self.assertTrue(np.all(np.isfinite(gt.seq_log_density(seqs))))
        est = stats.TemporalGraphGrammarEstimator(pseudo_count=0.5)
        acc = est.accumulator_factory().make()
        acc.seq_update(seqs, np.ones(len(seqs)), None)
        fit = est.estimate(float(len(seqs)), acc.value())
        self.assertLess(float(np.max(np.abs(fit.motif_weights - [0.3, 0.35, 0.2, 0.15]))), 0.05)  # sampler == scorer
        self.assertLess(float(np.max(np.abs(fit.remove_weights - [0.4, 0.3, 0.2, 0.1]))), 0.05)

    def test_scalable_sampler_handles_large_graph(self):
        rng = np.random.RandomState(0)
        gt = stats.TemporalGraphGrammarDistribution([0.4, 0.3, 0.2, 0.1], edge_rate=5.0, node_rate=1.0)
        big = [(int(rng.randint(40_000)), int(rng.randint(40_000))) for _ in range(120_000)]
        big = [(i, j) for i, j in big if i != j]
        snaps = gt.sampler(seed=1).sample_one_scalable(num_steps=2, seed_edges=big)  # dense would need ~13 GB
        self.assertTrue(all(sp.issparse(a) for a in snaps))
        self.assertGreaterEqual(snaps[-1].shape[0], 40_000)

    def test_scalable_directed_emits_asymmetric_sparse(self):
        gt = stats.TemporalGraphGrammarDistribution([0.25] * 4, edge_rate=3.0, node_rate=1.0, directed=True)
        snaps = gt.sampler(seed=0).sample_one_scalable(num_steps=4, seed_edges=[(0, 1), (1, 2), (2, 0), (3, 1)])
        self.assertTrue(all(sp.issparse(a) for a in snaps))
        g = snaps[-1].toarray()
        self.assertFalse(np.array_equal(g, g.T))  # directed: asymmetric adjacency


class ChurningTemporalGraphGrammarTest(unittest.TestCase):
    def test_nodes_leave_and_rate_recovers(self):
        rng = np.random.RandomState(5)
        edit = stats.TemporalGraphGrammarDistribution(
            [0.2, 0.4, 0.25, 0.15],
            edge_rate=5.0,
            node_rate=3.0,
            remove_weights=[0.5, 0.25, 0.15, 0.1],
            edge_remove_rate=1.5,
        )
        gt = stats.ChurningTemporalGraphGrammarDistribution(edit, node_remove_rate=2.0)
        obs = [gt.sampler(seed=s).sample_one(num_steps=8, seed_graph=_seed_graph(rng, n=30, p=0.3)) for s in range(120)]
        # nodes genuinely leave: some id present in one snapshot is gone in the next
        left = any(set(o[t - 1][1]) - set(o[t][1]) for o in obs for t in range(1, len(o)))
        self.assertTrue(left)
        self.assertTrue(np.all(np.isfinite(gt.seq_log_density(obs))))
        est = gt.estimator(pseudo_count=0.5)
        acc = est.accumulator_factory().make()
        acc.seq_update(obs, np.ones(len(obs)), gt)
        fit = est.estimate(float(len(obs)), acc.value())
        self.assertAlmostEqual(fit.node_remove_rate, 2.0, delta=0.2)  # node-removal rate recovered
        self.assertLess(float(np.max(np.abs(fit.edit_grammar.motif_weights - [0.2, 0.4, 0.25, 0.15]))), 0.05)
        self.assertLess(float(np.max(np.abs(fit.edit_grammar.remove_weights - [0.5, 0.25, 0.15, 0.1]))), 0.05)

    def test_removed_node_edges_not_charged_as_edge_removals(self):
        # dropping a node removes its incident edges -- those must NOT be scored as edge-grammar deletions
        # (a pure-growth edit grammar with node churn still has finite likelihood)
        rng = np.random.RandomState(0)
        edit = stats.TemporalGraphGrammarDistribution([0.25] * 4, edge_rate=4.0, node_rate=2.0)  # growth-only edges
        gt = stats.ChurningTemporalGraphGrammarDistribution(edit, node_remove_rate=2.0)
        obs = [gt.sampler(seed=s).sample_one(num_steps=6, seed_graph=_seed_graph(rng, n=25, p=0.3)) for s in range(40)]
        self.assertTrue(np.all(np.isfinite(gt.seq_log_density(obs))))


class LatentRegimeTemporalGraphGrammarTest(unittest.TestCase):
    def test_recovers_regimes_transition_and_beats_single_grammar(self):
        rng = np.random.RandomState(0)
        growth = stats.TemporalGraphGrammarDistribution(
            [0.1, 0.3, 0.35, 0.25], edge_rate=8.0, node_rate=1.0, edge_remove_rate=0.0
        )
        decay = stats.TemporalGraphGrammarDistribution(
            [0.25] * 4, edge_rate=1.0, node_rate=0.0, remove_weights=[0.4, 0.3, 0.2, 0.1], edge_remove_rate=6.0
        )
        A = [[0.85, 0.15], [0.15, 0.85]]
        gt = stats.LatentTemporalGraphGrammarDistribution([growth, decay], [0.5, 0.5], A)
        data = [gt.sampler(seed=s).sample_one(num_steps=8, seed_graph=_seed_graph(rng, n=30, p=0.3)) for s in range(40)]
        self.assertTrue(np.all(np.isfinite(gt.seq_log_density(data))))
        # single-grammar baseline
        se = stats.TemporalGraphGrammarEstimator(pseudo_count=0.5)
        sa = se.accumulator_factory().make()
        sa.seq_update(data, np.ones(len(data)), None)
        single = se.estimate(len(data), sa.value())
        # EM
        est = stats.LatentTemporalGraphGrammarEstimator([growth.estimator(0.2), decay.estimator(0.2)], pseudo_count=0.5)
        acc = est.accumulator_factory().make()
        acc.seq_initialize(data, np.ones(len(data)), np.random.RandomState(1))
        cur = est.estimate(len(data), acc.value())
        prev_ll = -np.inf
        for _ in range(6):
            acc = est.accumulator_factory().make()
            acc.seq_update(data, np.ones(len(data)), cur)
            cur = est.estimate(len(data), acc.value())
            ll = float(cur.seq_log_density(data).sum())
            self.assertGreaterEqual(ll, prev_ll - 1.0)  # EM does not decrease the likelihood
            prev_ll = ll
        order = np.argsort([s.edge_rate for s in cur.states])
        lo, hi = cur.states[order[0]], cur.states[order[1]]
        self.assertLess(lo.edge_rate, 3.0)  # decay regime
        self.assertGreater(hi.edge_rate, 5.0)  # growth regime
        self.assertGreater(lo.edge_remove_rate, hi.edge_remove_rate)  # decay removes, growth doesn't
        self.assertGreater(np.min(np.diag(cur.transition_matrix)), 0.6)  # regimes persist
        self.assertGreater(ll, float(single.seq_log_density(data).sum()))  # latent beats one grammar
        self.assertEqual(len(cur.decode(data[0])), len(data[0]) - 1)  # Viterbi labels every transition


class RegimeSwitchingAttributesTest(unittest.TestCase):
    def test_regime_drives_structure_and_node_and_edge_attributes(self):
        rng = np.random.RandomState(0)
        active = stats.TemporalGraphGrammarDistribution([0.1, 0.3, 0.35, 0.25], edge_rate=8.0, node_rate=2.0)
        quiet = stats.TemporalGraphGrammarDistribution(
            [0.25] * 4, edge_rate=1.0, node_rate=2.0, remove_weights=[0.4, 0.3, 0.2, 0.1], edge_remove_rate=5.0
        )
        gt = stats.LatentAttributedTemporalGraphGrammarDistribution(
            [active, quiet],
            [stats.GaussianDistribution(25.0, 16.0), stats.GaussianDistribution(55.0, 16.0)],  # young vs old
            [stats.PoissonDistribution(10.0), stats.PoissonDistribution(2.0)],  # chatty vs quiet edges
            [0.5, 0.5],
            [[0.85, 0.15], [0.15, 0.85]],
        )
        data = [gt.sampler(seed=s).sample_one(num_steps=8, seed_graph=_seed_graph(rng, n=30, p=0.3)) for s in range(35)]
        self.assertTrue(np.all(np.isfinite(gt.seq_log_density(data))))
        est = gt.estimator(pseudo_count=0.3)
        acc = est.accumulator_factory().make()
        acc.seq_initialize(data, np.ones(len(data)), np.random.RandomState(2))
        cur = est.estimate(len(data), acc.value())
        prev_ll = -np.inf
        for _ in range(7):
            acc = est.accumulator_factory().make()
            acc.seq_update(data, np.ones(len(data)), cur)
            cur = est.estimate(len(data), acc.value())
            ll = float(cur.seq_log_density(data).sum())
            self.assertGreaterEqual(ll, prev_ll - 1.0)  # EM monotone
            prev_ll = ll
        order = np.argsort([s.edge_rate for s in cur.structures])
        q, a = order[0], order[1]
        # the regime jointly drives structure AND both attribute streams
        self.assertGreater(cur.structures[a].edge_rate, 5.0)  # active densifies
        self.assertLess(cur.structures[q].edge_rate, 3.0)  # quiet doesn't
        self.assertLess(cur.node_dists[a].mu, cur.node_dists[q].mu)  # active nodes younger
        self.assertGreater(cur.edge_dists[a].lam, cur.edge_dists[q].lam)  # active edges chattier
        self.assertAlmostEqual(cur.node_dists[a].mu, 25.0, delta=4.0)
        self.assertAlmostEqual(cur.edge_dists[a].lam, 10.0, delta=2.0)
        self.assertEqual(len(cur.decode(data[0])), len(data[0][0]) - 1)  # Viterbi labels every transition


class GraphGrammarClosuresTest(unittest.TestCase):
    def test_directed_scalable_sampling(self):
        rng = np.random.RandomState(0)
        gt = stats.TemporalGraphGrammarDistribution(
            [0.3, 0.35, 0.2, 0.15],
            edge_rate=6.0,
            node_rate=1.0,
            remove_weights=[0.4, 0.3, 0.2, 0.1],
            edge_remove_rate=2.0,
            directed=True,
        )
        big = [(int(rng.randint(2000)), int(rng.randint(2000))) for _ in range(8000)]
        big = [(i, j) for i, j in big if i != j]
        s = gt.sampler(seed=1).sample_one_scalable(num_steps=6, seed_edges=big)
        g = s[-1].toarray()
        self.assertTrue(all(sp.issparse(a) for a in s))
        self.assertFalse(np.array_equal(g, g.T))  # genuinely directed
        self.assertTrue(np.all(np.isfinite(gt.seq_log_density([s]))))  # the directed scorer accepts it
        # the dominant (well-sampled) motifs recover; the cn>=3 motif is starved in a sparse directed graph
        seqs = [
            gt.sampler(seed=k).sample_one_scalable(
                num_steps=6, seed_edges=[(int(rng.randint(150)), int(rng.randint(150))) for _ in range(400)]
            )
            for k in range(80)
        ]
        est = stats.TemporalGraphGrammarEstimator(stats.CommonNeighbourMotif(directed=True), pseudo_count=0.5)
        acc = est.accumulator_factory().make()
        acc.seq_update(seqs, np.ones(len(seqs)), None)
        fit = est.estimate(len(seqs), acc.value())
        self.assertLess(float(np.max(np.abs(fit.motif_weights[:3] - [0.3, 0.35, 0.2]))), 0.08)

    def test_sparse_path_churn(self):
        rng = np.random.RandomState(0)
        edit = stats.TemporalGraphGrammarDistribution(
            [0.2, 0.4, 0.25, 0.15],
            edge_rate=5.0,
            node_rate=3.0,
            remove_weights=[0.5, 0.25, 0.15, 0.1],
            edge_remove_rate=1.5,
        )
        ch = stats.ChurningTemporalGraphGrammarDistribution(edit, node_remove_rate=2.0)
        dense = [
            ch.sampler(seed=s).sample_one(num_steps=8, seed_graph=_seed_graph(rng, n=30, p=0.3)) for s in range(50)
        ]
        sparse = [[(sp.csr_array(adj), ids) for adj, ids in obs] for obs in dense]
        self.assertTrue(np.allclose(ch.seq_log_density(dense), ch.seq_log_density(sparse)))  # dense==sparse churn
        ed = ch.estimator(0.5)
        ad = ed.accumulator_factory().make()
        ad.seq_update(dense, np.ones(len(dense)), ch)
        asp = ed.accumulator_factory().make()
        asp.seq_update(sparse, np.ones(len(sparse)), ch)
        fd, fs = ed.estimate(len(dense), ad.value()), ed.estimate(len(sparse), asp.value())
        self.assertAlmostEqual(fd.node_remove_rate, fs.node_remove_rate)
        self.assertTrue(np.allclose(fd.edit_grammar.motif_weights, fs.edit_grammar.motif_weights))


class LatentChurningTemporalGraphGrammarTest(unittest.TestCase):
    def test_regime_switches_turnover_and_grammar(self):
        rng = np.random.RandomState(0)
        stable = stats.TemporalGraphGrammarDistribution([0.1, 0.3, 0.35, 0.25], edge_rate=7.0, node_rate=3.0)
        churn = stats.TemporalGraphGrammarDistribution(
            [0.25] * 4, edge_rate=1.0, node_rate=3.0, remove_weights=[0.4, 0.3, 0.2, 0.1], edge_remove_rate=4.0
        )
        gt = stats.LatentChurningTemporalGraphGrammarDistribution(
            [stable, churn],
            node_remove_rates=[0.3, 4.0],
            initial_probs=[0.5, 0.5],
            transition_matrix=[[0.85, 0.15], [0.15, 0.85]],
        )
        data = [gt.sampler(seed=s).sample_one(num_steps=8, seed_graph=_seed_graph(rng, n=30, p=0.3)) for s in range(35)]
        # nodes genuinely leave (counts swing) and ids disappear
        self.assertTrue(any(set(o[t - 1][1]) - set(o[t][1]) for o in data for t in range(1, len(o))))
        self.assertTrue(np.all(np.isfinite(gt.seq_log_density(data))))
        est = gt.estimator(pseudo_count=0.3)
        acc = est.accumulator_factory().make()
        acc.seq_initialize(data, np.ones(len(data)), np.random.RandomState(3))
        cur = est.estimate(len(data), acc.value())
        prev_ll = -np.inf
        for _ in range(7):
            acc = est.accumulator_factory().make()
            acc.seq_update(data, np.ones(len(data)), cur)
            cur = est.estimate(len(data), acc.value())
            ll = float(cur.seq_log_density(data).sum())
            self.assertGreaterEqual(ll, prev_ll - 1.0)  # EM monotone
            prev_ll = ll
        order = np.argsort([s.edge_rate for s in cur.states])
        c, s = order[0], order[1]  # churn, stable
        # the regime jointly switches the grammar AND the node-turnover rate
        self.assertGreater(cur.states[s].edge_rate, 4.0)  # stable grows
        self.assertLess(cur.node_remove_rates[s], 1.5)  # stable: slow turnover
        self.assertGreater(cur.node_remove_rates[c], 2.5)  # churn: fast turnover
        self.assertEqual(len(cur.decode(data[0])), len(data[0]) - 1)


class RegimeMomentInitTest(unittest.TestCase):
    def test_moment_init_seeds_recoverable_em(self):
        from mixle.stats.graphs.temporal_graph_grammar import regime_moment_init

        rng = np.random.RandomState(0)
        a = stats.TemporalGraphGrammarDistribution([0.15, 0.3, 0.35, 0.2], edge_rate=4.0, node_rate=2.0)
        b = stats.TemporalGraphGrammarDistribution([0.2, 0.3, 0.3, 0.2], edge_rate=3.0, node_rate=2.0)
        gt = stats.LatentAttributedTemporalGraphGrammarDistribution(
            [a, b],
            [stats.GaussianDistribution(20.0, 9.0), stats.GaussianDistribution(50.0, 9.0)],
            [stats.PoissonDistribution(9.0), stats.PoissonDistribution(2.0)],
            [0.5, 0.5],
            [[0.85, 0.15], [0.15, 0.85]],
        )
        data = [
            gt.sampler(seed=s).sample_one(num_steps=14, seed_graph=_seed_graph(rng, n=28, p=0.3)) for s in range(60)
        ]
        est = gt.estimator(0.3)
        init = regime_moment_init(est, gt, data, 2, seed=1)  # signature-clustering seed (no random restarts)
        self.assertIsInstance(init, stats.LatentAttributedTemporalGraphGrammarDistribution)
        cur = init
        for _ in range(10):
            acc = est.accumulator_factory().make()
            acc.seq_update(data, np.ones(len(data)), cur)
            cur = est.estimate(len(data), acc.value())
        ages = sorted(d.mu for d in cur.node_dists)
        self.assertAlmostEqual(ages[0], 20.0, delta=4.0)  # the two attribute regimes are recovered from the seed
        self.assertAlmostEqual(ages[1], 50.0, delta=4.0)


if __name__ == "__main__":
    unittest.main()
