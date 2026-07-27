"""Temporal graph grammar: contract (sample/score/fit/combine) + motif-distribution recovery."""

import unittest

import numpy as np
import scipy.sparse as sp

import mixle.stats as stats
from mixle.stats.graphs.temporal_graph_grammar import (
    ApproximateTemporalGraphSample,
    ChurningTemporalGraphGrammarStatistics,
    HomophilyTemporalGraphGrammarStatistics,
    LabeledTemporalGraphGrammarDistribution,
    LatentTemporalGraphGrammarStatistics,
    TemporalGraphGrammarStatistics,
    _edge_diff,
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


class TemporalGraphGrammarContractTest(unittest.TestCase):
    def test_dense_and_sparse_adjacencies_share_one_validated_sample_space(self):
        dist = stats.TemporalGraphGrammarDistribution([1, 1, 1, 1])
        valid = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.assertEqual(dist.log_density([valid]), dist.log_density([sp.csr_array(valid)]))

        invalid = [
            np.array([[0.0, -1.0], [-1.0, 0.0]]),
            np.array([[0.0, np.nan], [np.nan, 0.0]]),
            np.array([[1.0, 0.0], [0.0, 0.0]]),
            np.array([[0.0, 1.0], [0.0, 0.0]]),
            np.zeros((2, 3)),
        ]
        for adjacency in invalid:
            with self.subTest(kind="dense", adjacency=adjacency):
                with self.assertRaises(ValueError):
                    dist.log_density([adjacency])
            with self.subTest(kind="sparse", adjacency=adjacency):
                with self.assertRaises(ValueError):
                    dist.log_density([sp.csr_array(adjacency)])

    def test_motif_partition_and_distribution_parameters_are_owned_and_validated(self):
        for bins in ((), (1,), (0, 0), (0, 1.5), (0, -1), (0, True)):
            with self.subTest(bins=bins):
                with self.assertRaises(ValueError):
                    stats.CommonNeighbourMotif(bins)
        with self.assertRaises(ValueError):
            stats.CommonNeighbourMotif((0, 1), directed=1)

        weights = np.array([1.0, 2.0, 3.0, 4.0])
        dist = stats.TemporalGraphGrammarDistribution(weights, edge_rate=2.0)
        weights[:] = 0.0
        np.testing.assert_allclose(dist.motif_weights, [0.1, 0.2, 0.3, 0.4])
        with self.assertRaises(ValueError):
            dist.motif_weights[0] = 0.5

        for bad_weights in ([0, 0, 0, 0], [-1, 1, 1, 1], [np.nan, 1, 1, 1], [1, 1]):
            with self.subTest(weights=bad_weights):
                with self.assertRaises(ValueError):
                    stats.TemporalGraphGrammarDistribution(bad_weights)
        for keyword in ("edge_rate", "node_rate", "edge_remove_rate"):
            for value in (-1.0, np.nan, np.inf):
                with self.subTest(keyword=keyword, value=value):
                    with self.assertRaises(ValueError):
                        stats.TemporalGraphGrammarDistribution([1, 1, 1, 1], **{keyword: value})
        with self.assertRaises(ValueError):
            stats.TemporalGraphGrammarDistribution(
                [1, 1], motif=stats.CommonNeighbourMotif((0, 1), directed=True)
            )

    def test_exact_finite_candidate_law_is_normalized_and_preserves_zeros(self):
        dist = stats.TemporalGraphGrammarDistribution([1, 0, 0, 0], edge_rate=2.0)
        empty_two = np.zeros((2, 2))
        edge_two = np.array([[0.0, 1.0], [1.0, 0.0]])
        probability = np.exp(dist.log_density([empty_two, empty_two])) + np.exp(
            dist.log_density([empty_two, edge_two])
        )
        self.assertAlmostEqual(probability, 1.0, places=12)
        self.assertAlmostEqual(np.exp(dist.log_density([empty_two, edge_two])), 1.0 - np.exp(-2.0), places=12)

        empty_three = np.zeros((3, 3))
        edges = ((0, 1), (0, 2), (1, 2))
        total = 0.0
        for mask in range(1 << len(edges)):
            current = empty_three.copy()
            for bit, (left, right) in enumerate(edges):
                if mask & (1 << bit):
                    current[left, right] = current[right, left] = 1.0
            total += np.exp(dist.log_density([empty_three, current]))
        self.assertAlmostEqual(total, 1.0, places=12)

        wedge = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        triangle = wedge.copy()
        triangle[0, 2] = triangle[2, 0] = 1.0
        self.assertEqual(dist.log_density([wedge, triangle]), float("-inf"))
        no_growth = stats.TemporalGraphGrammarDistribution([1, 1, 1, 1], node_rate=0.0)
        self.assertEqual(no_growth.log_density([np.zeros((1, 1)), np.zeros((2, 2))]), float("-inf"))

    def test_accumulation_and_estimation_fail_atomically_on_invalid_evidence(self):
        estimator = stats.TemporalGraphGrammarEstimator()
        accumulator = estimator.accumulator_factory().make()
        empty = np.zeros((2, 2))
        edge = np.array([[0.0, 1.0], [1.0, 0.0]])
        accumulator.update([empty, edge], 1.0, None)
        before = accumulator.value()

        with self.assertRaises(ValueError):
            accumulator.seq_update([[empty, edge]], np.ones(2), None)
        with self.assertRaises(ValueError):
            accumulator.seq_update([[empty, edge]], np.array([-1.0]), None)
        with self.assertRaises(ValueError):
            accumulator.update([edge, np.zeros((1, 1))], 1.0, None)
        after = accumulator.value()
        np.testing.assert_array_equal(after.add_counts, before.add_counts)
        self.assertEqual(after.steps, before.steps)

        corrupt = TemporalGraphGrammarStatistics(
            1,
            (0, 1, 2, 3),
            False,
            np.ones(4),
            np.zeros(4),
            -1.0,
            0.0,
            0.0,
            1.0,
        )
        with self.assertRaises(ValueError):
            estimator.estimate(1.0, corrupt)
        with self.assertRaises(ValueError):
            estimator.estimate(0.0, estimator.accumulator_factory().make().value())

    def test_scalable_samples_are_explicitly_typed_approximations(self):
        dist = stats.TemporalGraphGrammarDistribution([1, 1, 1, 1])
        approximate = dist.sampler(seed=1).sample_one_scalable(num_steps=1, n_init=4)
        self.assertIsInstance(approximate, ApproximateTemporalGraphSample)
        self.assertFalse(approximate.receipt.exact)
        with self.assertRaises(ValueError):
            dist.log_density(approximate)
        accumulator = dist.estimator().accumulator_factory().make()
        with self.assertRaises(ValueError):
            accumulator.update(approximate, 1.0, None)
        self.assertTrue(np.isfinite(dist.log_density(approximate.snapshots)))


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

    def test_features_are_bound_to_nodes_and_per_transition_edge_events(self):
        empty = np.zeros((2, 2))
        edge = np.array([[0.0, 1.0], [1.0, 0.0]])
        structure = stats.TemporalGraphGrammarDistribution([1, 0, 0, 0], edge_rate=1.0)
        dist = LabeledTemporalGraphGrammarDistribution(
            structure,
            stats.GaussianDistribution(0.0, 1.0),
            stats.PoissonDistribution(3.0),
        )
        observation = ([empty, edge], [0.0, 1.0], [[3]])
        self.assertTrue(np.isfinite(dist.log_density(observation)))
        self.assertEqual(dist.dist_to_encoder().row_count(dist.seq_encode([observation])), 1)

        invalid = [
            ([empty, edge], [0.0], [[3]]),
            ([empty, edge], [0.0, 1.0, 2.0], [[3]]),
            ([empty, edge], [0.0, 1.0], []),
            ([empty, edge], [0.0, 1.0], [[]]),
            ([empty, edge], [0.0, 1.0], [3]),
        ]
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    dist.log_density(value)

        estimator = dist.estimator()
        accumulator = estimator.accumulator_factory().make()
        accumulator.update(observation, 1.5, dist)
        value = accumulator.value()
        self.assertEqual(value.node_weight, 3.0)
        self.assertEqual(value.edge_weight, 1.5)
        before = accumulator.value()
        with self.assertRaises(ValueError):
            accumulator.seq_update([observation], np.ones(2), dist)
        self.assertEqual(accumulator.value().node_weight, before.node_weight)

    def test_sampler_emits_event_aligned_records_and_cold_initialization_uses_child_contracts(self):
        structure = stats.TemporalGraphGrammarDistribution([1, 1, 1, 1], edge_rate=2.0, node_rate=1.0)
        dist = LabeledTemporalGraphGrammarDistribution(
            structure,
            stats.GaussianDistribution(0.0, 1.0),
            stats.PoissonDistribution(2.0),
        )
        observation = dist.sampler(seed=4).sample_one(num_steps=3, n_init=4)
        snapshots, node_features, edge_groups = observation
        self.assertEqual(len(node_features), snapshots[-1].shape[0])
        self.assertEqual(len(edge_groups), len(snapshots) - 1)
        for previous, current, group in zip(snapshots, snapshots[1:], edge_groups):
            self.assertEqual(len(group), len(_edge_diff(previous, current)[0]))
        accumulator = dist.estimator().accumulator_factory().make()
        accumulator.initialize(observation, 1.0, np.random.RandomState(5))
        self.assertEqual(accumulator.value().node_weight, float(len(node_features)))


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

    def test_unordered_type_rate_law_is_normalized_and_contract_checked(self):
        motif = stats.CommonNeighbourMotif((0,))
        rate = np.array([[[0.0, 2.0], [2.0, 0.0]]])
        dist = stats.HomophilyTemporalGraphGrammarDistribution(rate, [0.5, 0.5], motif=motif)
        rate[:] = 0.0
        self.assertEqual(dist.rate[0, 0, 1], 2.0)
        empty = np.zeros((2, 2))
        edge = np.array([[0.0, 1.0], [1.0, 0.0]])
        types = np.array([0, 1])
        joint_mass = np.exp(dist.log_density(([empty, empty], types))) + np.exp(
            dist.log_density(([empty, edge], types))
        )
        self.assertAlmostEqual(joint_mass, 0.25, places=12)
        self.assertAlmostEqual(
            np.exp(dist.log_density(([empty, edge], types))) / 0.25,
            1.0 - np.exp(-2.0),
            places=12,
        )

        bad_rates = [
            np.ones((2, 2)),
            np.ones((2, 2, 2)),
            np.array([[[0.0, 1.0], [2.0, 0.0]]]),
            np.array([[[0.0, -1.0], [-1.0, 0.0]]]),
            np.array([[[0.0, np.nan], [np.nan, 0.0]]]),
        ]
        for bad_rate in bad_rates:
            with self.subTest(rate=bad_rate):
                with self.assertRaises(ValueError):
                    stats.HomophilyTemporalGraphGrammarDistribution(bad_rate, [0.5, 0.5], motif=motif)
        for bad_weights in ([0.0, 0.0], [-1.0, 2.0], [np.nan, 1.0], []):
            with self.subTest(weights=bad_weights):
                with self.assertRaises(ValueError):
                    stats.HomophilyTemporalGraphGrammarDistribution(
                        np.zeros((1, len(bad_weights), len(bad_weights))),
                        bad_weights,
                        motif=motif,
                    )

        for bad_types in ([0], [0, 2], [0.0, 1.0]):
            with self.subTest(types=bad_types):
                with self.assertRaises(ValueError):
                    dist.log_density(([empty, empty], bad_types))
        zero_cross = stats.HomophilyTemporalGraphGrammarDistribution(
            np.zeros((1, 2, 2)),
            [0.5, 0.5],
            motif=motif,
        )
        self.assertEqual(zero_cross.log_density(([empty, edge], types)), float("-inf"))

    def test_homophily_statistics_are_versioned_validated_and_atomic(self):
        motif = stats.CommonNeighbourMotif((0,))
        estimator = stats.HomophilyTemporalGraphGrammarEstimator(1, 2, motif)
        accumulator = estimator.accumulator_factory().make()
        empty = np.zeros((2, 2))
        observation = ([empty, empty], np.array([0, 1]))
        accumulator.update(observation, 1.0, None)
        before = accumulator.value()
        with self.assertRaises(ValueError):
            accumulator.seq_update([observation], np.ones(2), None)
        np.testing.assert_array_equal(accumulator.value().type_counts, before.type_counts)

        lower_count = np.zeros((1, 2, 2))
        lower_count[0, 1, 0] = 1.0
        corrupt = HomophilyTemporalGraphGrammarStatistics(
            1,
            (0,),
            2,
            lower_count,
            np.ones(2),
            0.0,
            1.0,
        )
        with self.assertRaises(ValueError):
            estimator.estimate(1.0, corrupt)
        with self.assertRaises(ValueError):
            estimator.estimate(0.0, estimator.accumulator_factory().make().value())


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
            gt.sampler(seed=s).sample_one_scalable(
                num_steps=6, seed_edges=self._seed_edges(rng, n=40)
            ).snapshots
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
        approximate = gt.sampler(seed=1).sample_one_scalable(
            num_steps=2, seed_edges=big
        )  # dense would need ~13 GB
        self.assertFalse(approximate.receipt.exact)
        snaps = approximate.snapshots
        self.assertTrue(all(sp.issparse(a) for a in snaps))
        self.assertGreaterEqual(snaps[-1].shape[0], 40_000)

    def test_scalable_directed_emits_asymmetric_sparse(self):
        gt = stats.TemporalGraphGrammarDistribution([0.25] * 4, edge_rate=3.0, node_rate=1.0, directed=True)
        snaps = gt.sampler(seed=0).sample_one_scalable(
            num_steps=4, seed_edges=[(0, 1), (1, 2), (2, 0), (3, 1)]
        ).snapshots
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

    def test_identity_contract_and_finite_population_removal_law(self):
        edit = stats.TemporalGraphGrammarDistribution([1, 1, 1, 1], edge_rate=0.0, node_rate=0.0)
        dist = stats.ChurningTemporalGraphGrammarDistribution(edit, node_remove_rate=2.0)
        one = np.zeros((1, 1))
        empty = np.zeros((0, 0))
        keep = [(one, ["node"]), (one, ["node"])]
        drop = [(one, ["node"]), (empty, [])]
        self.assertAlmostEqual(np.exp(dist.log_density(keep)) + np.exp(dist.log_density(drop)), 1.0, places=12)
        self.assertAlmostEqual(np.exp(dist.log_density(drop)), 1.0 - np.exp(-2.0), places=12)

        invalid = [
            [(np.zeros((2, 2)), ["duplicate", "duplicate"])],
            [(np.zeros((2, 2)), ["only-one"])],
            [(one, ["node"]), (empty, []), (one, ["node"])],
            [(np.array([[1.0]]), ["node"])],
        ]
        for sequence in invalid:
            with self.subTest(sequence=sequence):
                with self.assertRaises(ValueError):
                    dist.log_density(sequence)

    def test_churning_statistics_are_versioned_validated_and_atomic(self):
        edit = stats.TemporalGraphGrammarDistribution([1, 1, 1, 1], edge_rate=0.0, node_rate=0.0)
        dist = stats.ChurningTemporalGraphGrammarDistribution(edit, node_remove_rate=1.0)
        observation = [(np.zeros((1, 1)), [0]), (np.zeros((0, 0)), [])]
        estimator = dist.estimator()
        accumulator = estimator.accumulator_factory().make()
        accumulator.update(observation, 1.0, dist)
        before = accumulator.value()
        with self.assertRaises(ValueError):
            accumulator.seq_update([observation], np.ones(2), dist)
        self.assertEqual(accumulator.value().removed, before.removed)

        corrupt = ChurningTemporalGraphGrammarStatistics(
            1,
            False,
            before.edit,
            -1.0,
            1.0,
        )
        with self.assertRaises(ValueError):
            estimator.estimate(1.0, corrupt)
        with self.assertRaises(ValueError):
            estimator.estimate(0.0, estimator.accumulator_factory().make().value())


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

    def test_regime_laws_validate_dimensions_and_preserve_structural_zeros(self):
        fixed = stats.TemporalGraphGrammarDistribution([1, 1, 1, 1], edge_rate=0.0, edge_remove_rate=0.0)
        removal = stats.TemporalGraphGrammarDistribution(
            [1, 1, 1, 1],
            edge_rate=0.0,
            remove_weights=[1, 1, 1, 1],
            edge_remove_rate=1.0,
        )
        with self.assertRaises(ValueError):
            stats.LatentTemporalGraphGrammarDistribution([])
        with self.assertRaises(ValueError):
            stats.LatentTemporalGraphGrammarDistribution(
                [fixed, stats.TemporalGraphGrammarDistribution([1, 1, 1, 1], directed=True)]
            )

        invalid_initial = ([1.0], [-1.0, 2.0], [np.nan, 1.0], [0.0, 0.0])
        for initial in invalid_initial:
            with self.subTest(initial=initial):
                with self.assertRaises(ValueError):
                    stats.LatentTemporalGraphGrammarDistribution([fixed, removal], initial)
        invalid_transition = (
            [[1.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, -1.0], [0.0, 1.0]],
            [[1.0, np.nan], [0.0, 1.0]],
        )
        for transition in invalid_transition:
            with self.subTest(transition=transition):
                with self.assertRaises(ValueError):
                    stats.LatentTemporalGraphGrammarDistribution(
                        [fixed, removal],
                        [0.5, 0.5],
                        transition,
                    )

        dist = stats.LatentTemporalGraphGrammarDistribution(
            [fixed, removal],
            [1.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        edge = np.array([[0.0, 1.0], [1.0, 0.0]])
        empty = np.zeros((2, 2))
        impossible = [edge, empty]
        self.assertEqual(dist.log_density(impossible), float("-inf"))
        with self.assertRaises(ValueError):
            dist.decode(impossible)

    def test_zero_probability_evidence_is_recorded_and_fails_closed(self):
        fixed = stats.TemporalGraphGrammarDistribution([1, 1, 1, 1], edge_rate=0.0, edge_remove_rate=0.0)
        removal = stats.TemporalGraphGrammarDistribution(
            [1, 1, 1, 1],
            edge_rate=0.0,
            edge_remove_rate=1.0,
        )
        dist = stats.LatentTemporalGraphGrammarDistribution(
            [fixed, removal],
            [1.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        edge = np.array([[0.0, 1.0], [1.0, 0.0]])
        impossible = [edge, np.zeros((2, 2))]
        estimator = dist.estimator(pseudo_count=0.5)
        accumulator = estimator.accumulator_factory().make()
        accumulator.update(impossible, 2.0, dist)
        value = accumulator.value()
        self.assertEqual(value.accepted_weight, 0.0)
        self.assertEqual(value.rejected_weight, 2.0)
        with self.assertRaises(ValueError):
            estimator.estimate(1.0, value)
        with self.assertRaises(ValueError):
            accumulator.seq_update([impossible], np.ones(2), dist)

        corrupt = LatentTemporalGraphGrammarStatistics(
            1,
            (0, 1, 2, 3),
            False,
            2,
            np.ones(2),
            np.zeros((2, 2)),
            value.state_values,
            1.0,
            0.0,
            1.0,
        )
        with self.assertRaises(ValueError):
            estimator.estimate(1.0, corrupt)


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

    def test_attribute_records_are_event_aligned_and_counted_by_effective_regime_weight(self):
        structure_a = stats.TemporalGraphGrammarDistribution([1, 1, 1, 1], edge_rate=1.0, node_rate=1.0)
        structure_b = stats.TemporalGraphGrammarDistribution([1, 1, 1, 1], edge_rate=2.0, node_rate=2.0)
        node_dists = [stats.GaussianDistribution(0.0, 1.0), stats.GaussianDistribution(1.0, 1.0)]
        edge_dists = [stats.PoissonDistribution(2.0), stats.PoissonDistribution(3.0)]
        with self.assertRaises(ValueError):
            stats.LatentAttributedTemporalGraphGrammarDistribution(
                [structure_a, structure_b],
                node_dists[:1],
                edge_dists,
            )

        dist = stats.LatentAttributedTemporalGraphGrammarDistribution(
            [structure_a, structure_b],
            node_dists,
            edge_dists,
            [0.5, 0.5],
            [[0.8, 0.2], [0.2, 0.8]],
        )
        previous = np.zeros((1, 1))
        current = np.array([[0.0, 1.0], [1.0, 0.0]])
        observation = ([previous, current], [[0.5]], [[2]])
        self.assertTrue(np.isfinite(dist.log_density(observation)))
        invalid = [
            ([previous, current], [], [[2]]),
            ([previous, current], [[], [0.5]], [[2]]),
            ([previous, current], [[0.5, 1.5]], [[2]]),
            ([previous, current], [[0.5]], []),
            ([previous, current], [[0.5]], [[]]),
            ([previous, current], [[0.5]], [[2], []]),
        ]
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    dist.log_density(value)

        estimator = dist.estimator(pseudo_count=0.5)
        accumulator = estimator.accumulator_factory().make()
        accumulator.initialize(observation, 2.0, np.random.RandomState(3))
        value = accumulator.value()
        self.assertAlmostEqual(float(value.node_weights.sum()), 2.0)
        self.assertAlmostEqual(float(value.edge_weights.sum()), 2.0)
        self.assertEqual(value.accepted_weight, 2.0)
        with self.assertRaises(ValueError):
            accumulator.seq_initialize([observation], np.ones(2), np.random.RandomState(4))


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
        approximate = gt.sampler(seed=1).sample_one_scalable(num_steps=6, seed_edges=big)
        self.assertFalse(approximate.receipt.exact)
        s = approximate.snapshots
        g = s[-1].toarray()
        self.assertTrue(all(sp.issparse(a) for a in s))
        self.assertFalse(np.array_equal(g, g.T))  # genuinely directed
        self.assertTrue(np.all(np.isfinite(gt.seq_log_density([s]))))  # the directed scorer accepts it
        # the dominant (well-sampled) motifs recover; the cn>=3 motif is starved in a sparse directed graph
        seqs = [
            gt.sampler(seed=k).sample_one_scalable(
                num_steps=6, seed_edges=[(int(rng.randint(150)), int(rng.randint(150))) for _ in range(400)]
            ).snapshots
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

    def test_moment_init_validates_scope_preserves_structure_and_reports_coverage(self):
        from mixle.stats.graphs.temporal_graph_grammar import regime_moment_init

        state = stats.TemporalGraphGrammarDistribution([1, 1, 1, 1], edge_rate=1.0, node_rate=1.0)
        proto = stats.LatentAttributedTemporalGraphGrammarDistribution(
            [state],
            [stats.PoissonDistribution(2.0)],
            [stats.PoissonDistribution(3.0)],
            [1.0],
            [[1.0]],
        )
        previous = np.zeros((1, 1))
        current = np.array([[0.0, 1.0], [1.0, 0.0]])
        observation = ([previous, current], [[2]], [[3]])
        result = regime_moment_init(
            proto.estimator(pseudo_count=0.5),
            proto,
            [observation],
            1,
            seed=7,
            return_receipt=True,
        )
        self.assertIsInstance(result.model, stats.LatentAttributedTemporalGraphGrammarDistribution)
        self.assertEqual(result.receipt.num_transitions, 1)
        self.assertEqual(result.receipt.cluster_counts, (1,))

        structured = ([previous, current], [[{"value": 2.0}]], [[3]])
        with self.assertRaises(ValueError):
            regime_moment_init(proto.estimator(0.5), proto, [structured], 1)
        with self.assertRaises(ValueError):
            regime_moment_init(proto.estimator(0.5), proto, [], 1)
        with self.assertRaises(ValueError):
            regime_moment_init(proto.estimator(0.5), proto, [observation], 2)
        base_proto = stats.LatentTemporalGraphGrammarDistribution([state])
        with self.assertRaises(ValueError):
            regime_moment_init(proto.estimator(0.5), base_proto, [[previous, current]], 1)

    def test_churn_rate_vector_and_finite_population_law_are_exact(self):
        fixed = stats.TemporalGraphGrammarDistribution([1, 1, 1, 1], edge_rate=0.0, node_rate=0.0)
        rates = np.array([2.0])
        dist = stats.LatentChurningTemporalGraphGrammarDistribution(
            [fixed],
            rates,
            [1.0],
            [[1.0]],
        )
        rates[:] = 0.0
        self.assertEqual(dist.node_remove_rates[0], 2.0)
        one = np.zeros((1, 1))
        empty = np.zeros((0, 0))
        keep = [(one, ["node"]), (one, ["node"])]
        drop = [(one, ["node"]), (empty, [])]
        self.assertAlmostEqual(np.exp(dist.log_density(keep)) + np.exp(dist.log_density(drop)), 1.0, places=12)

        for bad_rates in ([], [1.0, 2.0], [-1.0], [np.nan]):
            with self.subTest(rates=bad_rates):
                with self.assertRaises(ValueError):
                    stats.LatentChurningTemporalGraphGrammarDistribution([fixed], bad_rates)
        with self.assertRaises(ValueError):
            dist.log_density([(np.zeros((2, 2)), [1, 1])])

    def test_impossible_latent_churn_is_recorded_and_not_discarded(self):
        fixed = stats.TemporalGraphGrammarDistribution([1, 1, 1, 1], edge_rate=0.0, node_rate=0.0)
        dist = stats.LatentChurningTemporalGraphGrammarDistribution(
            [fixed],
            [0.0],
            [1.0],
            [[1.0]],
        )
        impossible = [(np.zeros((1, 1)), [0]), (np.zeros((0, 0)), [])]
        estimator = dist.estimator(pseudo_count=0.5)
        accumulator = estimator.accumulator_factory().make()
        accumulator.update(impossible, 3.0, dist)
        value = accumulator.value()
        self.assertEqual(value.rejected_weight, 3.0)
        self.assertEqual(value.accepted_weight, 0.0)
        with self.assertRaises(ValueError):
            estimator.estimate(1.0, value)
        with self.assertRaises(ValueError):
            accumulator.seq_update([impossible], np.ones(2), dist)


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

    def test_moment_init_works_on_the_churning_variant(self):
        # Churning snapshots are (adjacency, node_ids) tuples, unlike every other regime-switching
        # variant's bare adjacency snapshots -- _regime_signatures previously crashed with AttributeError
        # (transition_components called directly on the tuple), and regime_moment_init's own
        # acc._accumulate call separately crashed once past that (Churning's _accumulate expects
        # already-identity-aligned tuples, not raw observations, unlike its Latent/Attributed siblings).
        from mixle.stats.graphs.temporal_graph_grammar import regime_moment_init

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
        est = gt.estimator(pseudo_count=0.3)
        init = regime_moment_init(est, gt, data, 2, seed=1)
        self.assertIsInstance(init, stats.LatentChurningTemporalGraphGrammarDistribution)
        self.assertTrue(np.all(np.isfinite(init.seq_log_density(data))))

        cur = init
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
        self.assertGreater(cur.states[s].edge_rate, 4.0)  # stable grows
        self.assertLess(cur.node_remove_rates[s], 1.5)  # stable: slow turnover
        self.assertGreater(cur.node_remove_rates[c], 2.5)  # churn: fast turnover


if __name__ == "__main__":
    unittest.main()
