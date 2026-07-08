import unittest

import numpy as np

import mixle.stats as stats
from mixle.models import ErdosRenyiGraphModel, StochasticBlockGraphModel


class GraphDistributionTestCase(unittest.TestCase):
    def test_erdos_renyi_matches_legacy_model_and_estimates_probability(self):
        adj = np.asarray(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
            ]
        )
        dist = stats.ErdosRenyiGraphDistribution(0.35)
        legacy = ErdosRenyiGraphModel(0.35)

        self.assertAlmostEqual(dist.log_density(adj), legacy.log_likelihood(adj), places=12)

        enc = dist.dist_to_encoder().seq_encode([adj, {"edges": [(0, 1), (1, 2)], "num_nodes": 3}])
        seq_ll = dist.seq_log_density(enc)
        self.assertEqual(seq_ll.shape, (2,))
        self.assertAlmostEqual(seq_ll[0], dist.log_density(adj), places=12)

        acc = stats.ErdosRenyiGraphEstimator().accumulator_factory().make()
        acc.update(adj, 1.0, None)
        fitted = stats.ErdosRenyiGraphEstimator().estimate(1.0, acc.value())
        self.assertAlmostEqual(fitted.p, 3.0 / 6.0, places=12)

        loaded = stats.ErdosRenyiGraphDistribution.from_json(fitted.to_json())
        self.assertIsInstance(loaded, stats.ErdosRenyiGraphDistribution)
        self.assertAlmostEqual(loaded.log_density(adj), fitted.log_density(adj), places=12)

    def test_erdos_renyi_sampler_and_model_bridge(self):
        dist = stats.ErdosRenyiGraphDistribution(0.4, num_nodes=8)
        sample = dist.sampler(seed=1).sample()

        self.assertEqual(sample.shape, (8, 8))
        self.assertTrue(np.all(sample == sample.T))
        self.assertTrue(np.all(np.diag(sample) == 0))

        legacy = dist.to_model()
        round_tripped = stats.ErdosRenyiGraphDistribution.from_model(legacy)
        self.assertAlmostEqual(round_tripped.p, dist.p, places=12)
        self.assertEqual(round_tripped.directed, dist.directed)

    def test_erdos_renyi_enumeration_matches_brute_force(self):
        import itertools

        from mixle.data.sources.graph_source import _edge_indices
        from mixle.enumeration.algorithms import freeze
        from mixle.enumeration.density_rank import density_rank

        def brute(dist, n):
            edges = list(_edge_indices(n, dist.directed, dist.self_loops))
            out = []
            for bits in itertools.product((0, 1), repeat=len(edges)):
                adj = np.zeros((n, n), dtype=np.int8)
                for (i, j), v in zip(edges, bits):
                    adj[i, j] = v
                    if not dist.directed:
                        adj[j, i] = v
                out.append((adj, dist.log_density(adj)))
            out.sort(key=lambda t: -t[1])
            return out

        for directed, self_loops, p, n in [(False, False, 0.3, 4), (True, False, 0.4, 3), (False, True, 0.25, 3)]:
            dist = stats.ErdosRenyiGraphDistribution(p, directed=directed, self_loops=self_loops, num_nodes=n)
            items = list(dist.enumerator())
            b = brute(dist, n)
            self.assertEqual(len(items), len(b))
            np.testing.assert_allclose([lp for _, lp in items], [lp for _, lp in b], atol=1e-9)
            for v, lp in items:
                self.assertAlmostEqual(lp, dist.log_density(v), places=9)
            self.assertEqual(len({freeze(v) for v, _ in items}), len(items))

        # all four capabilities now route through the enumerator (rank uses the exact head)
        dist = stats.ErdosRenyiGraphDistribution(0.3, num_nodes=5)
        dist.enumerator().quantized_index(max_bits=10.0)  # arbitrary-index unranking builds
        r = density_rank(dist, dist.sampler(0).sample(), n_samples=2000, seed=1)
        self.assertEqual(r.method, "exact-head")

    def test_erdos_renyi_enumeration_requires_num_nodes(self):
        from mixle.stats.compute.pdist import EnumerationError

        with self.assertRaises(EnumerationError):
            stats.ErdosRenyiGraphDistribution(0.3).enumerator()

    def test_stochastic_block_enumeration_matches_brute_force(self):
        import itertools

        from mixle.data.sources.graph_source import _edge_indices
        from mixle.stats.compute.pdist import EnumerationError

        bp = np.array([[0.7, 0.2], [0.2, 0.5]])
        assign = [0, 0, 1, 1]
        for directed, self_loops, prior in [(False, False, False), (False, False, True), (True, True, True)]:
            dist = stats.StochasticBlockGraphDistribution(
                bp,
                block_assignments=assign,
                directed=directed,
                self_loops=self_loops,
                include_assignment_prior=prior,
                block_prior=[0.5, 0.5],
            )
            n = len(assign)
            edges = list(_edge_indices(n, directed, self_loops))
            brute = []
            for bits in itertools.product((0, 1), repeat=len(edges)):
                adj = np.zeros((n, n), dtype=np.int8)
                for (i, j), v in zip(edges, bits):
                    adj[i, j] = v
                    if not directed:
                        adj[j, i] = v
                brute.append((adj, dist.log_density(adj)))
            # Every adjacency the enumerator can emit is already present (with its log_density) in
            # `brute`, since both walk the same `_edge_indices` edge set over the full 2**|edges|
            # space. Reuse those already-computed values instead of calling dist.log_density(v) again
            # per item -- log_density is the dominant cost here, and recomputing it for every one of
            # the (up to tens of thousands of) enumerated graphs was pure redundant work.
            brute_by_bytes = {a.tobytes(): lp for a, lp in brute}
            brute.sort(key=lambda t: -t[1])
            items = list(dist.enumerator())
            self.assertEqual(len(items), len(brute))
            np.testing.assert_allclose([lp for _, lp in items], [lp for _, lp in brute], atol=1e-9)
            for v, lp in items:
                self.assertIn(v.tobytes(), brute_by_bytes)
                self.assertAlmostEqual(lp, brute_by_bytes[v.tobytes()], places=9)

        with self.assertRaises(EnumerationError):
            stats.StochasticBlockGraphDistribution(bp).enumerator()  # no fixed assignments

    def test_stochastic_block_matches_legacy_model_and_estimates_probabilities(self):
        assignments = np.asarray([0, 0, 1, 1])
        adj = np.asarray(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
            ]
        )
        block_probs = np.asarray([[0.8, 0.2], [0.2, 0.7]])
        dist = stats.StochasticBlockGraphDistribution(block_probs, assignments)
        legacy = StochasticBlockGraphModel(block_probs, assignments)

        self.assertAlmostEqual(dist.log_density(adj), legacy.log_likelihood(adj), places=12)

        enc = dist.dist_to_encoder().seq_encode([adj, (adj, assignments)])
        seq_ll = dist.seq_log_density(enc)
        self.assertEqual(seq_ll.shape, (2,))
        self.assertAlmostEqual(seq_ll[0], dist.log_density(adj), places=12)

        acc = stats.StochasticBlockGraphEstimator(num_blocks=2).accumulator_factory().make()
        acc.update((adj, assignments), 1.0, None)
        fitted = stats.StochasticBlockGraphEstimator(num_blocks=2).estimate(1.0, acc.value())
        np.testing.assert_allclose(fitted.block_probs, [[1.0, 0.25], [0.25, 1.0]], atol=1.0e-11)
        np.testing.assert_allclose(fitted.block_prior, [0.5, 0.5], atol=1.0e-12)

        loaded = stats.StochasticBlockGraphDistribution.from_json(fitted.to_json())
        self.assertIsInstance(loaded, stats.StochasticBlockGraphDistribution)
        self.assertAlmostEqual(
            loaded.log_density((adj, assignments)), fitted.log_density((adj, assignments)), places=12
        )

    def test_stochastic_block_sampler_bridge_and_prior_predictive_marginals(self):
        block_probs = np.asarray([[0.8, 0.2], [0.2, 0.6]])
        block_prior = np.asarray([0.25, 0.75])
        dist = stats.StochasticBlockGraphDistribution(block_probs, block_prior=block_prior, self_loops=True)

        sample, assignments = dist.sampler(seed=3).sample(num_nodes=6, return_assignments=True)
        self.assertEqual(sample.shape, (6, 6))
        self.assertEqual(assignments.shape, (6,))
        self.assertTrue(np.all(sample == sample.T))
        self.assertTrue(np.all((assignments >= 0) & (assignments < 2)))

        edge_p = float(block_prior @ block_probs @ block_prior)
        loop_p = float(np.sum(block_prior * np.diag(block_probs)))
        marginals = dist.edge_marginals(num_nodes=4)
        np.testing.assert_allclose(marginals[np.triu_indices(4, k=1)], edge_p, atol=1.0e-12)
        np.testing.assert_allclose(np.diag(marginals), loop_p, atol=1.0e-12)
        self.assertAlmostEqual(dist.link_probability(0, 1), edge_p, places=12)
        self.assertAlmostEqual(dist.link_probability(0, 0), loop_p, places=12)

        fixed = stats.StochasticBlockGraphDistribution(block_probs, [0, 0, 1, 1])
        legacy = fixed.to_model()
        round_tripped = stats.StochasticBlockGraphDistribution.from_model(legacy)
        np.testing.assert_allclose(round_tripped.block_probs, fixed.block_probs)
        np.testing.assert_array_equal(round_tripped.block_assignments, fixed.block_assignments)

    def test_stochastic_block_estimator_json_round_trip(self):
        estimator = stats.StochasticBlockGraphEstimator(
            num_blocks=2, pseudo_count=1.0, block_prior=[0.4, 0.6], include_assignment_prior=True
        )
        loaded = stats.StochasticBlockGraphEstimator.from_json(estimator.to_json())

        self.assertIsInstance(loaded, stats.StochasticBlockGraphEstimator)
        self.assertEqual(loaded.num_blocks, 2)
        self.assertEqual(loaded.pseudo_count, 1.0)
        self.assertTrue(loaded.include_assignment_prior)
        np.testing.assert_allclose(loaded.block_prior, [0.4, 0.6])
        self.assertIsNotNone(loaded.accumulator_factory().make())

    def test_graph_distribution_input_validation(self):
        with self.assertRaises(ValueError):
            stats.ErdosRenyiGraphDistribution(1.25)

        block_probs = [[0.8, 0.2], [0.2, 0.6]]
        with self.assertRaises(ValueError):
            stats.StochasticBlockGraphDistribution(block_probs, [0, -1])

        dist = stats.StochasticBlockGraphDistribution(block_probs)
        with self.assertRaises(ValueError):
            dist.link_probability(0, 1, [0, 2])

        with self.assertRaises(ValueError):
            stats.ErdosRenyiGraphDistribution(0.5).log_density({"edges": [(0, 3)], "num_nodes": 3})


if __name__ == "__main__":
    unittest.main()
