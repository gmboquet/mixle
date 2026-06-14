import unittest

import numpy as np

import pysp.stats as stats
from pysp.models import ErdosRenyiGraphModel, StochasticBlockGraphModel


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
