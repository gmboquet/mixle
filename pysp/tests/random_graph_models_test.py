import unittest

import numpy as np

from pysp.models import ErdosRenyiGraphModel, StochasticBlockGraphModel, hard_em_stochastic_block_model


class RandomGraphModelsTestCase(unittest.TestCase):
    def test_erdos_renyi_mle_matches_observed_edge_fraction(self):
        adj = np.asarray(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
            ]
        )
        model = ErdosRenyiGraphModel.fit_mle(adj)

        self.assertAlmostEqual(model.p, 3.0 / 6.0)
        self.assertTrue(np.isfinite(model.log_likelihood(adj)))

    def test_erdos_renyi_sampling_respects_undirected_no_loop_structure(self):
        model = ErdosRenyiGraphModel(0.4)
        sample = model.sample(8, seed=1)

        self.assertEqual(sample.shape, (8, 8))
        self.assertTrue(np.all(sample == sample.T))
        self.assertTrue(np.all(np.diag(sample) == 0))

    def test_stochastic_block_mle_recovers_block_edge_frequencies(self):
        assignments = [0, 0, 1, 1]
        adj = np.asarray(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
            ]
        )
        model = StochasticBlockGraphModel.fit_mle(adj, assignments, num_blocks=2)

        np.testing.assert_allclose(model.block_probs, [[1.0, 0.25], [0.25, 1.0]])
        self.assertGreater(model.log_likelihood(adj), ErdosRenyiGraphModel.fit_mle(adj).log_likelihood(adj))

    def test_stochastic_block_sampling_and_bic(self):
        model = StochasticBlockGraphModel(
            [[0.8, 0.1], [0.1, 0.7]],
            [0, 0, 0, 1, 1, 1],
        )
        sample = model.sample(seed=3)

        self.assertTrue(np.all(sample == sample.T))
        self.assertTrue(np.all(np.diag(sample) == 0))
        self.assertTrue(np.isfinite(model.bic(sample)))

    def test_hard_em_sbm_returns_valid_monotone_history(self):
        truth = StochasticBlockGraphModel(
            [[0.9, 0.05], [0.05, 0.85]],
            [0, 0, 0, 0, 1, 1, 1, 1],
        )
        adj = truth.sample(seed=5)
        result = hard_em_stochastic_block_model(adj, num_blocks=2, max_its=8, restarts=2, seed=6)

        self.assertIsInstance(result.model, StochasticBlockGraphModel)
        self.assertEqual(result.model.block_assignments.shape[0], adj.shape[0])
        self.assertGreaterEqual(len(result.history), 1)
        self.assertTrue(np.all(np.isfinite(result.history)))
        self.assertTrue(np.all(np.diff(result.history) >= -1.0e-9))

    def test_hard_em_sbm_rejects_decreasing_simultaneous_reassignment(self):
        rng = np.random.RandomState(133)
        adj = (rng.rand(6, 6) < rng.uniform(0.05, 0.95)).astype(int)
        adj = np.triu(adj, 1)
        adj = adj + adj.T

        result = hard_em_stochastic_block_model(adj, num_blocks=3, max_its=12, restarts=1, seed=100133)

        self.assertGreaterEqual(len(result.history), 1)
        self.assertTrue(np.all(np.isfinite(result.history)))
        self.assertTrue(np.all(np.diff(result.history) >= -1.0e-9))


if __name__ == "__main__":
    unittest.main()
