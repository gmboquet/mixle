import unittest

import numpy as np

from mixle.models import ErdosRenyiGraphModel, StochasticBlockGraphModel, hard_em_stochastic_block_model


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

    def test_erdos_renyi_log_likelihood_rejects_asymmetric_adjacency_when_undirected(self):
        adj = np.asarray(
            [
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
            ]
        )
        model = ErdosRenyiGraphModel(0.3, directed=False, self_loops=False)

        with self.assertRaises(ValueError):
            model.log_likelihood(adj)

    def test_erdos_renyi_log_likelihood_rejects_self_loop_when_disallowed(self):
        adj = np.asarray(
            [
                [1, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
            ]
        )
        model = ErdosRenyiGraphModel(0.3, directed=False, self_loops=False)

        with self.assertRaises(ValueError):
            model.log_likelihood(adj)

    def test_erdos_renyi_log_likelihood_allows_asymmetric_adjacency_when_directed(self):
        adj = np.asarray(
            [
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
            ]
        )
        model = ErdosRenyiGraphModel(0.3, directed=True, self_loops=False)

        self.assertTrue(np.isfinite(model.log_likelihood(adj)))

    def test_mle_rejects_graphs_that_violate_declared_structure(self):
        asymmetric = np.asarray([[0, 1], [0, 0]])
        self_loop = np.asarray([[1, 0], [0, 0]])
        for fit in (
            lambda adj: ErdosRenyiGraphModel.fit_mle(adj),
            lambda adj: StochasticBlockGraphModel.fit_mle(adj, [0, 0], num_blocks=1),
        ):
            with self.subTest(model=fit), self.assertRaisesRegex(ValueError, "symmetric"):
                fit(asymmetric)
            with self.subTest(model=fit), self.assertRaisesRegex(ValueError, "diagonal"):
                fit(self_loop)

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

    def test_stochastic_block_log_likelihood_rejects_asymmetric_adjacency_when_undirected(self):
        adj = np.asarray(
            [
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
            ]
        )
        model = StochasticBlockGraphModel([[0.3, 0.3], [0.3, 0.3]], [0, 0, 1, 1])

        with self.assertRaises(ValueError):
            model.log_likelihood(adj)

    def test_stochastic_block_log_likelihood_rejects_self_loop_when_disallowed(self):
        adj = np.asarray(
            [
                [1, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
            ]
        )
        model = StochasticBlockGraphModel([[0.3, 0.3], [0.3, 0.3]], [0, 0, 1, 1])

        with self.assertRaises(ValueError):
            model.log_likelihood(adj)

    def test_stochastic_block_log_likelihood_allows_self_loop_when_enabled(self):
        adj = np.asarray(
            [
                [1, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 0],
            ]
        )
        model = StochasticBlockGraphModel([[0.3, 0.3], [0.3, 0.3]], [0, 0, 1, 1], self_loops=True)

        self.assertTrue(np.isfinite(model.log_likelihood(adj)))

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

    def test_graph_priors_assignments_and_budgets_are_not_coerced(self):
        adjacency = np.zeros((2, 2), dtype=int)
        for kwargs in (
            {"pseudo_count": -1.0},
            {"pseudo_count": np.nan},
            {"pseudo_count": True},
            {"prior_p": -0.1},
            {"prior_p": 1.1},
            {"prior_p": np.nan},
            {"prior_p": True},
            {"directed": 1},
            {"self_loops": 0},
        ):
            with self.subTest(family="erdos", kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                ErdosRenyiGraphModel.fit_mle(adjacency, **kwargs)
            with self.subTest(family="sbm", kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                StochasticBlockGraphModel.fit_mle(adjacency, [0, 0], num_blocks=1, **kwargs)

        for assignments, num_blocks in (
            ([[0, 0]], 1),
            ([0.0, 0.0], 1),
            (["0", "0"], 1),
            ([0, -1], 1),
            ([0, 1], 1),
        ):
            with self.subTest(assignments=assignments, num_blocks=num_blocks), self.assertRaises(ValueError):
                StochasticBlockGraphModel.fit_mle(adjacency, assignments, num_blocks=num_blocks)

        for controls in (
            {"num_blocks": 0},
            {"num_blocks": 1.5},
            {"num_blocks": True},
            {"max_its": 0},
            {"max_its": 1.5},
            {"restarts": 0},
            {"restarts": 1.5},
            {"seed": True},
            {"seed": -1},
            {"pseudo_count": np.inf},
            {"prior_p": np.nan},
        ):
            with self.subTest(controls=controls), self.assertRaises((TypeError, ValueError)):
                hard_em_stochastic_block_model(adjacency, num_blocks=1, **controls)

    def test_model_construction_and_sampling_validate_exact_schema(self):
        for assignments in ([[0, 0]], [0.0, 0.0], ["0", "0"], [0, 1]):
            with self.subTest(assignments=assignments), self.assertRaises(ValueError):
                StochasticBlockGraphModel([[0.5]], assignments)
        with self.assertRaises(ValueError):
            StochasticBlockGraphModel(np.empty((0, 0)), [])

        model = ErdosRenyiGraphModel(0.5)
        for num_nodes in (1.5, True, -1):
            with self.subTest(num_nodes=num_nodes), self.assertRaises((TypeError, ValueError)):
                model.sample(num_nodes)
        with self.assertRaises(ValueError):
            model.sample(1, seed=2**32)


if __name__ == "__main__":
    unittest.main()
