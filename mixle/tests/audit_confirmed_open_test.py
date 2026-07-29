"""Regressions for audit findings independently confirmed open after the first repair pass."""

from __future__ import annotations

import types
import unittest
from unittest.mock import patch

import numpy as np

from mixle.engines._optional_extension import load_optional_extension
from mixle.stats.graphs.erdos_renyi_graph import ErdosRenyiGraphDistribution
from mixle.stats.graphs.random_dot_product_graph import (
    RandomDotProductGraphDistribution,
    RandomDotProductGraphEstimator,
)
from mixle.stats.graphs.stochastic_block_graph import (
    StochasticBlockGraphDistribution,
    StochasticBlockGraphStatistics,
)
from mixle.stats.trees.chow_liu_tree import ChowLiuTreeDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution


class ChowLiuStructureTest(unittest.TestCase):
    def _make(self, parents, order):
        n = len(parents)
        marginals = [GaussianDistribution(0.0, 1.0) for _ in range(n)]
        return ChowLiuTreeDistribution(
            parents,
            marginals,
            [{} for _ in range(n)],
            default_dists=marginals,
            feature_order=order,
        )

    def test_root_and_every_parent_must_precede_child(self):
        invalid = [
            ([1, None], [0, 1]),
            ([None, 2], [0, 1]),
            ([None, 1], [0, 1]),
            ([None, 2, 1], [0, 1, 2]),
        ]
        for parents, order in invalid:
            with self.subTest(parents=repr(parents), order=repr(order)), self.assertRaises(ValueError):
                self._make(parents, order)
        self._make([None, 0, 1], [0, 1, 2])


class GraphSupportTest(unittest.TestCase):
    def test_erdos_renyi_preserves_deterministic_endpoints(self):
        absent = np.zeros((2, 2))
        present = np.asarray([[0, 1], [1, 0]])
        zero = ErdosRenyiGraphDistribution(0.0, num_nodes=2)
        one = ErdosRenyiGraphDistribution(1.0, num_nodes=2)
        self.assertEqual(zero.log_density(absent), 0.0)
        self.assertEqual(zero.log_density(present), -np.inf)
        self.assertEqual(one.log_density(present), 0.0)
        self.assertEqual(one.log_density(absent), -np.inf)
        self.assertEqual(len(list(zero.enumerator())), 1)
        self.assertEqual(len(list(one.enumerator())), 1)

    def test_erdos_renyi_rejects_invalid_statistics_and_controls(self):
        estimator = ErdosRenyiGraphDistribution(0.5, num_nodes=2).estimator()
        for value in ((-1.0, -2.0), (1.0, 2.0), (np.inf, 0.0)):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                estimator.estimate(None, value)
        with self.assertRaises(ValueError):
            estimator.estimate(None, (0.0, 0.0))
        with self.assertRaises(ValueError):
            ErdosRenyiGraphDistribution(0.5, num_nodes=2).sampler().sample(size=1.5)

    def test_rdpg_preserves_deterministic_edge_endpoints(self):
        zero = RandomDotProductGraphDistribution([[0.0], [0.0]])
        one = RandomDotProductGraphDistribution([[1.0], [1.0]])
        absent = np.zeros((2, 2))
        present = np.asarray([[0, 1], [1, 0]])
        self.assertEqual(zero.log_density(absent), 0.0)
        self.assertEqual(zero.log_density(present), -np.inf)
        self.assertEqual(one.log_density(present), 0.0)
        self.assertEqual(one.log_density(absent), -np.inf)

    def test_rdpg_owns_parameters_and_serialized_statistics(self):
        positions = np.asarray([[0.5], [0.5]])
        dist = RandomDotProductGraphDistribution(positions)
        positions[:] = 1.0
        self.assertEqual(dist.probs[0, 1], 0.25)
        marginals = dist.edge_marginals()
        marginals[0, 1] = 1.0
        self.assertEqual(dist.probs[0, 1], 0.25)

        graph = np.asarray([[0, 1], [1, 0]])
        acc = dist.estimator().accumulator_factory().make()
        acc.update(graph, 1.0, None)
        value = acc.value()
        value[1][0, 1] = 0.0
        self.assertEqual(acc.value()[1][0, 1], 1.0)
        restored = dist.estimator().accumulator_factory().make().from_value(acc.value())
        restored_value = restored.value()
        restored_value[1][0, 1] = 0.0
        self.assertEqual(restored.value()[1][0, 1], 1.0)

    def test_rdpg_rejects_invalid_evidence_and_empty_estimation(self):
        estimator = RandomDotProductGraphEstimator(2, num_nodes=3)
        acc = estimator.accumulator_factory().make()
        asymmetric = np.asarray([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        with self.assertRaises(ValueError):
            acc.update(asymmetric, 1.0, None)
        with self.assertRaises(ValueError):
            acc.update(np.zeros((3, 3)), -1.0, None)
        with self.assertRaises(ValueError):
            estimator.estimate(None, acc.value())

    def test_rdpg_ase_uses_positive_eigenvalues_before_negative_ones(self):
        adjacency = np.asarray(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 1, 1, 0],
            ],
            dtype=np.float64,
        )
        fitted = RandomDotProductGraphEstimator(2, num_nodes=4).estimate(None, (1.0, adjacency))
        self.assertEqual(np.linalg.matrix_rank(fitted.positions, tol=1.0e-10), 2)

    def test_sbm_separates_conditional_and_joint_sample_spaces(self):
        probs = np.asarray([[0.5, 0.2], [0.2, 0.5]])
        fixed = StochasticBlockGraphDistribution(probs, [0])
        self.assertEqual(fixed.log_density(np.zeros((1, 1))), 0.0)
        with self.assertRaises(ValueError):
            StochasticBlockGraphDistribution(probs, [0], include_assignment_prior=True)

        population = StochasticBlockGraphDistribution(probs, block_prior=[0.3, 0.7])
        empty = np.zeros((1, 1))
        mass = sum(np.exp(population.log_density((empty, np.asarray([z])))) for z in range(2))
        self.assertAlmostEqual(mass, 1.0, places=12)
        with self.assertRaises(ValueError):
            population.sampler(0).sample(num_nodes=1, return_assignments=False)

    def test_sbm_preserves_endpoints_and_owns_parameters(self):
        probs = np.asarray([[0.5, 0.0], [0.0, 0.5]])
        dist = StochasticBlockGraphDistribution(probs, [0, 1])
        probs[0, 1] = probs[1, 0] = 1.0
        absent = np.zeros((2, 2))
        present = np.asarray([[0, 1], [1, 0]])
        self.assertEqual(dist.log_density(absent), 0.0)
        self.assertEqual(dist.log_density(present), -np.inf)
        marginals = dist.edge_marginals()
        marginals[0, 1] = 1.0
        self.assertEqual(dist.link_probability(0, 1), 0.0)

    def test_sbm_rejects_fractional_assignments_and_incoherent_statistics(self):
        probs = np.asarray([[0.8, 0.2], [0.2, 0.6]])
        dist = StochasticBlockGraphDistribution(probs, [0, 1])
        with self.assertRaises(ValueError):
            dist.sampler(0).sample(block_assignments=[0.9, 1.0])
        acc = dist.estimator().accumulator_factory().make()
        with self.assertRaises(ValueError):
            acc.update((np.zeros((2, 2)), [0, 3]), 1.0, None)
        corrupt = StochasticBlockGraphStatistics(
            1,
            np.asarray([[2.0, 0.0], [0.0, 0.0]]),
            np.asarray([[1.0, 0.0], [0.0, 0.0]]),
            np.asarray([1.0, 1.0]),
            2.0,
            1.0,
            False,
            False,
        )
        with self.assertRaises(ValueError):
            dist.estimator().estimate(None, corrupt)

    def test_fixed_erdos_renyi_size_is_enforced_everywhere(self):
        dist = ErdosRenyiGraphDistribution(0.5, num_nodes=3)
        wrong = np.zeros((2, 2))
        with self.assertRaisesRegex(ValueError, "fixed-size"):
            dist.log_density(wrong)
        encoded = dist.dist_to_encoder().seq_encode([wrong])
        with self.assertRaisesRegex(ValueError, "fixed-size"):
            dist.seq_log_density(encoded)
        with self.assertRaisesRegex(ValueError, "fixed-size"):
            dist.posterior(wrong)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            dist.sampler(0).sample_graph(num_nodes=2)
        accumulator = dist.estimator().accumulator_factory().make()
        with self.assertRaisesRegex(ValueError, "fixed-size"):
            accumulator.initialize(wrong, 1.0, None)

    def test_num_nodes_rejects_lossy_controls(self):
        for value in (True, 2.5, -1):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                ErdosRenyiGraphDistribution(0.5, num_nodes=value)


class OptionalExtensionLoaderTest(unittest.TestCase):
    def test_absence_and_loader_incompatibility_fall_back_with_diagnostics(self):
        absent = ModuleNotFoundError("missing", name="example.extension")
        for error, status in ((absent, "absent"), (OSError("bad ABI"), "incompatible")):
            with (
                self.subTest(status=repr(status)),
                patch(
                    "mixle.engines._optional_extension.importlib.import_module",
                    side_effect=error,
                ),
            ):
                result = load_optional_extension("example.extension", ("run",))
            self.assertFalse(result.available)
            self.assertEqual(result.status, status)
            self.assertIn(type(error).__name__, result.diagnostic)

    def test_missing_abi_symbol_is_incompatible(self):
        with patch(
            "mixle.engines._optional_extension.importlib.import_module",
            return_value=types.SimpleNamespace(),
        ):
            result = load_optional_extension("example.extension", ("run",))
        self.assertEqual(result.status, "incompatible")
        self.assertIn("run", result.diagnostic)

    def test_unexpected_extension_defect_propagates(self):
        with (
            patch(
                "mixle.engines._optional_extension.importlib.import_module",
                side_effect=RuntimeError("implementation defect"),
            ),
            self.assertRaisesRegex(RuntimeError, "implementation defect"),
        ):
            load_optional_extension("example.extension", ("run",))


if __name__ == "__main__":
    unittest.main()
