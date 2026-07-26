"""Regressions for audit findings independently confirmed open after the first repair pass."""

from __future__ import annotations

import types
import unittest
from unittest.mock import patch

import numpy as np

from mixle.engines._optional_extension import load_optional_extension
from mixle.stats.graphs.erdos_renyi_graph import ErdosRenyiGraphDistribution
from mixle.stats.graphs.random_dot_product_graph import RandomDotProductGraphDistribution
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
            with self.subTest(parents=parents, order=order), self.assertRaises(ValueError):
                self._make(parents, order)
        self._make([None, 0, 1], [0, 1, 2])


class GraphSupportTest(unittest.TestCase):
    def test_rdpg_preserves_deterministic_edge_endpoints(self):
        zero = RandomDotProductGraphDistribution([[0.0], [0.0]])
        one = RandomDotProductGraphDistribution([[1.0], [1.0]])
        absent = np.zeros((2, 2))
        present = np.asarray([[0, 1], [1, 0]])
        self.assertEqual(zero.log_density(absent), 0.0)
        self.assertEqual(zero.log_density(present), -np.inf)
        self.assertEqual(one.log_density(present), 0.0)
        self.assertEqual(one.log_density(absent), -np.inf)

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
            with self.subTest(value=value), self.assertRaises(ValueError):
                ErdosRenyiGraphDistribution(0.5, num_nodes=value)


class OptionalExtensionLoaderTest(unittest.TestCase):
    def test_absence_and_loader_incompatibility_fall_back_with_diagnostics(self):
        absent = ModuleNotFoundError("missing", name="example.extension")
        for error, status in ((absent, "absent"), (OSError("bad ABI"), "incompatible")):
            with self.subTest(status=status), patch(
                "mixle.engines._optional_extension.importlib.import_module",
                side_effect=error,
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
