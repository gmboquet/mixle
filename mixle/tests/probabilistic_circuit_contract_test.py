"""Graph, encoder, and EM contracts for probabilistic circuits."""

import unittest

import numpy as np

import mixle.stats as stats
from mixle.stats.latent.probabilistic_circuit import (
    ProbabilisticCircuitDistribution,
    _Node,
    leaf,
    prod,
    summ,
)
from mixle.utils.vector import ImpossibleEvidenceError


def _one_variable_sum(weights=(0.7, 0.3)):
    return ProbabilisticCircuitDistribution(
        summ(
            [
                leaf(0, stats.CategoricalDistribution({"a": 0.8, "b": 0.2})),
                leaf(0, stats.CategoricalDistribution({"a": 0.2, "b": 0.8})),
            ],
            weights,
        ),
        num_vars=1,
    )


class CircuitGraphContractTest(unittest.TestCase):
    def test_builder_rejects_cycles_unknown_kinds_and_empty_internal_nodes(self):
        child = leaf(0, stats.CategoricalDistribution({"a": 1.0}))
        root = prod([child])
        root.children.append(root)
        with self.assertRaisesRegex(ValueError, "acyclic"):
            ProbabilisticCircuitDistribution(root, num_vars=1)

        unknown = _Node("mystery")
        with self.assertRaisesRegex(ValueError, "unknown"):
            ProbabilisticCircuitDistribution(unknown, num_vars=1)
        with self.assertRaises(ValueError):
            prod([])
        with self.assertRaises(ValueError):
            summ([])

    def test_builder_requires_exact_scopes_and_probability_weights(self):
        distribution = stats.CategoricalDistribution({"a": 1.0})
        for scope in (0.5, True, (), (0, 0)):
            with self.subTest(scope=scope), self.assertRaises((TypeError, ValueError)):
                leaf(scope, distribution)
        children = [leaf(0, distribution), leaf(0, distribution)]
        for weights in ([-1.0, 2.0], [1.0, 1.0], [np.nan, np.nan], [1.0]):
            with self.subTest(weights=weights), self.assertRaises((TypeError, ValueError)):
                summ(children, weights)

    def test_flattened_fast_path_requires_a_complete_topological_dag(self):
        d0 = stats.CategoricalDistribution({"a": 1.0})
        d1 = stats.CategoricalDistribution({"b": 1.0})
        valid = (
            [
                ("leaf", 0),
                ("leaf", 1),
                ("sum", [0, 1], [np.log(0.6), np.log(0.4)]),
            ],
            {0: d0, 1: d1},
            {0: (0,), 1: (0,)},
        )
        model = ProbabilisticCircuitDistribution(valid, num_vars=1)
        self.assertAlmostEqual(np.exp(model.log_density(["a"])), 0.6)

        invalid = (
            (([("leaf", 0), ("mystery", [0])], {0: d0}, {0: (0,)}), ValueError),
            (([("leaf", 0), ("product", [])], {0: d0}, {0: (0,)}), ValueError),
            (([("leaf", 0), ("product", [1])], {0: d0}, {0: (0,)}), ValueError),
            (([("leaf", 0), ("product", [0, 0])], {0: d0}, {0: (0,)}), ValueError),
            (([("leaf", 0)], {1: d0}, {1: (0,)}), ValueError),
            (([("leaf", 0)], {0: d0}, {0: (1,)}), ValueError),
            (
                (
                    [("leaf", 0), ("leaf", 1)],
                    {0: d0, 1: d1},
                    {0: (0,), 1: (0,)},
                ),
                ValueError,
            ),
            (
                (
                    [("leaf", 0), ("leaf", 1), ("sum", [0, 1], [0.0, 0.0])],
                    {0: d0, 1: d1},
                    {0: (0,), 1: (0,)},
                ),
                ValueError,
            ),
        )
        for flattened, error in invalid:
            with self.subTest(flattened=flattened), self.assertRaises(error):
                ProbabilisticCircuitDistribution(flattened, num_vars=1)

    def test_flattened_inputs_are_owned(self):
        distribution = stats.CategoricalDistribution({"a": 1.0})
        nodes = [("leaf", 0)]
        scopes = {0: [0]}
        model = ProbabilisticCircuitDistribution((nodes, {0: distribution}, scopes), num_vars=1)
        nodes[0] = ("mystery",)
        scopes[0][0] = 4
        self.assertEqual(model.nodes, [("leaf", 0)])
        self.assertEqual(model.leaf_scope, {0: (0,)})


class CircuitEncoderContractTest(unittest.TestCase):
    def test_encoder_equality_includes_ordered_leaf_semantics(self):
        supported_a = ProbabilisticCircuitDistribution(
            leaf(0, stats.CategoricalDistribution({"a": 1.0})),
            num_vars=1,
        )
        supported_b = ProbabilisticCircuitDistribution(
            leaf(0, stats.CategoricalDistribution({"b": 1.0})),
            num_vars=1,
        )
        self.assertNotEqual(supported_a.dist_to_encoder(), supported_b.dist_to_encoder())

        gaussian_a = ProbabilisticCircuitDistribution(leaf(0, stats.GaussianDistribution(0.0, 1.0)), num_vars=1)
        gaussian_b = ProbabilisticCircuitDistribution(leaf(0, stats.GaussianDistribution(5.0, 2.0)), num_vars=1)
        self.assertEqual(gaussian_a.dist_to_encoder(), gaussian_b.dist_to_encoder())

    def test_encoder_validates_leaf_payload_keys_and_row_counts(self):
        model = _one_variable_sum()
        encoder = model.dist_to_encoder()
        encoded = encoder.seq_encode([["a"], ["b"]])
        self.assertEqual(encoder.row_count(encoded), 2)
        with self.assertRaises(ValueError):
            encoder.row_count({0: encoded[0]})


class CircuitEMContractTest(unittest.TestCase):
    def test_impossible_positive_weight_evidence_is_transactional(self):
        model = ProbabilisticCircuitDistribution(
            summ(
                [
                    leaf(0, stats.CategoricalDistribution({"a": 1.0})),
                    leaf(0, stats.CategoricalDistribution({"b": 1.0})),
                ],
                [0.5, 0.5],
            ),
            num_vars=1,
        )
        accumulator = model.estimator().accumulator_factory().make()
        encoded = model.dist_to_encoder().seq_encode([["z"]])
        before = accumulator.value()
        with self.assertRaises(ImpossibleEvidenceError):
            accumulator.seq_update(encoded, [1.0], model)
        after = accumulator.value()
        for node_id in before[0]:
            np.testing.assert_array_equal(after[0][node_id], before[0][node_id])
        self.assertEqual(repr(after[1]), repr(before[1]))

        accumulator.seq_update(encoded, [0.0], model)
        for counts in accumulator.sum_counts.values():
            np.testing.assert_array_equal(counts, np.zeros_like(counts))

    def test_accumulator_weights_are_finite_nonnegative_and_exact_length(self):
        model = _one_variable_sum()
        encoded = model.dist_to_encoder().seq_encode([["a"]])
        for weights in ([], [-1.0], [np.nan], [1.0, 1.0]):
            with self.subTest(weights=weights), self.assertRaises((TypeError, ValueError)):
                model.estimator().accumulator_factory().make().seq_update(encoded, weights, model)

    def test_pseudo_counts_and_effective_sum_counts_are_validated(self):
        model = _one_variable_sum()
        for pseudo_count in (-1.0, np.nan, np.inf, "1"):
            with self.subTest(pseudo_count=pseudo_count), self.assertRaises((TypeError, ValueError)):
                model.estimator(pseudo_count=pseudo_count)

        estimator = model.estimator()
        accumulator = estimator.accumulator_factory().make()
        encoded = model.dist_to_encoder().seq_encode([["a"]])
        accumulator.seq_update(encoded, [1.0], model)
        sum_counts, leaf_values = accumulator.value()
        root_id = len(model.nodes) - 1
        for counts in ([-1.0, 2.0], [np.nan, 1.0], [np.inf, 1.0], [1.0]):
            malformed = ({root_id: np.asarray(counts)}, leaf_values)
            with self.subTest(counts=counts), self.assertRaises((TypeError, ValueError)):
                estimator.estimate(1.0, malformed)

        zero_counts = ({root_id: np.zeros(2)}, leaf_values)
        fitted = estimator.estimate(1.0, zero_counts)
        np.testing.assert_allclose(np.exp(fitted.nodes[root_id][2]), [0.7, 0.3])


if __name__ == "__main__":
    unittest.main()
