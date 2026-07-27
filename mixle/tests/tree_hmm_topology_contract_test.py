"""Rooted-tree schema contracts for TreeHiddenMarkovDataEncoder."""

import unittest
from itertools import product

import numpy as np

from mixle.stats import CategoricalDistribution
from mixle.stats.combinator.null_dist import NullDistribution
from mixle.stats.latent.tree_hidden_markov_model import TreeHiddenMarkovModelDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution


def _model(use_numba):
    return TreeHiddenMarkovModelDistribution(
        topics=[
            GaussianDistribution(mu=0.0, sigma2=1.0),
            GaussianDistribution(mu=3.0, sigma2=2.0),
        ],
        w=[0.6, 0.4],
        transitions=[[0.8, 0.2], [0.3, 0.7]],
        len_dist=NullDistribution(),
        use_numba=use_numba,
    )


CANONICAL = [
    ((0, -1), 0.1),
    ((1, 0), 2.8),
    ((2, 0), -0.2),
    ((3, 1), 3.2),
]


class TreeHmmTopologyContractTest(unittest.TestCase):
    def test_input_entry_order_is_canonicalized_by_node_id(self):
        shuffled = [CANONICAL[3], CANONICAL[1], CANONICAL[0], CANONICAL[2]]
        for use_numba in (False, True):
            with self.subTest(use_numba=use_numba):
                model = _model(use_numba)
                self.assertAlmostEqual(model.log_density(shuffled), model.log_density(CANONICAL), places=12)
                np.testing.assert_array_equal(model.viterbi(shuffled), model.viterbi(CANONICAL))

    def test_encoder_reports_tree_rows_for_both_layouts(self):
        for use_numba in (False, True):
            with self.subTest(use_numba=use_numba):
                encoder = _model(use_numba).dist_to_encoder()
                self.assertEqual(encoder.row_count(encoder.seq_encode([])), 0)
                self.assertEqual(encoder.row_count(encoder.seq_encode([CANONICAL, CANONICAL])), 2)

    def test_exact_dense_node_ids_are_required(self):
        invalid = [
            [((0, -1), 0.0), ((0, 0), 1.0)],
            [((0, -1), 0.0), ((2, 0), 1.0)],
            [((False, -1), 0.0)],
            [((0.0, -1), 0.0)],
        ]
        for use_numba in (False, True):
            encoder = _model(use_numba).dist_to_encoder()
            for tree in invalid:
                with self.subTest(use_numba=use_numba, tree=tree), self.assertRaises((TypeError, ValueError)):
                    encoder.seq_encode([tree])

    def test_exactly_one_node_zero_root_is_required(self):
        invalid = [
            [],
            [((0, 0), 0.0)],
            [((0, -1), 0.0), ((1, -1), 1.0)],
            [((0, -1.0), 0.0)],
            [((0, False), 0.0)],
        ]
        for use_numba in (False, True):
            encoder = _model(use_numba).dist_to_encoder()
            for tree in invalid:
                with self.subTest(use_numba=use_numba, tree=tree), self.assertRaises((TypeError, ValueError)):
                    encoder.seq_encode([tree])

    def test_parents_must_exist_and_precede_children(self):
        invalid = [
            [((0, -1), 0.0), ((1, 2), 1.0), ((2, 0), 2.0)],
            [((0, -1), 0.0), ((1, 3), 1.0)],
            [((0, -1), 0.0), ((1, 1), 1.0)],
            [((0, -1), 0.0), ((1, -2), 1.0)],
        ]
        for use_numba in (False, True):
            encoder = _model(use_numba).dist_to_encoder()
            for tree in invalid:
                with self.subTest(use_numba=use_numba, tree=tree), self.assertRaises(ValueError):
                    encoder.seq_encode([tree])

    def test_malformed_entries_fail_before_numeric_lowering(self):
        invalid = [
            [(0, -1)],
            [((0,), 1.0)],
            [((0, -1, 2), 1.0)],
        ]
        for use_numba in (False, True):
            encoder = _model(use_numba).dist_to_encoder()
            for tree in invalid:
                with self.subTest(use_numba=use_numba, tree=tree), self.assertRaises(TypeError):
                    encoder.seq_encode([tree])


class TreeHmmViterbiContractTest(unittest.TestCase):
    def test_numpy_decoder_tracks_parent_conditioned_backpointers(self):
        model = TreeHiddenMarkovModelDistribution(
            topics=[
                CategoricalDistribution({"a": 0.9, "b": 0.1}),
                CategoricalDistribution({"a": 0.1, "b": 0.9}),
            ],
            w=[0.5, 0.5],
            transitions=[[1.0, 0.0], [0.0, 1.0]],
            use_numba=False,
        )
        tree = [((0, -1), "a"), ((1, 0), "b"), ((2, 0), "b")]
        np.testing.assert_array_equal(model.viterbi(tree), [1, 1, 1])

    def test_numpy_decoder_matches_brute_force_joint_map(self):
        tree = [
            ((0, -1), "a"),
            ((1, 0), "b"),
            ((2, 0), "a"),
            ((3, 1), "b"),
        ]
        emissions = ({"a": 0.75, "b": 0.25}, {"a": 0.2, "b": 0.8})
        weights = np.asarray([0.55, 0.45])
        transitions = np.asarray([[0.7, 0.3], [0.15, 0.85]])
        model = TreeHiddenMarkovModelDistribution(
            topics=[CategoricalDistribution(p) for p in emissions],
            w=weights,
            transitions=transitions,
            use_numba=False,
        )

        def joint_log_probability(states):
            value = np.log(weights[states[0]])
            for node, ((_, parent), observation) in enumerate(tree):
                value += np.log(emissions[states[node]][observation])
                if parent >= 0:
                    value += np.log(transitions[states[parent], states[node]])
            return value

        expected = max(product(range(2), repeat=len(tree)), key=joint_log_probability)
        np.testing.assert_array_equal(model.viterbi(tree), expected)


class TreeHmmParameterCacheContractTest(unittest.TestCase):
    def test_source_array_mutation_cannot_stale_level_marginals(self):
        weights = np.asarray([0.6, 0.4])
        transitions = np.asarray([[0.8, 0.2], [0.3, 0.7]])
        model = TreeHiddenMarkovModelDistribution(
            topics=[GaussianDistribution(mu=0.0, sigma2=1.0), GaussianDistribution(mu=2.0, sigma2=1.0)],
            w=weights,
            transitions=transitions,
            use_numba=False,
        )
        expected = model._get_p_level(3).copy()
        weights[:] = [1.0, 0.0]
        transitions[:] = np.eye(2)
        np.testing.assert_allclose(model._get_p_level(3), expected)
        np.testing.assert_allclose(model.w, [0.6, 0.4])
        np.testing.assert_allclose(model.transitions, [[0.8, 0.2], [0.3, 0.7]])

    def test_in_place_parameter_mutation_invalidates_level_marginals(self):
        model = _model(False)
        original = model._get_p_level(3).copy()
        model.w[:] = [1.0, 0.0]
        model.transitions[:] = np.eye(2)
        updated = model._get_p_level(3)
        self.assertFalse(np.array_equal(original, updated))
        np.testing.assert_allclose(updated, [[1.0, 0.0]] * 3)

if __name__ == "__main__":
    unittest.main()
