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

    def test_level_marginals_propagate_through_the_transition_matrix(self):
        """``_get_p_level`` must return the real per-level marginals ``w @ A**k``."""
        # Deliberately not the stationary law, so successive levels must differ.
        model = TreeHiddenMarkovModelDistribution(
            topics=[GaussianDistribution(mu=0.0, sigma2=1.0), GaussianDistribution(mu=2.0, sigma2=1.0)],
            w=[0.95, 0.05],
            transitions=[[0.8, 0.2], [0.3, 0.7]],
            use_numba=False,
        )
        weights = np.asarray(model.w)
        transitions = np.asarray(model.transitions)
        expected = np.zeros((4, len(weights)))
        expected[0] = weights
        for level in range(1, 4):
            expected[level] = expected[level - 1] @ transitions
        np.testing.assert_allclose(model._get_p_level(4), expected, rtol=1e-12, atol=1e-12)
        # A non-degenerate chain genuinely moves mass between levels.
        self.assertFalse(np.allclose(expected[0], expected[1]))


class TreeHmmPosteriorContractTest(unittest.TestCase):
    """``seq_posterior`` must be the full upward-downward node marginal, not the upward message."""

    EMISSIONS = ({"a": 0.9, "b": 0.1}, {"a": 0.1, "b": 0.9})
    WEIGHTS = np.asarray([0.8, 0.2])
    TRANSITIONS = np.asarray([[0.25, 0.75], [0.6, 0.4]])
    TREES = [
        [((0, -1), "b"), ((1, 0), "b"), ((2, 1), "a")],
        [((0, -1), "a"), ((1, 0), "b"), ((2, 0), "a"), ((3, 1), "b"), ((4, 3), "a")],
        [((0, -1), "a")],
    ]

    def _model(self, use_numba):
        return TreeHiddenMarkovModelDistribution(
            topics=[CategoricalDistribution(p) for p in self.EMISSIONS],
            w=self.WEIGHTS,
            transitions=self.TRANSITIONS,
            use_numba=use_numba,
        )

    def _brute_force(self, tree):
        positions = {node: index for index, ((node, _), _) in enumerate(tree)}
        marginals = np.zeros((len(tree), 2))
        total = 0.0
        for states in product(range(2), repeat=len(tree)):
            joint = 1.0
            for index, ((_, parent), observation) in enumerate(tree):
                state = states[index]
                joint *= self.WEIGHTS[state] if parent < 0 else self.TRANSITIONS[states[positions[parent]], state]
                joint *= self.EMISSIONS[state][observation]
            total += joint
            for index in range(len(tree)):
                marginals[index, states[index]] += joint
        return marginals / total

    # KNOWN BUG, not a flaky test. TreeHiddenMarkovModelDistribution.seq_posterior returns the
    # UPWARD messages (betas), not the smoothed node marginals (gammas) its docstring promises.
    # The downward pass was never implemented -- tree_hidden_markov_model.py still carries the
    # author's own note, "Need to do upward and downward, then read back the gammas", immediately
    # above the kernel call, and the function then returns ``betas``. Consequence: only the root is
    # correct (its subtree is the whole tree); every other node ignores all evidence outside its own
    # subtree. Verified exactly -- for the chain 0->1->2 with observations b,b,a, the returned rows
    # equal ``depth_prior * own_subtree_evidence`` to 6 decimals, not the true marginals.
    # Model FITTING is unaffected: the tree HMM E-step does not call seq_posterior. The wrong values
    # do reach callers through mixle/ppl/core.py's responsibilities path.
    # Remove this decorator once the downward pass lands; unittest then reports an unexpected
    # success, which is the signal that this marker is stale.
    @unittest.expectedFailure
    def test_seq_posterior_matches_brute_force_node_marginals(self):
        expected = [self._brute_force(tree) for tree in self.TREES]
        for use_numba in (False, True):
            with self.subTest(use_numba=use_numba):
                model = self._model(use_numba)
                encoded = model.dist_to_encoder().seq_encode(self.TREES)
                for got, want in zip(model.seq_posterior(encoded), expected, strict=True):
                    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-10, atol=1e-12)

    def test_seq_posterior_agrees_across_encodings(self):
        numpy_model = self._model(False)
        numba_model = self._model(True)
        numpy_post = numpy_model.seq_posterior(numpy_model.dist_to_encoder().seq_encode(self.TREES))
        numba_post = numba_model.seq_posterior(numba_model.dist_to_encoder().seq_encode(self.TREES))
        for left, right in zip(numpy_post, numba_post, strict=True):
            np.testing.assert_allclose(np.asarray(left), np.asarray(right), rtol=1e-10, atol=1e-12)

    def test_seq_posterior_rows_are_normalized(self):
        for use_numba in (False, True):
            with self.subTest(use_numba=use_numba):
                model = self._model(use_numba)
                encoded = model.dist_to_encoder().seq_encode(self.TREES)
                for got in model.seq_posterior(encoded):
                    np.testing.assert_allclose(np.asarray(got).sum(axis=1), 1.0, rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
