"""Probability-law and structural-zero contracts for structured HMMs."""

import unittest

import numpy as np

import mixle.stats as stats
from mixle.stats.latent.structured_hmm import (
    BlockDiagonalTransition,
    DenseTransition,
    InputOutputHMM,
    KroneckerTransition,
    LowRankTransition,
    SparseTransition,
    StructuredHMM,
    TransitionOperator,
    dirichlet_transition,
    kron_initial,
    sticky_transition,
)
from mixle.utils.vector import ImpossibleEvidenceError


class _InconsistentTransition(TransitionOperator):
    n_states = 2

    def forward(self, alpha):
        return np.asarray(alpha)[::-1]

    def backward(self, values):
        return np.asarray(values)

    def as_matrix(self):
        return np.eye(2)


class TransitionProbabilityContractTest(unittest.TestCase):
    def test_dense_requires_an_owned_square_stochastic_matrix(self):
        source = np.array([[0.75, 0.25], [0.1, 0.9]])
        transition = DenseTransition(source)
        source[0] = [0.0, 1.0]
        np.testing.assert_allclose(transition.as_matrix()[0], [0.75, 0.25])

        invalid = (
            np.ones((2, 3)),
            np.array([[0.8, 0.8], [0.2, 0.8]]),
            np.array([[1.0, 0.0], [-0.1, 1.1]]),
            np.array([[1.0, 0.0], [np.nan, np.nan]]),
            np.array([[1.0, 0.0], [0.0, 0.0]]),
        )
        for matrix in invalid:
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                DenseTransition(matrix)

    def test_dense_prior_and_hyperparameters_are_validated(self):
        matrix = np.eye(2)
        for prior in (np.ones((3, 3)), [[1.0, -1.0], [0.0, 1.0]], [[np.inf, 0.0], [0.0, 1.0]]):
            with self.subTest(prior=prior), self.assertRaises(ValueError):
                DenseTransition(matrix, prior=prior)
        for constructor in (sticky_transition, dirichlet_transition):
            for value in (-1.0, np.nan, np.inf):
                with self.subTest(constructor=constructor.__name__, value=value), self.assertRaises(ValueError):
                    constructor(matrix, value)

    def test_low_rank_requires_aligned_row_stochastic_factors(self):
        with self.assertRaises(ValueError):
            LowRankTransition(np.ones((2, 2)), np.full((2, 2), 0.5))
        with self.assertRaises(ValueError):
            LowRankTransition(np.full((3, 2), 0.5), np.full((2, 2), 0.5))
        with self.assertRaises(ValueError):
            LowRankTransition(np.array([[1.0, 0.0], [np.nan, 0.0]]), np.eye(2))

    def test_sparse_schema_is_exact_and_every_row_has_outgoing_mass(self):
        invalid_calls = (
            (2.0, [(0, 0), (1, 1)], None),
            (2, [(0.0, 0), (1, 1)], None),
            (2, [(0, 0), (0, 0), (1, 1)], None),
            (2, [(0, 0), (1, 2)], None),
            (2, [(0, 0)], None),
            (2, [(0, 0), (1, 1)], [1.0]),
            (2, [(0, 0), (1, 1)], [1.0, -1.0]),
            (2, [(0, 0), (1, 1)], [1.0, np.nan]),
            (2, [(0, 0), (1, 1)], [1.0, 0.0]),
        )
        for n_states, edges, values in invalid_calls:
            with (
                self.subTest(n_states=n_states, edges=edges, values=values),
                self.assertRaises((TypeError, ValueError)),
            ):
                SparseTransition(n_states, edges, values)

    def test_composites_validate_children_and_accumulator_arity(self):
        with self.assertRaises(ValueError):
            BlockDiagonalTransition([])
        with self.assertRaises(TypeError):
            BlockDiagonalTransition([object()])
        with self.assertRaises(ValueError):
            BlockDiagonalTransition([_InconsistentTransition()])
        with self.assertRaises(TypeError):
            KroneckerTransition(DenseTransition(np.eye(2)), object())

        block = BlockDiagonalTransition([DenseTransition(np.eye(2)), DenseTransition(np.eye(2))])
        with self.assertRaises(ValueError):
            block.estimate([np.zeros((2, 2))])
        kron = KroneckerTransition(DenseTransition(np.eye(2)), DenseTransition(np.eye(2)))
        with self.assertRaises(ValueError):
            kron.estimate([np.zeros((2, 2))])

    def test_zero_count_estimation_retains_each_previous_row_law(self):
        dense = DenseTransition(np.array([[0.8, 0.2], [0.3, 0.7]]))
        np.testing.assert_allclose(dense.estimate(np.zeros((2, 2))).as_matrix(), dense.as_matrix())

        low_rank = LowRankTransition(
            np.array([[0.8, 0.2], [0.3, 0.7]]),
            np.array([[0.6, 0.4], [0.1, 0.9]]),
        )
        fitted_low_rank = low_rank.estimate([np.zeros((2, 2)), np.zeros((2, 2))])
        np.testing.assert_allclose(fitted_low_rank.g, low_rank.g)
        np.testing.assert_allclose(fitted_low_rank.phi, low_rank.phi)

        sparse = SparseTransition(2, [(0, 0), (0, 1), (1, 1)], [0.8, 0.2, 1.0])
        np.testing.assert_allclose(sparse.estimate(np.zeros(3)).as_matrix(), sparse.as_matrix())

        kron = KroneckerTransition(dense, dense)
        np.testing.assert_allclose(
            kron.estimate([np.zeros((2, 2)), np.zeros((2, 2))]).as_matrix(),
            kron.as_matrix(),
        )

    def test_initial_and_operator_contracts_are_not_silently_repaired(self):
        emissions = [stats.CategoricalDistribution({0: 1.0}), stats.CategoricalDistribution({1: 1.0})]
        for probabilities in ([1.0, 1.0], [1.1, -0.1], [np.nan, np.nan]):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                StructuredHMM(emissions, probabilities, DenseTransition(np.eye(2)))
        with self.assertRaises(ValueError):
            StructuredHMM(emissions, [0.5, 0.5], _InconsistentTransition())
        with self.assertRaises(ValueError):
            kron_initial([1.0, 1.0], [0.5, 0.5])


class StructuralZeroContractTest(unittest.TestCase):
    @staticmethod
    def _emissions():
        return [stats.CategoricalDistribution({0: 1.0}), stats.CategoricalDistribution({1: 1.0})]

    def test_impossible_structured_paths_remain_impossible(self):
        hmm = StructuredHMM(self._emissions(), [1.0, 0.0], DenseTransition(np.eye(2)))
        self.assertEqual(float(hmm.seq_log_density([[0, 1]])[0]), -np.inf)
        with self.assertRaises(ImpossibleEvidenceError):
            hmm.state_posteriors([0, 1])
        with self.assertRaises(ImpossibleEvidenceError):
            hmm.viterbi([0, 1])

    def test_final_state_constraint_preserves_exact_transition_zeros(self):
        hmm = StructuredHMM(
            self._emissions(),
            [1.0, 0.0],
            DenseTransition(np.array([[0.0, 1.0], [0.0, 1.0]])),
            final_states={1},
            len_dist=stats.CategoricalDistribution({2: 1.0}),
        )
        results = hmm.enumerator().top_k(5)
        self.assertEqual([sequence for sequence, _ in results], [[0, 1]])
        self.assertEqual(float(hmm.seq_log_density([[0, 0]])[0]), -np.inf)
        with self.assertRaises(ImpossibleEvidenceError):
            hmm.viterbi([0, 0])

    def test_iohmm_uses_the_same_impossible_evidence_contract(self):
        hmm = InputOutputHMM(
            self._emissions(),
            [1.0, 0.0],
            [DenseTransition(np.eye(2))],
        )
        record = [(0, 0), (1, 0)]
        self.assertEqual(float(hmm.log_density(record)), -np.inf)
        with self.assertRaises(ImpossibleEvidenceError):
            hmm.state_posteriors(record)
        with self.assertRaises(ImpossibleEvidenceError):
            hmm.viterbi(record)


if __name__ == "__main__":
    unittest.main()
