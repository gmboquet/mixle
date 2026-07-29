"""Probability, training, and encoded-row contracts for segmental HMMs."""

import unittest

import numpy as np

from mixle.stats import CategoricalDistribution
from mixle.stats.combinator.null_dist import NullEstimator
from mixle.stats.latent.segmental_hidden_markov_model import (
    SegmentalHiddenMarkovEstimator,
    SegmentalHiddenMarkovModelDistribution,
)
from mixle.utils.vector import ImpossibleEvidenceError


def _model(**overrides):
    arguments = {
        "emissions": [
            CategoricalDistribution({"a": 0.8, "b": 0.2}),
            CategoricalDistribution({"a": 0.3, "b": 0.7}),
        ],
        "w": [0.6, 0.4],
        "transitions": [[0.8, 0.2], [0.3, 0.7]],
    }
    arguments.update(overrides)
    return SegmentalHiddenMarkovModelDistribution(**arguments)


class SegmentalHmmDistributionContractTest(unittest.TestCase):
    def test_emission_and_initial_geometry_is_exact(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            _model(emissions=[], w=[])
        for weights in ([[0.6, 0.4]], [0.6, 0.5], [-0.1, 1.1], [np.nan, np.nan]):
            with self.subTest(weights=repr(weights)), self.assertRaises(ValueError):
                _model(w=weights)

    def test_transition_matrix_is_not_reshaped_or_normalized(self):
        invalid = (
            [0.8, 0.2, 0.3, 0.7],
            [[0.8, 0.3], [0.3, 0.7]],
            [[0.8, -0.2], [0.3, 0.7]],
            [[0.8, 0.2], [np.inf, 0.0]],
        )
        for transitions in invalid:
            with self.subTest(transitions=repr(transitions)), self.assertRaises(ValueError):
                _model(transitions=transitions)

    def test_reachable_nonterminal_zero_row_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "reachable non-terminal"):
            _model(transitions=[[0.0, 0.0], [0.3, 0.7]])

    def test_accepted_zero_rows_are_recorded_and_canonicalized(self):
        unreachable = _model(w=[1.0, 0.0], transitions=[[1.0, 0.0], [0.0, 0.0]])
        np.testing.assert_array_equal(unreachable.transitions, np.eye(2))
        self.assertEqual(unreachable.unreachable_transition_rows, (1,))

        terminal = _model(
            w=[1.0, 0.0],
            transitions=[[0.5, 0.5], [0.0, 0.0]],
            terminal_states={1},
        )
        np.testing.assert_array_equal(terminal.transitions[1], [0.0, 1.0])

    def test_mutable_inputs_are_owned(self):
        emissions = _model().emissions
        weights = np.asarray([0.6, 0.4])
        transitions = np.asarray([[0.8, 0.2], [0.3, 0.7]])
        model = _model(emissions=emissions, w=weights, transitions=transitions)
        emissions.clear()
        weights[:] = [1.0, 0.0]
        transitions[:] = np.eye(2)
        self.assertEqual(len(model.emissions), 2)
        np.testing.assert_allclose(model.w, [0.6, 0.4])
        np.testing.assert_allclose(model.transitions, [[0.8, 0.2], [0.3, 0.7]])


class SegmentalHmmTrainingContractTest(unittest.TestCase):
    def test_estimator_controls_and_terminal_ids_are_exact(self):
        for invalid in (True, -1.0, np.inf, (1.0,), (1.0, 2.0, 3.0)):
            with self.subTest(invalid=repr(invalid)), self.assertRaises((TypeError, ValueError)):
                SegmentalHiddenMarkovEstimator([NullEstimator()], pseudo_count=invalid)
        with self.assertRaises(TypeError):
            SegmentalHiddenMarkovEstimator([NullEstimator()], terminal_states={0.5})
        with self.assertRaises(ValueError):
            SegmentalHiddenMarkovEstimator([NullEstimator()], terminal_states={1})

    def test_estimator_validates_complete_sufficient_statistic_geometry(self):
        estimator = SegmentalHiddenMarkovEstimator([NullEstimator(), NullEstimator()])
        valid = (
            2,
            np.zeros(2),
            np.zeros(2),
            np.zeros((2, 2)),
            [None, None],
            None,
        )
        malformed = (
            (3, *valid[1:]),
            (*valid[:1], np.zeros(3), *valid[2:]),
            (*valid[:3], np.zeros(4), *valid[4:]),
            (*valid[:4], [None], valid[5]),
        )
        for value in malformed:
            with self.subTest(value=repr(value)), self.assertRaises((TypeError, ValueError)):
                estimator.estimate(None, value)

    def test_zero_count_rows_become_self_loops_not_uniform_fabrication(self):
        estimator = SegmentalHiddenMarkovEstimator(
            [NullEstimator(), NullEstimator()],
            pseudo_count=0.0,
        )
        accumulator = estimator.accumulator_factory().make()
        fitted = estimator.estimate(None, accumulator.value())
        np.testing.assert_allclose(fitted.w, [0.5, 0.5])
        np.testing.assert_allclose(fitted.transitions, np.eye(2))

    def test_impossible_batch_fails_before_any_statistics_mutate(self):
        model = SegmentalHiddenMarkovModelDistribution(
            [CategoricalDistribution({"a": 1.0}), CategoricalDistribution({"b": 1.0})],
            [0.5, 0.5],
            [[0.8, 0.2], [0.2, 0.8]],
        )
        accumulator = model.estimator().accumulator_factory().make()
        encoded = model.dist_to_encoder().seq_encode([["a"], ["never-seen"]])
        with self.assertRaisesRegex(ImpossibleEvidenceError, "batch rows \\[1\\]"):
            accumulator.seq_update(encoded, np.ones(2), model)
        np.testing.assert_array_equal(accumulator.init_counts, np.zeros(2))
        np.testing.assert_array_equal(accumulator.state_counts, np.zeros(2))
        np.testing.assert_array_equal(accumulator.trans_counts, np.zeros((2, 2)))

    def test_encoder_reports_and_validates_sequence_rows(self):
        encoder = _model().dist_to_encoder()
        self.assertEqual(encoder.row_count(encoder.seq_encode([])), 0)
        self.assertEqual(encoder.row_count(encoder.seq_encode([["a"], ["a", "b"]])), 2)


if __name__ == "__main__":
    unittest.main()
