"""Exact schema contracts for the typed lookback hidden Markov model."""

import unittest

import numpy as np

from mixle.stats.bayes.dirichlet import DirichletDistribution
from mixle.stats.combinator.null_dist import (
    NullAccumulatorFactory,
    NullDataEncoder,
    NullDistribution,
    NullEstimator,
)
from mixle.stats.latent.lookback_hidden_markov_model import (
    LookbackHiddenMarkovModelDataEncoder,
    LookbackHiddenMarkovModelDistribution,
    LookbackHiddenMarkovModelEstimator,
    LookbackHiddenMarkovModelEstimatorAccumulator,
    LookbackHiddenMarkovModelEstimatorAccumulatorFactory,
)


def _distribution(**overrides):
    arguments = {
        "topics": [NullDistribution(), NullDistribution()],
        "w": [0.6, 0.4],
        "transitions": [[0.8, 0.2], [0.3, 0.7]],
        "lag": 0,
    }
    arguments.update(overrides)
    return LookbackHiddenMarkovModelDistribution(**arguments)


def _estimator(**overrides):
    arguments = {
        "estimators": [NullEstimator(), NullEstimator()],
        "lag": 0,
        "init_estimators": [NullEstimator(), NullEstimator()],
    }
    arguments.update(overrides)
    return LookbackHiddenMarkovModelEstimator(**arguments)


class LookbackDistributionSchemaTest(unittest.TestCase):
    def test_lag_is_an_exact_nonnegative_integer(self):
        for invalid in (True, 1.5, "1"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    _distribution(lag=invalid)
        with self.assertRaises(ValueError):
            _distribution(lag=-1)

    def test_state_and_topic_geometry_is_exact(self):
        with self.assertRaisesRegex(ValueError, "one topic per state"):
            _distribution(topics=[NullDistribution()])
        with self.assertRaisesRegex(ValueError, "at least 1"):
            _distribution(topics=[], w=[])
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            _distribution(w=[[0.6, 0.4]])

    def test_initial_weights_are_a_probability_simplex(self):
        for invalid in ([0.6, 0.5], [-0.1, 1.1], [np.nan, np.nan]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    _distribution(w=invalid)

    def test_transition_geometry_is_not_reshaped_or_normalized(self):
        for invalid in (
            [0.8, 0.2, 0.3, 0.7],
            [[0.8, 0.3], [0.3, 0.7]],
            [[0.8, -0.2], [0.3, 0.7]],
            [[0.8, 0.2], [np.inf, 0.0]],
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    _distribution(transitions=invalid)

    def test_reachable_nonterminal_zero_transition_row_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "reachable non-terminal"):
            _distribution(transitions=[[0.0, 0.0], [0.3, 0.7]])

    def test_unreachable_zero_transition_row_is_canonicalized_without_mutating_input(self):
        transitions = np.array([[1.0, 0.0], [0.0, 0.0]])
        dist = _distribution(w=[1.0, 0.0], transitions=transitions)
        np.testing.assert_array_equal(dist.transitions, np.eye(2))
        np.testing.assert_array_equal(transitions, [[1.0, 0.0], [0.0, 0.0]])
        self.assertEqual(dist.unreachable_transition_rows, (1,))

    def test_positive_lag_requires_exact_initial_distribution_arity(self):
        with self.assertRaisesRegex(ValueError, "requires one initial distribution"):
            _distribution(lag=1)
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            _distribution(lag=1, init_dist=[NullDistribution()])

    def test_mutable_probability_and_component_inputs_are_owned(self):
        topics = [NullDistribution(), NullDistribution()]
        initial = [NullDistribution(), NullDistribution()]
        weights = np.array([0.6, 0.4])
        transitions = np.array([[0.8, 0.2], [0.3, 0.7]])
        dist = _distribution(
            topics=topics,
            init_dist=initial,
            w=weights,
            transitions=transitions,
        )
        topics.clear()
        initial.clear()
        weights[:] = [1.0, 0.0]
        transitions[:] = np.eye(2)
        self.assertEqual(len(dist.topics), 2)
        self.assertEqual(len(dist.init_dist), 2)
        np.testing.assert_allclose(dist.w, [0.6, 0.4])
        np.testing.assert_allclose(dist.transitions, [[0.8, 0.2], [0.3, 0.7]])

    def test_chain_prior_geometry_matches_the_state_space(self):
        invalid = (
            DirichletDistribution(np.ones(3)),
            [DirichletDistribution(np.ones(2)), DirichletDistribution(np.ones(2))],
        )
        with self.assertRaisesRegex(ValueError, "exactly 2 concentrations"):
            _distribution(prior=invalid)


class LookbackTrainingSchemaTest(unittest.TestCase):
    def test_all_training_layers_validate_lag(self):
        constructors = (
            lambda lag: LookbackHiddenMarkovModelDataEncoder(NullDataEncoder(), lag),
            lambda lag: _estimator(lag=lag),
            lambda lag: LookbackHiddenMarkovModelEstimatorAccumulator(
                [NullAccumulatorFactory().make()],
                [NullAccumulatorFactory().make()],
                lag=lag,
            ),
            lambda lag: LookbackHiddenMarkovModelEstimatorAccumulatorFactory(
                lag,
                [NullAccumulatorFactory()],
                [NullAccumulatorFactory()],
            ),
        )
        for constructor in constructors:
            for invalid in (False, 0.5):
                with self.subTest(constructor=constructor, invalid=invalid):
                    with self.assertRaises(TypeError):
                        constructor(invalid)
            with self.assertRaises(ValueError):
                constructor(-1)

    def test_positive_lag_requires_initial_training_components(self):
        with self.assertRaisesRegex(ValueError, "initial estimator"):
            LookbackHiddenMarkovModelEstimator([NullEstimator()], lag=1)
        with self.assertRaisesRegex(ValueError, "initial accumulator"):
            LookbackHiddenMarkovModelEstimatorAccumulator(
                [NullAccumulatorFactory().make()],
                lag=1,
            )
        with self.assertRaisesRegex(ValueError, "initial factory"):
            LookbackHiddenMarkovModelEstimatorAccumulatorFactory(
                1,
                [NullAccumulatorFactory()],
            )

    def test_estimator_arity_and_controls_are_exact(self):
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            _estimator(init_estimators=[NullEstimator()])
        for invalid in ((1.0,), (1.0, 2.0, 3.0)):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    _estimator(pseudo_count=invalid)
        for invalid in (-1.0, np.inf, True):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    _estimator(pseudo_count=invalid)
        short_prior = (
            DirichletDistribution(np.ones(2)),
            [DirichletDistribution(np.ones(2))],
        )
        with self.assertRaisesRegex(ValueError, "exactly 2 item"):
            _estimator(prior=short_prior)

    def test_estimator_rejects_mismatched_or_malformed_sufficient_statistics(self):
        estimator = _estimator()
        valid = (
            0,
            2,
            np.zeros(2),
            np.zeros(2),
            np.zeros((2, 2)),
            [None, None],
            [None, None],
            None,
        )
        malformed = [
            (1, *valid[1:]),
            (valid[0], 3, *valid[2:]),
            (*valid[:2], np.zeros(3), *valid[3:]),
            (*valid[:4], np.zeros(4), *valid[5:]),
            (*valid[:5], [None], *valid[6:]),
        ]
        for value in malformed:
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    estimator.estimate(None, value)


if __name__ == "__main__":
    unittest.main()
