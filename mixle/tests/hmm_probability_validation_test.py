"""Probability-law and ownership contracts for HiddenMarkovModelDistribution."""

import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.stats.compute.backend import backend_seq_log_density
from mixle.stats.latent.hidden_markov import HiddenMarkovModelDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution


def _gaussian_topics():
    return [GaussianDistribution(-2.0, 1.0), GaussianDistribution(2.0, 1.0)]


def _categorical_topics():
    return [
        CategoricalDistribution(pmap={"a": 0.8, "b": 0.2}),
        CategoricalDistribution(pmap={"a": 0.1, "b": 0.9}),
    ]


class HiddenMarkovInitialWeightValidationTestCase(unittest.TestCase):
    """Validation of ``w``, the initial hidden-state probability vector."""

    def test_nan_w_rejected_at_construction(self):
        with self.assertRaisesRegex(ValueError, "w"):
            HiddenMarkovModelDistribution(_gaussian_topics(), [float("nan"), 0.5], [[0.9, 0.1], [0.2, 0.8]])

    def test_negative_w_rejected_at_construction(self):
        with self.assertRaisesRegex(ValueError, "w"):
            HiddenMarkovModelDistribution(_gaussian_topics(), [-0.5, 1.5], [[0.9, 0.1], [0.2, 0.8]])

    def test_infinite_w_rejected_at_construction(self):
        with self.assertRaisesRegex(ValueError, "w"):
            HiddenMarkovModelDistribution(_gaussian_topics(), [float("inf"), 0.5], [[0.9, 0.1], [0.2, 0.8]])

    def test_non_simplex_w_rejected_at_construction(self):
        with self.assertRaisesRegex(ValueError, "sum to one"):
            HiddenMarkovModelDistribution(_gaussian_topics(), [1.0, 1.0], [[0.9, 0.1], [0.2, 0.8]])

    def test_valid_w_still_constructs(self):
        hmm = HiddenMarkovModelDistribution(_gaussian_topics(), [0.5, 0.5], [[0.9, 0.1], [0.2, 0.8]])
        self.assertTrue(np.isfinite(hmm.log_density([-2.0, -2.1])))


class HiddenMarkovTransitionsValidationTestCase(unittest.TestCase):
    """Validation of ``transitions``, the hidden-state transition matrix."""

    def test_nan_transitions_rejected_at_construction(self):
        with self.assertRaisesRegex(ValueError, "transitions"):
            HiddenMarkovModelDistribution(_gaussian_topics(), [0.5, 0.5], [[float("nan"), 0.1], [0.2, 0.8]])

    def test_negative_transitions_rejected_at_construction(self):
        with self.assertRaisesRegex(ValueError, "transitions"):
            HiddenMarkovModelDistribution(_gaussian_topics(), [0.5, 0.5], [[1.3, -0.3], [0.2, 0.8]])

    def test_infinite_transitions_rejected_at_construction(self):
        with self.assertRaisesRegex(ValueError, "transitions"):
            HiddenMarkovModelDistribution(_gaussian_topics(), [0.5, 0.5], [[float("inf"), 0.1], [0.2, 0.8]])

    def test_transitions_not_summing_to_one_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "sum to one"):
            HiddenMarkovModelDistribution(_gaussian_topics(), [0.5, 0.5], [[0.9, 0.4], [0.2, 0.8]])

    def test_reachable_zero_transition_row_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "reachable states"):
            HiddenMarkovModelDistribution(_gaussian_topics(), [0.5, 0.5], [[0.0, 0.0], [0.2, 0.8]])

    def test_unreachable_zero_transition_row_is_canonicalized_and_recorded(self):
        hmm = HiddenMarkovModelDistribution(_gaussian_topics(), [1.0, 0.0], [[1.0, 0.0], [0.0, 0.0]])
        np.testing.assert_array_equal(hmm.transitions, [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(hmm.unreachable_transition_rows, (1,))

    def test_valid_transitions_still_construct(self):
        hmm = HiddenMarkovModelDistribution(_gaussian_topics(), [0.5, 0.5], [[0.9, 0.1], [0.2, 0.8]])
        self.assertTrue(np.isfinite(hmm.log_density([-2.0, -2.1])))
        for use_numba in (True, False):
            hmm_nb = HiddenMarkovModelDistribution(
                _gaussian_topics(), [0.5, 0.5], [[0.9, 0.1], [0.2, 0.8]], use_numba=use_numba
            )
            enc = hmm_nb.dist_to_encoder().seq_encode([[-2.0, -2.1], [2.0, 1.9, 2.2]])
            ll = np.asarray(hmm_nb.seq_log_density(enc), dtype=float)
            self.assertTrue(np.all(np.isfinite(ll)), f"use_numba={use_numba}: {ll}")


class HiddenMarkovTausValidationTestCase(unittest.TestCase):
    """Validation of ``taus``, the optional per-state mixture weights over ``topics``."""

    def test_nan_taus_rejected_at_construction(self):
        with self.assertRaisesRegex(ValueError, "taus"):
            HiddenMarkovModelDistribution(
                _categorical_topics(),
                [0.6, 0.4],
                [[0.9, 0.1], [0.3, 0.7]],
                taus=[[float("nan"), 0.3], [0.2, 0.8]],
            )

    def test_negative_taus_rejected_at_construction(self):
        with self.assertRaisesRegex(ValueError, "taus"):
            HiddenMarkovModelDistribution(
                _categorical_topics(),
                [0.6, 0.4],
                [[0.9, 0.1], [0.3, 0.7]],
                taus=[[-0.1, 1.1], [0.2, 0.8]],
            )

    def test_infinite_taus_rejected_at_construction(self):
        with self.assertRaisesRegex(ValueError, "taus"):
            HiddenMarkovModelDistribution(
                _categorical_topics(),
                [0.6, 0.4],
                [[0.9, 0.1], [0.3, 0.7]],
                taus=[[float("inf"), 0.3], [0.2, 0.8]],
            )

    def test_taus_not_summing_to_one_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "sum to one"):
            HiddenMarkovModelDistribution(
                _categorical_topics(),
                [0.6, 0.4],
                [[0.9, 0.1], [0.3, 0.7]],
                taus=[[0.3, 0.3], [0.2, 0.8]],
            )

    def test_valid_taus_still_construct(self):
        hmm = HiddenMarkovModelDistribution(
            _categorical_topics(),
            [0.6, 0.4],
            [[0.9, 0.1], [0.3, 0.7]],
            taus=[[0.7, 0.3], [0.2, 0.8]],
        )
        self.assertTrue(hmm.has_topics)
        self.assertTrue(np.isfinite(hmm.log_density(["a", "b", "a"])))


class HiddenMarkovGeometryAndOwnershipTestCase(unittest.TestCase):
    def test_direct_emissions_require_one_topic_per_state(self):
        with self.assertRaisesRegex(ValueError, "one topic per state"):
            HiddenMarkovModelDistribution([_gaussian_topics()[0]], [0.5, 0.5], [[0.9, 0.1], [0.2, 0.8]])

    def test_taus_geometry_must_match_states_and_topics(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            HiddenMarkovModelDistribution(
                _categorical_topics(),
                [1.0],
                [[1.0]],
                taus=[[1.0]],
            )

    def test_constructor_owns_probability_arrays_and_topic_container(self):
        topics = _gaussian_topics()
        w = np.asarray([0.5, 0.5])
        transitions = np.asarray([[0.9, 0.1], [0.2, 0.8]])
        hmm = HiddenMarkovModelDistribution(topics, w, transitions)
        score = hmm.log_density([-2.0, -2.1])

        topics.clear()
        w[:] = [1.0, 0.0]
        transitions[:] = np.eye(2)

        self.assertEqual(len(hmm.topics), 2)
        np.testing.assert_array_equal(hmm.w, [0.5, 0.5])
        np.testing.assert_array_equal(hmm.transitions, [[0.9, 0.1], [0.2, 0.8]])
        self.assertEqual(hmm.log_density([-2.0, -2.1]), score)


class HiddenMarkovEmptyBatchTestCase(unittest.TestCase):
    def test_empty_batches_are_valid_neutral_inputs(self):
        for use_numba in (False, True):
            with self.subTest(use_numba=use_numba):
                hmm = HiddenMarkovModelDistribution(
                    _gaussian_topics(),
                    [0.5, 0.5],
                    [[0.9, 0.1], [0.2, 0.8]],
                    use_numba=use_numba,
                )
                encoder = hmm.dist_to_encoder()
                encoded = encoder.seq_encode([])
                self.assertEqual(encoder.row_count(encoded), 0)
                self.assertEqual(hmm.seq_log_density(encoded).shape, (0,))
                self.assertEqual(np.asarray(backend_seq_log_density(hmm, encoded, NUMPY_ENGINE)).shape, (0,))

                estimator = hmm.estimator()
                host_accumulator = estimator.accumulator_factory().make()
                host_accumulator.seq_update(encoded, np.zeros(0), hmm)
                np.testing.assert_array_equal(host_accumulator.init_counts, [0.0, 0.0])
                np.testing.assert_array_equal(host_accumulator.state_counts, [0.0, 0.0])
                np.testing.assert_array_equal(host_accumulator.trans_counts, np.zeros((2, 2)))

                engine_accumulator = estimator.accumulator_factory().make()
                engine_accumulator.seq_update_engine(encoded, np.zeros(0), hmm, NUMPY_ENGINE)
                np.testing.assert_array_equal(engine_accumulator.init_counts, [0.0, 0.0])
                np.testing.assert_array_equal(engine_accumulator.state_counts, [0.0, 0.0])
                np.testing.assert_array_equal(engine_accumulator.trans_counts, np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
