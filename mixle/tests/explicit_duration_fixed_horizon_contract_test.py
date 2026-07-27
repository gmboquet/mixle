"""Fixed-horizon probability contracts for the explicit-duration HMM."""

import itertools
import unittest
from collections import Counter

import numpy as np

import mixle.stats as stats
from mixle.stats.latent.structured_hmm import ExplicitDurationHMM


class ExplicitDurationFixedHorizonTest(unittest.TestCase):
    @staticmethod
    def _categorical_model():
        return ExplicitDurationHMM(
            [
                stats.CategoricalDistribution({0: 0.8, 1: 0.2}),
                stats.CategoricalDistribution({0: 0.3, 1: 0.7}),
            ],
            [0.6, 0.4],
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.25, 0.75], [0.5, 0.5]]),
            2,
        )

    def test_each_fixed_horizon_is_a_normalized_observation_law(self):
        model = self._categorical_model()
        for horizon in range(1, 5):
            mass = sum(
                np.exp(model.forward_loglik(sequence))
                for sequence in itertools.product((0, 1), repeat=horizon)
            )
            self.assertAlmostEqual(mass, 1.0, places=12)

    def test_expanded_hmm_matches_the_right_censored_law(self):
        model = self._categorical_model()
        expanded = model.to_structured_hmm()
        for horizon in range(1, 5):
            for sequence in itertools.product((0, 1), repeat=horizon):
                expected = float(expanded.seq_log_density([list(sequence)])[0])
                self.assertAlmostEqual(model.forward_loglik(sequence), expected, places=12)

    def test_sampler_and_scorer_agree_when_states_are_observable(self):
        model = ExplicitDurationHMM(
            [stats.CategoricalDistribution({0: 1.0}), stats.CategoricalDistribution({1: 1.0})],
            [0.6, 0.4],
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.25, 0.75], [0.5, 0.5]]),
            2,
        )
        expected = {
            (0, 0): 0.6 * 0.75,
            (0, 1): 0.6 * 0.25,
            (1, 1): 0.4 * 0.5,
            (1, 0): 0.4 * 0.5,
        }
        for sequence, probability in expected.items():
            self.assertAlmostEqual(np.exp(model.forward_loglik(sequence)), probability, places=12)

        samples = Counter(tuple(model.sampler(seed=seed).sample(2)) for seed in range(4000))
        for sequence, probability in expected.items():
            self.assertAlmostEqual(samples[sequence] / 4000.0, probability, delta=0.025)

    def test_censored_duration_statistics_marginalize_the_unseen_tail(self):
        model = ExplicitDurationHMM(
            [stats.CategoricalDistribution({0: 1.0}), stats.CategoricalDistribution({1: 1.0})],
            [1.0, 0.0],
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.25, 0.75], [0.5, 0.5]]),
            2,
        )
        log_likelihood, initial, transitions, durations, occupancy = model._estep([0])
        self.assertAlmostEqual(log_likelihood, 0.0)
        np.testing.assert_allclose(initial, [1.0, 0.0])
        np.testing.assert_allclose(transitions, np.zeros((2, 2)))
        np.testing.assert_allclose(durations[0], [0.25, 0.75])
        np.testing.assert_allclose(occupancy, [[1.0, 0.0]])

    def test_viterbi_and_sampling_allow_the_horizon_to_censor_a_duration(self):
        model = ExplicitDurationHMM(
            [stats.CategoricalDistribution({0: 1.0}), stats.CategoricalDistribution({1: 1.0})],
            [1.0, 0.0],
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
            3,
        )
        self.assertEqual(model.sampler(seed=3).sample(1), [0])
        self.assertAlmostEqual(model.forward_loglik([0]), 0.0)
        self.assertEqual(model.viterbi_segments([0]), [(0, 0, 1)])

    def test_sampler_length_is_an_exact_positive_integer(self):
        sampler = self._categorical_model().sampler(seed=1)
        for length in (0, -1, 1.5, True):
            with self.subTest(length=length), self.assertRaises((TypeError, ValueError)):
                sampler.sample(length)


class ExplicitDurationSchemaTest(unittest.TestCase):
    @staticmethod
    def _args():
        return (
            [stats.CategoricalDistribution({0: 1.0}), stats.CategoricalDistribution({1: 1.0})],
            [0.5, 0.5],
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.5, 0.5], [0.5, 0.5]]),
            2,
        )

    def test_constructor_rejects_silent_probability_repairs(self):
        emissions, pi, transition, durations, maximum = self._args()
        invalid = (
            ([emissions[0]], [1.0], np.array([[1.0]]), np.array([[1.0, 0.0]]), 2),
            (emissions, [1.0, 1.0], transition, durations, maximum),
            (emissions, pi, np.array([[0.1, 0.9], [1.0, 0.0]]), durations, maximum),
            (emissions, pi, np.array([[0.0, 2.0], [1.0, 0.0]]), durations, maximum),
            (emissions, pi, transition, np.array([[0.5, 0.4], [0.5, 0.5]]), maximum),
            (emissions, pi, transition, durations, 2.0),
        )
        for args in invalid:
            with self.subTest(args=args), self.assertRaises((TypeError, ValueError)):
                ExplicitDurationHMM(*args)

    def test_constructor_owns_all_probability_arrays(self):
        emissions, pi, transition, durations, maximum = self._args()
        pi = np.asarray(pi)
        model = ExplicitDurationHMM(emissions, pi, transition, durations, maximum)
        pi[0] = 0.0
        transition[0] = [1.0, 0.0]
        durations[0] = [1.0, 0.0]
        np.testing.assert_allclose(model.pi, [0.5, 0.5])
        np.testing.assert_allclose(model.a, [[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_allclose(model.dur, [[0.5, 0.5], [0.5, 0.5]])


if __name__ == "__main__":
    unittest.main()
