"""Schema, probability, and heterogeneous-estimation contracts for scheduled HMMs."""

import unittest

import numpy as np

from mixle.stats import CategoricalDistribution, IntegerCategoricalDistribution
from mixle.stats.combinator.null_dist import NullEstimator
from mixle.stats.latent.scheduled_hidden_markov_model import (
    ByLength,
    ByPosition,
    ByRelativePosition,
    Homogeneous,
    PhaseSchedule,
    ScheduledHiddenMarkovModelDistribution,
    ScheduledHMMEstimator,
)
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution


def _model(**overrides):
    arguments = {
        "inits": np.asarray([[0.6, 0.4]]),
        "transitions": np.asarray([[[0.8, 0.2], [0.3, 0.7]]]),
        "emissions": [
            [
                CategoricalDistribution({"a": 0.8, "b": 0.2}),
                CategoricalDistribution({"a": 0.3, "b": 0.7}),
            ]
        ],
        "schedule": Homogeneous(),
    }
    arguments.update(overrides)
    return ScheduledHiddenMarkovModelDistribution(**arguments)


class _BadPhaseSchedule(PhaseSchedule):
    n_phases = 1

    def phase(self, t, length):
        return 1

    def to_dict(self):
        return {"kind": "_BadPhaseSchedule"}


class _FractionalPhaseCount(PhaseSchedule):
    n_phases = 1.5

    def phase(self, t, length):
        return 0

    def to_dict(self):
        return {"kind": "_FractionalPhaseCount"}


class ScheduledHmmScheduleContractTest(unittest.TestCase):
    def test_phase_counts_are_exact_positive_integers(self):
        for schedule_type in (ByPosition, ByRelativePosition):
            for invalid in (True, 1.5, "2"):
                with (
                    self.subTest(schedule_type=repr(schedule_type), invalid=repr(invalid)),
                    self.assertRaises(TypeError),
                ):
                    schedule_type(invalid)
            with self.assertRaises(ValueError):
                schedule_type(0)

    def test_length_boundaries_are_exact_nonnegative_and_increasing(self):
        for invalid in ([1.5], [True], ["2"]):
            with self.subTest(invalid=repr(invalid)), self.assertRaises(TypeError):
                ByLength(invalid)
        with self.assertRaises(ValueError):
            ByLength([-1])
        with self.assertRaises(ValueError):
            ByLength([3, 3])

    def test_phase_arguments_are_not_fractionally_coerced(self):
        with self.assertRaises(TypeError):
            ByPosition(2).phase(0.5, 3)
        with self.assertRaises(TypeError):
            ByRelativePosition(2).phase(0, 3.5)
        with self.assertRaises(TypeError):
            ByLength([2]).phase(0, True)

    def test_custom_schedule_phase_count_and_outputs_are_checked(self):
        with self.assertRaisesRegex(TypeError, "n_phases"):
            _model(schedule=_FractionalPhaseCount())
        model = _model(schedule=_BadPhaseSchedule())
        with self.assertRaisesRegex(ValueError, "outside"):
            model.log_density(["a"])


class ScheduledHmmProbabilityContractTest(unittest.TestCase):
    def test_initial_probabilities_are_exact_phase_simplexes(self):
        invalid = (
            [0.6, 0.4],
            [[0.6, 0.5]],
            [[-0.1, 1.1]],
            [[np.nan, np.nan]],
        )
        for value in invalid:
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                _model(inits=value)

    def test_transition_probabilities_are_exact_row_simplexes(self):
        invalid = (
            [[0.8, 0.2], [0.3, 0.7]],
            [[[0.8, 0.3], [0.3, 0.7]]],
            [[[0.8, -0.2], [0.3, 0.7]]],
            [[[0.8, 0.2], [np.inf, 0.0]]],
        )
        for value in invalid:
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                _model(transitions=value)

    def test_emission_grid_geometry_is_exact(self):
        with self.assertRaisesRegex(ValueError, "exact 1 x 2"):
            _model(emissions=[[CategoricalDistribution({"a": 1.0})]])

    def test_probability_and_component_inputs_are_owned(self):
        inits = np.asarray([[0.6, 0.4]])
        transitions = np.asarray([[[0.8, 0.2], [0.3, 0.7]]])
        emissions = _model().emissions
        model = _model(inits=inits, transitions=transitions, emissions=emissions)
        inits[:] = [[1.0, 0.0]]
        transitions[:] = np.eye(2)
        emissions.clear()
        np.testing.assert_allclose(model.inits, [[0.6, 0.4]])
        np.testing.assert_allclose(model.transitions, [[[0.8, 0.2], [0.3, 0.7]]])
        self.assertEqual(len(model.emissions), 1)


class ScheduledHmmEstimatorContractTest(unittest.TestCase):
    def test_estimator_controls_are_exact_and_bounded(self):
        model = _model()
        for invalid in (True, -1.0, np.inf):
            with self.subTest(invalid=repr(invalid)), self.assertRaises((TypeError, ValueError)):
                model.estimator(pseudo_count=invalid)

    def test_distribution_estimator_preserves_every_emission_family(self):
        model = ScheduledHiddenMarkovModelDistribution(
            inits=[[1.0], [1.0]],
            transitions=[[[1.0]], [[1.0]]],
            emissions=[
                [GaussianDistribution(mu=0.0, sigma2=1.0)],
                [IntegerCategoricalDistribution(0, [0.5, 0.5])],
            ],
            schedule=ByPosition(2),
        )
        estimator = model.estimator(pseudo_count=0.1)
        accumulator = estimator.accumulator_factory().make()
        accumulator.seq_initialize([[0.25, 1]], np.ones(1), np.random.RandomState(1))
        fitted = estimator.estimate(None, accumulator.value())
        self.assertIsInstance(fitted.emissions[0][0], GaussianDistribution)
        self.assertIsInstance(fitted.emissions[1][0], IntegerCategoricalDistribution)

    def test_zero_count_phases_still_produce_probability_laws(self):
        estimator = ScheduledHMMEstimator(2, Homogeneous(), NullEstimator(), pseudo_count=0.0)
        accumulator = estimator.accumulator_factory().make()
        fitted = estimator.estimate(None, accumulator.value())
        np.testing.assert_allclose(fitted.inits, [[0.5, 0.5]])
        np.testing.assert_allclose(fitted.transitions, [np.eye(2)])


if __name__ == "__main__":
    unittest.main()
