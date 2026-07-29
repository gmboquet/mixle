"""Uniform initial-law and state-evidence contracts for semi-supervised HMMs."""

import unittest

import numpy as np

from mixle.stats import CategoricalDistribution
from mixle.stats.latent.semi_supervised_hidden_markov_model import (
    SemiSupervisedHiddenMarkovModelDistribution,
    _as_prior,
)


def _model(**overrides):
    arguments = {
        "topics": [
            CategoricalDistribution({"x": 1.0}),
            CategoricalDistribution({"x": 1.0}),
        ],
        "transitions": [[0.5, 0.5], [0.5, 0.5]],
    }
    arguments.update(overrides)
    return SemiSupervisedHiddenMarkovModelDistribution(**arguments)


class SemiSupervisedHmmPriorContractTest(unittest.TestCase):
    def test_absent_evidence_uses_a_normalized_uniform_initial_law(self):
        model = _model()
        self.assertAlmostEqual(model.density((["x"], None)), 1.0)
        self.assertAlmostEqual(model.density((["x", "x"], None)), 1.0)

    def test_all_ones_evidence_is_no_constraint(self):
        model = _model()
        emissions = ["x", "x", "x"]
        self.assertAlmostEqual(
            model.log_density((emissions, np.ones((3, 2)))),
            model.log_density((emissions, None)),
        )

    def test_likelihood_potential_scale_does_not_change_evidence(self):
        model = _model()
        base = np.asarray([[2.0, 1.0], [0.25, 1.0]])
        scaled = base * np.asarray([[17.0], [0.01]])
        self.assertAlmostEqual(
            model.log_density((["x", "x"], base)),
            model.log_density((["x", "x"], scaled)),
            places=12,
        )

    def test_potential_rows_are_owned_and_normalized_to_unit_maximum(self):
        source = np.asarray([[2.0, 1.0]])
        normalized = _as_prior(source, 3, 2)
        source[:] = 0.0
        np.testing.assert_allclose(normalized, [[1.0, 0.5]] * 3)

    def test_invalid_or_empty_evidence_rows_are_rejected(self):
        invalid = (
            [[0.0, 0.0]],
            [[np.nan, 1.0]],
            [[np.inf, 1.0]],
            [[-1.0, 2.0]],
        )
        model = _model()
        for prior in invalid:
            with self.subTest(prior=repr(prior)), self.assertRaises(ValueError):
                model.log_density((["x"], prior))
        with self.assertRaises(TypeError):
            model.log_density((["x"], [["bad", "evidence"]]))

    def test_terminal_scoring_uses_the_same_uniform_initial_law(self):
        model = _model(terminal_states={0, 1})
        self.assertAlmostEqual(model.density((["x"], None)), 1.0)
        self.assertAlmostEqual(model.density((["x"], [[1.0, 1.0]])), 1.0)


if __name__ == "__main__":
    unittest.main()
