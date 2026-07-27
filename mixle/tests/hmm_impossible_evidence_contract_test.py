"""Cross-family HMM E-steps reject impossible evidence before mutating statistics."""

import unittest

import numpy as np

from mixle.stats import CategoricalDistribution
from mixle.stats.latent.scheduled_hidden_markov_model import (
    Homogeneous,
    ScheduledHiddenMarkovModelDistribution,
)
from mixle.stats.latent.semi_supervised_hidden_markov_model import (
    SemiSupervisedHiddenMarkovModelDistribution,
)
from mixle.stats.latent.structured_hmm import DenseTransition, StructuredHMM
from mixle.utils.vector import ImpossibleEvidenceError


def _emissions():
    return [CategoricalDistribution({"a": 1.0}), CategoricalDistribution({"b": 1.0})]


class HMMImpossibleEvidenceContractTestCase(unittest.TestCase):
    def test_scheduled_hmm_batch_failure_is_transactional(self):
        model = ScheduledHiddenMarkovModelDistribution(
            np.asarray([[0.5, 0.5]]),
            np.asarray([[[0.8, 0.2], [0.2, 0.8]]]),
            [_emissions()],
            Homogeneous(),
        )
        accumulator = model.estimator().accumulator_factory().make()
        with self.assertRaisesRegex(ImpossibleEvidenceError, "batch rows \\[1\\]"):
            accumulator.seq_update([["a"], ["never-seen"]], np.ones(2), model)
        np.testing.assert_array_equal(accumulator.init_counts, np.zeros((1, 2)))
        np.testing.assert_array_equal(accumulator.trans_counts, np.zeros((1, 2, 2)))

    def test_semi_supervised_hmm_batch_failure_is_transactional(self):
        model = SemiSupervisedHiddenMarkovModelDistribution(
            _emissions(),
            np.asarray([[0.8, 0.2], [0.2, 0.8]]),
        )
        records = [(["a"], None), (["never-seen"], None)]
        encoded = model.dist_to_encoder().seq_encode(records)
        accumulator = model.estimator().accumulator_factory().make()
        with self.assertRaisesRegex(ImpossibleEvidenceError, "batch rows \\[1\\]"):
            accumulator.seq_update(encoded, np.ones(2), model)
        np.testing.assert_array_equal(accumulator.trans_counts, np.zeros((2, 2)))

    def test_structured_hmm_batch_failure_is_transactional(self):
        model = StructuredHMM(
            _emissions(),
            np.asarray([0.5, 0.5]),
            DenseTransition(np.asarray([[0.8, 0.2], [0.2, 0.8]])),
        )
        accumulator = model.estimator().accumulator_factory().make()
        with self.assertRaisesRegex(ImpossibleEvidenceError, "batch rows \\[1\\]"):
            accumulator.seq_update([["a"], ["never-seen"]], np.ones(2), model)
        np.testing.assert_array_equal(accumulator.pi_acc, np.zeros(2))
        np.testing.assert_array_equal(accumulator.trans_acc, np.zeros((2, 2)))
        np.testing.assert_array_equal(accumulator.nk, np.zeros(2))

    def test_structured_state_posterior_rejects_impossible_evidence(self):
        model = StructuredHMM(
            _emissions(),
            np.asarray([0.5, 0.5]),
            DenseTransition(np.asarray([[0.8, 0.2], [0.2, 0.8]])),
        )
        with self.assertRaisesRegex(ImpossibleEvidenceError, "zero-probability"):
            model.state_posteriors(["never-seen"])


if __name__ == "__main__":
    unittest.main()
