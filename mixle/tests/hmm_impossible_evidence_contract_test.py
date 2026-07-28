"""Cross-family HMM E-steps reject impossible evidence before mutating statistics."""

import unittest

import numpy as np

from mixle.stats import CategoricalDistribution
from mixle.stats.latent.hidden_markov import HiddenMarkovModelDistribution
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

    def test_hidden_markov_batch_failure_is_transactional(self):
        for kwargs in ({"use_numba": False}, {"use_numba": True}, {"terminal_states": [1]}):
            with self.subTest(**kwargs):
                model = HiddenMarkovModelDistribution(
                    _emissions(),
                    w=np.asarray([0.5, 0.5]),
                    transitions=np.asarray([[0.8, 0.2], [0.2, 0.8]]),
                    **kwargs,
                )
                accumulator = model.estimator().accumulator_factory().make()
                encoded = accumulator.acc_to_encoder().seq_encode([["a"], ["never-seen"]])
                with self.assertRaises(ImpossibleEvidenceError):
                    accumulator.seq_update(encoded, np.ones(2), model)
                np.testing.assert_array_equal(accumulator.init_counts, np.zeros(2))
                np.testing.assert_array_equal(accumulator.trans_counts, np.zeros((2, 2)))
                np.testing.assert_array_equal(accumulator.state_counts, np.zeros(2))

    def test_hidden_markov_estep_scores_each_emission_model_once(self):
        """The E-step validates evidence from its own forward pass, not a second scoring pass.

        Guarding seq_update with a separate seq_log_density re-ran every emission model over every
        observation to learn what the E-step was about to compute anyway -- about a third of
        seq_update. This pins the emission-scoring count so that cannot come back unnoticed.
        """
        for kwargs in ({"use_numba": False}, {"use_numba": True}, {"terminal_states": [1]}):
            with self.subTest(**kwargs):
                self.assertEqual(
                    self._emission_scorings_during_seq_update(**kwargs),
                    2,
                    "one emission scoring pass per state, not two",
                )

    @staticmethod
    def _emission_scorings_during_seq_update(**kwargs) -> int:
        calls = [0]

        class _Counting(CategoricalDistribution):
            def seq_log_density(self, x):
                calls[0] += 1
                return super().seq_log_density(x)

        model = HiddenMarkovModelDistribution(
            [_Counting({"a": 0.6, "b": 0.4}), _Counting({"a": 0.3, "b": 0.7})],
            w=np.asarray([0.5, 0.5]),
            transitions=np.asarray([[0.8, 0.2], [0.2, 0.8]]),
            **kwargs,
        )
        accumulator = model.estimator().accumulator_factory().make()
        encoded = accumulator.acc_to_encoder().seq_encode([["a", "b"], ["b", "a", "b"]])
        calls[0] = 0
        accumulator.seq_update(encoded, np.ones(2), model)
        return calls[0]

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
