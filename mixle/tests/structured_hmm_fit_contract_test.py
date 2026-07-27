"""Training-schema, diagnostics, and statistic-ownership contracts for structured HMMs."""

import unittest

import numpy as np

import mixle.stats as stats
from mixle.stats.latent.structured_hmm import (
    DenseTransition,
    ExplicitDurationHMM,
    InputOutputHMM,
    LowRankTransition,
    StructuredHMM,
    chunked_state_posteriors,
    fit_chunked,
)


class StructuredFitContractTest(unittest.TestCase):
    @staticmethod
    def _model():
        return StructuredHMM(
            [stats.GaussianDistribution(-1.0, 1.0), stats.GaussianDistribution(1.0, 1.0)],
            [0.5, 0.5],
            DenseTransition(np.array([[0.8, 0.2], [0.3, 0.7]])),
        )

    def test_fit_result_preserves_unpacking_and_reports_the_final_model(self):
        data = [[-1.2, -0.8, 0.7], [1.1, 0.9, -0.4]]
        weights = np.array([2.0, 0.5])
        result = self._model().fit(data, max_its=2, fast=False, weights=weights)
        fitted, trace = result
        expected = float(np.dot(weights, fitted.seq_log_density(data)))

        self.assertIs(fitted, result.model)
        self.assertIs(trace, result.log_likelihood_trace)
        self.assertEqual(result.diagnostics.log_likelihood_trace, tuple(trace))
        self.assertAlmostEqual(result.diagnostics.final_log_likelihood, expected, places=10)
        self.assertEqual(result.diagnostics.iterations, len(trace) - 1)
        self.assertTrue(result.diagnostics.monotone)
        self.assertEqual(result.diagnostics.n_sequences, 2)
        self.assertEqual(result.diagnostics.total_weight, 2.5)

    def test_zero_weight_sequences_are_excluded_from_the_fit(self):
        used = [-1.0, -0.7, 0.8]
        ignored = [50.0, 50.0, 50.0]
        weighted = self._model()
        reference = self._model()
        weighted.fit([used, ignored], weights=[1.0, 0.0], max_its=1, fast=False)
        reference.fit([used], max_its=1, fast=False)
        np.testing.assert_allclose(weighted.pi, reference.pi)
        np.testing.assert_allclose(weighted.transition.as_matrix(), reference.transition.as_matrix())
        np.testing.assert_allclose(
            [emission.mu for emission in weighted.emissions],
            [emission.mu for emission in reference.emissions],
        )

    def test_fit_controls_data_and_weights_are_exact(self):
        data = [[-1.0, 1.0]]
        invalid_calls = (
            {"max_its": 1.5},
            {"max_its": True},
            {"max_its": 0},
            {"tol": "0.1"},
            {"tol": -1.0},
            {"tol": np.nan},
            {"fast": 1},
            {"weights": []},
            {"weights": [-1.0]},
            {"weights": [np.nan]},
            {"weights": [0.0]},
        )
        for kwargs in invalid_calls:
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                self._model().fit(data, **kwargs)
        for invalid_data in ([], [[]]):
            with self.subTest(data=invalid_data), self.assertRaises(ValueError):
                self._model().fit(invalid_data)

    def test_chunk_controls_and_receipt_are_validated(self):
        data = [[-1.0, -0.5, 0.5, 1.0]]
        for kwargs in (
            {"chunk": 0, "overlap": 0},
            {"chunk": 2.0, "overlap": 0},
            {"chunk": 2, "overlap": -1},
            {"chunk": 2, "overlap": 2},
            {"chunk": 2, "overlap": 0, "workers": 1.5},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                fit_chunked(self._model(), data, max_its=1, **kwargs)
        result = fit_chunked(self._model(), data, chunk=4, overlap=0, max_its=1)
        self.assertTrue(result.diagnostics.approximate)
        self.assertEqual(result.diagnostics.log_likelihood_trace, tuple(result[1]))
        with self.assertRaises(ValueError):
            chunked_state_posteriors(self._model(), data[0], chunk=2, overlap=2)


class IOHMMFitContractTest(unittest.TestCase):
    @staticmethod
    def _model():
        return InputOutputHMM(
            [stats.GaussianDistribution(-1.0, 1.0), stats.GaussianDistribution(1.0, 1.0)],
            [0.5, 0.5],
            [
                DenseTransition(np.array([[0.8, 0.2], [0.3, 0.7]])),
                DenseTransition(np.array([[0.2, 0.8], [0.7, 0.3]])),
            ],
        )

    def test_input_batches_lengths_and_symbols_are_exact(self):
        observations = [[-1.0, 1.0]]
        invalid_inputs = (
            [],
            [[0]],
            [[0, 1.0]],
            [[0, True]],
            [[0, -1]],
            [[0, 2]],
        )
        for inputs in invalid_inputs:
            with self.subTest(inputs=inputs), self.assertRaises((TypeError, ValueError)):
                self._model().fit(observations, inputs, max_its=1)
        with self.assertRaises(TypeError):
            self._model().log_density([(-1.0, 0.0), (1.0, 1)])
        with self.assertRaises(ValueError):
            self._model().seq_log_density([[-1.0, 1.0]], [[0]])

    def test_weighted_fit_reports_the_final_controlled_model(self):
        observations = [[-1.2, -0.8, 0.7], [1.1, 0.9, -0.4]]
        inputs = [[0, 0, 1], [1, 0, 1]]
        weights = np.array([1.5, 0.5])
        result = self._model().fit(observations, inputs, weights=weights, max_its=2)
        expected = float(np.dot(weights, result.model.seq_log_density(observations, inputs)))
        self.assertAlmostEqual(result.diagnostics.final_log_likelihood, expected, places=10)
        self.assertTrue(result.diagnostics.monotone)
        self.assertFalse(result.diagnostics.approximate)


class StatisticOwnershipContractTest(unittest.TestCase):
    @staticmethod
    def _structured_accumulator():
        model = StructuredHMM(
            [stats.GaussianDistribution(-1.0, 1.0), stats.GaussianDistribution(1.0, 1.0)],
            [0.5, 0.5],
            LowRankTransition(
                np.array([[0.8, 0.2], [0.3, 0.7]]),
                np.array([[0.6, 0.4], [0.1, 0.9]]),
            ),
            keys=("initial", "transition"),
        )
        return model.estimator().accumulator_factory().make()

    def test_structured_transition_statistics_never_alias_payloads_or_key_store(self):
        accumulator = self._structured_accumulator()
        accumulator.trans_acc[0][0, 0] = 2.0
        payload = accumulator.value()
        payload[1][0][0, 0] = 9.0
        self.assertEqual(accumulator.trans_acc[0][0, 0], 2.0)

        restored = self._structured_accumulator().from_value(accumulator.value())
        source = accumulator.value()
        restored.from_value(source)
        source[1][0][0, 0] = 11.0
        self.assertEqual(restored.trans_acc[0][0, 0], 2.0)

        store = {}
        accumulator.key_merge(store)
        accumulator.trans_acc[0][0, 0] = 13.0
        self.assertEqual(store["transition"][0][0, 0], 2.0)
        replaced = self._structured_accumulator()
        replaced.key_replace(store)
        store["transition"][0][0, 0] = 17.0
        self.assertEqual(replaced.trans_acc[0][0, 0], 2.0)

    def test_io_transition_statistics_never_alias_serialized_payloads(self):
        model = IOHMMFitContractTest._model()
        accumulator = model.estimator().accumulator_factory().make()
        accumulator.trans_accs[0][0, 0] = 3.0
        payload = accumulator.value()
        payload[1][0][0, 0] = 7.0
        self.assertEqual(accumulator.trans_accs[0][0, 0], 3.0)
        restored = model.estimator().accumulator_factory().make().from_value(accumulator.value())
        source = accumulator.value()
        restored.from_value(source)
        source[1][0][0, 0] = 8.0
        self.assertEqual(restored.trans_accs[0][0, 0], 3.0)


class ExplicitDurationFitContractTest(unittest.TestCase):
    @staticmethod
    def _model():
        return ExplicitDurationHMM(
            [stats.GaussianDistribution(-1.0, 1.0), stats.GaussianDistribution(1.0, 1.0)],
            [0.5, 0.5],
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.6, 0.4], [0.3, 0.7]]),
            2,
        )

    def test_weighted_fit_has_a_finite_final_receipt(self):
        data = [[-1.0, -0.8, 1.0], [1.1, 0.7, -1.0]]
        weights = np.array([2.0, 0.5])
        result = self._model().fit(data, max_its=2, weights=weights)
        expected = float(np.dot(weights, result.model.seq_log_density(data)))
        self.assertAlmostEqual(result.diagnostics.final_log_likelihood, expected, places=10)
        self.assertTrue(result.diagnostics.monotone)
        self.assertFalse(result.diagnostics.approximate)


if __name__ == "__main__":
    unittest.main()
