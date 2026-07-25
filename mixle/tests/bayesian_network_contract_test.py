import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import mixle.stats as st
from mixle.inference.bayesian_network import (
    HeterogeneousBayesianNetwork,
    MixtureOfBayesianNetworks,
    _DiscreteConditionalFactor,
    _fit_multinomial_logistic,
    _hard_em_run,
    _MarginalFactor,
    _soft_em_run,
    learn_bayesian_network,
)


def _categorical_network():
    factor = _MarginalFactor(0, st.CategoricalDistribution({"a": 1.0}))
    return HeterogeneousBayesianNetwork([factor])


class BayesianNetworkWeightedContractTest(unittest.TestCase):
    def test_public_weighted_fit_rejects_inconsistent_objectives(self):
        data = [(0.0,), (1.0,)]
        invalid = ([], [1.0], [-1.0, 2.0], [np.nan, 1.0], [0.0, 0.0])
        for weights in invalid:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                learn_bayesian_network(data, weights=weights)

    def test_multinomial_optimizer_failure_is_not_a_fitted_factor(self):
        failed = SimpleNamespace(
            success=False,
            fun=1.0,
            x=np.zeros(2),
            jac=np.zeros(2),
            message="iteration limit",
        )
        with mock.patch("scipy.optimize.minimize", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "iteration limit"):
                _fit_multinomial_logistic(
                    np.asarray([[1.0], [1.0], [1.0]]),
                    np.asarray([0, 1, 2]),
                    3,
                )

    def test_conditional_parameter_count_includes_backoff_and_each_table(self):
        backoff = st.CategoricalDistribution({"a": 0.5, "b": 0.5})
        table = {
            ("x",): st.CategoricalDistribution({"a": 0.2, "b": 0.3, "c": 0.5}),
            ("y",): st.CategoricalDistribution({"a": 0.4, "b": 0.6}),
        }
        factor = _DiscreteConditionalFactor(0, [1], table, backoff)
        self.assertEqual(factor.n_params(), 4)


class BayesianNetworkMixtureContractTest(unittest.TestCase):
    def test_constructor_and_impossible_rows_fail_closed(self):
        network = _categorical_network()
        with self.assertRaises(ValueError):
            MixtureOfBayesianNetworks([], [])
        for weights in ([1.0], [0.4, 0.4], [1.1, -0.1], [np.nan, np.nan]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                MixtureOfBayesianNetworks([network, network], weights)
        model = MixtureOfBayesianNetworks([network, network], [0.5, 0.5])
        with self.assertRaisesRegex(ValueError, "zero probability"):
            model.responsibilities([("missing",)])

    def test_hard_em_uses_adjacent_assignments_and_honest_rescue_weights(self):
        data = [("a",)] * 4
        network = _categorical_network()
        initial = np.asarray([0, 0, 1, 1])
        stable = np.asarray([0, 1, 0, 1])
        learn = mock.Mock(return_value=network)

        def responsibilities(_model, _data):
            result = np.zeros((len(stable), 2))
            result[np.arange(len(stable)), stable] = 1.0
            return result

        with mock.patch.object(MixtureOfBayesianNetworks, "responsibilities", responsibilities):
            _hard_em_run(data, 2, initial, learn, 5, 4, np.random.RandomState(0))
        self.assertEqual(learn.call_count, 6)

        learn.reset_mock()

        def all_first(_model, _data):
            result = np.zeros((len(data), 2))
            result[:, 0] = 1.0
            return result

        with mock.patch.object(MixtureOfBayesianNetworks, "responsibilities", all_first):
            model, _ = _hard_em_run(
                data,
                2,
                np.zeros(len(data), dtype=np.int64),
                learn,
                1,
                4,
                np.random.RandomState(0),
            )
        np.testing.assert_array_equal(model.weights, [1.0, 0.0])

    def test_soft_em_retains_last_non_decreasing_model(self):
        data = [("a",)] * 4
        created = []

        class Encoder:
            def seq_encode(self, rows):
                return rows

        class Candidate:
            def __init__(self, objective):
                self.objective = objective

            def dist_to_encoder(self):
                return Encoder()

            def seq_log_density(self, encoded):
                return np.full(len(encoded), self.objective / len(encoded))

            def responsibilities(self, rows):
                return np.full((len(rows), 2), 0.5)

        def mixture_factory(_components, _weights):
            candidate = Candidate(10.0 if not created else 9.0)
            created.append(candidate)
            return candidate

        with mock.patch(
            "mixle.inference.bayesian_network.MixtureOfBayesianNetworks",
            side_effect=mixture_factory,
        ):
            model, value = _soft_em_run(
                data,
                2,
                np.asarray([0, 0, 1, 1]),
                lambda rows, w=None: object(),
                max_iter=4,
            )
        self.assertIs(model, created[0])
        self.assertEqual(value, 10.0)
        self.assertEqual(len(created), 2)


if __name__ == "__main__":
    unittest.main()
