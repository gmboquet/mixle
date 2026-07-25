import unittest

import numpy as np

from mixle.inference.em import (
    EMStepResult,
    MonotonicEM,
    MonteCarloEM,
    RestartEM,
    SampledSufficientStatistics,
    run_em,
)
from mixle.stats import GaussianDistribution, GaussianEstimator, seq_encode


class _RecordingEstimator:
    def __init__(self):
        self.args = None

    def estimate(self, nobs, suff_stat):
        self.args = (nobs, suff_stat)
        return GaussianDistribution(0.0, 1.0)


class _RejectingStatefulStrategy:
    def __init__(self, seed=4):
        self.iteration = 0
        self.state = {"values": []}
        self.cache = {"before": 1}
        self.rng = np.random.RandomState(seed)

    def step(self, enc_data, estimator, model, engine=None, objective=None):
        self.iteration += 1
        self.state["values"].append(float(self.rng.normal()))
        self.cache["after"] = self.iteration
        return EMStepResult(GaussianDistribution(50.0, 1.0))


class EMTransactionStateTest(unittest.TestCase):
    def setUp(self):
        self.model = GaussianDistribution(0.0, 1.0)
        self.estimator = GaussianEstimator()
        self.enc = seq_encode(np.asarray([-1.0, 0.0, 1.0]), model=self.model)
        self.objective = lambda candidate: -abs(candidate.mu)

    def test_run_em_rejection_restores_complete_strategy_and_rng_state(self):
        strategy = _RejectingStatefulStrategy()
        before_rng = strategy.rng.get_state()

        result = run_em(
            self.enc,
            self.estimator,
            self.model,
            strategy=strategy,
            max_its=1,
            objective=self.objective,
        )

        self.assertIs(result, self.model)
        self.assertEqual(strategy.iteration, 0)
        self.assertEqual(strategy.state, {"values": []})
        self.assertEqual(strategy.cache, {"before": 1})
        after_rng = strategy.rng.get_state()
        self.assertEqual(before_rng[0], after_rng[0])
        np.testing.assert_array_equal(before_rng[1], after_rng[1])
        self.assertEqual(before_rng[2:], after_rng[2:])

    def test_direct_monotonic_rejection_restores_nested_strategy_state(self):
        base = _RejectingStatefulStrategy()
        result = MonotonicEM(base).step(
            self.enc, self.estimator, self.model, objective=self.objective
        )
        self.assertFalse(result.accepted)
        self.assertEqual(base.iteration, 0)
        self.assertEqual(base.state, {"values": []})
        self.assertEqual(base.cache, {"before": 1})

    def test_objective_exception_restores_strategy_state_before_propagating(self):
        strategy = _RejectingStatefulStrategy()

        def failing_objective(candidate):
            if candidate.mu == 50.0:
                raise LookupError("objective failed")
            return 0.0

        with self.assertRaisesRegex(LookupError, "objective failed"):
            run_em(
                self.enc,
                self.estimator,
                self.model,
                strategy=strategy,
                max_its=1,
                objective=failing_objective,
            )
        self.assertEqual(strategy.iteration, 0)
        self.assertEqual(strategy.state, {"values": []})
        self.assertEqual(strategy.cache, {"before": 1})

    def test_mcem_two_tuple_is_a_statistic_not_an_implicit_result_protocol(self):
        estimator = _RecordingEstimator()
        statistic = ("left", "right")
        MonteCarloEM(lambda *args: statistic).step(self.enc, estimator, self.model)
        self.assertEqual(estimator.args, (None, statistic))

    def test_mcem_explicit_result_carries_nobs(self):
        estimator = _RecordingEstimator()
        statistic = ("left", "right")
        MonteCarloEM(lambda *args: SampledSufficientStatistics(statistic, nobs=3)).step(
            self.enc, estimator, self.model
        )
        self.assertEqual(estimator.args, (3, statistic))

    def test_restarts_receive_independent_strategy_state(self):
        seen_iterations = []

        class RecordingStrategy:
            def __init__(self):
                self.iteration = 0

            def step(self, enc_data, estimator, model, engine=None, objective=None):
                seen_iterations.append(self.iteration)
                self.iteration += 1
                return EMStepResult(model, objective(model))

        starts = (GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0))
        strategy = RecordingStrategy()
        RestartEM(starts, strategy=strategy, max_its=1, delta=None).run(
            self.enc, self.estimator, objective=lambda candidate: -abs(candidate.mu)
        )
        self.assertEqual(seen_iterations, [0, 0])
        self.assertEqual(strategy.iteration, 0)


if __name__ == "__main__":
    unittest.main()
