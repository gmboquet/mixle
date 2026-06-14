import unittest

import numpy as np

from pysp.stats import (
    GaussianDistribution,
    GaussianEstimator,
    MixtureDistribution,
    MixtureEstimator,
    seq_encode,
    seq_estimate,
)
from pysp.utils.em import (
    AcceleratedEM,
    AnnealedEM,
    ConditionalMaximizationEM,
    GeneralizedEM,
    HardEM,
    IncrementalEM,
    MonteCarloEM,
    OnlineEM,
    PosteriorTransformEM,
    RestartEM,
    StandardEM,
    VariationalEM,
    observed_log_likelihood,
    run_em,
)
from pysp.utils.estimation import IncrementalEstimator, StreamingEstimator, constant


def _assert_mixture_close(test_case, actual, expected):
    np.testing.assert_allclose(actual.w, expected.w, rtol=1.0e-12, atol=1.0e-12)
    for got, exp in zip(actual.components, expected.components):
        np.testing.assert_allclose([got.mu, got.sigma2], [exp.mu, exp.sigma2], rtol=1.0e-12, atol=1.0e-12)


class EMStrategiesTestCase(unittest.TestCase):
    def setUp(self):
        self.truth = MixtureDistribution(
            [
                GaussianDistribution(-3.0, 0.7),
                GaussianDistribution(3.0, 1.1),
            ],
            [0.45, 0.55],
        )
        self.start = MixtureDistribution(
            [
                GaussianDistribution(-2.0, 2.0),
                GaussianDistribution(2.0, 2.0),
            ],
            [0.5, 0.5],
        )
        self.estimator = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        self.data = self.truth.sampler(seed=10).sample(size=120)
        self.enc = seq_encode(self.data, model=self.start)

    def test_standard_em_matches_seq_estimate(self):
        actual = StandardEM().step(self.enc, self.estimator, self.start).model
        expected = seq_estimate(self.enc, self.estimator, self.start)
        _assert_mixture_close(self, actual, expected)

    def test_temperature_one_posterior_transform_matches_standard_em(self):
        actual = PosteriorTransformEM(temperature=1.0).step(self.enc, self.estimator, self.start).model
        expected = StandardEM().step(self.enc, self.estimator, self.start).model
        _assert_mixture_close(self, actual, expected)

    def test_hard_em_returns_valid_mixture(self):
        fitted = HardEM().step(self.enc, self.estimator, self.start).model
        self.assertIsInstance(fitted, MixtureDistribution)
        self.assertAlmostEqual(float(np.sum(fitted.w)), 1.0)
        self.assertTrue(np.all(np.isfinite(fitted.seq_log_density(fitted.dist_to_encoder().seq_encode(self.data)))))

    def test_annealed_em_advances_temperature_schedule(self):
        strategy = AnnealedEM([2.0, 1.0, 0.0], hard_final=True)

        self.assertEqual(strategy.current_temperature, 2.0)
        model1 = strategy.step(self.enc, self.estimator, self.start).model
        self.assertEqual(strategy.current_temperature, 1.0)
        model2 = strategy.step(self.enc, self.estimator, model1).model
        self.assertEqual(strategy.current_temperature, 0.0)
        model3 = strategy.step(self.enc, self.estimator, model2).model

        self.assertIsInstance(model3, MixtureDistribution)
        self.assertAlmostEqual(float(np.sum(model3.w)), 1.0)
        strategy.reset()
        self.assertEqual(strategy.current_temperature, 2.0)

    def test_annealed_em_validates_temperature_schedule(self):
        with self.assertRaises(ValueError):
            AnnealedEM([])
        with self.assertRaises(ValueError):
            AnnealedEM([1.0, -0.1])

    def test_generalized_em_rejects_worse_candidate(self):
        awful = MixtureDistribution(
            [
                GaussianDistribution(50.0, 0.5),
                GaussianDistribution(60.0, 0.5),
            ],
            [0.5, 0.5],
        )

        def candidate_fn(enc_data, estimator, model, engine):
            return awful

        objective = observed_log_likelihood(self.enc)
        result = GeneralizedEM(candidate_fn).step(self.enc, self.estimator, self.start, objective=objective)
        self.assertFalse(result.accepted)
        self.assertIs(result.model, self.start)

    def test_conditional_maximization_accepts_improving_steps_only(self):
        good = StandardEM().step(self.enc, self.estimator, self.start).model
        awful = MixtureDistribution(
            [
                GaussianDistribution(50.0, 0.5),
                GaussianDistribution(60.0, 0.5),
            ],
            [0.5, 0.5],
        )

        def good_step(enc_data, estimator, model, engine):
            return good

        def bad_step(enc_data, estimator, model, engine):
            return awful

        objective = observed_log_likelihood(self.enc)
        result = ConditionalMaximizationEM([good_step, bad_step]).step(
            self.enc, self.estimator, self.start, objective=objective
        )

        self.assertFalse(result.accepted)
        self.assertIs(result.model, good)
        self.assertGreater(result.objective, objective(self.start))

    def test_monte_carlo_em_estimates_from_sampled_sufficient_statistics(self):
        data = np.asarray([1.0, 2.0, 3.0])
        model = GaussianDistribution(0.0, 1.0)
        estimator = GaussianEstimator()

        def sampled_suff_stat(enc_data, estimator, model, rng, num_samples, engine):
            return len(enc_data), (enc_data.sum(), np.dot(enc_data, enc_data), len(enc_data), len(enc_data))

        result = MonteCarloEM(sampled_suff_stat, num_samples=5, seed=1).step(data, estimator, model)

        self.assertIsInstance(result.model, GaussianDistribution)
        self.assertAlmostEqual(result.model.mu, 2.0)
        self.assertAlmostEqual(result.model.sigma2, 2.0 / 3.0)

    def test_variational_em_carries_state_into_free_energy_m_step(self):
        data = np.asarray([1.0, 2.0, 3.0])
        model = GaussianDistribution(0.0, 1.0)
        estimator = GaussianEstimator()

        def variational_step(enc_data, estimator, model, state, engine):
            return len(enc_data), (enc_data.sum(), np.dot(enc_data, enc_data), len(enc_data), len(enc_data))

        def m_step(enc_data, estimator, model, state, engine):
            nobs, suff_stat = state
            return estimator.estimate(nobs, suff_stat)

        def free_energy(enc_data, estimator, model, state, engine):
            return -abs(model.mu - 2.0)

        result = VariationalEM(variational_step, m_step, free_energy_fn=free_energy).step(data, estimator, model)

        self.assertAlmostEqual(result.model.mu, 2.0)
        self.assertAlmostEqual(result.objective, 0.0)

    def test_online_em_delegates_to_streaming_estimator(self):
        model = GaussianDistribution(0.0, 1.0)
        estimator = GaussianEstimator()
        batches = [np.asarray([-1.0, 0.0, 1.0]), np.asarray([2.0, 3.0])]
        strategy = OnlineEM(schedule=constant(0.25))
        stream = StreamingEstimator(estimator, schedule=constant(0.25), model=model)

        current = model
        expected = model
        for batch in batches:
            enc = seq_encode(batch, model=current)
            result = strategy.step(enc, estimator, current)
            expected = stream.update(enc_data=enc)
            current = result.model

        self.assertAlmostEqual(current.mu, expected.mu, places=12)
        self.assertAlmostEqual(current.sigma2, expected.sigma2, places=12)
        self.assertEqual(result.metadata["online_step"], 2)
        self.assertEqual(result.metadata["nobs"], stream.nobs)

    def test_incremental_em_delegates_to_incremental_estimator(self):
        model = GaussianDistribution(0.0, 1.0)
        estimator = GaussianEstimator()
        strategy = IncrementalEM()
        incremental = IncrementalEstimator(estimator, model=model)
        chunks = [
            ("a", np.asarray([-2.0, -1.0, 0.0])),
            ("b", np.asarray([1.0, 2.0])),
            ("a", np.asarray([-3.0, -2.0])),
        ]

        current = model
        expected = model
        for chunk_id, data in chunks:
            enc = seq_encode(data, model=current)
            result = strategy.step_chunk(chunk_id, enc, estimator, current)
            expected = incremental.update(chunk_id, enc_data=enc)
            current = result.model

        self.assertAlmostEqual(current.mu, expected.mu, places=12)
        self.assertAlmostEqual(current.sigma2, expected.sigma2, places=12)
        self.assertEqual(result.metadata["chunk_id"], "a")
        self.assertEqual(result.metadata["incremental_step"], 3)
        self.assertEqual(set(strategy._incremental.chunk_values.keys()), {"a", "b"})
        with self.assertRaises(KeyError):
            strategy.chunk_value("missing")

    def test_incremental_em_step_uses_chunk_id_function(self):
        model = GaussianDistribution(0.0, 1.0)
        estimator = GaussianEstimator()
        chunk_ids = iter(["a", "b"])
        strategy = IncrementalEM(chunk_id_fn=lambda enc_data, estimator, model, engine: next(chunk_ids))

        enc_a = seq_encode(np.asarray([0.0, 1.0]), model=model)
        result_a = strategy.step(enc_a, estimator, model)
        enc_b = seq_encode(np.asarray([2.0, 3.0]), model=result_a.model)
        result_b = strategy.step(enc_b, estimator, result_a.model)

        self.assertEqual(result_a.metadata["chunk_id"], "a")
        self.assertEqual(result_b.metadata["chunk_id"], "b")
        with self.assertRaises(ValueError):
            IncrementalEM().step(enc_a, estimator, model)

    def test_accelerated_em_accepts_objective_improving_proposal(self):
        data = np.asarray([1.0, 2.0, 3.0])
        model = GaussianDistribution(0.0, 1.0)
        estimator = GaussianEstimator()
        enc = seq_encode(data, model=model)

        def proposal(old_model, base_model, step_factor, enc_data, estimator, engine):
            return GaussianDistribution(3.0, max(base_model.sigma2, 1.0e-6))

        objective = lambda candidate: -abs(candidate.mu - 3.0)
        result = AcceleratedEM(proposal, step_factors=(1.0,)).step(enc, estimator, model, objective=objective)

        self.assertAlmostEqual(result.model.mu, 3.0)
        self.assertTrue(result.metadata["accelerated"])
        self.assertEqual(result.metadata["step_factor"], 1.0)
        self.assertGreater(result.objective, result.metadata["base_objective"])

    def test_accelerated_em_rejects_worse_proposal_and_keeps_base_step(self):
        data = np.asarray([1.0, 2.0, 3.0])
        model = GaussianDistribution(0.0, 1.0)
        estimator = GaussianEstimator()
        enc = seq_encode(data, model=model)

        def proposal(old_model, base_model, step_factor, enc_data, estimator, engine):
            return GaussianDistribution(50.0, 1.0)

        objective = lambda candidate: -abs(candidate.mu - 2.0)
        result = AcceleratedEM(proposal, step_factors=(1.0,)).step(enc, estimator, model, objective=objective)

        self.assertAlmostEqual(result.model.mu, 2.0)
        self.assertFalse(result.metadata["accelerated"])
        self.assertIsNone(result.metadata["step_factor"])
        self.assertAlmostEqual(result.objective, 0.0)

    def test_accelerated_em_validates_proposal_and_step_factors(self):
        with self.assertRaises(TypeError):
            AcceleratedEM(None)
        with self.assertRaises(ValueError):
            AcceleratedEM(lambda *args: args[1], step_factors=())
        with self.assertRaises(ValueError):
            AcceleratedEM(lambda *args: args[1], step_factors=(1.0, 0.0))

    def test_restart_em_keeps_best_initialization(self):
        awful = MixtureDistribution(
            [
                GaussianDistribution(50.0, 0.5),
                GaussianDistribution(60.0, 0.5),
            ],
            [0.5, 0.5],
        )
        objective = observed_log_likelihood(self.enc)

        fitted = RestartEM([awful, self.start], strategy=StandardEM(), max_its=2, delta=None).run(
            self.enc, self.estimator, objective=objective
        )

        self.assertGreater(objective(fitted), objective(awful))

    def test_run_em_improves_observed_log_likelihood(self):
        objective = observed_log_likelihood(self.enc)
        before = objective(self.start)
        fitted = run_em(self.enc, self.estimator, self.start, strategy=StandardEM(), max_its=3, delta=None)
        self.assertGreater(objective(fitted), before)

    def test_optimize_accepts_em_strategy(self):
        # the strategy hook lets em.py strategies drive optimize's loop; the
        # default and an explicit StandardEM must reach the same fit, and a
        # callable strategy must also be accepted
        import io

        from pysp.utils.estimation import optimize

        default_fit = optimize(self.data, self.estimator, max_its=15, rng=np.random.RandomState(2), out=io.StringIO())
        strategy_fit = optimize(
            self.data,
            self.estimator,
            max_its=15,
            rng=np.random.RandomState(2),
            out=io.StringIO(),
            strategy=StandardEM(),
        )
        _assert_mixture_close(self, strategy_fit, default_fit)

        callable_fit = optimize(
            self.data,
            self.estimator,
            max_its=5,
            rng=np.random.RandomState(2),
            out=io.StringIO(),
            strategy=lambda enc, est, model: seq_estimate(enc, est, model),
        )
        self.assertIsInstance(callable_fit, MixtureDistribution)
        self.assertTrue(np.all(np.isfinite(callable_fit.w)))


if __name__ == "__main__":
    unittest.main()
