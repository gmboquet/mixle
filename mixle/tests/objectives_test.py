import importlib
import unittest

import numpy as np

from mixle.stats import GaussianDistribution

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
else:
    torch = None


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class ObjectiveProjectionTorchTest(unittest.TestCase):
    def setUp(self):
        from mixle.engines import TorchEngine

        self.engine = TorchEngine(dtype=torch.float64)

    def assertObjectiveResultDiagnostics(self, result):
        self.assertEqual(len(result.history), result.iterations + 1)
        self.assertAlmostEqual(result.initial_value, result.history[0])
        self.assertAlmostEqual(result.value, result.best_value)
        self.assertAlmostEqual(result.final_delta, result.history[-1] - result.history[-2])
        self.assertIsInstance(result.converged, bool)
        self.assertGreater(result.improvement, 0.0)
        self.assertIsNotNone(result.best_value)
        self.assertIsInstance(result.best_iteration, int)
        self.assertGreaterEqual(result.best_iteration, 0)
        self.assertLess(result.best_iteration, len(result.history))
        expected_best = max(result.history) if result.maximize else min(result.history)
        self.assertAlmostEqual(result.best_value, expected_best)
        self.assertGreater(result.best_improvement, 0.0)
        self.assertIsNotNone(result.final_gradient_norm)
        self.assertTrue(np.isfinite(result.final_gradient_norm))
        self.assertGreaterEqual(result.final_gradient_norm, 0.0)

    def three_class_classification_fixture(self):
        centers = np.asarray([[-1.25, -1.0], [1.25, -1.0], [0.0, 1.35]], dtype=float)
        offsets = np.asarray(
            [
                [-0.18, -0.04],
                [-0.12, 0.10],
                [-0.05, -0.14],
                [0.02, 0.05],
                [0.08, -0.08],
                [0.15, 0.12],
                [0.20, -0.02],
                [-0.22, 0.16],
            ],
            dtype=float,
        )
        x = np.vstack([center + offsets for center in centers])
        y = np.repeat(np.arange(3, dtype=np.int64), len(offsets))
        return x, y

    def zero_module_parameters(self, model):
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()

    def test_fit_objective_maximizes_user_supplied_distribution_objective(self):
        from mixle.inference.objectives import ExpectedLogDensity, fit_objective

        truth = GaussianDistribution(1.5, 0.7)
        start = GaussianDistribution(-1.0, 3.0)
        data = truth.sampler(seed=1).sample(size=300)
        enc = start.dist_to_encoder().seq_encode(data)
        start_obj = float(ExpectedLogDensity()(start, enc, self.engine).detach().cpu().item())

        fitted, value = fit_objective(
            enc, start, ExpectedLogDensity(), engine=self.engine, max_its=250, lr=0.05, tol=0.0
        )

        self.assertGreater(value, start_obj)
        self.assertLess(abs(fitted.mu - np.mean(data)), 0.15)
        self.assertLess(abs(fitted.sigma2 - np.var(data)), 0.20)

    def test_variational_projection_matches_source_moments(self):
        from mixle.inference.objectives import variational_projection

        source = GaussianDistribution(2.0, 0.5)
        target = GaussianDistribution(-2.0, 4.0)
        data = source.sampler(seed=2).sample(size=500)

        projected, value = variational_projection(
            source, target, data=data, engine=self.engine, max_its=300, lr=0.05, tol=0.0
        )

        self.assertTrue(np.isfinite(value))
        self.assertLess(abs(projected.mu - np.mean(data)), 0.12)
        self.assertLess(abs(projected.sigma2 - np.var(data)), 0.15)

    def test_variational_projection_result_reports_diagnostics(self):
        from mixle.inference.objectives import variational_projection

        source = GaussianDistribution(1.0, 0.8)
        target = GaussianDistribution(-1.5, 3.5)
        data = source.sampler(seed=3).sample(size=300)

        result = variational_projection(
            source, target, data=data, engine=self.engine, max_its=180, lr=0.05, tol=0.0, return_result=True
        )

        self.assertTrue(np.isfinite(result.value))
        self.assertObjectiveResultDiagnostics(result)

    def test_unnormalized_log_likelihood_with_exact_partition(self):
        from mixle.inference.objectives import UnnormalizedLogLikelihood, fit_objective

        truth = GaussianDistribution(1.25, 0.6)
        start = GaussianDistribution(-1.0, 3.0)
        data = truth.sampler(seed=5).sample(size=250)
        enc = start.dist_to_encoder().seq_encode(data)

        def log_unnormalized(model, enc_data, engine):
            x = engine.asarray(enc_data)
            return -0.5 * (x - model.mu) * (x - model.mu) / model.sigma2

        def log_partition(model, engine):
            return 0.5 * engine.log(engine.asarray(2.0 * np.pi) * model.sigma2)

        objective = UnnormalizedLogLikelihood(log_unnormalized, log_partition=log_partition)
        fitted, value = fit_objective(enc, start, objective, engine=self.engine, max_its=250, lr=0.05, tol=0.0)

        self.assertTrue(np.isfinite(value))
        self.assertLess(abs(fitted.mu - np.mean(data)), 0.15)
        self.assertLess(abs(fitted.sigma2 - np.var(data)), 0.20)

    def test_optimize_torch_objective_accepts_arbitrary_parameters(self):
        from mixle.inference.objectives import optimize_torch_objective

        x = torch.tensor(-4.0, dtype=torch.float64, requires_grad=True)
        value, _ = optimize_torch_objective(
            [x], lambda: -((x - 3.0) ** 2), engine=self.engine, max_its=200, lr=0.1, tol=0.0
        )

        self.assertGreater(value, -1.0e-4)
        self.assertLess(abs(float(x.detach().cpu().item()) - 3.0), 0.02)

    def test_optimize_torch_objective_restores_best_seen_tensor_state(self):
        from mixle.inference.objectives import optimize_torch_objective

        x = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        value, _ = optimize_torch_objective(
            [x], lambda: -((x - 1.0) ** 2), engine=self.engine, max_its=1, lr=10.0, tol=0.0
        )

        self.assertAlmostEqual(value, -1.0)
        self.assertAlmostEqual(float(x.detach().cpu().item()), 0.0)

        y = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        final_value, _ = optimize_torch_objective(
            [y], lambda: -((y - 1.0) ** 2), engine=self.engine, max_its=1, lr=10.0, tol=0.0, restore_best=False
        )

        self.assertLess(final_value, -1.0)
        self.assertGreater(abs(float(y.detach().cpu().item())), 5.0)

    def test_optimize_torch_objective_result_reports_diagnostics(self):
        from mixle.inference.objectives import optimize_torch_objective

        x = torch.tensor(-4.0, dtype=torch.float64, requires_grad=True)
        result = optimize_torch_objective(
            [x],
            lambda: -((x - 3.0) ** 2),
            engine=self.engine,
            max_its=200,
            lr=0.1,
            tol=0.0,
            return_result=True,
        )

        self.assertGreater(result.value, -1.0e-4)
        self.assertLess(abs(result.model[0] - 3.0), 0.02)
        self.assertObjectiveResultDiagnostics(result)
        self.assertLess(result.final_gradient_norm, 0.1)

    def test_fit_parameter_objective_handles_named_constraints(self):
        from mixle.inference.objectives import ObjectiveParameter, fit_parameter_objective

        params, value = fit_parameter_objective(
            [
                ObjectiveParameter("loc", -2.0),
                ObjectiveParameter("scale", 4.0, constraint="positive"),
                ObjectiveParameter("gate", 0.2, constraint="unit_interval"),
            ],
            lambda p, enc, engine: -((p["loc"] - 1.5) ** 2 + (p["scale"] - 0.75) ** 2 + (p["gate"] - 0.8) ** 2),
            engine=self.engine,
            max_its=350,
            lr=0.07,
            tol=0.0,
        )

        self.assertGreater(value, -1.0e-4)
        self.assertLess(abs(params["loc"] - 1.5), 0.02)
        self.assertLess(abs(params["scale"] - 0.75), 0.02)
        self.assertLess(abs(params["gate"] - 0.8), 0.02)

    def test_fit_parameter_objective_handles_simplex_vectors(self):
        from mixle.inference.objectives import ObjectiveParameter, fit_parameter_objective

        target = np.asarray([0.1, 0.25, 0.65], dtype=float)

        def objective(params, enc, engine):
            delta = params["w"] - engine.asarray(target)
            return -engine.sum(delta * delta)

        params, value = fit_parameter_objective(
            [ObjectiveParameter("w", [0.7, 0.2, 0.1], constraint="simplex_vector")],
            objective,
            engine=self.engine,
            max_its=300,
            lr=0.08,
            tol=0.0,
        )

        self.assertGreater(value, -1.0e-4)
        self.assertAlmostEqual(float(np.sum(params["w"])), 1.0, places=10)
        self.assertLess(np.linalg.norm(params["w"] - target), 0.02)

    def test_fit_parameter_objective_accepts_simplex_alias_and_positive_matrix(self):
        from mixle.inference.objectives import ObjectiveParameter, fit_parameter_objective

        target_w = np.asarray([0.2, 0.8], dtype=float)
        target_mat = np.asarray([[0.5, 1.0], [1.5, 2.0]], dtype=float)
        target_row = np.asarray([[0.1, 0.3, 0.6], [0.7, 0.2, 0.1]], dtype=float)
        target_col = np.asarray([[0.2, 0.8], [0.3, 0.1], [0.5, 0.1]], dtype=float)

        def objective(params, enc, engine):
            w_delta = params["w"] - engine.asarray(target_w)
            mat_delta = params["scale"] - engine.asarray(target_mat)
            row_delta = params["transition"] - engine.asarray(target_row)
            col_delta = params["topic"] - engine.asarray(target_col)
            return -(
                engine.sum(w_delta * w_delta)
                + engine.sum(mat_delta * mat_delta)
                + engine.sum(row_delta * row_delta)
                + engine.sum(col_delta * col_delta)
            )

        params, value = fit_parameter_objective(
            [
                ObjectiveParameter("w", [0.75, 0.25], constraint="simplex"),
                ObjectiveParameter("scale", [[2.0, 1.5], [1.0, 0.5]], constraint="positive_matrix"),
                ObjectiveParameter("transition", [[0.4, 0.4, 0.2], [0.2, 0.3, 0.5]], constraint="row_simplex_matrix"),
                ObjectiveParameter("topic", [[0.6, 0.2], [0.2, 0.3], [0.2, 0.5]], constraint="column_simplex_matrix"),
            ],
            objective,
            engine=self.engine,
            max_its=450,
            lr=0.06,
            tol=0.0,
        )

        self.assertGreater(value, -1.0e-4)
        self.assertAlmostEqual(float(np.sum(params["w"])), 1.0, places=10)
        self.assertTrue(np.all(params["scale"] > 0.0))
        np.testing.assert_allclose(np.sum(params["transition"], axis=1), np.ones(2), rtol=1.0e-10, atol=1.0e-10)
        np.testing.assert_allclose(np.sum(params["topic"], axis=0), np.ones(2), rtol=1.0e-10, atol=1.0e-10)
        self.assertLess(np.linalg.norm(params["w"] - target_w), 0.02)
        self.assertLess(np.linalg.norm(params["scale"] - target_mat), 0.03)
        self.assertLess(np.linalg.norm(params["transition"] - target_row), 0.03)
        self.assertLess(np.linalg.norm(params["topic"] - target_col), 0.03)

    def test_fit_parameter_objective_accepts_coupled_bound_constraints(self):
        from mixle.inference.objectives import ObjectiveParameter, fit_parameter_objective

        def objective(params, enc, engine):
            return -(
                (params["low"] - 0.5) ** 2
                + (params["high"] - 1.75) ** 2
                + (params["ceiling"] - 3.0) ** 2
                + (params["floor"] - 2.0) ** 2
            )

        params, value = fit_parameter_objective(
            [
                ObjectiveParameter("low", -2.0),
                ObjectiveParameter("high", 4.0, constraint="greater_than:low"),
                ObjectiveParameter("ceiling", 5.0),
                ObjectiveParameter("floor", 1.0, constraint="less_than:ceiling"),
            ],
            objective,
            engine=self.engine,
            max_its=350,
            lr=0.07,
            tol=0.0,
        )

        self.assertGreater(value, -1.0e-4)
        self.assertGreater(params["high"], params["low"])
        self.assertLess(params["floor"], params["ceiling"])
        self.assertLess(abs(params["low"] - 0.5), 0.02)
        self.assertLess(abs(params["high"] - 1.75), 0.02)
        self.assertLess(abs(params["ceiling"] - 3.0), 0.02)
        self.assertLess(abs(params["floor"] - 2.0), 0.02)

    def test_fit_parameter_objective_accepts_prebuilt_parameter_set(self):
        from mixle.inference.objectives import ObjectiveParameter, ObjectiveParameterSet, fit_parameter_objective

        param_set = ObjectiveParameterSet([ObjectiveParameter("x", -3.0)], engine=self.engine)
        params, value = fit_parameter_objective(
            param_set,
            lambda p, enc, engine: -((p["x"] - 2.0) ** 2),
            max_its=200,
            lr=0.08,
            tol=0.0,
        )

        self.assertGreater(value, -1.0e-4)
        self.assertLess(abs(params["x"] - 2.0), 0.02)

    def test_fit_parameter_objective_restores_best_seen_named_parameters(self):
        from mixle.inference.objectives import ObjectiveParameter, fit_parameter_objective

        result = fit_parameter_objective(
            [ObjectiveParameter("x", 0.0)],
            lambda p, enc, engine: -((p["x"] - 1.0) ** 2),
            engine=self.engine,
            max_its=1,
            lr=10.0,
            tol=0.0,
            return_result=True,
        )

        self.assertEqual(result.best_iteration, 0)
        self.assertAlmostEqual(result.value, -1.0)
        self.assertAlmostEqual(result.model["x"], 0.0)
        self.assertAlmostEqual(result.best_value, -1.0)
        self.assertLess(result.history[-1], result.best_value)

    def test_fit_parameter_objective_fits_user_normal_likelihood(self):
        from mixle.inference.objectives import ObjectiveParameter, fit_parameter_objective

        rng = np.random.RandomState(11)
        data = rng.normal(1.25, np.sqrt(0.55), size=260)

        def objective(params, enc, engine):
            x = engine.asarray(enc)
            sigma2 = params["sigma2"]
            return engine.sum(
                -0.5 * (((x - params["mu"]) ** 2) / sigma2 + engine.log(engine.asarray(2.0 * np.pi) * sigma2))
            )

        result = fit_parameter_objective(
            [
                ObjectiveParameter("mu", -2.0),
                ObjectiveParameter("sigma2", 3.0, constraint="positive"),
            ],
            objective,
            enc=data,
            engine=self.engine,
            max_its=300,
            lr=0.05,
            tol=0.0,
            return_result=True,
        )

        self.assertTrue(np.isfinite(result.value))
        self.assertLess(abs(result.model["mu"] - np.mean(data)), 0.12)
        self.assertLess(abs(result.model["sigma2"] - np.var(data)), 0.15)
        self.assertGreater(result.iterations, 0)
        self.assertObjectiveResultDiagnostics(result)

    def test_gaussian_process_marginal_likelihood_improves(self):
        from mixle.models import GaussianProcessRegressor

        x = np.linspace(-2.0, 2.0, 18)[:, None]
        y = np.sin(x[:, 0])
        gp = GaussianProcessRegressor(lengthscale=0.3, amplitude=0.5, noise=0.6, engine=self.engine)
        before = float(gp.log_marginal_likelihood(x, y).detach().cpu().item())
        fitted = gp.fit(x, y, max_its=120, lr=0.03, tol=0.0)
        after = float(gp.log_marginal_likelihood(x, y).detach().cpu().item())

        self.assertIs(fitted, gp)  # fit returns the model; (value, iterations) moved behind return_result
        self.assertGreater(after, before)
        pred = gp.predict(x, y, x)
        self.assertLess(np.mean((pred - y) ** 2), 0.20)

    def test_gaussian_process_fit_reports_diagnostics(self):
        from mixle.models import GaussianProcessRegressor

        x = np.linspace(-1.5, 1.5, 14)[:, None]
        y = np.cos(1.3 * x[:, 0]) + 0.15 * x[:, 0]
        gp = GaussianProcessRegressor(lengthscale=0.25, amplitude=0.4, noise=0.8, engine=self.engine)
        before = float(gp.log_marginal_likelihood(x, y).detach().cpu().item())

        result = gp.fit(x, y, max_its=100, lr=0.03, tol=0.0, return_result=True)

        self.assertGreater(result.value, before)
        self.assertEqual(len(result.model), 4)
        self.assertObjectiveResultDiagnostics(result)
        pred = gp.predict(x, y, x)
        self.assertLess(np.mean((pred - y) ** 2), 0.20)

    def test_neural_regression_objective_improves_fit(self):
        from mixle.models import GaussianRegressionNeuralNetwork, make_mlp

        torch.manual_seed(4)
        x = np.linspace(-1.0, 1.0, 50)[:, None]
        y = 2.0 * x - 0.5
        model = GaussianRegressionNeuralNetwork(make_mlp(1, [8], 1, activation="tanh"), noise=0.8, engine=self.engine)
        before = np.mean((model.predict(x) - y) ** 2)
        value, _ = model.fit(x, y, max_its=300, lr=0.03, tol=0.0)
        after = np.mean((model.predict(x) - y) ** 2)

        self.assertTrue(np.isfinite(value))
        self.assertLess(after, 0.25 * before)
        self.assertLess(after, 0.05)

    def test_neural_regression_fit_reports_diagnostics(self):
        from mixle.models import GaussianRegressionNeuralNetwork, make_mlp

        torch.manual_seed(8)
        x = np.linspace(-1.0, 1.0, 36)[:, None]
        y = -1.25 * x + 0.3
        model = GaussianRegressionNeuralNetwork(make_mlp(1, [], 1), noise=0.9, engine=self.engine)
        before = np.mean((model.predict(x) - y) ** 2)

        result = model.fit(x, y, max_its=180, lr=0.04, tol=0.0, return_result=True)
        after = np.mean((model.predict(x) - y) ** 2)

        self.assertTrue(np.isfinite(result.value))
        self.assertEqual(len(result.model), len(list(model.parameters())))
        self.assertObjectiveResultDiagnostics(result)
        self.assertLess(after, 0.25 * before)
        self.assertLess(after, 0.05)

    def test_neural_classification_objective_improves_accuracy(self):
        from mixle.models import CategoricalClassificationNeuralNetwork, make_mlp

        x, y = self.three_class_classification_fixture()
        model = CategoricalClassificationNeuralNetwork(make_mlp(2, [], 3), engine=self.engine)
        self.zero_module_parameters(model)
        before = np.mean(model.predict(x) == y)

        value, _ = model.fit(x, y, max_its=260, lr=0.06, tol=0.0)
        proba = model.predict_proba(x)
        after = np.mean(model.predict(x) == y)

        self.assertTrue(np.isfinite(value))
        self.assertGreater(after, before)
        self.assertGreaterEqual(after, 0.95)
        np.testing.assert_allclose(np.sum(proba, axis=1), np.ones(len(x)), rtol=1.0e-10, atol=1.0e-10)

    def test_neural_classification_fit_reports_diagnostics(self):
        from mixle.models import CategoricalClassificationNeuralNetwork, make_mlp

        x, y = self.three_class_classification_fixture()
        model = CategoricalClassificationNeuralNetwork(make_mlp(2, [], 3), engine=self.engine)
        self.zero_module_parameters(model)

        result = model.fit(x, y, max_its=220, lr=0.06, tol=0.0, return_result=True)
        after = np.mean(model.predict(x) == y)

        self.assertTrue(np.isfinite(result.value))
        self.assertEqual(len(result.model), len(list(model.parameters())))
        self.assertObjectiveResultDiagnostics(result)
        self.assertGreaterEqual(after, 0.95)

    def test_neural_poisson_objective_improves_count_rate_fit(self):
        from mixle.models import PoissonRegressionNeuralNetwork, make_mlp

        x = np.linspace(-1.0, 1.0, 48)[:, None]
        y = np.asarray(np.round(np.exp(0.35 + 1.1 * x[:, 0])), dtype=np.float64)
        model = PoissonRegressionNeuralNetwork(make_mlp(1, [], 1), engine=self.engine)
        self.zero_module_parameters(model)
        before = np.mean((model.predict_rate(x)[:, 0] - y) ** 2)

        value, _ = model.fit(x, y, max_its=260, lr=0.05, tol=0.0)
        after = np.mean((model.predict_rate(x)[:, 0] - y) ** 2)

        self.assertTrue(np.isfinite(value))
        self.assertLess(after, 0.35 * before)
        self.assertLess(after, 0.35)

    def test_neural_poisson_fit_reports_diagnostics(self):
        from mixle.models import PoissonRegressionNeuralNetwork, make_mlp

        x = np.linspace(-1.0, 1.0, 40)[:, None]
        y = np.asarray(np.round(np.exp(-0.15 + 0.9 * x[:, 0])), dtype=np.float64)
        model = PoissonRegressionNeuralNetwork(make_mlp(1, [], 1), engine=self.engine)
        self.zero_module_parameters(model)
        before = float(model.log_likelihood(x, y).detach().cpu().item())

        result = model.fit(x, y, max_its=220, lr=0.05, tol=0.0, return_result=True)
        after = float(model.log_likelihood(x, y).detach().cpu().item())

        self.assertTrue(np.isfinite(result.value))
        self.assertEqual(len(result.model), len(list(model.parameters())))
        self.assertObjectiveResultDiagnostics(result)
        self.assertGreater(after, before)
        self.assertAlmostEqual(after, result.value)


if __name__ == "__main__":
    unittest.main()
