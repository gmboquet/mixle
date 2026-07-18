"""PINNRegression: a physics-informed neural network model (data-fit NLL + PDE-residual penalty on collocation
points), composing into the same NeuralGaussian-style estimator/accumulator contract."""

import unittest

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _mlp(dims):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.Tanh())  # smooth activation: PINN residuals need well-behaved autograd
    return torch.nn.Sequential(*layers)


def _ode_residual(module, coll):
    """du/dx = cos(x) -- the analytic solution through u(0)=0 is u(x) = sin(x)."""
    u = module(coll)
    (du,) = torch.autograd.grad(u, coll, grad_outputs=torch.ones_like(u), create_graph=True)
    return du - torch.cos(coll)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class PINNRegressionTest(unittest.TestCase):
    def test_fits_the_analytic_solution_from_one_boundary_point_plus_the_residual(self):
        from mixle.models.pinn import PINNRegression

        torch.manual_seed(0)
        module = _mlp([1, 32, 32, 1])
        model = PINNRegression(
            module,
            _ode_residual,
            domain=([-2.0], [2.0]),
            noise=0.05,
            residual_weight=5.0,
            n_collocation=64,
            m_steps=400,
            lr=0.01,
            seed=0,
        )
        est = model.estimator()
        acc = est.accumulator_factory().make()
        data = [(np.array([0.0]), np.array([0.0]))]  # the one boundary condition: u(0) = 0
        enc = model.dist_to_encoder().seq_encode(data)
        acc.seq_update(enc, np.ones(len(data)), model)
        fitted = est.estimate(None, acc.value())

        grid = np.linspace(-2.0, 2.0, 41)[:, None]
        pred = fitted._forward(grid)[:, 0]
        truth = np.sin(grid[:, 0])
        self.assertLess(float(np.mean((pred - truth) ** 2)), 0.02)

    def test_residual_shrinks_after_training(self):
        from mixle.models.pinn import PINNRegression

        torch.manual_seed(1)
        module = _mlp([1, 32, 32, 1])
        probe = torch.linspace(-2.0, 2.0, 50, requires_grad=True).reshape(-1, 1)
        before = float((_ode_residual(module, probe) ** 2).mean().detach())

        model = PINNRegression(
            module,
            _ode_residual,
            domain=([-2.0], [2.0]),
            residual_weight=1.0,
            n_collocation=64,
            m_steps=300,
            lr=0.01,
            seed=0,
        )
        fitted = model.estimator().estimate(None, (np.zeros((0, 1)), np.zeros((0, 1)), np.zeros((0,))))
        probe2 = torch.linspace(-2.0, 2.0, 50, requires_grad=True).reshape(-1, 1)
        after = float((_ode_residual(fitted.module, probe2) ** 2).mean().detach())
        self.assertLess(after, before * 0.1)

    def test_trains_with_zero_labeled_data_pure_pde_fit(self):
        from mixle.models.pinn import PINNRegression

        torch.manual_seed(2)
        module = _mlp([1, 16, 16, 1])
        model = PINNRegression(
            module,
            _ode_residual,
            domain=([-2.0], [2.0]),
            residual_weight=1.0,
            n_collocation=64,
            m_steps=200,
            lr=0.01,
            seed=0,
        )
        est = model.estimator()
        empty = (np.zeros((0, 1)), np.zeros((0, 1)), np.zeros((0,)))
        fitted = est.estimate(None, empty)  # no labeled data at all -- must not raise, must still train

        probe = torch.linspace(-2.0, 2.0, 50, requires_grad=True).reshape(-1, 1)
        resid = float((_ode_residual(fitted.module, probe) ** 2).mean().detach())
        self.assertLess(resid, 0.05)  # honors du/dx = cos(x) even with no boundary condition to pin the constant

    def test_reported_density_is_the_data_fit_nll_only(self):
        from mixle.models.pinn import PINNRegression

        torch.manual_seed(3)
        module = _mlp([1, 16, 1])
        model = PINNRegression(module, _ode_residual, domain=([-2.0], [2.0]), noise=1.0, seed=0)
        x, y = np.array([0.3]), np.array([0.0])
        ld = model.log_density((x, y))
        self.assertTrue(np.isfinite(ld))
        # matches the plain Gaussian NLL formula exactly (no residual term folded into the density), whatever
        # the (untrained, effectively random) module output at x happens to be
        mean = float(model._forward(x[None, :])[0, 0])
        expected = -0.5 * (y[0] - mean) ** 2 - 0.5 * np.log(2.0 * np.pi)
        self.assertAlmostEqual(ld, expected, places=4)

    def test_deterministic_given_seed(self):
        from mixle.models.pinn import PINNRegression

        def _run():
            torch.manual_seed(7)
            module = _mlp([1, 16, 1])
            model = PINNRegression(
                module,
                _ode_residual,
                domain=([-2.0], [2.0]),
                residual_weight=2.0,
                n_collocation=32,
                m_steps=60,
                lr=0.01,
                seed=5,
            )
            fitted = model.estimator().estimate(None, (np.zeros((0, 1)), np.zeros((0, 1)), np.zeros((0,))))
            grid = np.linspace(-2.0, 2.0, 10)[:, None]
            return fitted._forward(grid)[:, 0]

        a, b = _run(), _run()
        np.testing.assert_allclose(a, b, atol=1e-6)


def _module_level_residual(module, coll):
    return _ode_residual(module, coll)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class PINNRegressionSerializationTest(unittest.TestCase):
    def test_to_dict_from_dict_roundtrip_preserves_predictions(self):
        from mixle.models.pinn import PINNRegression

        torch.manual_seed(4)
        module = _mlp([1, 16, 1])
        model = PINNRegression(
            module,
            _module_level_residual,
            domain=([-2.0], [2.0]),
            noise=0.2,
            residual_weight=1.5,
            n_collocation=16,
            m_steps=1,
            lr=0.01,
            seed=9,
            name="ode-model",
        )
        payload = model.to_dict()
        from mixle.utils.serialization import trusted_deserialization

        with trusted_deserialization():  # embedded torch module: a self-produced, trusted round-trip
            restored = PINNRegression.from_dict(payload)

        grid = np.linspace(-2.0, 2.0, 5)[:, None]
        np.testing.assert_allclose(model._forward(grid), restored._forward(grid), atol=1e-6)
        self.assertEqual(restored.residual_weight, model.residual_weight)
        self.assertEqual(restored.n_collocation, model.n_collocation)
        self.assertEqual(restored.name, "ode-model")


if __name__ == "__main__":
    unittest.main()
