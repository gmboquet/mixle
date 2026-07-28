"""Tests for autograd MCMC gradients and the NUTS/injected-gradient parameter-posterior path."""

import unittest

import numpy as np

from mixle.inference.mcmc import (
    MCMCResult,
    nuts,
    sample_parameter_posterior,
    torch_available,
    torch_gradient,
    value_and_torch_gradient,
)
from mixle.inference.mcmc.parameter_bridge import _finite_difference_gradient

HAS_TORCH = torch_available()


class FiniteDifferenceBoundaryTestCase(unittest.TestCase):
    def test_uses_one_sided_difference_at_a_support_boundary(self):
        def target(x):
            value = float(x)
            return value if value >= 0.0 else -np.inf

        gradient = _finite_difference_gradient(target)
        self.assertAlmostEqual(gradient(0.0), 1.0, places=6)

    def test_rejects_an_isolated_point_instead_of_inventing_zero_gradient(self):
        gradient = _finite_difference_gradient(lambda x: 0.0 if float(x) == 0.0 else -np.inf)
        with self.assertRaisesRegex(ValueError, "no in-support neighbor"):
            gradient(0.0)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TorchGradientTestCase(unittest.TestCase):
    def test_matches_finite_difference_scalar(self):
        # log N(x; 2, 1) up to constant; d/dx = -(x - 2).
        log_target_np = lambda x: -0.5 * (float(np.asarray(x).reshape(-1)[0]) - 2.0) ** 2
        log_target_t = lambda t: -0.5 * (t - 2.0) ** 2
        g = torch_gradient(log_target_t)
        fd = _finite_difference_gradient(log_target_np)
        for x in (-3.0, 0.0, 5.0):
            self.assertAlmostEqual(g(x), fd(x), places=4)
            self.assertAlmostEqual(g(x), -(x - 2.0), places=10)  # exact

    def test_matches_finite_difference_vector(self):
        import torch

        mean = np.array([1.0, -2.0, 0.5])
        log_target_np = lambda x: float(-0.5 * np.sum((np.asarray(x) - mean) ** 2))
        m_t = torch.tensor(mean)
        log_target_t = lambda t: -0.5 * torch.sum((t - m_t) ** 2)
        g = torch_gradient(log_target_t)
        fd = _finite_difference_gradient(log_target_np)
        x = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(g(x), fd(x), atol=1e-4)
        np.testing.assert_allclose(g(x), -(x - mean), atol=1e-10)  # exact

    def test_value_and_gradient_agree(self):
        import torch

        m_t = torch.tensor([3.0, 4.0])
        log_target_t = lambda t: -0.5 * torch.sum((t - m_t) ** 2)
        vg = value_and_torch_gradient(log_target_t)
        x = np.array([1.0, 1.0])
        val, grad = vg(x)
        self.assertAlmostEqual(val, -0.5 * ((1 - 3) ** 2 + (1 - 4) ** 2), places=10)
        np.testing.assert_allclose(grad, -(x - np.array([3.0, 4.0])), atol=1e-10)

    def test_value_and_gradient_evaluate_the_target_exactly_once(self):
        # MXR-080-1632: the contract is one forward/backward pass. A target that advances its own
        # state per call must not be evaluated twice -- the reported value and gradient would then
        # come from two different objectives.
        import torch

        calls = {"n": 0}

        def log_target_t(t):
            calls["n"] += 1
            return -float(calls["n"]) * torch.sum(t**2)

        vg = value_and_torch_gradient(log_target_t)
        val, grad = vg(np.array([1.0, 1.0]))
        self.assertEqual(calls["n"], 1)
        # value == -c * 2 and grad == -2c * x for the SAME c; consistency check independent of c.
        c = -val / 2.0
        np.testing.assert_allclose(grad, -2.0 * c * np.array([1.0, 1.0]), atol=1e-10)

    def test_value_and_gradient_scalar_path_agrees(self):
        vg = value_and_torch_gradient(lambda t: -0.5 * (t - 2.0) ** 2)
        val, grad = vg(5.0)
        self.assertAlmostEqual(val, -4.5, places=10)
        self.assertIsInstance(grad, float)
        self.assertAlmostEqual(grad, -3.0, places=10)

    def test_nuts_samples_gaussian_with_exact_gradient(self):
        log_target_np = lambda x: -0.5 * float(np.asarray(x).reshape(-1)[0]) ** 2
        grad = torch_gradient(lambda t: -0.5 * t**2)
        res = nuts(
            log_target_np,
            grad_log_target=grad,
            initial=0.0,
            num_samples=3000,
            warmup=1000,
            rng=np.random.RandomState(0),
        )
        arr = res.sample_array()
        self.assertAlmostEqual(float(arr.mean()), 0.0, delta=0.15)
        self.assertAlmostEqual(float(arr.std()), 1.0, delta=0.2)


class ParameterPosteriorSamplerOptionTestCase(unittest.TestCase):
    """The bridge accepts nuts + an injected gradient (works without torch via finite-diff)."""

    def _poisson_setup(self):
        from mixle.stats.univariate.discrete.poisson import PoissonDistribution

        truth = PoissonDistribution(4.0)
        data = truth.sampler(0).sample(300)
        return truth, data

    def test_nuts_sampler_runs(self):
        truth, data = self._poisson_setup()
        res = sample_parameter_posterior(truth, data, sampler="nuts", steps=400, burn_in=300, seed=1)
        self.assertIsInstance(res, MCMCResult)
        lam = np.array([float(np.asarray(s).reshape(-1)[0]) for s in res.samples])
        self.assertAlmostEqual(lam.mean(), 4.0, delta=0.6)

    def test_unknown_sampler_raises(self):
        truth, data = self._poisson_setup()
        with self.assertRaises(ValueError):
            sample_parameter_posterior(truth, data, sampler="gibbs", steps=10)

    def test_injected_gradient_is_used_for_hmc(self):
        truth, data = self._poisson_setup()
        calls = {"n": 0}

        def counting_grad(x):
            calls["n"] += 1
            return _finite_difference_gradient(lambda y: 0.0)(x) * 0.0  # zero grad: just confirm it's invoked

        # A zero gradient turns HMC into a momentum-only walk; we only assert the hook was called.
        sample_parameter_posterior(
            truth, data, sampler="hmc", steps=20, burn_in=10, num_steps=3, grad_log_target=counting_grad, seed=2
        )
        self.assertGreater(calls["n"], 0)


if __name__ == "__main__":
    unittest.main()
