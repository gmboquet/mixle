"""Contracts shared by numerical and analytic PPL targets."""

import unittest
from unittest import mock

import numpy as np

from mixle.ppl import Gamma, Mix, Normal, free
from mixle.ppl.autograd import torch_available


@unittest.skipUnless(torch_available(), "requires a torch autodiff backend")
class AutogradTargetContractTest(unittest.TestCase):
    @staticmethod
    def _noncentered_model():
        tau = Gamma(2.0, 1.0, name="tau")
        mu = Normal(0.0, tau, name="mu").noncentered()
        return Normal(mu, 1.0)

    def test_noncentered_map_matches_constrained_numeric_target(self):
        from mixle.ppl.autograd import grad_target
        from mixle.ppl.inference import _build_target

        data = np.array([-0.5, 0.25, 1.0])
        model = self._noncentered_model()
        analytic = grad_target(model, data, jacobian=False)
        numeric, slots, *_ = _build_target(model, data, jacobian=False)
        self.assertEqual([s.name for s in slots], ["tau", "mu"])
        for u in (np.array([-0.4, 0.3]), np.array([0.6, -0.2])):
            self.assertAlmostEqual(analytic.log_target(u), numeric(u), places=10)

    def test_hierarchical_noncentered_mixture_value_and_gradient_match_numeric(self):
        from mixle.ppl.autograd import grad_target
        from mixle.ppl.inference import _build_target

        tau0 = Gamma(2.0, 1.0, name="tau0")
        tau1 = Gamma(2.0, 1.0, name="tau1")
        mu0 = Normal(0.0, tau0, name="mu0").noncentered()
        mu1 = Normal(0.0, tau1, name="mu1").noncentered()
        model = Mix([Normal(mu0, 1.0), Normal(mu1, 1.0)], free)
        data = np.array([-2.0, -1.5, 1.75, 2.25])
        analytic = grad_target(model, data, jacobian=False)
        numeric, slots, *_ = _build_target(model, data, jacobian=False)
        self.assertEqual(type(analytic).__name__, "MixtureGradTarget")
        self.assertEqual([s.name for s in slots], ["tau0", "mu0", "tau1", "mu1", "w0", "w1"])
        u = np.array([0.2, -0.1, -0.3, 0.25, 0.1, -0.2])
        value, gradient = analytic.value_and_grad(u)
        self.assertAlmostEqual(value, numeric(u), places=10)
        finite_difference = np.empty_like(u)
        eps = 1.0e-6
        for i in range(len(u)):
            upper, lower = u.copy(), u.copy()
            upper[i] += eps
            lower[i] -= eps
            finite_difference[i] = (numeric(upper) - numeric(lower)) / (2.0 * eps)
        np.testing.assert_allclose(gradient, finite_difference, atol=2.0e-5, rtol=2.0e-5)

    def test_observation_and_parameter_boundaries_reject_nonfinite_values(self):
        from mixle.ppl.autograd import grad_target

        model = Normal(free, 1.0)
        with self.assertRaisesRegex(ValueError, "infinity"):
            grad_target(model, [0.0, np.inf])
        with self.assertRaisesRegex(ValueError, "infinity"):
            grad_target(model, [0.0, -np.inf], missing="marginalize")
        with self.assertRaisesRegex(ValueError, "NaN"):
            grad_target(model, [0.0, np.nan])
        target = grad_target(model, [0.0, np.nan, 1.0], missing="marginalize")
        self.assertTrue(np.isfinite(target.log_target(np.array([0.2]))))
        with self.assertRaisesRegex(ValueError, "finite"):
            target.log_target(np.array([np.nan]))
        with self.assertRaisesRegex(ValueError, "length 1"):
            target.log_target(np.array([0.1, 0.2]))

    def test_positive_infinite_target_is_not_recast_as_impossible(self):
        import torch

        from mixle.ppl.autograd import grad_target

        target = grad_target(Normal(free, 1.0), [0.0])
        with mock.patch.object(target, "_logtarget_tensor", return_value=torch.tensor(float("inf"))):
            with self.assertRaises(FloatingPointError):
                target.log_target(np.array([0.0]))
            with self.assertRaises(FloatingPointError):
                target.value_and_grad(np.array([0.0]))

    def test_internal_scorer_failure_propagates(self):
        from mixle.ppl import autograd

        with mock.patch.object(autograd, "_scorers", side_effect=RuntimeError("broken scorer registry")):
            with self.assertRaisesRegex(RuntimeError, "broken scorer registry"):
                autograd.grad_target(Normal(free, 1.0), [0.0])


if __name__ == "__main__":
    unittest.main()
