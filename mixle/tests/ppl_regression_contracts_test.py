"""Release contracts for regression and mixed-model routes."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from mixle.ppl import Bernoulli, Field, Gamma, Group, Laplace, Normal, Poisson, StudentT, free


class RegressionRoutingAndDataContractTest(unittest.TestCase):
    def test_explicit_unsupported_inference_method_is_rejected(self):
        model = Normal(free * Field("x") + free, free)
        with self.assertRaisesRegex(NotImplementedError, "mcmc"):
            model.fit([0.0, 1.0], given={"x": [0.0, 1.0]}, how="mcmc")

    def test_response_covariate_group_and_extra_rows_must_align(self):
        model = Normal(free * Field("x") + free, free)
        for data, given in (
            ([0.0], {"x": [0.0, 1.0]}),
            ([0.0, 1.0], {"x": [0.0]}),
            ([0.0, 1.0], {"x": [0.0, 1.0], "unused": [1.0]}),
            ([0.0, np.nan], {"x": [0.0, 1.0]}),
            ([0.0, 1.0], {"x": [0.0, np.inf]}),
        ):
            with self.subTest(data=repr(data), given=repr(given)):
                with self.assertRaises(ValueError):
                    model.fit(data, given=given)
        grouped = Normal(free + Group("g"), free)
        with self.assertRaises(ValueError):
            grouped.fit([0.0, 1.0], given={"g": ["a"]})

    def test_parameter_aliases_and_handles_are_unambiguous(self):
        with self.assertRaisesRegex(ValueError, "aliases"):
            Normal(free * Field("x") + free * Field("x"), free).fit([0.0, 1.0], given={"x": [0.0, 1.0]})
        shared = Normal(0.0, 1.0, name="shared")
        with self.assertRaisesRegex(ValueError, "same coefficient handle"):
            Normal(shared * Field("x") + shared * Field("z"), free).fit(
                [0.0, 1.0], given={"x": [0.0, 1.0], "z": [1.0, 0.0]}
            )
        fitted = Normal(free * Field("x") + free, 1.0).fit([0.0, 1.0, 2.0], given={"x": [0.0, 1.0, 2.0]})
        self.assertEqual(fitted.result.parameter_ids, ["coef:0", "coef:1"])
        self.assertEqual(fitted.result.samples("coef:0", n=2, rng=np.random.RandomState(0)).shape, (2,))
        with self.assertRaisesRegex(KeyError, "addressable"):
            fitted.result.samples(free, n=2)


class MixedModelContractTest(unittest.TestCase):
    def test_lmm_and_glmm_reject_unimplemented_coefficient_priors(self):
        prior = Normal(0.0, 1.0)
        with self.assertRaisesRegex(NotImplementedError, "coefficient priors"):
            Normal(prior * Field("x") + free + Group("g"), free).fit([0.0, 1.0], given={"x": [0.0, 1.0], "g": [0, 1]})
        with self.assertRaisesRegex(NotImplementedError, "coefficient priors"):
            Poisson(prior * Field("x") + free + Group("g")).fit([0, 1], given={"x": [0.0, 1.0], "g": [0, 1]})

    def test_glmm_support_and_nested_convergence_receipt(self):
        model = Bernoulli(free + Group("g"))
        with self.assertRaisesRegex(ValueError, "exactly 0 or 1"):
            model.fit([0.0, 2.0], given={"g": [0, 1]})
        fitted = model.fit(
            [0.0, 1.0, 0.0, 1.0],
            given={"g": [0, 0, 1, 1]},
            max_iter=1,
            inner_max_iter=1,
        )
        result = fitted.result
        self.assertFalse(result.converged)
        self.assertEqual(result.outer_iterations, 1)
        self.assertEqual(result.inner_iterations, (1,))
        self.assertEqual(len(result.inner_converged), 1)
        self.assertIsInstance(result.inner_converged[0], bool)
        self.assertEqual(result.termination_reason, "max_iterations")


class RegressionObjectiveContractTest(unittest.TestCase):
    def test_location_scale_gradient_matches_clipped_objective_and_failure_surfaces(self):
        model = Normal(0.0, free * Field("x"))

        def inspect_and_fail(fun, x0, jac, **_kwargs):
            theta = np.array([30.0])
            eps = 1.0e-5
            numeric = (fun(theta + eps) - fun(theta - eps)) / (2.0 * eps)
            self.assertAlmostEqual(float(jac(theta)[0]), float(numeric), places=6)
            return SimpleNamespace(
                success=False,
                fun=fun(theta),
                x=theta,
                message="forced optimizer failure",
                nit=1,
                jac=jac(theta),
            )

        with mock.patch("scipy.optimize.minimize", side_effect=inspect_and_fail):
            with self.assertRaisesRegex(RuntimeError, "forced optimizer failure"):
                model.fit([0.0, 1.0], given={"x": [1.0, 1.0]})

    def test_coordinate_descent_honors_budget_and_reports_nonconvergence(self):
        fitted = Normal(Laplace(0.0, 1.0) * Field("x") + free, free).fit(
            [0.0, 2.0, 4.0], given={"x": [0.0, 1.0, 2.0]}, max_its=1
        )
        self.assertEqual(fitted.result.iterations, 1)
        self.assertFalse(fitted.result.converged)
        self.assertEqual(fitted.result.termination_reason, "max_iterations")
        self.assertIsNone(fitted.result.cov)

    def test_quantile_uncertainty_and_ignored_semantics_are_explicit(self):
        fitted = Normal(free * Field("x") + free, free).fit([0.0, 1.0, 2.0], given={"x": [0.0, 1.0, 2.0]}, quantile=0.5)
        self.assertIsNone(fitted.result.cov)
        with self.assertRaisesRegex(NotImplementedError, "uncertainty"):
            fitted.result.samples()
        with self.assertRaisesRegex(NotImplementedError, "coefficient-prior"):
            Normal(Normal(0.0, 1.0) * Field("x") + free, free).fit([0.0, 1.0], given={"x": [0.0, 1.0]}, quantile=0.5)
        with self.assertRaisesRegex(NotImplementedError, "scale"):
            Normal(free * Field("x") + free, 1.0).fit([0.0, 1.0], given={"x": [0.0, 1.0]}, quantile=0.5)

    def test_glm_support_prior_scaling_and_scale_semantics(self):
        with self.assertRaisesRegex(ValueError, "exactly 0 or 1"):
            Bernoulli(free * Field("x")).fit([0.0, 0.5], given={"x": [0.0, 1.0]})
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            Poisson(free * Field("x")).fit([0.0, 1.5], given={"x": [0.0, 1.0]})
        with self.assertRaisesRegex(NotImplementedError, "StudentT"):
            Normal(StudentT(3.0, 0.0, 1.0) * Field("x"), 1.0).fit([0.0, 1.0], given={"x": [0.0, 1.0]})
        with self.assertRaisesRegex(NotImplementedError, "scale"):
            Normal(free * Field("x"), Gamma(2.0, 1.0)).fit([0.0, 1.0], given={"x": [0.0, 1.0]})
        with self.assertRaises(ValueError):
            Normal(free * Field("x"), 1.0).fit([0.0], given={"x": [1.0]}, max_its=0)

        prior1 = Normal(0.0, 1.0)
        prior2 = Normal(0.0, 1.0)
        low_noise = Normal(prior1 * Field("x"), 1.0).fit([10.0], given={"x": [1.0]})
        high_noise = Normal(prior2 * Field("x"), 3.0).fit([10.0], given={"x": [1.0]})
        self.assertGreater(low_noise.result.beta[0], high_noise.result.beta[0])
        self.assertAlmostEqual(low_noise.result.beta[0], 5.0)
        self.assertAlmostEqual(high_noise.result.beta[0], 1.0)

        fixed_scale = Normal(Laplace(0.0, 1.0) * Field("x"), 2.5).fit([0.0, 3.0], given={"x": [0.0, 1.0]})
        self.assertAlmostEqual(fixed_scale.result.sigma, 2.5)


if __name__ == "__main__":
    unittest.main()
