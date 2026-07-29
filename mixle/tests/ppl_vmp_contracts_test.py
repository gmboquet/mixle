"""Release contracts for variational message passing."""

import unittest

import numpy as np

from mixle.ppl import Graph, Mix, Normal, Poisson, free
from mixle.ppl.vmp import mixture_vmp


class VMPGraphContractTest(unittest.TestCase):
    def test_refit_is_fresh_and_prior_result_is_a_snapshot(self):
        mu = Normal(0.0, 1.0, name="mu")
        graph = Graph().observe(Normal(mu, 1.0), [10.0])
        first = graph.fit()
        first_before = first.posterior(mu)
        second = graph.fit()
        self.assertAlmostEqual(first_before["mean"], 5.0)
        self.assertEqual(first.posterior(mu), first_before)
        self.assertEqual(second.posterior(mu), first_before)

    def test_graph_validates_observations_and_controls(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            Graph().fit()
        mu = Normal(0.0, 1.0)
        for data in ([], [0.0, np.nan], [[0.0], [1.0]]):
            with self.subTest(data=repr(data)):
                with self.assertRaises(ValueError):
                    Graph().observe(Normal(mu, 1.0), data).fit()
        graph = Graph().observe(Normal(mu, 1.0), [0.0])
        for kwargs in ({"max_its": 0}, {"max_its": 1.5}, {"tol": 0.0}, {"tol": np.inf}):
            with self.subTest(kwargs=repr(kwargs)):
                with self.assertRaises(ValueError):
                    graph.fit(**kwargs)

    def test_nonconverged_result_is_explicit(self):
        mu = Normal(0.0, 1.0)
        result = Graph().observe(Normal(mu, 1.0), [2.0]).fit(max_its=1)
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.termination_reason, "max_iterations")


class MixtureVMPContractTest(unittest.TestCase):
    def test_route_rejects_non_gaussian_or_partially_declared_models(self):
        with self.assertRaisesRegex(NotImplementedError, "only"):
            Mix([Poisson(free), Poisson(free)]).fit([0, 1, 2], how="vmp")
        with self.assertRaisesRegex(NotImplementedError, "only"):
            Mix([Normal(0.0, free), Normal(1.0, free)]).fit([0.0, 1.0], how="vmp")
        with self.assertRaisesRegex(NotImplementedError, "only"):
            Mix([Normal(free, free), Normal(free, free)], [0.5, 0.5]).fit([0.0, 1.0], how="vmp")

    def test_complete_input_and_termination_validation(self):
        invalid_calls = [
            ([], 1, {}),
            ([0.0, np.inf], 1, {}),
            ([0.0], 0, {}),
            ([0.0], 2, {}),
            ([0.0], 1, {"max_its": 0}),
            ([0.0], 1, {"tol": 0.0}),
            ([0.0], 1, {"s0": -1.0}),
            ([0.0], 1, {"a0": 0.0}),
            ([0.0], 1, {"b0": np.nan}),
            ([0.0], 1, {"alpha0": -1.0}),
        ]
        for data, k, kwargs in invalid_calls:
            with self.subTest(data=repr(data), k=repr(k), kwargs=repr(kwargs)):
                with self.assertRaises(ValueError):
                    mixture_vmp(data, k, **kwargs)
        fit = mixture_vmp([0.0, 1.0], 1, max_its=1)
        self.assertFalse(fit.result.converged)
        self.assertEqual(fit.result.iterations, 1)
        self.assertEqual(fit.result.termination_reason, "max_iterations")
        self.assertEqual(fit.result.responsibility_normalizer_trace.shape, fit.result.elbo_trace.shape)

    def test_predictive_integrates_variational_parameter_uncertainty(self):
        fitted = Mix([Normal(free, free)]).fit([0.0], how="vmp", rng=np.random.RandomState(1))
        component = fitted.result.components[0]
        self.assertGreater(component["predictive_variance"], component["plugin_variance"])
        self.assertIs(fitted.result.plugin_distribution, fitted.dist)
        draws = np.asarray(fitted.predict(20_000, rng=np.random.RandomState(2)))
        self.assertGreater(float(draws.var()), 1.25 * component["plugin_variance"])


if __name__ == "__main__":
    unittest.main()
