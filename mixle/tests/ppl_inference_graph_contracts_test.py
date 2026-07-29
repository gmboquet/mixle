"""Contracts for identity-preserving inference graphs and shaped posterior parameters."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mixle.inference.mcmc import MCMCResult
from mixle.ppl import (
    MVN,
    Bernoulli,
    Beta,
    Categorical,
    Dirichlet,
    Field,
    Gamma,
    Mix,
    Normal,
    Poisson,
    free,
    lower,
)
from mixle.ppl.inference import (
    _HYPER_LP,
    Posterior,
    _build_target,
    _collect_composite,
    _ensemble_worker,
    _finalize_chains,
    _grouped_target,
    _hmc_worker,
    _indexed_target,
    _laplace_covariance,
    _mcmc_worker,
    _nuts_worker,
    _ParameterDomainError,
    _prepare_target,
    _Slot,
    _slots_of,
    _to_value,
    ensemble_fit,
    hmc_fit,
    laplace_fit,
    map_fit,
    mcmc_fit,
    nuts_fit,
    vi_fit,
)
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution


class SharedParameterGraphContractTest(unittest.TestCase):
    def test_reused_scalar_prior_is_one_slot_in_flat_and_composite_models(self):
        shared = Normal(0.0, 3.0, name="shared")
        flat = Normal(shared, shared)
        slots, _ = _slots_of(flat, flat._family)
        self.assertEqual(len(slots), 1)
        self.assertIs(slots[0].handle, shared)
        self.assertEqual(slots[0].support, "positive")
        target, slots, _family, build, unpack, _seed = _build_target(flat, [2.0])
        values, _ = unpack([math.log(2.0)])
        distribution = build(values)
        self.assertEqual(distribution.mu, 2.0)
        self.assertEqual(distribution.sigma2, 4.0)
        self.assertTrue(math.isfinite(target([math.log(2.0)])))

        component = Normal(shared, 1.0)
        composite = Mix([component, component], weights=np.array([0.5, 0.5]))
        slots, rebuild = _collect_composite(composite)
        self.assertEqual(len(slots), 1)
        rebuilt = rebuild({slots[0].index: 2.0})
        first, second = rebuilt._args[0]
        self.assertEqual(first._args[0], 2.0)
        self.assertEqual(second._args[0], 2.0)


class StructuralPosteriorContractTest(unittest.TestCase):
    def test_vector_simplex_and_cholesky_handles_reassemble_declared_shapes(self):
        mean = free(2, name="mu")
        mean_slots, _ = _collect_composite(MVN(2, mean=mean, cov=np.eye(2)))
        mean_draws = np.array([[1.0, 2.0], [3.0, 4.0]])
        mean_post = Posterior(mean_slots, mean_draws, object())
        np.testing.assert_array_equal(mean_post.samples(mean), mean_draws)
        np.testing.assert_array_equal(mean_post.samples("mu"), mean_draws)

        probability = free(3, name="p", kind="simplex")
        probability_slots, _ = _collect_composite(Categorical(probability))
        probability_post = Posterior(probability_slots, np.array([[1.0, 2.0, 1.0]]), object())
        np.testing.assert_allclose(probability_post.samples(probability), [[0.25, 0.5, 0.25]])

        covariance = free(2, name="cov", kind="cholesky")
        covariance_slots, _ = _collect_composite(MVN(2, mean=np.zeros(2), cov=covariance))
        covariance_post = Posterior(covariance_slots, np.array([[2.0, 0.5, 1.0]]), object())
        expected = np.array([[[4.0, 1.0], [1.0, 1.25]]])
        np.testing.assert_allclose(covariance_post.samples(covariance), expected)


class TargetFailureContractTest(unittest.TestCase):
    def test_support_transforms_are_stable_and_overflow_is_a_typed_rejection(self):
        with self.assertRaises(_ParameterDomainError):
            _to_value("positive", 10_000.0)
        value, log_jacobian = _to_value("unit", 10_000.0)
        self.assertGreater(value, 0.0)
        self.assertLess(value, 1.0)
        self.assertTrue(math.isfinite(log_jacobian))

        model = Normal(free, free)
        target, *_ = _build_target(model, [0.0, 1.0])
        self.assertEqual(target([0.0, 10_000.0]), -1e300)

    def test_likelihood_implementation_failures_are_not_disguised_as_domain_rejection(self):
        model = Normal(free, 1.0)
        target, *_ = _build_target(model, [0.0, 1.0])
        with patch.object(GaussianDistribution, "seq_log_density", side_effect=TypeError("scorer defect")):
            with self.assertRaisesRegex(RuntimeError, "likelihood evaluation"):
                target([0.0])
        with patch.object(GaussianDistribution, "seq_log_density", return_value=np.array([np.inf, 0.0])):
            with self.assertRaisesRegex(FloatingPointError, "invalid value"):
                target([0.0])


class ConstraintSemanticsContractTest(unittest.TestCase):
    def test_noncentered_constraints_see_declared_values_and_hard_constraints_stay_hard(self):
        location = Normal(5.0, 2.0, name="location").noncentered()
        model = Normal(location, 1.0)
        _target, _grad, _slots, _build, _mean, _std, feasible = _prepare_target(
            model,
            [5.0],
            location > 4.0,
            None,
            want_grad=False,
            numpy_only=True,
        )
        self.assertTrue(feasible([0.0]))

        parameter = Normal(0.0, 3.0, name="parameter")
        constrained = Normal(parameter, 1.0)
        hard = parameter > 0.0
        soft = parameter.eq(1.0)
        mixed_target, _grad, _slots, _build, _mean, _std, mixed_feasible = _prepare_target(
            constrained,
            [1.0],
            [hard, soft],
            None,
            want_grad=False,
            numpy_only=True,
        )
        hard_target, *_ = _prepare_target(
            constrained,
            [1.0],
            hard,
            None,
            want_grad=False,
            numpy_only=True,
        )
        self.assertFalse(mixed_feasible([-1.0]))
        self.assertEqual(mixed_target([-1.0]), -math.inf)
        self.assertLess(mixed_target([2.0]), hard_target([2.0]))


class GroupedParameterizationContractTest(unittest.TestCase):
    def test_grouped_gamma_prior_uses_public_shape_rate_parameterization(self):
        value = 1.7
        declared = Gamma(2.5, 4.0)
        expected = lower(declared, target="dist").log_density(value)
        actual = _HYPER_LP["Gamma"](value, [2.5, 4.0], np)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_grouped_map_uses_constrained_density_and_public_gamma_rate(self):
        tau = Gamma(2.5, 4.0, name="tau")
        model = Normal(Normal(0.0, tau, name="theta").each(), 1.0)
        groups = [[1.0], [2.0]]
        map_target, *_ = _grouped_target(model, groups, want_grad=False, jacobian=False)
        sampling_target, *_ = _grouped_target(model, groups, want_grad=False, jacobian=True)
        coordinates = np.array([math.log(2.0), 1.0, 2.0])
        likelihood = sum(lower(Normal(theta, 1.0), target="dist").log_density(y) for theta, y in zip([1, 2], [1, 2]))
        group_prior = sum(lower(Normal(0.0, 2.0), target="dist").log_density(theta) for theta in [1, 2])
        hyper_prior = lower(tau, target="dist").log_density(2.0)
        self.assertAlmostEqual(map_target(coordinates), likelihood + group_prior + hyper_prior, places=12)
        self.assertAlmostEqual(
            sampling_target(coordinates) - map_target(coordinates),
            coordinates[0],
            places=12,
        )

    def test_grouped_map_retains_the_fitted_group_vector(self):
        population = Normal(0.0, 2.0, name="theta").each()
        fitted = Normal(population, 1.0).fit([[1.0], [2.0]], how="map")
        np.testing.assert_allclose(fitted.result.estimate(population), [0.8, 1.6], atol=1e-5)
        with self.assertRaises(NotImplementedError):
            fitted.result.samples(population)


class MultiChainContractTest(unittest.TestCase):
    def test_raw_result_retains_every_chain_and_aggregates_acceptance(self):
        slot = _Slot(0, None, False, "mu", None, "real")
        first_samples = [[value] for value in np.linspace(-0.2, 0.2, 20)]
        second_samples = [[value] for value in np.linspace(-0.1, 0.3, 20)]
        first = MCMCResult(first_samples, np.zeros(20), np.array([True] * 10 + [False] * 10))
        second = MCMCResult(second_samples, np.zeros(20), np.array([True] * 15 + [False] * 5))
        fitted = _finalize_chains(
            Normal(free, 1.0),
            [slot],
            [first, second],
            lambda values: GaussianDistribution(values[0], 1.0),
        )
        self.assertEqual(len(fitted.result.raw.chains), 2)
        self.assertEqual(fitted.result.raw.samples.shape, (40, 1))
        self.assertAlmostEqual(fitted.result.acceptance_rate, 0.625)
        self.assertEqual(fitted.result.n_chains, 2)

    def test_parallel_workers_receive_the_declared_missing_data_policy(self):
        for worker in (_mcmc_worker, _hmc_worker, _nuts_worker, _ensemble_worker):
            with self.subTest(worker=repr(worker.__name__)):
                with patch("mixle.ppl.inference._prepare_target", side_effect=RuntimeError("stop")) as prepare:
                    with self.assertRaisesRegex(RuntimeError, "stop"):
                        worker(7, Normal(free, 1.0), [0.0], {"missing": "marginalize"})
                self.assertEqual(prepare.call_args.kwargs["missing"], "marginalize")


class InferenceControlContractTest(unittest.TestCase):
    def setUp(self):
        self.model = Normal(Normal(0.0, 2.0, name="mu"), 1.0)
        self.data = [0.0, 1.0]

    def test_sampler_controls_reject_invalid_values_before_sampling(self):
        invalid = [
            (mcmc_fit, {"draws": 0}),
            (mcmc_fit, {"burn": -1}),
            (mcmc_fit, {"thin": 1.5}),
            (mcmc_fit, {"chains": 0}),
            (mcmc_fit, {"scale": np.inf}),
            (hmc_fit, {"step_size": np.nan}),
            (hmc_fit, {"num_steps": 0}),
            (nuts_fit, {"target_accept": 1.0}),
            (nuts_fit, {"max_tree_depth": 0}),
            (ensemble_fit, {"walkers": 5}),
        ]
        for fitter, controls in invalid:
            with (
                self.subTest(fitter=repr(fitter.__name__), controls=repr(controls)),
                self.assertRaises((TypeError, ValueError)),
            ):
                fitter(self.model, self.data, **controls)

    def test_map_controls_and_unsuccessful_termination_are_enforced(self):
        with self.assertRaises(ValueError):
            map_fit(self.model, self.data, max_iter=0)
        with self.assertRaises(ValueError):
            map_fit(self.model, self.data, tol=np.nan)
        failed = SimpleNamespace(success=False, fun=1.0, x=np.array([0.0]), message="iteration limit")
        with patch("scipy.optimize.minimize", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "iteration limit"):
                map_fit(self.model, self.data)

    def test_laplace_uses_explicit_posterior_contract_and_hessian_receipt(self):
        fitted = laplace_fit(self.model, self.data, draws=20, rng=np.random.RandomState(3))
        self.assertEqual(fitted.result.raw.hessian["rank"], 1)
        self.assertEqual(fitted.result.raw.hessian["regularization"], 0.0)
        self.assertTrue(fitted.result.raw.optimizer["success"])
        with self.assertRaisesRegex(NotImplementedError, "hard-truncated"):
            laplace_fit(self.model, self.data, constraints=self.model._args[0] > 0.0)
        with self.assertRaisesRegex(RuntimeError, "not positive definite"):
            _laplace_covariance(np.array([[-1.0]]))

    def test_vi_rejects_unexecuted_variants_and_records_the_executed_algorithm(self):
        with self.assertRaises(ValueError):
            vi_fit(self.model, self.data, samples=1)
        with self.assertRaises(ValueError):
            vi_fit(self.model, self.data, batch_size=len(self.data))
        with patch("mixle.ppl.autograd.grad_target", return_value=None):
            with self.assertRaisesRegex(NotImplementedError, "fullrank"):
                vi_fit(self.model, self.data, family="fullrank")
        fitted = vi_fit(
            self.model,
            self.data,
            samples=20,
            mc=2,
            steps=5,
            max_iter=5,
            rng=np.random.RandomState(4),
        )
        self.assertEqual(fitted.result.raw.algorithm, "advi_meanfield")
        self.assertEqual(fitted.result.raw.iterations, 5)
        self.assertEqual(fitted.result.raw.termination_reason, "fixed_steps_completed")


class ConjugateSupportContractTest(unittest.TestCase):
    def test_discrete_conjugate_routes_reject_coercible_off_support_observations(self):
        invalid = [
            (Poisson(Gamma(2.0, 1.0)), [1.2]),
            (Bernoulli(Beta(2.0, 2.0)), [0.5]),
            (Categorical(Dirichlet(np.ones(3))), [1.2]),
        ]
        for model, data in invalid:
            with self.subTest(model=repr(model._family.name)), self.assertRaisesRegex(ValueError, "requires"):
                model.fit(data, how="conjugate")

    def test_all_free_normal_uses_a_proper_default_prior(self):
        fitted = Normal(free, free).fit([2.0, 2.0], how="conjugate")
        self.assertTrue(fitted.result.prior["proper"])
        self.assertGreater(fitted.result.prior["kappa"], 0.0)
        self.assertGreater(fitted.result.prior["alpha"], 0.0)
        self.assertGreater(fitted.result.prior["beta"], 0.0)
        self.assertTrue(np.isfinite(fitted.params["sd"]))
        with self.assertRaises(ValueError):
            Normal(free, free).fit([2.0], how="conjugate", prior={"kappa": 0.0})


class HierarchicalPriorContractTest(unittest.TestCase):
    def test_empirical_bayes_receipt_discloses_initialization_and_conditional_draws(self):
        population = Normal(100.0, 0.5, name="theta").each()
        fitted = Normal(population, 1.0).fit([[0.0, 0.1], [1.0, 1.1]], how="hierarchical")
        receipt = fitted.result.hyper
        self.assertEqual(receipt["procedure"], "empirical_bayes_em")
        self.assertEqual(receipt["initialization"]["location"], 100.0)
        self.assertEqual(receipt["initialization"]["scale"], 0.5)
        self.assertTrue(receipt["conditional_group_posterior"])
        self.assertFalse(receipt["includes_hyperparameter_uncertainty"])
        draws = fitted.result.samples(n=7, rng=np.random.RandomState(2))
        self.assertEqual(draws.shape, (7, 2))
        self.assertFalse(np.array_equal(draws[0], fitted.result.group_means))

    def test_degenerate_binary_groups_remain_proper_under_declared_beta_prior(self):
        population = Beta(2.0, 3.0, name="p").each()
        fitted = Bernoulli(population).fit([[0.0, 0.0], [1.0, 1.0]], how="hierarchical")
        self.assertTrue(np.all((fitted.result.group_means > 0.0) & (fitted.result.group_means < 1.0)))
        self.assertEqual(fitted.result.samples(n=3, rng=np.random.RandomState(3)).shape, (3, 2))


class IndexedLatentContractTest(unittest.TestCase):
    def test_numeric_indices_are_exact_and_bounded(self):
        theta = free(2, name="theta")
        model = Normal(theta[Field("group")], 1.0)
        for labels in ([0.0, 1.2], [0, -1], [0, 2]):
            with self.subTest(labels=repr(labels)), self.assertRaisesRegex(ValueError, "exact integer|outside"):
                model.fit([0.0, 1.0], given={"group": labels})

    def test_symbolic_labels_and_original_vector_handle_survive_map(self):
        theta = free(2, name="theta")
        fitted = Normal(theta[Field("group")], 1.0).fit(
            [3.0, -2.0, 3.1, -1.9],
            given={"group": ["beta", "alpha", "beta", "alpha"]},
        )
        self.assertEqual(fitted.result.group_labels, ("beta", "alpha"))
        self.assertEqual(fitted.result.group_index, {"beta": 0, "alpha": 1})
        np.testing.assert_array_equal(fitted.result.estimate(theta), fitted.result.latents["theta"])
        with self.assertRaisesRegex(NotImplementedError, "point estimates"):
            fitted.result.samples(theta)
        self.assertTrue(fitted.result.optimizer["success"])

    def test_mcmc_returns_shaped_samples_keyed_by_original_vector_handle(self):
        theta = free(2, name="theta")
        fitted = Normal(theta[Field("group")], 1.0).fit(
            [0.0, 1.0, 0.1, 0.9],
            given={"group": ["left", "right", "left", "right"]},
            how="mcmc",
            draws=20,
            burn=10,
            rng=np.random.RandomState(7),
        )
        self.assertEqual(fitted.result.samples(theta).shape, (20, 2))
        self.assertEqual(fitted.result.group_labels, ("left", "right"))

    def test_indexed_constraints_and_sampler_controls_are_not_discarded(self):
        theta = free(2, name="theta")
        model = Normal(theta[Field("group")], 1.0)
        fitted = model.fit(
            [1.0, 2.0, 1.1, 2.1],
            given={"group": [0, 1, 0, 1]},
            constraints=theta > 0.0,
        )
        self.assertTrue(np.all(fitted.result.estimate(theta) > 0.0))
        with self.assertRaises(ValueError):
            model.fit([1.0, 2.0], given={"group": [0, 1]}, how="mcmc", draws=0)

    def test_map_omits_and_mcmc_includes_support_transform_jacobian(self):
        theta = free(2, name="theta", support="positive")
        model = Normal(theta[Field("group")], 1.0)
        given = {"group": [0, 1]}
        map_target, *_ = _indexed_target(model, [1.0, 2.0], given, jacobian=False)
        sampling_target, *_ = _indexed_target(model, [1.0, 2.0], given, jacobian=True)
        coordinates = np.array([math.log(1.5), math.log(2.5)])
        self.assertAlmostEqual(
            sampling_target(coordinates) - map_target(coordinates),
            float(np.sum(coordinates)),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
