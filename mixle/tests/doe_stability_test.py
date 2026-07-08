"""Bug-hunting pass over mixle/doe/: six real crash/hang bugs found and fixed here.

1. propagate.unscented_transform crashed with LinAlgError on a singular/near-singular input
   covariance (a fixed/zero-variance input dimension) -- a realistic input, not contrived.
2. sensitivity.morris_screening crashed with ZeroDivisionError for levels=1.
3. bayesopt._fit_surrogate's default-GP amplitude/noise silently became NaN when fit with zero
   observations (`float(np.std([])) or 1.0` -- nan is truthy, so the `or` fallback never caught it) --
   a real, documented path: BayesianOptimizer.ask(q) with q > n_init before any tell().
4. multifidelity.multi_fidelity_minimize hung forever (infinite loop) given a zero-cost fidelity: the
   per-cost score divides by that fidelity's cost, so it wins every round, and the budget accounting
   never advances.
5. Several propose_* functions (bayesopt.propose_next/propose_batch via the shared _propose_one,
   bayesopt.propose_knowledge_gradient, entropy.propose_mes, active.propose_active_learning,
   multifidelity.multi_fidelity_minimize) crashed with an unhelpful "argmax of an empty sequence" on
   n_candidates <= 0 instead of a clear ValueError -- inconsistent with propose_batch's own q <= 0
   validation.
6. active.expected_information_gain_nmc crashed (ValueError from a zero-size reduction, or
   ZeroDivisionError from a plain int division) on n_inner=0 or n_outer=0.
"""

import importlib.util
import unittest

import numpy as np

from mixle.doe import (
    expected_information_gain_nmc,
    morris_screening,
    multi_fidelity_minimize,
    propose_active_learning,
    propose_knowledge_gradient,
    propose_mes,
    propose_next,
    unscented_transform,
)

# propose_next's default GP surrogate (mixle.models.gaussian_process.GaussianProcessRegressor) is
# torch-only; the n_candidates<=0 validation tests never reach the GP fit (they raise first), but the
# two tests that actually run a fit need to be skipped in a torch-free environment like everything
# else in the suite does.
HAS_TORCH = importlib.util.find_spec("torch") is not None


class UnscentedTransformSingularCovarianceTest(unittest.TestCase):
    def test_a_zero_variance_input_dimension_does_not_crash(self):
        cov = [[1.0, 0.0], [0.0, 0.0]]  # one dimension is a fixed/known constant
        mean, out_cov = unscented_transform(lambda x: x, [0.0, 0.0], cov)
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(np.isfinite(out_cov)))

    def test_still_correct_on_a_well_conditioned_covariance(self):
        # regression guard: the jitter fallback must not perturb a healthy input's result meaningfully.
        cov = np.array([[2.0, 0.3], [0.3, 1.5]])
        mean, out_cov = unscented_transform(lambda x: x, [1.0, -1.0], cov)
        np.testing.assert_allclose(mean, [1.0, -1.0], atol=1e-6)
        np.testing.assert_allclose(out_cov, cov, atol=1e-4)


class MorrisScreeningLevelsValidationTest(unittest.TestCase):
    def test_levels_1_raises_a_clear_error_instead_of_zerodivisionerror(self):
        with self.assertRaises(ValueError):
            morris_screening(lambda x: float(np.sum(x)), [(0, 1), (0, 1)], levels=1, trajectories=2)

    def test_levels_2_still_works(self):
        report = morris_screening(lambda x: float(np.sum(x)), [(0, 1), (0, 1)], levels=2, trajectories=3)
        self.assertEqual(len(report["mu_star"]), 2)


class ProposeNCandidatesValidationTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.x = rng.uniform(0, 1, size=(5, 2))
        self.y = rng.normal(size=5)
        self.bounds = [(0.0, 1.0), (0.0, 1.0)]

    def test_propose_next_rejects_non_positive_n_candidates(self):
        with self.assertRaises(ValueError):
            propose_next(self.x, self.y, self.bounds, n_candidates=0)

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_propose_next_still_works_with_a_normal_n_candidates(self):
        point = propose_next(self.x, self.y, self.bounds, n_candidates=16, seed=0)
        self.assertEqual(point.shape, (2,))

    def test_propose_knowledge_gradient_rejects_non_positive_n_candidates(self):
        with self.assertRaises(ValueError):
            propose_knowledge_gradient(self.x, self.y, self.bounds, n_candidates=0)

    def test_propose_mes_rejects_non_positive_n_candidates(self):
        with self.assertRaises(ValueError):
            propose_mes(self.x, self.y, self.bounds, n_candidates=0)

    def test_propose_active_learning_rejects_non_positive_n_candidates(self):
        with self.assertRaises(ValueError):
            propose_active_learning(self.x, self.y, self.bounds, n_candidates=0)


class MultiFidelityZeroCostTest(unittest.TestCase):
    def test_a_zero_cost_fidelity_raises_instead_of_hanging(self):
        def objective(x, s):
            return float(np.sum(x))

        with self.assertRaises(ValueError):
            multi_fidelity_minimize(
                objective, [(0.0, 1.0)], fidelities=(0.0, 1.0), costs=(0.0, 1.0), max_cost=5.0, n_init=1
            )

    def test_non_positive_n_candidates_also_raises(self):
        def objective(x, s):
            return float(np.sum(x))

        with self.assertRaises(ValueError):
            multi_fidelity_minimize(objective, [(0.0, 1.0)], n_candidates=0, max_cost=5.0, n_init=1)


class ExpectedInformationGainZeroBudgetTest(unittest.TestCase):
    def _sampler(self, rng, n):
        return rng.normal(size=(n, 1))

    def _log_likelihood(self, thetas, y):
        return -0.5 * np.sum((thetas - y) ** 2, axis=-1)

    def _simulate(self, theta, rng):
        return theta + rng.normal(scale=0.1, size=theta.shape)

    def test_n_inner_zero_raises_a_clear_error(self):
        with self.assertRaises(ValueError):
            expected_information_gain_nmc(
                self._sampler, self._log_likelihood, self._simulate, n_outer=4, n_inner=0, seed=0
            )

    def test_n_outer_zero_raises_a_clear_error(self):
        with self.assertRaises(ValueError):
            expected_information_gain_nmc(
                self._sampler, self._log_likelihood, self._simulate, n_outer=0, n_inner=4, seed=0
            )

    def test_positive_budgets_still_compute_a_finite_value(self):
        eig = expected_information_gain_nmc(
            self._sampler, self._log_likelihood, self._simulate, n_outer=8, n_inner=8, seed=0
        )
        self.assertTrue(np.isfinite(eig))


class AskBeforeTellDoesNotProduceNanSurrogateTest(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_a_batch_ask_larger_than_n_init_before_any_tell_does_not_nan_out(self):
        # the exact real path that hit the `nan or 1.0` bug: propose a batch of candidates with an
        # empty y (before any objective evaluation is told back), via active_learning_design's
        # underlying machinery -- exercised indirectly through the documented empty-observation path.
        from mixle.doe.bayesopt import _fit_surrogate

        gp = _fit_surrogate(np.empty((0, 2)), np.empty(0), None, None)
        self.assertTrue(np.isfinite(gp.amplitude))
        self.assertTrue(np.isfinite(gp.noise))


if __name__ == "__main__":
    unittest.main()
