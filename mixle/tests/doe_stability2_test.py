"""Second pass over the mixle/doe/ bug backlog (the first six -- propagate/sensitivity/bayesopt NaN/
multifidelity hang/n_candidates validation -- are a separate, already-shipped PR). Seven more real
bugs found and fixed here.

1. turbo_minimize's TuRBO restart branch unconditionally drew a full n_init-point Latin-hypercube
   batch on trust-region collapse, overshooting max_evals by up to n_init real objective calls; the
   very first design (before the loop) had the same problem for max_evals < n_init.
2. trust_region._thompson_batch silently returned fewer than q picks when q > cand.shape[0].
3. batch.propose_local_penalization silently duplicated points when q > n_candidates (once every
   candidate's merit hits -inf, argmax deterministically returns index 0 again).
4. calibrate.calibrate's no-discrepancy branch lacked the noise floor its discrepancy-branch sibling
   has, an inconsistency (not independently crash-prone, but a real defensive gap).
5. sensitivity.fast_indices's Tarantola bias correction could silently distort S1 (even flip its
   sign) if harmonics was large relative to n, then get masked by the trailing clip(0, 1).
6. analysis.design_diagnostics could silently return NaN for max_correlation when a design column
   has zero variance (np.corrcoef produces 0/0 = NaN for that column).
7. BayesianOptimizer.ask() under-counted in-flight (asked-but-not-told) points -- gated on
   n_observations (told count) instead of _init_used (dispensed count) -- causing duplicate
   initial-design draws in the parallel/async campaigns this class explicitly supports. Fixing this
   surfaced an EIGHTH bug: _propose_one crashed with an opaque "zero-size array" ValueError when
   called with zero observations (ask(q > n_init) before any tell()), now a clear, named error.
"""

import importlib.util
import unittest

import numpy as np

from mixle.doe import design_diagnostics, fast_indices, polynomial_features
from mixle.doe.batch import propose_local_penalization
from mixle.doe.optimizer import BayesianOptimizer
from mixle.doe.trust_region import _thompson_batch, turbo_minimize

# The default GP surrogate (mixle.models.gaussian_process.GaussianProcessRegressor) is torch-only;
# tests that actually run a fit need to be skipped in a torch-free environment like everything else
# in the suite does.
HAS_TORCH = importlib.util.find_spec("torch") is not None


class TurboBudgetOvershootTest(unittest.TestCase):
    def test_max_evals_less_than_n_init_raises_instead_of_overshooting(self):
        with self.assertRaises(ValueError):
            turbo_minimize(lambda x: float(np.sum(x**2)), [(0.0, 1.0)] * 3, n_init=8, max_evals=3)

    def test_a_collapse_near_the_budget_does_not_overshoot_max_evals(self):
        # a deliberately tiny max_evals relative to n_init forces the restart branch to fire near
        # the end of the run; the total evaluation count must never exceed max_evals.
        result = turbo_minimize(
            lambda x: float(np.sum(x**2)), [(-1.0, 1.0)] * 2, n_init=4, max_evals=6, seed=0, batch_size=1
        )
        self.assertLessEqual(result["Y"].shape[0], 6)

    def test_a_normal_run_still_converges_reasonably(self):
        result = turbo_minimize(lambda x: float(np.sum(x**2)), [(-2.0, 2.0)] * 2, n_init=6, max_evals=30, seed=0)
        self.assertLess(result["y"], 1.0)


class ThompsonBatchCandidateValidationTest(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_q_greater_than_candidates_raises_instead_of_silently_truncating(self):
        from mixle.doe.bayesopt import _fit_surrogate

        rng = np.random.RandomState(0)
        xs = rng.uniform(0, 1, size=(5, 2))
        ys = rng.normal(size=5)
        gp = _fit_surrogate(xs, ys, None, None)
        cand = rng.uniform(0, 1, size=(3, 2))
        with self.assertRaises(ValueError):
            _thompson_batch(gp, xs, ys, cand, q=5, rng=rng)


class LocalPenalizationDuplicateTest(unittest.TestCase):
    def test_q_greater_than_n_candidates_raises_instead_of_duplicating(self):
        rng = np.random.RandomState(0)
        x = rng.uniform(0, 1, size=(5, 2))
        y = rng.normal(size=5)
        with self.assertRaises(ValueError):
            propose_local_penalization(x, y, [(0.0, 1.0), (0.0, 1.0)], q=10, n_candidates=5, seed=0)

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_q_within_n_candidates_still_works(self):
        rng = np.random.RandomState(0)
        x = rng.uniform(0, 1, size=(5, 2))
        y = rng.normal(size=5)
        batch = propose_local_penalization(x, y, [(0.0, 1.0), (0.0, 1.0)], q=3, n_candidates=64, seed=0)
        self.assertEqual(batch.shape, (3, 2))


class FastIndicesHarmonicsValidationTest(unittest.TestCase):
    def test_too_many_harmonics_for_n_raises_instead_of_silently_distorting(self):
        with self.assertRaises(ValueError):
            fast_indices(lambda x: np.sum(x, axis=1), [(0, 1), (0, 1)], n=8, harmonics=6)

    def test_a_well_posed_configuration_still_works(self):
        rng = np.random.RandomState(0)

        def f(x):
            return x[:, 0] + 0.01 * rng.standard_normal(x.shape[0])

        report = fast_indices(f, [(0, 1), (0, 1)], n=300, harmonics=4, seed=0)
        self.assertGreater(report["S1"][0], report["S1"][1])  # x0 dominates, x1 is noise


class DesignDiagnosticsZeroVarianceColumnTest(unittest.TestCase):
    def test_a_constant_column_does_not_produce_nan_max_correlation(self):
        rng = np.random.RandomState(0)
        design = np.column_stack([rng.uniform(-1, 1, 20), np.full(20, 0.5)])  # second factor never varies
        diag = design_diagnostics(design, polynomial_features(1))
        self.assertTrue(np.isfinite(diag["max_correlation"]))

    def test_a_healthy_design_is_unaffected(self):
        rng = np.random.RandomState(1)
        design = rng.uniform(-1, 1, size=(30, 3))
        diag = design_diagnostics(design, polynomial_features(1))
        self.assertTrue(np.isfinite(diag["max_correlation"]))
        self.assertGreaterEqual(diag["max_correlation"], 0.0)


class AskBeforeTellDuplicateInitPointsTest(unittest.TestCase):
    def test_repeated_ask_before_any_tell_does_not_redispense_the_same_init_point(self):
        opt = BayesianOptimizer([(0.0, 1.0), (0.0, 1.0)], n_init=3, seed=0)
        points = [opt.ask(1) for _ in range(3)]
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                self.assertFalse(np.allclose(points[i], points[j]))

    def test_asking_past_the_exhausted_init_design_with_no_tell_raises_a_clear_error(self):
        # rather than crash on the opaque "zero-size array" numpy error the fix for bug #7 surfaced.
        opt = BayesianOptimizer([(0.0, 1.0), (0.0, 1.0)], n_init=2, seed=0)
        opt.ask(1)
        opt.ask(1)
        with self.assertRaises(ValueError):
            opt.ask(1)

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_the_normal_ask_tell_ask_workflow_still_works(self):
        opt = BayesianOptimizer([(0.0, 1.0), (0.0, 1.0)], n_init=3, seed=0)
        for _ in range(3):
            x = opt.ask(1)
            opt.tell(x, float(np.sum(x**2)))
        x_gp = opt.ask(1)
        self.assertEqual(x_gp.shape, (2,))
        self.assertEqual(opt.n_observations, 3)


if __name__ == "__main__":
    unittest.main()
