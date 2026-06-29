"""Tests for the DoE Bayesian-optimization loop (WS-E).

``expected_improvement`` is torch-free and tested directly; the GP-surrogate ``propose_next`` /
``minimize`` paths require torch and are skipped when it is unavailable.
"""

import importlib.util
import unittest

import numpy as np

from mixle.doe import (
    available_acquisitions,
    expected_improvement,
    minimize,
    probability_of_improvement,
    propose_batch,
    propose_next,
    register_acquisition,
    upper_confidence_bound,
)
from mixle.doe.bayesopt import _get_acquisition

HAS_TORCH = importlib.util.find_spec("torch") is not None


class ExpectedImprovementTest(unittest.TestCase):
    def test_zero_std_gives_zero_ei(self):
        ei = expected_improvement(mean=np.array([0.0, 1.0]), std=np.array([0.0, 0.0]), best=0.5)
        np.testing.assert_array_equal(ei, np.zeros(2))

    def test_minimize_rewards_lower_mean(self):
        # Equal std: a smaller predicted mean is more improving under minimization.
        ei = expected_improvement(mean=np.array([-1.0, 0.0, 1.0]), std=np.array([1.0, 1.0, 1.0]), best=0.0)
        self.assertGreater(ei[0], ei[1])
        self.assertGreater(ei[1], ei[2])

    def test_maximize_rewards_higher_mean(self):
        ei = expected_improvement(
            mean=np.array([-1.0, 0.0, 1.0]), std=np.array([1.0, 1.0, 1.0]), best=0.0, maximize=True
        )
        self.assertGreater(ei[2], ei[1])
        self.assertGreater(ei[1], ei[0])

    def test_nonnegative_and_increases_with_uncertainty(self):
        # At the incumbent mean (improve=0), EI grows with predictive std.
        low = expected_improvement(mean=np.array([0.0]), std=np.array([0.3]), best=0.0)
        high = expected_improvement(mean=np.array([0.0]), std=np.array([1.5]), best=0.0)
        self.assertTrue(np.all(low >= 0.0) and np.all(high >= 0.0))
        self.assertGreater(high[0], low[0])


class ProbabilityOfImprovementTest(unittest.TestCase):
    def test_zero_std_is_deterministic_indicator(self):
        # std=0: PI is 1 where the mean strictly improves on best, else 0.
        pi = probability_of_improvement(mean=np.array([-1.0, 1.0]), std=np.array([0.0, 0.0]), best=0.0)
        np.testing.assert_array_equal(pi, np.array([1.0, 0.0]))

    def test_bounded_in_unit_interval_and_monotone(self):
        pi = probability_of_improvement(mean=np.array([-1.0, 0.0, 1.0]), std=np.array([1.0, 1.0, 1.0]), best=0.0)
        self.assertTrue(np.all(pi >= 0.0) and np.all(pi <= 1.0))
        self.assertGreater(pi[0], pi[1])  # lower mean is more likely to improve (minimization)
        self.assertGreater(pi[1], pi[2])

    def test_maximize_flips_direction(self):
        pi = probability_of_improvement(mean=np.array([-1.0, 1.0]), std=np.array([1.0, 1.0]), best=0.0, maximize=True)
        self.assertGreater(pi[1], pi[0])


class UpperConfidenceBoundTest(unittest.TestCase):
    def test_minimization_merit_prefers_low_mean_and_high_std(self):
        # Merit is maximized; for minimization that is kappa*std - mean.
        merit = upper_confidence_bound(mean=np.array([0.0, 0.0]), std=np.array([0.1, 2.0]), kappa=2.0)
        self.assertGreater(merit[1], merit[0])  # more uncertain point is more attractive (exploration)
        lo = upper_confidence_bound(mean=np.array([-1.0]), std=np.array([1.0]), kappa=1.0)
        hi = upper_confidence_bound(mean=np.array([1.0]), std=np.array([1.0]), kappa=1.0)
        self.assertGreater(lo[0], hi[0])  # lower mean -> higher merit under minimization

    def test_maximization_uses_optimistic_upper_bound(self):
        merit = upper_confidence_bound(mean=np.array([0.0, 1.0]), std=np.array([1.0, 1.0]), kappa=2.0, maximize=True)
        self.assertGreater(merit[1], merit[0])


class AcquisitionRegistryTest(unittest.TestCase):
    def test_builtin_names_and_aliases_resolve(self):
        names = available_acquisitions()
        for expected in (
            "expected_improvement",
            "ei",
            "probability_of_improvement",
            "pi",
            "upper_confidence_bound",
            "ucb",
        ):
            self.assertIn(expected, names)
        self.assertIs(_get_acquisition("ei"), expected_improvement)
        self.assertIs(_get_acquisition("UCB"), upper_confidence_bound)  # case-insensitive

    def test_callable_passes_through(self):
        self.assertIs(_get_acquisition(expected_improvement), expected_improvement)

    def test_unknown_acquisition_lists_registered(self):
        with self.assertRaises(ValueError) as ctx:
            _get_acquisition("banana")
        self.assertIn("banana", str(ctx.exception))
        self.assertIn("ei", str(ctx.exception))

    def test_non_callable_rejected(self):
        with self.assertRaises(TypeError):
            register_acquisition("bad", object())

    def test_custom_acquisition_is_registered(self):
        def const_acq(mean, std, best, *, maximize=False, **_):
            return np.zeros_like(np.asarray(std, dtype=float))

        register_acquisition("const-test-acq", const_acq, aliases=("cta",))
        try:
            self.assertIs(_get_acquisition("CTA"), const_acq)
        finally:
            from mixle.doe.bayesopt import _ACQUISITIONS

            _ACQUISITIONS.pop("const-test-acq", None)
            _ACQUISITIONS.pop("cta", None)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class BayesOptLoopTest(unittest.TestCase):
    def test_propose_next_in_bounds(self):
        bounds = [(-2.0, 2.0), (0.0, 5.0)]
        rng = np.random.RandomState(0)
        x = rng.uniform([-2.0, 0.0], [2.0, 5.0], size=(6, 2))
        y = np.sum((x - np.array([0.5, 2.0])) ** 2, axis=1)
        nxt = np.asarray(propose_next(x, y, bounds, n_candidates=128, seed=1, fit_kwargs={"max_its": 60}))
        self.assertEqual(nxt.shape, (2,))
        self.assertTrue(np.all(nxt >= [-2.0, 0.0]) and np.all(nxt <= [2.0, 5.0]))

    def test_minimize_finds_near_optimum_of_a_bowl(self):
        target = np.array([0.5, -1.0])
        bounds = [(-3.0, 3.0), (-3.0, 3.0)]

        def objective(p):
            return float(np.sum((p - target) ** 2))

        result = minimize(objective, bounds, n_init=6, n_iter=20, seed=0, n_candidates=256, fit_kwargs={"max_its": 60})
        self.assertEqual(result.x.shape[0], result.y.shape[0])
        self.assertEqual(result.x.shape[0], 26)
        # BO should beat the best of the initial random design and land near the optimum.
        best_init = float(np.min(result.y[:6]))
        self.assertLessEqual(result.best_y, best_init)
        self.assertLess(result.best_y, 0.5)

    def test_propose_next_honors_acq_choice(self):
        bounds = [(-2.0, 2.0), (0.0, 5.0)]
        rng = np.random.RandomState(0)
        x = rng.uniform([-2.0, 0.0], [2.0, 5.0], size=(6, 2))
        y = np.sum((x - np.array([0.5, 2.0])) ** 2, axis=1)
        for acq, kw in (("ei", None), ("pi", None), ("ucb", {"kappa": 2.0})):
            nxt = np.asarray(
                propose_next(x, y, bounds, n_candidates=128, seed=1, acq=acq, acq_kwargs=kw, fit_kwargs={"max_its": 60})
            )
            self.assertEqual(nxt.shape, (2,))
            self.assertTrue(np.all(nxt >= [-2.0, 0.0]) and np.all(nxt <= [2.0, 5.0]))

    def test_minimize_with_ucb_finds_near_optimum(self):
        target = np.array([0.5, -1.0])
        bounds = [(-3.0, 3.0), (-3.0, 3.0)]

        def objective(p):
            return float(np.sum((p - target) ** 2))

        result = minimize(
            objective,
            bounds,
            n_init=6,
            n_iter=20,
            seed=0,
            acq="ucb",
            acq_kwargs={"kappa": 2.0},
            n_candidates=256,
            fit_kwargs={"max_its": 60},
        )
        self.assertLess(result.best_y, 0.5)

    def test_propose_batch_returns_distinct_in_bounds_points(self):
        bounds = [(-2.0, 2.0), (0.0, 5.0)]
        rng = np.random.RandomState(0)
        x = rng.uniform([-2.0, 0.0], [2.0, 5.0], size=(6, 2))
        y = np.sum((x - np.array([0.5, 2.0])) ** 2, axis=1)
        batch = propose_batch(x, y, bounds, q=3, n_candidates=128, seed=1, fit_kwargs={"max_its": 60})
        self.assertEqual(batch.shape, (3, 2))
        self.assertTrue(np.all(batch >= [-2.0, 0.0]) and np.all(batch <= [2.0, 5.0]))
        # kriging-believer steers picks apart: the batch should not collapse to one repeated point.
        self.assertGreater(len(np.unique(batch, axis=0)), 1)

    def test_propose_batch_rejects_nonpositive_q(self):
        with self.assertRaises(ValueError):
            propose_batch(np.zeros((3, 2)), np.zeros(3), [(0.0, 1.0), (0.0, 1.0)], q=0)


if __name__ == "__main__":
    unittest.main()
