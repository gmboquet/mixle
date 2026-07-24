"""Tests for the DoE Bayesian-optimization loop (WS-E).

``expected_improvement`` is torch-free and tested directly; the GP-surrogate ``propose_next`` /
``minimize`` paths require torch and are skipped when it is unavailable.
"""

import importlib.util
import unittest

import numpy as np
from scipy.stats import norm

from mixle.doe import (
    available_acquisitions,
    expected_improvement,
    log_expected_improvement,
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
    def test_zero_std_gives_deterministic_limit(self):
        # A point-mass posterior (std=0) has no uncertainty to average over: EI is the GUARANTEED
        # improvement max(best - mean, 0), not 0 (MXR-080-0168). mean=0 improves on best=0.5 by 0.5;
        # mean=1.0 does not improve at all, so it stays at the 0 floor.
        ei = expected_improvement(mean=np.array([0.0, 1.0]), std=np.array([0.0, 0.0]), best=0.5)
        np.testing.assert_array_equal(ei, np.array([0.5, 0.0]))

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


class LogExpectedImprovementTest(unittest.TestCase):
    def test_zero_std_gives_deterministic_limit(self):
        # log(EI) at the deterministic limit is log(max(best - mean, 0)): finite (log 0.5) where the
        # point-mass outcome improves, -inf where it does not (MXR-080-0168; mirrors EI's fix).
        log_ei = log_expected_improvement(mean=np.array([0.0, 1.0]), std=np.array([0.0, 0.0]), best=0.5)
        np.testing.assert_allclose(log_ei, np.array([np.log(0.5), -np.inf]))

    def test_minimize_rewards_lower_mean(self):
        log_ei = log_expected_improvement(mean=np.array([-1.0, 0.0, 1.0]), std=np.array([1.0, 1.0, 1.0]), best=0.0)
        self.assertGreater(log_ei[0], log_ei[1])
        self.assertGreater(log_ei[1], log_ei[2])

    def test_maximize_rewards_higher_mean(self):
        log_ei = log_expected_improvement(
            mean=np.array([-1.0, 0.0, 1.0]), std=np.array([1.0, 1.0, 1.0]), best=0.0, maximize=True
        )
        self.assertGreater(log_ei[2], log_ei[1])
        self.assertGreater(log_ei[1], log_ei[0])

    def test_matches_log_of_ei_away_from_the_underflow_tail(self):
        # Where EI itself is far from underflowing, log(EI) and log_expected_improvement must agree.
        mean, std, best = np.array([-1.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0]), 0.0
        ei = expected_improvement(mean=mean, std=std, best=best)
        log_ei = log_expected_improvement(mean=mean, std=std, best=best)
        np.testing.assert_allclose(log_ei, np.log(ei), atol=1e-10)


class DeterministicCandidateEiTest(unittest.TestCase):
    """MXR-080-0168: EI/log-EI at (near-)zero predictive variance must equal the deterministic limit
    max(improve, 0), continuously as std -> 0, without disturbing the ordinary nonzero-variance formula.
    """

    def test_audit_case_deterministic_candidate_has_guaranteed_improvement(self):
        # The audit's exact repro: a deterministic candidate (std=0) predicted to land exactly at 0,
        # against a minimizing incumbent of 2 -- a GUARANTEED improvement of 2, not 0.
        ei = expected_improvement(mean=np.array([0.0]), std=np.array([0.0]), best=2.0)
        self.assertEqual(ei[0], 2.0)
        log_ei = log_expected_improvement(mean=np.array([0.0]), std=np.array([0.0]), best=2.0)
        self.assertAlmostEqual(log_ei[0], np.log(2.0), places=12)

    def test_continuity_as_variance_shrinks_to_zero(self):
        # No discontinuous jump: EI/log-EI must smoothly approach the deterministic limit as std -> 0,
        # not just be correct in isolation at the exact std=0 point.
        stds = [1.0e-3, 1.0e-6, 1.0e-9, 0.0]
        ei_values = [expected_improvement(mean=np.array([0.0]), std=np.array([s]), best=2.0)[0] for s in stds]
        for value in ei_values:
            self.assertAlmostEqual(value, 2.0, places=8)

        log_ei_values = [log_expected_improvement(mean=np.array([0.0]), std=np.array([s]), best=2.0)[0] for s in stds]
        for value in log_ei_values:
            self.assertAlmostEqual(value, np.log(2.0), places=8)

    def test_deterministic_candidate_with_no_improvement_stays_at_the_floor(self):
        # The other side of the limit: a deterministic candidate that does NOT improve on the incumbent
        # gets EI=0 / log-EI=-inf, exactly as before -- the fix only changes the improving case.
        worse = expected_improvement(mean=np.array([3.0]), std=np.array([0.0]), best=2.0)
        tied = expected_improvement(mean=np.array([2.0]), std=np.array([0.0]), best=2.0)
        np.testing.assert_array_equal(worse, np.array([0.0]))
        np.testing.assert_array_equal(tied, np.array([0.0]))
        log_worse = log_expected_improvement(mean=np.array([3.0]), std=np.array([0.0]), best=2.0)
        log_tied = log_expected_improvement(mean=np.array([2.0]), std=np.array([0.0]), best=2.0)
        np.testing.assert_array_equal(log_worse, np.array([-np.inf]))
        np.testing.assert_array_equal(log_tied, np.array([-np.inf]))

    def test_maximize_convention_deterministic_limit(self):
        # maximize=True: improve = mean - best - xi; a deterministic mean=5 against incumbent best=2
        # guarantees an improvement of 3.
        ei = expected_improvement(mean=np.array([5.0]), std=np.array([0.0]), best=2.0, maximize=True)
        self.assertEqual(ei[0], 3.0)

    def test_negative_control_nonzero_variance_matches_standard_closed_form(self):
        # The fix must not perturb the ordinary (nonzero-std) formula: compare against an independent
        # scipy.stats.norm implementation of the textbook closed form.
        mean_, std_, best_ = 0.3, 1.4, 1.0
        z = (best_ - mean_) / std_
        expected_ei = std_ * (z * norm.cdf(z) + norm.pdf(z))
        got_ei = expected_improvement(mean=np.array([mean_]), std=np.array([std_]), best=best_)[0]
        self.assertAlmostEqual(got_ei, expected_ei, places=12)
        got_log_ei = log_expected_improvement(mean=np.array([mean_]), std=np.array([std_]), best=best_)[0]
        self.assertAlmostEqual(got_log_ei, np.log(expected_ei), places=10)


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
