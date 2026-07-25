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
    propose_knowledge_gradient,
    propose_next,
    register_acquisition,
    thompson_sampling,
    upper_confidence_bound,
)
from mixle.doe.bayesopt import _get_acquisition

HAS_TORCH = importlib.util.find_spec("torch") is not None


class _MockSurrogate:
    """A minimal duck-typed :class:`~mixle.doe._contracts.Surrogate` returning caller-controlled,
    possibly-broken predictions -- lets the MXR-080-0170 surrogate-boundary validation be tested
    without depending on torch or the real GP.
    """

    def __init__(self, mean, cov=None):
        self._mean = mean
        self._cov = cov

    def fit(self, x, y, **kwargs):
        return self

    def predict(self, x_train, y_train, x_new, return_cov=False):
        return (self._mean, self._cov) if return_cov else self._mean


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


class AcquisitionInputValidationTest(unittest.TestCase):
    """MXR-080-0169: every built-in acquisition validates the common posterior-moment / parameter
    contract up front, instead of accepting or silently broadcasting a malformed input.
    """

    _ACQUISITIONS = (
        ("ei", expected_improvement, {}),
        ("log_ei", log_expected_improvement, {}),
        ("pi", probability_of_improvement, {}),
        ("ucb", upper_confidence_bound, {}),
        ("thompson", thompson_sampling, {}),
    )

    def test_probability_of_improvement_rejects_negative_std_as_invalid_not_deterministic(self):
        # The specific historical bug: PI tested `std <= threshold`, so a NEGATIVE std (an upstream
        # error, never a legitimate input) fell into the same branch as the deterministic std==0 case
        # and was silently scored as if certain. It must now be rejected outright.
        with self.assertRaises(ValueError):
            probability_of_improvement(mean=np.array([0.0]), std=np.array([-5.0]), best=2.0)

    def test_negative_std_rejected_by_every_builtin_acquisition(self):
        for name, fn, kw in self._ACQUISITIONS:
            with self.subTest(acquisition=name), self.assertRaises(ValueError):
                fn(mean=np.array([0.0]), std=np.array([-1.0]), best=0.0, **kw)

    def test_nonfinite_std_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(std=value), self.assertRaises(ValueError):
                expected_improvement(mean=np.array([0.0]), std=np.array([value]), best=0.0)

    def test_nonfinite_mean_rejected(self):
        with self.assertRaises(ValueError):
            expected_improvement(mean=np.array([np.nan]), std=np.array([1.0]), best=0.0)

    def test_nonfinite_best_rejected(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(best=value), self.assertRaises(ValueError):
                expected_improvement(mean=np.array([0.0]), std=np.array([1.0]), best=value)

    def test_mismatched_mean_std_shapes_rejected(self):
        # mean/std are per-candidate arrays that must line up one-to-one; this is not an intentionally
        # broadcastable pair (unlike best/xi/kappa, which are genuine scalars broadcast against them).
        with self.assertRaises(ValueError):
            expected_improvement(mean=np.array([0.0, 1.0]), std=np.array([1.0]), best=0.0)

    def test_negative_xi_rejected(self):
        for fn in (expected_improvement, log_expected_improvement, probability_of_improvement):
            with self.subTest(fn=fn.__name__), self.assertRaises(ValueError):
                fn(mean=np.array([0.0]), std=np.array([1.0]), best=0.0, xi=-0.1)

    def test_nonfinite_xi_rejected(self):
        with self.assertRaises(ValueError):
            expected_improvement(mean=np.array([0.0]), std=np.array([1.0]), best=0.0, xi=float("nan"))

    def test_negative_kappa_rejected(self):
        with self.assertRaises(ValueError):
            upper_confidence_bound(mean=np.array([0.0]), std=np.array([1.0]), best=0.0, kappa=-0.1)

    def test_invalid_rng_rejected(self):
        for bad_rng in ("banana", 42, object()):
            with self.subTest(rng=bad_rng), self.assertRaises(ValueError):
                thompson_sampling(mean=np.array([0.0]), std=np.array([1.0]), best=0.0, rng=bad_rng)

    def test_valid_rng_types_accepted(self):
        # Negative control: both the legacy RandomState and the modern Generator are genuine RNGs.
        for rng in (np.random.RandomState(0), np.random.default_rng(0)):
            with self.subTest(rng=type(rng).__name__):
                out = thompson_sampling(mean=np.array([0.0]), std=np.array([1.0]), best=0.0, rng=rng)
                self.assertEqual(out.shape, (1,))
                self.assertTrue(np.all(np.isfinite(out)))

    def test_negative_control_well_formed_calls_are_unaffected(self):
        # Ordinary, valid calls to every built-in acquisition keep computing exactly as before.
        mean, std, best = np.array([-1.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0]), 0.0
        for name, fn, kw in self._ACQUISITIONS:
            with self.subTest(acquisition=name):
                out = fn(mean=mean, std=std, best=best, **kw)
                self.assertEqual(out.shape, (3,))
                self.assertTrue(np.all(np.isfinite(out)))
        # PI's own legitimate deterministic case (std == 0, not negative) is untouched by the fix.
        pi = probability_of_improvement(mean=np.array([-1.0, 1.0]), std=np.array([0.0, 0.0]), best=0.0)
        np.testing.assert_array_equal(pi, np.array([1.0, 0.0]))


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

    def test_propose_batch_all_nonfinite_merit_raises_clearly(self):
        # MXR-080-0170, the batch path: propose_batch always fits its own (real) surrogate -- no gp=
        # override -- so this exercises the shared _propose_one merit-selection boundary through the
        # real GP rather than a mock, via a pathological acquisition that returns an all-NaN merit.
        bounds = [(-2.0, 2.0), (0.0, 5.0)]
        rng = np.random.RandomState(0)
        x = rng.uniform([-2.0, 0.0], [2.0, 5.0], size=(6, 2))
        y = np.sum((x - np.array([0.5, 2.0])) ** 2, axis=1)

        def nan_acquisition(mean, std, best, *, maximize=False, **_):
            return np.full(mean.shape, np.nan)

        with self.assertRaises(ValueError):
            propose_batch(x, y, bounds, q=2, n_candidates=16, seed=1, acq=nan_acquisition, fit_kwargs={"max_its": 20})


class SurrogateBoundaryTest(unittest.TestCase):
    """MXR-080-0170: the proposal machinery validates the surrogate's predicted mean/covariance and the
    acquisition merit derived from them, instead of letting a broken ``gp=`` (wrong shape, non-finite,
    asymmetric covariance) or an all-non-finite merit reach ``argmax``/``argmin`` unchecked. Uses
    ``_MockSurrogate`` so these run without torch, covering the sequential (``propose_next``) and
    knowledge-gradient (``propose_knowledge_gradient``) proposal paths -- both accept a ``gp=``
    override. ``propose_batch`` shares the exact same validated boundary (via ``_propose_one``, the
    same helper ``propose_next`` uses) but always fits its own surrogate internally with no ``gp=``
    override, so its coverage of this same boundary lives with the other torch-dependent tests in
    :class:`BayesOptLoopTest` below.
    """

    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.x = self.rng.uniform(-2.0, 2.0, size=(4, 2))
        self.y = np.sum(self.x**2, axis=1)
        self.bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        self.n_candidates = 5

    def test_propose_next_rejects_wrong_length_mean(self):
        gp = _MockSurrogate(mean=np.zeros(3), cov=np.eye(3))  # 3 != n_candidates (5)
        with self.assertRaises(ValueError):
            propose_next(self.x, self.y, self.bounds, n_candidates=self.n_candidates, seed=1, gp=gp)

    def test_propose_next_rejects_nonfinite_mean(self):
        gp = _MockSurrogate(mean=np.full(self.n_candidates, np.nan), cov=np.eye(self.n_candidates))
        with self.assertRaises(ValueError):
            propose_next(self.x, self.y, self.bounds, n_candidates=self.n_candidates, seed=1, gp=gp)

    def test_propose_next_rejects_wrong_shape_covariance(self):
        gp = _MockSurrogate(mean=np.zeros(self.n_candidates), cov=np.eye(self.n_candidates + 1))
        with self.assertRaises(ValueError):
            propose_next(self.x, self.y, self.bounds, n_candidates=self.n_candidates, seed=1, gp=gp)

    def test_propose_next_rejects_asymmetric_covariance(self):
        cov = np.eye(self.n_candidates)
        cov[0, 1] = 5.0  # break symmetry
        gp = _MockSurrogate(mean=np.zeros(self.n_candidates), cov=cov)
        with self.assertRaises(ValueError):
            propose_next(self.x, self.y, self.bounds, n_candidates=self.n_candidates, seed=1, gp=gp)

    def test_propose_next_rejects_nonfinite_observations_before_fitting(self):
        # Non-finite y never reaches the (mock) surrogate at all -- _fit_surrogate's own boundary
        # check fires first, regardless of what the surrogate would have done with it.
        bad_y = np.array([1.0, np.nan, 2.0, 3.0])
        gp = _MockSurrogate(mean=np.zeros(self.n_candidates), cov=np.eye(self.n_candidates))
        with self.assertRaises(ValueError):
            propose_next(self.x, bad_y, self.bounds, n_candidates=self.n_candidates, seed=1, gp=gp)

    def test_propose_next_all_nonfinite_merit_raises_clearly(self):
        # A pathological/custom acquisition that returns an all-NaN merit must not let np.argmax
        # silently pick an arbitrary (NaN) index.
        def nan_acquisition(mean, std, best, *, maximize=False, **_):
            return np.full(mean.shape, np.nan)

        gp = _MockSurrogate(mean=np.zeros(self.n_candidates), cov=np.eye(self.n_candidates) * 0.01)
        with self.assertRaises(ValueError):
            propose_next(
                self.x, self.y, self.bounds, n_candidates=self.n_candidates, seed=1, gp=gp, acq=nan_acquisition
            )

    def test_propose_next_negative_control_selects_the_obviously_best_candidate(self):
        # One candidate's predicted mean is far better (lower) than the rest with tight uncertainty;
        # a well-posed surrogate must still let propose_next select a sensible, high-merit candidate.
        mean = np.array([0.0, 0.0, -100.0, 0.0, 0.0])
        cov = np.eye(self.n_candidates) * 0.01
        gp = _MockSurrogate(mean=mean, cov=cov)
        _, merit = propose_next(
            self.x, self.y, self.bounds, n_candidates=self.n_candidates, seed=1, gp=gp, return_acquisition=True
        )
        # EI at the -100 candidate is roughly (best - (-100)); every other candidate's EI is tiny by
        # comparison (best is a small value from x**2 over [-2, 2]^2).
        self.assertGreater(merit, 50.0)

    def test_propose_knowledge_gradient_rejects_wrong_length_mean(self):
        gp = _MockSurrogate(mean=np.zeros(3), cov=np.eye(3))
        with self.assertRaises(ValueError):
            propose_knowledge_gradient(self.x, self.y, self.bounds, n_candidates=self.n_candidates, gp=gp)

    def test_propose_knowledge_gradient_rejects_asymmetric_covariance(self):
        cov = np.eye(self.n_candidates)
        cov[0, 1] = 5.0
        gp = _MockSurrogate(mean=np.zeros(self.n_candidates), cov=cov)
        with self.assertRaises(ValueError):
            propose_knowledge_gradient(self.x, self.y, self.bounds, n_candidates=self.n_candidates, gp=gp)

    def test_propose_knowledge_gradient_negative_control_selects_a_candidate_in_bounds(self):
        mean = np.array([0.0, 0.0, 5.0, 0.0, 0.0])
        cov = np.eye(self.n_candidates) * 0.01
        gp = _MockSurrogate(mean=mean, cov=cov)
        point = propose_knowledge_gradient(self.x, self.y, self.bounds, n_candidates=self.n_candidates, gp=gp)
        self.assertEqual(point.shape, (2,))
        self.assertTrue(np.all(point >= [-2.0, -2.0]) and np.all(point <= [2.0, 2.0]))


class ThompsonSamplingRngThreadingTest(unittest.TestCase):
    """``thompson_sampling`` used to silently fall back to a fresh, unseeded ``np.random.RandomState()``
    whenever ``propose_next``/``propose_batch``'s own (correctly seeded) ``rng`` was not manually
    re-threaded into ``acq_kwargs`` -- so ``BayesianOptimizer(acq="thompson", seed=...)`` reproduced its
    Latin-hypercube init design bit-for-bit (that path already used the seeded rng) but drew a
    DIFFERENT GP-guided proposal on every run, because the acquisition's own randomness came from OS
    entropy instead of the caller's seed. ``_propose_one`` now threads its own ``rng`` into every
    acquisition call (an explicit ``acq_kwargs["rng"]``, if given, still wins), so ``acq="thompson"`` is
    reproducible under a fixed seed with no extra caller wiring, exactly like every other acquisition.
    Uses ``_MockSurrogate`` so this runs without torch -- the bug and the fix are entirely in the
    proposal loop's own kwarg wiring, not in the GP surrogate. See also
    ``doe_optimizer_test.ThompsonSamplingBayesianOptimizerReproducibilityTest`` for the same fix proven
    end to end against the real ``BayesianOptimizer``/GP.
    """

    def setUp(self):
        self.x = np.array([[0.1, 0.1], [0.4, -0.3], [-0.9, 0.7], [1.2, -1.5]])
        self.y = np.sum(self.x**2, axis=1)
        self.bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        self.n_candidates = 5

    def _mock_gp(self, std: float = 0.5) -> _MockSurrogate:
        # Nonzero std so Thompson's draw is genuinely stochastic, not a degenerate std=0 point mass.
        return _MockSurrogate(mean=np.zeros(self.n_candidates), cov=np.eye(self.n_candidates) * std * std)

    def _propose_thompson(self, seed, acq_kwargs=None):
        return propose_next(
            self.x,
            self.y,
            self.bounds,
            n_candidates=self.n_candidates,
            seed=seed,
            acq="thompson",
            acq_kwargs=acq_kwargs,
            gp=self._mock_gp(),
            return_acquisition=True,
        )

    def test_repeated_calls_with_the_same_seed_are_bit_identical(self):
        # The direct reproduction of the reported bug: same seed, same everything else, twice.
        point_a, merit_a = self._propose_thompson(seed=2)
        point_b, merit_b = self._propose_thompson(seed=2)
        np.testing.assert_array_equal(point_a, point_b)
        self.assertEqual(merit_a, merit_b)

    def test_different_seeds_give_different_draws(self):
        # Sanity check that the mock's nonzero std makes the draw genuinely stochastic -- confirms the
        # bit-identity above is a real reproducibility guarantee, not a coincidence of a degenerate
        # (std=0 or seed-insensitive) setup.
        point_a, merit_a = self._propose_thompson(seed=2)
        point_b, merit_b = self._propose_thompson(seed=3)
        self.assertFalse(np.array_equal(point_a, point_b) and merit_a == merit_b)

    def test_explicit_acq_kwargs_rng_still_overrides_the_ambient_seed(self):
        # A caller that manually threads its own rng via acq_kwargs (the pre-fix workaround, still a
        # documented, supported override -- see thompson_acquisition_test.py) must still win over the
        # ambient seed='s rng for the ACQUISITION's own randomness, not be silently shadowed by it.
        # seed= is held FIXED across every call below: it also controls the Latin-hypercube candidate
        # SET itself (independent of acq_kwargs), so varying it here would confound "which candidates
        # exist" with "which one gets picked" -- holding it fixed isolates the acquisition's own draw
        # as the only thing that can differ. Compares the raw MERIT (a continuous score, returned via
        # return_acquisition=True inside _propose_thompson), not just the final selected point: with
        # only n_candidates=5 options, two genuinely-different random draws can coincidentally agree
        # on the argmax roughly 1-in-5 of the time, which the underlying merit score will not.
        _, ambient_merit = self._propose_thompson(seed=2)  # ambient seed's rng feeds the Thompson draw
        _, overridden_merit = self._propose_thompson(seed=2, acq_kwargs={"rng": np.random.RandomState(99)})
        self.assertNotEqual(ambient_merit, overridden_merit)  # the override actually took effect

        _, overridden_merit_again = self._propose_thompson(seed=2, acq_kwargs={"rng": np.random.RandomState(99)})
        self.assertEqual(overridden_merit, overridden_merit_again)  # ...and is itself reproducible

    def test_rng_kwarg_is_a_no_op_for_non_thompson_acquisitions(self):
        # _propose_one now unconditionally passes rng= to every acquisition (so thompson_sampling gets
        # threaded); every other builtin must ignore it via its own **_ catch-all with zero effect --
        # proven directly at the function contract, not just "two runs happened to match".
        mean = np.array([-1.0, 0.0, 1.0, 2.0, -2.0])
        std = np.array([1.0, 0.5, 0.2, 0.1, 0.3])
        for fn in (expected_improvement, log_expected_improvement, probability_of_improvement, upper_confidence_bound):
            with self.subTest(fn=fn.__name__):
                without_rng = fn(mean=mean, std=std, best=0.0)
                with_rng = fn(mean=mean, std=std, best=0.0, rng=np.random.RandomState(0))
                np.testing.assert_array_equal(without_rng, with_rng)

    def test_default_acq_ei_path_produces_identical_proposals_regardless_of_the_threading_fix(self):
        # BayesianOptimizer/propose_next default to acq="ei", which takes no rng at all -- the fix must
        # be a complete no-op for the default path end to end (not just at the bare acquisition
        # function above), across two independent mock surrogates standing in for two separate fits.
        mean = np.array([0.0, 0.0, -50.0, 0.0, 0.0])
        cov = np.eye(self.n_candidates) * 0.01
        point_a, merit_a = propose_next(
            self.x,
            self.y,
            self.bounds,
            n_candidates=self.n_candidates,
            seed=2,
            return_acquisition=True,
            gp=_MockSurrogate(mean=mean, cov=cov),
        )
        point_b, merit_b = propose_next(
            self.x,
            self.y,
            self.bounds,
            n_candidates=self.n_candidates,
            seed=2,
            return_acquisition=True,
            gp=_MockSurrogate(mean=mean, cov=cov),
        )
        np.testing.assert_array_equal(point_a, point_b)
        self.assertEqual(merit_a, merit_b)

    def test_thompson_acquisition_advances_the_shared_rng_state_beyond_candidate_generation(self):
        # RNG-state-hashing mechanism proof: starting from identical RandomState(seed) instances, ei
        # (which draws no randomness of its own) advances the rng only through Latin-hypercube
        # candidate generation, while thompson (which draws one extra standard_normal per candidate
        # from the SAME stream) must leave the rng at a strictly later position in its underlying
        # random stream -- direct evidence the acquisition's randomness is genuinely pulled from the
        # threaded rng, not merely coincidentally reproducible.
        mean = np.zeros(self.n_candidates)
        cov = np.eye(self.n_candidates) * 0.25

        rng_ei = np.random.RandomState(2)
        propose_next(
            self.x, self.y, self.bounds, n_candidates=self.n_candidates, seed=rng_ei, acq="ei", gp=self._mock_gp()
        )
        pos_after_ei = rng_ei.get_state()[2]

        rng_thompson = np.random.RandomState(2)
        propose_next(
            self.x,
            self.y,
            self.bounds,
            n_candidates=self.n_candidates,
            seed=rng_thompson,
            acq="thompson",
            gp=self._mock_gp(),
        )
        pos_after_thompson = rng_thompson.get_state()[2]

        self.assertGreater(
            pos_after_thompson,
            pos_after_ei,
            "thompson consumed no more of the shared rng's random stream than ei did -- its own draw "
            "is not actually being pulled from the threaded rng",
        )


if __name__ == "__main__":
    unittest.main()
