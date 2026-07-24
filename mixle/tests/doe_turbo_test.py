"""Trust-region Bayesian optimization, TuRBO (mixle.doe.trust_region)."""

import importlib.util
import unittest
import warnings
from unittest import mock

import numpy as np

from mixle.doe import trust_region as tr_module
from mixle.doe.trust_region import TrustRegion, turbo_minimize

HAS_TORCH = importlib.util.find_spec("torch") is not None


class _FakeGP:
    """Stub surrogate: a valid ``predict()`` (mean 0, small-variance diagonal covariance) that lets
    ``turbo_minimize``'s loop run end-to-end without torch. Used by the MXR-080-0196/0197 tests below
    that exercise the loop's *control flow* (validation, restarts, exception handling, diagnostics) --
    not the actual GP math, which the (torch-gated) ``TurboOptimizeTest`` covers separately.
    """

    def predict(self, xn, yn, cand, return_cov=True):
        n = cand.shape[0]
        return np.zeros(n), np.eye(n) * 0.01


def _fake_fit_surrogate_ok(x, y, gp, fit_kwargs):
    return _FakeGP()


class TrustRegionStateTest(unittest.TestCase):
    def test_expands_on_success_shrinks_on_failure(self):
        tr = TrustRegion(dim=4)
        start = tr.length
        for _ in range(tr.success_tol):
            tr.update(True)
        self.assertGreater(tr.length, start)  # doubled after consecutive successes

        tr = TrustRegion(dim=4)
        start = tr.length
        for _ in range(tr.failure_tol):
            tr.update(False)
        self.assertLess(tr.length, start)  # halved after consecutive failures

    def test_collapses(self):
        tr = TrustRegion(dim=2)
        for _ in range(200):
            tr.update(False)
        self.assertTrue(tr.collapsed)


# --------------------------------------------------------------------------- MXR-080-0196
# TrustRegion.__post_init__ and turbo_minimize's own parameter handling accepted impossible geometry
# and budgets outright: a negative batch_size silently produced zero picks (range(-1) iterates zero
# times), a batch_size that could never fit inside n_candidates repeatedly raised *inside* the search
# loop where it was indistinguishable from a genuine model failure, and invalid trust lengths/tolerances
# (inverted min/max, non-finite length, non-positive tolerances) were never checked at all. Everything
# below is now validated up front with a clear, specific error instead.
class TrustRegionValidationTest(unittest.TestCase):
    def test_rejects_nonpositive_or_fractional_dim(self):
        for bad in (0, -3, 2.5):
            with self.assertRaises((ValueError, TypeError)):
                TrustRegion(dim=bad)

    def test_rejects_nonpositive_length_min(self):
        for bad in (0.0, -0.1):
            with self.assertRaises(ValueError):
                TrustRegion(dim=3, length_min=bad)

    def test_rejects_length_max_below_length_min(self):
        # min(length * 2, length_max) could otherwise shrink the region on a "successful" expansion.
        with self.assertRaises(ValueError):
            TrustRegion(dim=3, length_min=0.5, length_max=0.1)

    def test_rejects_length_outside_min_max_bounds(self):
        with self.assertRaises(ValueError):
            TrustRegion(dim=3, length=2.0, length_min=0.1, length_max=1.6)  # length > length_max
        with self.assertRaises(ValueError):
            TrustRegion(dim=3, length=0.01, length_min=0.1, length_max=1.6)  # length < length_min

    def test_rejects_non_finite_length(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                TrustRegion(dim=3, length=bad)

    def test_rejects_nonpositive_success_tol(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                TrustRegion(dim=3, success_tol=bad)

    def test_rejects_negative_failure_tol(self):
        # 0 is the documented "default to dim" sentinel and must stay legal.
        with self.assertRaises(ValueError):
            TrustRegion(dim=3, failure_tol=-1)
        TrustRegion(dim=3, failure_tol=0)  # still fine

    def test_a_valid_custom_configuration_still_constructs_and_cycles(self):
        # Negative control: validation does not reject legitimate, non-default geometry, and the
        # expand/shrink cycle (already covered end to end by TrustRegionStateTest for the defaults)
        # works the same way for a custom-but-valid region. Independent instances per direction, same
        # as TrustRegionStateTest, since an expand-then-shrink round trip can land back on the start
        # value rather than below it.
        kwargs = dict(dim=5, length=0.5, length_min=0.05, length_max=1.0, success_tol=2, failure_tol=3)
        tr = TrustRegion(**kwargs)
        start = tr.length
        for _ in range(tr.success_tol):
            tr.update(True)
        self.assertGreater(tr.length, start)

        tr = TrustRegion(**kwargs)
        start = tr.length
        for _ in range(tr.failure_tol):
            tr.update(False)
        self.assertLess(tr.length, start)


class TurboMinimizeValidationTest(unittest.TestCase):
    """turbo_minimize's own count/budget validation -- torch-free: every case here is rejected before
    the loop ever reaches a GP fit, so none of it needs the real surrogate.
    """

    def _obj(self, x):
        return float(np.sum(x**2))

    def test_rejects_negative_batch_size_instead_of_silently_producing_no_picks(self):
        # Previously: range(-1) iterates zero times, so _thompson_batch silently returned an empty
        # batch and the run quietly degraded to restarts-only with no error at all.
        with self.assertRaises(ValueError):
            turbo_minimize(self._obj, [(-1.0, 1.0)] * 2, n_init=4, max_evals=12, batch_size=-1, seed=0)

    def test_rejects_zero_batch_size(self):
        with self.assertRaises(ValueError):
            turbo_minimize(self._obj, [(-1.0, 1.0)] * 2, n_init=4, max_evals=12, batch_size=0, seed=0)

    def test_rejects_batch_size_greater_than_n_candidates_up_front(self):
        # Previously: unsatisfiable no matter how many times the loop retries (n_candidates is fixed
        # for the whole run), so it just kept raising ValueError from inside _thompson_batch, getting
        # caught by turbo_minimize's except clause and misattributed as "the model failed". Now
        # rejected once, immediately, with a message naming the actual constraint -- not a model-failure
        # diagnostic. Regression-test this end to end (not just via _thompson_batch in isolation): the
        # GP is stubbed (never actually invoked, since validation must happen before the loop starts)
        # so this runs without torch.
        with mock.patch.object(tr_module, "_fit_surrogate", side_effect=AssertionError("must not be reached")):
            with self.assertRaisesRegex(ValueError, "batch_size <= n_candidates"):
                turbo_minimize(
                    self._obj, [(-1.0, 1.0)] * 2, n_init=4, max_evals=12, batch_size=10, n_candidates=3, seed=0
                )

    def test_enforces_the_1_le_batch_size_le_n_candidates_invariant(self):
        # batch_size == n_candidates is the boundary and must be accepted (validation only, GP stubbed).
        with mock.patch.object(tr_module, "_fit_surrogate", side_effect=_fake_fit_surrogate_ok):
            res = turbo_minimize(
                self._obj, [(-1.0, 1.0)] * 2, n_init=4, max_evals=8, batch_size=3, n_candidates=3, seed=0
            )
        self.assertLessEqual(res["Y"].shape[0], 8)

    def test_rejects_fractional_or_negative_max_evals_and_n_init(self):
        with self.assertRaises((ValueError, TypeError)):
            turbo_minimize(self._obj, [(-1.0, 1.0)] * 2, n_init=4, max_evals=10.5, seed=0)
        with self.assertRaises((ValueError, TypeError)):
            turbo_minimize(self._obj, [(-1.0, 1.0)] * 2, n_init=3.5, max_evals=12, seed=0)
        with self.assertRaises((ValueError, TypeError)):
            turbo_minimize(self._obj, [(-1.0, 1.0)] * 2, n_init=4, max_evals=-5, seed=0)

    def test_rejects_fractional_n_candidates(self):
        with self.assertRaises((ValueError, TypeError)):
            turbo_minimize(self._obj, [(-1.0, 1.0)] * 2, n_init=4, max_evals=12, n_candidates=5.5, seed=0)

    def test_a_normal_well_configured_search_still_runs_through_shrink_expand_restart(self):
        # Negative control for the whole validation pass: a sensible configuration is untouched by any
        # of the new checks and still drives the loop through real shrink/expand/restart cycles (GP
        # stubbed so this runs without torch; TurboOptimizeTest below covers the real GP end to end).
        with mock.patch.object(tr_module, "_fit_surrogate", side_effect=_fake_fit_surrogate_ok):
            res = turbo_minimize(
                self._obj, [(-1.0, 1.0)] * 2, n_init=4, max_evals=40, batch_size=2, n_candidates=50, seed=0
            )
        self.assertEqual(res["Y"].shape[0], 40)
        self.assertGreaterEqual(res["n_restarts"], 0)


@unittest.skipUnless(HAS_TORCH, "GP surrogate requires torch")
class TurboOptimizeTest(unittest.TestCase):
    def test_finds_quadratic_optimum(self):
        from mixle.doe import turbo_minimize

        opt = np.array([0.3, -0.7, 1.2, -1.5, 0.0, 0.9])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # The GP-hyperparameter refit (Adam, default max_its=500) dominates runtime: profiling
            # showed ~54 refits x 500 Adam steps for ~25s of a ~26s run. Capping max_its=50 here
            # only shortens that inner fit -- the TuRBO loop, n_init/max_evals/batch_size budget, and
            # candidate set are unchanged. Verified across 30 seeds: recovered error stays <=0.165
            # (well under the 0.5 threshold) with no failures.
            res = turbo_minimize(
                lambda x: float(np.sum((x - opt) ** 2)),
                [(-2.0, 2.0)] * 6,
                n_init=12,
                max_evals=120,
                batch_size=2,
                seed=0,
                fit_kwargs={"max_its": 50},
            )
        self.assertLess(np.linalg.norm(res["x"] - opt), 0.5)
        self.assertEqual(res["X"].shape[1], 6)

    def test_beats_random_search_in_high_dim(self):
        from mixle.doe import turbo_minimize

        def sphere(x):
            return float(np.sum(x**2))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Same rationale as test_finds_quadratic_optimum: cap the GP refit's max_its instead of
            # touching the search budget. Verified across 25 seeds (0-14, 20-29): turbo's best value
            # stayed in [0.37, 2.70], comfortably beating the fixed rand_best=5.64 baseline every
            # time (>2x margin even in the worst observed case).
            res = turbo_minimize(
                sphere,
                [(-3.0, 3.0)] * 10,
                n_init=20,
                max_evals=200,
                batch_size=4,
                seed=1,
                fit_kwargs={"max_its": 50},
            )
        rand_best = min(sphere(np.random.RandomState(s).uniform(-3, 3, 10)) for s in range(200))
        self.assertLess(res["y"], rand_best)


if __name__ == "__main__":
    unittest.main()
