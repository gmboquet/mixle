"""Campaign eight (release-checklists/0.8.0-candidate-31d8e1fd-campaign.md), numerical/statistical
correctness lens: robust_regression()'s IRLS scale estimate (median(|r - median(r)|) / 0.6745) is
EXACTLY zero by construction whenever half or more of the rows share (near enough) the same
residual under the current fit -- an ordinary shape for zero-inflated counts or capped/rounded
measurements, not a contrived one -- and floors at the hardcoded 1e-8. For method='tukey' the
redescending biweight then hard-zeros essentially every row, either tripping the existing "assigned
zero weight to every observation" crash outright, or -- on a design with genuine minority signal --
settling at a self-consistent degenerate fixed point that discards the real signal with
converged=True and no warning. The verifier's extension found method='huber' is not a safe
alternative either: given enough iterations it reaches the same silent degenerate answer, just more
slowly.

The >50% breakdown point itself is real and inherent to any MAD-scaled M-estimator -- not something
this fix attempts to overcome. What was missing, and what these tests pin down, is disclosure,
matching the convention this module already uses elsewhere (GLM separation / saturation / rank
deficiency): a UserWarning plus a boolean field (RegressionFit.degenerate_scale) on a fit that
still returns, and a named-cause RuntimeError on a fit that cannot.

Both reproduction scenarios below are the campaign's own two findings:
  * a single-column (intercept-only) design where most responses are exactly zero and a minority
    carries the only real signal -- crashes under tukey with, pre-fix, no explanation of why.
  * a two-column zero-inflated design with a genuine, design-dependent minority signal -- tukey
    silently returns coefficients of [0, 0] with converged=True and no warning, pre-fix.

Every assertion below that checks for a warning, a RuntimeError substring, or
`degenerate_scale` fails against the pre-fix code (no such field existed; no warning was ever
issued; the crash message named no cause) and passes after it.
"""

import unittest
import warnings

import numpy as np

from mixle.inference import robust_regression


class PointMassSingleColumnTest(unittest.TestCase):
    """Scenario 1: an intercept-only design where >50% of responses are exactly zero."""

    def setUp(self):
        rng = np.random.RandomState(0)
        n = 200
        self.X = np.ones((n, 1))
        self.y = np.zeros(n)
        n_nonzero = int(0.3 * n)  # minority carries the only real signal (true mean ~10)
        self.y[:n_nonzero] = rng.normal(10.0, 1.0, n_nonzero)

    def test_tukey_crash_now_names_the_breakdown(self):
        # pre-fix: raises RuntimeError("robust regression assigned zero weight to every
        # observation") with no indication of why. The crash itself is legitimate (this is past
        # the estimator's real breakdown point, not a bug to paper over) -- only the message
        # changes, to actually name the mechanism instead of leaving the caller to guess.
        with self.assertRaises(RuntimeError) as ctx:
            robust_regression(self.X, self.y, method="tukey")
        message = str(ctx.exception)
        self.assertIn("breakdown", message)
        self.assertIn("zero-inflated", message)
        self.assertIn("RegressionFit.degenerate_scale", message)

    def test_huber_silently_degenerates_and_now_warns(self):
        # huber's weight (c / |u|) is never exactly zero, so it never trips the crash guard -- it
        # instead settles on a near-zero-coefficient fit (true signal ~10, fitted ~0) and, pre-fix,
        # reports converged=True with no warning at all. The verifier's extension is that this
        # persists past the default max_iter=100, so exercise a raised max_iter here too.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = robust_regression(self.X, self.y, method="huber", max_iter=500)
        self.assertTrue(fit.converged)
        self.assertLess(abs(fit.coef[0]), 1e-4)  # true signal (~10) discarded entirely
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(len(user_warnings), 1, caught)
        self.assertIn("breakdown", str(user_warnings[0].message))
        self.assertTrue(fit.degenerate_scale)


class ZeroInflatedTwoColumnTest(unittest.TestCase):
    """Scenario 2: a two-column zero-inflated design with genuine, design-dependent signal."""

    def setUp(self):
        rng = np.random.RandomState(7)
        n = 600
        predictor = rng.normal(0, 1, n)
        self.X = np.column_stack([np.ones(n), predictor])
        self.true_beta = np.array([5.0, 2.0])
        is_nonzero = rng.rand(n) < 0.35  # minority carries the design-dependent real signal
        self.y = np.zeros(n)
        self.y[is_nonzero] = self.X[is_nonzero] @ self.true_beta + rng.normal(
            0.0, 0.3, int(np.count_nonzero(is_nonzero))
        )

    def test_scenario_is_a_genuine_majority_point_mass(self):
        # documents the fixture's shape so a future edit that accidentally drops it below the
        # >50% breakdown threshold fails here, with a clear reason, instead of surfacing as a
        # confusing failure in the disclosure tests below
        self.assertGreater(np.mean(self.y == 0.0), 0.5)
        self.assertGreater(np.count_nonzero(self.y != 0.0), 20)

    def test_tukey_silently_returns_near_zero_and_now_warns(self):
        # the campaign's own "tukey currently silently returns [0, 0]" reproduction
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = robust_regression(self.X, self.y, method="tukey")
        self.assertTrue(fit.converged)
        np.testing.assert_allclose(fit.coef, [0.0, 0.0], atol=1e-6)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(len(user_warnings), 1, caught)
        self.assertIn("breakdown", str(user_warnings[0].message))
        self.assertTrue(fit.degenerate_scale)

    def test_huber_at_raised_max_iter_also_degenerates_and_now_warns(self):
        # the verifier's extension: huber reaches the SAME degenerate fixed point, just more
        # slowly, so this raises max_iter well past the 100 default to give it room to get there
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = robust_regression(self.X, self.y, method="huber", max_iter=2000)
        self.assertTrue(fit.converged)
        self.assertLess(np.max(np.abs(fit.coef)), 1e-4)  # true_beta = [5.0, 2.0] discarded entirely
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(len(user_warnings), 1, caught)
        self.assertIn("breakdown", str(user_warnings[0].message))
        self.assertTrue(fit.degenerate_scale)


class OrdinaryContaminationStaysUndisclosedTest(unittest.TestCase):
    """The existing RobustQuantileTest.test_robust_ignores_outliers shape must not newly warn.

    8% gross contamination on otherwise continuous, non-tied noise never floors the scale estimate
    (the bulk's residuals keep a real, non-degenerate spread), so this must behave exactly as
    before this fix: no warning, no `degenerate_scale`, and the true coefficients recovered.
    """

    def setUp(self):
        self.rng = np.random.RandomState(2)
        self.n = 1000
        self.X = np.column_stack([np.ones(self.n), self.rng.normal(0, 1, self.n)])
        self.beta = np.array([1.0, 2.0])

    def test_smooth_contamination_produces_no_new_warning(self):
        y = self.X @ self.beta + self.rng.normal(0, 0.3, self.n)
        y[:80] += 50.0  # gross contamination, but continuous underlying noise -- no point mass
        for method in ("huber", "tukey"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = robust_regression(self.X, y, method=method)
            user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
            self.assertEqual(user_warnings, [], f"unexpected warning(s) for method={method!r}: {user_warnings}")
            self.assertFalse(fit.degenerate_scale, f"method={method!r} incorrectly flagged as degenerate")
            np.testing.assert_allclose(fit.coef, self.beta, atol=0.2)


if __name__ == "__main__":
    unittest.main()
