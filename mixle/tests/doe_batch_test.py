"""Rigorous batch (multi-point) Bayesian optimization (mixle.doe.batch)."""

import importlib.util
import unittest
import warnings

import numpy as np
from scipy.stats import norm

from mixle.doe.batch import monte_carlo_qei, propose_local_penalization, propose_qei_batch

HAS_TORCH = importlib.util.find_spec("torch") is not None  # the proposal drivers fit the torch GP surrogate


class _StubSurrogate:
    """Fake GP surrogate for exercising propose_qei_batch/propose_local_penalization's OWN validation
    and selection logic without requiring torch. The real default surrogate
    (mixle.models.gaussian_process.GaussianProcessRegressor) is torch-only, but both drivers accept any
    object with .fit/.predict via their `gp=` injection point, so a deterministic stand-in exercises the
    batch-building loop directly. Mean/covariance are a well-formed function of the candidate points alone
    (optimum at the origin, an RBF-like correlation structure that is PSD by construction) -- optionally
    with a NaN patch over a caller-chosen region, to simulate a localized GP pathology.
    """

    def __init__(self, nan_where=None):
        self._nan_where = nan_where

    def fit(self, x, y, **kwargs):
        return self

    def predict(self, x, y, pts, return_cov=True):
        pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
        mean = -np.sum(pts**2, axis=1)
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        cov = 0.1 * np.exp(-(d**2)) + 1e-9 * np.eye(pts.shape[0])
        if self._nan_where is not None:
            bad = self._nan_where(pts)
            mean = mean.copy()
            mean[bad] = np.nan
            cov = cov.copy()
            cov[np.ix_(bad, bad)] = np.nan
        return mean, cov


class MonteCarloQeiTest(unittest.TestCase):
    """The q-EI estimator is pure NumPy and checkable in closed form."""

    def test_q1_matches_analytic_ei(self):
        best, mu, sigma = 1.0, 0.5, 0.8
        z = (best - mu) / sigma
        analytic = (best - mu) * norm.cdf(z) + sigma * norm.pdf(z)  # EI, minimization
        mc = monte_carlo_qei([mu], [[sigma**2]], best, maximize=False, samples=200000, seed=0)
        self.assertAlmostEqual(mc, analytic, places=2)

    def test_correlated_duplicate_gives_no_batch_gain(self):
        # two perfectly-correlated identical points have the q-EI of a single point (the property
        # kriging-believer violates): batching duplicates is worthless.
        best, mu, var = 1.0, 0.5, 0.64
        single = monte_carlo_qei([mu], [[var]], best, samples=200000, seed=0)
        dup = monte_carlo_qei([mu, mu], [[var, var], [var, var]], best, samples=200000, seed=0)
        self.assertAlmostEqual(dup, single, places=2)

    def test_independent_points_increase_qei(self):
        best, mu, var = 1.0, 0.5, 0.64
        single = monte_carlo_qei([mu], [[var]], best, samples=200000, seed=0)
        indep = monte_carlo_qei([mu, mu], [[var, 0.0], [0.0, var]], best, samples=200000, seed=0)
        self.assertGreater(indep, single)


class SafeCholeskyValidationTest(unittest.TestCase):
    """MXR-080-0166: _safe_cholesky (via monte_carlo_qei) used to replace ANY covariance it couldn't
    factor -- including a genuinely invalid one -- with its diagonal, silently downgrading the module's
    central "true joint posterior" claim to an independent approximation while still returning a
    plausible-looking number. It must now distinguish "not a valid covariance at all" (reject) from
    "valid but numerically singular" (heal with small, quantified, reported jitter) and never blur the
    two.
    """

    def test_indefinite_covariance_is_rejected_not_silently_downgraded(self):
        # the audit's exact example: [[1,2],[2,1]] has eigenvalues 3 and -1 (ac=1 < b^2=4) -- not a
        # covariance matrix at all, for ANY jitter budget. Must raise, not quietly fall back to diagonal.
        with self.assertRaisesRegex(ValueError, "positive semi-definite"):
            monte_carlo_qei([0.0, 0.0], [[1, 2], [2, 1]], best=1.0, samples=1000, seed=0)

    def test_a_severely_non_psd_covariance_is_also_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive semi-definite"):
            monte_carlo_qei([0.0, 0.0], [[-1.0, 0.0], [0.0, -1.0]], best=1.0, samples=1000, seed=0)

    def test_near_singular_covariance_is_healed_with_quantified_reported_jitter(self):
        # an exactly rank-deficient (perfectly correlated duplicate) covariance IS a legitimate joint
        # covariance -- min eigenvalue 0.0, not negative -- but a direct Cholesky factorization fails on
        # it, so this must heal via jitter rather than raise. The healing must not be invisible: a
        # RuntimeWarning should report it, and the jitter must be small (a numerical nudge, not a change
        # of model) relative to the covariance's own scale.
        best, mu, var = 1.0, 0.5, 0.64
        cov = [[var, var], [var, var]]  # exactly singular: eigenvalues are 0 and 2*var
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            val = monte_carlo_qei([mu, mu], cov, best, samples=200000, seed=0)
        jitter_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(jitter_warnings), 1)
        message = str(jitter_warnings[0].message)
        self.assertIn("jitter=", message)
        reported_jitter = float(message.split("jitter=")[1].split()[0])
        self.assertGreater(reported_jitter, 0.0)
        self.assertLess(reported_jitter, 1e-4 * var)  # a small numerical nudge, not a material perturbation
        # healed value must still reflect the TRUE JOINT computation (perfectly-correlated duplicate has
        # the single-point q-EI), not the independent-model value the old diagonal fallback would give.
        single = monte_carlo_qei([mu], [[var]], best, samples=200000, seed=0)
        indep = monte_carlo_qei([mu, mu], [[var, 0.0], [0.0, var]], best, samples=200000, seed=0)
        self.assertAlmostEqual(val, single, places=2)
        self.assertGreater(indep - single, 0.01)  # confirms independent really is a different number here

    def test_well_conditioned_covariance_needs_no_jitter(self):
        # negative control: a genuinely well-conditioned covariance must factor on the first, unperturbed
        # attempt -- no warning, and the usual closed-form-matching value from test_q1_matches_analytic_ei.
        best, mu, sigma = 1.0, 0.5, 0.8
        z = (best - mu) / sigma
        analytic = (best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            val = monte_carlo_qei([mu], [[sigma**2]], best, samples=200000, seed=0)
        self.assertEqual([w for w in caught if issubclass(w.category, RuntimeWarning)], [])
        self.assertAlmostEqual(val, analytic, places=2)

    def test_non_finite_covariance_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not finite"):
            monte_carlo_qei([0.0, 0.0], [[1.0, 0.0], [0.0, np.nan]], best=1.0, samples=100, seed=0)

    def test_zero_covariance_is_valid_and_reports_absolute_jitter(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = monte_carlo_qei([0.5, 0.5], np.zeros((2, 2)), best=1.0, samples=32, seed=0)
        self.assertAlmostEqual(value, 0.5, places=9)
        messages = [str(item.message) for item in caught if issubclass(item.category, RuntimeWarning)]
        self.assertEqual(len(messages), 1)
        self.assertIn("absolute jitter for zero covariance scale", messages[0])

    def test_asymmetric_covariance_is_rejected_not_replaced(self):
        with self.assertRaisesRegex(ValueError, "not symmetric"):
            monte_carlo_qei([0.0, 0.0], [[1.0, 0.2], [0.0, 1.0]], best=1.0, samples=32, seed=0)


class MonteCarloQeiInputValidationTest(unittest.TestCase):
    """MXR-080-0167 at the monte_carlo_qei level: nonpositive/fractional sample counts, mean/cov shape
    mismatches, and non-finite posterior moments used to be silently accepted (truncated, broadcast into
    an opaque numpy crash, or propagated as a NaN result) instead of rejected with a clear error.
    """

    def test_nonpositive_samples_rejected(self):
        with self.assertRaisesRegex(ValueError, "samples"):
            monte_carlo_qei([0.5], [[0.64]], best=1.0, samples=0, seed=0)
        with self.assertRaisesRegex(ValueError, "samples"):
            monte_carlo_qei([0.5], [[0.64]], best=1.0, samples=-5, seed=0)

    def test_fractional_samples_rejected(self):
        # int(10.5) == 10 would silently truncate instead of naming the invalid input.
        with self.assertRaisesRegex(ValueError, "samples"):
            monte_carlo_qei([0.5], [[0.64]], best=1.0, samples=10.5, seed=0)

    def test_boolean_samples_rejected(self):
        for bad in (True, False, np.bool_(True)):
            with self.assertRaises((TypeError, ValueError)):
                monte_carlo_qei([0.5], [[0.64]], best=1.0, samples=bad, seed=0)

    def test_cov_mean_shape_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            monte_carlo_qei([0.5, 0.5, 0.5], [[1.0, 0.0], [0.0, 1.0]], best=1.0, samples=100, seed=0)

    def test_non_finite_mean_rejected(self):
        with self.assertRaisesRegex(ValueError, "not finite|non-finite"):
            monte_carlo_qei([np.nan, 0.5], [[1.0, 0.0], [0.0, 1.0]], best=1.0, samples=100, seed=0)

    def test_non_finite_incumbent_and_non_boolean_sense_are_rejected(self):
        for invalid in (np.nan, np.inf, -np.inf):
            with self.assertRaisesRegex(ValueError, "best"):
                monte_carlo_qei([0.5], [[0.64]], best=invalid, samples=10, seed=0)
        for invalid in ("false", 0, np.bool_(True)):
            with self.assertRaises(TypeError):
                monte_carlo_qei([0.5], [[0.64]], best=1.0, maximize=invalid, samples=10, seed=0)

    def test_overflowing_finite_inputs_do_not_publish_infinite_qei(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            monte_carlo_qei([1e308], [[1.0]], best=-1e308, maximize=True, samples=10, seed=0)

    def test_a_well_posed_call_is_unaffected(self):
        # negative control: ordinary, valid inputs still work exactly as before.
        best, mu, sigma = 1.0, 0.5, 0.8
        z = (best - mu) / sigma
        analytic = (best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        val = monte_carlo_qei([mu], [[sigma**2]], best, maximize=False, samples=200000, seed=0)
        self.assertAlmostEqual(val, analytic, places=2)


class BatchDriverInputValidationTest(unittest.TestCase):
    """MXR-080-0167 at the propose_qei_batch / propose_local_penalization level. Uses _StubSurrogate
    (dependency-injected via `gp=`) instead of the real torch GP, so these run with or without torch and
    isolate the drivers' OWN validation/selection logic from surrogate-fit behavior.
    """

    def setUp(self):
        self.bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        rng = np.random.RandomState(0)
        self.x = rng.uniform(-1, 1, size=(5, 2))
        self.y = -np.sum(self.x**2, axis=1)

    # -- nonpositive/fractional counts --------------------------------------------------------------

    def test_propose_qei_batch_rejects_nonpositive_or_fractional_q(self):
        for bad_q in (0, -1, 2.5):
            with self.assertRaises(ValueError):
                propose_qei_batch(
                    self.x, self.y, self.bounds, q=bad_q, n_candidates=10, mc_samples=32, gp=_StubSurrogate()
                )

    def test_propose_qei_batch_rejects_nonpositive_or_fractional_n_candidates(self):
        for bad_n in (0, -1, 10.5):
            with self.assertRaises(ValueError):
                propose_qei_batch(
                    self.x, self.y, self.bounds, q=2, n_candidates=bad_n, mc_samples=32, gp=_StubSurrogate()
                )

    def test_propose_qei_batch_rejects_nonpositive_or_fractional_mc_samples(self):
        for bad_mc in (0, -1, 10.5):
            with self.assertRaises(ValueError):
                propose_qei_batch(
                    self.x, self.y, self.bounds, q=2, n_candidates=10, mc_samples=bad_mc, gp=_StubSurrogate()
                )

    def test_boolean_counts_are_rejected_by_both_batch_drivers(self):
        for bad in (True, np.bool_(True)):
            with self.assertRaises((TypeError, ValueError)):
                propose_qei_batch(
                    self.x, self.y, self.bounds, q=bad, n_candidates=10, mc_samples=32, gp=_StubSurrogate()
                )
            with self.assertRaises((TypeError, ValueError)):
                propose_local_penalization(self.x, self.y, self.bounds, q=bad, n_candidates=10, gp=_StubSurrogate())

    def test_propose_qei_batch_rejects_batch_larger_than_candidates(self):
        with self.assertRaisesRegex(ValueError, "q <= n_candidates"):
            propose_qei_batch(self.x, self.y, self.bounds, q=3, n_candidates=2, mc_samples=16, gp=_StubSurrogate())

    def test_propose_local_penalization_rejects_nonpositive_or_fractional_q(self):
        for bad_q in (0, -1, 2.5):
            with self.assertRaises(ValueError):
                propose_local_penalization(self.x, self.y, self.bounds, q=bad_q, n_candidates=10, gp=_StubSurrogate())

    def test_propose_local_penalization_rejects_nonpositive_or_fractional_n_candidates(self):
        for bad_n in (0, -1, 10.5):
            with self.assertRaises(ValueError):
                propose_local_penalization(self.x, self.y, self.bounds, q=2, n_candidates=bad_n, gp=_StubSurrogate())

    # -- all-NaN merit --------------------------------------------------------------------------------

    def test_propose_qei_batch_all_nan_merit_raises_instead_of_leaving_best_c_none(self):
        # MXR-080-0170's shared _fit_surrogate (bayesopt._validate_observations) now rejects a NaN
        # observation before either driver ever reaches its own merit loop, so a NaN in y no longer
        # reaches this check -- that path is covered instead by bayesopt's own validation tests. To
        # still exercise a NaN surrogate prediction specifically, use a well-formed y and make the
        # *surrogate's prediction* uniformly NaN instead (x/y stay perfectly finite).
        #
        # That NaN is now caught even earlier than this function's own "no candidate produced a finite"
        # loop guard (lines below): monte_carlo_qei itself validates its mean argument is finite and
        # raises immediately (MXR-080-0167), before ever returning a value the per-candidate comparison
        # loop could see. So the original best_c=None scenario this test targeted is no longer reachable
        # via a NaN-surrogate route -- monte_carlo_qei's own guard fires first, on the very first
        # candidate, which is a strictly earlier and equally-safe outcome (never a silent best_c=None).
        # The loop-level "no candidate produced a finite" guard remains as defense in depth for any other
        # future path that might return -- rather than raise on -- a non-finite merit.
        all_nan_gp = _StubSurrogate(nan_where=lambda pts: np.ones(len(pts), dtype=bool))
        with self.assertRaisesRegex(ValueError, "mean contains non-finite values"):
            propose_qei_batch(self.x, self.y, self.bounds, q=2, n_candidates=10, mc_samples=32, gp=all_nan_gp)

    def test_propose_local_penalization_all_nan_merit_raises_instead_of_arbitrary_pick(self):
        # See the twin qei test above: a NaN in y is now caught earlier by bayesopt._fit_surrogate's
        # shared observation validation (MXR-080-0170), so this drives the surrogate's own prediction to
        # all-NaN instead, keeping x/y finite, to exercise this function's own merit-level guard.
        all_nan_gp = _StubSurrogate(nan_where=lambda pts: np.ones(len(pts), dtype=bool))
        with self.assertRaisesRegex(ValueError, "no candidate produced a finite"):
            propose_local_penalization(self.x, self.y, self.bounds, q=2, n_candidates=10, gp=all_nan_gp)

    def test_propose_local_penalization_never_selects_a_nan_merit_candidate_over_a_finite_one(self):
        # the more precise failure np.argmax([1, nan, 5]) == 1 (NOT 2) causes: with SOME but not all
        # candidates NaN, argmax on the raw merit array would hand a NaN-scored candidate a win over
        # genuinely better finite ones. Candidates with x0 > 0.5 get a NaN posterior from the stub; the
        # fix must never place a pick there, even though plenty of finite-merit candidates remain.
        gp = _StubSurrogate(nan_where=lambda pts: pts[:, 0] > 0.5)
        batch = propose_local_penalization(self.x, self.y, self.bounds, q=3, n_candidates=40, seed=0, gp=gp)
        self.assertFalse((batch[:, 0] > 0.5).any())

    # -- negative controls ----------------------------------------------------------------------------

    def test_propose_qei_batch_negative_control_still_selects_sensible_candidates(self):
        batch = propose_qei_batch(
            self.x, self.y, self.bounds, q=2, n_candidates=20, mc_samples=64, seed=0, gp=_StubSurrogate()
        )
        self.assertEqual(batch.shape, (2, 2))
        self.assertTrue(np.isfinite(batch).all())
        self.assertTrue(((batch >= -1.0) & (batch <= 1.0)).all())
        self.assertEqual(np.unique(batch, axis=0).shape[0], 2)

    def test_propose_local_penalization_negative_control_still_selects_sensible_candidates(self):
        batch = propose_local_penalization(
            self.x, self.y, self.bounds, q=2, n_candidates=20, seed=0, gp=_StubSurrogate()
        )
        self.assertEqual(batch.shape, (2, 2))
        self.assertTrue(np.isfinite(batch).all())
        self.assertTrue(((batch >= -1.0) & (batch <= 1.0)).all())


@unittest.skipUnless(HAS_TORCH, "GP surrogate requires torch")
class BatchProposalTest(unittest.TestCase):
    def _problem(self):
        rng = np.random.RandomState(0)
        x = rng.uniform(-3, 3, (12, 1))
        y = np.sin(3 * x[:, 0]) + 0.3 * x[:, 0] ** 2  # two basins
        return x, y, [(-3.0, 3.0)]

    def test_qei_batch_is_diverse(self):
        from mixle.doe import propose_qei_batch

        x, y, bounds = self._problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = propose_qei_batch(x, y, bounds, q=3, n_candidates=120, mc_samples=128, seed=1)
        self.assertEqual(b.shape, (3, 1))
        self.assertTrue((b >= -3).all() and (b <= 3).all())
        pdist = [abs(b[i, 0] - b[j, 0]) for i in range(3) for j in range(i + 1, 3)]
        self.assertGreater(min(pdist), 0.1)  # not near-duplicate points

    def test_local_penalization_is_diverse(self):
        from mixle.doe import propose_local_penalization

        x, y, bounds = self._problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = propose_local_penalization(x, y, bounds, q=3, n_candidates=400, seed=1)
        self.assertEqual(b.shape, (3, 1))
        pdist = [abs(b[i, 0] - b[j, 0]) for i in range(3) for j in range(i + 1, 3)]
        self.assertGreater(min(pdist), 0.3)  # floored spacing keeps the batch spread


if __name__ == "__main__":
    unittest.main()
