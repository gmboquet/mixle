"""Release 0.8.0 wave-3 GLM regressions: B4 covariance conditioning, minimum-norm claim, separation.

Locks in three verified-defect fixes in :mod:`mixle.inference.glm`:

* B4 (blocking): the coefficient covariance was ``pinv(X'WX)`` -- the normal-equations matrix,
  whose condition is ``cond(X)^2`` -- so pinv's default cutoff silently truncated singular values
  once ``cond(X)`` passed ~1e8 and collapsed every standard error by up to 8 orders of magnitude
  with ``rank`` full, ``converged=True``, and zero warnings (measured: sunspots Poisson quadratic
  gave se 9.8e-08 where statsmodels gives 3.899; Longley intercept se off by 1.9e8x vs the NIST
  certified value). The covariance now comes from an SVD of the WEIGHTED DESIGN ``sqrt(W)X``, so
  rank and cutoff decisions happen at ``cond(X)``; a numerically rank-deficient design is warned
  about and refused per-coefficient Wald inference instead of silently full-ranked.
* t1 (major): the rank-deficiency refusal message claims the returned coefficients are the
  minimum-norm split, but the normal-equations solve returned a vector whose L2 norm was 16.7%
  above the true minimum-norm solution. The IRLS step now solves ``min ||sqrt(W)(Xb - z)||`` by
  ``lstsq`` on the weighted design, which really is minimum-norm.
* t2 (minor): perfect separation in a binomial GLM raised a bare ``RuntimeError`` describing an
  internal symptom ("IRLS produced binomial means outside (0, 1)"). It now raises
  :class:`~mixle.inference.glm.PerfectSeparationError` (a RuntimeError subclass) naming the
  separation and pointing at penalized alternatives.

The statsmodels reference constants below were computed with statsmodels 0.14.6 on the exact
seeded datasets these tests rebuild (``np.random.RandomState`` streams are stable across numpy
versions); the tests themselves need only numpy.
"""

import unittest
import warnings

import numpy as np

from mixle.inference.glm import PerfectSeparationError, glm


def _gaussian_qr_se(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Independent OLS standard errors via QR of the (unsquared) design."""
    q, r = np.linalg.qr(X)
    beta = np.linalg.solve(r, q.T @ y)
    resid = y - X @ beta
    phi = resid @ resid / (X.shape[0] - X.shape[1])
    r_inv = np.linalg.solve(r, np.eye(X.shape[1]))
    return np.sqrt(phi * np.sum(r_inv**2, axis=1))


class B4CovarianceConditioningTest(unittest.TestCase):
    """B4: standard errors must survive cond(X) ~ 1e8; cutoff decisions at cond(X), not cond(X)^2."""

    def _collinear_design(self, eps: float):
        rng = np.random.RandomState(0)
        n = 400
        x1 = rng.standard_normal(n)
        noise_dir = rng.standard_normal(n)
        y = 1.0 + 2.0 * x1 + rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x1, x1 + eps * noise_dir])
        return X, y

    def test_two_collinear_predictors_at_cond_2e8_match_statsmodels(self):
        # the acceptance criterion from the finding: rtol < 1e-4 against statsmodels' bse
        # wherever statsmodels is well-posed. statsmodels 0.14.6 OLS on this exact design
        # (cond(X) = 1.978e8): the old pinv(X'WX) covariance reported se ~ [0.047, 0.33, 0.33]
        # here -- the collinear pair 7 orders of magnitude too small.
        X, y = self._collinear_design(1e-8)
        self.assertGreater(np.linalg.cond(X), 1e8)
        fit = glm(X, y, family="gaussian")
        statsmodels_bse = np.array([4.705440745544149e-02, 4.668782318790662e06, 4.668782322109617e06])
        np.testing.assert_allclose(fit.se, statsmodels_bse, rtol=1e-4)
        self.assertEqual(fit.rank, 3)

    def test_conditioning_sweep_matches_qr_reference(self):
        # eps in {1e-2 .. 1e-8}: agreement with an independent QR-based reference throughout.
        # Before the fix the relative error crossed 1e-4 at eps=1e-6 and hit 1.0 at eps=1e-8.
        for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
            with self.subTest(eps=eps):
                X, y = self._collinear_design(eps)
                fit = glm(X, y, family="gaussian")
                np.testing.assert_allclose(fit.se, _gaussian_qr_se(X, y), rtol=1e-6)
                self.assertEqual(fit.rank, 3)

    def test_poisson_quadratic_trend_matches_weighted_qr_reference(self):
        # sunspots-shaped failure: intercept + year + year^2 on raw calendar years,
        # cond(X) ~ 1.7e9. The reference factorizes sqrt(W)X at the fitted mu -- the same
        # numbers statsmodels produced on the original sunspots reproduction.
        rng = np.random.RandomState(5)
        year = np.arange(1700.0, 2009.0)
        X = np.column_stack([np.ones_like(year), year, year**2])
        self.assertGreater(np.linalg.cond(X), 1e9)
        y = rng.poisson(50.0, size=year.size).astype(float)
        fit = glm(X, y, family="poisson")
        q, r = np.linalg.qr(X * np.sqrt(fit.fitted)[:, None])
        r_inv = np.linalg.solve(r, np.eye(3))
        reference_se = np.sqrt(np.sum(r_inv**2, axis=1))
        np.testing.assert_allclose(fit.se, reference_se, rtol=1e-6)
        # regression guard: the fixed se must NOT be the pinv(X'WX) collapse, which loses
        # the intercept column's se by many orders of magnitude on this design
        collapsed = np.sqrt(np.clip(np.diag(np.linalg.pinv((X.T * fit.fitted) @ X)), 0.0, None))
        self.assertLess(collapsed[0], 1e-3 * fit.se[0])

    def test_robust_sandwich_uses_the_factorized_bread(self):
        # robust=True reuses the same (X'WX)^-1 bread; before the fix the sandwich inherited
        # the collapse. Reference: HC0 with the bread from an SVD of the unweighted design and
        # the meat formed in a different operation order. (The B M B triple product has an
        # intrinsic cross-implementation noise floor that grows with cond(X)^2 -- statsmodels
        # HC0 itself differs from any reordering by ~1% at cond 2e7 -- so the tight comparison
        # runs at cond ~2e5 and the collapse teeth run separately at cond ~2e8.)
        X, y = self._collinear_design(1e-5)
        fit = glm(X, y, family="gaussian", robust=True)
        _, s, vt = np.linalg.svd(X, full_matrices=False)
        bread = (vt.T / s**2) @ vt
        resid = y - fit.fitted
        meat = (X * resid[:, None] ** 2).T @ X
        reference_se = np.sqrt(np.diag(bread @ meat @ bread))
        np.testing.assert_allclose(fit.se, reference_se, rtol=1e-4)

    def test_robust_sandwich_does_not_collapse_at_cond_2e8(self):
        # statsmodels 0.14.6 HC0 on this design: [4.648e-02, 5.065e+06, 5.065e+06]. The old
        # pinv(X'WX) bread reported the collinear pair ~7 ORDERS OF MAGNITUDE smaller. The
        # sandwich's intrinsic noise floor grows with cond(X)^2 and is BLAS-dependent: the same
        # tree measures se/statsmodels ratios up to 1.82 on macOS Accelerate and past 3 on
        # ubuntu OpenBLAS (the factor-3 first version of this band failed only in CI). A
        # factor-100 band is platform-robust and still refuses the guarded defect with five
        # orders of margin -- what is being pinned here is "no truncation collapse", not
        # cross-BLAS agreement, which test_robust_matches_reference_at_cond_2e5 pins where the
        # arithmetic is well-posed.
        X, y = self._collinear_design(1e-8)
        fit = glm(X, y, family="gaussian", robust=True)
        statsmodels_hc0 = np.array([4.648168808710404e-02, 5.064743770828102e06, 5.064743776942883e06])
        self.assertTrue(np.all(fit.se > statsmodels_hc0 / 100.0))
        self.assertTrue(np.all(fit.se < statsmodels_hc0 * 100.0))

    def test_numerically_rank_deficient_design_is_said_not_silently_full_ranked(self):
        # cond(X) ~ 2e15: statsmodels itself is ill-posed here. mixle must flag it -- reduced
        # rank, an explicit near-collinearity warning, and the Wald refusal -- instead of
        # reporting confident nonsense.
        X, y = self._collinear_design(1e-15)
        with self.assertWarnsRegex(UserWarning, "rank-deficient at working precision"):
            fit = glm(X, y, family="gaussian")
        self.assertEqual(fit.rank, 2)
        with self.assertRaisesRegex(ValueError, "not identified"):
            fit.p_values()

    def test_well_conditioned_poisson_matches_statsmodels(self):
        # IRLS family sanity after the refactor (statsmodels 0.14.6 GLM Poisson, tol=1e-12)
        rng = np.random.RandomState(7)
        X = np.column_stack([np.ones(300), rng.standard_normal(300), rng.standard_normal(300)])
        y = rng.poisson(np.exp(0.3 + 0.5 * X[:, 1] - 0.25 * X[:, 2])).astype(float)
        fit = glm(X, y, family="poisson")
        np.testing.assert_allclose(fit.coef, [0.27271109206000943, 0.5436848458886792, -0.2657664716810219], rtol=1e-8)
        np.testing.assert_allclose(fit.se, [0.05379878596704998, 0.05161168820573635, 0.04475353491254872], rtol=1e-6)

    def test_well_conditioned_binomial_matches_statsmodels(self):
        # IRLS family sanity after the refactor (statsmodels 0.14.6 GLM Binomial, tol=1e-12)
        rng = np.random.RandomState(11)
        X = np.column_stack([np.ones(200), rng.standard_normal(200), rng.standard_normal(200)])
        p = 1.0 / (1.0 + np.exp(-(0.4 + 1.2 * X[:, 1] - 0.8 * X[:, 2])))
        y = (rng.rand(200) < p).astype(float)
        fit = glm(X, y, family="binomial")
        np.testing.assert_allclose(fit.coef, [0.7314567043523569, 1.659414014946538, -0.9492653181404538], rtol=1e-8)
        np.testing.assert_allclose(fit.se, [0.19713749583486548, 0.2611214088096989, 0.20423861227920523], rtol=1e-6)

    def test_zero_weight_rows_do_not_disturb_rank_or_residual_df(self):
        rng = np.random.RandomState(9)
        X = np.column_stack([np.ones(50), rng.standard_normal(50), rng.standard_normal(50)])
        y = X @ np.array([1.0, 0.5, -0.5]) + rng.standard_normal(50)
        w = np.ones(50)
        w[:10] = 0.0
        fit = glm(X, y, family="gaussian", weights=w)
        self.assertEqual(fit.rank, 3)
        self.assertEqual(fit.residual_df, 40 - 3)
        np.testing.assert_allclose(fit.se, _gaussian_qr_se(X[10:], y[10:]), rtol=1e-8)


class MinimumNormClaimTest(unittest.TestCase):
    """t1: the refusal message's 'minimum-norm split' claim must be true of the returned coef."""

    def _duplicated_column_fit(self):
        rng = np.random.RandomState(1)
        base = rng.standard_normal((100, 3))
        X = np.column_stack([base, base[:, 1]])  # duplicated column: rank 3 of 4
        y = base @ np.array([1.0, 2.0, -1.0]) + rng.standard_normal(100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # the (asserted-elsewhere) rank warning
            return X, y, glm(X, y, family="gaussian")

    def test_rank_deficient_gaussian_coef_is_the_minimum_norm_solution(self):
        # before the fix ||coef|| exceeded the true minimum-norm vector's norm by 16.7%
        X, y, fit = self._duplicated_column_fit()
        min_norm = np.linalg.pinv(X) @ y
        np.testing.assert_allclose(fit.coef, min_norm, atol=1e-8)
        self.assertAlmostEqual(np.linalg.norm(fit.coef), np.linalg.norm(min_norm), places=10)

    def test_min_norm_representative_splits_the_duplicated_effect_equally(self):
        X, y, fit = self._duplicated_column_fit()
        self.assertAlmostEqual(fit.coef[1], fit.coef[3], places=8)

    def test_refusal_still_names_the_minimum_norm_split(self):
        _, _, fit = self._duplicated_column_fit()
        self.assertEqual(fit.rank, 3)
        with self.assertRaisesRegex(ValueError, "minimum-norm"):
            fit.z_values()

    def test_exactly_duplicated_column_fits_finite_and_warns(self):
        # the historical no-crash contract (glm_robust_solve_test) plus the new announcement
        rng = np.random.RandomState(0)
        x0 = rng.randn(200)
        X = np.column_stack([np.ones(200), x0, x0])
        y = (1.0 / (1.0 + np.exp(-(0.5 + 1.5 * x0))) > rng.rand(200)).astype(float)
        with self.assertWarnsRegex(UserWarning, "rank-deficient at working precision"):
            fit = glm(X, y, family="binomial")
        self.assertTrue(np.all(np.isfinite(fit.coef)))
        self.assertTrue(np.all(np.isfinite(fit.se)))
        self.assertEqual(fit.rank, 2)


class PerfectSeparationTest(unittest.TestCase):
    """t2: separation is named as a statistical condition, not an internal numeric symptom."""

    def setUp(self):
        x = np.concatenate([np.linspace(-3.0, -1.0, 20), np.linspace(1.0, 3.0, 20)])
        self.X = np.column_stack([np.ones(40), x])
        self.y = (x > 0).astype(float)

    def test_perfect_separation_raises_the_named_error(self):
        with self.assertRaisesRegex(PerfectSeparationError, "perfect separation detected") as ctx:
            glm(self.X, self.y, family="binomial")
        message = str(ctx.exception)
        self.assertIn("no finite maximum-likelihood estimate", message)
        self.assertIn("ridge_regression", message)  # actionable guidance, not just a diagnosis

    def test_separation_error_is_a_runtime_error_subclass(self):
        # pre-existing `except RuntimeError` handlers must keep catching the condition
        with self.assertRaises(RuntimeError):
            glm(self.X, self.y, family="binomial")

    def test_probit_and_cloglog_separation_are_also_named(self):
        for link in ("probit", "cloglog"):
            with self.subTest(link=link), self.assertRaises(PerfectSeparationError):
                glm(self.X, self.y, family="binomial", link=link)

    def test_overlapping_classes_fit_cleanly_with_no_warning(self):
        # guard-overreach control: a legitimate overlapped binomial fit must not be rejected
        # or warned about by the separation/collinearity machinery
        rng = np.random.RandomState(3)
        x = rng.standard_normal(120)
        p = 1.0 / (1.0 + np.exp(-(0.5 + 1.0 * x)))
        y = (rng.rand(120) < p).astype(float)
        X = np.column_stack([np.ones(120), x])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = glm(X, y, family="binomial")
        self.assertEqual([str(w.message) for w in caught], [])
        self.assertTrue(fit.converged)
        self.assertTrue(np.all(np.isfinite(fit.se)))


if __name__ == "__main__":
    unittest.main()
