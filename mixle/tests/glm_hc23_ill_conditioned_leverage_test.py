# T1-02: HC2/HC3 leverage was computed as diag(X @ xtwx_inv @ X.T), where xtwx_inv is built from
# 1/singular**2 -- squaring the design's condition number right back in, exactly what audit B4
# rejected pinv(X'WX) for elsewhere in this same function. At cond(X) ~ 1e9 this produced
# "leverage" outside the [0, 1] bound (observed max 4.11, sum 24.6 for a rank-4 design) and a
# false-positive "leverage 1, e.g. a dummy level with a single observation" refusal on a design
# with no dummy coding at all -- just two highly-correlated continuous predictors. The fix reads
# leverage straight off U (diag(U @ U.T)) from the SVD already factored for xtwx_inv, which is
# correct by construction (bounded in [0, 1], sums to the rank) at cond(X) precision.

import unittest

import numpy as np

from mixle.inference.glm import glm


def _high_collinearity_design(n=200, cond_target=1e9, seed=0):
    rng = np.random.default_rng(seed)
    intercept = np.ones(n)
    x1 = rng.normal(size=n)
    x2 = x1 + rng.normal(scale=1.0 / cond_target, size=n)  # near-duplicate of x1
    x3 = rng.normal(size=n)
    X = np.column_stack([intercept, x1, x2, x3])
    beta_true = np.array([1.0, 0.5, -0.3, 0.2])
    y = X @ beta_true + rng.normal(scale=1.0, size=n)
    return X, y


def _svd_leverage(X):
    # the numerically stable reference: the hat diagonal read straight off U, never through a
    # quadratic form against 1/singular**2
    u, _, _ = np.linalg.svd(X, full_matrices=False)
    return np.sum(u * u, axis=1)


class Hc23IllConditionedLeverageTest(unittest.TestCase):
    def test_no_false_positive_singleton_refusal_on_non_dummy_high_collinearity_design(self):
        # this design has zero dummy/discrete coding -- just two correlated continuous
        # predictors at cond(X) ~ 1.9e9 -- so HC2/HC3 must fit, not refuse
        X, y = _high_collinearity_design()
        self.assertGreater(np.linalg.cond(X), 1e8)
        for robust in ("HC2", "HC3"):
            with self.subTest(robust=robust):
                res = glm(X, y, family="gaussian", link="identity", robust=robust)
                self.assertTrue(np.all(np.isfinite(res.se)))

    def test_leverage_matches_svd_reference_to_high_precision_across_condition_sweep(self):
        # rebuild the same HC3 sandwich the library computes, but swap in the library's own
        # leverage figure at each condition number and compare its downstream standard errors
        # against a build that instead uses diag(U U') taken straight from a fresh SVD of X --
        # equivalent by construction, so any daylight between them is exactly the cancellation
        # the fix removes
        for exponent in (5, 6, 7, 8, 9):
            cond_target = 10.0**exponent
            X, y = _high_collinearity_design(cond_target=cond_target)
            with self.subTest(cond_target=cond_target):
                ref_leverage = _svd_leverage(X)
                # bounded in [0, 1] and sums to the rank (4), at every condition number tested
                self.assertTrue(np.all(ref_leverage >= -1e-9))
                self.assertTrue(np.all(ref_leverage <= 1.0 + 1e-9))
                self.assertAlmostEqual(ref_leverage.sum(), 4.0, places=6)

                res = glm(X, y, family="gaussian", link="identity", robust="HC3")
                mu = X @ np.linalg.lstsq(X, y, rcond=None)[0]
                resid = y - mu
                u, s, vt = np.linalg.svd(X, full_matrices=False)
                xtx_inv = (vt.T * (1.0 / s**2)) @ vt
                unit_score = X * resid[:, None]
                row_scale = 1.0 / (1.0 - ref_leverage)
                half = (row_scale[:, None] * unit_score) @ xtx_inv
                se_ref = np.sqrt(np.diag(half.T @ half))
                np.testing.assert_allclose(res.se, se_ref, rtol=1e-8)

    def test_genuine_singleton_dummy_level_still_refuses(self):
        # a real degenerate case -- a dummy level fitted by exactly one observation, h_i == 1 --
        # must still be caught; the fix must not widen the refusal away
        x = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([0.9, 1.4, 1.1, 0.6, 3.2])
        for robust in ("HC2", "HC3"):
            with self.subTest(robust=robust):
                with self.assertRaisesRegex(ValueError, "leverage 1"):
                    glm(x, y, family="gaussian", robust=robust)

    def test_hc0_hc1_and_model_covariance_unaffected_by_ill_conditioning(self):
        # HC0/HC1/model-based covariance never consume the leverage computation -- confirm they
        # still fit finite, sane values at the same condition number that broke HC2/HC3
        X, y = _high_collinearity_design()
        for robust in (False, True, "HC0", "HC1"):
            with self.subTest(robust=robust):
                res = glm(X, y, family="gaussian", link="identity", robust=robust)
                self.assertTrue(np.all(np.isfinite(res.se)))


if __name__ == "__main__":
    unittest.main()
