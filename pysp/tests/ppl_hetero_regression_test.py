"""Heteroskedastic (location-scale) regression in pysp.ppl: a linear predictor in the *scale* slot.

`Normal(mean_pred, free*Field("x") + free)` and the same for `LogNormal` model the dispersion as a
log-linear function of covariates (the "not homoskedastic" capability). The mean keeps its identity
link; the scale uses a log link. These tests check coefficient recovery, that the previously-failing
scale-slot regression no longer raises, and that the homoskedastic path is unchanged.
"""

import unittest

import numpy as np

from pysp.ppl import Field, LogNormal, Normal, free


class HeteroskedasticRegressionTestCase(unittest.TestCase):
    def test_scale_slot_no_longer_raises(self):
        rng = np.random.RandomState(0)
        x = rng.uniform(-1, 1, 500)
        y = rng.normal(0.0, np.exp(0.1 * x))
        # used to raise TypeError: float() argument ... not '_LinearPredictor'
        fit = Normal(free, free * Field("x") + free).fit(y, given={"x": x})
        self.assertIn("x", fit.result.scale_coefficients)

    def test_heteroskedastic_normal_recovers_mean_and_scale(self):
        rng = np.random.RandomState(1)
        x = rng.uniform(-1, 1, 8000)
        y = rng.normal(2.0 + 1.5 * x, np.exp(0.2 + 0.6 * x))
        fit = Normal(free * Field("x") + free, free * Field("x") + free).fit(y, given={"x": x})
        mc = fit.result.coefficients
        sc = fit.result.scale_coefficients
        self.assertAlmostEqual(mc["x"]["mean"], 1.5, delta=0.15)
        self.assertAlmostEqual(mc["intercept"]["mean"], 2.0, delta=0.1)
        self.assertAlmostEqual(sc["x"]["mean"], 0.6, delta=0.12)
        self.assertAlmostEqual(sc["intercept"]["mean"], 0.2, delta=0.1)
        # predict returns per-row loc and scale, with scale increasing in x
        p = fit.result.predict({"x": np.array([-1.0, 1.0])})
        self.assertLess(p["scale"][0], p["scale"][1])

    def test_heteroskedastic_lognormal_scale(self):
        rng = np.random.RandomState(2)
        z = rng.uniform(0, 1, 8000)
        y = np.exp(rng.normal(0.5, np.exp(-0.5 + 0.8 * z)))  # positive, right-skewed offset
        fit = LogNormal(free, free * Field("z") + free).fit(y, given={"z": z})
        self.assertAlmostEqual(fit.result.coefficients["intercept"]["mean"], 0.5, delta=0.1)
        self.assertAlmostEqual(fit.result.scale_coefficients["z"]["mean"], 0.8, delta=0.15)

    def test_lognormal_requires_positive_data(self):
        with self.assertRaises(ValueError):
            LogNormal(free, free * Field("z") + free).fit(
                np.array([-1.0, 2.0, 3.0]), given={"z": np.array([0.0, 0.5, 1.0])}
            )

    def test_homoskedastic_regression_unchanged(self):
        rng = np.random.RandomState(3)
        x = rng.uniform(-1, 1, 3000)
        y = rng.normal(1.0 - 2.0 * x, 0.7)
        fit = Normal(free * Field("x") + free, free).fit(y, given={"x": x})  # constant-scale path
        self.assertAlmostEqual(fit.result.coefficients["x"]["mean"], -2.0, delta=0.1)
        self.assertAlmostEqual(fit.result.sigma, 0.7, delta=0.05)

    def test_normal_regression_prior_handles_use_ridge_convention(self):
        x = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = np.asarray([-4.1, -1.4, 0.2, 2.1, 4.8])
        slope = Normal(0.5, 2.0, name="slope")
        intercept = Normal(-0.25, 3.0, name="intercept")
        fit = Normal(slope * Field("x") + intercept, 2.0).fit(y, given={"x": x})

        X = np.column_stack([x, np.ones_like(x)])
        m0 = np.asarray([0.5, -0.25])
        P0 = np.diag([1.0 / 2.0**2, 1.0 / 3.0**2])
        A = X.T @ X + P0
        expected_beta = np.linalg.inv(A) @ (X.T @ y + P0 @ m0)
        expected_cov = 2.0**2 * np.linalg.inv(A)

        np.testing.assert_allclose(fit.result.beta, expected_beta, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(fit.result.cov, expected_cov, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
