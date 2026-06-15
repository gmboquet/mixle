"""Tests for pysp.ppl linear regression (Field + linear predictor)."""
import numpy as np
import unittest

from pysp.ppl import Normal, Field, free


class RegressionTestCase(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(0)
        self.N = 4000
        self.x = rng.normal(0, 1, self.N)
        self.z = rng.normal(0, 1, self.N)
        # true: y = 2 x - 1.5 z + 0.7 + N(0, 0.5)
        self.y = 2.0 * self.x - 1.5 * self.z + 0.7 + rng.normal(0, 0.5, self.N)

    def test_ols_multi_covariate(self):
        m = Normal(free * Field("x") + free * Field("z") + free, free).fit(
            list(self.y), given={"x": list(self.x), "z": list(self.z)})
        c = m.params
        self.assertAlmostEqual(c["x"]["mean"], 2.0, delta=0.05)
        self.assertAlmostEqual(c["z"]["mean"], -1.5, delta=0.05)
        self.assertAlmostEqual(c["intercept"]["mean"], 0.7, delta=0.05)
        self.assertAlmostEqual(m.result.sigma, 0.5, delta=0.05)

    def test_bayesian_regression_posterior_and_predict(self):
        a, b = Normal(0, 10), Normal(0, 10)
        m = Normal(a * Field("x") + b, free).fit(
            list(self.y - (-1.5 * self.z)), given={"x": list(self.x)})  # drop z term
        # coefficient posterior available by handle, name, index
        self.assertAlmostEqual(m.posterior(a).mean(), 2.0, delta=0.1)
        self.assertAlmostEqual(m.result.coefficients["x"]["mean"], 2.0, delta=0.1)
        # prediction at new covariates
        pred = m.result.predict({"x": [0.0, 1.0, 2.0]})
        self.assertAlmostEqual(pred[0], 0.7, delta=0.1)
        self.assertAlmostEqual(pred[1], 2.7, delta=0.1)
        self.assertAlmostEqual(pred[2], 4.7, delta=0.15)

    def test_known_sigma(self):
        m = Normal(free * Field("x") + free, 0.5).fit(
            list(2.0 * self.x + 0.7 + np.random.RandomState(1).normal(0, 0.5, self.N)),
            given={"x": list(self.x)})
        self.assertAlmostEqual(m.result.sigma, 0.5, delta=1e-9)   # fixed, not estimated
        self.assertAlmostEqual(m.params["x"]["mean"], 2.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
