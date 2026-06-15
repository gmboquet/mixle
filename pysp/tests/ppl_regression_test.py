"""Tests for pysp.ppl linear regression (Field + linear predictor)."""
import numpy as np
import unittest

from pysp.ppl import Normal, Bernoulli, Poisson, Field, free


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


class GLMTestCase(unittest.TestCase):

    def test_logistic_regression(self):
        rng = np.random.RandomState(0)
        N = 6000
        x, z = rng.normal(0, 1, N), rng.normal(0, 1, N)
        p = 1.0 / (1.0 + np.exp(-(2.0 * x - 1.0 * z + 0.5)))
        y = (rng.random(N) < p).astype(float)
        m = Bernoulli(free * Field("x") + free * Field("z") + free).fit(
            list(y), given={"x": list(x), "z": list(z)})
        c = m.params
        self.assertAlmostEqual(c["x"]["mean"], 2.0, delta=0.2)
        self.assertAlmostEqual(c["z"]["mean"], -1.0, delta=0.2)
        self.assertAlmostEqual(c["intercept"]["mean"], 0.5, delta=0.2)
        # prediction returns a probability through the logit link
        prob = float(m.result.predict({"x": [0.0], "z": [0.0]})[0])
        self.assertAlmostEqual(prob, 1.0 / (1.0 + np.exp(-0.5)), delta=0.05)

    def test_poisson_regression(self):
        rng = np.random.RandomState(1)
        N = 6000
        x = rng.normal(0, 1, N)
        y = rng.poisson(np.exp(0.5 * x + 0.3)).astype(float)
        m = Poisson(free * Field("x") + free).fit(list(y), given={"x": list(x)})
        self.assertAlmostEqual(m.params["x"]["mean"], 0.5, delta=0.1)
        self.assertAlmostEqual(m.params["intercept"]["mean"], 0.3, delta=0.1)
        self.assertGreater(float(m.result.predict({"x": [0.0]})[0]), 0.0)   # a rate


if __name__ == "__main__":
    unittest.main()
