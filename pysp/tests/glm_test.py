"""GLM + penalized / robust / quantile regression (pysp.inference.glm)."""

import unittest

import numpy as np

from pysp.inference import (
    elastic_net,
    glm,
    lasso,
    quantile_regression,
    ridge_regression,
    robust_regression,
)


class GLMTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.n = 3000
        self.X = np.column_stack([np.ones(self.n), self.rng.normal(0, 1, self.n), self.rng.normal(0, 1, self.n)])

    def test_logistic_recovers_coefficients(self):
        beta = np.array([0.5, 1.5, -1.0])
        p = 1.0 / (1.0 + np.exp(-self.X @ beta))
        y = (self.rng.rand(self.n) < p).astype(float)
        r = glm(self.X, y, family="binomial")
        np.testing.assert_allclose(r.coef, beta, atol=0.15)
        self.assertEqual(r.link, "logit")
        self.assertTrue(np.all(r.se > 0))

    def test_poisson_recovers_coefficients(self):
        beta = np.array([0.2, 0.5, -0.3])
        mu = np.exp(self.X @ beta)
        y = self.rng.poisson(mu).astype(float)
        r = glm(self.X, y, family="poisson")
        np.testing.assert_allclose(r.coef, beta, atol=0.1)

    def test_gaussian_glm_equals_ols(self):
        y = self.X @ np.array([1.0, 2.0, -1.0]) + self.rng.normal(0, 1, self.n)
        r = glm(self.X, y, family="gaussian")
        ols = np.linalg.lstsq(self.X, y, rcond=None)[0]
        np.testing.assert_allclose(r.coef, ols, atol=1e-6)

    def test_offset_shifts_poisson_rate(self):
        # with a log-exposure offset the rate per unit exposure is recovered
        beta = np.array([0.1, 0.4])
        x = np.column_stack([np.ones(self.n), self.rng.normal(0, 1, self.n)])
        exposure = self.rng.uniform(1, 5, self.n)
        mu = exposure * np.exp(x @ beta)
        y = self.rng.poisson(mu).astype(float)
        r = glm(x, y, family="poisson", offset=np.log(exposure))
        np.testing.assert_allclose(r.coef, beta, atol=0.1)

    def test_probit_and_cloglog_links_fit(self):
        beta = np.array([0.2, 1.0, -0.5])
        p = 1.0 / (1.0 + np.exp(-self.X @ beta))
        y = (self.rng.rand(self.n) < p).astype(float)
        for link in ("probit", "cloglog"):
            r = glm(self.X, y, family="binomial", link=link)
            self.assertEqual(r.link, link)
            self.assertTrue(np.all(np.isfinite(r.coef)))

    def test_robust_se_option(self):
        y = self.X @ np.array([1.0, 2.0, -1.0]) + self.rng.normal(0, 1, self.n) * (0.5 + np.abs(self.X[:, 1]))
        model = glm(self.X, y, family="gaussian", robust=False)
        robust = glm(self.X, y, family="gaussian", robust=True)
        # robust SE on the heteroscedastic coefficient exceeds the model-based one
        self.assertGreater(robust.se[1], model.se[1])

    def test_negative_binomial_runs(self):
        mu = np.exp(self.X @ np.array([0.3, 0.4, -0.2]))
        y = self.rng.poisson(mu).astype(float)
        r = glm(self.X, y, family="negativebinomial")
        self.assertEqual(r.family, "negativebinomial")
        self.assertTrue(np.all(np.isfinite(r.coef)))

    def test_predict(self):
        beta = np.array([0.5, 1.5, -1.0])
        p = 1.0 / (1.0 + np.exp(-self.X @ beta))
        y = (self.rng.rand(self.n) < p).astype(float)
        r = glm(self.X, y, family="binomial")
        pred = r.predict(self.X[:5])
        self.assertEqual(pred.shape, (5,))
        self.assertTrue(np.all((pred > 0) & (pred < 1)))


class PenalizedTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(1)
        self.X = self.rng.normal(0, 1, (300, 10))
        self.beta = np.zeros(10)
        self.beta[:3] = [3.0, -2.0, 1.0]
        self.y = self.X @ self.beta + self.rng.normal(0, 0.5, 300)

    def test_ridge_shrinks_toward_zero(self):
        big = ridge_regression(self.X, self.y, alpha=1e-6).coef
        small = ridge_regression(self.X, self.y, alpha=100.0).coef
        self.assertLess(np.linalg.norm(small), np.linalg.norm(big))

    def test_ridge_recovers_when_lightly_penalized(self):
        r = ridge_regression(self.X, self.y, alpha=1e-3)
        np.testing.assert_allclose(r.coef[:3], self.beta[:3], atol=0.15)

    def test_lasso_selects_sparse_support(self):
        r = lasso(self.X, self.y, alpha=0.3)
        # only the 3 true predictors survive
        self.assertEqual(int(np.sum(np.abs(r.coef) > 1e-6)), 3)
        self.assertTrue(np.all(np.abs(r.coef[3:]) < 1e-6))

    def test_elastic_net_between_ridge_and_lasso(self):
        r = elastic_net(self.X, self.y, alpha=0.3, l1_ratio=0.5)
        nonzero = int(np.sum(np.abs(r.coef) > 1e-6))
        self.assertLessEqual(nonzero, 10)
        self.assertGreaterEqual(nonzero, 3)


class RobustQuantileTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(2)
        self.n = 1000
        self.X = np.column_stack([np.ones(self.n), self.rng.normal(0, 1, self.n)])
        self.beta = np.array([1.0, 2.0])

    def test_robust_ignores_outliers(self):
        y = self.X @ self.beta + self.rng.normal(0, 0.3, self.n)
        y[:80] += 50.0  # gross contamination
        for method in ("huber", "tukey"):
            r = robust_regression(self.X, y, method=method)
            np.testing.assert_allclose(r.coef, self.beta, atol=0.2)
        ols = np.linalg.lstsq(self.X, y, rcond=None)[0]
        self.assertGreater(abs(ols[0] - 1.0), 1.0)  # OLS badly biased

    def test_quantile_median_near_ols_for_symmetric(self):
        y = self.X @ self.beta + self.rng.normal(0, 1, self.n)
        qm = quantile_regression(self.X, y, 0.5)
        ols = np.linalg.lstsq(self.X, y, rcond=None)[0]
        np.testing.assert_allclose(qm.coef, ols, atol=0.15)

    def test_quantile_levels_ordered(self):
        y = self.X @ self.beta + self.rng.normal(0, 1, self.n)
        q10 = quantile_regression(self.X, y, 0.1).coef[0]
        q90 = quantile_regression(self.X, y, 0.9).coef[0]
        self.assertLess(q10, q90)  # lower quantile intercept below upper

    def test_quantile_invalid_tau(self):
        with self.assertRaises(ValueError):
            quantile_regression(self.X, np.zeros(self.n), 1.5)


if __name__ == "__main__":
    unittest.main()
