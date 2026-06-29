"""Tests for mixle.ppl linear regression (Field + linear predictor)."""

import unittest

import numpy as np

from mixle.ppl import Bernoulli, Field, Group, Normal, Poisson, free


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
            list(self.y), given={"x": list(self.x), "z": list(self.z)}
        )
        c = m.params
        self.assertAlmostEqual(c["x"]["mean"], 2.0, delta=0.05)
        self.assertAlmostEqual(c["z"]["mean"], -1.5, delta=0.05)
        self.assertAlmostEqual(c["intercept"]["mean"], 0.7, delta=0.05)
        self.assertAlmostEqual(m.result.sigma, 0.5, delta=0.05)

    def test_bayesian_regression_posterior_and_predict(self):
        a, b = Normal(0, 10), Normal(0, 10)
        m = Normal(a * Field("x") + b, free).fit(
            list(self.y - (-1.5 * self.z)), given={"x": list(self.x)}
        )  # drop z term
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
            list(2.0 * self.x + 0.7 + np.random.RandomState(1).normal(0, 0.5, self.N)), given={"x": list(self.x)}
        )
        self.assertAlmostEqual(m.result.sigma, 0.5, delta=1e-9)  # fixed, not estimated
        self.assertAlmostEqual(m.params["x"]["mean"], 2.0, delta=0.05)


class GLMTestCase(unittest.TestCase):
    def test_logistic_regression(self):
        rng = np.random.RandomState(0)
        N = 6000
        x, z = rng.normal(0, 1, N), rng.normal(0, 1, N)
        p = 1.0 / (1.0 + np.exp(-(2.0 * x - 1.0 * z + 0.5)))
        y = (rng.random(N) < p).astype(float)
        m = Bernoulli(free * Field("x") + free * Field("z") + free).fit(list(y), given={"x": list(x), "z": list(z)})
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
        self.assertGreater(float(m.result.predict({"x": [0.0]})[0]), 0.0)  # a rate


class MixedEffectsTestCase(unittest.TestCase):
    def test_random_intercept_lmm(self):
        rng = np.random.RandomState(0)
        G, n_per = 40, 30
        u = rng.normal(0, 1.5, G)
        ys, xs, subj = [], [], []
        for gi in range(G):
            x = rng.normal(0, 1, n_per)
            y = 1.0 + 2.0 * x + u[gi] + rng.normal(0, 0.7, n_per)
            ys += list(y)
            xs += list(x)
            subj += [gi] * n_per
        m = Normal(free * Field("x") + free + Group("subject"), free).fit(ys, given={"x": xs, "subject": subj})
        r = m.result
        self.assertAlmostEqual(r.coefficients["x"]["mean"], 2.0, delta=0.1)  # fixed slope
        self.assertAlmostEqual(r.tau, 1.5, delta=0.4)  # random-intercept sd
        self.assertAlmostEqual(r.sigma, 0.7, delta=0.1)  # residual sd
        ge = np.array([r.group_effects[i] for i in range(G)])
        self.assertGreater(np.corrcoef(ge, u)[0, 1], 0.95)  # recovers BLUPs
        # intercept absorbs the sample mean of the random effects
        self.assertAlmostEqual(r.coefficients["intercept"]["mean"] + ge.mean() - u.mean(), 1.0, delta=0.15)

    def test_poisson_glmm(self):
        # non-Normal mixed model: log-rate = b0 + b1 x + u_g, u_g ~ N(0, tau^2), via PQL
        rng = np.random.RandomState(0)
        G, n_per = 40, 40
        b0, b1, tau = 0.2, 0.5, 0.6
        u = rng.normal(0, tau, G)
        ys, xs, subj = [], [], []
        for gi in range(G):
            x = rng.normal(0, 1, n_per)
            ys += list(rng.poisson(np.exp(b0 + b1 * x + u[gi])))
            xs += list(x)
            subj += [gi] * n_per
        m = Poisson(free * Field("x") + free + Group("g")).fit(ys, given={"x": xs, "g": subj})
        r = m.result
        self.assertEqual(r.link, "log")
        self.assertAlmostEqual(r.coefficients["x"]["mean"], b1, delta=0.15)
        self.assertAlmostEqual(r.tau, tau, delta=0.2)
        ge = np.array([r.group_effects[i] for i in range(G)])
        self.assertGreater(np.corrcoef(ge, u)[0, 1], 0.95)

    def test_bernoulli_glmm(self):
        rng = np.random.RandomState(0)
        G, n_per = 60, 60
        b0, b1, tau = -0.3, 0.8, 0.7
        u = rng.normal(0, tau, G)
        ys, xs, subj = [], [], []
        for gi in range(G):
            x = rng.normal(0, 1, n_per)
            p = 1.0 / (1.0 + np.exp(-(b0 + b1 * x + u[gi])))
            ys += list((rng.random(n_per) < p).astype(float))
            xs += list(x)
            subj += [gi] * n_per
        m = Bernoulli(free * Field("x") + free + Group("g")).fit(ys, given={"x": xs, "g": subj})
        r = m.result
        self.assertEqual(r.link, "logit")
        self.assertAlmostEqual(r.coefficients["x"]["mean"], b1, delta=0.2)
        self.assertAlmostEqual(r.tau, tau, delta=0.25)
        ge = np.array([r.group_effects[i] for i in range(G)])
        self.assertGreater(np.corrcoef(ge, u)[0, 1], 0.85)

    def test_random_intercept_only_no_fixed_covariate(self):
        # intercept-only fixed part (no fixed covariate) used to fail to size the design matrix.
        rng = np.random.RandomState(0)
        G, n_per = 40, 30
        u = rng.normal(0, 1.5, G)
        ys, subj = [], []
        for gi in range(G):
            ys += list(3.0 + u[gi] + rng.normal(0, 0.6, n_per))
            subj += [gi] * n_per
        m = Normal(Group("subject") + free, free).fit(ys, given={"subject": subj})
        r = m.result
        ge = np.array([r.group_effects[i] for i in range(G)])
        # fixed intercept absorbs the grand mean; random effects are mean-zero BLUPs around it
        self.assertAlmostEqual(r.coefficients["intercept"]["mean"] + ge.mean(), 3.0 + u.mean(), delta=0.2)
        self.assertGreater(np.corrcoef(ge, u)[0, 1], 0.95)
        self.assertAlmostEqual(r.sigma, 0.6, delta=0.1)

    def test_random_slope_only_no_fixed_covariate(self):
        rng = np.random.RandomState(1)
        G, n_per = 60, 40
        u0 = rng.normal(0, 1.0, G)
        u1 = rng.normal(0, 0.8, G)
        ys, xs, subj = [], [], []
        for gi in range(G):
            x = rng.normal(0, 1, n_per)
            ys += list(2.0 + u0[gi] + (1.5 + u1[gi]) * x + rng.normal(0, 0.5, n_per))
            xs += list(x)
            subj += [gi] * n_per
        # fixed part is intercept-only; the slope lives entirely in the random effect
        m = Normal(Group("subject", slopes=["x"]) + free, free).fit(ys, given={"x": xs, "subject": subj})
        r = m.result
        bslope = np.array([r.group_effects_full[i][1] for i in range(G)])
        self.assertGreater(np.corrcoef(bslope, u1)[0, 1], 0.9)  # per-group slope deviations recovered
        self.assertAlmostEqual(r.sigma, 0.5, delta=0.1)

    def test_random_slopes_lmm(self):
        rng = np.random.RandomState(0)
        G, n_per = 60, 40
        u0 = rng.normal(0, 1.0, G)
        u1 = rng.normal(0, 0.8, G)
        ys, xs, subj = [], [], []
        for gi in range(G):
            x = rng.normal(0, 1, n_per)
            y = 1.0 + 2.0 * x + u0[gi] + u1[gi] * x + rng.normal(0, 0.5, n_per)
            ys += list(y)
            xs += list(x)
            subj += [gi] * n_per
        m = Normal(free * Field("x") + free + Group("subject", slopes=["x"]), free).fit(
            ys, given={"x": xs, "subject": subj}
        )
        r = m.result
        sds = np.sqrt(np.diag(r.random_cov))
        self.assertAlmostEqual(sds[0], 1.0, delta=0.25)  # random intercept sd
        self.assertAlmostEqual(sds[1], 0.8, delta=0.25)  # random slope sd
        self.assertAlmostEqual(r.sigma, 0.5, delta=0.1)
        bslope = np.array([r.group_effects_full[i][1] for i in range(G)])
        self.assertGreater(np.corrcoef(bslope, u1)[0, 1], 0.95)


class QuantileRegressionTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.n = 3000
        self.x = rng.uniform(0, 5, self.n)
        # heteroskedastic: noise scale grows with x, so the quantiles fan out
        self.y = 2.0 + 1.5 * self.x + rng.normal(0, 0.4 + 0.6 * self.x, self.n)

    def test_quantiles_fan_out_and_recover_median(self):
        fits = {
            tau: Normal(free * Field("x") + free, free).fit(list(self.y), given={"x": list(self.x)}, quantile=tau)
            for tau in (0.1, 0.5, 0.9)
        }
        slopes = {tau: fits[tau].result.coefficients["x"]["mean"] for tau in fits}
        self.assertAlmostEqual(slopes[0.5], 1.5, delta=0.2)  # median slope ~ the mean slope
        self.assertLess(slopes[0.1], slopes[0.5])  # spread grows with x -> steeper upper quantile
        self.assertLess(slopes[0.5], slopes[0.9])
        self.assertEqual(fits[0.9].result.quantile, 0.9)

    def test_band_coverage(self):
        lo_fit = Normal(free * Field("x") + free, free).fit(list(self.y), given={"x": list(self.x)}, quantile=0.1)
        hi_fit = Normal(free * Field("x") + free, free).fit(list(self.y), given={"x": list(self.x)}, quantile=0.9)
        lo = lo_fit.result.predict({"x": list(self.x)})
        hi = hi_fit.result.predict({"x": list(self.x)})
        cov = ((self.y >= lo) & (self.y <= hi)).mean()
        self.assertAlmostEqual(cov, 0.8, delta=0.04)

    def test_invalid_quantile_raises(self):
        with self.assertRaises(ValueError):
            Normal(free * Field("x") + free, free).fit(list(self.y), given={"x": list(self.x)}, quantile=1.5)


class RegularizedRegressionTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        n, self.p, self.k = 200, 12, 3
        self.X = rng.normal(0, 1, (n, self.p))
        self.beta = np.zeros(self.p)
        self.beta[: self.k] = np.array([3.0, -2.5, 2.0])
        self.y = self.X @ self.beta + rng.normal(0, 0.5, n)
        self.given = {f"x{j}": list(self.X[:, j]) for j in range(self.p)}

    def _build(self, coef):
        t = coef(0) * Field("x0")
        for j in range(1, self.p):
            t = t + coef(j) * Field(f"x{j}")
        return t + coef("intercept")

    def _coefs(self, m):
        return np.array([m.result.coefficients[f"x{j}"]["mean"] for j in range(self.p)])

    def test_free_recovers_ols(self):
        m = Normal(self._build(lambda j: free), free).fit(list(self.y), given=self.given)
        np.testing.assert_allclose(self._coefs(m)[: self.k], self.beta[: self.k], atol=0.15)

    def test_lasso_selects_sparse_support(self):
        from mixle.ppl import Laplace

        m = Normal(self._build(lambda j: Laplace(0, 0.3)), free).fit(list(self.y), given=self.given)
        coefs = self._coefs(m)
        nonzero = np.flatnonzero(np.abs(coefs) > 1e-6)
        self.assertTrue(set(range(self.k)).issubset(set(nonzero)))  # keeps the true features
        self.assertLess(len(nonzero), self.p)  # but zeros some irrelevant ones (sparsity)

    def test_ridge_shrinks_without_zeroing(self):
        m = Normal(self._build(lambda j: Normal(0, 0.4)), free).fit(list(self.y), given=self.given)
        coefs = self._coefs(m)
        self.assertEqual(np.sum(np.abs(coefs) < 1e-6), 0)  # ridge keeps all nonzero
        ols = Normal(self._build(lambda j: free), free).fit(list(self.y), given=self.given)
        self.assertLess(np.abs(coefs).max(), np.abs(self._coefs(ols)).max() + 1e-9)  # shrunk

    def test_elastic_net_groups_correlated_features(self):
        from mixle.ppl import Laplace

        rng = np.random.RandomState(0)
        nn, p = 200, 8
        z = rng.normal(0, 1, (nn, 1))
        Xc = np.concatenate([z + 0.05 * rng.normal(0, 1, (nn, 3)), rng.normal(0, 1, (nn, p - 3))], axis=1)
        y = Xc @ np.r_[np.full(3, 2.0), np.zeros(p - 3)] + rng.normal(0, 0.5, nn)
        given = {f"x{j}": list(Xc[:, j]) for j in range(p)}

        def build(coef):
            t = coef(0) * Field("x0")
            for j in range(1, p):
                t = t + coef(j) * Field(f"x{j}")
            return t + coef("intercept")

        def coefs(m):
            return np.array([m.result.coefficients[f"x{j}"]["mean"] for j in range(p)])

        lasso = Normal(build(lambda j: Laplace(0, 0.1)), free).fit(list(y), given=given)
        enet = Normal(build(lambda j: Laplace(0, 0.1)), free).fit(list(y), given=given, l2=2.0)
        # the global L2 spreads weight across the correlated group instead of concentrating on one
        self.assertGreater(np.abs(coefs(enet)[:3]).min(), np.abs(coefs(lasso)[:3]).min())


if __name__ == "__main__":
    unittest.main()
