"""Tests for mixle.ppl linear regression (Field + linear predictor)."""

import unittest

import numpy as np

from mixle.ppl import Bernoulli, Field, Group, Normal, Poisson, compare, free


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


class ConditionalFitModelSelectionTestCase(unittest.TestCase):
    """A fitted regression / mixed-effects model must report a real log-likelihood.

    Before D-0190, `aic`/`bic`/`log_likelihood`/`plugin_log_likelihood`/`compare` all crashed on any
    model containing a `Field(...)` with a leaked `AttributeError: 'NoneType' object has no attribute
    'dist_to_encoder'`: `lower()` returned None for a conditional fit and every caller but `.params`
    dereferenced it. The EM never computed a likelihood at all, so there was nothing to report.
    """

    def _radonish(self, seed=0, n=240, groups=12):
        rng = np.random.default_rng(seed)
        g = np.array([f"g{i % groups}" for i in range(n)])
        effect = {f"g{i}": rng.normal(0.0, 0.7) for i in range(groups)}
        x = rng.normal(0.0, 1.0, n)
        y = 1.5 + 2.0 * x + np.array([effect[k] for k in g]) + rng.normal(0.0, 0.5, n)
        return list(y), list(x), list(g)

    @staticmethod
    def _dense_marginal_loglik(y, X, Z, beta, Sigma, sigma2, g):
        """Reference marginal log-likelihood built the DENSE way: form V = Z Sigma Z' + sigma^2 I
        over the whole sample and evaluate the multivariate normal directly. This shares no code
        path with the implementation's per-group Woodbury / determinant-lemma route."""
        y = np.asarray(y, float)
        n = y.size
        V = np.eye(n) * sigma2
        for a in range(n):
            for b in range(n):
                if g[a] == g[b]:
                    V[a, b] += Z[a] @ Sigma @ Z[b]
        r = y - X @ beta
        # Cholesky rather than slogdet: V is positive definite by construction, and this gives both
        # the log-determinant and the quadratic form stably (slogdet on the dense V underflows here).
        L = np.linalg.cholesky(V)
        logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
        quad = float(np.linalg.solve(L, r) @ np.linalg.solve(L, r))
        return float(-0.5 * (n * np.log(2 * np.pi) + logdet + quad))

    def test_mixed_model_loglik_matches_a_dense_reference(self):
        y, x, g = self._radonish()
        m = Normal(free * Field("x") + free + Group("gid"), free).fit(y, given={"x": x, "gid": g})
        r = m.result
        n = len(y)
        X = (
            np.column_stack([np.asarray(x), np.ones(n)])
            if r.names[0] == "x"
            else np.column_stack([np.ones(n), np.asarray(x)])
        )
        Z = np.ones((n, 1))
        reference = self._dense_marginal_loglik(y, X, Z, r.beta, r.random_cov, r.sigma**2, g)
        self.assertAlmostEqual(m.log_likelihood(y), reference, places=6)
        self.assertEqual(r.nobs, n)
        # fixed effects (2) + Sigma upper triangle (1) + residual variance (1)
        self.assertEqual(r.n_params, 4)

    def test_mixed_model_aic_bic_are_consistent_with_the_loglik(self):
        y, x, g = self._radonish()
        m = Normal(free * Field("x") + free + Group("gid"), free).fit(y, given={"x": x, "gid": g})
        ll, k, n = m.log_likelihood(y), m.result.n_params, len(y)
        self.assertAlmostEqual(m.aic(y), 2 * k - 2 * ll, places=9)
        self.assertAlmostEqual(m.bic(y), k * np.log(n) - 2 * ll, places=9)

    def test_plain_gaussian_regression_loglik_matches_closed_form(self):
        y, x, _g = self._radonish()
        m = Normal(free * Field("x") + free, free).fit(y, given={"x": x})
        yv = np.asarray(y)
        fitted = (
            m.result.beta @ np.column_stack([np.asarray(x), np.ones(yv.size)]).T if m.result.names[0] == "x" else None
        )
        self.assertIsNotNone(fitted)
        resid = yv - fitted
        sigma = m.result.sigma
        expected = float(-0.5 * yv.size * (np.log(2 * np.pi) + 2 * np.log(sigma)) - resid @ resid / (2 * sigma**2))
        self.assertAlmostEqual(m.log_likelihood(y), expected, places=6)

    def test_compare_ranks_conditional_fits_and_prefers_the_grouped_model(self):
        y, x, g = self._radonish()
        mixed = Normal(free * Field("x") + free + Group("gid"), free).fit(y, given={"x": x, "gid": g})
        plain = Normal(free * Field("x") + free, free).fit(y, given={"x": x})
        rows = compare([mixed, plain], y)
        self.assertEqual(len(rows), 2)
        # the data really do have group structure, so the mixed model must win on AIC
        self.assertLess(rows[0]["aic"], rows[1]["aic"])
        self.assertEqual(rows[0]["model"], "LMMResult")

    def test_compare_refuses_a_single_model_instead_of_hanging(self):
        """compare(model, data) used to iterate the RandomVariable through the legacy __getitem__
        protocol -- which never terminates -- and hung indefinitely instead of raising."""
        y, x, _g = self._radonish()
        m = Normal(free * Field("x") + free, free).fit(y, given={"x": x})
        with self.assertRaises(TypeError) as ctx:
            compare(m, y)
        self.assertIn("LIST", str(ctx.exception))

    def test_unavailable_quantities_refuse_clearly_and_do_not_recommend_broken_calls(self):
        y, x, g = self._radonish()
        m = Normal(free * Field("x") + free + Group("gid"), free).fit(y, given={"x": x, "gid": g})
        # a mixed model's marginal likelihood factors per GROUP, so there is no per-observation vector
        with self.assertRaises(NotImplementedError) as plug:
            m.plugin_log_likelihood(y)
        self.assertIn("per GROUP", str(plug.exception))
        for name in ("waic", "loo"):
            with self.assertRaises(NotImplementedError) as ctx:
                getattr(m, name)(y)
            message = str(ctx.exception)
            # the old message recommended plugin_log_likelihood/AIC/BIC, all of which then crashed
            self.assertNotIn("plugin_log_likelihood(data), AIC, or BIC instead", message)
            self.assertIn("log_likelihood(data)", message)

    def test_lowering_a_conditional_fit_refuses_instead_of_returning_none(self):
        from mixle.ppl.core import lower

        y, x, g = self._radonish()
        m = Normal(free * Field("x") + free + Group("gid"), free).fit(y, given={"x": x, "gid": g})
        with self.assertRaises(NotImplementedError):
            lower(m, target="dist")

    def test_params_still_reports_coefficients_for_a_conditional_fit(self):
        """`.params` was the one caller that handled lower()'s None, so it must keep working now
        that the sentinel is gone."""
        y, x, g = self._radonish()
        m = Normal(free * Field("x") + free + Group("gid"), free).fit(y, given={"x": x, "gid": g})
        self.assertIn("x", m.params)
        self.assertAlmostEqual(m.params["x"]["mean"], 2.0, delta=0.15)

    def test_log_likelihood_refuses_a_different_sized_dataset(self):
        y, x, g = self._radonish()
        m = Normal(free * Field("x") + free + Group("gid"), free).fit(y, given={"x": x, "gid": g})
        with self.assertRaises(ValueError) as ctx:
            m.log_likelihood(y[:50])
        self.assertIn("observations", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
