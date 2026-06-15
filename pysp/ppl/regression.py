"""Regression / GLMs for pysp.ppl.

A linear predictor in a parameter slot makes a model a regression; the outer family sets the
link:

    Normal(a*Field("x") + b, sigma)   identity link  -> linear regression
    Bernoulli(a*Field("x") + b)       logit link     -> logistic regression
    Poisson(a*Field("x") + b)         log link       -> Poisson regression

Coefficients may be Normal priors (Bayesian / ridge — MAP) or ``free`` (MLE). Fitting is
IRLS (Fisher scoring) with optional Gaussian-prior penalty and a Laplace coefficient
covariance. Fit with ``.fit(y, given={"x": xs})``.
"""
from __future__ import annotations

import numpy as np

from pysp.ppl.core import RandomVariable, Field, _LinearPredictor, free as FREE

# family -> canonical link name
_LINK = {"Normal": "identity", "Bernoulli": "logit", "Poisson": "log"}


def _link_inv(link, eta):
    if link == "identity":
        return eta
    if link == "logit":
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
    if link == "log":
        return np.exp(np.clip(eta, -30, 30))
    raise ValueError(link)


def _irls_weight(link, mu):
    if link == "identity":
        return np.ones_like(mu)
    if link == "logit":
        return np.clip(mu * (1.0 - mu), 1e-9, None)
    if link == "log":
        return np.clip(mu, 1e-9, None)
    raise ValueError(link)


class RegressionResult:
    """Posterior over regression coefficients + residual scale, with prediction."""

    def __init__(self, names, idx_of, beta, cov, sigma, columns, link="identity"):
        self.names = names                       # column names (covariates + 'intercept')
        self.beta = beta                         # posterior mean coefficients
        self.cov = cov                           # posterior covariance
        self.sigma = float(sigma)
        self.link = link
        self._idx_of = idx_of                    # id(coef handle) -> column index
        self._columns = columns                  # list of (kind, payload) for predict
        self.coefficients = {names[i]: {"mean": float(beta[i]), "sd": float(np.sqrt(cov[i, i]))}
                             for i in range(len(names))}
        self.acceptance_rate = None
        self.predictive = None

    def _resolve(self, param):
        if isinstance(param, str):
            return self.names.index(param)
        if isinstance(param, (int, np.integer)):
            return int(param)
        return self._idx_of[id(param)]

    def samples(self, param=None, n: int = 4000, rng=None):
        rng = rng or np.random.RandomState()
        if param is None:
            return rng.multivariate_normal(self.beta, self.cov, n)
        i = self._resolve(param)
        return rng.normal(self.beta[i], np.sqrt(self.cov[i, i]), n)

    def predict(self, given, *, n=None, rng=None):
        """Predict the response mean at covariates ``given`` (dict of arrays): the fitted
        value through the link (probabilities for logistic, rates for Poisson, mean for
        linear). With ``n``, returns ``n`` posterior-predictive draws of the linear-response
        mean (integrating coefficient uncertainty)."""
        X, offset = _design(self._columns, given)
        eta = offset + X @ self.beta
        if n is None:
            return _link_inv(self.link, eta)
        rng = rng or np.random.RandomState()
        out = np.empty((n, eta.size))
        for k in range(n):
            beta_k = rng.multivariate_normal(self.beta, self.cov)
            out[k] = _link_inv(self.link, offset + X @ beta_k)
        return out

    def summary(self):
        return {"coefficients": self.coefficients, "sigma": self.sigma}


def _columns_of(linpred: _LinearPredictor):
    """Return (est_columns, fixed_columns): estimated coefs (RV prior / free) vs constants."""
    cols = list(linpred.terms)                                  # (coef, Field)
    if linpred.intercept is not None:
        cols.append((linpred.intercept, None))                 # None field -> intercept (ones)
    est, fixed = [], []
    for coef, field in cols:
        if isinstance(coef, RandomVariable) or coef is FREE:
            est.append((coef, field))
        else:
            fixed.append((float(coef), field))
    return est, fixed


def _design(columns, given):
    """Build the design matrix for the estimated columns and the fixed offset."""
    est, fixed = columns
    n = None
    for _, field in est + fixed:
        if field is not None:
            n = len(np.asarray(given[field.name]).reshape(-1)); break
    if n is None:
        raise ValueError("need at least one covariate to size the design matrix.")
    mat = []
    for _, field in est:
        mat.append(np.ones(n) if field is None else np.asarray(given[field.name], float).reshape(-1))
    X = np.column_stack(mat) if mat else np.zeros((n, 0))
    offset = np.zeros(n)
    for c, field in fixed:
        offset += c * (np.ones(n) if field is None else np.asarray(given[field.name], float).reshape(-1))
    return X, offset


class LMMResult:
    """Linear mixed model: fixed-effect coefficients + variance components + group effects."""

    def __init__(self, names, beta, cov, Sigma, sigma, b, group_levels, re_names):
        self.names = names
        self.beta = beta
        self.cov = cov
        self.random_cov = np.asarray(Sigma)        # random-effects covariance (q x q)
        self.random_names = re_names               # ['intercept', slope names...]
        self.tau = float(np.sqrt(Sigma[0, 0]))     # random-intercept sd (back-compat)
        self.sigma = float(sigma)                  # residual sd
        # per-group effects: intercept for back-compat, full vector under group_effects_full
        self.group_effects = {lv: float(b[i, 0]) for i, lv in enumerate(group_levels)}
        self.group_effects_full = {lv: b[i] for i, lv in enumerate(group_levels)}
        self.coefficients = {names[i]: {"mean": float(beta[i]), "sd": float(np.sqrt(cov[i, i]))}
                             for i in range(len(names))}
        self.acceptance_rate = None
        self.predictive = None

    def summary(self):
        return {"coefficients": self.coefficients, "random_cov": self.random_cov,
                "sigma": self.sigma, "n_groups": len(self.group_effects)}


def _lmm_fit(rv, y, given, linpred, max_iter, tol):
    """Linear mixed model with one grouping factor (random intercept + optional random
    slopes): y = X beta + Z b_g + eps, b_g ~ N(0, Sigma), eps ~ N(0, sigma^2). EM."""
    if len(linpred.groups) != 1:
        raise NotImplementedError("exactly one grouping factor is supported.")
    gname, slopes = linpred.groups[0]
    if gname not in given:
        raise ValueError(f"group column {gname!r} not in given=.")
    columns = _columns_of(linpred)
    est, _fixed = columns
    X, offset = _design(columns, given) if (est or _fixed) else (np.zeros((y.size, 0)), np.zeros(y.size))
    names = [f.name if f is not None else "intercept" for _, f in est]
    if not names:
        X = np.ones((y.size, 1)); names = ["intercept"]
    N, p = X.shape

    # random-effects design Z: intercept + slope columns
    re_names = ["intercept"] + list(slopes)
    zcols = [np.ones(N)] + [np.asarray(given[s], dtype=float).reshape(-1) for s in slopes]
    Z = np.column_stack(zcols)
    q = Z.shape[1]
    levels, g = np.unique(np.asarray(given[gname]), return_inverse=True)
    G = levels.size
    yv = y - offset

    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    var0 = max(float(np.var(yv)), 1e-3)
    Sigma = np.eye(q) * (0.5 * var0)
    sigma2 = 0.5 * var0
    b = np.zeros((G, q))
    # precompute per-group Z slices
    groups = [np.where(g == gi)[0] for gi in range(G)]
    for _ in range(max_iter):
        resid = yv - X @ beta
        Sinv = np.linalg.inv(Sigma)
        SS = np.zeros((q, q))
        err2 = 0.0
        trace_term = 0.0
        for gi, idx in enumerate(groups):
            Zg, rg = Z[idx], resid[idx]
            cov_g = np.linalg.inv(Sinv + Zg.T @ Zg / sigma2)     # E-step posterior of b_g
            b_g = cov_g @ (Zg.T @ rg / sigma2)
            b[gi] = b_g
            SS += np.outer(b_g, b_g) + cov_g
            pred = Zg @ b_g
            err2 += float((rg - pred) @ (rg - pred))
            trace_term += float(np.trace(Zg @ cov_g @ Zg.T))
        Sigma = SS / G                                            # M-step
        sigma2 = max((err2 + trace_term) / N, 1e-8)
        Zb = np.einsum("nq,nq->n", Z, b[g])
        beta_new = np.linalg.lstsq(X, yv - Zb, rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    cov = np.linalg.inv(X.T @ X / sigma2) if p else np.zeros((0, 0))
    result = LMMResult(names, beta, cov, Sigma, np.sqrt(sigma2), b, list(levels), re_names)
    return RandomVariable._bound(None, name=rv._name, result=result)


def regression_fit(rv: RandomVariable, data, *, given=None, max_iter: int = 100,
                   tol: float = 1e-9, **_) -> RandomVariable:
    linpred0 = next((a for a in rv._args if isinstance(a, _LinearPredictor)), None)
    if linpred0 is not None and linpred0.groups:           # mixed-effects model
        if rv._family.name != "Normal":
            raise NotImplementedError("mixed-effects models require a Normal response.")
        return _lmm_fit(rv, np.asarray(data, float).reshape(-1), given or {}, linpred0,
                        max_iter, tol)
    fam = rv._family.name
    link = _LINK.get(fam)
    if link is None:
        raise NotImplementedError(f"regression for family {fam} is not supported "
                                  f"(have {sorted(_LINK)}).")
    linpred = next(a for a in rv._args if isinstance(a, _LinearPredictor))
    scale = rv._args[1] if fam == "Normal" else None
    given = given or {}
    y = np.asarray(data, dtype=float).reshape(-1)
    columns = _columns_of(linpred)
    est, _fixed = columns
    X, offset = _design(columns, given)
    N, p = X.shape

    # Gaussian priors per coef (free -> flat, precision 0)
    m0, p0 = np.zeros(p), np.zeros(p)
    idx_of, names = {}, []
    for i, (coef, field) in enumerate(est):
        names.append(field.name if field is not None else "intercept")
        idx_of[id(coef)] = i
        if isinstance(coef, RandomVariable) and coef._family.name == "Normal":
            m0[i] = float(coef._args[0])
            p0[i] = 1.0 / float(coef._args[1]) ** 2
    P0 = np.diag(p0)

    # IRLS / Fisher scoring (one step is OLS for the Gaussian identity link)
    beta = np.zeros(p)
    cov = np.eye(p)
    for _ in range(max_iter):
        eta = offset + X @ beta
        mu = _link_inv(link, eta)
        W = _irls_weight(link, mu)
        z = (eta - offset) + (y - mu) / W                 # working response (predictor space)
        WX = X * W[:, None]
        A = X.T @ WX + P0
        cov = np.linalg.inv(A)
        new_beta = cov @ (X.T @ (W * z) + P0 @ m0)
        if np.max(np.abs(new_beta - beta)) < tol:
            beta = new_beta
            break
        beta = new_beta

    if fam == "Normal":                                    # residual scale
        sigma_fixed = not (scale is FREE or isinstance(scale, RandomVariable))
        if sigma_fixed:
            sigma = float(scale)
        else:
            resid = y - (offset + X @ beta)
            sigma = float(np.sqrt(max(resid @ resid / N, 1e-8)))
        cov = cov * (sigma ** 2)                            # OLS coef cov scales with sigma^2
    else:
        sigma = float("nan")

    result = RegressionResult(names, idx_of, beta, cov, sigma, columns, link=link)
    return RandomVariable._bound(None, name=rv._name, result=result)
