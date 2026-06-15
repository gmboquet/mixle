"""Linear regression for pysp.ppl.

A model ``Normal(a*Field("x") + b, sigma)`` is a regression: the mean is a linear predictor
over covariates. Coefficients may be Normal priors (Bayesian / ridge — closed-form Gaussian
posterior) or ``free`` (OLS); ``sigma`` may be a constant or ``free`` (estimated from
residuals). Fit with ``.fit(y, given={"x": xs})``.
"""
from __future__ import annotations

import numpy as np

from pysp.ppl.core import RandomVariable, Field, _LinearPredictor, free as FREE


class RegressionResult:
    """Posterior over regression coefficients + residual scale, with prediction."""

    def __init__(self, names, idx_of, beta, cov, sigma, columns):
        self.names = names                       # column names (covariates + 'intercept')
        self.beta = beta                         # posterior mean coefficients
        self.cov = cov                           # posterior covariance
        self.sigma = float(sigma)
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
        """Predict the response at covariates ``given`` (dict of arrays). Returns the mean
        prediction; with ``n`` set, returns ``n`` posterior-predictive draws per row."""
        X, offset = _design(self._columns, given)
        mean = offset + X @ self.beta
        if n is None:
            return mean
        rng = rng or np.random.RandomState()
        out = np.empty((n, mean.size))
        for k in range(n):
            beta_k = rng.multivariate_normal(self.beta, self.cov)
            out[k] = offset + X @ beta_k + rng.normal(0.0, self.sigma, mean.size)
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


def regression_fit(rv: RandomVariable, data, *, given=None, max_iter: int = 100,
                   tol: float = 1e-9, **_) -> RandomVariable:
    linpred, scale = rv._args
    given = given or {}
    y = np.asarray(data, dtype=float).reshape(-1)
    columns = _columns_of(linpred)
    est, _fixed = columns
    X, offset = _design(columns, given)
    yv = y - offset
    N, p = X.shape

    # priors: Gaussian per coef (free -> flat, precision 0)
    m0 = np.zeros(p)
    p0 = np.zeros(p)
    idx_of, names = {}, []
    for i, (coef, field) in enumerate(est):
        names.append(field.name if field is not None else "intercept")
        idx_of[id(coef)] = i
        if isinstance(coef, RandomVariable) and coef._family.name == "Normal":
            m0[i] = float(coef._args[0])
            p0[i] = 1.0 / float(coef._args[1]) ** 2

    sigma_fixed = not (scale is FREE or isinstance(scale, RandomVariable))
    sigma2 = float(scale) ** 2 if sigma_fixed else max(float(np.var(yv)), 1e-8)

    P0 = np.diag(p0)
    XtX, Xty = X.T @ X, X.T @ yv
    beta, cov = np.zeros(p), np.eye(p)
    for _ in range(max_iter):
        post_prec = P0 + XtX / sigma2
        cov = np.linalg.inv(post_prec)
        beta = cov @ (P0 @ m0 + Xty / sigma2)
        if sigma_fixed:
            break
        resid = yv - X @ beta
        new_sigma2 = max(float(resid @ resid / N), 1e-8)
        if abs(new_sigma2 - sigma2) < tol:
            sigma2 = new_sigma2
            break
        sigma2 = new_sigma2

    result = RegressionResult(names, idx_of, beta, cov, np.sqrt(sigma2), columns)
    return RandomVariable._bound(None, name=rv._name, result=result)
