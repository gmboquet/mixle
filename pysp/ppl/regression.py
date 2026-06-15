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


def regression_fit(rv: RandomVariable, data, *, given=None, max_iter: int = 100,
                   tol: float = 1e-9, **_) -> RandomVariable:
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
