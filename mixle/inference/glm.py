"""Generalized linear models and penalized / robust / quantile regression on plain arrays.

A array-level regression toolkit (operating on a design matrix ``X`` and response ``y``, independent of
the PPL DSL in :mod:`mixle.ppl.regression`):

  * :func:`glm` -- exponential-family GLMs by iteratively reweighted least squares, with explicit
    family/link objects (Gaussian, Binomial, Poisson, Gamma, inverse-Gaussian, negative-binomial),
    offsets, prior weights, and optional sandwich (robust) standard errors.
  * :func:`ridge_regression`, :func:`elastic_net` (and :func:`lasso`) -- L2 / L1 / mixed penalised
    linear regression; the elastic net is solved by coordinate descent.
  * :func:`robust_regression` -- Huber / Tukey M-estimation, down-weighting outliers via IRLS on a
    robust scale.
  * :func:`quantile_regression` -- the conditional ``tau``-quantile by IRLS on the check loss.

Each fit returns a small result object exposing the coefficients, standard errors, fitted values, a
``predict`` method, and the relevant goodness-of-fit summary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy import special, stats

# --------------------------------------------------------------------------- links


@dataclass(frozen=True)
class Link:
    """A link function ``eta = g(mu)`` with its inverse and derivative ``dmu/deta``."""

    name: str
    g: Callable[[np.ndarray], np.ndarray]
    inv: Callable[[np.ndarray], np.ndarray]
    mu_eta: Callable[[np.ndarray], np.ndarray]  # dmu/deta as a function of eta


def _solve_psd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve a (weighted) normal-equations system, robust to a singular/ill-conditioned design.

    IRLS on collinear predictors (e.g. high-dim modality feature vectors as parents in a factor) yields
    a singular ``X'WX``; a bare ``solve`` would raise. Fall back to the minimum-norm least-squares
    solution (``lstsq``), which is well-defined and stable there and identical when the system is full
    rank -- so a well-conditioned fit is unchanged and a rank-deficient one no longer crashes."""
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(a, b, rcond=None)[0]


def _clip01(p: np.ndarray) -> np.ndarray:
    eps = 1e-10
    return np.clip(p, eps, 1.0 - eps)


_LINKS: dict[str, Link] = {
    "identity": Link("identity", lambda mu: mu, lambda eta: eta, lambda eta: np.ones_like(eta)),
    "log": Link("log", lambda mu: np.log(mu), lambda eta: np.exp(eta), lambda eta: np.exp(eta)),
    "logit": Link(
        "logit",
        lambda mu: np.log(_clip01(mu) / (1.0 - _clip01(mu))),
        lambda eta: special.expit(eta),
        lambda eta: special.expit(eta) * (1.0 - special.expit(eta)),
    ),
    "probit": Link(
        "probit",
        lambda mu: stats.norm.ppf(_clip01(mu)),
        lambda eta: stats.norm.cdf(eta),
        lambda eta: stats.norm.pdf(eta),
    ),
    "cloglog": Link(
        "cloglog",
        lambda mu: np.log(-np.log(1.0 - _clip01(mu))),
        lambda eta: 1.0 - np.exp(-np.exp(eta)),
        lambda eta: np.exp(eta - np.exp(eta)),
    ),
    "inverse": Link("inverse", lambda mu: 1.0 / mu, lambda eta: 1.0 / eta, lambda eta: -1.0 / eta**2),
    "sqrt": Link("sqrt", lambda mu: np.sqrt(mu), lambda eta: eta**2, lambda eta: 2.0 * eta),
}


# --------------------------------------------------------------------------- families


@dataclass(frozen=True)
class Family:
    """An exponential-family error model: variance function, canonical link, deviance, dispersion."""

    name: str
    variance: Callable[[np.ndarray], np.ndarray]
    canonical: str
    unit_deviance: Callable[[np.ndarray, np.ndarray], np.ndarray]
    estimate_dispersion: bool
    extra: float = 1.0  # negative-binomial theta


def _binom_dev(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    mu = _clip01(mu)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(y > 0, y * np.log(y / mu), 0.0)
        t2 = np.where(y < 1, (1 - y) * np.log((1 - y) / (1 - mu)), 0.0)
    return 2.0 * (t1 + t2)


def _pois_dev(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2.0 * (t - (y - mu))


def _gamma_dev(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return 2.0 * (-np.log(y / mu) + (y - mu) / mu)


def _ig_dev(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return (y - mu) ** 2 / (y * mu**2)


def _make_negbin(theta: float) -> Family:
    def dev(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = np.where(y > 0, y * np.log(y / mu), 0.0)
            t2 = (y + theta) * np.log((y + theta) / (mu + theta))
        return 2.0 * (t1 - t2)

    return Family("negativebinomial", lambda mu: mu + mu**2 / theta, "log", dev, False, extra=theta)


_FAMILIES: dict[str, Family] = {
    "gaussian": Family("gaussian", lambda mu: np.ones_like(mu), "identity", lambda y, mu: (y - mu) ** 2, True),
    "binomial": Family("binomial", lambda mu: _clip01(mu) * (1 - _clip01(mu)), "logit", _binom_dev, False),
    "poisson": Family("poisson", lambda mu: mu, "log", _pois_dev, False),
    "gamma": Family("gamma", lambda mu: mu**2, "log", _gamma_dev, True),
    "inverse_gaussian": Family("inverse_gaussian", lambda mu: mu**3, "log", _ig_dev, True),
}


def _resolve_family(family: str | Family, theta: float) -> Family:
    if isinstance(family, Family):
        return family
    if family == "negativebinomial":
        return _make_negbin(theta)
    if family not in _FAMILIES:
        raise ValueError(f"unknown family '{family}'.")
    return _FAMILIES[family]


# --------------------------------------------------------------------------- GLM


@dataclass
class GLMResult:
    """Fitted GLM.

    Attributes:
        coef: ``(p,)`` coefficient estimates.
        se: ``(p,)`` standard errors (model-based, or robust if requested).
        fitted: ``(n,)`` fitted means ``mu``.
        deviance: residual deviance.
        dispersion: estimated/assumed dispersion ``phi``.
        log_likelihood: maximised log-likelihood.
        n_iter: IRLS iterations to convergence.
        family / link: names.
        cov: ``(p, p)`` coefficient covariance.
    """

    coef: np.ndarray
    se: np.ndarray
    fitted: np.ndarray
    deviance: float
    dispersion: float
    log_likelihood: float
    n_iter: int
    family: str
    link: str
    cov: np.ndarray
    _link: Link = field(repr=False, default=None)

    def predict(self, x: np.ndarray, *, offset: np.ndarray | None = None) -> np.ndarray:
        """Predict the mean response ``mu`` at new design rows ``x``."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        eta = x @ self.coef
        if offset is not None:
            eta = eta + np.asarray(offset, dtype=float)
        return self._link.inv(eta)

    @property
    def aic(self) -> float:
        """Akaike information criterion for the fitted GLM."""
        return float(-2.0 * self.log_likelihood + 2.0 * self.coef.size)

    @property
    def bic(self) -> float:
        """Bayesian information criterion for the fitted GLM."""
        return float(-2.0 * self.log_likelihood + np.log(self.fitted.size) * self.coef.size)

    def z_values(self) -> np.ndarray:
        """Return Wald z statistics for fitted coefficients."""
        return self.coef / self.se

    def p_values(self) -> np.ndarray:
        """Return two-sided normal-approximation p-values for coefficients."""
        return 2.0 * stats.norm.sf(np.abs(self.z_values()))


def _loglik(family: Family, y: np.ndarray, mu: np.ndarray, phi: float, weights: np.ndarray) -> float:
    name = family.name
    if name == "gaussian":
        return float(np.sum(weights * stats.norm.logpdf(y, mu, np.sqrt(phi))))
    if name == "poisson":
        return float(np.sum(weights * stats.poisson.logpmf(y, mu)))
    if name == "binomial":
        m = _clip01(mu)
        return float(np.sum(weights * (y * np.log(m) + (1 - y) * np.log(1 - m))))
    if name == "gamma":
        shape = 1.0 / phi
        return float(np.sum(weights * stats.gamma.logpdf(y, shape, scale=mu * phi)))
    if name == "negativebinomial":
        theta = family.extra
        return float(np.sum(weights * stats.nbinom.logpmf(y, theta, theta / (theta + mu))))
    # inverse gaussian / fallback: use -deviance/2 as a proxy
    return float(-0.5 * np.sum(weights * family.unit_deviance(y, mu)) / phi)


def glm(
    x: np.ndarray,
    y: np.ndarray,
    *,
    family: str | Family = "gaussian",
    link: str | None = None,
    offset: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    robust: bool = False,
) -> GLMResult:
    """Fit a generalized linear model by iteratively reweighted least squares.

    Args:
        x: ``(n, p)`` design matrix (include an intercept column explicitly if wanted).
        y: ``(n,)`` response (counts, 0/1 or proportions, positive reals, ... per the family).
        family: ``"gaussian"``, ``"binomial"``, ``"poisson"``, ``"gamma"``, ``"inverse_gaussian"``,
            ``"negativebinomial"``, or a :class:`Family`.
        link: link name; defaults to the family's canonical link.
        offset: ``(n,)`` known additive term on the linear-predictor scale (e.g. ``log`` exposure).
        weights: ``(n,)`` prior weights.
        max_iter, tol: IRLS controls (convergence on the relative deviance change).
        robust: if True report Huber--White sandwich standard errors instead of model-based ones.

    Returns:
        A :class:`GLMResult`.
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    fam = _resolve_family(
        family, getattr(family, "extra", 1.0) if isinstance(family, Family) else _nb_theta_default(family)
    )
    lk = _LINKS[link or fam.canonical]
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float).ravel()
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float).ravel()

    # initialise mu in the interior of the family's support
    if fam.name == "binomial":
        mu = (y + 0.5) / 2.0
    elif fam.name in ("poisson", "gamma", "inverse_gaussian", "negativebinomial"):
        mu = np.maximum(y, 0.1) + 0.1
    else:
        mu = y.copy()
    eta = lk.g(mu)

    beta = np.zeros(p)
    dev_old = np.inf
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        dmu = lk.mu_eta(eta)
        var = fam.variance(mu)
        wls_w = w * dmu**2 / var
        z = (eta - off) + (y - mu) / dmu
        XtW = X.T * wls_w
        new_beta = _solve_psd(XtW @ X, XtW @ z)
        new_eta = X @ new_beta + off
        if not (np.all(np.isfinite(new_beta)) and np.all(np.isfinite(new_eta))):
            break  # divergence (e.g. complete separation): keep the last finite iterate
        beta, eta = new_beta, new_eta
        mu = lk.inv(eta)
        dev = float(np.sum(w * fam.unit_deviance(y, mu)))
        if np.abs(dev - dev_old) <= tol * (np.abs(dev) + 0.1):
            break
        dev_old = dev

    dmu = lk.mu_eta(eta)
    var = fam.variance(mu)
    wls_w = np.nan_to_num(w * dmu**2 / var, nan=0.0, posinf=0.0, neginf=0.0)
    xtwx_inv = np.linalg.pinv((X.T * wls_w) @ X)  # pinv: robust to collinear high-dim parents
    dev = float(np.sum(w * fam.unit_deviance(y, mu)))
    if fam.estimate_dispersion:
        phi = float(np.sum(w * (y - mu) ** 2 / var) / max(n - p, 1))
    else:
        phi = 1.0
    if robust:
        # per-observation score x_i * w_i (y-mu) (dmu/deta) / V(mu); sandwich B (sum gg') B
        score = X * (w * (y - mu) * dmu / var)[:, None]
        meat = score.T @ score
        cov = xtwx_inv @ meat @ xtwx_inv
    else:
        cov = phi * xtwx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    ll = _loglik(fam, y, mu, phi, w)
    return GLMResult(beta, se, mu, dev, phi, ll, n_iter, fam.name, lk.name, cov, _link=lk)


def _nb_theta_default(family: str) -> float:
    return 1.0


# --------------------------------------------------------------------------- penalized


@dataclass
class PenalizedResult:
    """Fitted penalized linear regression.

    Attributes:
        coef: ``(p,)`` coefficients (excluding the intercept).
        intercept: fitted intercept.
        alpha: overall penalty strength.
        l1_ratio: elastic-net mixing (1 = lasso, 0 = ridge).
        n_iter: coordinate-descent iterations (0 for the closed-form ridge).
    """

    coef: np.ndarray
    intercept: float
    alpha: float
    l1_ratio: float
    n_iter: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict penalized-regression responses for design rows."""
        return np.atleast_2d(np.asarray(x, dtype=float)) @ self.coef + self.intercept


def ridge_regression(
    x: np.ndarray, y: np.ndarray, alpha: float = 1.0, *, fit_intercept: bool = True
) -> PenalizedResult:
    """Ridge (L2-penalised) linear regression in closed form.

    Minimises ``||y - X b||^2 + alpha ||b||^2``; the intercept (if fitted) is not penalised.
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    if fit_intercept:
        xm, ym = X.mean(axis=0), y.mean()
        Xc, yc = X - xm, y - ym
    else:
        Xc, yc, xm, ym = X, y, np.zeros(X.shape[1]), 0.0
    p = Xc.shape[1]
    beta = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(p), Xc.T @ yc)
    intercept = float(ym - xm @ beta) if fit_intercept else 0.0
    return PenalizedResult(beta, intercept, alpha, 0.0, 0)


def elastic_net(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    *,
    fit_intercept: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-7,
) -> PenalizedResult:
    """Elastic-net linear regression by cyclic coordinate descent.

    Minimises ``(1/2n) ||y - X b||^2 + alpha ( l1_ratio ||b||_1 + (1 - l1_ratio)/2 ||b||^2 )``.
    ``l1_ratio = 1`` is the lasso (sparse), ``l1_ratio = 0`` is ridge.
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    if fit_intercept:
        xm, ym = X.mean(axis=0), y.mean()
        Xc, yc = X - xm, y - ym
    else:
        Xc, yc, xm, ym = X, y, np.zeros(p), 0.0
    beta = np.zeros(p)
    col_sq = np.sum(Xc**2, axis=0) / n
    r = yc - Xc @ beta
    lam1 = alpha * l1_ratio
    lam2 = alpha * (1.0 - l1_ratio)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        max_delta = 0.0
        for j in range(p):
            if col_sq[j] == 0:
                continue
            r = r + Xc[:, j] * beta[j]
            rho = (Xc[:, j] @ r) / n
            new = np.sign(rho) * max(abs(rho) - lam1, 0.0) / (col_sq[j] + lam2)
            max_delta = max(max_delta, abs(new - beta[j]))
            beta[j] = new
            r = r - Xc[:, j] * beta[j]
        if max_delta < tol:
            break
    intercept = float(ym - xm @ beta) if fit_intercept else 0.0
    return PenalizedResult(beta, intercept, alpha, l1_ratio, n_iter)


def lasso(x: np.ndarray, y: np.ndarray, alpha: float = 1.0, **kw) -> PenalizedResult:
    """Lasso (L1) linear regression -- :func:`elastic_net` with ``l1_ratio = 1``."""
    return elastic_net(x, y, alpha, 1.0, **kw)


# --------------------------------------------------------------------------- robust / quantile


@dataclass
class RegressionFit:
    """Coefficients + fitted values from :func:`robust_regression` / :func:`quantile_regression`."""

    coef: np.ndarray
    fitted: np.ndarray
    scale: float
    n_iter: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict fitted regression values for design rows."""
        return np.atleast_2d(np.asarray(x, dtype=float)) @ self.coef


def robust_regression(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str = "huber",
    c: float | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> RegressionFit:
    """Robust (M-estimator) linear regression by IRLS with a robust scale.

    Down-weights observations with large residuals so a few outliers cannot dominate the fit. ``huber``
    uses the Huber weight (tuning ``c = 1.345`` for 95% Gaussian efficiency); ``tukey`` uses the
    redescending Tukey biweight (``c = 4.685``), which rejects gross outliers entirely.
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    if c is None:
        c = 1.345 if method == "huber" else 4.685
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    n_iter = 0
    scale = 1.0
    for n_iter in range(1, max_iter + 1):
        r = y - X @ beta
        scale = max(np.median(np.abs(r - np.median(r))) / 0.6745, 1e-8)
        u = r / scale
        if method == "huber":
            w = np.where(np.abs(u) <= c, 1.0, c / np.maximum(np.abs(u), 1e-12))
        elif method == "tukey":
            w = np.where(np.abs(u) <= c, (1.0 - (u / c) ** 2) ** 2, 0.0)
        else:
            raise ValueError("method must be 'huber' or 'tukey'.")
        XtW = X.T * w
        new = _solve_psd(XtW @ X, XtW @ y)
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    return RegressionFit(beta, X @ beta, float(scale), n_iter)


def quantile_regression(
    x: np.ndarray, y: np.ndarray, tau: float = 0.5, *, max_iter: int = 200, tol: float = 1e-7, eps: float = 1e-6
) -> RegressionFit:
    """Linear quantile regression: the conditional ``tau``-quantile by IRLS on the check loss.

    Minimises the pinball loss ``sum rho_tau(y - X b)`` via iteratively reweighted least squares with
    weights ``tau / |r|`` for positive residuals and ``(1 - tau) / |r|`` for negative ones (a smoothed
    Newton scheme; ``eps`` floors ``|r|`` for stability).
    """
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must be in (0, 1).")
    X = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        r = y - X @ beta
        w = np.where(r >= 0, tau, 1.0 - tau) / np.maximum(np.abs(r), eps)
        XtW = X.T * w
        new = _solve_psd(XtW @ X, XtW @ y)
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    return RegressionFit(beta, X @ beta, float(np.mean(np.abs(y - X @ beta))), n_iter)


__all__ = [
    "Link",
    "Family",
    "GLMResult",
    "glm",
    "PenalizedResult",
    "ridge_regression",
    "elastic_net",
    "lasso",
    "RegressionFit",
    "robust_regression",
    "quantile_regression",
]
