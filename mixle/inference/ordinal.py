"""Ordinal regression and rank-correlation (concordance) measures.

When the response is *ordered categories* (none < mild < severe; 1--5 stars) the spacing between
levels is unknown, so neither plain regression (assumes equal spacing) nor multinomial logit (throws
away the order) is right. The cumulative (proportional-odds / proportional-hazards) model is:

    P(Y <= k | x) = F(alpha_k - x' beta),  alpha_1 < ... < alpha_{K-1},

a single coefficient vector ``beta`` with ``K-1`` ordered thresholds. :func:`ordinal_regression`
fits this by maximum likelihood with ``F`` the logistic (ordered logit / proportional odds) or normal
(ordered probit) CDF.

The concordance measures summarise the monotone association between two ordinal variables from the
counts of concordant/discordant pairs: :func:`kendall_tau` (tau-b, tie-corrected),
:func:`goodman_kruskal_gamma`, and :func:`somers_d` (asymmetric). :func:`concordance_summary` returns
all of them with the underlying pair counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


def _cdf(link: str, z: np.ndarray) -> np.ndarray:
    if link == "logit":
        return stats.logistic.cdf(z)
    if link == "probit":
        return stats.norm.cdf(z)
    raise ValueError("link must be 'logit' or 'probit'.")


def _logcdf(link: str, z: np.ndarray) -> np.ndarray:
    if link == "logit":
        return stats.logistic.logcdf(z)
    if link == "probit":
        return stats.norm.logcdf(z)
    raise ValueError("link must be 'logit' or 'probit'.")


def _ordinal_design(x: np.ndarray, p: int | None = None) -> np.ndarray:
    X = np.asarray(x, dtype=float)
    if p is not None and X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] < 1:
        raise ValueError("x must be a non-empty two-dimensional design matrix")
    if p is not None and X.shape[1] != p:
        raise ValueError(f"x must have shape (n, {p})")
    if not np.all(np.isfinite(X)):
        raise ValueError("x must contain only finite values")
    return X


def _paired_ordinal_values(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size < 2:
        raise ValueError("x and y must be aligned one-dimensional arrays with at least two values")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("x and y must contain only finite values")
    return a, b


@dataclass
class OrdinalResult:
    """Fitted ordinal (cumulative-link) regression.

    Attributes:
        coef: ``(p,)`` slope coefficients (positive ``beta_j`` raises the latent score, shifting mass
            toward higher categories).
        thresholds: ``(K-1,)`` ordered cut points ``alpha``.
        se: ``(p,)`` standard errors for ``coef``.
        log_likelihood: maximised log-likelihood.
        link: ``"logit"`` or ``"probit"``.
        n_categories: number of ordered categories ``K``.
    """

    coef: np.ndarray
    thresholds: np.ndarray
    se: np.ndarray
    log_likelihood: float
    link: str
    n_categories: int
    n_iter: int = 0
    converged: bool = True
    rank: int | None = None

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Per-category probabilities ``(n, K)`` at design rows ``x``."""
        x = _ordinal_design(x, self.coef.size)
        eta = x @ self.coef
        cuts = np.concatenate([[-np.inf], self.thresholds, [np.inf]])
        cdfs = _cdf(self.link, cuts[None, :] - eta[:, None])
        return np.diff(cdfs, axis=1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Most-probable ordered category per row."""
        return np.argmax(self.predict_proba(x), axis=1)


def ordinal_regression(x: np.ndarray, y: np.ndarray, *, link: str = "logit", max_iter: int = 200) -> OrdinalResult:
    """Fit a cumulative-link ordinal regression (ordered logit / probit) by maximum likelihood.

    Args:
        x: ``(n, p)`` covariates (no intercept -- the thresholds play that role).
        y: ``(n,)`` integer category labels ``0..K-1`` (ordered).
        link: ``"logit"`` (proportional odds) or ``"probit"``.
        max_iter: optimiser iterations.

    Returns:
        An :class:`OrdinalResult`.
    """
    if link not in {"logit", "probit"}:
        raise ValueError("link must be 'logit' or 'probit'.")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    X = _ordinal_design(x)
    raw_y = np.asarray(y)
    if raw_y.ndim != 1 or raw_y.shape[0] != X.shape[0]:
        raise ValueError("y must be a one-dimensional category array aligned with x")
    try:
        numeric_y = np.asarray(raw_y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("y must contain finite integer category labels") from exc
    if not np.all(np.isfinite(numeric_y)) or np.any(numeric_y != np.floor(numeric_y)):
        raise ValueError("y must contain finite integer category labels")
    y = numeric_y.astype(np.intp)
    if np.any(y < 0):
        raise ValueError("category labels must be contiguous integers starting at 0")
    n, p = X.shape
    categories = np.unique(y)
    K = int(categories[-1]) + 1
    if K < 2:
        raise ValueError("need at least two ordered categories.")
    if not np.array_equal(categories, np.arange(K)):
        raise ValueError("category labels must be contiguous integers starting at 0")
    rank = int(np.linalg.matrix_rank(X))
    if rank < p:
        raise ValueError(f"ordinal design is rank deficient ({rank} < {p})")

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        beta = theta[:p]
        first = theta[p]
        with np.errstate(over="ignore", invalid="ignore"):
            incs = np.exp(theta[p + 1 :])  # positive increments -> ordered thresholds
        cuts = np.concatenate([[first], first + np.cumsum(incs)])
        return beta, cuts

    def neg_ll(theta: np.ndarray) -> float:
        beta, cuts = unpack(theta)
        if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(cuts)) or np.any(np.diff(cuts) <= 0):
            return np.inf
        eta = X @ beta
        full = np.concatenate([[-np.inf], cuts, [np.inf]])
        log_lo = _logcdf(link, full[y] - eta)
        log_hi = _logcdf(link, full[y + 1] - eta)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_prob = log_hi + np.log(-np.expm1(log_lo - log_hi))
        if not np.all(np.isfinite(log_prob)):
            return np.inf
        return float(-np.sum(log_prob))

    # init: zero slopes, thresholds at standard-normal quantiles of the category frequencies
    cumfreq = np.cumsum(np.bincount(y, minlength=K)[:-1]) / n
    init_cuts = stats.logistic.ppf(cumfreq) if link == "logit" else stats.norm.ppf(cumfreq)
    theta0 = (
        np.concatenate([np.zeros(p), [init_cuts[0]], np.log(np.maximum(np.diff(init_cuts), 0.1))])
        if K > 2
        else np.concatenate([np.zeros(p), [init_cuts[0]]])
    )
    res = optimize.minimize(neg_ll, theta0, method="L-BFGS-B", options={"maxiter": max_iter})
    if not res.success or not np.isfinite(res.fun) or not np.all(np.isfinite(res.x)):
        raise RuntimeError(f"ordinal regression failed to converge: {res.message}")
    beta, cuts = unpack(res.x)
    cov = res.hess_inv if isinstance(res.hess_inv, np.ndarray) else np.asarray(res.hess_inv.todense())
    if cov.shape != (res.x.size, res.x.size) or not np.all(np.isfinite(cov)):
        raise RuntimeError("ordinal regression produced an invalid inverse Hessian")
    se = np.sqrt(np.clip(np.diag(cov)[:p], 0.0, None))
    return OrdinalResult(beta, cuts, se, float(-res.fun), link, K, int(res.nit), True, rank)


# --------------------------------------------------------------------------- concordance


def _pair_counts(x: np.ndarray, y: np.ndarray) -> dict[str, int]:
    """Concordant / discordant / tie pair counts for two ordinal variables (O(n log n) via sort)."""
    x, y = _paired_ordinal_values(x, y)
    n = x.shape[0]
    c = d = tx = ty = txy = 0
    for i in range(n):
        dx = np.sign(x[i + 1 :] - x[i])
        dy = np.sign(y[i + 1 :] - y[i])
        prod = dx * dy
        c += int(np.sum(prod > 0))
        d += int(np.sum(prod < 0))
        txy += int(np.sum((dx == 0) & (dy == 0)))
        tx += int(np.sum((dx == 0) & (dy != 0)))
        ty += int(np.sum((dx != 0) & (dy == 0)))
    return {"concordant": c, "discordant": d, "tx": tx, "ty": ty, "txy": txy}


def concordance_summary(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """All pairwise concordance measures between two ordinal variables.

    Returns:
        ``{'kendall_tau_b', 'gamma', 'somers_d_yx', 'somers_d_xy', 'concordant', 'discordant',
        'tx', 'ty', 'txy'}`` -- ``tx`` = pairs tied on ``x`` only, ``ty`` on ``y`` only, ``txy`` on both.
    """
    pc = _pair_counts(x, y)
    c, d, tx, ty = pc["concordant"], pc["discordant"], pc["tx"], pc["ty"]
    cd = c + d
    tau_b = (c - d) / np.sqrt((cd + tx) * (cd + ty)) if (cd + tx) > 0 and (cd + ty) > 0 else 0.0
    gamma = (c - d) / cd if cd > 0 else 0.0
    d_yx = (c - d) / (cd + ty) if (cd + ty) > 0 else 0.0
    d_xy = (c - d) / (cd + tx) if (cd + tx) > 0 else 0.0
    return {
        "kendall_tau_b": float(tau_b),
        "gamma": float(gamma),
        "somers_d_yx": float(d_yx),
        "somers_d_xy": float(d_xy),
        **{k: float(v) for k, v in pc.items()},
    }


def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall's tau-b rank correlation (tie-corrected) between two ordinal variables."""
    return concordance_summary(x, y)["kendall_tau_b"]


def goodman_kruskal_gamma(x: np.ndarray, y: np.ndarray) -> float:
    """Goodman--Kruskal gamma: ``(C - D) / (C + D)`` ignoring ties."""
    return concordance_summary(x, y)["gamma"]


def somers_d(x: np.ndarray, y: np.ndarray, *, dependent: str = "y") -> float:
    """Somers' D, the asymmetric rank association treating ``dependent`` as the response."""
    if dependent not in {"x", "y"}:
        raise ValueError("dependent must be 'x' or 'y'")
    s = concordance_summary(x, y)
    return s["somers_d_yx"] if dependent == "y" else s["somers_d_xy"]


__all__ = [
    "OrdinalResult",
    "ordinal_regression",
    "concordance_summary",
    "kendall_tau",
    "goodman_kruskal_gamma",
    "somers_d",
]
