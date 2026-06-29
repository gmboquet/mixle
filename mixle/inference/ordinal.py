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

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Per-category probabilities ``(n, K)`` at design rows ``x``."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
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
    X = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.asarray(y).astype(int).ravel()
    n, p = X.shape
    K = int(y.max()) + 1
    if K < 2:
        raise ValueError("need at least two ordered categories.")

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        beta = theta[:p]
        first = theta[p]
        incs = np.exp(theta[p + 1 :])  # positive increments -> ordered thresholds
        cuts = np.concatenate([[first], first + np.cumsum(incs)])
        return beta, cuts

    def neg_ll(theta: np.ndarray) -> float:
        beta, cuts = unpack(theta)
        eta = X @ beta
        full = np.concatenate([[-np.inf], cuts, [np.inf]])
        lo = _cdf(link, full[y] - eta)
        hi = _cdf(link, full[y + 1] - eta)
        prob = np.clip(hi - lo, 1e-12, 1.0)
        return float(-np.sum(np.log(prob)))

    # init: zero slopes, thresholds at standard-normal quantiles of the category frequencies
    cumfreq = np.cumsum(np.bincount(y, minlength=K)[:-1]) / n
    init_cuts = stats.norm.ppf(np.clip(cumfreq, 0.01, 0.99))
    theta0 = (
        np.concatenate([np.zeros(p), [init_cuts[0]], np.log(np.maximum(np.diff(init_cuts), 0.1))])
        if K > 2
        else np.concatenate([np.zeros(p), [init_cuts[0]]])
    )
    res = optimize.minimize(neg_ll, theta0, method="BFGS", options={"maxiter": max_iter})
    beta, cuts = unpack(res.x)
    cov = res.hess_inv if isinstance(res.hess_inv, np.ndarray) else np.asarray(res.hess_inv.todense())
    se = np.sqrt(np.clip(np.diag(cov)[:p], 0.0, None))
    return OrdinalResult(beta, cuts, se, float(-res.fun), link, K)


# --------------------------------------------------------------------------- concordance


def _pair_counts(x: np.ndarray, y: np.ndarray) -> dict[str, int]:
    """Concordant / discordant / tie pair counts for two ordinal variables (O(n log n) via sort)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
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
