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

``glm`` returns a result with coefficients, standard errors, and Wald inference; the penalized,
robust, and quantile fits return coefficients and fits WITHOUT standard errors (their correct
inferential machinery -- debiased lasso, M-estimator sandwiches, quantile sparsity estimates --
is deliberately not faked here; audit G-6).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, sparse, special, stats

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
    rank -- so a well-conditioned fit is unchanged and a rank-deficient one no longer crashes.

    Used by the ridge / robust / quantile solvers, whose inference machinery is deliberately absent
    (audit G-6). :func:`glm` does NOT use it: its coefficients and covariance carry Wald inference,
    so it factorizes the weighted design ``sqrt(W) X`` directly -- forming the normal equations
    squares the condition number and silently corrupts the standard errors (audit B4)."""
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(a, b, rcond=None)[0]


def _clip01(p: np.ndarray) -> np.ndarray:
    eps = 1e-10
    return np.clip(p, eps, 1.0 - eps)


def _regression_data(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a finite, aligned two-dimensional design and one-dimensional response."""
    X = np.asarray(x, dtype=float)
    response = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError("x must be a two-dimensional (n, p) design matrix")
    if response.ndim != 1:
        raise ValueError("y must be a one-dimensional response")
    n, p = X.shape
    if n < 1 or p < 1:
        raise ValueError("x must have at least 1 row and 1 column")
    if response.shape[0] != n:
        raise ValueError("x and y must have the same number of rows")
    if not np.all(np.isfinite(X)):
        raise ValueError("x must contain only finite values")
    if not np.all(np.isfinite(response)):
        raise ValueError("y must contain only finite values")
    return X, response


def _solver_controls(max_iter: int, tol: float) -> None:
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and > 0")


def _response_support(family: Family, y: np.ndarray) -> None:
    if family.name == "binomial" and np.any((y < 0) | (y > 1)):
        raise ValueError("binomial responses must lie in [0, 1]")
    if family.name in {"poisson", "negativebinomial"}:
        if np.any(y < 0) or np.any(y != np.floor(y)):
            raise ValueError(f"{family.name} responses must be non-negative integers")
    if family.name in {"gamma", "inverse_gaussian"} and np.any(y <= 0):
        raise ValueError(f"{family.name} responses must be strictly positive")


class PerfectSeparationError(RuntimeError):
    """The binomial classes are perfectly separated (or perfectly predicted): no finite MLE exists.

    Raised by :func:`glm` when IRLS drives fitted binomial probabilities to exactly 0 or 1 -- the
    numerical signature of (quasi-)complete separation, where ``|coef|`` diverges and the working
    weights vanish. A subclass of :class:`RuntimeError`, so pre-existing handlers keep working."""


def _mean_support(family: Family, mu: np.ndarray) -> None:
    if family.name == "binomial":
        # A monotone inverse link keeps mu inside [0, 1]; hitting the endpoints EXACTLY is the
        # numerical signature of separation (diverging coefficients, vanishing working weights),
        # not a solver defect -- so name the statistical condition, not the internal symptom.
        pinned = (mu == 0.0) | (mu == 1.0)
        if np.any(pinned):
            raise PerfectSeparationError(
                "perfect separation detected between the binomial classes: the design predicts "
                f"y exactly on {int(np.count_nonzero(pinned))} of {mu.size} observations (fitted "
                "probabilities reached exactly 0 or 1), so the coefficients diverge and no finite "
                "maximum-likelihood estimate exists. Remove or coarsen the separating "
                "predictor(s), or use penalized estimation (ridge_regression / elastic_net), "
                "which stays finite under separation."
            )
        if np.any((mu < 0) | (mu > 1)):
            raise RuntimeError("IRLS produced binomial means outside (0, 1)")
    if family.name in {"poisson", "negativebinomial", "gamma", "inverse_gaussian"} and np.any(mu <= 0):
        raise RuntimeError(f"IRLS produced non-positive {family.name} means")


def _prediction_design(x: np.ndarray, p: int) -> np.ndarray:
    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2 or X.shape[1] != p:
        raise ValueError(f"x must have shape (n, {p})")
    if not np.all(np.isfinite(X)):
        raise ValueError("x must contain only finite values")
    return X


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
    "inverse_squared": Link(
        "inverse_squared", lambda mu: 1.0 / mu**2, lambda eta: 1.0 / np.sqrt(eta), lambda eta: -0.5 * eta**-1.5
    ),
    "sqrt": Link("sqrt", lambda mu: np.sqrt(mu), lambda eta: eta**2, lambda eta: 2.0 * eta),
}


# --------------------------------------------------------------------------- families


@dataclass(frozen=True)
class Family:
    """An exponential-family error model: variance function, canonical link, deviance, dispersion.

    ``canonical`` names the family's *mathematical* canonical link; ``default_link`` (when set) is
    the link :func:`glm` fits with when none is requested. They are separate so families whose
    canonical link is numerically awkward (gamma's ``inverse``, inverse-Gaussian's
    ``inverse_squared`` -- neither keeps ``mu`` positive) can default to ``log`` without mislabeling
    the canonical link.
    """

    name: str
    variance: Callable[[np.ndarray], np.ndarray]
    canonical: str
    unit_deviance: Callable[[np.ndarray, np.ndarray], np.ndarray]
    estimate_dispersion: bool
    extra: float = 1.0  # negative-binomial theta
    default_link: str | None = None  # fitting default when it differs from the canonical link


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
    "gamma": Family(
        name="gamma",
        variance=lambda mu: mu**2,
        canonical="inverse",
        unit_deviance=_gamma_dev,
        estimate_dispersion=True,
        default_link="log",
    ),
    "inverse_gaussian": Family(
        name="inverse_gaussian",
        variance=lambda mu: mu**3,
        canonical="inverse_squared",
        unit_deviance=_ig_dev,
        estimate_dispersion=True,
        default_link="log",
    ),
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
        dispersion: estimated/assumed dispersion ``phi`` -- the residual-df-corrected (Pearson)
            estimate the standard errors use, NOT the value the log-likelihood is evaluated at.
        log_likelihood: maximised log-likelihood, or ``None`` when the supplied
            family/response does not define one. For dispersion-estimating families this is
            evaluated at the family's own dispersion MLE (see ``_dispersion_mle``), which
            differs from ``dispersion`` by design: the covariance wants the df-corrected
            estimate, the likelihood wants its maximiser.
        n_iter: IRLS iterations to convergence.
        converged: whether the IRLS convergence criterion was met. Public fits
            currently raise instead of returning this as false.
        rank: effective NUMERICAL rank of the weighted design ``sqrt(W) X`` at the solution,
            from the same SVD the covariance is computed from (cutoff at ``cond(X)``, not
            ``cond(X)^2``; audit B4). ``rank < p`` -- exact or near collinearity -- is announced
            by a ``UserWarning`` at fit time and makes ``z_values`` / ``p_values`` refuse.
        family / link: names.
        cov: ``(p, p)`` coefficient covariance.
    """

    coef: np.ndarray
    se: np.ndarray
    fitted: np.ndarray
    deviance: float
    dispersion: float
    log_likelihood: float | None
    n_iter: int
    family: str
    link: str
    cov: np.ndarray
    converged: bool = True
    rank: int | None = None
    # audit G-1/G-3: inference metadata -- the t reference needs the residual degrees of freedom
    # when the dispersion was estimated, and per-coefficient Wald inference is only defined when
    # the design has full rank (the pinv min-norm split otherwise fabricates smaller SEs)
    residual_df: int | None = None
    dispersion_estimated: bool = False
    _link: Link = field(repr=False, default=None)

    def predict(self, x: np.ndarray, *, offset: np.ndarray | None = None) -> np.ndarray:
        """Predict the mean response ``mu`` at new design rows ``x``."""
        x = _prediction_design(x, self.coef.size)
        eta = x @ self.coef
        if offset is not None:
            off = np.asarray(offset, dtype=float)
            if off.ndim == 0:
                off = np.full(x.shape[0], float(off))
            if off.ndim != 1 or off.shape[0] != x.shape[0] or not np.all(np.isfinite(off)):
                raise ValueError("offset must be a finite scalar or one value per prediction row")
            eta = eta + off
        prediction = np.asarray(self._link.inv(eta), dtype=float)
        if prediction.shape != eta.shape or not np.all(np.isfinite(prediction)):
            raise RuntimeError("link inverse returned invalid predictions")
        return prediction

    @property
    def _n_parameters(self) -> int:
        # audit G-2: count what was actually ESTIMATED -- the design rank (not the raw column
        # count, which penalizes rank-deficient fits for parameters they never estimated) plus
        # the dispersion when it was estimated (the Gaussian/Gamma/IG families' extra parameter,
        # previously uncounted, which biased model selection toward larger mean models)
        rank = self.rank if self.rank is not None else self.coef.size
        return int(rank + (1 if self.dispersion_estimated else 0))

    @property
    def aic(self) -> float:
        """Akaike information criterion: ``-2 ll + 2 k`` with ``k = rank + estimated dispersion``."""
        if self.log_likelihood is None:
            raise ValueError("AIC is unavailable because this fit has no defined likelihood")
        return float(-2.0 * self.log_likelihood + 2.0 * self._n_parameters)

    @property
    def bic(self) -> float:
        """Bayesian information criterion over the positive-weight observations actually fit."""
        if self.log_likelihood is None:
            raise ValueError("BIC is unavailable because this fit has no defined likelihood")
        n_effective = self.fitted.size if self.residual_df is None else self.residual_df + (self.rank or 0)
        return float(-2.0 * self.log_likelihood + np.log(n_effective) * self._n_parameters)

    def _require_identifiable(self) -> None:
        # audit G-3: with rank < p the individual coefficients are NOT identified -- the SVD
        # least-squares solve returns the minimum-norm representative (verified: identical to
        # ``pinv(X) @ y`` for the Gaussian/identity fit), which splits a shared effect across collinear columns
        # and shrinks each SE to match (a duplicated column halved both the coefficient and its
        # SE, leaving z unchanged and the collinearity invisible). Only estimable functions have
        # sampling distributions there, so per-coefficient Wald inference refuses.
        if self.rank is not None and self.rank < self.coef.size:
            raise ValueError(
                f"per-coefficient Wald inference is undefined: design rank {self.rank} < "
                f"{self.coef.size} columns, so individual coefficients are not identified (the "
                "reported minimum-norm split is arbitrary). Drop or combine collinear columns, "
                "or test an estimable linear combination instead."
            )

    def z_values(self) -> np.ndarray:
        """Return Wald statistics for fitted coefficients (full-rank designs only)."""
        self._require_identifiable()
        return self.coef / self.se

    def p_values(self) -> np.ndarray:
        """Two-sided Wald p-values: Student-t on the residual df when the dispersion was
        ESTIMATED (Gaussian/Gamma/inverse-Gaussian -- the plug-in-dispersion normal reference
        rejected 9% at nominal 5% at n=8; audit G-1), normal otherwise (fixed-dispersion
        families), both asymptotic in the non-Gaussian mean model."""
        z = np.abs(self.z_values())
        if self.dispersion_estimated and self.residual_df is not None and self.residual_df > 0:
            return 2.0 * stats.t.sf(z, self.residual_df)
        return 2.0 * stats.norm.sf(z)


def _dispersion_mle(family: Family, y: np.ndarray, mu: np.ndarray, weights: np.ndarray, phi_fallback: float) -> float:
    """The maximiser of this family's OWN log-likelihood over the dispersion, at fixed ``mu``.

    Each estimate solves d/d(phi) of the weighted log-density sum ``_loglik`` evaluates -- so the
    reported "maximised log-likelihood" really is evaluated at its maximum:

    * gaussian: ``sum w (y - mu)^2 / sum w`` (the RSS form).
    * inverse_gaussian: ``sum w (y - mu)^2 / (y mu^2) / sum w`` -- the mean unit deviance. The
      Pearson form divides by ``V(mu) = mu^3`` instead of ``y mu^2`` and does NOT maximise the IG
      density (measured 0.1 nats short on an 8-point fit).
    * gamma: no closed form -- the shape ``nu = 1/phi`` solves ``log(nu) - digamma(nu) = c`` with
      ``c`` the weighted mean of ``y/mu - log(y/mu) - 1`` (half the mean unit deviance);
      ``log(nu) - digamma(nu)`` is strictly decreasing from +inf to 0, so the root is unique and
      bracketed in log-space. A numerically perfect fit (``c ~ 0``) sends ``phi -> 0`` and the
      likelihood to +inf; the covariance-phi fallback is returned there rather than a fake maximum.
    """
    total_weight = float(np.sum(weights))
    if family.name == "gaussian":
        return float(np.sum(weights * (y - mu) ** 2) / total_weight)
    if family.name == "inverse_gaussian":
        return float(np.sum(weights * (y - mu) ** 2 / (y * mu**2)) / total_weight)
    if family.name == "gamma":
        c = float(np.sum(weights * (y / mu - np.log(y / mu) - 1.0)) / total_weight)
        if not np.isfinite(c) or c <= 1e-12:
            return phi_fallback
        from scipy.optimize import brentq

        def gap(log_nu: float) -> float:
            return log_nu - float(special.digamma(np.exp(log_nu))) - c

        log_nu_hat = brentq(gap, -30.0, 30.0, xtol=1e-12)
        return float(np.exp(-log_nu_hat))
    return phi_fallback


def _loglik(family: Family, y: np.ndarray, mu: np.ndarray, phi: float, weights: np.ndarray) -> float | None:
    name = family.name
    if name == "gaussian":
        return float(np.sum(weights * stats.norm.logpdf(y, mu, np.sqrt(phi))))
    if name == "poisson":
        return float(np.sum(weights * stats.poisson.logpmf(y, mu)))
    if name == "binomial":
        if np.any((y != 0.0) & (y != 1.0)):
            return None
        m = _clip01(mu)
        return float(np.sum(weights * (y * np.log(m) + (1 - y) * np.log(1 - m))))
    if name == "gamma":
        shape = 1.0 / phi
        return float(np.sum(weights * stats.gamma.logpdf(y, shape, scale=mu * phi)))
    if name == "negativebinomial":
        theta = family.extra
        return float(np.sum(weights * stats.nbinom.logpmf(y, theta, theta / (theta + mu))))
    if name == "inverse_gaussian":
        log_density = -0.5 * (np.log(2.0 * np.pi * phi) + 3.0 * np.log(y) + (y - mu) ** 2 / (phi * y * mu**2))
        return float(np.sum(weights * log_density))
    return None


def glm(
    x: np.ndarray,
    y: np.ndarray,
    *,
    family: str | Family = "gaussian",
    link: str | Link | None = None,
    offset: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    robust: bool = False,
    theta: float = 1.0,
) -> GLMResult:
    """Fit a generalized linear model by iteratively reweighted least squares.

    Args:
        x: ``(n, p)`` design matrix (include an intercept column explicitly if wanted).
        y: ``(n,)`` response (counts, 0/1 or proportions, positive reals, ... per the family).
        family: ``"gaussian"``, ``"binomial"``, ``"poisson"``, ``"gamma"``, ``"inverse_gaussian"``,
            ``"negativebinomial"``, or a :class:`Family`.
        link: a link name, or a :class:`Link` instance to use directly (e.g. a custom link); defaults
            to the family's default link (the canonical link, except gamma / inverse-Gaussian which
            default to ``log`` -- their canonical ``inverse`` / ``inverse_squared`` links do not keep
            ``mu`` positive).
        offset: ``(n,)`` known additive term on the linear-predictor scale (e.g. ``log`` exposure).
        weights: ``(n,)`` prior weights.
        max_iter, tol: IRLS controls (convergence on the relative deviance change).
        robust: if True report Huber--White (HC0) sandwich standard errors instead of
            model-based ones. HC0 carries no finite-sample leverage correction, assumes
            INDEPENDENT observations (it is not cluster-robust), and still requires the mean/link
            to be correct; small-sample intervals lean narrow (audit G-7).
        theta: the negative-binomial dispersion parameter (``family="negativebinomial"`` only;
            ignored otherwise). Not estimated from data -- pass the value appropriate to your data
            (e.g. from a prior fit or a method-of-moments estimate); the default ``1.0`` is a plain
            placeholder, not a fitted value. WARNING (audit
            G-4): the MODEL-BASED standard errors assume ``theta`` is correct -- with the
            placeholder against true theta = 10, measured SEs were 1.87x the true sampling
            spread (nominal 95% Wald intervals covering ~100%, tests losing essentially all
            power). Pass ``robust=True`` unless ``theta`` is trusted, and if ``theta`` came from
            the same data, remember AIC/BIC do not count it.

    Returns:
        A :class:`GLMResult`.
    """
    X, y = _regression_data(x, y)
    n, p = X.shape
    _solver_controls(max_iter, tol)
    theta_arg = getattr(family, "extra", theta) if isinstance(family, Family) else theta
    fam = _resolve_family(family, theta_arg)
    if fam.name == "negativebinomial" and (not np.isfinite(fam.extra) or fam.extra <= 0):
        raise ValueError("theta must be finite and > 0")
    _response_support(fam, y)
    if isinstance(link, Link):
        lk = link
    else:
        link_name = link or fam.default_link or fam.canonical
        if link_name not in _LINKS:
            raise ValueError(f"unknown link '{link_name}'.")
        lk = _LINKS[link_name]
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if off.ndim != 1 or off.shape[0] != n or not np.all(np.isfinite(off)):
        raise ValueError("offset must be a finite one-dimensional array aligned with y")
    if w.ndim != 1 or w.shape[0] != n or not np.all(np.isfinite(w)):
        raise ValueError("weights must be a finite one-dimensional array aligned with y")
    if np.any(w < 0) or not np.any(w > 0):
        raise ValueError("weights must be non-negative with at least one positive value")
    active = w > 0

    # initialise mu in the interior of the family's support
    if fam.name == "binomial":
        mu = (y + 0.5) / 2.0
    elif fam.name in ("poisson", "gamma", "inverse_gaussian", "negativebinomial"):
        mu = np.maximum(y, 0.1) + 0.1
    else:
        mu = y.copy()
    eta = np.asarray(lk.g(mu), dtype=float)
    if eta.shape != mu.shape or not np.all(np.isfinite(eta)):
        raise ValueError("link is incompatible with the initial response mean")

    beta = np.zeros(p)
    dev_old = np.inf
    n_iter = 0
    converged = False
    for n_iter in range(1, max_iter + 1):
        dmu = np.asarray(lk.mu_eta(eta), dtype=float)
        var = np.asarray(fam.variance(mu), dtype=float)
        if (
            dmu.shape != mu.shape
            or var.shape != mu.shape
            or not np.all(np.isfinite(dmu))
            or not np.all(np.isfinite(var))
            or np.any(dmu == 0)
            or np.any(var <= 0)
        ):
            raise RuntimeError("IRLS encountered an invalid link derivative or variance")
        wls_w = w * dmu**2 / var
        z = (eta - off) + (y - mu) / dmu
        if not np.all(np.isfinite(wls_w)) or not np.all(np.isfinite(z)):
            raise RuntimeError("IRLS produced non-finite working values")
        # audit B4: solve the WLS step on the weighted design sqrt(W)X itself, never via the
        # normal equations X'WX -- squaring the design squares its condition number, so every
        # rank/cutoff decision would happen at cond(X)^2 instead of cond(X). lstsq's SVD also
        # returns the true minimum-norm solution when the design is rank-deficient, which is
        # exactly the representative _require_identifiable's message describes.
        sqrt_wls = np.sqrt(wls_w)
        new_beta = np.linalg.lstsq(X * sqrt_wls[:, None], z * sqrt_wls, rcond=None)[0]
        new_eta = X @ new_beta + off
        if not (np.all(np.isfinite(new_beta)) and np.all(np.isfinite(new_eta))):
            raise RuntimeError("IRLS diverged before convergence")
        beta, eta = new_beta, new_eta
        mu = np.asarray(lk.inv(eta), dtype=float)
        if mu.shape != y.shape or not np.all(np.isfinite(mu)):
            raise RuntimeError("IRLS link inverse produced invalid fitted means")
        _mean_support(fam, mu)
        dev = float(np.sum(w * fam.unit_deviance(y, mu)))
        if not np.isfinite(dev):
            raise RuntimeError("IRLS produced non-finite deviance")
        if np.abs(dev - dev_old) <= tol * (np.abs(dev) + 0.1):
            converged = True
            break
        dev_old = dev
    if not converged:
        raise RuntimeError(f"IRLS failed to converge in {max_iter} iterations")

    dmu = np.asarray(lk.mu_eta(eta), dtype=float)
    var = np.asarray(fam.variance(mu), dtype=float)
    wls_w = w * dmu**2 / var
    # audit B4: the coefficient covariance comes from an SVD of the WEIGHTED DESIGN sqrt(W)X,
    # never from pinv(X'WX). cond(X'WX) = cond(sqrt(W)X)^2, so pinv's default cutoff silently
    # truncated singular values once cond(X) passed ~1e8 and collapsed the standard errors by
    # up to 8 orders of magnitude -- with rank reported full, converged=True, and no warning.
    # Factorizing sqrt(W)X keeps rank and cutoff decisions at cond(X); zero-weight rows enter
    # as zero rows, so this is still the effective rank among positive-weight observations.
    _, singular, vt = np.linalg.svd(X * np.sqrt(wls_w)[:, None], full_matrices=False)
    cutoff = np.finfo(float).eps * max(n, p) * (singular[0] if singular.size else 0.0)
    significant = singular > cutoff
    rank = int(np.count_nonzero(significant))
    inv_sq_singular = np.zeros_like(singular)
    inv_sq_singular[significant] = 1.0 / singular[significant] ** 2
    xtwx_inv = (vt.T * inv_sq_singular) @ vt
    if rank < p:
        # exact OR numerical rank deficiency: say so instead of silently full-ranking the fit.
        # The fit itself still returns (minimum-norm coefficients, as documented); z_values /
        # p_values refuse via _require_identifiable because rank < p.
        warnings.warn(
            f"the design is rank-deficient at working precision: effective rank {rank} < {p} "
            "columns, so some predictors are exactly or nearly collinear. Coefficients are the "
            "minimum-norm least-squares representative and per-coefficient Wald inference "
            "(z_values / p_values) will refuse; drop, combine, or center/rescale the collinear "
            "columns to make individual coefficients identifiable.",
            UserWarning,
            stacklevel=2,
        )
    dev = float(np.sum(w * fam.unit_deviance(y, mu)))
    if fam.estimate_dispersion:
        residual_df = int(np.count_nonzero(active)) - rank
        if residual_df <= 0:
            raise ValueError("dispersion is not identifiable without positive residual degrees of freedom")
        phi = float(np.sum(w * (y - mu) ** 2 / var) / residual_df)
    else:
        phi = 1.0
    if not np.isfinite(phi) or phi <= 0:
        raise RuntimeError("fit produced an invalid dispersion estimate")
    if robust:
        # per-observation score x_i * w_i (y-mu) (dmu/deta) / V(mu); sandwich B (sum gg') B
        score = X * (w * (y - mu) * dmu / var)[:, None]
        meat = score.T @ score
        cov = xtwx_inv @ meat @ xtwx_inv
    else:
        cov = phi * xtwx_inv
    if not np.all(np.isfinite(cov)):
        raise RuntimeError("fit produced a non-finite covariance matrix")
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    if fam.estimate_dispersion:
        # audit G-2: "maximised log-likelihood" must be evaluated at the MLE of the dispersion,
        # not the residual-df-corrected phi used for the covariance -- the mixed convention
        # understated ll and, with the uncounted dispersion parameter, distorted AIC/BIC model
        # selection toward larger mean models. The MLE is FAMILY-SPECIFIC (the Pearson-form
        # sum (y-mu)^2/V(mu) / n maximises only the Gaussian likelihood; the first cut of this
        # fix used it for every family and the IG contract test caught the 0.1-nat gap).
        phi_mle = _dispersion_mle(fam, y, mu, w, phi)
        ll = _loglik(fam, y, mu, phi_mle, w)
    else:
        ll = _loglik(fam, y, mu, phi, w)
    if ll is not None and not np.isfinite(ll):
        raise RuntimeError("fit produced a non-finite log likelihood")
    return GLMResult(
        beta,
        se,
        mu,
        dev,
        phi,
        ll,
        n_iter,
        fam.name,
        lk.name,
        cov,
        converged=True,
        rank=rank,
        residual_df=int(np.count_nonzero(active)) - rank,
        dispersion_estimated=bool(fam.estimate_dispersion),
        _link=lk,
    )


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
    converged: bool = True
    rank: int | None = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict penalized-regression responses for design rows."""
        return _prediction_design(x, self.coef.size) @ self.coef + self.intercept


def ridge_regression(
    x: np.ndarray, y: np.ndarray, alpha: float = 1.0, *, fit_intercept: bool = True
) -> PenalizedResult:
    """Ridge (L2-penalised) linear regression in closed form.

    Minimises ``||y - X b||^2 + alpha ||b||^2``; the intercept (if fitted) is not penalised.
    """
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and >= 0")
    X, y = _regression_data(x, y)
    if fit_intercept:
        xm, ym = X.mean(axis=0), y.mean()
        Xc, yc = X - xm, y - ym
    else:
        Xc, yc, xm, ym = X, y, np.zeros(X.shape[1]), 0.0
    p = Xc.shape[1]
    beta = _solve_psd(Xc.T @ Xc + alpha * np.eye(p), Xc.T @ yc)
    intercept = float(ym - xm @ beta) if fit_intercept else 0.0
    return PenalizedResult(beta, intercept, alpha, 0.0, 0, converged=True, rank=int(np.linalg.matrix_rank(Xc)))


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
    ``l1_ratio = 1`` is the lasso (sparse); ``l1_ratio = 0`` is ridge-shaped, but note the
    ``1/(2n)`` data term means ``elastic_net(X, y, a, 0.0)`` equals ``ridge_regression(X, y, n*a)``
    -- the penalties differ by a factor of ``n`` when cross-walking alpha (audit G-5).
    """
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and >= 0")
    if not np.isfinite(l1_ratio) or not 0.0 <= l1_ratio <= 1.0:
        raise ValueError("l1_ratio must be in [0, 1]")
    _solver_controls(max_iter, tol)
    X, y = _regression_data(x, y)
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
    converged = False
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
            converged = True
            break
    if not converged:
        raise RuntimeError(f"elastic net failed to converge in {max_iter} iterations")
    intercept = float(ym - xm @ beta) if fit_intercept else 0.0
    return PenalizedResult(
        beta,
        intercept,
        alpha,
        l1_ratio,
        n_iter,
        converged=True,
        rank=int(np.linalg.matrix_rank(Xc)),
    )


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
    converged: bool = True
    rank: int | None = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict fitted regression values for design rows."""
        return _prediction_design(x, self.coef.size) @ self.coef


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
    _solver_controls(max_iter, tol)
    X, y = _regression_data(x, y)
    if method not in {"huber", "tukey"}:
        raise ValueError("method must be 'huber' or 'tukey'.")
    if c is None:
        c = 1.345 if method == "huber" else 4.685
    if not np.isfinite(c) or c <= 0:
        raise ValueError("c must be finite and > 0")
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    n_iter = 0
    scale = 1.0
    converged = False
    for n_iter in range(1, max_iter + 1):
        r = y - X @ beta
        scale = max(np.median(np.abs(r - np.median(r))) / 0.6745, 1e-8)
        u = r / scale
        if method == "huber":
            w = np.where(np.abs(u) <= c, 1.0, c / np.maximum(np.abs(u), 1e-12))
        elif method == "tukey":
            w = np.where(np.abs(u) <= c, (1.0 - (u / c) ** 2) ** 2, 0.0)
        if not np.any(w > 0):
            raise RuntimeError("robust regression assigned zero weight to every observation")
        XtW = X.T * w
        new = _solve_psd(XtW @ X, XtW @ y)
        if not np.all(np.isfinite(new)):
            raise RuntimeError("robust regression produced non-finite coefficients")
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            converged = True
            break
        beta = new
    if not converged:
        raise RuntimeError(f"robust regression failed to converge in {max_iter} iterations")
    return RegressionFit(beta, X @ beta, float(scale), n_iter, converged=True, rank=int(np.linalg.matrix_rank(X)))


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
    _solver_controls(max_iter, tol)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and > 0")
    X, y = _regression_data(x, y)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    n_iter = 0
    converged = False
    for n_iter in range(1, max_iter + 1):
        r = y - X @ beta
        w = np.where(r >= 0, tau, 1.0 - tau) / np.maximum(np.abs(r), eps)
        XtW = X.T * w
        new = _solve_psd(XtW @ X, XtW @ y)
        if not np.all(np.isfinite(new)):
            raise RuntimeError("quantile regression produced non-finite coefficients")
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            converged = True
            break
        beta = new
    if not converged:
        # IRLS can cycle around a non-smooth optimum. Resolve that case with the
        # exact linear-program formulation rather than returning the last iterate.
        n, p = X.shape
        objective = np.concatenate([np.zeros(p), np.full(n, tau), np.full(n, 1.0 - tau)])
        equality = sparse.hstack(
            [sparse.csr_matrix(X), sparse.eye(n, format="csr"), -sparse.eye(n, format="csr")],
            format="csr",
        )
        lp = optimize.linprog(
            objective,
            A_eq=equality,
            b_eq=y,
            bounds=[(None, None)] * p + [(0.0, None)] * (2 * n),
            method="highs",
        )
        if not lp.success or lp.x is None or not np.all(np.isfinite(lp.x[:p])):
            raise RuntimeError(f"quantile regression failed to converge: {lp.message}")
        beta = lp.x[:p]
        n_iter += int(getattr(lp, "nit", 0))
    return RegressionFit(
        beta,
        X @ beta,
        float(np.mean(np.abs(y - X @ beta))),
        n_iter,
        converged=True,
        rank=int(np.linalg.matrix_rank(X)),
    )


__all__ = [
    "Link",
    "Family",
    "GLMResult",
    "PerfectSeparationError",
    "glm",
    "PenalizedResult",
    "ridge_regression",
    "elastic_net",
    "lasso",
    "RegressionFit",
    "robust_regression",
    "quantile_regression",
]
