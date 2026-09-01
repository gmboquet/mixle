"""Generalized linear models and penalized / robust / quantile regression on plain arrays.

A array-level regression toolkit (operating on a design matrix ``X`` and response ``y``, independent of
the PPL DSL in :mod:`mixle.ppl.regression`):

  * :func:`glm` -- exponential-family GLMs by iteratively reweighted least squares, with explicit
    family/link objects (Gaussian, Binomial, Poisson, Gamma, inverse-Gaussian, negative-binomial),
    offsets, prior weights under a stated convention (``weight_type``: analytic precision weights
    by default, frequency counts on request), and optional sandwich (robust) standard errors in
    the HC0-HC3 leverage-corrected variants.
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

    Raised by :func:`glm` when IRLS runs the fitted binomial probabilities all the way onto 0 or 1
    and can no longer take a step -- the terminal form of (quasi-)complete separation, where
    ``|coef|`` diverges and the working weights vanish. A subclass of :class:`RuntimeError`, so
    pre-existing handlers keep working.

    Separation that stops the iteration just SHORT of that point is the same statistical condition
    and is reported too, but as a ``UserWarning`` plus ``GLMResult.separated``, because there the
    fit still carries a correct deviance / fitted / likelihood; only its Wald branch is undefined
    and refuses. Which of the two a given data set produces is a numerical accident, so never read
    the absence of this exception as evidence of no separation -- read ``separated``."""


_SEPARATION_SCREEN = 1e-4


def _binomial_separation(x: np.ndarray, y: np.ndarray) -> bool:
    """Does some direction in the design space separate the binomial classes?

    Separation -- complete or quasi-complete -- is exactly the condition under which no finite MLE
    exists: some ``b`` has ``x_i'b >= 0`` wherever ``y_i = 1`` and ``<= 0`` wherever ``y_i = 0``,
    strictly for at least one row, so the likelihood keeps rising as the coefficients run off along
    ``b`` (Silvapulle 1981; solved as a linear feasibility program after Konis 2007).

    It is deliberately a property of the DATA rather than of the iterate. Testing the fitted
    probabilities instead ties the answer to how the predictors happen to be coded: a two-point
    (dummy-coded) separated design stalls IRLS at ``min(mu) = 2.1e-11``, never exactly 0, so the
    saturation check in :func:`_mean_support` never fires, while a continuously coded separation of
    the SAME data saturates and does fire. Nor can a threshold repair that, because the measure
    does not order the two states: a perfectly identifiable logistic fit with a true slope of 8
    reaches ``min(mu) = 3.4e-11``, i.e. closer to the boundary than the separated design, so any
    cutoff on ``mu`` or ``|eta|`` loose enough to catch the separated case also rejects a
    legitimate one. The program answers the actual question and returns False for every
    identifiable design, however extreme its fitted probabilities.

    Rows with ``0 < y < 1`` (proportion responses) carry both classes, so they enter twice, in both
    directions -- a separating direction has to be orthogonal to them.
    """
    scale = np.max(np.abs(x), axis=0)
    scale[scale == 0.0] = 1.0
    columns = x / scale  # one common column scale keeps the margin tolerance below meaningful
    directed = columns * np.where(y >= 1.0, 1.0, -1.0)[:, None]
    mixed = (y > 0.0) & (y < 1.0)
    if np.any(mixed):
        directed = np.vstack([directed, -columns[mixed]])
    # separation depends only on the SET of directed rows, and the designs this matters most for
    # (dummy / factor codings, the shape separation actually arrives in) collapse to a handful
    directed = np.unique(directed, axis=0)
    solution = optimize.linprog(
        -directed.sum(axis=0),
        A_ub=-directed,
        b_ub=np.zeros(directed.shape[0]),
        bounds=[(-1.0, 1.0)] * x.shape[1],
        method="highs",
    )
    if not solution.success or solution.x is None:
        # an unsolved feasibility program is not evidence of separation: stay silent rather than
        # attach a statistical claim to a solver failure
        return False
    return bool(np.max(directed @ solution.x) > 1e-7)


def _mean_support(family: Family, mu: np.ndarray) -> None:
    if family.name == "binomial":
        # A monotone inverse link keeps mu inside [0, 1]; hitting the endpoints EXACTLY leaves IRLS
        # with no finite step to take, and only separation gets it there -- so name the statistical
        # condition, not the internal symptom. Separation that stalls just short of the endpoints
        # is caught after convergence by _binomial_separation, which does not depend on the coding.
        pinned = (mu == 0.0) | (mu == 1.0)
        if np.any(pinned):
            raise PerfectSeparationError(
                "perfect separation detected between the binomial classes: the design predicts "
                f"y exactly on {int(np.count_nonzero(pinned))} of {mu.size} observations (fitted "
                "probabilities reached exactly 0 or 1), so the coefficients diverge and no finite "
                "maximum-likelihood estimate exists. Remove or coarsen the separating "
                "predictor(s), pool sparse levels, or compare nested models by their deviance -- a "
                "likelihood-ratio test stays valid under separation where the Wald test does not. "
                "Note that ridge_regression / elastic_net are least-squares fits with no family "
                "argument, so they give a penalized LINEAR PROBABILITY model here, not penalized "
                "logistic regression."
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
        # name the menu: a typo ('guassian') used to cost a source dive to find the valid names
        raise ValueError(
            f"unknown family '{family}'; expected one of "
            f"{', '.join(sorted([*_FAMILIES, 'negativebinomial']))}, or a Family instance"
        )
    return _FAMILIES[family]


# --------------------------------------------------------------------------- GLM


@dataclass
class GLMResult:
    """Fitted GLM.

    Attributes:
        coef: ``(p,)`` coefficient estimates.
        se: ``(p,)`` standard errors (model-based, or the robust variant named by ``cov_type``).
            All NaN on a saturated dispersion-estimating fit -- see ``dispersion``.
        cov_type: which estimator ``se`` / ``cov`` came from: ``"model"`` for ``phi (X'WX)^-1``,
            or ``"HC0"`` | ``"HC1"`` | ``"HC2"`` | ``"HC3"`` for the sandwich variants (see
            :func:`glm`'s ``robust`` argument). Recorded on the result for the same reason
            ``weight_type`` is: the meaning of a headline output must be readable off the result.
        fitted: ``(n,)`` fitted means ``mu``.
        deviance: residual deviance.
        dispersion: estimated/assumed dispersion ``phi`` -- the ``residual_df``-corrected (Pearson)
            estimate the standard errors use, NOT the value the log-likelihood is evaluated at.
            NaN when a dispersion-estimating family is fit with ``residual_df <= 0`` (a SATURATED
            fit -- as many estimated parameters as observations): the coefficients, fitted values
            and deviance are still returned, but the dispersion, every standard error and the
            likelihood are undefined there, announced by a ``UserWarning`` at fit time, and
            ``z_values`` / ``p_values`` / ``aic`` / ``bic`` refuse with that reason.
        residual_df: residual degrees of freedom, ``n - rank``, where ``n`` counts rows under
            ``weight_type="analytic"`` and ``sum(weights)`` under ``weight_type="frequency"``.
            This is the divisor of ``dispersion`` and the t reference ``p_values`` uses, so it is
            where the weight convention shows up in the standard errors.
        log_likelihood: maximised log-likelihood, or ``None`` when the supplied
            family/response does not define one -- or when the fit is saturated (see
            ``dispersion``), where the likelihood is unbounded in the dispersion and so has no
            maximised value. For dispersion-estimating families this is
            evaluated at the family's own dispersion MLE (see ``_dispersion_mle``), which
            differs from ``dispersion`` by design: the covariance wants the df-corrected
            estimate, the likelihood wants its maximiser. It is also computed under
            ``weight_type``: analytic weights enter as per-observation precisions
            (``Var = phi / w_i``), frequency weights as replicate counts.
        ll_nobs: how many observations ``log_likelihood`` is a sum over -- the sample size ``bic``
            penalises with. Row count for an analytic-weighted dispersion-estimating family;
            ``sum(weights)`` whenever the likelihood is a weighted sum of per-unit densities (every
            frequency-weighted fit, and every fixed-dispersion family, whose discrete densities
            have no analytic-weight form). Pairing a likelihood over ``sum(w)`` units with a
            ``log(rows)`` penalty is what made BIC agree with neither convention.
        weight_type: the convention ``weights`` were read under; see :func:`glm`.
        separated: True when the binomial design separates the classes, so at least one coefficient
            has no finite maximum-likelihood estimate. ``deviance``, ``fitted`` and
            ``log_likelihood`` are still correct (compare nested models by deviance), but ``coef``
            and ``se`` along the separating direction record only where IRLS stopped, so
            ``z_values`` / ``p_values`` refuse. Announced by a ``UserWarning`` at fit time.
        n_iter: IRLS iterations to convergence.
        converged: whether the IRLS convergence criterion was met. Public fits
            currently raise instead of returning this as false.
        rank: effective NUMERICAL rank of the weighted design ``sqrt(W) X`` at the solution,
            from the same SVD the covariance is computed from (cutoff at ``cond(X)``, not
            ``cond(X)^2``; audit B4). ``rank < p`` -- exact or near collinearity -- is announced
            by a ``UserWarning`` at fit time and makes ``z_values`` / ``p_values`` refuse.
        family / link: names.
        cov: ``(p, p)`` coefficient covariance.

    Call forms: ``aic`` / ``bic`` are PROPERTIES (``fit.aic``, no parentheses) while ``z_values``
    / ``p_values`` are METHODS (``fit.z_values()``). Both call forms are frozen for
    compatibility -- changing either direction breaks every existing caller -- so the split is
    stated here instead: mixing them up fails with ``TypeError: 'float' object is not callable``
    (calling the property) or a bound-method object where an array was expected (reading the
    method), and neither traceback names the actual mistake. All four refuse with a stated
    reason (no likelihood, rank deficiency, separation, saturation) rather than return NaN.
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
    # the weight convention, and the sample size the reported likelihood actually spans -- BIC
    # needs the latter explicitly, because it is not always the residual-df sample size
    weight_type: str = "analytic"
    ll_nobs: float | None = None
    separated: bool = False
    # which covariance estimator `se`/`cov` came from -- recorded for the same reason weight_type
    # is: the meaning of a headline output must be readable off the result, not off a remembered
    # call argument
    cov_type: str = "model"
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

    def _no_likelihood_reason(self, criterion: str) -> str:
        # log_likelihood is None for two different reasons, and the refusal must name the right
        # one: "no defined likelihood" sent users of a saturated gaussian fit hunting through
        # the family/response combination when the actual problem was too few observations
        if self.dispersion_estimated and self.residual_df is not None and self.residual_df <= 0:
            return (
                f"{criterion} is unavailable: the fit is saturated (residual degrees of freedom "
                "are 0), so its likelihood is unbounded in the dispersion and has no maximised "
                "value. Add observations beyond the number of identifiable parameters."
            )
        return (
            f"{criterion} is unavailable because this family/response combination defines no "
            "likelihood (e.g. a binomial fit to proportion responses strictly between 0 and 1)"
        )

    @property
    def aic(self) -> float:
        """Akaike information criterion: ``-2 ll + 2 k`` with ``k = rank + estimated dispersion``."""
        if self.log_likelihood is None:
            raise ValueError(self._no_likelihood_reason("AIC"))
        return float(-2.0 * self.log_likelihood + 2.0 * self._n_parameters)

    @property
    def bic(self) -> float:
        """Bayesian information criterion over the observations the log-likelihood spans.

        The penalty has to count the SAME observations the likelihood does (``ll_nobs``). Deriving
        it from ``residual_df`` instead pairs a likelihood summed over ``sum(weights)`` replicates
        with a ``log(rows)`` penalty on a frequency table, which is correct under neither weight
        convention -- 107.7 where the two defensible answers were 116.5 and a likelihood of -3.5.
        """
        if self.log_likelihood is None:
            raise ValueError(self._no_likelihood_reason("BIC"))
        n_effective = self.ll_nobs if self.ll_nobs else self.fitted.size
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
                "reported minimum-norm split is arbitrary). Drop or combine collinear columns "
                "(or supply more rows than columns), or test an estimable linear combination "
                "instead."
            )
        # the same refusal for the same reason one step further out: under separation the
        # coefficient itself has no finite maximiser, so coef/se record where the iteration
        # stopped rather than an estimate and its sampling spread. The Wald statistic then
        # collapses toward zero as the coefficient grows (Hauck--Donner), which is why the
        # untreated fit reports "no effect" precisely where the data separate completely.
        if self.separated:
            raise ValueError(
                "per-coefficient Wald inference is undefined under perfect separation: at least "
                "one coefficient has no finite maximum-likelihood estimate, so its value and "
                "standard error are artifacts of where IRLS stopped and its Wald p-value tends to "
                "1 however strong the association is. Compare nested models by their deviance (a "
                "likelihood-ratio test), or drop / coarsen / pool the separating predictor(s); "
                "deviance, fitted and log_likelihood on this fit are unaffected."
            )
        # saturation: with residual_df <= 0 the estimated dispersion -- and with it every
        # standard error -- does not exist, so there is no Wald statistic to form. Only
        # dispersion-estimating families can get here; fixed-dispersion ones keep their
        # (dispersion-free) standard errors at residual_df == 0.
        if self.dispersion_estimated and self.residual_df is not None and self.residual_df <= 0:
            raise ValueError(
                "per-coefficient Wald inference is undefined on a saturated fit: residual "
                "degrees of freedom are 0, so the dispersion -- and with it every standard "
                "error -- cannot be estimated. Add observations beyond the number of "
                "identifiable parameters, or drop columns."
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


def _dispersion_mle(
    family: Family,
    y: np.ndarray,
    mu: np.ndarray,
    weights: np.ndarray,
    phi_fallback: float,
    *,
    frequency: bool,
) -> float:
    """The maximiser of this family's OWN log-likelihood over the dispersion, at fixed ``mu``.

    Each estimate solves d/d(phi) of exactly the log-density sum ``_loglik`` evaluates under the
    same convention -- so the reported "maximised log-likelihood" really is evaluated at its
    maximum. Arrays arrive already restricted to the positive-weight rows.

    The two conventions differ only in how many observations the sum spans. Under ``frequency`` a
    row stands for ``w`` replicates, so the divisor is ``sum(w)``; under analytic weights it is one
    observation of precision ``w``, so the divisor is the row count and the weights stay inside the
    sum. With unit weights the two coincide, which is why an unweighted fit is untouched.

    * gaussian: ``sum w (y - mu)^2`` over that divisor (the RSS form).
    * inverse_gaussian: ``sum w (y - mu)^2 / (y mu^2)`` over it -- the mean unit deviance. The
      Pearson form divides by ``V(mu) = mu^3`` instead of ``y mu^2`` and does NOT maximise the IG
      density (measured 0.1 nats short on an 8-point fit).
    * gamma: no closed form. Writing ``d_i = y_i/mu_i - log(y_i/mu_i) - 1`` (half the unit
      deviance) and ``nu_i`` for the shape, the score is ``sum nu_i (log nu_i - digamma nu_i - d_i)
      = 0``. Under frequency weights every observation shares ``nu = 1/phi``, leaving
      ``log(nu) - digamma(nu) = mean(d)``; under analytic weights ``nu_i = w_i/phi`` differs per
      observation, and clearing the ``1/phi`` leaves ``sum w_i (log nu_i - digamma nu_i) = sum w_i
      d_i``. ``log(nu) - digamma(nu)`` falls strictly from +inf to 0, so both are strictly monotone
      in ``phi`` with a unique root, bracketed in log-space. A numerically perfect fit (all
      ``d_i ~ 0``) sends ``phi -> 0`` and the likelihood to +inf; the covariance-phi fallback is
      returned there rather than a fake maximum.
    """
    divisor = float(np.sum(weights)) if frequency else float(weights.size)
    if family.name == "gaussian":
        return float(np.sum(weights * (y - mu) ** 2) / divisor)
    if family.name == "inverse_gaussian":
        return float(np.sum(weights * (y - mu) ** 2 / (y * mu**2)) / divisor)
    if family.name == "gamma":
        deficit = float(np.sum(weights * (y / mu - np.log(y / mu) - 1.0)))
        if not np.isfinite(deficit) or deficit <= 1e-12 * divisor:
            return phi_fallback
        if frequency:

            def gap(log_nu: float) -> float:
                return log_nu - float(special.digamma(np.exp(log_nu))) - deficit / divisor

            return float(np.exp(-optimize.brentq(gap, -30.0, 30.0, xtol=1e-12)))

        def analytic_gap(log_phi: float) -> float:
            # phi * (score = sum nu_i (log nu_i - digamma nu_i - d_i)) with nu_i = w_i / phi, so
            # the leading factor is w_i and the data enter only through the phi-free deficit
            shape = weights * np.exp(-log_phi)
            return float(np.sum(weights * (np.log(shape) - special.digamma(shape)))) - deficit

        return float(np.exp(optimize.brentq(analytic_gap, -30.0, 30.0, xtol=1e-12)))
    return phi_fallback


def _loglik(
    family: Family, y: np.ndarray, mu: np.ndarray, phi: float, weights: np.ndarray, *, frequency: bool
) -> float | None:
    """Log-likelihood under the stated weight convention, over the positive-weight rows.

    Discrete families have no analytic-weight density -- there is no exponential-dispersion form of
    a Poisson or Bernoulli observation "measured with precision w" -- so their weighted likelihood
    is ``sum w log f`` under both conventions, and the convention shows up only in the sample size
    (``GLMResult.ll_nobs``). The dispersion-estimating families do distinguish: a frequency weight
    replicates a log-density, an analytic weight divides that observation's dispersion.
    """
    name = family.name
    if name == "binomial":
        if np.any((y != 0.0) & (y != 1.0)):
            return None
        m = _clip01(mu)
        return float(np.sum(weights * (y * np.log(m) + (1 - y) * np.log(1 - m))))
    if name == "poisson":
        return float(np.sum(weights * stats.poisson.logpmf(y, mu)))
    if name == "negativebinomial":
        theta = family.extra
        return float(np.sum(weights * stats.nbinom.logpmf(y, theta, theta / (theta + mu))))
    scale = phi if frequency else phi / weights
    if name == "gaussian":
        log_density = stats.norm.logpdf(y, mu, np.sqrt(scale))
    elif name == "gamma":
        log_density = stats.gamma.logpdf(y, 1.0 / scale, scale=mu * scale)
    elif name == "inverse_gaussian":
        log_density = -0.5 * (np.log(2.0 * np.pi * scale) + 3.0 * np.log(y) + (y - mu) ** 2 / (scale * y * mu**2))
    else:
        return None
    return float(np.sum(weights * log_density)) if frequency else float(np.sum(log_density))


def glm(
    x: np.ndarray,
    y: np.ndarray,
    *,
    family: str | Family = "gaussian",
    link: str | Link | None = None,
    offset: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    weight_type: str = "analytic",
    max_iter: int = 100,
    tol: float = 1e-8,
    robust: bool | str = False,
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
        weights: ``(n,)`` non-negative prior weights, at least one positive. Zero-weight rows drop
            out of the fit entirely. What a weight MEANS is set by ``weight_type`` -- the two
            readings agree on the coefficients and on nothing else.
        weight_type: the convention ``weights`` are read under.

            * ``"analytic"`` (default; also called prior, precision or variance weights -- the
              McCullagh--Nelder reading, and R's for ``glm``): row ``i`` is ONE observation measured
              with precision ``w_i``, so ``Var(y_i) = phi V(mu_i) / w_i``. The sample size is the
              number of rows: ``residual_df = rows - rank``, ``dispersion`` divides by that, and the
              log-likelihood is that of ``n`` independent observations with dispersions ``phi/w_i``.
              Where the dispersion is ESTIMATED, ``phi`` then absorbs the weights' scale, so
              multiplying every weight by a constant leaves the whole fit unchanged --
              coefficients, standard errors, log-likelihood, AIC and BIC alike -- and only the
              relative precisions matter. (Not so where ``phi`` is fixed at 1: for binomial /
              Poisson / negative-binomial the weight scale IS the amount of information, and
              halving every weight widens the standard errors by ``sqrt(2)``.) Use this for
              inverse-variance weighting, measurement precisions or sampling weights; fractional
              weights are accepted.
            * ``"frequency"``: ``w_i`` is a COUNT -- row ``i`` stands for ``w_i`` identical
              observations -- so it must be a whole number. Every reported quantity then equals
              what fitting the expanded ``sum(w)``-row data set gives: ``residual_df = sum(w) -
              rank``, ``dispersion`` divides by that, ``robust`` standard errors replicate the
              score, and AIC/BIC span ``sum(w)`` observations. Use it for aggregated frequency
              tables -- survey cells, contingency tables, "distinct rows plus a count" extracts.

            The gap is not cosmetic. On a 6-row table whose counts sum to 114, the same call gives
            ``se = [0.369, 0.135]`` analytically and ``[0.070, 0.025]`` as frequencies -- a factor
            ``sqrt((114 - 2) / (6 - 2)) = 5.29`` -- with p-values of 5e-04 against 6e-82. Fixed
            dispersion families (binomial, Poisson, negative-binomial) return identical
            coefficients AND standard errors either way, because ``phi`` is never estimated there;
            only ``residual_df`` and the BIC sample size move. Their likelihood is ``sum w log f``
            under both conventions (a discrete density has no analytic-weight form), so ``ll_nobs``
            is ``sum(w)`` for them regardless -- see ``GLMResult.ll_nobs``.
        max_iter, tol: IRLS controls (convergence on the relative deviance change).
        robust: which coefficient-covariance estimator to report (recorded on the result as
            ``cov_type``). ``False`` -- the default -- is the model-based ``phi (X'WX)^-1``;
            ``True`` or ``"HC0"`` is the Huber--White sandwich; ``"HC1"`` | ``"HC2"`` | ``"HC3"``
            apply the standard finite-sample leverage corrections to the same
            PSD-by-construction sandwich (MacKinnon--White 1985):

            * HC0: the plain sandwich. No finite-sample correction, so on small or leveraged
              designs it is biased LOW -- measured 16-51% below HC3 on a 12-point design with one
              leveraged row, i.e. exactly where robust standard errors are reached for.
              (``robust=True`` keeps meaning HC0 for compatibility.)
            * HC1: HC0 scaled by ``n / (n - rank)`` -- the df correction Stata's ``robust``
              applies. Fixes the average small-sample bias but ignores WHERE the leverage sits.
            * HC2: per-observation scores scaled by ``1 / sqrt(1 - h_i)``; exactly unbiased under
              homoskedasticity. Use when leverage is uneven but n is moderate.
            * HC3: scores scaled by ``1 / (1 - h_i)`` (jackknife-like) -- the standard
              recommendation for ``n <~ 50`` or clearly leveraged designs (Long--Ervin 2000);
              the most conservative of the four.

            All four assume INDEPENDENT observations (none is cluster-robust) and still require
            the mean/link to be correct (audit G-7). ``h_i`` is the observation's leverage in the
            converged working weighted regression; under ``weight_type="frequency"`` each
            replicate carries its own leverage, so every estimator equals the expanded data set's.
            HC1-3 are refused on a saturated fit (their corrections divide by zero there), and
            HC2/HC3 on a unit-leverage observation (one fitted by itself alone).
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

    Raises:
        PerfectSeparationError: a binomial fit whose fitted probabilities reached exactly 0 or 1,
            leaving IRLS no finite step. Separation that stalls just short of that returns instead,
            with ``GLMResult.separated`` set, a ``UserWarning``, and ``z_values`` / ``p_values``
            refusing -- so treat ``separated``, not the absence of this exception, as the answer to
            "were my classes separated?" (the exception fires on a continuously coded separation
            and not on the dummy-coded form of the very same data).
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
            # same precision as the family message: the menu is short, so print it
            raise ValueError(
                f"unknown link '{link_name}'; expected one of {', '.join(sorted(_LINKS))}, or a Link instance"
            )
        lk = _LINKS[link_name]
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if off.ndim != 1 or off.shape[0] != n or not np.all(np.isfinite(off)):
        raise ValueError("offset must be a finite one-dimensional array aligned with y")
    if w.ndim != 1 or w.shape[0] != n or not np.all(np.isfinite(w)):
        raise ValueError("weights must be a finite one-dimensional array aligned with y")
    if np.any(w < 0) or not np.any(w > 0):
        raise ValueError("weights must be non-negative with at least one positive value")
    if weight_type not in {"analytic", "frequency"}:
        raise ValueError("weight_type must be 'analytic' (precision weights) or 'frequency' (counts)")
    frequency = weight_type == "frequency"
    if frequency and np.any(w != np.floor(w)):
        # a fractional count has no expanded data set to be equivalent to, and would leave
        # residual_df non-integral; the analytic reading is what fractional weights mean
        raise ValueError(
            "weight_type='frequency' reads weights as observation COUNTS, so they must be whole "
            "numbers; pass weight_type='analytic' for fractional precision / inverse-variance weights"
        )
    if isinstance(robust, str):
        # strings used to be read as a truthy bool, so robust="HC3" SILENTLY computed HC0 --
        # the named estimators must dispatch, and a name outside the menu must refuse
        cov_estimator = robust.upper()
        if cov_estimator not in {"HC0", "HC1", "HC2", "HC3"}:
            raise ValueError(
                "robust must be False (model-based covariance), True (HC0), or one of "
                f"'HC0' | 'HC1' | 'HC2' | 'HC3'; got {robust!r}"
            )
    else:
        # non-string truthiness keeps its historical meaning so robust=1 callers are untouched
        cov_estimator = "HC0" if robust else None
    active = w > 0
    # the sample size the residual df, dispersion and t reference are all taken over: rows under
    # analytic weights (one observation apiece), replicates under frequency weights
    n_units = float(np.sum(w)) if frequency else float(np.count_nonzero(active))

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

    separated = False
    if fam.name == "binomial":
        # Deviance convergence does NOT mean the coefficients converged: under separation the
        # deviance flattens while beta keeps marching, and IRLS stops wherever tol happens to
        # bite. Screen on how close the fit came to the boundary (widened for a loose tol, which
        # stops the march earlier), then let the coding-independent feasibility program decide --
        # the screen alone cannot, since identifiable fits reach the boundary just as closely.
        boundary = np.minimum(mu[active], 1.0 - mu[active])
        if boundary.size and float(np.min(boundary)) < max(_SEPARATION_SCREEN, float(np.sqrt(tol))):
            separated = _binomial_separation(X[active], y[active])
        if separated:
            warnings.warn(
                "perfect separation detected between the binomial classes: some direction in the "
                f"design separates them ({int(np.count_nonzero(boundary < 1e-8))} of "
                f"{int(boundary.size)} fitted probabilities are within 1e-8 of 0 or 1), so at "
                "least one coefficient has no finite maximum-likelihood estimate and IRLS stopped "
                "at an arbitrary point along it. coef and se there record only where it stopped, "
                "so z_values / p_values refuse (their Wald p-value would tend to 1 however strong "
                "the association is); deviance, fitted and log_likelihood are unaffected, so "
                "compare nested models by their deviance instead. Drop, coarsen or pool the "
                "separating predictor(s) to recover finite coefficients.",
                UserWarning,
                stacklevel=2,
            )

    dmu = np.asarray(lk.mu_eta(eta), dtype=float)
    var = np.asarray(fam.variance(mu), dtype=float)
    wls_w = w * dmu**2 / var
    # audit B4: the coefficient covariance comes from an SVD of the WEIGHTED DESIGN sqrt(W)X,
    # never from pinv(X'WX). cond(X'WX) = cond(sqrt(W)X)^2, so pinv's default cutoff silently
    # truncated singular values once cond(X) passed ~1e8 and collapsed the standard errors by
    # up to 8 orders of magnitude -- with rank reported full, converged=True, and no warning.
    # Factorizing sqrt(W)X keeps rank and cutoff decisions at cond(X); zero-weight rows enter
    # as zero rows, so this is still the effective rank among positive-weight observations.
    u, singular, vt = np.linalg.svd(X * np.sqrt(wls_w)[:, None], full_matrices=False)
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
            "columns, so some predictors are exactly or nearly collinear (with fewer rows than "
            "columns this is automatic). Coefficients are the "
            "minimum-norm least-squares representative and per-coefficient Wald inference "
            "(z_values / p_values) will refuse; drop, combine, or center/rescale the collinear "
            "columns to make individual coefficients identifiable.",
            UserWarning,
            stacklevel=2,
        )
    dev = float(np.sum(w * fam.unit_deviance(y, mu)))
    residual_df = int(round(n_units)) - rank
    # A dispersion-estimating family with residual_df <= 0 is SATURATED: the model interpolates
    # the observations, so the dispersion (and with it every standard error and the maximised
    # likelihood, which is unbounded as phi -> 0) genuinely cannot be computed -- but the
    # coefficients, fitted values and deviance still can. Refusing the whole fit here (the old
    # behavior) also refused those, while the sibling degeneracy (rank deficiency with
    # residual_df > 0) already returns the computable part and refuses only inference; route
    # saturation through the same disclose-and-refuse machinery instead of a blanket raise.
    saturated = fam.estimate_dispersion and residual_df <= 0
    if saturated:
        phi = float("nan")
        warnings.warn(
            f"the {fam.name} fit is saturated: it estimates as many parameters (rank {rank}) as "
            f"it has observations ({int(round(n_units))}), so residual degrees of freedom are 0 "
            "and the dispersion cannot be estimated. Coefficients, fitted values and deviance "
            "are returned; dispersion and standard errors are NaN (robust ones included: every "
            "residual is numerically zero, so a sandwich would claim exact knowledge), the "
            "log-likelihood is None (it is unbounded in the dispersion at an interpolating fit), "
            "and z_values / p_values / aic / bic refuse. Add observations beyond the number of "
            "identifiable parameters, or drop columns, to recover inference.",
            UserWarning,
            stacklevel=2,
        )
    elif fam.estimate_dispersion:
        phi = float(np.sum(w * (y - mu) ** 2 / var) / residual_df)
    else:
        phi = 1.0
    if not saturated and (not np.isfinite(phi) or phi <= 0):
        raise RuntimeError("fit produced an invalid dispersion estimate")
    if saturated:
        # no covariance of any kind is defined at an interpolating fit -- NaN, not 0 or a raise
        cov = np.full((p, p), np.nan)
    elif cov_estimator is not None:
        # per-observation score x_i (y-mu) (dmu/deta) / V(mu) scaled by the weight; sandwich
        # B (sum gg') B, formed as (S B)' (S B) so it is positive semidefinite BY CONSTRUCTION. The
        # algebraically equal B (S'S) B loses PSD-ness to cancellation at cond(X)^2: OpenBLAS
        # produced negative sandwich diagonals on a cond ~2e8 design (clipped to se = 0.0 -- a false
        # claim of EXACT knowledge of coefficients the design cannot identify) where Accelerate
        # stayed positive, so the same wheel reported different answers per platform (wave-3 CI).
        # The weight enters the meat differently per convention, which is not a free choice: a
        # frequency weight REPLICATES a score (meat sum w g g', matching the expanded data set),
        # an analytic weight SCALES one (meat sum w^2 g g', HC0 for the weighted estimator).
        if cov_estimator != "HC0" and residual_df <= 0:
            # only reachable for fixed-dispersion families (dispersion-estimating ones took the
            # saturated branch above): n/(n - rank) and 1/(1 - h) both divide by zero here, so
            # no finite-sample-corrected answer exists -- HC0 and the model covariance still do
            raise ValueError(
                f"{cov_estimator} is undefined on a saturated fit (residual degrees of freedom "
                "= 0): its finite-sample correction divides by zero. Use robust=True/'HC0' or "
                "the model-based covariance, or add observations beyond the number of "
                "identifiable parameters."
            )
        unit_score = X * ((y - mu) * dmu / var)[:, None]
        row_scale = np.sqrt(w) if frequency else w
        if cov_estimator in {"HC2", "HC3"}:
            # h_i is the observation's leverage in the CONVERGED working weighted regression --
            # the hat diagonal of sqrt(W_wls) X. An analytic-weighted row is one observation, so
            # its leverage carries its full working weight; a frequency-weighted row stands for w
            # replicates that each carry the per-replicate weight dmu^2/var, so the correction
            # (like the meat) matches the expanded data set. Folding 1/(1-h)^k into the row scale
            # keeps the (S B)'(S B) PSD-by-construction form.
            #
            # audit B4 (again): the hat diagonal of sqrt(W_wls) X is diag(U U') where U is the
            # left singular vectors already factored above -- read straight off U, never rebuilt
            # via the quadratic form X @ xtwx_inv @ X.T. xtwx_inv is built from 1/singular**2,
            # which spans ~cond(X)^2 for an ill-conditioned design; squaring back through it in a
            # quadratic form suffers the same catastrophic cancellation X'WX itself was rejected
            # for, producing "leverage" outside [0, 1] (observed: max 4.11, sum 24.6 for a
            # rank-4 design, instead of every h_i in [0, 1] summing to exactly 4). diag(U U') is
            # bounded in [0, 1] and sums to the rank by construction, at cond(X) precision. U's
            # insignificant columns (beyond `rank`) must be masked out exactly like xtwx_inv
            # masks them via inv_sq_singular -- otherwise a rank-deficient design (collinear or
            # duplicated columns, not just an ill-conditioned one) sums the hat diagonal to p
            # instead of rank, reintroducing spurious leverage-near-1 false positives.
            hat_diag = np.sum((u * significant[None, :]) ** 2, axis=1)
            if frequency:
                # hat_diag is the leverage of the whole replicate-count row (weight w * dmu^2/var);
                # dividing out w recovers the per-replicate figure described above
                per_replicate = np.zeros_like(hat_diag)
                np.divide(hat_diag, w, out=per_replicate, where=active)
                leverage = np.where(active, np.clip(per_replicate, 0.0, None), 0.0)
            else:
                leverage = np.where(active, np.clip(hat_diag, 0.0, None), 0.0)
            degenerate = leverage > 1.0 - 1e-10
            if np.any(degenerate):
                raise ValueError(
                    f"{cov_estimator} is undefined here: {int(np.count_nonzero(degenerate))} "
                    "observation(s) have leverage 1 in the working regression (each is fitted by "
                    "itself alone, e.g. a dummy level with a single observation), so the "
                    "1/(1 - h) leverage correction divides by zero. Pool the singleton level(s) "
                    "or use 'HC0'/'HC1'."
                )
            row_scale = row_scale / (1.0 - leverage) ** (0.5 if cov_estimator == "HC2" else 1.0)
        half = (row_scale[:, None] * unit_score) @ xtwx_inv
        cov = half.T @ half
        if cov_estimator == "HC1":
            # the df rescaling counts the same units residual_df does: rows under analytic
            # weights, replicates under frequency weights
            cov = cov * (n_units / (n_units - rank))
    else:
        cov = phi * xtwx_inv
    if not saturated and not np.all(np.isfinite(cov)):
        raise RuntimeError("fit produced a non-finite covariance matrix")
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    # zero-weight rows are absent from the fit, so they are absent from the likelihood too -- and
    # under analytic weights their dispersion phi/0 is not even defined
    y_fit, mu_fit, w_fit = y[active], mu[active], w[active]
    if saturated:
        # the profile likelihood of an interpolating fit rises without bound as phi -> 0: there
        # is no maximised value to report, which is a different fact from "this family defines
        # no likelihood" and is reported as such by aic / bic
        ll = None
    elif fam.estimate_dispersion:
        # audit G-2: "maximised log-likelihood" must be evaluated at the MLE of the dispersion,
        # not the residual-df-corrected phi used for the covariance -- the mixed convention
        # understated ll and, with the uncounted dispersion parameter, distorted AIC/BIC model
        # selection toward larger mean models. The MLE is FAMILY-SPECIFIC (the Pearson-form
        # sum (y-mu)^2/V(mu) / n maximises only the Gaussian likelihood; the first cut of this
        # fix used it for every family and the IG contract test caught the 0.1-nat gap).
        phi_mle = _dispersion_mle(fam, y_fit, mu_fit, w_fit, phi, frequency=frequency)
        ll = _loglik(fam, y_fit, mu_fit, phi_mle, w_fit, frequency=frequency)
    else:
        ll = _loglik(fam, y_fit, mu_fit, phi, w_fit, frequency=frequency)
    if ll is not None and not np.isfinite(ll):
        raise RuntimeError("fit produced a non-finite log likelihood")
    return GLMResult(
        coef=beta,
        se=se,
        fitted=mu,
        deviance=dev,
        dispersion=phi,
        log_likelihood=ll,
        n_iter=n_iter,
        family=fam.name,
        link=lk.name,
        cov=cov,
        converged=True,
        rank=rank,
        residual_df=residual_df,
        dispersion_estimated=bool(fam.estimate_dispersion),
        weight_type=weight_type,
        # BIC must penalise the observations the LIKELIHOOD spans, which is the residual-df sample
        # size only when the likelihood is per-row: a fixed-dispersion family's sum w log f counts
        # sum(w) units whichever convention the standard errors follow
        ll_nobs=n_units if fam.estimate_dispersion else float(np.sum(w)),
        separated=separated,
        cov_type="model" if cov_estimator is None else cov_estimator,
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
    """Coefficients + fitted values from :func:`robust_regression` / :func:`quantile_regression`.

    Attributes:
        degenerate_scale: set by :func:`robust_regression` when it detects a sign of its own >50%
            breakdown point -- EITHER the IRLS scale has collapsed (at or within a convergence step
            of its numerical floor, or far below the raw response's own spread) with a real share
            of rows assigned (near) zero weight as a result, OR at least ~44% of the raw response
            is tied (near enough) to a single value regardless of the fit's residuals (the shape a
            capped/censored/zero-inflated measurement produces, which need not collapse the scale
            at all; see the module comment above ``_robust_weight_collapse`` in this file for the
            full design and, just as importantly, what it still cannot see). Either signal firing
            means this is not a transient numerical accident, and ``coef`` MAY reflect a majority
            pattern rather than a fully recovered relationship -- but may equally reflect a fit
            that is completely correct, with the flagged pattern arising because the majority
            genuinely fits (near-)exactly; see :func:`robust_regression`. This field, and the
            ``UserWarning`` announcing it at fit time, name the observed CONDITION only, never a
            verdict: neither can establish whether a real minority signal was discarded or a
            genuine minority of contamination was correctly rejected (both look identical from this
            pattern alone), and -- a further, structural limitation, not merely an unresolved
            ambiguity -- a majority and minority that follow two genuinely different relationships
            (as opposed to sharing one tied value) is largely invisible to this mechanism in the
            first place unless the majority's own fit is exceptionally tight; see the module comment
            for exactly how far that reaches and where it stops. Always ``False`` for
            :func:`quantile_regression`, which has no comparable mechanism.
    """

    coef: np.ndarray
    fitted: np.ndarray
    scale: float
    n_iter: int
    converged: bool = True
    rank: int | None = None
    degenerate_scale: bool = False

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict fitted regression values for design rows."""
        return _prediction_design(x, self.coef.size) @ self.coef


# audit R-1 (04ef6ae1): median(|r - median(r)|) -- the numerator of the scale estimate in
# robust_regression's IRLS loop below -- is EXACTLY zero whenever half or more of the rows share
# (near enough) the same residual under the current fit, which floors the whole scale estimate at
# _ROBUST_SCALE_FLOOR. That is a genuine, inherent limit of any MAD-scaled M-estimator (the classic
# >50% breakdown point; ordinary zero-inflated or capped/rounded data reaches it easily, no
# adversarial construction needed) and is not something a fix should try to overcome -- but every
# OTHER degenerate branch this module recognizes (separation, saturation, rank deficiency,
# non-convergence) already discloses its own failure, and this one previously did not.
#
# audit R-2 (743a2185): an independent adversarial review of R-1's fix, by direct execution against
# it, found real gaps in the two-part heuristic it shipped (scale bit-exactly AT _ROBUST_SCALE_FLOOR,
# AND >=5% of weights near zero). Detection became three signals, not one combined check -- the first
# two of which are unchanged by audit R-3 below and remain exactly as R-2 shipped them:
#
#  1. _robust_weight_collapse -- R-1's own idea, repaired. "Floored" is now a NEAR-floor band sized
#     off the caller's OWN `tol` instead of bit-exact equality: `tol` and _ROBUST_SCALE_FLOOR happen
#     to share the same 1e-8 default, so IRLS's convergence check (on `beta`, not on the scale) could
#     stop moving before the scale finished settling onto the floor, resting a `tol`-sized nudge
#     above it -- already just as degenerate as sitting exactly on it, but invisible to a bit-exact
#     comparison. Measured directly on R-1's own 60-seed target reproduction (35%-minority
#     zero-inflated design, method='huber', default max_iter=100): the true coefficients were
#     discarded in 60/60 seeds, but only 38/60 were flagged, because 22 converged with scale
#     strictly above the floor (by 0.04-0.86 times `tol`, always under 1x). The crushed-weight
#     fraction required also drops, from 5% to 1%: see _ROBUST_DEGENERACY_FRACTION for why that is
#     safe rather than arbitrary. Audit R-3 below ADDS a second, relative way for the scale part of
#     this check to fire; the crushed-weight part is untouched.
#  2. _robust_response_point_mass -- new in R-2, and fit-INDEPENDENT: read directly off the raw
#     response before IRLS ever runs, rather than waiting for a converged fit's residuals to (maybe)
#     show the same pattern. Two holes this closes that signal 1 cannot, by construction:
#       - A minority's crushed-weight fraction and its share of the raw data are not always the same
#         number. A 1%-minority single-column reproduction crushes exactly 1.00% of rows (matching
#         signal 1's own new threshold exactly), but the two-column version of the same 1% minority
#         was found, by direct execution, to crush as little as 0.67% -- under signal 1's threshold
#         even at its new, lowered value -- while the response itself is 99.33% tied, nowhere near
#         ITS threshold. No single crushed-weight cutoff covers both; reading the response directly
#         does not need one.
#       - Capped/censored measurements (named in R-1's own docstring as in scope, alongside
#         zero-inflation) tie rows' raw Y values to a shared ceiling but do NOT, in general, tie
#         their RESIDUALS once the fit carries any slope: the ceiling is one constant, but a capped
#         row's fitted prediction still varies with that row's X, so ceiling-minus-a-varying-
#         prediction is not one repeated number. Confirmed by direct execution on a
#         covariate-dependent design right-censored for 70% of rows: under method='huber' the fitted
#         slope collapses by over 50% relative, yet the raw MAD sits at ~0.9-1.2 throughout (nowhere
#         near _ROBUST_SCALE_FLOOR) and not one row is ever crushed to near-zero weight -- signal 1
#         is structurally blind here, at ANY threshold, because huber's soft down-weighting never
#         fully lets go of the uncensored minority's slope information. method='tukey' on the exact
#         same data instead collapses the slope completely (to bit-level zero) and, as a side effect
#         of collapsing that far, DOES also drive the scale to its floor with ~30% of rows crushed --
#         so signal 1 happens to fire there too, but only incidentally, and only because tukey chose
#         to collapse all the way; a method that stopped at a partial, huber-like collapse would
#         still slip past it regardless of threshold. The raw response, sitting 70% tied at the
#         ceiling regardless of method or how far the fit collapsed, is the one signal both share.
#  3. (R-2 only; REMOVED by audit R-3 -- kept here as attributed history, not current behavior)
#     _robust_diverges_from_ols was never a third detector: R-2 added it as a SUPPRESSOR of signal 1
#     alone, to avoid warning on a benign heteroskedastic fit (a noiseless majority, a noisier
#     minority of the SAME relationship) where signal 1's scale+weight condition is genuinely
#     satisfied even though nothing is discarded. See audit R-3 immediately below for why this was
#     removed rather than repaired.
#
# audit R-3 (this commit): a second independent adversarial review of R-2's fix, again by direct
# execution against the actual shipped code, found two further problems. This is a third design pass,
# not a patch on the second -- one mechanism is removed outright rather than re-tuned:
#
#  GAP A (blocking) -- signal 3 (_robust_diverges_from_ols) is REMOVED entirely, not repaired, for
#     two independently-confirmed reasons:
#       - Structural blindness signal 3 made WORSE, not better. Every signal here reasons about a
#         majority TIED to one value or one near-exact residual; NONE can see a majority following
#         one real (e.g. sloped) relationship while a minority follows a genuinely different one --
#         an entirely ordinary reason to reach for robust regression in the first place (pooling
#         across instruments/conditions/regimes with different slopes), and qualitatively different
#         from a value tie. Reproduced directly: a two-column design with a majority at slope 3 and
#         noise sigma=0.05 and a minority at 2x-4x that slope, swept across minority fractions
#         1%-95%, method='tukey' converges to the MAJORITY's relationship to within 1e-3 absolute
#         error at every one of 12 tested (fraction, slope-ratio) combinations from 1% to 20%
#         minority, discarding the minority's real, different relationship completely -- and
#         degenerate_scale was False in all 12, because sigma=0.05 never comes close to flooring the
#         scale (signal 1's scale condition never fires in the first place; signal 3 is not even
#         reached). Signal 3 cannot help this half of the gap by construction -- it only ever
#         suppresses signal 1, and signal 1 was never triggered here to suppress.
#       - Worse: in the narrow window where signal 1's scale condition DOES fire on its own (majority
#         noise small enough to floor the scale, e.g. sigma=1e-9), signal 3 actively suppresses that
#         CORRECT detection at ordinary parameters. Reproduced directly on the same sloped-mixture
#         design at sigma=1e-9: signal 1 alone fires in every one of 12 (fraction x slope-ratio x
#         method) combinations tested, but signal 3's <10% OLS-agreement bar suppresses it at
#         minority_frac=5%/slope_ratio=2x (OLS disagreement 4.81%, both methods) and
#         minority_frac=10%/slope_ratio=2x (9.77%, both methods) -- ordinary fractions and a
#         straight, not-subtle 2x slope difference -- turning a correct signal-1-alone detection back
#         into silent, undisclosed complete discard. This is a direct regression relative to what
#         signal 1 ALONE (R-2's own repair, minus R-2's own suppressor) would have caught.
#     R-2's original motivation for signal 3 -- not warning on the benign heteroskedastic fixture,
#     where signal 1's condition is genuinely satisfied but nothing was actually discarded -- is
#     real, but is now addressed by HONEST WORDING instead of suppression: see
#     _robust_breakdown_explanation below. Signal 1 firing states the observed weight/scale
#     CONDITION; it never again asserts that a real minority signal was necessarily discarded, so a
#     caller who already knows their majority is exactly (or near-exactly) correct by design is not
#     given a false alarm, while a caller who does not expect this is told plainly what was observed
#     and left to judge it -- rather than having the detection hidden from them either way.
#
#     Separately investigated per the same gap: whether signal 1's scale condition could be made
#     scale-RELATIVE as well as absolute, to catch a sloped majority whose own noise is "very precise
#     but not exactly exact" (sigma=1e-6, 1e-4 -- too big for the hardcoded 1e-8 floor's `tol`-sized
#     band, however that band is sized, yet just as completely a discard of the minority). This DOES
#     have a well-reasoned, validated answer for part of the gap: _robust_weight_collapse's scale
#     check now ALSO fires when the scale is far smaller than the raw response's own robust spread
#     (_response_robust_scale), not only when it is near the hardcoded absolute floor -- full
#     reasoning, validation, and the part that remains OUT of reach even with this change, are at
#     _ROBUST_RELATIVE_SCALE_FLOOR below.
#
#  GAP B (major) -- _RESPONSE_POINT_MASS_FRACTION's hard 50% cutoff does not track where fit quality
#     for the shape signal 2 targets (a tied response value) actually degrades. My own sweep (not the
#     reviewer's numbers re-quoted, re-run independently against this module's own two-column
#     zero-inflated fixture family, method='huber', default max_iter=100, 200 seeds per point) found a
#     smooth, roughly symmetric SEVERE-degradation band from response-tie-fraction ~44% through ~56%:
#     mean/median relative coefficient error there ranges 0.36-0.90 (worse, in fact, at some points
#     below 50% than just above it), and this band is not an artifact of one fixture's noise level or
#     effect size -- re-run across noise sigma 0.02-1.0, n=600-6000, single- and two-column designs,
#     and minority separations from 2 to 10 true SDs, EVERY configuration shows severe or unreliably
#     bimodal error (median deceptively good in one low-noise case, but mean dragged to 0.21 by a
#     catastrophic-failure tail) throughout that same band. At the exact 50% point itself: 51.0% of
#     200 seeds are flagged, with corr(realized-tie-fraction, flagged) = 0.82 but corr(actual relative
#     error, flagged) = only 0.34 -- confirming the reviewer's framing that at the exact cutoff,
#     whether a given seed is flagged is driven mostly by which way its own finite-sample Bernoulli
#     draw landed, not by how bad that seed's fit actually is. _RESPONSE_POINT_MASS_FRACTION is
#     lowered to 0.44 (from 0.5) on that evidence: see its own comment below for why 0.44 specifically,
#     and for the coin-flip-at-the-boundary caveat that moves with it rather than disappearing.
#
# What audit R-3 did NOT close, stated here plainly rather than left for a future review to
# rediscover (see also robust_regression's own docstring and RegressionFit.degenerate_scale):
#   - A sloped (or otherwise differently-related, non-tied) majority/minority mixture is closed by
#     the new relative scale check for method='tukey' at every majority noise level reproduced here
#     (down to a relative scale ratio of ~1.5e-7). Under method='huber', R-3 left a gap NOT quite down
#     to ~1e-4 majority noise: huber's soft down-weighting settles the minority's weight around 1e-5 to
#     1e-4 there -- functionally negligible for the fit's outcome (the recovered coefficients already
#     match the majority to within 4e-6) but stayed just above the FIXED _ROBUST_NEGLIGIBLE_WEIGHT bar
#     R-3 shipped with (1e-6), so the crushed-weight fraction never reached _ROBUST_DEGENERACY_FRACTION
#     and the fit passed silently. R-3 considered widening that bar and deliberately deferred it as a
#     materially bigger, riskier change than the scale check alone. Audit R-4 tested the deferral
#     directly rather than accepting it: widening _ROBUST_NEGLIGIBLE_WEIGHT to 1e-5 through 1e-3
#     produced ZERO new failures anywhere in this module's full validated test suite (the 8%-outlier
#     contamination fixture, the capped/censored fixtures, the low-minority-fraction sweep, all of it)
#     -- huber's weight decays so slowly (w=c/|u|) that even a 1000x widening only requires |u| to
#     shrink to ~1345 standard deviations, astronomically implausible for real data, while tukey is
#     unaffected regardless (it already hard-zeros outliers past c=4.685). Closed by widening
#     _ROBUST_NEGLIGIBLE_WEIGHT to 1e-4 (see its own comment below); huber now closes to the same
#     ~1e-4 noise floor as tukey's relative-scale path.
#   - A majority/minority split that differs ONLY in location -- no covariate for the response's own
#     spread to track -- and forms TIGHT-but-not-exactly-tied clusters (as opposed to signal 2's
#     literal repeated value) is not reliably caught by anything here. Confirmed directly: for an
#     intercept-only design with a majority cluster at one mean and a minority cluster at another,
#     both with noise sigma=1e-6, the relative check's own denominator (the raw response's robust
#     spread) collapses to the SAME tiny scale as the fit's residuals, because there is no x-driven
#     variability holding them apart -- the ratio sits at ~1.0, nowhere near collapsed, in every
#     fraction tested (5%/10%/20% minority). Only an even smaller absolute noise (sigma=1e-9, small
#     enough to hit the unchanged hardcoded floor directly) is caught, via the ORIGINAL absolute path.
#   - As before R-3, and unchanged by it: nothing here -- the weight pattern, the response, or (now
#     removed) an OLS comparison -- can distinguish "a real minority signal" from "a minority of
#     contamination correctly discarded," because both leave an identical fingerprint; the question is
#     about what GENERATED the majority/minority split, which none of these signals observe. Past
#     whichever breakdown point applies, this module discloses the CONDITION it can actually see (a
#     collapsed scale with crushed weights, or a majority-shared response value) rather than a verdict
#     on which case this is, and the wording below says so rather than asserting the discarded-signal
#     reading as fact.
#   - As before R-3, and unchanged by it: a response point mass affecting LESS than the (now 44%,
#     was 50%) cutoff is ordinary estimation bias, not disclosed, and any hard cutoff on a smooth,
#     continuous underlying degradation curve has some neighborhood around it where realized-sample
#     noise -- not the true underlying severity -- decides whether a given fit gets flagged. GAP B
#     above moved that neighborhood and narrowed how much genuinely-severe territory sits on its
#     wrong side; it did not, and structurally could not, eliminate it.
_ROBUST_SCALE_FLOOR = 1e-8
# a fitted weight this far below the "well-fit" ceiling (u=0 gives w=1 under both methods) reflects
# a residual the floored scale has pushed out of range, not a considered judgment. R-3 left this at
# its original 1e-6 (see its GAP A note above for why widening this instead of the scale check was
# investigated and deliberately deferred). Widened to 1e-4 by audit R-4, which tested the deferral
# directly (see the "did NOT close" note above): confirmed zero new failures across this module's
# entire validated test suite at every value from 1e-5 through 1e-3, and confirmed the reason this is
# safe rather than merely untested -- huber's weight w=c/|u| decays so slowly that even 1e-3 (1000x
# the original) only requires |u| to shrink to ~1345 standard deviations to cross it, astronomically
# implausible under real Gaussian-like noise, while tukey's own hard cutoff (c=4.685) is unaffected by
# this constant regardless of its value.
_ROBUST_NEGLIGIBLE_WEIGHT = 1e-4
# lowered from R-1's 5% (see audit R-2 above). Safe rather than arbitrary because false positives
# here are prevented almost entirely by _robust_weight_collapse's near-floor requirement, not by
# this fraction: an M-estimator correctly rejecting ordinary, non-tied contamination never drives
# the SCALE anywhere near the floor in the first place (verified directly: the 8%-contamination
# design already used elsewhere in this test suite sits at scale ~0.32 under both methods -- tens of
# millions of floors away), so this fraction's only remaining job is to rule out the ZERO-crushed
# "genuinely perfect fit" case. 1% clears every reproduced minority down to the smallest tested with
# room to spare (a single-column 1% minority crushes exactly 1.00% of rows). audit R-3 re-confirmed
# this same protection holds with the new relative scale path active: a uniformly-tiny-noise, single-
# relationship fit with NO true minority (sigma swept 1e-9 through 0.3, an O(1) response) never
# crushes so much as one row's weight below this bar, at any of those noise levels -- IRLS's weight is
# self-normalizing on the standardized residual `u = r / scale`, so a purely Gaussian residual pattern
# essentially never produces |u| beyond the enormous multiple of `c` a near-zero weight requires,
# regardless of how tiny the absolute scale itself is. This fraction, not _ROBUST_RELATIVE_SCALE_FLOOR,
# is doing the real false-positive protection for the new relative path too.
_ROBUST_DEGENERACY_FRACTION = 0.01
# how far above _ROBUST_SCALE_FLOOR a converged scale can sit and still count as "collapsed", sized
# off the caller's OWN convergence tolerance rather than a hardcoded constant (audit R-2, signal 1):
# IRLS can stop moving beta -- and hence stop moving the scale -- as soon as consecutive iterates are
# within `tol`, so a scale still settling toward the floor at that moment can rest anywhere in
# roughly a `tol`-sized neighborhood above it. Measured directly against R-1's own 60-seed target
# reproduction, every one of the 22 seeds R-1 missed had its converged scale under 1x of `tol` above
# the floor; an 8x margin is kept for headroom that sample did not need but a different design might.
_ROBUST_FLOOR_SLACK = 8.0
# audit R-3, GAP A: a SECOND, independent way for _robust_weight_collapse's scale check to fire --
# the converged scale sits at or below this fraction of the raw response's own robust spread
# (_response_robust_scale), regardless of how that compares to the hardcoded absolute floor above.
# This is what lets signal 1 see a sloped majority/minority mixture whose majority noise is small but
# not literally floor-sized (sigma=1e-6 to 1e-4 against an O(1) response, in the reproductions below),
# which the absolute-only check could never reach at any slack multiple, because that noise never gets
# near 1e-8 in the first place. Unit-invariant by construction (a ratio of two spreads in the
# response's own units), unlike the absolute floor, which is only ever as meaningful as the caller's
# choice of units for y.
#
# Chosen value (1e-3) and why it is not a delicate choice: false positives on ordinary, well-
# conditioned continuous data are overwhelmingly prevented by _ROBUST_DEGENERACY_FRACTION's
# crushed-weight requirement (see that constant's own comment), not by this one -- a uniformly-tiny-
# noise, no-true-breakdown fit never crushes any row's weight regardless of noise scale, so this
# constant only controls how far the relative-collapse SIGNAL reaches, not whether a healthy fit can
# misfire. Verified directly at 1e-4, 1e-3, and 1e-2 alike: zero new false positives across every
# scenario checked (8%-contamination, mild 30%/4% capping, the 35%-minority huber sweep, the 1%-5%
# low-fraction sweep, and the uniform-tiny-noise probe above) at all three values. 1e-3 is chosen from
# that safe range with margin on both sides: 2-3 orders of magnitude below the smallest ratio measured
# on any benign fixture here (~0.0048 at 1% noise relative to an O(1) response, in the uniform-noise
# probe), and 1-2 orders of magnitude above the target reproduction's ratios (~1.5e-7 to ~2.4e-5 for
# majority noise 1e-6 to 1e-4 against an O(1) response, sloped two-column mixture, both methods).
#
# What this closes and what it still does not (see the module comment above for the full account):
# method='huber' at majority noise around 1e-4, which used to settle the minority's weight in the
# 1e-5-1e-4 range -- below "well-fit" but (before audit R-4 widened _ROBUST_NEGLIGIBLE_WEIGHT to
# 1e-4) never below that bar -- is now closed alongside tukey. Still out of reach: an intercept-only
# (no-covariate) location-only majority/minority split with tight but not exactly tied clusters,
# where the response's own spread collapses in step with the fit's residual scale for the same
# reason the fit's scale does, leaving no separation for a ratio-based check to see -- widening
# _ROBUST_NEGLIGIBLE_WEIGHT does not help here either, since this check's OWN floor/relative-floor
# gate (not the weight-fraction count) is what never fires for this shape in the first place.
_ROBUST_RELATIVE_SCALE_FLOOR = 1e-3
# at least this fraction of the raw response (near enough) sharing one value; see
# _robust_response_point_mass and audit R-2 (signal 2). Originally 0.5, matching the >50% M-estimator
# breakdown point in theory. Lowered to 0.44 by audit R-3 (GAP B in the module comment above): my own
# sweep of this module's own two-column zero-inflated fixture family (noise 0.02-1.0, n=600-6000,
# single- and two-column shapes, 2-10 true SD minority separations, 200 seeds per point) found mean/
# median relative coefficient error already severe (roughly 0.35-0.90, comparable to or worse than the
# cases this module already discloses) throughout a roughly symmetric band from ~44% through ~56%
# response-tie-fraction, essentially independent of noise level or effect size -- i.e. this is a
# property of the estimator's own breakdown dynamics near 50%, not of any one fixture's parameters.
# 0.44 sits at the edge of, not deep inside, that band: below it (40-42% in the same sweep), error is
# measurably and consistently better (roughly 0.06-0.27 depending on configuration) though not
# uniformly excellent, so this is a real, evidence-based improvement over 0.5 rather than a claim that
# 44% is itself a clean boundary. It also does not create new false positives on well-conditioned,
# non-degenerate data: per _response_point_mass_fraction's own design, continuous, untied data
# essentially never produces a 44%+ tie run by chance, so lowering this constant only pulls MORE of
# the already-tied-response family (the shape this signal exists to catch) into disclosure -- it does
# not newly catch anything without a real, near-exact response tie. What lowering this constant does
# NOT do: eliminate the coin-flip-like uncertainty AT whatever cutoff is current. Measured directly at
# the OLD 50% cutoff: 51% of 200 seeds were flagged, but whether a given seed was flagged correlated
# strongly with its own realized tie-fraction (r=0.82, i.e. mostly which way that seed's Bernoulli
# draw happened to land) and only weakly with its actual fit quality (r=0.34) -- a hard threshold on a
# smooth, continuous degradation curve cannot avoid this at ANY value, and 0.44 is not an exception;
# it only narrows how much clearly-severe territory sits on the wrong (undisclosed) side of the line.
_RESPONSE_POINT_MASS_FRACTION = 0.44
# ties within this fraction of the response's own range count as "the same value": generous enough
# to survive any floating-point noise a real capping/rounding pipeline introduces, far too tight to
# ever merge two genuinely distinct values from continuous data into a false point mass -- for
# responses whose absolute scale is not itself tiny. See _RESPONSE_POINT_MASS_ULP_FLOOR_MULT below
# for the companion term this alone is not sufficient without.
_RESPONSE_POINT_MASS_REL_TOL = 1e-9
# GAP (audit R-4, found reviewing R-3): the tie tolerance used to be
# `max(1e-12, spread * _RESPONSE_POINT_MASS_REL_TOL)` -- a FIXED absolute floor whenever `spread`
# itself was small enough (roughly <1e-3) for that floor to bind instead of the relative term. That
# floor does not scale with `n`, but the natural spacing between adjacent order statistics of a
# dense continuous sample DOES shrink with `n` -- at a small enough absolute response scale and
# large enough `n`, adjacent sorted values can land within (or even collide at, via ordinary float64
# rounding) 1e-12 of each other purely from sampling density, with no real repeated/capped/rounded
# value anywhere. Confirmed: y = N(10.0, 1e-9, 100_000) has a perfect, correct fit (coefficient error
# ~3e-12) but used to register 94% "point mass" and fire a UserWarning claiming zero-inflation/
# capping/rounding that was not present -- 2.5% of adjacent sorted values were exactly bit-identical
# from rounding collisions alone, and the old floor treated 99% of gaps as "tied".
#
# Fixed by replacing the fixed absolute floor with one anchored to the VALUES' OWN float64
# resolution: two adjacent sorted values are "tied" only if their gap is within this many ULPs of
# the larger value's own magnitude (via `np.spacing`), OR within the spread-relative tolerance above
# -- whichever is more permissive. A genuine repeated/capped/rounded value produces either an exact
# bit-for-bit tie or a few-ULP one from ordinary arithmetic (e.g. `min(z, cap)`); two independent
# draws from a continuous distribution landing within a handful of ULPs of each other by pure chance
# is vanishingly unlikely REGARDLESS of `n`, unlike a fixed absolute epsilon. Verified this holds at
# every one of dense-small-scale n in {1e3, 1e4, 5e4, 1e5, 5e5} (point-mass fraction stays <=0.001 in
# every case, vs. >=0.44 -- a false positive -- at the old floor for the larger `n` values in that
# set), while the genuine zero-inflated (65%) and capped (70%) fixtures this signal exists to catch
# are unaffected (still register their true fraction exactly, independent of the multiplier chosen
# from 4 to 64). Chosen value has a >100x margin over the largest incidental drift (a handful of
# ULPs) either fixture family plausibly introduces.
_RESPONSE_POINT_MASS_ULP_FLOOR_MULT = 8


def _response_robust_scale(y: np.ndarray) -> float:
    """Robust (MAD-based) spread of the raw response, independent of any fit or IRLS iteration.

    Used by :func:`_robust_weight_collapse` (audit R-3, GAP A) as the denominator of its relative
    floor-band check -- see :data:`_ROBUST_RELATIVE_SCALE_FLOOR` for the full reasoning. Floored to a
    tiny epsilon (not :data:`_ROBUST_SCALE_FLOOR`, which is a units-dependent constant for the
    RESIDUAL scale, not a generic epsilon) purely to avoid division by exact zero when ``y`` itself
    is majority-tied -- a case :func:`_robust_response_point_mass` already handles independently, so
    this function returning a degenerate value there is harmless, never a silent miss.
    """
    return max(float(np.median(np.abs(y - np.median(y)))) / 0.6745, 1e-12)


def _robust_weight_collapse(scale: float, w: np.ndarray, tol: float, y_scale: float) -> bool:
    """Has the IRLS scale collapsed -- in absolute or relative terms -- with a real share of rows
    crushed by it?

    Signal 1 of audit R-2 (module comment above), extended by audit R-3 (GAP A). Two independent
    ways for "collapsed" to be true:

      * absolute (R-2, unchanged): a band within :data:`_ROBUST_FLOOR_SLACK` multiples of ``tol``
        above :data:`_ROBUST_SCALE_FLOOR`, rather than bit-exact equality to it, so a scale that
        IRLS's OWN convergence check let stop just short of the floor is no longer read as healthy.
      * relative (R-3, new): at or below :data:`_ROBUST_RELATIVE_SCALE_FLOOR` times ``y_scale`` (see
        :func:`_response_robust_scale`) -- unit-invariant, and able to see a majority sub-population
        whose own noise is small but not literally floor-sized (sigma around 1e-6 to 1e-4 against an
        O(1) response, in this module's own reproductions), which the absolute check cannot reach at
        any slack multiple because that noise never approaches the hardcoded 1e-8 in the first place.

    Either way, a collapsed scale is still not YET a problem by itself: a genuinely perfect (or
    near-perfect) fit -- every row's residual near the same tiny value, absolutely or relative to
    the response's own scale -- collapses it too, harmlessly, leaving every weight near 1. What
    tells the two apart is whether the collapsed scale has gone on to treat a real share of the
    OTHER rows as outliers -- see :data:`_ROBUST_DEGENERACY_FRACTION`, whose own comment documents
    why this fraction (not the scale check) is what actually protects ordinary, well-conditioned
    data from a false positive here, for either the absolute or the relative path.
    """
    absolute_floor = scale <= _ROBUST_SCALE_FLOOR + _ROBUST_FLOOR_SLACK * tol
    relative_floor = scale <= _ROBUST_RELATIVE_SCALE_FLOOR * y_scale
    if not (absolute_floor or relative_floor):
        return False
    return bool(np.mean(w <= _ROBUST_NEGLIGIBLE_WEIGHT) >= _ROBUST_DEGENERACY_FRACTION)


def _response_point_mass_fraction(y: np.ndarray) -> float:
    """Largest share of ``y`` (near enough) tied to a single value, independent of any fit.

    Sorts ``y`` and finds the longest run of consecutive values each within tolerance of its run's
    predecessor -- a near-exact-tie detector, not a density-estimation bandwidth. Continuous,
    untied data essentially never produces a long such run by chance; zero-inflation, capping,
    flooring, and rounding all produce one by construction, regardless of what the design matrix or
    any fitted coefficient looks like.

    The per-pair tolerance is the MORE PERMISSIVE of two terms (see :data:`_RESPONSE_POINT_MASS_REL_TOL`
    and :data:`_RESPONSE_POINT_MASS_ULP_FLOOR_MULT` for the full reasoning behind each): a share of
    the response's overall range (survives realistic-scale floating-point noise from a real
    capping/rounding pipeline), and a multiple of the adjacent values' own float64 resolution
    (survives being defeated by high sampling density at a small absolute response scale, where a
    fixed absolute epsilon would not).
    """
    n = y.size
    if n < 2:
        return 1.0
    ys = np.sort(y)
    spread = ys[-1] - ys[0]
    local_scale = np.maximum(np.abs(ys[:-1]), np.abs(ys[1:]))
    ulp_floor = _RESPONSE_POINT_MASS_ULP_FLOOR_MULT * np.spacing(np.maximum(local_scale, 1.0))
    tie_tol = np.maximum(ulp_floor, spread * _RESPONSE_POINT_MASS_REL_TOL)
    breaks = np.flatnonzero(np.diff(ys) > tie_tol)
    run_lengths = np.diff(np.concatenate(([0], breaks + 1, [n])))
    return float(np.max(run_lengths)) / n


def _robust_response_point_mass(y: np.ndarray) -> bool:
    """Does at least :data:`_RESPONSE_POINT_MASS_FRACTION` of the raw response share (near enough)
    the same value?

    Signal 2 of audit R-2 (module comment above), threshold lowered by audit R-3 (GAP B) from 0.5 to
    0.44 -- see :data:`_RESPONSE_POINT_MASS_FRACTION`'s own comment for the sweep that justifies the
    new value. Fit-independent, computed once on ``y`` alone before IRLS ever runs. See
    :func:`_response_point_mass_fraction` for the tie detector and the module comment for why this
    catches a capped/censored breakdown that the weight/scale signal (:func:`_robust_weight_collapse`)
    structurally cannot.
    """
    return _response_point_mass_fraction(y) >= _RESPONSE_POINT_MASS_FRACTION


def _robust_breakdown_explanation(collapse: bool, point_mass: bool, point_mass_fraction: float, w: np.ndarray) -> str:
    """Shared wording for the RuntimeError/UserWarning naming a majority-tied pattern.

    Audit R-3 rewrite (module comment above, GAP A): states the observed CONDITION plainly and
    NEVER asserts a verdict on what it means. The module comment documents why that verdict is not
    something the weight pattern or the response can establish on their own -- a genuine minority
    signal discarded, and a genuine minority of contamination correctly rejected, produce an
    identical fingerprint; so, separately, does a majority that is simply exactly (or near-exactly)
    correct, with nothing discarded at all (audit R-2's original "benign heteroskedastic" concern,
    now resolved by this wording rather than by suppressing the signal -- see the module comment).
    Only the signal(s) that actually fired are named, and the advice offered differs by which one:
    a response-level tie (signal 2) supports naming the data SHAPE it typically comes from
    (zero-inflation, capping, rounding) because that is read directly off ``y`` itself; a weight/
    scale collapse alone (signal 1) does not license that same claim, so it gets more general
    advice instead (compare against OLS, inspect which rows were down-weighted) plus an explicit
    "if you already expect this, it's not news" out.
    """
    conditions = []
    if point_mass:
        conditions.append(
            f"{point_mass_fraction:.0%} of the raw response values are (near enough) tied to the "
            "same value -- the shape zero-inflated, capped/censored, or rounded/floored "
            "measurements typically produce"
        )
    if collapse:
        frac = float(np.mean(w <= _ROBUST_NEGLIGIBLE_WEIGHT))
        conditions.append(
            f"{frac:.0%} of observations were assigned (near) zero weight during robust fitting, "
            "with the IRLS scale estimate that produced those weights collapsed -- at (or within a "
            "convergence step of) its numerical floor, or far smaller than the response's own "
            "spread"
        )
    observed = "; separately, ".join(conditions) if conditions else "a majority-tied pattern"
    hedge = (
        f"{observed}. This is the shape of the estimator's own >50% breakdown point: past it, a "
        "MAD-scaled M-estimator cannot tell, from this pattern alone, a majority that is "
        "legitimately exact or near-exact (nothing discarded) apart from a majority that has "
        "overwhelmed and discarded a real, differently-behaved minority (coef reflects only the "
        "majority) -- both produce this identical pattern. See RegressionFit.degenerate_scale and "
        "robust_regression's docstring for what this disclosure can and cannot tell you."
    )
    if point_mass:
        advice = (
            " If this response shape was not expected, consider a model built for it instead: "
            "zero-inflated or hurdle regression for excess zeros, Tobit / censored regression for "
            "a capped or floored measurement."
        )
    else:
        advice = (
            " If you did not expect this many observations to collapse this precisely onto one "
            "fit, this is worth investigating before trusting coef -- e.g. compare against a plain "
            "ordinary-least-squares fit, or inspect which rows were assigned near-zero weight; if "
            "you already know your data has an exact or near-exact majority by design, this is "
            "naming that shape, not flagging a new problem."
        )
    return f"{hedge}{advice}"


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

    The scale estimate (median absolute residual deviation / 0.6745) is only ever as robust as any
    MAD-based estimator is: past its own breakdown point -- half or more of the rows sharing (near
    enough) the same residual under the current fit, an ordinary shape for zero-inflated or
    capped/rounded data and not a contrived one -- it is exactly zero by construction and floors at
    ``1e-8`` (or, as of audit R-3, is treated the same way once it is merely far smaller than the
    response's own spread, even without literally reaching that floor). That is an inherent limit of
    this estimator family, not something iterating further or switching ``method`` fixes:
    ``tukey``'s redescending weight then hard-zeros essentially every row (raised below), and
    ``huber``'s never-quite-zero weight instead settles -- sometimes only after many iterations --
    on a near-zero-coefficient fit matching the majority tie that reports ``converged=True``. A
    capped or censored measurement breaks down the same way without ever collapsing the scale
    estimate itself: a shared ceiling ties rows' raw responses together but not their residuals once
    the fit carries any slope, so this is instead read directly off the response (see the module
    comment above ``_robust_weight_collapse`` in this file for the full two-signal detection design,
    including what audit R-3 closed and what it plainly did not: most notably, a majority and a
    minority that follow two genuinely DIFFERENT relationships -- as opposed to sharing one tied
    value -- is only ever PARTIALLY visible to this mechanism, and a location-only split with no
    covariate to anchor the response's own spread is not visible to it at all). All of these are
    disclosed rather than left looking like an ordinary converged fit: see
    ``RegressionFit.degenerate_scale`` and the ``UserWarning`` issued at fit time. What is NOT, and
    cannot be, disclosed: WHICH of "a real minority signal was discarded," "a minority of
    contamination was correctly rejected," or "nothing was discarded at all, because the majority is
    genuinely this exact" produced a flagged pattern -- all three leave the same fingerprint, so the
    warning names the observed condition, never a verdict on which case it is.

    Raises:
        RuntimeError: every observation lands at zero weight (the message names the breakdown
            pattern above when one was detected), coefficients become non-finite, or IRLS does not
            converge in ``max_iter`` iterations.
    """
    _solver_controls(max_iter, tol)
    X, y = _regression_data(x, y)
    if method not in {"huber", "tukey"}:
        raise ValueError("method must be 'huber' or 'tukey'.")
    if c is None:
        c = 1.345 if method == "huber" else 4.685
    if not np.isfinite(c) or c <= 0:
        raise ValueError("c must be finite and > 0")
    ols_beta = np.linalg.lstsq(X, y, rcond=None)[0]  # the IRLS starting point
    beta = ols_beta
    # both fit-independent; computed once on y alone before IRLS ever runs (audit R-2; y_scale is
    # new in audit R-3, GAP A -- see _response_robust_scale)
    point_mass_fraction = _response_point_mass_fraction(y)
    point_mass = point_mass_fraction >= _RESPONSE_POINT_MASS_FRACTION
    y_scale = _response_robust_scale(y)
    n_iter = 0
    scale = 1.0
    converged = False
    degenerate = False
    collapse_flagged = False
    for n_iter in range(1, max_iter + 1):
        r = y - X @ beta
        scale = max(np.median(np.abs(r - np.median(r))) / 0.6745, _ROBUST_SCALE_FLOOR)
        u = r / scale
        if method == "huber":
            w = np.where(np.abs(u) <= c, 1.0, c / np.maximum(np.abs(u), 1e-12))
        elif method == "tukey":
            w = np.where(np.abs(u) <= c, (1.0 - (u / c) ** 2) ** 2, 0.0)
        # audit R-3 (module comment, GAP A) removed the OLS-divergence suppressor that used to gate
        # this: signal 1 now stands on its own, honestly WORDED rather than silenced when it fires
        # on a fit that turns out to be fine -- see _robust_breakdown_explanation
        collapse_flagged = _robust_weight_collapse(scale, w, tol, y_scale)
        degenerate = collapse_flagged or point_mass
        if not np.any(w > 0):
            if degenerate:
                raise RuntimeError(
                    "robust regression assigned zero weight to every observation: "
                    + _robust_breakdown_explanation(collapse_flagged, point_mass, point_mass_fraction, w)
                )
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
        if degenerate:
            raise RuntimeError(
                f"robust regression failed to converge in {max_iter} iterations: "
                + _robust_breakdown_explanation(collapse_flagged, point_mass, point_mass_fraction, w)
                + " More iterations are unlikely to help."
            )
        raise RuntimeError(f"robust regression failed to converge in {max_iter} iterations")
    if degenerate:
        warnings.warn(
            f"robust_regression (method={method!r}) converged, but "
            + _robust_breakdown_explanation(collapse_flagged, point_mass, point_mass_fraction, w)
            + " converged=True here reflects a self-consistent fixed point, not confirmation of "
            "whether coef is the fully correct answer or reflects only the majority pattern above "
            "(also recorded as RegressionFit.degenerate_scale).",
            UserWarning,
            stacklevel=2,
        )
    return RegressionFit(
        beta,
        X @ beta,
        float(scale),
        n_iter,
        converged=True,
        rank=int(np.linalg.matrix_rank(X)),
        degenerate_scale=degenerate,
    )


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
