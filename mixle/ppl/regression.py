"""Regression / GLMs for mixle.ppl.

A linear predictor in a parameter slot makes a model a regression; the outer family sets the
link:

    Normal(a*Field("x") + b, sigma)   identity link  -> linear regression
    Bernoulli(a*Field("x") + b)       logit link     -> logistic regression
    Poisson(a*Field("x") + b)         log link       -> Poisson regression

Coefficients may be ``free`` or may carry Normal penalty handles.  Fitting is
IRLS/Fisher scoring for a likelihood or penalized-likelihood point estimate.
For Normal responses a Normal coefficient prior is scaled by the residual
variance (the fixed ``sigma``, or the working ``sigma^2`` when it is free), so
``result.beta`` / ``result.cov`` are the exact Gaussian posterior mean and
covariance at that plug-in scale.  Non-Gaussian GLM families (Bernoulli /
Poisson) have unit dispersion, so the prior enters the IRLS equations
unscaled -- the exact penalized-likelihood (MAP) mode.  Fit with
``.fit(y, given={"x": xs})``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from mixle.ppl.core import RandomVariable, _LinearPredictor
from mixle.ppl.core import free as FREE
from mixle.utils.exact import require_exact_bool

# family -> canonical link name
_LINK = {"Normal": "identity", "Bernoulli": "logit", "Poisson": "log"}


def _positive_int(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return int(value)


def _positive_float(value, label):
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite positive scalar.") from exc
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{label} must be a finite positive scalar.")
    return out


def _numeric_vector(value, label, *, length=None):
    try:
        out = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite numeric one-dimensional sequence.") from exc
    if out.ndim != 1 or out.size == 0 or not np.isfinite(out).all():
        raise ValueError(f"{label} must be a non-empty finite numeric one-dimensional sequence.")
    if length is not None and out.size != length:
        raise ValueError(f"{label} has {out.size} rows; expected exactly {length}.")
    return out.copy()


def _group_vector(value, label, *, length):
    out = np.asarray(value)
    if out.ndim != 1 or out.size != length:
        raise ValueError(f"{label} must be one-dimensional with exactly {length} rows.")
    for item in out:
        if item is None or (isinstance(item, (float, np.floating)) and not math.isfinite(float(item))):
            raise ValueError(f"{label} contains a missing or non-finite group label.")
    return out.copy()


def _validate_conditional_data(rv, data, given):
    """Validate one aligned conditional dataset before any NumPy broadcasting."""
    y = _numeric_vector(data, "response")
    if given is None:
        given = {}
    if not isinstance(given, Mapping):
        raise TypeError("given must be a mapping from field names to one-dimensional arrays.")
    given = dict(given)
    numeric_fields, group_fields = set(), set()
    for arg in rv._args:
        if not isinstance(arg, _LinearPredictor):
            continue
        numeric_fields.update(field.name for _, field in arg.terms)
        for group, slopes in arg.groups:
            group_fields.add(group)
            numeric_fields.update(slopes)
    required = numeric_fields | group_fields
    missing = sorted(required - given.keys())
    if missing:
        raise ValueError(f"given is missing required field(s): {missing}.")
    clean = {}
    for name, values in given.items():
        if name in group_fields and name not in numeric_fields:
            clean[name] = _group_vector(values, f"given[{name!r}]", length=y.size)
        else:
            clean[name] = _numeric_vector(values, f"given[{name!r}]", length=y.size)
    return y, clean


def _parameter_layout(est):
    """Return unambiguous display names and handle lookup for coefficient columns."""
    names = [field.name if field is not None else "intercept" for _, field in est]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"regression coefficient aliases must be unique; duplicates: {duplicates}.")
    idx_of, seen_handles = {}, set()
    for i, (coef, _field) in enumerate(est):
        if not isinstance(coef, RandomVariable):
            continue  # the singleton `free` token is not an addressable parameter handle
        key = id(coef)
        if key in seen_handles:
            raise ValueError("the same coefficient handle cannot occupy multiple regression columns.")
        seen_handles.add(key)
        idx_of[key] = i
    return names, idx_of


def _validate_response(family, y):
    if family == "Bernoulli" and np.any((y != 0.0) & (y != 1.0)):
        raise ValueError("Bernoulli regression responses must be exactly 0 or 1.")
    if family == "Poisson" and (np.any(y < 0.0) or np.any(y != np.floor(y))):
        raise ValueError("Poisson regression responses must be exact non-negative integers.")


def _validate_supported_priors(est, *, allowed=("Normal", "Laplace")):
    for coef, _field in est:
        if isinstance(coef, RandomVariable) and coef._family.name not in allowed:
            raise NotImplementedError(
                f"regression coefficient prior {coef._family.name!r} is not implemented; "
                f"supported prior families are {sorted(allowed)}."
            )


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
    """Regression point estimate with coefficient curvature diagnostics."""

    def __init__(
        self,
        names,
        idx_of,
        beta,
        cov,
        sigma,
        columns,
        link="identity",
        *,
        converged=True,
        iterations=1,
        termination_reason="closed_form",
        objective_delta=0.0,
    ):
        self.names = list(names)  # unique display aliases
        self.parameter_ids = [f"coef:{i}" for i in range(len(self.names))]
        self.beta = np.asarray(beta, dtype=float).copy()
        self.cov = None if cov is None else np.asarray(cov, dtype=float).copy()
        self.sigma = float(sigma)
        self.link = link
        self._idx_of = idx_of  # id(coef handle) -> column index
        self._columns = columns  # list of (kind, payload) for predict
        self.converged = require_exact_bool(converged, "converged")
        self.iterations = int(iterations)
        self.termination_reason = str(termination_reason)
        self.objective_delta = float(objective_delta)
        self.coefficients = {
            self.names[i]: {
                "id": self.parameter_ids[i],
                "mean": float(self.beta[i]),
                "sd": None if self.cov is None else float(np.sqrt(max(self.cov[i, i], 0.0))),
            }
            for i in range(len(self.names))
        }
        self.acceptance_rate = None
        self.predictive = None

    def _resolve(self, param):
        if isinstance(param, str):
            if param in self.parameter_ids:
                return self.parameter_ids.index(param)
            if param in self.names:
                return self.names.index(param)
            raise KeyError(f"unknown regression parameter {param!r}.")
        if isinstance(param, (int, np.integer)):
            index = int(param)
            if index < 0 or index >= len(self.names):
                raise IndexError(f"regression parameter index {index} is out of range.")
            return index
        try:
            return self._idx_of[id(param)]
        except KeyError as exc:
            raise KeyError("object is not an addressable coefficient handle in this regression.") from exc

    def samples(self, param=None, n: int = 4000, rng=None):
        """Draw from the Gaussian coefficient approximation represented by ``beta`` and ``cov``."""
        if self.cov is None:
            raise NotImplementedError("coefficient uncertainty was not estimated for this regression route.")
        rng = rng or np.random.RandomState()
        if param is None:
            return rng.multivariate_normal(self.beta, self.cov, n)
        i = self._resolve(param)
        return rng.normal(self.beta[i], np.sqrt(self.cov[i, i]), n)

    def predict(self, given, *, n=None, rng=None):
        """Predict the response mean at covariates ``given`` (dict of arrays): the fitted
        value through the link (probabilities for logistic, rates for Poisson, mean for
        linear). With ``n``, returns ``n`` draws of the fitted mean under the Gaussian
        coefficient approximation; observation noise is not added."""
        X, offset = _design(self._columns, given)
        eta = offset + X @ self.beta
        if n is None:
            return _link_inv(self.link, eta)
        if self.cov is None:
            raise NotImplementedError("coefficient uncertainty was not estimated for this regression route.")
        rng = rng or np.random.RandomState()
        out = np.empty((n, eta.size))
        for k in range(n):
            beta_k = rng.multivariate_normal(self.beta, self.cov)
            out[k] = _link_inv(self.link, offset + X @ beta_k)
        return out

    def summary(self):
        """Return coefficient summaries and residual scale metadata."""
        return {
            "coefficients": self.coefficients,
            "sigma": self.sigma,
            "converged": self.converged,
            "iterations": self.iterations,
            "termination_reason": self.termination_reason,
            "objective_delta": self.objective_delta,
        }

    def to_exponential_family(self, engine=None):
        """Return the conditional exponential-family view ``p(y|x)`` for a canonical link.

        For a canonical link the linear predictor *is* the natural parameter:
        ``eta(x) = offset + X @ beta`` is the logit (Bernoulli) / log-rate (Poisson)
        directly, and the mean ``mu(x)/sigma^2`` paired with ``-1/(2 sigma^2)`` for the
        Normal.  The returned
        :class:`~mixle.stats.compute.exp_family.ConditionalExponentialFamilyForm` exposes
        ``natural_parameters(x)``, ``sufficient_statistics(y)``, ``log_partition``,
        ``log_base_measure(y)``, ``mean(x)`` (the inverse link == :meth:`predict`), and
        ``log_density(y, x)``.
        """
        from mixle.engines import NUMPY_ENGINE
        from mixle.stats.compute.exp_family import ConditionalExponentialFamilyForm

        eng = NUMPY_ENGINE if engine is None else engine
        link = self.link

        def _eta_linear(given):
            X, offset = _design(self._columns, given)
            return offset + X @ self.beta

        if link == "logit":
            from mixle.stats.univariate.discrete.bernoulli import BernoulliDistribution

            response = BernoulliDistribution(0.5)

            def natural_fn(given):
                return _eta_linear(given)[:, None]

            def log_partition_fn(eta):
                e = np.asarray(eta, float).reshape(-1)
                return np.logaddexp(0.0, e)  # log(1 + e^eta)

            dispersion = None
        elif link == "log":
            from mixle.stats.univariate.discrete.poisson import PoissonDistribution

            response = PoissonDistribution(1.0)

            def natural_fn(given):
                return _eta_linear(given)[:, None]

            def log_partition_fn(eta):
                return np.exp(np.asarray(eta, float).reshape(-1))  # A = lambda = e^eta

            dispersion = None
        elif link == "identity":
            from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

            sigma2 = self.sigma**2
            response = GaussianDistribution(0.0, sigma2)

            def natural_fn(given):
                mu = _eta_linear(given)
                eta1 = mu / sigma2
                eta2 = np.full_like(mu, -0.5 / sigma2)
                return np.column_stack([eta1, eta2])

            def log_partition_fn(eta):
                e = np.atleast_2d(np.asarray(eta, float))
                eta1, eta2 = e[:, 0], e[:, 1]
                # A(eta) = -eta1^2/(4 eta2) - 0.5 log(-eta2/pi)
                #        = mu^2/(2 sigma^2) + 0.5 log(2 pi sigma^2)
                return -(eta1 * eta1) / (4.0 * eta2) - 0.5 * np.log(-eta2 / np.pi)

            dispersion = sigma2
        else:
            raise NotImplementedError("no canonical exponential-family map for link %r." % link)

        def mean_fn(given):
            return _link_inv(link, _eta_linear(given))

        return ConditionalExponentialFamilyForm(
            response_family=response,
            natural_fn=natural_fn,
            log_partition_fn=log_partition_fn,
            mean_fn=mean_fn,
            dispersion=dispersion,
            engine=eng,
        )


def _columns_of(linpred: _LinearPredictor):
    """Return (est_columns, fixed_columns): estimated coefs (RV prior / free) vs constants."""
    cols = list(linpred.terms)  # (coef, Field)
    if linpred.intercept is not None:
        cols.append((linpred.intercept, None))  # None field -> intercept (ones)
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
    if not isinstance(given, Mapping):
        raise TypeError("given must be a mapping from field names to one-dimensional arrays.")
    n = None
    for _, field in est + fixed:
        if field is not None:
            if field.name not in given:
                raise ValueError(f"given is missing required field {field.name!r}.")
            n = _numeric_vector(given[field.name], f"given[{field.name!r}]").size
            break
    if n is None:  # intercept-only fixed part (e.g. a random-effects-only model): size from given
        for arr in (given or {}).values():
            raw = np.asarray(arr)
            if raw.ndim != 1 or raw.size == 0:
                raise ValueError("given arrays must be non-empty and one-dimensional.")
            n = raw.size
            break
    if n is None:
        raise ValueError("need at least one covariate or a given= array to size the design matrix.")
    mat = []
    for _, field in est:
        mat.append(
            np.ones(n) if field is None else _numeric_vector(given[field.name], f"given[{field.name!r}]", length=n)
        )
    X = np.column_stack(mat) if mat else np.zeros((n, 0))
    offset = np.zeros(n)
    for c, field in fixed:
        offset += c * (
            np.ones(n) if field is None else _numeric_vector(given[field.name], f"given[{field.name!r}]", length=n)
        )
    return X, offset


class LMMResult:
    """Linear mixed model: fixed-effect coefficients + variance components + group effects."""

    def __init__(
        self,
        names,
        beta,
        cov,
        Sigma,
        sigma,
        b,
        group_levels,
        re_names,
        *,
        converged,
        iterations,
        objective_delta,
    ):
        self.names = names
        self.parameter_ids = [f"fixed:{i}" for i in range(len(names))]
        self.beta = beta
        self.cov = cov
        self.random_cov = np.asarray(Sigma)  # random-effects covariance (q x q)
        self.random_names = re_names  # ['intercept', slope names...]
        self.tau = float(np.sqrt(Sigma[0, 0]))  # random-intercept sd (back-compat)
        self.sigma = float(sigma)  # residual sd
        # per-group effects: intercept for back-compat, full vector under group_effects_full
        self.group_effects = {lv: float(b[i, 0]) for i, lv in enumerate(group_levels)}
        self.group_effects_full = {lv: b[i] for i, lv in enumerate(group_levels)}
        self.coefficients = {
            names[i]: {
                "id": self.parameter_ids[i],
                "mean": float(beta[i]),
                "sd": float(np.sqrt(cov[i, i])),
            }
            for i in range(len(names))
        }
        self.acceptance_rate = None
        self.predictive = None
        self.converged = bool(converged)
        self.iterations = int(iterations)
        self.objective_delta = float(objective_delta)
        self.termination_reason = "tolerance" if converged else "max_iterations"

    def summary(self):
        """Return fixed effects, random-effects covariance, scale, and group count."""
        return {
            "coefficients": self.coefficients,
            "random_cov": self.random_cov,
            "sigma": self.sigma,
            "n_groups": len(self.group_effects),
            "converged": self.converged,
            "iterations": self.iterations,
            "termination_reason": self.termination_reason,
            "objective_delta": self.objective_delta,
        }


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
    names, _ = _parameter_layout(est)
    if not names:
        X = np.ones((y.size, 1))
        names = ["intercept"]
    N, p = X.shape

    # random-effects design Z: intercept + slope columns
    re_names = ["intercept"] + list(slopes)
    zcols = [np.ones(N)] + [_numeric_vector(given[s], f"given[{s!r}]", length=N) for s in slopes]
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
    converged, delta = False, math.inf
    for iteration in range(1, max_iter + 1):
        resid = yv - X @ beta
        Sinv = np.linalg.inv(Sigma)
        SS = np.zeros((q, q))
        err2 = 0.0
        trace_term = 0.0
        for gi, idx in enumerate(groups):
            Zg, rg = Z[idx], resid[idx]
            cov_g = np.linalg.inv(Sinv + Zg.T @ Zg / sigma2)  # E-step posterior of b_g
            b_g = cov_g @ (Zg.T @ rg / sigma2)
            b[gi] = b_g
            SS += np.outer(b_g, b_g) + cov_g
            pred = Zg @ b_g
            err2 += float((rg - pred) @ (rg - pred))
            trace_term += float(np.trace(Zg @ cov_g @ Zg.T))
        Sigma_new = SS / G  # M-step
        sigma2_new = max((err2 + trace_term) / N, 1e-8)
        Zb = np.einsum("nq,nq->n", Z, b[g])
        beta_new = np.linalg.lstsq(X, yv - Zb, rcond=None)[0]
        # Converge on (beta, Sigma, sigma^2) JOINTLY. In a balanced design beta is an exact
        # fixed point after one update (the BLUP means cancel), so a beta-only test exits at
        # iteration ~0 with the variance components one EM step from their crude init.
        delta = max(
            float(np.max(np.abs(beta_new - beta))),
            float(np.max(np.abs(Sigma_new - Sigma))),
            abs(sigma2_new - sigma2),
        )
        beta, Sigma, sigma2 = beta_new, Sigma_new, sigma2_new
        if delta < tol:
            converged = True
            break

    # GLS fixed-effect covariance (X' V^-1 X)^-1 with V = Z Sigma Z' + sigma^2 I (block-diagonal
    # per group). inv(X'X / sigma^2) would ignore the random-effect term and is badly
    # anti-conservative under grouping. Woodbury per group reuses the E-step posterior:
    # V_g^-1 = (I - Z_g cov_g Z_g' / sigma^2) / sigma^2 with cov_g = (Sigma^-1 + Z_g'Z_g/sigma^2)^-1.
    if p:
        Sinv = np.linalg.inv(Sigma)
        xtvx = np.zeros((p, p))
        for idx in groups:
            Xg, Zg = X[idx], Z[idx]
            cov_g = np.linalg.inv(Sinv + Zg.T @ Zg / sigma2)
            XtZ = Xg.T @ Zg
            xtvx += (Xg.T @ Xg - XtZ @ cov_g @ XtZ.T / sigma2) / sigma2
        cov = np.linalg.inv(xtvx)
    else:
        cov = np.zeros((0, 0))
    result = LMMResult(
        names,
        beta,
        cov,
        Sigma,
        np.sqrt(sigma2),
        b,
        list(levels),
        re_names,
        converged=converged,
        iterations=iteration,
        objective_delta=delta,
    )
    return RandomVariable._bound(None, name=rv._name, result=result)


class GLMMResult:
    """Generalized linear mixed model: fixed-effect coefficients (on the link scale) + the
    random-effects covariance + per-group effects. No residual scale (the family sets dispersion)."""

    def __init__(
        self,
        family,
        link,
        names,
        beta,
        cov,
        Sigma,
        b,
        group_levels,
        re_names,
        *,
        converged,
        outer_iterations,
        inner_iterations,
        inner_converged,
        objective_delta,
    ):
        self.family = family
        self.link = link
        self.names = names
        self.parameter_ids = [f"fixed:{i}" for i in range(len(names))]
        self.beta = beta
        self.cov = cov
        self.random_cov = np.asarray(Sigma)
        self.random_names = re_names
        self.tau = float(np.sqrt(Sigma[0, 0]))  # random-intercept sd
        self.group_effects = {lv: float(b[i, 0]) for i, lv in enumerate(group_levels)}
        self.group_effects_full = {lv: b[i] for i, lv in enumerate(group_levels)}
        self.coefficients = {
            names[i]: {
                "id": self.parameter_ids[i],
                "mean": float(beta[i]),
                "sd": float(np.sqrt(max(cov[i, i], 0.0))),
            }
            for i in range(len(names))
        }
        self.acceptance_rate = None
        self.predictive = None
        self.converged = bool(converged)
        self.outer_iterations = int(outer_iterations)
        self.inner_iterations = tuple(int(v) for v in inner_iterations)
        self.inner_converged = tuple(bool(v) for v in inner_converged)
        self.objective_delta = float(objective_delta)
        self.termination_reason = "tolerance" if converged else "max_iterations"

    def summary(self):
        """Return fixed effects, random-effects covariance, link, and group count."""
        return {
            "coefficients": self.coefficients,
            "random_cov": self.random_cov,
            "link": self.link,
            "n_groups": len(self.group_effects),
            "converged": self.converged,
            "outer_iterations": self.outer_iterations,
            "inner_iterations": self.inner_iterations,
            "inner_converged": self.inner_converged,
            "termination_reason": self.termination_reason,
            "objective_delta": self.objective_delta,
        }


def _glmm_fit(rv, y, given, linpred, link, max_iter, inner_max_iter, tol):
    """Generalized linear mixed model with one grouping factor, by penalized quasi-likelihood (PQL).

    ``eta = X beta + Z b_g``, ``b_g ~ N(0, Sigma)``, ``y ~ Family(link^-1(eta))`` (Poisson log /
    Bernoulli logit). Alternates IRLS over the fixed effects, a per-group penalized-IRLS update of the
    random effects (ridge ``Sigma^-1``), and an EM update of ``Sigma`` from the group-effect second
    moments + Laplace posterior covariances. PQL is the standard GLMM estimator; it is mildly biased
    for binary data with very few observations per group (use more obs/group there).
    """
    gname, slopes = linpred.groups[0]
    if gname not in given:
        raise ValueError(f"group column {gname!r} not in given=.")
    columns = _columns_of(linpred)
    est, _fixed = columns
    X, offset = _design(columns, given) if (est or _fixed) else (np.zeros((y.size, 0)), np.zeros(y.size))
    names, _ = _parameter_layout(est)
    if not names:
        X = np.ones((y.size, 1))
        names = ["intercept"]
    N, p = X.shape

    re_names = ["intercept"] + list(slopes)
    zcols = [np.ones(N)] + [_numeric_vector(given[s], f"given[{s!r}]", length=N) for s in slopes]
    Z = np.column_stack(zcols)
    q = Z.shape[1]
    levels, g = np.unique(np.asarray(given[gname]), return_inverse=True)
    G = levels.size
    groups = [np.where(g == gi)[0] for gi in range(G)]

    # PQL is mildly biased for binary data with few observations per group. Warn so the user reads the estimates as
    # approximate rather than treating them as a full posterior.
    if link == "logit":
        min_per_group = min((ix.size for ix in groups), default=0)
        if min_per_group < 5:
            import warnings

            warnings.warn(
                "GLMM fit by penalized quasi-likelihood (PQL), which is mildly biased for binary (logit) "
                f"data with few observations per group (smallest group has {min_per_group}). Treat these "
                "estimates as approximate; use more observations per group, or how='mcmc'/'nuts' for a "
                "less-biased posterior.",
                RuntimeWarning,
                stacklevel=2,
            )

    beta = np.zeros(p)
    b = np.zeros((G, q))
    Sigma = np.eye(q) * 0.5
    cov = np.eye(p)
    inner_iterations, inner_converged = [], []
    converged, delta = False, math.inf
    for outer_iteration in range(1, max_iter + 1):
        Sinv = np.linalg.inv(Sigma)
        beta_prev = beta.copy()
        # inner PQL: alternate IRLS fixed-effect and penalized random-effect updates to the joint mode
        this_inner_converged = False
        for inner_iteration in range(1, inner_max_iter + 1):
            b_prev = b.copy()
            eta = np.clip(offset + X @ beta + np.einsum("nq,nq->n", Z, b[g]), -30, 30)
            mu = _link_inv(link, eta)
            w = _irls_weight(link, mu)
            zwork = (eta - offset) + (y - mu) / w  # working response in predictor space
            zb = np.einsum("nq,nq->n", Z, b[g])
            WX = X * w[:, None]
            A = X.T @ WX + 1e-8 * np.eye(p)
            cov = np.linalg.inv(A)
            beta_new = cov @ (X.T @ (w * (zwork - zb)))
            cov_groups = []
            for gi, idx in enumerate(groups):
                Zg, wg = Z[idx], w[idx]
                zg = zwork[idx] - X[idx] @ beta_new
                cov_g = np.linalg.inv(Zg.T @ (wg[:, None] * Zg) + Sinv)
                b[gi] = cov_g @ (Zg.T @ (wg * zg))
                cov_groups.append(cov_g)
            inner_delta = max(
                float(np.max(np.abs(beta_new - beta))),
                float(np.max(np.abs(b - b_prev))),
            )
            if inner_delta < tol:
                beta = beta_new
                this_inner_converged = True
                break
            beta = beta_new
        inner_iterations.append(inner_iteration)
        inner_converged.append(this_inner_converged)
        Sigma_new = (sum(np.outer(b[gi], b[gi]) + cov_groups[gi] for gi in range(G))) / G  # M-step
        # Converge on (beta, Sigma) JOINTLY -- a beta-only test can exit while the variance
        # components are still moving (the same failure as the LMM in balanced designs).
        delta = max(float(np.max(np.abs(beta - beta_prev))), float(np.max(np.abs(Sigma_new - Sigma))))
        Sigma = Sigma_new
        if delta < tol and this_inner_converged:
            converged = True
            break

    result = GLMMResult(
        rv._family.name,
        link,
        names,
        beta,
        cov,
        Sigma,
        b,
        list(levels),
        re_names,
        converged=converged,
        outer_iterations=outer_iteration,
        inner_iterations=inner_iterations,
        inner_converged=inner_converged,
        objective_delta=delta,
    )
    return RandomVariable._bound(None, name=rv._name, result=result)


def _slot_design(slot, given, n):
    """Design pieces for one parameter slot.

    Returns ``(X, offset, names, spec, m0, p0)`` where ``spec`` lets ``predict`` rebuild the slot:
    ``("lp", columns)`` for a linear predictor, ``("free",)`` for a free / Normal-prior intercept,
    or ``("const", c)`` for a fixed value. ``m0``/``p0`` are the Gaussian-prior mean / precision per
    estimated coefficient (precision 0 == flat / MLE).
    """
    if isinstance(slot, _LinearPredictor):
        if slot.groups:
            raise NotImplementedError("location-scale regression does not support group effects yet.")
        columns = _columns_of(slot)
        est, _ = columns
        _validate_supported_priors(est, allowed=("Normal",))
        X, offset = _design(columns, given)
        names, _ = _parameter_layout(est)
        m0, p0 = [], []
        for coef, field in est:
            if isinstance(coef, RandomVariable) and coef._family.name == "Normal":
                m0.append(float(coef._args[0]))
                p0.append(1.0 / float(coef._args[1]) ** 2)
            else:
                m0.append(0.0)
                p0.append(0.0)
        return X, offset, names, ("lp", columns), np.asarray(m0), np.asarray(p0)
    if slot is FREE:
        return np.ones((n, 1)), np.zeros(n), ["intercept"], ("free",), np.zeros(1), np.zeros(1)
    if isinstance(slot, RandomVariable) and slot._family.name == "Normal":
        m0 = np.asarray([float(slot._args[0])])
        p0 = np.asarray([1.0 / float(slot._args[1]) ** 2])
        return np.ones((n, 1)), np.zeros(n), ["intercept"], ("free",), m0, p0
    return np.zeros((n, 0)), float(slot) * np.ones(n), [], ("const", float(slot)), np.zeros(0), np.zeros(0)


def _build_from_spec(spec, given, n):
    """Rebuild ``(X, offset)`` for a stored slot spec at prediction time."""
    kind = spec[0]
    if kind == "lp":
        return _design(spec[1], given)
    if kind == "free":
        return np.ones((n, 1)), np.zeros(n)
    return np.zeros((n, 0)), spec[1] * np.ones(n)


class LocationScaleResult:
    """Heteroskedastic (location-scale) regression: separate mean and log-scale coefficients.

    The scale follows a log link, ``scale = exp(eta_scale)``, so the dispersion can vary with
    covariates (``Normal(mean_pred, free*Field("x") + free)``). ``predict`` returns per-row ``loc``
    and ``scale``.
    """

    def __init__(
        self,
        family,
        names_m,
        names_s,
        beta,
        cov,
        spec_m,
        spec_s,
        *,
        iterations,
        objective_delta,
        termination_reason,
    ):
        self.family = family
        self.names = list(names_m) + list(names_s)
        self.names_mean = list(names_m)
        self.names_scale = list(names_s)
        self.parameter_ids_mean = [f"mean:{i}" for i in range(len(names_m))]
        self.parameter_ids_scale = [f"scale:{i}" for i in range(len(names_s))]
        self.beta = np.asarray(beta, dtype=float).copy()
        self.cov = np.asarray(cov, dtype=float).copy()
        self._pm = len(names_m)
        self._spec_m = spec_m
        self._spec_s = spec_s
        sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        self.coefficients = {
            names_m[i]: {"id": self.parameter_ids_mean[i], "mean": float(beta[i]), "sd": float(sd[i])}
            for i in range(self._pm)
        }
        self.scale_coefficients = {
            names_s[j]: {
                "id": self.parameter_ids_scale[j],
                "mean": float(beta[self._pm + j]),
                "sd": float(sd[self._pm + j]),
            }
            for j in range(len(names_s))
        }
        self.link = "identity"
        self.scale_link = "log"
        self.acceptance_rate = None
        self.predictive = None
        self.converged = True
        self.iterations = int(iterations)
        self.objective_delta = float(objective_delta)
        self.termination_reason = str(termination_reason)

    def predict(self, given, **_):
        """Return ``{'loc': array, 'scale': array}`` at covariates ``given``."""
        n = 1
        for v in (given or {}).values():
            n = max(n, len(np.asarray(v).reshape(-1)))
        beta_m, beta_s = self.beta[: self._pm], self.beta[self._pm :]
        Xm, offm = _build_from_spec(self._spec_m, given or {}, n)
        Xs, offs = _build_from_spec(self._spec_s, given or {}, n)
        loc = offm + (Xm @ beta_m if beta_m.size else np.zeros(n))
        scale = np.exp(np.clip(offs + (Xs @ beta_s if beta_s.size else np.zeros(n)), -20, 20))
        return {"loc": loc, "scale": scale}

    def summary(self):
        """Return separate coefficient summaries for location and scale predictors."""
        return {
            "mean_coefficients": self.coefficients,
            "scale_coefficients": self.scale_coefficients,
            "converged": self.converged,
            "iterations": self.iterations,
            "termination_reason": self.termination_reason,
            "objective_delta": self.objective_delta,
        }


def _locscale_fit(rv, data, given, *, max_iter=200, tol=1e-8):
    """Fit a heteroskedastic Normal/LogNormal: mean (identity) + log-scale linear predictors.

    Maximizes the (optionally ridge-penalized) log-likelihood with analytic gradients; the
    coefficient covariance is the Laplace approximation (inverse Hessian at the optimum).
    """
    from scipy.optimize import minimize

    fam = rv._family.name
    y = np.asarray(data, dtype=float)
    if fam == "LogNormal":
        if np.any(y <= 0):
            raise ValueError("LogNormal regression requires positive observations.")
        w = np.log(y)  # log y ~ Normal(mean, scale); fit on the log scale
    else:
        w = y
    n = w.size

    Xm, offm, names_m, spec_m, m0m, p0m = _slot_design(rv._args[0], given, n)
    Xs, offs, names_s, spec_s, m0s, p0s = _slot_design(rv._args[1], given, n)
    pm, ps = Xm.shape[1], Xs.shape[1]

    def unpack(theta):
        return theta[:pm], theta[pm:]

    def nll(theta):
        bm, bs = unpack(theta)
        mu = offm + (Xm @ bm if pm else 0.0)
        eta = np.clip(offs + (Xs @ bs if ps else 0.0), -20.0, 20.0)
        r = w - mu
        inv2 = np.exp(-2.0 * eta)
        val = np.sum(eta + 0.5 * r * r * inv2)
        val += 0.5 * np.sum(p0m * (bm - m0m) ** 2) + 0.5 * np.sum(p0s * (bs - m0s) ** 2)
        return val

    def grad(theta):
        bm, bs = unpack(theta)
        mu = offm + (Xm @ bm if pm else 0.0)
        raw_eta = offs + (Xs @ bs if ps else 0.0)
        eta = np.clip(raw_eta, -20.0, 20.0)
        active = ((raw_eta > -20.0) & (raw_eta < 20.0)).astype(float)
        r = w - mu
        inv2 = np.exp(-2.0 * eta)
        gm = (-Xm.T @ (r * inv2) + p0m * (bm - m0m)) if pm else np.zeros(0)
        gs = (Xs.T @ ((1.0 - r * r * inv2) * active) + p0s * (bs - m0s)) if ps else np.zeros(0)
        return np.concatenate([gm, gs])

    # warm start: OLS mean, unit scale
    theta0 = np.zeros(pm + ps)
    if pm:
        try:
            theta0[:pm] = np.linalg.lstsq(Xm, w - offm, rcond=None)[0]
        except np.linalg.LinAlgError:
            pass
    res = minimize(nll, theta0, jac=grad, method="L-BFGS-B", options={"maxiter": max_iter, "ftol": tol})
    if not res.success or not math.isfinite(float(res.fun)) or not np.isfinite(res.x).all():
        raise RuntimeError(f"location-scale regression optimization failed: {res.message}")
    theta = res.x
    bm, bs = unpack(theta)

    # Laplace covariance from the analytic Hessian at the optimum
    mu = offm + (Xm @ bm if pm else 0.0)
    raw_eta = offs + (Xs @ bs if ps else 0.0)
    eta = np.clip(raw_eta, -20.0, 20.0)
    active = ((raw_eta > -20.0) & (raw_eta < 20.0)).astype(float)
    r = w - mu
    inv2 = np.exp(-2.0 * eta)
    Hmm = (Xm.T @ (Xm * inv2[:, None]) + np.diag(p0m)) if pm else np.zeros((0, 0))
    Hss = Xs.T @ (Xs * (2.0 * r * r * inv2 * active)[:, None]) + np.diag(p0s) if ps else np.zeros((0, 0))
    Hms = (2.0 * Xm.T @ (Xs * (r * inv2 * active)[:, None])) if (pm and ps) else np.zeros((pm, ps))
    H = np.block([[Hmm, Hms], [Hms.T, Hss]])
    try:
        cov = np.linalg.inv(H + 1e-8 * np.eye(H.shape[0]))
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)

    result = LocationScaleResult(
        fam,
        names_m,
        names_s,
        theta,
        cov,
        spec_m,
        spec_s,
        iterations=int(res.nit),
        objective_delta=float(np.linalg.norm(res.jac, ord=np.inf)),
        termination_reason=str(res.message),
    )
    return RandomVariable._bound(None, name=rv._name, result=result)


def _coord_descent(X, target, p0, m0, l1, loc1, max_iter, tol, *, fixed_sigma2=None):
    """Cyclic coordinate descent for penalized least squares with per-coefficient L1/L2.

    Minimizes the Normal negative log posterior, scaling coefficient-prior
    penalties by the fixed or iteratively estimated residual variance.
    Each coordinate has a closed-form soft-threshold update, so a ``free`` coefficient reduces to
    the OLS update, a Normal prior to ridge, and a Laplace prior to lasso (the families mix freely).
    """
    n, p = X.shape
    beta = np.zeros(p)
    z = (X * X).sum(axis=0)  # squared column norms
    resid = target - X @ beta
    sigma2 = (
        _positive_float(fixed_sigma2, "fixed residual variance")
        if fixed_sigma2 is not None
        else max(float(resid @ resid) / n, 1.0e-8)
    )
    converged, delta = False, math.inf
    for iteration in range(1, max_iter + 1):
        delta = 0.0
        for j in range(p):
            resid = resid + X[:, j] * beta[j]  # partial residual excluding coordinate j
            # In RSS units, a coefficient prior's negative log density is
            # multiplied by sigma^2.  This keeps Normal/Laplace prior strength
            # invariant to the response's declared likelihood scale.
            a = z[j] + sigma2 * p0[j]
            c = X[:, j] @ resid + sigma2 * p0[j] * m0[j]
            threshold = sigma2 * l1[j]
            if a <= 0.0:
                bj = 0.0
            else:
                d = a * loc1[j] - c
                if d > threshold:
                    u = -(d - threshold) / a
                elif d < -threshold:
                    u = -(d + threshold) / a
                else:
                    u = 0.0
                bj = loc1[j] + u
            delta = max(delta, abs(bj - beta[j]))
            beta[j] = bj
            resid = resid - X[:, j] * beta[j]
        sigma_delta = 0.0
        if fixed_sigma2 is None:
            new_sigma2 = max(float(resid @ resid) / n, 1.0e-8)
            sigma_delta = abs(new_sigma2 - sigma2)
            sigma2 = new_sigma2
        delta = max(delta, sigma_delta)
        if delta < tol:
            converged = True
            break
    return beta, sigma2, converged, iteration, delta


def _quantile_fit(rv: RandomVariable, data, given, tau: float) -> RandomVariable:
    """Fit the conditional ``tau``-quantile by minimizing the pinball (check) loss.

    Distribution-free: no Gaussian assumption is used. The check-loss minimization is the
    exact linear program ``min tau*sum(u) + (1-tau)*sum(v)`` subject to
    ``X beta + u - v = y - offset``, ``u, v >= 0``, solved with HiGHS. The returned
    :class:`RegressionResult` predicts the fitted quantile through the identity link;
    coefficient standard errors for quantile regression need a bootstrap, so
    uncertainty is explicitly unavailable rather than represented by zero variance.
    """
    if not 0.0 < tau < 1.0:
        raise ValueError(f"quantile must be in (0, 1); got {tau}.")
    if rv._family.name != "Normal":
        raise NotImplementedError("quantile regression requires a Normal (continuous) response.")
    from scipy import sparse
    from scipy.optimize import linprog

    linpred = next(a for a in rv._args if isinstance(a, _LinearPredictor))
    if linpred.groups:
        raise NotImplementedError("quantile regression does not support grouped effects.")
    if rv._args[1] is not FREE:
        raise NotImplementedError(
            "quantile regression requires a free scale slot; declared scale semantics are unused."
        )
    columns = _columns_of(linpred)
    est, _fixed = columns
    if any(isinstance(coef, RandomVariable) for coef, _field in est):
        raise NotImplementedError("quantile regression does not implement coefficient-prior semantics.")
    X, offset = _design(columns, given)
    y = np.asarray(data, dtype=float)
    n, p = X.shape
    c = np.concatenate([np.zeros(p), tau * np.ones(n), (1.0 - tau) * np.ones(n)])
    a_eq = sparse.hstack([sparse.csr_matrix(X), sparse.eye(n), -sparse.eye(n)], format="csr")
    bounds = [(None, None)] * p + [(0.0, None)] * (2 * n)
    res = linprog(c, A_eq=a_eq, b_eq=y - offset, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"quantile regression LP did not converge: {res.message}")
    beta = np.asarray(res.x[:p], dtype=float)
    names, idx_of = _parameter_layout(est)
    result = RegressionResult(
        names,
        idx_of,
        beta,
        None,
        float("nan"),
        columns,
        link="identity",
        converged=True,
        iterations=int(getattr(res, "nit", 0)),
        termination_reason="optimal",
    )
    result.quantile = float(tau)
    return RandomVariable._bound(None, name=rv._name, result=result)


def regression_fit(
    rv: RandomVariable,
    data,
    *,
    given=None,
    how="auto",
    max_iter: int = 100,
    inner_max_iter: int = 100,
    tol: float = 1e-9,
    quantile=None,
    l2=0.0,
    **unknown,
) -> RandomVariable:
    """Fit a PPL regression expression using the appropriate linear-model route."""
    if unknown:
        raise TypeError(f"unsupported regression fit control(s): {', '.join(sorted(unknown))}")
    if how not in {"auto", "map"}:
        raise NotImplementedError(
            f"regression's specialized backend implements point-estimate how='map' only, not how={how!r}."
        )
    max_iter = _positive_int(max_iter, "max_iter")
    inner_max_iter = _positive_int(inner_max_iter, "inner_max_iter")
    tol = _positive_float(tol, "tol")
    try:
        l2 = float(l2)
    except (TypeError, ValueError) as exc:
        raise ValueError("l2 must be a finite non-negative scalar.") from exc
    if not math.isfinite(l2) or l2 < 0.0:
        raise ValueError("l2 must be a finite non-negative scalar.")
    y, given = _validate_conditional_data(rv, data, given)
    fam = rv._family.name
    _validate_response(fam, y)
    linpred0 = next((a for a in rv._args if isinstance(a, _LinearPredictor)), None)
    if quantile is not None:  # pinball-loss quantile regression (same linear-predictor syntax)
        if how != "auto":
            raise NotImplementedError("quantile regression is a separate loss route and requires how='auto'.")
        return _quantile_fit(rv, y, given, float(quantile))
    if linpred0 is not None and linpred0.groups:  # mixed-effects model
        est, _fixed = _columns_of(linpred0)
        if any(isinstance(coef, RandomVariable) for coef, _field in est):
            raise NotImplementedError(
                "mixed-model coefficient priors are not implemented by the current LMM/GLMM routes."
            )
        _parameter_layout(est)
        if fam == "Normal":
            if rv._args[1] is not FREE:
                raise NotImplementedError("the LMM route currently requires a free residual-scale slot.")
            return _lmm_fit(rv, y, given, linpred0, max_iter, tol)
        glmm_link = _LINK.get(fam)
        if glmm_link is None:
            raise NotImplementedError(f"mixed-effects models support {sorted(_LINK)} responses (got {fam!r}).")
        return _glmm_fit(rv, y, given, linpred0, glmm_link, max_iter, inner_max_iter, tol)
    # heteroskedastic location-scale: a linear predictor in the *scale* slot (log link)
    if fam in ("Normal", "LogNormal") and isinstance(rv._args[1], _LinearPredictor):
        return _locscale_fit(rv, y, given, max_iter=max_iter, tol=tol)
    link = _LINK.get(fam)
    if link is None:
        raise NotImplementedError(f"regression for family {fam} is not supported (have {sorted(_LINK)}).")
    linpred = next(a for a in rv._args if isinstance(a, _LinearPredictor))
    scale = rv._args[1] if fam == "Normal" else None
    columns = _columns_of(linpred)
    est, _fixed = columns
    _validate_supported_priors(est)
    names, idx_of = _parameter_layout(est)
    X, offset = _design(columns, given)
    N, p = X.shape
    if p == 0:
        raise ValueError("regression has no estimated coefficient columns.")

    # coefficient priors per slot: Normal -> L2 (ridge), Laplace -> L1 (lasso), free -> none
    m0, p0 = np.zeros(p), np.zeros(p)  # L2 mean / precision
    l1, loc1 = np.zeros(p), np.zeros(p)  # L1 strength / center
    for i, (coef, field) in enumerate(est):
        if isinstance(coef, RandomVariable) and coef._family.name == "Normal":
            m0[i] = float(coef._args[0])
            p0[i] = 1.0 / float(coef._args[1]) ** 2
        elif isinstance(coef, RandomVariable) and coef._family.name == "Laplace":
            loc1[i] = float(coef._args[0])
            l1[i] = 1.0 / float(coef._args[1])  # Laplace scale b -> L1 penalty 1/b
    if l2 > 0.0:  # global ridge added to every non-intercept coefficient (elastic net with Laplace priors)
        not_intercept = np.array([field is not None for (_coef, field) in est], dtype=float)
        p0 = p0 + l2 * not_intercept
    P0 = np.diag(p0)

    if np.any(l1 > 0.0) or l2 > 0.0:  # L1 and/or global L2 -> coordinate descent (lasso / ridge / elastic net)
        if fam != "Normal":
            raise NotImplementedError("penalized (L1 / elastic-net) regression is supported for Normal responses.")
        if isinstance(scale, RandomVariable):
            raise NotImplementedError("a prior-bearing Normal scale is not implemented by regression.")
        fixed_sigma2 = None if scale is FREE else _positive_float(scale, "Normal scale") ** 2
        beta, sigma2, converged, iterations, objective_delta = _coord_descent(
            X,
            y - offset,
            p0,
            m0,
            l1,
            loc1,
            max_iter,
            tol,
            fixed_sigma2=fixed_sigma2,
        )
        sigma = math.sqrt(sigma2)
        cov = None if np.any(l1 > 0.0) else np.linalg.inv(X.T @ X / sigma2 + P0)
        result = RegressionResult(
            names,
            idx_of,
            beta,
            cov,
            sigma,
            columns,
            link="identity",
            converged=converged,
            iterations=iterations,
            termination_reason="tolerance" if converged else "max_iterations",
            objective_delta=objective_delta,
        )
        return RandomVariable._bound(None, name=rv._name, result=result)

    # IRLS / Fisher scoring (one step is OLS for the Gaussian identity link).
    # For a Normal response the likelihood precision is X'X / sigma^2, so a Normal coefficient
    # prior (precision P0, in 1/coef^2 units) must enter the working normal equations -- which
    # are in X'X units -- scaled by sigma^2: (X'X + sigma^2 P0) beta = X'y + sigma^2 P0 m0 is
    # the exact Gaussian posterior mode. Unscaled, the "Bayesian" fit is ridge-at-sigma=1.
    # Non-Gaussian GLM families (Bernoulli/Poisson) have unit dispersion, so X'WX is already the
    # likelihood precision and P0 enters unscaled (the exact penalized-likelihood / MAP mode).
    if fam == "Normal" and isinstance(scale, RandomVariable):
        raise NotImplementedError("a prior-bearing Normal scale is not implemented by regression.")
    sigma_fixed = fam != "Normal" or scale is not FREE
    sigma2_work = _positive_float(scale, "Normal scale") ** 2 if (fam == "Normal" and sigma_fixed) else 1.0
    beta = np.zeros(p)
    cov = np.eye(p)
    converged, objective_delta = False, math.inf
    for iteration in range(1, max_iter + 1):
        eta = offset + X @ beta
        mu = _link_inv(link, eta)
        W = _irls_weight(link, mu)
        z = (eta - offset) + (y - mu) / W  # working response (predictor space)
        if fam == "Normal" and not sigma_fixed and np.any(p0 > 0.0):
            # free sigma: plug in the working residual variance (stabilizes as beta converges)
            sigma2_work = max(float((y - mu) @ (y - mu)) / N, 1e-8)
        WX = X * W[:, None]
        A = X.T @ WX + sigma2_work * P0
        cov = np.linalg.inv(A)
        new_beta = cov @ (X.T @ (W * z) + sigma2_work * (P0 @ m0))
        objective_delta = float(np.max(np.abs(new_beta - beta)))
        if objective_delta < tol:
            beta = new_beta
            converged = True
            break
        beta = new_beta

    if fam == "Normal":  # residual scale
        if sigma_fixed:
            sigma = float(scale)
        else:
            resid = y - (offset + X @ beta)
            sigma = float(np.sqrt(max(resid @ resid / N, 1e-8)))
        # coef cov scales with sigma^2: with the sigma^2-scaled prior above this is exactly
        # (X'X / sigma^2 + P0)^-1, the Gaussian posterior covariance (OLS curvature when P0 = 0)
        cov = cov * (sigma**2)
    else:
        sigma = float("nan")

    result = RegressionResult(
        names,
        idx_of,
        beta,
        cov,
        sigma,
        columns,
        link=link,
        converged=converged,
        iterations=iteration,
        termination_reason="tolerance" if converged else "max_iterations",
        objective_delta=objective_delta,
    )
    return RandomVariable._bound(None, name=rv._name, result=result)
