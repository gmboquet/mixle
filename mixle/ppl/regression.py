"""Regression / GLMs for mixle.ppl.

A linear predictor in a parameter slot makes a model a regression; the outer family sets the
link:

    Normal(a*Field("x") + b, sigma)   identity link  -> linear regression
    Bernoulli(a*Field("x") + b)       logit link     -> logistic regression
    Poisson(a*Field("x") + b)         log link       -> Poisson regression

Coefficients may be ``free`` or may carry Normal penalty handles.  Fitting is
IRLS/Fisher scoring for a likelihood or penalized-likelihood point estimate.
For Normal responses this module uses the ridge/penalized-least-squares
convention documented in the book and reports a scale-adjusted
inverse-curvature diagnostic; it is not a full Gaussian-prior posterior
unless the likelihood and prior precisions are scaled consistently.  Fit with
``.fit(y, given={"x": xs})``.
"""

from __future__ import annotations

import numpy as np

from mixle.ppl.core import RandomVariable, _LinearPredictor
from mixle.ppl.core import free as FREE

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
    """Regression point estimate with coefficient curvature diagnostics."""

    def __init__(self, names, idx_of, beta, cov, sigma, columns, link="identity"):
        self.names = names  # column names (covariates + 'intercept')
        self.beta = beta  # fitted coefficients
        self.cov = cov  # route-specific covariance / inverse-curvature diagnostic
        self.sigma = float(sigma)
        self.link = link
        self._idx_of = idx_of  # id(coef handle) -> column index
        self._columns = columns  # list of (kind, payload) for predict
        self.coefficients = {
            names[i]: {"mean": float(beta[i]), "sd": float(np.sqrt(cov[i, i]))} for i in range(len(names))
        }
        self.acceptance_rate = None
        self.predictive = None

    def _resolve(self, param):
        if isinstance(param, str):
            return self.names.index(param)
        if isinstance(param, (int, np.integer)):
            return int(param)
        return self._idx_of[id(param)]

    def samples(self, param=None, n: int = 4000, rng=None):
        """Draw from the Gaussian coefficient approximation represented by ``beta`` and ``cov``."""
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
        rng = rng or np.random.RandomState()
        out = np.empty((n, eta.size))
        for k in range(n):
            beta_k = rng.multivariate_normal(self.beta, self.cov)
            out[k] = _link_inv(self.link, offset + X @ beta_k)
        return out

    def summary(self):
        """Return coefficient summaries and residual scale metadata."""
        return {"coefficients": self.coefficients, "sigma": self.sigma}

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
    n = None
    for _, field in est + fixed:
        if field is not None:
            n = len(np.asarray(given[field.name]).reshape(-1))
            break
    if n is None:  # intercept-only fixed part (e.g. a random-effects-only model): size from given
        for arr in (given or {}).values():
            n = len(np.asarray(arr).reshape(-1))
            break
    if n is None:
        raise ValueError("need at least one covariate or a given= array to size the design matrix.")
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
        self.random_cov = np.asarray(Sigma)  # random-effects covariance (q x q)
        self.random_names = re_names  # ['intercept', slope names...]
        self.tau = float(np.sqrt(Sigma[0, 0]))  # random-intercept sd (back-compat)
        self.sigma = float(sigma)  # residual sd
        # per-group effects: intercept for back-compat, full vector under group_effects_full
        self.group_effects = {lv: float(b[i, 0]) for i, lv in enumerate(group_levels)}
        self.group_effects_full = {lv: b[i] for i, lv in enumerate(group_levels)}
        self.coefficients = {
            names[i]: {"mean": float(beta[i]), "sd": float(np.sqrt(cov[i, i]))} for i in range(len(names))
        }
        self.acceptance_rate = None
        self.predictive = None

    def summary(self):
        """Return fixed effects, random-effects covariance, scale, and group count."""
        return {
            "coefficients": self.coefficients,
            "random_cov": self.random_cov,
            "sigma": self.sigma,
            "n_groups": len(self.group_effects),
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
    names = [f.name if f is not None else "intercept" for _, f in est]
    if not names:
        X = np.ones((y.size, 1))
        names = ["intercept"]
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
            cov_g = np.linalg.inv(Sinv + Zg.T @ Zg / sigma2)  # E-step posterior of b_g
            b_g = cov_g @ (Zg.T @ rg / sigma2)
            b[gi] = b_g
            SS += np.outer(b_g, b_g) + cov_g
            pred = Zg @ b_g
            err2 += float((rg - pred) @ (rg - pred))
            trace_term += float(np.trace(Zg @ cov_g @ Zg.T))
        Sigma = SS / G  # M-step
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


class GLMMResult:
    """Generalized linear mixed model: fixed-effect coefficients (on the link scale) + the
    random-effects covariance + per-group effects. No residual scale (the family sets dispersion)."""

    def __init__(self, family, link, names, beta, cov, Sigma, b, group_levels, re_names):
        self.family = family
        self.link = link
        self.names = names
        self.beta = beta
        self.cov = cov
        self.random_cov = np.asarray(Sigma)
        self.random_names = re_names
        self.tau = float(np.sqrt(Sigma[0, 0]))  # random-intercept sd
        self.group_effects = {lv: float(b[i, 0]) for i, lv in enumerate(group_levels)}
        self.group_effects_full = {lv: b[i] for i, lv in enumerate(group_levels)}
        self.coefficients = {
            names[i]: {"mean": float(beta[i]), "sd": float(np.sqrt(max(cov[i, i], 0.0)))} for i in range(len(names))
        }
        self.acceptance_rate = None
        self.predictive = None

    def summary(self):
        """Return fixed effects, random-effects covariance, link, and group count."""
        return {
            "coefficients": self.coefficients,
            "random_cov": self.random_cov,
            "link": self.link,
            "n_groups": len(self.group_effects),
        }


def _glmm_fit(rv, y, given, linpred, link, max_iter, tol):
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
    names = [f.name if f is not None else "intercept" for _, f in est]
    if not names:
        X = np.ones((y.size, 1))
        names = ["intercept"]
    N, p = X.shape

    re_names = ["intercept"] + list(slopes)
    zcols = [np.ones(N)] + [np.asarray(given[s], dtype=float).reshape(-1) for s in slopes]
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
    for _ in range(max_iter):
        Sinv = np.linalg.inv(Sigma)
        beta_prev = beta.copy()
        # inner PQL: alternate IRLS fixed-effect and penalized random-effect updates to the joint mode
        for _inner in range(100):
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
            if np.max(np.abs(beta_new - beta)) < tol:
                beta = beta_new
                break
            beta = beta_new
        Sigma = (sum(np.outer(b[gi], b[gi]) + cov_groups[gi] for gi in range(G))) / G  # M-step
        if np.max(np.abs(beta - beta_prev)) < tol:
            break

    result = GLMMResult(rv._family.name, link, names, beta, cov, Sigma, b, list(levels), re_names)
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
        X, offset = _design(columns, given)
        names, m0, p0 = [], [], []
        for coef, field in est:
            names.append(field.name if field is not None else "intercept")
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

    def __init__(self, family, names_m, names_s, beta, cov, spec_m, spec_s):
        self.family = family
        self.names = list(names_m) + list(names_s)
        self.names_mean = list(names_m)
        self.names_scale = list(names_s)
        self.beta = beta
        self.cov = cov
        self._pm = len(names_m)
        self._spec_m = spec_m
        self._spec_s = spec_s
        sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        self.coefficients = {names_m[i]: {"mean": float(beta[i]), "sd": float(sd[i])} for i in range(self._pm)}
        self.scale_coefficients = {
            names_s[j]: {"mean": float(beta[self._pm + j]), "sd": float(sd[self._pm + j])} for j in range(len(names_s))
        }
        self.link = "identity"
        self.scale_link = "log"
        self.acceptance_rate = None
        self.predictive = None

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
        return {"mean_coefficients": self.coefficients, "scale_coefficients": self.scale_coefficients}


def _locscale_fit(rv, data, given, *, max_iter=200, tol=1e-8):
    """Fit a heteroskedastic Normal/LogNormal: mean (identity) + log-scale linear predictors.

    Maximizes the (optionally ridge-penalized) log-likelihood with analytic gradients; the
    coefficient covariance is the Laplace approximation (inverse Hessian at the optimum).
    """
    from scipy.optimize import minimize

    fam = rv._family.name
    y = np.asarray(data, dtype=float).reshape(-1)
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
        eta = np.clip(offs + (Xs @ bs if ps else 0.0), -20, 20)
        r = w - mu
        inv2 = np.exp(-2.0 * eta)
        val = np.sum(eta + 0.5 * r * r * inv2)
        val += 0.5 * np.sum(p0m * (bm - m0m) ** 2) + 0.5 * np.sum(p0s * (bs - m0s) ** 2)
        return val

    def grad(theta):
        bm, bs = unpack(theta)
        mu = offm + (Xm @ bm if pm else 0.0)
        eta = np.clip(offs + (Xs @ bs if ps else 0.0), -20, 20)
        r = w - mu
        inv2 = np.exp(-2.0 * eta)
        gm = (-Xm.T @ (r * inv2) + p0m * (bm - m0m)) if pm else np.zeros(0)
        gs = (Xs.T @ (1.0 - r * r * inv2) + p0s * (bs - m0s)) if ps else np.zeros(0)
        return np.concatenate([gm, gs])

    # warm start: OLS mean, unit scale
    theta0 = np.zeros(pm + ps)
    if pm:
        try:
            theta0[:pm] = np.linalg.lstsq(Xm, w - offm, rcond=None)[0]
        except np.linalg.LinAlgError:
            pass
    res = minimize(nll, theta0, jac=grad, method="L-BFGS-B", options={"maxiter": max_iter, "ftol": tol})
    theta = res.x
    bm, bs = unpack(theta)

    # Laplace covariance from the analytic Hessian at the optimum
    mu = offm + (Xm @ bm if pm else 0.0)
    eta = np.clip(offs + (Xs @ bs if ps else 0.0), -20, 20)
    r = w - mu
    inv2 = np.exp(-2.0 * eta)
    Hmm = (Xm.T @ (Xm * inv2[:, None]) + np.diag(p0m)) if pm else np.zeros((0, 0))
    Hss = (Xs.T @ (Xs * (2.0 * r * r * inv2)[:, None]) + np.diag(p0s)) if ps else np.zeros((0, 0))
    Hms = (2.0 * Xm.T @ (Xs * (r * inv2)[:, None])) if (pm and ps) else np.zeros((pm, ps))
    H = np.block([[Hmm, Hms], [Hms.T, Hss]])
    try:
        cov = np.linalg.inv(H + 1e-8 * np.eye(H.shape[0]))
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)

    result = LocationScaleResult(fam, names_m, names_s, theta, cov, spec_m, spec_s)
    return RandomVariable._bound(None, name=rv._name, result=result)


def _coord_descent(X, target, p0, m0, l1, loc1, max_iter, tol: float = 1e-8):
    """Cyclic coordinate descent for penalized least squares with per-coefficient L1/L2.

    Minimizes ``0.5||target - X beta||^2 + sum_i [0.5 p0_i (beta_i - m0_i)^2 + l1_i |beta_i - loc1_i|]``.
    Each coordinate has a closed-form soft-threshold update, so a ``free`` coefficient reduces to
    the OLS update, a Normal prior to ridge, and a Laplace prior to lasso (the families mix freely).
    """
    n, p = X.shape
    beta = np.zeros(p)
    z = (X * X).sum(axis=0)  # squared column norms
    resid = target - X @ beta
    for _ in range(max(int(max_iter), 200)):
        delta = 0.0
        for j in range(p):
            resid = resid + X[:, j] * beta[j]  # partial residual excluding coordinate j
            a = z[j] + p0[j]
            c = X[:, j] @ resid + p0[j] * m0[j]
            if a <= 0.0:
                bj = 0.0
            else:
                d = a * loc1[j] - c  # objective in u = beta_j - loc1[j]: 0.5 a u^2 + d u + l1|u|
                if d > l1[j]:
                    u = -(d - l1[j]) / a
                elif d < -l1[j]:
                    u = -(d + l1[j]) / a
                else:
                    u = 0.0
                bj = loc1[j] + u
            delta = max(delta, abs(bj - beta[j]))
            beta[j] = bj
            resid = resid - X[:, j] * beta[j]
        if delta < tol:
            break
    return beta


def _quantile_fit(rv: RandomVariable, data, given, tau: float) -> RandomVariable:
    """Fit the conditional ``tau``-quantile by minimizing the pinball (check) loss.

    Distribution-free: no Gaussian assumption is used. The check-loss minimization is the
    exact linear program ``min tau*sum(u) + (1-tau)*sum(v)`` subject to
    ``X beta + u - v = y - offset``, ``u, v >= 0``, solved with HiGHS. The returned
    :class:`RegressionResult` predicts the fitted quantile through the identity link;
    coefficient standard errors for quantile regression need a bootstrap, so ``cov`` is
    left at zero rather than reporting a misleading OLS curvature.
    """
    if not 0.0 < tau < 1.0:
        raise ValueError(f"quantile must be in (0, 1); got {tau}.")
    if rv._family.name != "Normal":
        raise NotImplementedError("quantile regression requires a Normal (continuous) response.")
    from scipy import sparse
    from scipy.optimize import linprog

    linpred = next(a for a in rv._args if isinstance(a, _LinearPredictor))
    columns = _columns_of(linpred)
    est, _fixed = columns
    X, offset = _design(columns, given)
    y = np.asarray(data, dtype=float).reshape(-1)
    n, p = X.shape
    c = np.concatenate([np.zeros(p), tau * np.ones(n), (1.0 - tau) * np.ones(n)])
    a_eq = sparse.hstack([sparse.csr_matrix(X), sparse.eye(n), -sparse.eye(n)], format="csr")
    bounds = [(None, None)] * p + [(0.0, None)] * (2 * n)
    res = linprog(c, A_eq=a_eq, b_eq=y - offset, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"quantile regression LP did not converge: {res.message}")
    beta = np.asarray(res.x[:p], dtype=float)
    names, idx_of = [], {}
    for i, (coef, field) in enumerate(est):
        names.append(field.name if field is not None else "intercept")
        idx_of[id(coef)] = i
    result = RegressionResult(names, idx_of, beta, np.zeros((p, p)), float("nan"), columns, link="identity")
    result.quantile = float(tau)
    return RandomVariable._bound(None, name=rv._name, result=result)


def regression_fit(
    rv: RandomVariable, data, *, given=None, max_iter: int = 100, tol: float = 1e-9, quantile=None, l2=0.0, **_
) -> RandomVariable:
    """Fit a PPL regression expression using the appropriate linear-model route."""
    linpred0 = next((a for a in rv._args if isinstance(a, _LinearPredictor)), None)
    if quantile is not None:  # pinball-loss quantile regression (same linear-predictor syntax)
        return _quantile_fit(rv, data, given or {}, float(quantile))
    if linpred0 is not None and linpred0.groups:  # mixed-effects model
        y = np.asarray(data, float).reshape(-1)
        if rv._family.name == "Normal":
            return _lmm_fit(rv, y, given or {}, linpred0, max_iter, tol)
        glmm_link = _LINK.get(rv._family.name)
        if glmm_link is None:
            raise NotImplementedError(
                f"mixed-effects models support {sorted(_LINK)} responses (got {rv._family.name!r})."
            )
        return _glmm_fit(rv, y, given or {}, linpred0, glmm_link, max(max_iter, 100), tol)
    fam = rv._family.name
    # heteroskedastic location-scale: a linear predictor in the *scale* slot (log link)
    if fam in ("Normal", "LogNormal") and isinstance(rv._args[1], _LinearPredictor):
        return _locscale_fit(rv, data, given or {}, max_iter=max(max_iter, 200))
    link = _LINK.get(fam)
    if link is None:
        raise NotImplementedError(f"regression for family {fam} is not supported (have {sorted(_LINK)}).")
    linpred = next(a for a in rv._args if isinstance(a, _LinearPredictor))
    scale = rv._args[1] if fam == "Normal" else None
    given = given or {}
    y = np.asarray(data, dtype=float).reshape(-1)
    columns = _columns_of(linpred)
    est, _fixed = columns
    X, offset = _design(columns, given)
    N, p = X.shape

    # coefficient priors per slot: Normal -> L2 (ridge), Laplace -> L1 (lasso), free -> none
    m0, p0 = np.zeros(p), np.zeros(p)  # L2 mean / precision
    l1, loc1 = np.zeros(p), np.zeros(p)  # L1 strength / center
    idx_of, names = {}, []
    for i, (coef, field) in enumerate(est):
        names.append(field.name if field is not None else "intercept")
        idx_of[id(coef)] = i
        if isinstance(coef, RandomVariable) and coef._family.name == "Normal":
            m0[i] = float(coef._args[0])
            p0[i] = 1.0 / float(coef._args[1]) ** 2
        elif isinstance(coef, RandomVariable) and coef._family.name == "Laplace":
            loc1[i] = float(coef._args[0])
            l1[i] = 1.0 / float(coef._args[1])  # Laplace scale b -> L1 penalty 1/b
    if l2 > 0.0:  # global ridge added to every non-intercept coefficient (elastic net with Laplace priors)
        not_intercept = np.array([field is not None for (_coef, field) in est], dtype=float)
        p0 = p0 + float(l2) * not_intercept
    P0 = np.diag(p0)

    if np.any(l1 > 0.0) or l2 > 0.0:  # L1 and/or global L2 -> coordinate descent (lasso / ridge / elastic net)
        if fam != "Normal":
            raise NotImplementedError("penalized (L1 / elastic-net) regression is supported for Normal responses.")
        beta = _coord_descent(X, y - offset, p0, m0, l1, loc1, max_iter)
        resid = y - (offset + X @ beta)
        sigma = float(np.sqrt(max(resid @ resid / N, 1e-8)))
        # L1 coefficient standard errors need a bootstrap; leave cov at zero (no OLS curvature)
        result = RegressionResult(names, idx_of, beta, np.zeros((p, p)), sigma, columns, link="identity")
        return RandomVariable._bound(None, name=rv._name, result=result)

    # IRLS / Fisher scoring (one step is OLS for the Gaussian identity link)
    beta = np.zeros(p)
    cov = np.eye(p)
    for _ in range(max_iter):
        eta = offset + X @ beta
        mu = _link_inv(link, eta)
        W = _irls_weight(link, mu)
        z = (eta - offset) + (y - mu) / W  # working response (predictor space)
        WX = X * W[:, None]
        A = X.T @ WX + P0
        cov = np.linalg.inv(A)
        new_beta = cov @ (X.T @ (W * z) + P0 @ m0)
        if np.max(np.abs(new_beta - beta)) < tol:
            beta = new_beta
            break
        beta = new_beta

    if fam == "Normal":  # residual scale
        sigma_fixed = not (scale is FREE or isinstance(scale, RandomVariable))
        if sigma_fixed:
            sigma = float(scale)
        else:
            resid = y - (offset + X @ beta)
            sigma = float(np.sqrt(max(resid @ resid / N, 1e-8)))
        cov = cov * (sigma**2)  # OLS coef cov scales with sigma^2
    else:
        sigma = float("nan")

    result = RegressionResult(names, idx_of, beta, cov, sigma, columns, link=link)
    return RandomVariable._bound(None, name=rv._name, result=result)
