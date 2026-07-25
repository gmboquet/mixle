"""Black-box likelihood/posterior curvature approximations for composable model parameters.

The conjugate path needs a registered closed form; the autograd VI/NUTS path needs a per-family
from-parameters scorer (so it covers flat models + mixtures of leaves). Neither works for an arbitrary
composable model. This module does: it treats a *fitted* model's parameters as the latent, flattens them
to an UNCONSTRAINED vector (positive params via log, unit via logit, simplex via softmax), and fits a
Gaussian approximation in that space from a finite-difference Hessian of the model's own
``seq_log_density`` -- which every model has. With an explicit log prior this is a Laplace posterior;
without one it is a likelihood-curvature approximation and is labelled accordingly.

    post = laplace_posterior(fitted_model, data)
    post.sample(...)        # parameter draws (a fitted model per draw)
    post.cov                # unconstrained-space posterior covariance

Coverage is the parameter round-trip in :func:`_flatten` -- the scalar exponential-family leaves, the
``Categorical`` simplex, plus ``Composite``, ``Mixture`` and ``HeterogeneousBayesianNetwork``
(recursively), so heterogeneous records, mixtures-of-anything, and learned Bayesian networks (categorical
CPTs + conditional-linear-Gaussian coefficients) are covered out of the box. It is extensible exactly
like ``register_family``: add a leaf's
(extract, rebuild) and every composite over it works. A model whose structure is not yet flattenable
raises a clear error rather than returning a wrong answer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


# --- unconstrained transforms: constrained value(s) <-> real coordinates ---------------------------
def _pos_to_u(x):
    value = float(x)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("positive parameters must be finite and strictly positive at the Laplace mode")
    return [float(np.log(value))]


def _pos_from_u(u):
    return float(np.exp(u[0])), u[1:]


def _real_to_u(x):
    return [float(x)]


def _real_from_u(u):
    return float(u[0]), u[1:]


def _unit_to_u(p):
    p = float(p)
    if not np.isfinite(p) or not 0 < p < 1:
        raise ValueError("unit-interval parameters must be strictly inside (0, 1) at the Laplace mode")
    return [float(np.log(p / (1 - p)))]


def _unit_from_u(u):
    return float(1.0 / (1.0 + np.exp(-u[0]))), u[1:]


def _simplex_to_u(p):  # length-K probability vector -> K-1 reals (softmax with the last logit anchored at 0)
    p = np.asarray(p, dtype=float)
    if (
        p.ndim != 1
        or p.size < 2
        or not np.all(np.isfinite(p))
        or np.any(p <= 0)
        or not np.isclose(np.sum(p), 1.0, rtol=1e-10, atol=1e-12)
    ):
        raise ValueError("simplex parameters must be finite, strictly positive, and sum to one at the Laplace mode")
    return list(np.log(p[:-1]) - np.log(p[-1]))


def _simplex_from_u(u, k):
    logits = np.concatenate([np.asarray(u[: k - 1], dtype=float), [0.0]])
    e = np.exp(logits - logits.max())
    return e / e.sum(), u[k - 1 :]


# --- per-leaf parameter round-trip (extend this to cover a new leaf family) -------------------------
def _leaf_flatteners():
    import mixle.stats as S

    return {
        "GaussianDistribution": (
            lambda d: _real_to_u(d.mu) + _pos_to_u(d.sigma2),
            lambda u: (lambda mu, u1: (lambda s2, u2: (S.GaussianDistribution(mu, s2), u2))(*_pos_from_u(u1)))(
                *_real_from_u(u)
            ),
        ),
        "PoissonDistribution": (
            lambda d: _pos_to_u(d.lam),
            lambda u: (lambda lam, r: (S.PoissonDistribution(lam), r))(*_pos_from_u(u)),
        ),
        "ExponentialDistribution": (
            lambda d: _pos_to_u(d.beta),
            lambda u: (lambda b, r: (S.ExponentialDistribution(b), r))(*_pos_from_u(u)),
        ),
        "GammaDistribution": (
            lambda d: _pos_to_u(d.k) + _pos_to_u(d.theta),
            lambda u: (lambda k, u1: (lambda th, u2: (S.GammaDistribution(k, th), u2))(*_pos_from_u(u1)))(
                *_pos_from_u(u)
            ),
        ),
        "BernoulliDistribution": (
            lambda d: _unit_to_u(d.p),
            lambda u: (lambda p, r: (S.BernoulliDistribution(p), r))(*_unit_from_u(u)),
        ),
    }


def _flatten(model) -> tuple[np.ndarray, Callable[[np.ndarray], Any]]:
    """Return (u0, rebuild): the model's parameters as one unconstrained vector, and a function that
    rebuilds the model from such a vector. Recurses through Composite and Mixture."""
    import mixle.stats as S

    leaves = _leaf_flatteners()
    name = type(model).__name__

    if name in leaves:
        to_u, from_u = leaves[name]
        return np.asarray(to_u(model), dtype=float), (lambda u, _f=from_u: _f(u))

    if isinstance(model, S.CategoricalDistribution):
        # the category-probability simplex over the (fixed) support -> K-1 softmax logits
        keys = sorted(model.pmap.keys(), key=repr)
        u0 = np.asarray(_simplex_to_u([model.pmap[k] for k in keys]), dtype=float)

        def rebuild(u, _keys=keys, _k=len(keys), _dv=model.default_value, _nm=model.name):
            p, rest = _simplex_from_u(u, _k)
            return S.CategoricalDistribution(dict(zip(_keys, p)), default_value=_dv, name=_nm), rest

        return u0, rebuild

    if isinstance(model, S.CompositeDistribution):
        parts = [_flatten(d) for d in model.dists]
        u0 = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0)

        def rebuild(u, _parts=parts):
            dists, rest = [], u
            for _, rb in _parts:
                d, rest = rb(rest)
                dists.append(d)
            return S.CompositeDistribution(tuple(dists)), rest

        return u0, rebuild

    if isinstance(model, S.MixtureDistribution):
        comp_parts = [_flatten(c) for c in model.components]
        w = np.asarray(model.w, dtype=float)
        u0 = np.concatenate([np.concatenate([p[0] for p in comp_parts]), np.asarray(_simplex_to_u(w))])
        kk = len(model.components)

        def rebuild(u, _parts=comp_parts, _k=kk):
            comps, rest = [], u
            for _, rb in _parts:
                c, rest = rb(rest)
                comps.append(c)
            weights, rest = _simplex_from_u(rest, _k)
            return S.MixtureDistribution(comps, list(weights)), rest

        return u0, rebuild

    from mixle.inference.bayesian_network import HeterogeneousBayesianNetwork

    if isinstance(model, HeterogeneousBayesianNetwork):
        parts = [_flatten_factor(f) for f in model.factors]
        u0 = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0)

        def rebuild(u, _parts=parts):
            facs, rest = [], u
            for _, rb in _parts:
                f, rest = rb(rest)
                facs.append(f)
            return HeterogeneousBayesianNetwork(facs), rest

        return u0, rebuild

    raise NotImplementedError(
        f"laplace_posterior cannot flatten a {name}; add it to _leaf_flatteners (the same per-family "
        "extend point as register_family), or use the model's bespoke inference."
    )


def _flatten_factor(f) -> tuple[np.ndarray, Callable[[np.ndarray], Any]]:
    """Flatten one Bayesian-network factor's numeric parameters to unconstrained coords (keeping its fixed
    structure -- child, parent set, discrete levels, GLM kind -- outside the vector) and return
    ``(u0, rebuild)`` where ``rebuild`` reconstructs the factor from such a vector plus the remaining tail."""
    from mixle.inference.bayesian_network import (
        _DiscreteConditionalFactor,
        _GLMFactor,
        _LinearGaussianFactor,
        _MarginalFactor,
    )

    if isinstance(f, _MarginalFactor):  # a root field: flatten its fitted marginal (categorical / Gaussian / count)
        u0, rb = _flatten(f.dist)
        return u0, (lambda u, _rb=rb, _c=f.child: (lambda d, r: (_MarginalFactor(_c, d), r))(*_rb(u)))

    if isinstance(f, _LinearGaussianFactor):  # CLG node: regression coefficients (real) + a log scale
        u0 = np.concatenate([np.asarray(f.coef, dtype=float), np.asarray(_pos_to_u(f.sigma), dtype=float)])
        nc = int(np.asarray(f.coef).shape[0])

        def rb(u, _c=f.child, _p=f.parents, _d=f.discrete, _nc=nc):
            coef = np.asarray(u[:_nc], dtype=float)
            sigma, rest = _pos_from_u(u[_nc:])
            return _LinearGaussianFactor(_c, _p, _d, coef, sigma), rest

        return u0, rb

    if isinstance(f, _GLMFactor):  # GLM node: the logistic / Poisson / softmax weights are already unconstrained
        w = np.asarray(f.weights, dtype=float)
        u0 = w.ravel()

        def rb(u, _c=f.child, _p=f.parents, _d=f.discrete, _k=f.kind, _lv=f.levels, _sh=w.shape):
            n = int(np.prod(_sh)) if _sh else 0
            weights = np.asarray(u[:n], dtype=float).reshape(_sh)
            return _GLMFactor(_c, _p, _d, _k, _lv, weights), u[n:]

        return u0, rb

    if isinstance(f, _DiscreteConditionalFactor):  # per-config CPTs: flatten the backoff + each config's child dist
        cfgs = sorted(f.table.keys(), key=repr)
        subs = [_flatten(f.backoff)] + [_flatten(f.table[c]) for c in cfgs]
        u0 = np.concatenate([s[0] for s in subs]) if subs else np.zeros(0)

        def rb(u, _c=f.child, _p=f.parents, _cfgs=cfgs, _subs=subs):
            dists, rest = [], u
            for _, rbf in _subs:
                d, rest = rbf(rest)
                dists.append(d)
            table = {cfg: dists[i + 1] for i, cfg in enumerate(_cfgs)}
            return _DiscreteConditionalFactor(_c, _p, table, dists[0]), rest

        return u0, rb

    raise NotImplementedError(
        f"laplace_posterior cannot flatten a {type(f).__name__} Bayesian-network factor; add it to _flatten_factor."
    )


class LaplacePosterior:
    """Validated Gaussian Laplace approximation in unconstrained parameter space.

    The compatibility name is retained, but :attr:`is_posterior` is true only
    when the target included an explicit prior.
    """

    def __init__(self, mode_model, u_mode, cov, rebuild, metadata):
        self.mean_model = mode_model
        self.u_mode = np.asarray(u_mode, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        if (
            self.u_mode.ndim != 1
            or self.u_mode.size < 1
            or not np.all(np.isfinite(self.u_mode))
            or self.cov.shape != (self.u_mode.size, self.u_mode.size)
            or not np.all(np.isfinite(self.cov))
            or not np.allclose(self.cov, self.cov.T, rtol=1e-10, atol=1e-12)
        ):
            raise ValueError("Laplace mode and covariance must be finite, aligned, and symmetric")
        self.cov = 0.5 * (self.cov + self.cov.T)
        try:
            self._chol = np.linalg.cholesky(self.cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Laplace covariance must be positive definite") from exc
        self._rebuild = rebuild
        self.metadata = dict(metadata)
        self.is_posterior = bool(self.metadata["prior_included"])
        self.acceptance_rate = None

    def sample(self, n: int = 1, rng=None):
        """Draw model samples from the Laplace Gaussian approximation. ``rng`` may be ``None``
        (unseeded), an int seed, or an existing ``RandomState``."""
        # `rng or np.random.RandomState()` silently discarded seed 0 (falsy) and crashed on any
        # other int (no .standard_normal method) -- 0 is a perfectly valid seed.
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be a positive integer")
        rng = rng if isinstance(rng, np.random.RandomState) else np.random.RandomState(rng)
        zs = rng.standard_normal((int(n), len(self.u_mode)))
        draws = self.u_mode[None, :] + zs @ self._chol.T
        models = []
        for u in draws:
            model, rest = self._rebuild(u.copy())
            if np.asarray(rest).size:
                raise RuntimeError("Laplace parameter rebuild left unconsumed coordinates")
            models.append(model)
        return models[0] if n == 1 else models

    def summary(self) -> dict:
        """Return Laplace approximation metadata."""
        return {
            "n_params": int(len(self.u_mode)),
            "approximation": "laplace (unconstrained Gaussian)",
            "target": self.metadata["target"],
            "is_posterior": self.is_posterior,
            **self.metadata,
        }


def laplace_posterior(
    model,
    data,
    *,
    log_prior: Callable[[np.ndarray], float] | None = None,
    eps: float = 1e-4,
    ridge: float = 1e-6,
    mode_tol: float = 1e-3,
) -> LaplacePosterior:
    """Build a validated Laplace approximation around ``model``'s parameters.

    ``log_prior`` is evaluated in the flattened unconstrained coordinates. If
    supplied, the target is the log posterior; otherwise the result is
    explicitly a likelihood-curvature approximation, despite the compatibility
    name of this function and result class.
    """
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and > 0")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and >= 0")
    if not np.isfinite(mode_tol) or mode_tol <= 0:
        raise ValueError("mode_tol must be finite and > 0")
    u0, rebuild = _flatten(model)
    d = len(u0)
    if d == 0:
        raise ValueError("model has no flattenable parameters.")
    if not np.all(np.isfinite(u0)):
        raise ValueError("model parameters are non-finite at the proposed Laplace mode")
    try:
        observations = tuple(data)
    except TypeError as exc:
        raise ValueError("data must be a reusable iterable of observations") from exc
    if not observations:
        raise ValueError("data must contain at least one observation")

    def log_target(u):
        coordinates = np.asarray(u, dtype=float)
        m, rest = rebuild(coordinates)
        if np.asarray(rest).size:
            raise RuntimeError("parameter rebuild left unconsumed coordinates")
        enc = m.dist_to_encoder().seq_encode(observations)
        likelihood = float(np.sum(np.asarray(m.seq_log_density(enc), dtype=float)))
        if not np.isfinite(likelihood):
            raise ValueError("model log likelihood is non-finite near the proposed Laplace mode")
        prior_value = 0.0 if log_prior is None else float(log_prior(coordinates.copy()))
        if not np.isfinite(prior_value):
            raise ValueError("log_prior must return a finite scalar near the proposed Laplace mode")
        return likelihood + prior_value

    # Finite-difference gradient and Hessian of the target at the proposed mode.
    h = eps * (1.0 + np.abs(u0))
    f0 = log_target(u0)
    gradient = np.zeros(d)
    hess = np.zeros((d, d))
    for i in range(d):
        ui = u0.copy()
        ui[i] += h[i]
        fi = log_target(ui)
        ui[i] -= 2 * h[i]
        fmi = log_target(ui)
        gradient[i] = (fi - fmi) / (2.0 * h[i])
        hess[i, i] = (fi - 2 * f0 + fmi) / (h[i] ** 2)
        for j in range(i + 1, d):
            upp = u0.copy()
            upp[i] += h[i]
            upp[j] += h[j]
            umm = u0.copy()
            umm[i] -= h[i]
            umm[j] -= h[j]
            upm = u0.copy()
            upm[i] += h[i]
            upm[j] -= h[j]
            ump = u0.copy()
            ump[i] -= h[i]
            ump[j] += h[j]
            hess[i, j] = hess[j, i] = (
                log_target(upp) + log_target(umm) - log_target(upm) - log_target(ump)
            ) / (4 * h[i] * h[j])

    gradient_norm = float(np.linalg.norm(gradient, ord=np.inf))
    gradient_limit = float(mode_tol * max(1.0, abs(f0)))
    if gradient_norm > gradient_limit:
        raise ValueError(
            f"proposed Laplace mode is not stationary: gradient norm {gradient_norm:.6g} "
            f"exceeds {gradient_limit:.6g}"
        )
    raw_precision = -0.5 * (hess + hess.T)
    eigenvalues = np.linalg.eigvalsh(raw_precision)
    curvature_tolerance = 1e-8 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -curvature_tolerance:
        raise ValueError("proposed Laplace mode has saddle/positive log-target curvature")
    rank = int(np.count_nonzero(eigenvalues > curvature_tolerance))
    if rank < d and ridge == 0:
        raise ValueError("Laplace curvature is rank deficient; explicit positive ridge regularization is required")
    precision = raw_precision + ridge * np.eye(d)
    regularized_eigenvalues = np.linalg.eigvalsh(precision)
    if float(np.min(regularized_eigenvalues)) <= 0:
        raise ValueError("regularized Laplace precision is not positive definite")
    cov = np.linalg.inv(precision)
    cov = 0.5 * (cov + cov.T)
    metadata = {
        "target": "posterior" if log_prior is not None else "likelihood",
        "prior_included": log_prior is not None,
        "regularization": float(ridge),
        "regularized": bool(ridge > 0),
        "curvature_rank": rank,
        "dimension": d,
        "raw_min_precision_eigenvalue": float(np.min(eigenvalues)),
        "mode_gradient_norm": gradient_norm,
        "mode_gradient_limit": gradient_limit,
        "approximation_status": (
            "regularized_rank_deficient_local_mode" if rank < d else "validated_local_mode"
        ),
    }
    return LaplacePosterior(model, u0, cov, rebuild, metadata)


__all__ = ["LaplacePosterior", "laplace_posterior"]
