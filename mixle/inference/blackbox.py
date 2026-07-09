"""Black-box Bayesian inference for *any* model: a Laplace posterior over the parameters.

The conjugate path needs a registered closed form; the autograd VI/NUTS path needs a per-family
from-parameters scorer (so it covers flat models + mixtures of leaves). Neither works for an arbitrary
composable model. This module does: it treats a *fitted* model's parameters as the latent, flattens them
to an UNCONSTRAINED vector (positive params via log, unit via logit, simplex via softmax), and fits a
Gaussian posterior in that space from a finite-difference Hessian of the model's own
``seq_log_density`` -- which every model has. No conjugacy, no autograd, no per-model inference code.

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
_EPS = 1e-12


def _pos_to_u(x):
    return [float(np.log(max(float(x), _EPS)))]


def _pos_from_u(u):
    return float(np.exp(u[0])), u[1:]


def _real_to_u(x):
    return [float(x)]


def _real_from_u(u):
    return float(u[0]), u[1:]


def _unit_to_u(p):
    p = min(max(float(p), _EPS), 1 - _EPS)
    return [float(np.log(p / (1 - p)))]


def _unit_from_u(u):
    return float(1.0 / (1.0 + np.exp(-u[0]))), u[1:]


def _simplex_to_u(p):  # length-K probability vector -> K-1 reals (softmax with the last logit anchored at 0)
    p = np.asarray(p, dtype=float)
    return list(np.log(np.maximum(p[:-1], _EPS)) - np.log(max(p[-1], _EPS)))


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
    """Gaussian Laplace posterior over a model's parameters (in the unconstrained space), with draws
    rebuilt back into fitted models. ``mean_model`` is the mode; ``cov`` the unconstrained covariance."""

    def __init__(self, mode_model, u_mode, cov, rebuild):
        self.mean_model = mode_model
        self.u_mode = np.asarray(u_mode, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        self._rebuild = rebuild
        self._chol = np.linalg.cholesky(cov + 1e-9 * np.eye(len(u_mode)))
        self.acceptance_rate = None

    def sample(self, n: int = 1, rng=None):
        """Draw model samples from the Laplace Gaussian approximation."""
        rng = rng or np.random.RandomState()
        zs = rng.standard_normal((int(n), len(self.u_mode)))
        draws = self.u_mode[None, :] + zs @ self._chol.T
        models = [self._rebuild(u.copy())[0] for u in draws]
        return models[0] if n == 1 else models

    def summary(self) -> dict:
        """Return Laplace approximation metadata."""
        return {"n_params": int(len(self.u_mode)), "posterior": "laplace (unconstrained Gaussian)"}


def laplace_posterior(model, data, *, eps: float = 1e-4, ridge: float = 1e-6) -> LaplacePosterior:
    """Laplace posterior over ``model``'s parameters from a finite-difference Hessian of its own
    ``seq_log_density`` -- works for ANY model whose parameters :func:`_flatten` covers (the scalar
    exponential-family leaves, ``Composite`` and ``Mixture``, recursively), conjugate or not, with no
    per-model inference code. ``model`` should be the fitted (MLE/MAP) model -- its parameters are the
    Laplace mode. Returns a :class:`LaplacePosterior` you can ``.sample()`` (a fitted model per draw)."""
    u0, rebuild = _flatten(model)
    d = len(u0)
    if d == 0:
        raise ValueError("model has no flattenable parameters.")

    def loglik(u):
        m, _ = rebuild(np.asarray(u, dtype=float))
        enc = m.dist_to_encoder().seq_encode(data)
        return float(np.sum(np.asarray(m.seq_log_density(enc))))

    # finite-difference Hessian of the log-likelihood at the mode (unconstrained coords)
    h = eps * (1.0 + np.abs(u0))
    f0 = loglik(u0)
    hess = np.zeros((d, d))
    for i in range(d):
        ui = u0.copy()
        ui[i] += h[i]
        fi = loglik(ui)
        ui[i] -= 2 * h[i]
        fmi = loglik(ui)
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
            hess[i, j] = hess[j, i] = (loglik(upp) + loglik(umm) - loglik(upm) - loglik(ump)) / (4 * h[i] * h[j])

    precision = -hess + ridge * np.eye(d)  # posterior precision ~ negative Hessian of the log-density
    # project to PSD (guard against a non-concave finite-difference Hessian)
    w, v = np.linalg.eigh(precision)
    w = np.maximum(w, 1e-8)
    cov = (v * (1.0 / w)) @ v.T
    return LaplacePosterior(model, u0, cov, rebuild)


__all__ = ["LaplacePosterior", "laplace_posterior"]
