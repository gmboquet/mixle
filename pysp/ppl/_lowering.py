"""Family/composite registration for the pysp.ppl dialect — the lowering registry.

Maps each user-facing distribution name to its underlying pysp ``*Distribution`` / ``*Estimator`` and
the arg / seed / read quartet that lowers a :class:`~pysp.ppl.core.RandomVariable` into a fittable pysp
model. This module is imported for its **side effects**: the ``register_family`` / ``register_composite``
calls run at import, so ``pysp.ppl`` imports it before any dialect constructor (``pysp.ppl.distributions``)
is used. Previously inlined in ``pysp/ppl/__init__.py``; extracted so the package init is a thin surface.
"""

from __future__ import annotations

import numpy as np

from pysp.ppl.core import register_composite, register_family
from pysp.stats.bayes.dirichlet import DirichletDistribution, DirichletEstimator
from pysp.stats.combinator.sequence import SequenceDistribution, SequenceEstimator
from pysp.stats.latent.hidden_markov import HiddenMarkovEstimator, HiddenMarkovModelDistribution
from pysp.stats.latent.lda import LDADistribution, LDAEstimator
from pysp.stats.latent.mixture import MixtureDistribution, MixtureEstimator
from pysp.stats.latent.semi_supervised_mixture import SemiSupervisedMixtureDistribution, SemiSupervisedMixtureEstimator
from pysp.stats.multivariate.diagonal_gaussian import DiagonalGaussianDistribution, DiagonalGaussianEstimator
from pysp.stats.multivariate.multivariate_gaussian import (
    MultivariateGaussianDistribution,
    MultivariateGaussianEstimator,
)
from pysp.stats.univariate.continuous.beta import BetaDistribution, BetaEstimator
from pysp.stats.univariate.continuous.exgaussian import (
    ExponentiallyModifiedGaussianDistribution,
    ExponentiallyModifiedGaussianEstimator,
)
from pysp.stats.univariate.continuous.exponential import ExponentialDistribution, ExponentialEstimator
from pysp.stats.univariate.continuous.gamma import GammaDistribution, GammaEstimator
from pysp.stats.univariate.continuous.gaussian import GaussianDistribution, GaussianEstimator
from pysp.stats.univariate.continuous.laplace import LaplaceDistribution, LaplaceEstimator
from pysp.stats.univariate.continuous.log_gaussian import LogGaussianDistribution, LogGaussianEstimator
from pysp.stats.univariate.continuous.logistic import LogisticDistribution, LogisticEstimator
from pysp.stats.univariate.continuous.pareto import ParetoDistribution, ParetoEstimator
from pysp.stats.univariate.continuous.rayleigh import RayleighDistribution, RayleighEstimator
from pysp.stats.univariate.continuous.student_t import StudentTDistribution, StudentTEstimator
from pysp.stats.univariate.continuous.uniform import UniformDistribution, UniformEstimator
from pysp.stats.univariate.continuous.weibull import WeibullDistribution, WeibullEstimator
from pysp.stats.univariate.discrete.bernoulli import BernoulliDistribution, BernoulliEstimator
from pysp.stats.univariate.discrete.binomial import BinomialDistribution, BinomialEstimator
from pysp.stats.univariate.discrete.categorical import CategoricalDistribution, CategoricalEstimator
from pysp.stats.univariate.discrete.geometric import GeometricDistribution, GeometricEstimator
from pysp.stats.univariate.discrete.integer_categorical import (
    IntegerCategoricalDistribution,
    IntegerCategoricalEstimator,
)
from pysp.stats.univariate.discrete.negative_binomial import NegativeBinomialDistribution, NegativeBinomialEstimator
from pysp.stats.univariate.discrete.poisson import PoissonDistribution, PoissonEstimator

# --- family registration: (user-facing args) -> (underlying *Distribution kwargs) ---
register_family(
    "Normal",
    GaussianDistribution,
    GaussianEstimator,
    lambda mean, sd: {"mu": float(mean), "sigma2": float(sd) ** 2},
    arity=2,
    seed_at=lambda v, s: {"mu": float(v), "sigma2": (float(s) ** 2) or 1.0},
    positive=(False, True),
    read=lambda d: {"mean": d.mu, "sd": float(np.sqrt(d.sigma2))},
)
register_family(
    "Poisson",
    PoissonDistribution,
    PoissonEstimator,
    lambda rate: {"lam": float(rate)},
    arity=1,
    seed_at=lambda v, s: {"lam": max(float(v), 1e-2)},
    positive=(True,),
    read=lambda d: {"rate": d.lam},
)
register_family(
    "Gamma",
    GammaDistribution,
    GammaEstimator,
    lambda shape, rate: {"k": float(shape), "theta": 1.0 / float(rate)},
    arity=2,
    seed_at=lambda v, s: {"k": 1.0, "theta": max(float(v), 1e-2)},
    positive=(True, True),
    read=lambda d: {"shape": d.k, "rate": 1.0 / d.theta},
)
register_family(
    "Exponential",
    ExponentialDistribution,
    ExponentialEstimator,
    lambda rate: {"beta": 1.0 / float(rate)},
    arity=1,
    seed_at=lambda v, s: {"beta": max(float(v), 1e-2)},
    positive=(True,),
    read=lambda d: {"rate": 1.0 / d.beta},
)
register_family(
    "Bernoulli",
    BernoulliDistribution,
    BernoulliEstimator,
    lambda p: {"p": float(p)},
    arity=1,
    support=("unit",),
    read=lambda d: {"p": d.p},
)
register_family(
    "Geometric",
    GeometricDistribution,
    GeometricEstimator,
    lambda p: {"p": float(p)},
    arity=1,
    support=("unit",),
    read=lambda d: {"p": d.p},
)
register_family(
    "Beta",
    BetaDistribution,
    BetaEstimator,
    lambda a, b: {"a": float(a), "b": float(b)},
    arity=2,
    positive=(True, True),
    read=lambda d: {"a": d.a, "b": d.b},
)
register_family(
    "Dirichlet",
    DirichletDistribution,
    DirichletEstimator,
    lambda alpha: {"alpha": np.asarray(alpha, dtype=float)},
    arity=1,
    read=lambda d: {"alpha": np.asarray(d.alpha), "mean": np.asarray(d.alpha) / float(np.sum(d.alpha))},
)
register_family(
    "StudentT",
    StudentTDistribution,
    StudentTEstimator,
    lambda df, loc, scale: {"df": float(df), "loc": float(loc), "scale": float(scale)},
    arity=3,
    seed_at=lambda v, s: {"df": 5.0, "loc": float(v), "scale": (float(s) or 1.0)},
    positive=(True, False, True),
    read=lambda d: {"df": d.df, "loc": d.loc, "scale": d.scale},
)
register_family(
    "LogNormal",
    LogGaussianDistribution,
    LogGaussianEstimator,
    lambda mu, sigma: {"mu": float(mu), "sigma2": float(sigma) ** 2},
    arity=2,
    seed_at=lambda v, s: {"mu": float(np.log(max(v, 1e-3))), "sigma2": 1.0},
    positive=(False, True),
    read=lambda d: {"mu": d.mu, "sigma": float(np.sqrt(d.sigma2))},
)


register_family(
    "EMG",
    ExponentiallyModifiedGaussianDistribution,
    ExponentiallyModifiedGaussianEstimator,
    lambda mu, sigma, rate: {"mu": float(mu), "sigma2": float(sigma) ** 2, "lam": float(rate)},
    arity=3,
    seed_at=lambda v, s: {"mu": float(v), "sigma2": (float(s) ** 2) or 1.0, "lam": 1.0},
    positive=(False, True, True),
    read=lambda d: {"mu": d.mu, "sigma": float(np.sqrt(d.sigma2)), "rate": d.lam},
)


def _nb_init(data):
    a = np.asarray(data, dtype=float)
    mu, var = float(a.mean()), float(a.var())
    r0 = mu * mu / max(var - mu, 1e-3)  # moment match: var = mu + mu^2/r
    p0 = r0 / (r0 + mu)
    return NegativeBinomialDistribution(max(r0, 1e-2), min(max(p0, 1e-3), 1 - 1e-3))


register_family(
    "NegativeBinomial",
    NegativeBinomialDistribution,
    NegativeBinomialEstimator,
    lambda r, p: {"r": float(r), "p": float(p)},
    arity=2,
    positive=(True, False),
    support=("positive", "unit"),
    init_fit=_nb_init,
    read=lambda d: {"r": d.r, "p": d.p},
)


def _cat_args(probs):
    if isinstance(probs, dict):
        return {"pmap": dict(probs)}
    return {"pmap": {i: float(v) for i, v in enumerate(probs)}}


register_family("Categorical", CategoricalDistribution, CategoricalEstimator, _cat_args, arity=1)

register_family(
    "Weibull",
    WeibullDistribution,
    WeibullEstimator,
    lambda shape, scale: {"shape": float(shape), "scale": float(scale)},
    arity=2,
    positive=(True, True),
    seed_at=lambda v, s: {"shape": 1.5, "scale": max(float(v), 1e-2)},
    read=lambda d: {"shape": d.shape, "scale": d.scale},
)
register_family(
    "Laplace",
    LaplaceDistribution,
    LaplaceEstimator,
    lambda loc, scale: {"mu": float(loc), "b": float(scale)},
    arity=2,
    positive=(False, True),
    seed_at=lambda v, s: {"mu": float(v), "b": max(float(s), 1e-2)},
    read=lambda d: {"loc": d.mu, "scale": d.b},
)
register_family(
    "Logistic",
    LogisticDistribution,
    LogisticEstimator,
    lambda loc, scale: {"loc": float(loc), "scale": float(scale)},
    arity=2,
    positive=(False, True),
    seed_at=lambda v, s: {"loc": float(v), "scale": max(float(s), 1e-2)},
    read=lambda d: {"loc": d.loc, "scale": d.scale},
)
register_family(
    "Uniform",
    UniformDistribution,
    UniformEstimator,
    lambda low, high: {"low": float(low), "high": float(high)},
    arity=2,
    read=lambda d: {"low": d.low, "high": d.high},
)
register_family(
    "Rayleigh",
    RayleighDistribution,
    RayleighEstimator,
    lambda sigma: {"sigma": float(sigma)},
    arity=1,
    positive=(True,),
    seed_at=lambda v, s: {"sigma": max(float(v), 1e-2)},
    read=lambda d: {"sigma": d.sigma},
)
register_family(
    "Pareto",
    ParetoDistribution,
    ParetoEstimator,
    lambda xm, alpha: {"xm": float(xm), "alpha": float(alpha)},
    arity=2,
    positive=(True, True),
    read=lambda d: {"xm": d.xm, "alpha": d.alpha},
)
register_family(
    "Binomial",
    BinomialDistribution,
    BinomialEstimator,
    lambda n, p: {"p": float(p), "n": int(n)},
    arity=2,
    support=("real", "unit"),
    read=lambda d: {"n": d.n, "p": d.p},
)


def _mix_dist(args, lower_child):
    comps, weights = args
    children = [lower_child(c) for c in comps]
    w = np.ones(len(children)) / len(children) if weights is None else np.asarray(weights, float)
    return MixtureDistribution(children, w=w)


def _mix_est(args, lower_child_est, name, keys):
    comps, weights = args
    estimators = [lower_child_est(c) for c in comps]
    fixed = None if weights is None else np.asarray(weights, float)
    return MixtureEstimator(estimators, fixed_weights=fixed, name=name)


def _kmeanspp_idx(arr, k, rng):
    """k-means++ seed indices: spread seeds across clusters (distance^2 weighted)."""
    n = len(arr)
    chosen = [int(rng.randint(n))]
    d2 = np.sum((arr - arr[chosen[0]]) ** 2, axis=-1) if arr.ndim > 1 else (arr - arr[chosen[0]]) ** 2
    for _ in range(1, k):
        total = d2.sum()
        probs = d2 / total if total > 0 else np.ones(n) / n
        j = int(rng.choice(n, p=probs))
        chosen.append(j)
        dj = np.sum((arr - arr[j]) ** 2, axis=-1) if arr.ndim > 1 else (arr - arr[j]) ** 2
        d2 = np.minimum(d2, dj)
    return chosen


def _mix_seed(args, data, rng, seed_child):
    comps, weights = args
    arr = np.asarray(data, dtype=float)
    scale = float(np.std(arr)) if arr.ndim == 1 and arr.size > 1 else 1.0
    idx = _kmeanspp_idx(arr, len(comps), rng)
    children = [seed_child(c, data[i], scale, rng) for c, i in zip(comps, idx)]
    if any(ch is None for ch in children):
        return None  # a component can't be seeded -> fall back to pysp default init
    w = np.ones(len(comps)) / len(comps) if weights is None else np.asarray(weights, float)
    return MixtureDistribution(children, w=w)


def _mix_read(d, read_params):
    return {"components": [read_params(c) for c in d.components], "weights": np.asarray(d.w)}


register_composite("Mixture", _mix_dist, _mix_est, seed_fn=_mix_seed, dist_cls=MixtureDistribution, read=_mix_read)


# --- SemiMix: semi-supervised mixture (observations are (value, prior) pairs) ------
def _semimix_dist(args, lower_child):
    comps, weights = args
    children = [lower_child(c) for c in comps]
    k = len(children)
    w = np.ones(k) / k if weights is None else np.asarray(weights, float)
    return SemiSupervisedMixtureDistribution(children, w=w)


def _semimix_est(args, lower_child_est, name, keys):
    comps, _weights = args
    estimators = [lower_child_est(c) for c in comps]
    return SemiSupervisedMixtureEstimator(estimators, name=name)


def _semimix_read(d, read_params):
    return {"components": [read_params(c) for c in d.components], "weights": np.asarray(d.w)}


register_composite(
    "SemiMix", _semimix_dist, _semimix_est, dist_cls=SemiSupervisedMixtureDistribution, read=_semimix_read
)


# --- Sequence: iid elements (+ optional length model) -----------------------------
def _seq_dist(args, lower_child):
    (elem,) = args
    return SequenceDistribution(lower_child(elem))


def _seq_est(args, lower_child_est, name, keys):
    (elem,) = args
    return SequenceEstimator(lower_child_est(elem), name=name)


def _seq_read(d, read_params):
    return {"element": read_params(d.dist)}


register_composite("Sequence", _seq_dist, _seq_est, dist_cls=SequenceDistribution, read=_seq_read)


# --- Markov / HMM: latent-state sequence model ------------------------------------
def _hmm_dist(args, lower_child):
    comps = args[0]
    topics = [lower_child(c) for c in comps]
    k = len(topics)
    trans = args[1] if len(args) > 1 else None
    init = args[2] if len(args) > 2 else None
    T = np.asarray(trans, dtype=float) if isinstance(trans, np.ndarray) else np.ones((k, k)) / k
    w = np.asarray(init, dtype=float) if isinstance(init, np.ndarray) else np.ones(k) / k
    return HiddenMarkovModelDistribution(topics, w=w, transitions=T)


def _hmm_est(args, lower_child_est, name, keys):
    comps = args[0]
    return HiddenMarkovEstimator([lower_child_est(c) for c in comps], name=name)


def _flatten_obs(data):
    parts = [np.asarray(s, dtype=float).reshape(-1) for s in data if len(s) > 0]
    return np.concatenate(parts) if parts else np.asarray([], dtype=float)


def _hmm_seed(args, data, rng, seed_child):
    comps = args[0]
    k = len(comps)
    arr = _flatten_obs(data)
    if arr.size < k:
        return None
    scale = float(np.std(arr)) or 1.0
    idx = _kmeanspp_idx(arr, k, rng)
    topics = [seed_child(c, arr[i], scale, rng) for c, i in zip(comps, idx)]
    if any(t is None for t in topics):
        return None
    return HiddenMarkovModelDistribution(topics, w=np.ones(k) / k, transitions=np.ones((k, k)) / k)


def _hmm_read(d, read_params):
    return {
        "states": [read_params(t) for t in d.topics],
        "transitions": np.asarray(d.transitions),
        "initial": np.asarray(d.w),
    }


register_composite(
    "Markov", _hmm_dist, _hmm_est, seed_fn=_hmm_seed, dist_cls=HiddenMarkovModelDistribution, read=_hmm_read
)


# --- LDA / topic model: documents are bags of (word_id, count) --------------------
def _lda_dist(args, lower_child):
    k, V, alpha = args
    topics = [IntegerCategoricalDistribution(0, np.ones(V) / V) for _ in range(k)]
    return LDADistribution(topics, np.full(k, float(alpha)))


def _lda_est(args, lower_child_est, name, keys):
    k, V, alpha = args
    return LDAEstimator(
        [IntegerCategoricalEstimator(min_val=0, max_val=V - 1) for _ in range(k)], fixed_alpha=np.full(k, float(alpha))
    )


def _lda_seed(args, data, rng, seed_child):
    k, V, alpha = args
    topics = [IntegerCategoricalDistribution(0, rng.dirichlet(0.5 * np.ones(V))) for _ in range(k)]
    return LDADistribution(topics, np.full(k, float(alpha)))


def _lda_read(d, read_params):
    return {"topics": [np.asarray(t.p_vec) for t in d.topics], "alpha": np.asarray(d.alpha)}


register_composite("LDA", _lda_dist, _lda_est, seed_fn=_lda_seed, dist_cls=LDADistribution, read=_lda_read)


# --- multivariate Gaussian (data are vectors) -------------------------------------
def _mvn_dist(args, lower_child):
    dim = args[0]
    mean = args[1] if len(args) > 1 else None
    cov = args[2] if len(args) > 2 else None
    mu = np.asarray(mean, dtype=float) if isinstance(mean, np.ndarray) else np.zeros(dim)
    covar = np.asarray(cov, dtype=float) if isinstance(cov, np.ndarray) else np.eye(dim)
    return MultivariateGaussianDistribution(mu, covar)


def _mvn_est(args, lower_child_est, name, keys):
    return MultivariateGaussianEstimator(args[0])


def _mvn_read(d, read_params):
    return {"mean": np.asarray(d.mu), "cov": np.asarray(d.covar)}


register_composite("MVN", _mvn_dist, _mvn_est, dist_cls=MultivariateGaussianDistribution, read=_mvn_read)


def _diag_dist(args, lower_child):
    dim = args[0]
    mean = args[1] if len(args) > 1 else None
    var = args[2] if len(args) > 2 else None
    mu = np.asarray(mean, dtype=float) if isinstance(mean, np.ndarray) else np.zeros(dim)
    covar = np.asarray(var, dtype=float) if isinstance(var, np.ndarray) else np.ones(dim)
    return DiagonalGaussianDistribution(mu, covar)


def _diag_est(args, lower_child_est, name, keys):
    return DiagonalGaussianEstimator(dim=args[0])


def _diag_read(d, read_params):
    return {"mean": np.asarray(d.mu), "var": np.asarray(d.covar)}


register_composite("DiagGaussian", _diag_dist, _diag_est, dist_cls=DiagonalGaussianDistribution, read=_diag_read)


# Linear-Gaussian state space (time series): the StateSpace composite self-registers (with its
# Kalman/RTS+EM fit_fn) when statespace.py is imported. The PDE-constrained PDEStateSpace composite
# self-registers from the pysparkplug-pde plugin's modules when that package is imported -- so this
# lowering hub (and pysp generally) no longer references the PDE stack.
from pysp.ppl import statespace  # noqa: F401, E402  (import-time StateSpace registration)
