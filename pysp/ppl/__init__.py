"""pysp.ppl — an elegant probabilistic-programming surface over pysparkplug.

A model is plain pysparkplug construction with two new things allowed in a parameter
slot: the token ``free`` (estimate it) or another distribution (make it random). Fit
with ``.fit(data)``; query with ``.sample`` / ``.log_prob`` / ``.posterior``.

    from pysp.ppl import Normal, free
    m = Normal(free, free).fit(data)
    m.sample(100)

The 86 ``pysp.stats`` distribution classes are untouched; this is a thin, optional
dialect. See notes/ppl-syntax-spec.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pysp.ppl.conformal import (
    ConformalClassifier,
    ConformalKnowledgeGraph,
    ConformalLinkPredictor,
    ConformalQuantileRegressor,
    ConformalRegressor,
    ConformalStructure,
    conformal,
)
from pysp.ppl.core import (
    Constraint,
    Event,
    Field,
    Group,
    RandomVariable,
    _CholeskySpec,
    _OrderedSpec,
    _SimplexSpec,
    _VectorSpec,
    compare,
    concave,
    constrain,
    convex,
    decreasing,
    eq,
    equal,
    free,
    increasing,
    lipschitz,
    lower,
    monotone,
    ne,
    ode_residual,
    ordered,
    register_composite,
    register_family,
)
from pysp.ppl.field import (
    GP,
    RBF,
    Cox,
    CustomProxy,
    FieldKernel,
    FieldModel,
    FieldPosterior,
    Gaussian,
    GaussianField,
    GaussianProxy,
    LogisticNicheProxy,
    Niche,
    PoissonProxy,
    Proxy,
    RandomWalk,
    fit_field,
    joint,
    multistart,
)
from pysp.ppl.flow import NavierStokes2D
from pysp.ppl.inverse import Differential
from pysp.ppl.priors import Potts, TotalVariation
from pysp.ppl.shape import level_set_material, shape_optimize

# Low-level numeric primitives (sparse_solve, grid assembly, the ODE integrator) live in
# pysp.ppl.pde_solve and are reached through the `ops` namespace handed to forward callbacks; they are
# deliberately not exported at the top level, which is reserved for the modeling surface.
from pysp.stats.bayes.dirichlet import DirichletDistribution, DirichletEstimator
from pysp.stats.combinator.sequence import SequenceDistribution, SequenceEstimator
from pysp.stats.latent.hidden_markov import HiddenMarkovEstimator, HiddenMarkovModelDistribution
from pysp.stats.latent.lda import LDADistribution, LDAEstimator
from pysp.stats.latent.mixture import MixtureDistribution, MixtureEstimator
from pysp.stats.latent.ss_mixture import SemiSupervisedMixtureDistribution, SemiSupervisedMixtureEstimator
from pysp.stats.leaf.bernoulli import BernoulliDistribution, BernoulliEstimator
from pysp.stats.leaf.beta import BetaDistribution, BetaEstimator
from pysp.stats.leaf.binomial import BinomialDistribution, BinomialEstimator
from pysp.stats.leaf.categorical import CategoricalDistribution, CategoricalEstimator
from pysp.stats.leaf.exgaussian import (
    ExponentiallyModifiedGaussianDistribution,
    ExponentiallyModifiedGaussianEstimator,
)
from pysp.stats.leaf.exponential import ExponentialDistribution, ExponentialEstimator
from pysp.stats.leaf.gamma import GammaDistribution, GammaEstimator
from pysp.stats.leaf.gaussian import GaussianDistribution, GaussianEstimator
from pysp.stats.leaf.geometric import GeometricDistribution, GeometricEstimator
from pysp.stats.leaf.int_range import IntegerCategoricalDistribution, IntegerCategoricalEstimator
from pysp.stats.leaf.laplace import LaplaceDistribution, LaplaceEstimator
from pysp.stats.leaf.log_gaussian import LogGaussianDistribution, LogGaussianEstimator
from pysp.stats.leaf.logistic import LogisticDistribution, LogisticEstimator
from pysp.stats.leaf.negative_binomial import NegativeBinomialDistribution, NegativeBinomialEstimator
from pysp.stats.leaf.pareto import ParetoDistribution, ParetoEstimator
from pysp.stats.leaf.poisson import PoissonDistribution, PoissonEstimator
from pysp.stats.leaf.rayleigh import RayleighDistribution, RayleighEstimator
from pysp.stats.leaf.student_t import StudentTDistribution, StudentTEstimator
from pysp.stats.leaf.uniform import UniformDistribution, UniformEstimator
from pysp.stats.leaf.weibull import WeibullDistribution, WeibullEstimator
from pysp.stats.multivariate.dmvn import DiagonalGaussianDistribution, DiagonalGaussianEstimator
from pysp.stats.multivariate.mvn import MultivariateGaussianDistribution, MultivariateGaussianEstimator

__all__ = [
    "RandomVariable",
    "free",
    "ordered",
    "lower",
    "Normal",
    "Poisson",
    "Gamma",
    "Exponential",
    "Categorical",
    "Bernoulli",
    "Geometric",
    "Binomial",
    "Weibull",
    "Laplace",
    "Logistic",
    "Uniform",
    "Rayleigh",
    "Pareto",
    "Beta",
    "StudentT",
    "LogNormal",
    "EMG",
    "NegativeBinomial",
    "Dirichlet",
    "Mix",
    "SemiMix",
    "Seq",
    "Markov",
    "LDA",
    "MVN",
    "DiagGaussian",
    "LocalLevel",
    "AR1",
    "PDE",
    "DiffusionOperator",
    "AdvectionOperator",
    "AdvectionDiffusionOperator",
    "make_operator",
    "register_dynamics_operator",
    "available_dynamics_operators",
    "Graph",
    "Field",
    "Group",
    "compare",
    "constrain",
    "Constraint",
    "Event",
    "eq",
    "equal",
    "ne",
    "increasing",
    "decreasing",
    "monotone",
    "convex",
    "concave",
    "lipschitz",
    "ode_residual",
    "GaussianField",
    "FieldKernel",
    "RandomWalk",
    "RBF",
    "Proxy",
    "GaussianProxy",
    "LogisticNicheProxy",
    "PoissonProxy",
    "CustomProxy",
    "fit_field",
    "FieldPosterior",
    "GP",
    "Gaussian",
    "Niche",
    "Cox",
    "joint",
    "FieldModel",
    "Differential",
    "shape_optimize",
    "level_set_material",
    "TotalVariation",
    "Potts",
    "multistart",
    "NavierStokes2D",
    "conformal",
    "ConformalRegressor",
    "ConformalClassifier",
    "ConformalQuantileRegressor",
    "ConformalStructure",
    "ConformalLinkPredictor",
    "ConformalKnowledgeGraph",
]


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


# --- linear-Gaussian state space (time series) ------------------------------------
def _ss_err(*a, **k):
    raise NotImplementedError("state-space models are fit via fit(); they have no single dist.")


register_composite("StateSpace", _ss_err, _ss_err)


# --- PDE-constrained latent field (spatiotemporal) --------------------------------
def _pde_err(*a, **k):
    raise NotImplementedError("PDE models are fit via fit(); they have no single dist.")


register_composite("PDEStateSpace", _pde_err, _pde_err)

from pysp.ppl.dynamics import (  # noqa: E402  (after composite registration)
    AdvectionDiffusionOperator,
    AdvectionOperator,
    DiffusionOperator,
    available_dynamics_operators,
    make_operator,
    register_dynamics_operator,
)


# --- constructors: conventional parameterizations, return symbolic RandomVariables ---
def Normal(mean: Any, sd: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Normal with mean and standard deviation (lowers to GaussianDistribution(mu, sd**2))."""
    return RandomVariable._sample("Normal", (mean, sd), name=name, keys=keys)


def Poisson(rate: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    return RandomVariable._sample("Poisson", (rate,), name=name, keys=keys)


def Gamma(shape: Any, rate: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Gamma with shape and rate (lowers to GammaDistribution(k=shape, theta=1/rate))."""
    return RandomVariable._sample("Gamma", (shape, rate), name=name, keys=keys)


def Exponential(rate: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Exponential with rate (mean 1/rate; lowers to ExponentialDistribution(beta=1/rate))."""
    return RandomVariable._sample("Exponential", (rate,), name=name, keys=keys)


def Bernoulli(p: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    return RandomVariable._sample("Bernoulli", (p,), name=name, keys=keys)


def Geometric(p: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    return RandomVariable._sample("Geometric", (p,), name=name, keys=keys)


def Beta(a: Any, b: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    return RandomVariable._sample("Beta", (a, b), name=name, keys=keys)


def Dirichlet(
    alpha: Any, *, dim: int | None = None, name: str | None = None, keys: str | None = None
) -> RandomVariable:
    """Dirichlet over a simplex; used as a prior on Categorical probabilities (VMP). The
    concentration ``alpha`` is also an inferable parameter: ``Dirichlet(free, dim=K)`` estimates
    a positive ``K``-vector from observed simplex data via ``how='mcmc'|'ensemble'|'map'``."""
    if alpha is free:
        if dim is None:
            raise ValueError("Dirichlet(free, dim=K) needs the dimension dim.")
        alpha = _VectorSpec(int(dim), "positive", name="alpha")
    return RandomVariable._sample("Dirichlet", (alpha,), name=name, keys=keys)


def Graph():
    """A VMP factor graph for arbitrary conjugate-Gaussian DAGs with shared variables.
    See pysp.ppl.vmp.Graph."""
    from pysp.ppl.vmp import Graph as _Graph

    return _Graph()


def StudentT(df: Any, loc: Any, scale: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Student-t with degrees of freedom, location, scale (heavy-tailed Normal)."""
    return RandomVariable._sample("StudentT", (df, loc, scale), name=name, keys=keys)


def LogNormal(mu: Any, sigma: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Log-normal: log(X) ~ Normal(mu, sigma)."""
    return RandomVariable._sample("LogNormal", (mu, sigma), name=name, keys=keys)


def EMG(mu: Any, sigma: Any, rate: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Exponentially-modified Gaussian: ``X = Normal(mu, sigma) + Exponential(rate)`` (right-skewed).

    Lowers to ``ExponentiallyModifiedGaussianDistribution(mu, sigma**2, lam=rate)``; ``rate`` is the
    exponential component's rate (its mean is ``1/rate``). The MLE is iterative with no closed form,
    so ``EMG(free, free, free).fit(data)`` uses a consistent method-of-moments estimate."""
    return RandomVariable._sample("EMG", (mu, sigma, rate), name=name, keys=keys)


def NegativeBinomial(r: Any, p: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Negative binomial with r failures and success probability p."""
    return RandomVariable._sample("NegativeBinomial", (r, p), name=name, keys=keys)


def Categorical(
    probs: Any, *, dim: int | None = None, name: str | None = None, keys: str | None = None
) -> RandomVariable:
    """Categorical from a probability dict {value: p} or a list of probabilities. The probability
    vector is also an inferable simplex parameter: ``Categorical(free, dim=K)`` estimates the K
    category probabilities (on the simplex) via ``how='mcmc'|'ensemble'|'map'``."""
    if probs is free:
        if dim is None:
            raise ValueError("Categorical(free, dim=K) needs the number of categories dim.")
        probs = _SimplexSpec(np.ones(int(dim)), rows=1, name="p")
    return RandomVariable._sample("Categorical", (probs,), name=name, keys=keys)


def Weibull(shape: Any, scale: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Weibull with shape (k) and scale (lambda)."""
    return RandomVariable._sample("Weibull", (shape, scale), name=name, keys=keys)


def Laplace(loc: Any, scale: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Laplace (double-exponential) with location and scale (b)."""
    return RandomVariable._sample("Laplace", (loc, scale), name=name, keys=keys)


def Logistic(loc: Any, scale: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Logistic with location and scale."""
    return RandomVariable._sample("Logistic", (loc, scale), name=name, keys=keys)


def Uniform(low: Any, high: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Continuous uniform on [low, high]."""
    return RandomVariable._sample("Uniform", (low, high), name=name, keys=keys)


def Rayleigh(sigma: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Rayleigh with scale sigma."""
    return RandomVariable._sample("Rayleigh", (sigma,), name=name, keys=keys)


def Pareto(scale: Any, shape: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Pareto with minimum value xm (scale) and tail index alpha (shape)."""
    return RandomVariable._sample("Pareto", (scale, shape), name=name, keys=keys)


def Binomial(n: Any, p: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Binomial with n trials and success probability p (n is fixed/known)."""
    return RandomVariable._sample("Binomial", (n, p), name=name, keys=keys)


def _as_rv(c: Any) -> RandomVariable:
    if isinstance(c, RandomVariable):
        return c
    return RandomVariable._bound(c)  # a concrete pysp distribution


def Mix(components, weights=None, *, name: str | None = None) -> RandomVariable:
    """Finite mixture over component RandomVariables (or concrete distributions).

    ``Mix([Normal(free, free), Normal(free, free)]).fit(data)`` fits a 2-component
    Gaussian mixture; ``.posterior(data)`` returns the responsibilities.
    """
    comps = tuple(_as_rv(c) for c in components)
    return RandomVariable._sample("Mixture", (comps, weights), name=name)


def SemiMix(components, weights=None, *, name: str | None = None) -> RandomVariable:
    """Semi-supervised finite mixture over component RandomVariables (or concrete distributions).

    Like :func:`Mix`, but each observation is a ``(value, prior)`` pair where ``prior`` is either
    ``None`` (unlabeled) or a sequence of ``(component_index, probability)`` pairs giving a partial
    label. Labeled rows restrict/re-weight the responsibilities to the listed components, so a few
    labels can anchor the components. ``SemiMix([Normal(free, free), Normal(free, free)]).fit(data)``
    fits a 2-component Gaussian mixture from a mix of labeled and unlabeled rows.
    """
    comps = tuple(_as_rv(c) for c in components)
    return RandomVariable._sample("SemiMix", (comps, weights), name=name)


def Seq(element, *, name: str | None = None) -> RandomVariable:
    """IID sequence of ``element``. Fit on a list of sequences (each a list/array)."""
    return RandomVariable._sample("Sequence", (_as_rv(element),), name=name)


def LocalLevel(*, name: str | None = None) -> RandomVariable:
    """Local-level state-space model (random walk + noise) for a time series. Fit on a 1-D
    series; recovers level/observation noise and smoothed states (Kalman/RTS + EM)."""
    return RandomVariable._sample("StateSpace", (False,), name=name)


def AR1(*, name: str | None = None) -> RandomVariable:
    """AR(1)-plus-noise state-space model; estimates the autoregressive coefficient phi."""
    return RandomVariable._sample("StateSpace", (True,), name=name)


def PDE(operator: Any, *, name: str | None = None) -> RandomVariable:
    """PDE-constrained latent-field model for spatiotemporal data.

    ``operator`` is a :class:`pysp.ppl.dynamics.DynamicsOperator` (e.g. ``DiffusionOperator``,
    ``AdvectionOperator``) whose method-of-lines discretization fixes the linear state
    transition. Fit on a ``(T, m)`` array of noisy field observations: the Kalman/RTS smoother
    recovers the latent field and EM estimates the process/observation noise levels while the
    physics-derived dynamics are held fixed. Pass ``dt=`` and an optional sensor operator ``H=``
    to ``fit()``."""
    return RandomVariable._sample("PDEStateSpace", (operator,), name=name)


def _mean_spec(mean, dim):
    """Mean-vector parameter spec: ``free`` -> real vector, ``ordered`` -> increasing vector."""
    if mean is free:
        return _VectorSpec(dim, "real", name="m")
    if mean is ordered:
        return _OrderedSpec(dim, name="m")
    return mean


def MVN(dim: int, *, mean=None, cov=None, name: str | None = None) -> RandomVariable:
    """Multivariate Gaussian of dimension ``dim`` (full covariance). Fit on a list of
    length-``dim`` vectors; ``MVN(dim).fit(X)`` recovers mean and covariance by EM.

    The **mean vector** and **covariance matrix** are also inferable parameters: pass
    ``mean=free`` (a ``dim``-vector on the real line) or ``mean=ordered`` (increasing entries,
    for identifiability) and/or ``cov=free`` (a full SPD covariance via its Cholesky factor) and
    fit with ``how='mcmc'|'ensemble'|'map'``."""
    dim = int(dim)
    cov_spec = _CholeskySpec(dim, name="S") if cov is free else cov
    return RandomVariable._sample("MVN", (dim, _mean_spec(mean, dim), cov_spec), name=name)


def DiagGaussian(dim: int, *, mean=None, var=None, name: str | None = None) -> RandomVariable:
    """Diagonal-covariance multivariate Gaussian of dimension ``dim``. ``DiagGaussian(dim).fit(X)``
    recovers mean and per-axis variance by EM; the **mean vector** (``mean=free`` / ``ordered``)
    and **diagonal variances** (``var=free``, a positive vector) are also inferable parameters via
    ``how='mcmc'|'ensemble'|'map'``."""
    dim = int(dim)
    var_spec = _VectorSpec(dim, "positive", name="s2") if var is free else var
    return RandomVariable._sample("DiagGaussian", (dim, _mean_spec(mean, dim), var_spec), name=name)


def LDA(num_topics: int, vocab_size: int, *, alpha: float = 1.0, name: str | None = None) -> RandomVariable:
    """Latent Dirichlet allocation. Fit on a list of documents, each a bag of
    ``(word_id, count)`` pairs over word ids ``0..vocab_size-1``. Topics are recovered
    as word distributions; alpha (the document-topic Dirichlet) is fixed by default."""
    return RandomVariable._sample("LDA", (int(num_topics), int(vocab_size), float(alpha)), name=name)


def _simplex_arg(spec, rows: int, k: int):
    """Turn a transitions=/initial= argument into a stored value: a fixed array stays an array;
    ``free`` or a ``Dirichlet`` prior becomes a ``_SimplexSpec`` (estimable simplex / simplex
    rows); ``None`` stays None (EM estimates / uniform default)."""
    if spec is None:
        return None
    if isinstance(spec, RandomVariable) and spec._kind == "sample" and spec._family.name == "Dirichlet":
        return _SimplexSpec(spec._args[0], rows=rows, name=spec._name)
    if spec is free:
        return _SimplexSpec(np.ones(k), rows=rows)
    return np.asarray(spec, dtype=float)  # a fixed transition matrix / initial distribution


def Markov(
    emission, states: int | None = None, *, transitions=None, initial=None, name: str | None = None
) -> RandomVariable:
    """Hidden Markov model over latent states emitting ``emission``.

    ``Markov(Normal(free, free), states=2).fit(sequences)`` fits a 2-state Gaussian HMM by EM
    (emissions k-means++ seeded so states separate); ``.posterior(sequences)`` gives state
    posteriors. For per-state priors pass a **list** of emissions, one per state:
    ``Markov([Normal(m0, 1), Normal(m1, 1)])``. The **transition matrix** and **initial
    distribution** are inferable parameters too: pass ``transitions=free`` /
    ``transitions=Dirichlet(alpha)`` (each row a simplex) and/or ``initial=free`` /
    ``initial=Dirichlet(alpha)`` and fit with ``how='mcmc'|'ensemble'|'map'`` (typically with an
    ordered-emission constraint for identifiability).
    """
    if isinstance(emission, (list, tuple)):
        comps = tuple(_as_rv(e) for e in emission)
        states = len(comps)
    else:
        if states is None:
            raise ValueError("Markov(emission, states=...) needs states, or a list of emissions.")
        comps = tuple(_as_rv(emission) for _ in range(states))
    trans = _simplex_arg(transitions, states, states)
    init = _simplex_arg(initial, 1, states)
    return RandomVariable._sample("Markov", (comps, trans, init), name=name)
