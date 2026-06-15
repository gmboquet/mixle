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

from typing import Any, Optional

import numpy as np

from pysp.ppl.core import (
    RandomVariable, free, lower, register_family, register_composite, Family, Field,
)

from pysp.stats.mixture import MixtureDistribution, MixtureEstimator
from pysp.stats.sequence import SequenceDistribution, SequenceEstimator
from pysp.stats.hidden_markov import HiddenMarkovModelDistribution, HiddenMarkovEstimator
from pysp.stats.gaussian import GaussianDistribution, GaussianEstimator
from pysp.stats.poisson import PoissonDistribution, PoissonEstimator
from pysp.stats.gamma import GammaDistribution, GammaEstimator
from pysp.stats.exponential import ExponentialDistribution, ExponentialEstimator
from pysp.stats.categorical import CategoricalDistribution, CategoricalEstimator
from pysp.stats.bernoulli import BernoulliDistribution, BernoulliEstimator
from pysp.stats.geometric import GeometricDistribution, GeometricEstimator
from pysp.stats.beta import BetaDistribution, BetaEstimator
from pysp.stats.student_t import StudentTDistribution, StudentTEstimator
from pysp.stats.log_gaussian import LogGaussianDistribution, LogGaussianEstimator
from pysp.stats.negative_binomial import NegativeBinomialDistribution, NegativeBinomialEstimator
from pysp.stats.dirichlet import DirichletDistribution, DirichletEstimator

__all__ = [
    "RandomVariable", "free", "lower",
    "Normal", "Poisson", "Gamma", "Exponential", "Categorical", "Bernoulli", "Geometric",
    "Beta", "StudentT", "LogNormal", "NegativeBinomial", "Dirichlet",
    "Mix", "Seq", "Markov", "Graph", "Field",
]


# --- family registration: (user-facing args) -> (underlying *Distribution kwargs) ---
register_family("Normal", GaussianDistribution, GaussianEstimator,
                lambda mean, sd: {"mu": float(mean), "sigma2": float(sd) ** 2}, arity=2,
                seed_at=lambda v, s: {"mu": float(v), "sigma2": (float(s) ** 2) or 1.0},
                positive=(False, True),
                read=lambda d: {"mean": d.mu, "sd": float(np.sqrt(d.sigma2))})
register_family("Poisson", PoissonDistribution, PoissonEstimator,
                lambda rate: {"lam": float(rate)}, arity=1,
                seed_at=lambda v, s: {"lam": max(float(v), 1e-2)}, positive=(True,),
                read=lambda d: {"rate": d.lam})
register_family("Gamma", GammaDistribution, GammaEstimator,
                lambda shape, rate: {"k": float(shape), "theta": 1.0 / float(rate)}, arity=2,
                seed_at=lambda v, s: {"k": 1.0, "theta": max(float(v), 1e-2)},
                positive=(True, True),
                read=lambda d: {"shape": d.k, "rate": 1.0 / d.theta})
register_family("Exponential", ExponentialDistribution, ExponentialEstimator,
                lambda rate: {"beta": 1.0 / float(rate)}, arity=1,
                seed_at=lambda v, s: {"beta": max(float(v), 1e-2)}, positive=(True,),
                read=lambda d: {"rate": 1.0 / d.beta})
register_family("Bernoulli", BernoulliDistribution, BernoulliEstimator,
                lambda p: {"p": float(p)}, arity=1, read=lambda d: {"p": d.p})
register_family("Geometric", GeometricDistribution, GeometricEstimator,
                lambda p: {"p": float(p)}, arity=1, read=lambda d: {"p": d.p})
register_family("Beta", BetaDistribution, BetaEstimator,
                lambda a, b: {"a": float(a), "b": float(b)}, arity=2, positive=(True, True),
                read=lambda d: {"a": d.a, "b": d.b})
register_family("Dirichlet", DirichletDistribution, DirichletEstimator,
                lambda alpha: {"alpha": np.asarray(alpha, dtype=float)}, arity=1,
                read=lambda d: {"alpha": np.asarray(d.alpha),
                                "mean": np.asarray(d.alpha) / float(np.sum(d.alpha))})
register_family("StudentT", StudentTDistribution, StudentTEstimator,
                lambda df, loc, scale: {"df": float(df), "loc": float(loc), "scale": float(scale)},
                arity=3, seed_at=lambda v, s: {"df": 5.0, "loc": float(v), "scale": (float(s) or 1.0)},
                positive=(True, False, True),
                read=lambda d: {"df": d.df, "loc": d.loc, "scale": d.scale})
register_family("LogNormal", LogGaussianDistribution, LogGaussianEstimator,
                lambda mu, sigma: {"mu": float(mu), "sigma2": float(sigma) ** 2}, arity=2,
                seed_at=lambda v, s: {"mu": float(np.log(max(v, 1e-3))), "sigma2": 1.0},
                positive=(False, True),
                read=lambda d: {"mu": d.mu, "sigma": float(np.sqrt(d.sigma2))})
def _nb_init(data):
    a = np.asarray(data, dtype=float)
    mu, var = float(a.mean()), float(a.var())
    r0 = mu * mu / max(var - mu, 1e-3)          # moment match: var = mu + mu^2/r
    p0 = r0 / (r0 + mu)
    return NegativeBinomialDistribution(max(r0, 1e-2), min(max(p0, 1e-3), 1 - 1e-3))


register_family("NegativeBinomial", NegativeBinomialDistribution, NegativeBinomialEstimator,
                lambda r, p: {"r": float(r), "p": float(p)}, arity=2, positive=(True, False),
                init_fit=_nb_init, read=lambda d: {"r": d.r, "p": d.p})


def _cat_args(probs):
    if isinstance(probs, dict):
        return {"pmap": dict(probs)}
    return {"pmap": {i: float(v) for i, v in enumerate(probs)}}


register_family("Categorical", CategoricalDistribution, CategoricalEstimator, _cat_args, arity=1)


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
    children = [seed_child(c, data[i], scale) for c, i in zip(comps, idx)]
    if any(ch is None for ch in children):
        return None  # a component can't be seeded -> fall back to pysp default init
    w = np.ones(len(comps)) / len(comps) if weights is None else np.asarray(weights, float)
    return MixtureDistribution(children, w=w)


def _mix_read(d, read_params):
    return {"components": [read_params(c) for c in d.components],
            "weights": np.asarray(d.w)}


register_composite("Mixture", _mix_dist, _mix_est, seed_fn=_mix_seed,
                   dist_cls=MixtureDistribution, read=_mix_read)


# --- Sequence: iid elements (+ optional length model) -----------------------------
def _seq_dist(args, lower_child):
    (elem,) = args
    return SequenceDistribution(lower_child(elem))


def _seq_est(args, lower_child_est, name, keys):
    (elem,) = args
    return SequenceEstimator(lower_child_est(elem), name=name)


def _seq_read(d, read_params):
    return {"element": read_params(d.dist)}


register_composite("Sequence", _seq_dist, _seq_est,
                   dist_cls=SequenceDistribution, read=_seq_read)


# --- Markov / HMM: latent-state sequence model ------------------------------------
def _hmm_dist(args, lower_child):
    comps, _ = args
    topics = [lower_child(c) for c in comps]
    k = len(topics)
    return HiddenMarkovModelDistribution(topics, w=np.ones(k) / k,
                                         transitions=np.ones((k, k)) / k)


def _hmm_est(args, lower_child_est, name, keys):
    comps, _ = args
    return HiddenMarkovEstimator([lower_child_est(c) for c in comps], name=name)


def _flatten_obs(data):
    parts = [np.asarray(s, dtype=float).reshape(-1) for s in data if len(s) > 0]
    return np.concatenate(parts) if parts else np.asarray([], dtype=float)


def _hmm_seed(args, data, rng, seed_child):
    comps, _ = args
    k = len(comps)
    arr = _flatten_obs(data)
    if arr.size < k:
        return None
    scale = float(np.std(arr)) or 1.0
    idx = _kmeanspp_idx(arr, k, rng)
    topics = [seed_child(c, arr[i], scale) for c, i in zip(comps, idx)]
    if any(t is None for t in topics):
        return None
    return HiddenMarkovModelDistribution(topics, w=np.ones(k) / k,
                                         transitions=np.ones((k, k)) / k)


def _hmm_read(d, read_params):
    return {"states": [read_params(t) for t in d.topics],
            "transitions": np.asarray(d.transitions), "initial": np.asarray(d.w)}


register_composite("Markov", _hmm_dist, _hmm_est, seed_fn=_hmm_seed,
                   dist_cls=HiddenMarkovModelDistribution, read=_hmm_read)


# --- constructors: conventional parameterizations, return symbolic RandomVariables ---
def Normal(mean: Any, sd: Any, *, name: Optional[str] = None, keys: Optional[str] = None) -> RandomVariable:
    """Normal with mean and standard deviation (lowers to GaussianDistribution(mu, sd**2))."""
    return RandomVariable._sample("Normal", (mean, sd), name=name, keys=keys)


def Poisson(rate: Any, *, name: Optional[str] = None, keys: Optional[str] = None) -> RandomVariable:
    return RandomVariable._sample("Poisson", (rate,), name=name, keys=keys)


def Gamma(shape: Any, rate: Any, *, name: Optional[str] = None, keys: Optional[str] = None) -> RandomVariable:
    """Gamma with shape and rate (lowers to GammaDistribution(k=shape, theta=1/rate))."""
    return RandomVariable._sample("Gamma", (shape, rate), name=name, keys=keys)


def Exponential(rate: Any, *, name: Optional[str] = None, keys: Optional[str] = None) -> RandomVariable:
    """Exponential with rate (mean 1/rate; lowers to ExponentialDistribution(beta=1/rate))."""
    return RandomVariable._sample("Exponential", (rate,), name=name, keys=keys)


def Bernoulli(p: Any, *, name: Optional[str] = None, keys: Optional[str] = None) -> RandomVariable:
    return RandomVariable._sample("Bernoulli", (p,), name=name, keys=keys)


def Geometric(p: Any, *, name: Optional[str] = None, keys: Optional[str] = None) -> RandomVariable:
    return RandomVariable._sample("Geometric", (p,), name=name, keys=keys)


def Beta(a: Any, b: Any, *, name: Optional[str] = None, keys: Optional[str] = None) -> RandomVariable:
    return RandomVariable._sample("Beta", (a, b), name=name, keys=keys)


def Dirichlet(alpha: Any, *, name: Optional[str] = None, keys: Optional[str] = None) -> RandomVariable:
    """Dirichlet over a simplex; used as a prior on Categorical probabilities (VMP)."""
    return RandomVariable._sample("Dirichlet", (alpha,), name=name, keys=keys)


def Graph():
    """A VMP factor graph for arbitrary conjugate-Gaussian DAGs with shared variables.
    See pysp.ppl.vmp.Graph."""
    from pysp.ppl.vmp import Graph as _Graph
    return _Graph()


def StudentT(df: Any, loc: Any, scale: Any, *, name: Optional[str] = None,
             keys: Optional[str] = None) -> RandomVariable:
    """Student-t with degrees of freedom, location, scale (heavy-tailed Normal)."""
    return RandomVariable._sample("StudentT", (df, loc, scale), name=name, keys=keys)


def LogNormal(mu: Any, sigma: Any, *, name: Optional[str] = None,
              keys: Optional[str] = None) -> RandomVariable:
    """Log-normal: log(X) ~ Normal(mu, sigma)."""
    return RandomVariable._sample("LogNormal", (mu, sigma), name=name, keys=keys)


def NegativeBinomial(r: Any, p: Any, *, name: Optional[str] = None,
                     keys: Optional[str] = None) -> RandomVariable:
    """Negative binomial with r failures and success probability p."""
    return RandomVariable._sample("NegativeBinomial", (r, p), name=name, keys=keys)


def Categorical(probs: Any, *, name: Optional[str] = None, keys: Optional[str] = None) -> RandomVariable:
    """Categorical from a probability dict {value: p} or a list of probabilities."""
    return RandomVariable._sample("Categorical", (probs,), name=name, keys=keys)


def _as_rv(c: Any) -> RandomVariable:
    if isinstance(c, RandomVariable):
        return c
    return RandomVariable._bound(c)  # a concrete pysp distribution


def Mix(components, weights=None, *, name: Optional[str] = None) -> RandomVariable:
    """Finite mixture over component RandomVariables (or concrete distributions).

    ``Mix([Normal(free, free), Normal(free, free)]).fit(data)`` fits a 2-component
    Gaussian mixture; ``.posterior(data)`` returns the responsibilities.
    """
    comps = tuple(_as_rv(c) for c in components)
    return RandomVariable._sample("Mixture", (comps, weights), name=name)


def Seq(element, *, name: Optional[str] = None) -> RandomVariable:
    """IID sequence of ``element``. Fit on a list of sequences (each a list/array)."""
    return RandomVariable._sample("Sequence", (_as_rv(element),), name=name)


def Markov(emission, states: int, *, name: Optional[str] = None) -> RandomVariable:
    """Hidden Markov model with ``states`` latent states emitting ``emission``.

    ``Markov(Normal(free, free), states=2).fit(sequences)`` fits a 2-state Gaussian HMM;
    emissions are k-means++ seeded so states separate. ``.posterior(sequences)`` gives
    state posteriors.
    """
    comps = tuple(_as_rv(emission) for _ in range(states))
    return RandomVariable._sample("Markov", (comps, None), name=name)
