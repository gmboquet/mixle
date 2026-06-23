"""The pysp.ppl distribution dialect — conventional constructors returning symbolic RandomVariables.

``Normal(mean, sd)``, ``Gamma(shape, rate)``, ``Mix([...])``, ``Markov(emission, states=k)``,
``MVN(dim)``, … — each returns a :class:`~pysp.ppl.core.RandomVariable` in a registered family (the
registrations live in :mod:`pysp.ppl._lowering`). A parameter slot accepts a concrete value, the token
``free`` (estimate it), or another ``RandomVariable``. Extracted from ``pysp/ppl/__init__.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pysp.ppl.core import (
    RandomVariable,
    _CholeskySpec,
    _OrderedSpec,
    _SimplexSpec,
    _VectorSpec,
    free,
    ordered,
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
