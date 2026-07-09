"""The mixle.ppl distribution dialect — conventional constructors returning symbolic RandomVariables.

``Normal(mean, sd)``, ``Gamma(shape, rate)``, ``Mix([...])``, ``Markov(emission, states=k)``,
``MVN(dim)``, … — each returns a :class:`~mixle.ppl.core.RandomVariable` in a registered family (the
registrations live in :mod:`mixle.ppl._lowering`). A parameter slot accepts a concrete value, the token
``free`` (estimate it), or another ``RandomVariable``. Extracted from ``mixle/ppl/__init__.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.ppl.core import (
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
    """Symbolic normal distribution parameterized by mean and standard deviation.

    The user-facing scale is ``sd``. Lowering converts it to the variance
    parameter used by ``GaussianDistribution`` while fitted artifacts should
    remain interpretable in the constructor's scale parameterization.
    """
    return RandomVariable._sample("Normal", (mean, sd), name=name, keys=keys)


def Poisson(rate: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Symbolic Poisson count distribution parameterized by non-negative rate."""
    return RandomVariable._sample("Poisson", (rate,), name=name, keys=keys)


def Gamma(shape: Any, rate: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Symbolic Gamma distribution parameterized by shape and rate.

    The underlying stats family uses scale ``theta``; lowering maps
    ``rate`` to ``theta = 1 / rate``.
    """
    return RandomVariable._sample("Gamma", (shape, rate), name=name, keys=keys)


def Exponential(rate: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Symbolic exponential distribution parameterized by event rate."""
    return RandomVariable._sample("Exponential", (rate,), name=name, keys=keys)


def Bernoulli(p: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Bernoulli distribution for binary outcomes with success probability ``p``."""
    return RandomVariable._sample("Bernoulli", (p,), name=name, keys=keys)


def Geometric(p: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Symbolic geometric count distribution with success probability ``p``."""
    return RandomVariable._sample("Geometric", (p,), name=name, keys=keys)


def Beta(a: Any, b: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Beta distribution over probabilities with concentration parameters ``a`` and ``b``."""
    return RandomVariable._sample("Beta", (a, b), name=name, keys=keys)


def Dirichlet(
    alpha: Any, *, dim: int | None = None, name: str | None = None, keys: str | None = None
) -> RandomVariable:
    """Dirichlet over a simplex; used as a prior on Categorical probabilities (VMP). The
    concentration ``alpha`` is also an inferable parameter. ``Dirichlet(free)`` learns the
    concentration by maximum likelihood, inferring the dimension ``K`` from the observed simplex
    data (no ``dim=`` needed); pass ``dim=K`` to request the explicit positive-``K``-vector parameter
    treatment for ``how='mcmc'|'ensemble'|'map'``."""
    if alpha is free and dim is not None:
        alpha = _VectorSpec(int(dim), "positive", name="alpha")
    return RandomVariable._sample("Dirichlet", (alpha,), name=name, keys=keys)


def Graph():
    """A VMP factor graph for arbitrary conjugate-Gaussian DAGs with shared variables.
    See mixle.ppl.vmp.Graph."""
    from mixle.ppl.vmp import Graph as _Graph

    return _Graph()


def StudentT(df: Any, loc: Any, scale: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Symbolic Student-t distribution with degrees of freedom, location, and scale."""
    return RandomVariable._sample("StudentT", (df, loc, scale), name=name, keys=keys)


def LogNormal(mu: Any, sigma: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Symbolic log-normal distribution where ``log(X)`` is normal."""
    return RandomVariable._sample("LogNormal", (mu, sigma), name=name, keys=keys)


def EMG(mu: Any, sigma: Any, rate: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Exponentially-modified Gaussian: ``X = Normal(mu, sigma) + Exponential(rate)`` (right-skewed).

    Lowers to ``ExponentiallyModifiedGaussianDistribution(mu, sigma**2, lam=rate)``; ``rate`` is the
    exponential component's rate (its mean is ``1/rate``). The MLE is iterative with no closed form,
    so ``EMG(free, free, free).fit(data)`` uses a consistent method-of-moments estimate."""
    return RandomVariable._sample("EMG", (mu, sigma, rate), name=name, keys=keys)


def NegativeBinomial(r: Any, p: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Symbolic negative-binomial distribution with shape ``r`` and success probability ``p``."""
    return RandomVariable._sample("NegativeBinomial", (r, p), name=name, keys=keys)


def HalfNormal(sigma: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Half-normal on ``[0, inf)`` with scale ``sigma`` -- the standard weakly-informative scale prior."""
    return RandomVariable._sample("HalfNormal", (sigma,), name=name, keys=keys)


def InverseGamma(alpha: Any, beta: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Inverse-gamma(``alpha``, ``beta``) -- the classic conjugate prior for a variance."""
    return RandomVariable._sample("InverseGamma", (alpha, beta), name=name, keys=keys)


def InverseGaussian(mu: Any, lam: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Inverse-Gaussian (Wald) with mean ``mu`` and shape ``lam`` -- a positive, right-skewed law."""
    return RandomVariable._sample("InverseGaussian", (mu, lam), name=name, keys=keys)


def Gumbel(loc: Any, scale: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Gumbel (type-I extreme-value) with ``loc`` and ``scale`` -- for maxima / extremes."""
    return RandomVariable._sample("Gumbel", (loc, scale), name=name, keys=keys)


def SkewNormal(loc: Any, scale: Any, shape: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Skew-normal with ``loc``, ``scale``, and ``shape`` (skewness; ``shape=0`` recovers the Normal)."""
    return RandomVariable._sample("SkewNormal", (loc, scale, shape), name=name, keys=keys)


def Skellam(mu1: Any, mu2: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Skellam: the difference of two independent ``Poisson(mu1)`` and ``Poisson(mu2)`` counts."""
    return RandomVariable._sample("Skellam", (mu1, mu2), name=name, keys=keys)


def LogSeries(p: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Logarithmic (log-series) distribution on ``{1, 2, ...}`` with parameter ``p`` in ``(0, 1)``."""
    return RandomVariable._sample("LogSeries", (p,), name=name, keys=keys)


def VonMises(mu: Any, kappa: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Von Mises (circular normal) on the angle ``(-pi, pi]`` with mean ``mu`` and concentration ``kappa``."""
    return RandomVariable._sample("VonMises", (mu, kappa), name=name, keys=keys)


def GEV(loc: Any, scale: Any, shape: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Generalized extreme value with ``loc``, ``scale``, ``shape`` (the block-maxima limit law)."""
    return RandomVariable._sample("GEV", (loc, scale, shape), name=name, keys=keys)


def Tweedie(mu: Any, phi: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Tweedie compound Poisson-Gamma (power ``p=1.5``) with mean ``mu`` and dispersion ``phi``.

    A positive distribution with an atom at zero -- the standard model for non-negative data that is
    part-zero, part-continuous (insurance claims, rainfall, ecological biomass).
    """
    return RandomVariable._sample("Tweedie", (mu, phi), name=name, keys=keys)


def GeneralizedGaussian(
    mu: Any, alpha: Any, beta: Any, *, name: str | None = None, keys: str | None = None
) -> RandomVariable:
    """Generalized Gaussian (exponential power) with ``mu``, scale ``alpha``, shape ``beta``.

    ``beta=2`` is the Normal and ``beta=1`` is the Laplace, so it interpolates between light and heavy
    tails -- a flexible symmetric error model.
    """
    return RandomVariable._sample("GeneralizedGaussian", (mu, alpha, beta), name=name, keys=keys)


def GeneralizedPareto(
    scale: Any, shape: Any, loc: Any = 0.0, *, name: str | None = None, keys: str | None = None
) -> RandomVariable:
    """Generalized Pareto (``scale``, tail ``shape``, threshold ``loc``) -- the peaks-over-threshold tail law."""
    return RandomVariable._sample("GeneralizedPareto", (scale, shape, loc), name=name, keys=keys)


def Nakagami(m: Any, omega: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Nakagami-``m`` distribution with shape ``m`` and spread ``omega`` (signal-fading amplitudes)."""
    return RandomVariable._sample("Nakagami", (m, omega), name=name, keys=keys)


def Rician(nu: Any, sigma: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Rician (Rice) distribution with non-centrality ``nu`` and scale ``sigma`` (signal-plus-noise magnitude)."""
    return RandomVariable._sample("Rician", (nu, sigma), name=name, keys=keys)


def Categorical(
    probs: Any = None, *, logits: Any = None, dim: int | None = None, name: str | None = None, keys: str | None = None
) -> RandomVariable:
    """Categorical from a probability dict {value: p} or a list of probabilities. The probability
    vector is also an inferable parameter. ``Categorical(free)`` learns the category probabilities
    by maximum likelihood, discovering the categories (and their count) from the data -- no ``dim=``
    needed; pass ``dim=K`` to request the explicit simplex-parameter treatment for
    ``how='mcmc'|'ensemble'|'map'``.

    ``Categorical(logits=Net(out=K))`` is **neural classification**: ``p(y|x) = softmax(Net(x))``, the
    softmax-link sibling of logistic regression. Fit with the conditional verb ``.fit(y, given={"x": X})``."""
    if logits is not None:
        return RandomVariable._sample("Categorical", (logits,), name=name, keys=keys)
    if probs is free and dim is not None:
        probs = _SimplexSpec(np.ones(int(dim)), rows=1, name="p")
    return RandomVariable._sample("Categorical", (probs,), name=name, keys=keys)


def Weibull(shape: Any, scale: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Symbolic Weibull distribution with positive shape and scale parameters."""
    return RandomVariable._sample("Weibull", (shape, scale), name=name, keys=keys)


def Laplace(loc: Any, scale: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Symbolic Laplace distribution with location and positive scale."""
    return RandomVariable._sample("Laplace", (loc, scale), name=name, keys=keys)


def Logistic(loc: Any, scale: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Symbolic logistic distribution with location and positive scale."""
    return RandomVariable._sample("Logistic", (loc, scale), name=name, keys=keys)


def Uniform(low: Any, high: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Symbolic continuous uniform distribution on the closed interval ``[low, high]``."""
    return RandomVariable._sample("Uniform", (low, high), name=name, keys=keys)


def Rayleigh(sigma: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Symbolic Rayleigh distribution with positive scale ``sigma``."""
    return RandomVariable._sample("Rayleigh", (sigma,), name=name, keys=keys)


def Pareto(scale: Any, shape: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Symbolic Pareto distribution with minimum value ``scale`` and tail index ``shape``."""
    return RandomVariable._sample("Pareto", (scale, shape), name=name, keys=keys)


def Binomial(n: Any, p: Any, *, name: str | None = None, keys: str | None = None) -> RandomVariable:
    """Symbolic binomial distribution with ``n`` trials and success probability ``p``."""
    return RandomVariable._sample("Binomial", (n, p), name=name, keys=keys)


def _as_rv(c: Any) -> RandomVariable:
    if isinstance(c, RandomVariable):
        return c
    return RandomVariable._bound(c)  # a concrete mixle distribution


def Mix(components, weights=None, *, name: str | None = None) -> RandomVariable:
    """Symbolic finite mixture over PPL variables or concrete distributions.

    ``Mix([Normal(free, free), Normal(free, free)]).fit(data)`` fits a 2-component
    Gaussian mixture. The fitted object exposes mixture responsibilities through
    posterior helpers and inherits the numeric stability contract of the stats
    mixture implementation.
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
    """Symbolic iid sequence whose observations are lists or arrays of ``element``."""
    return RandomVariable._sample("Sequence", (_as_rv(element),), name=name)


def LocalLevel(*, name: str | None = None) -> RandomVariable:
    """Local-level state-space model (random walk + noise) for a time series. Fit on a 1-D
    series; recovers level/observation noise and smoothed states (Kalman/RTS + EM)."""
    return RandomVariable._sample("StateSpace", (False,), name=name)


def AR1(*, name: str | None = None) -> RandomVariable:
    """AR(1)-plus-noise state-space model; estimates the autoregressive coefficient phi."""
    return RandomVariable._sample("StateSpace", (True,), name=name)


# The PDE(operator) constructor lives in the mixle-pde plugin (it lowers to the PDEStateSpace
# family that plugin registers); mixle.ppl no longer ships PDE-constrained modeling.


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
