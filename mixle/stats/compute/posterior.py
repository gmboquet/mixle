"""The ``Posterior`` hierarchy: a uniform, samplable object for any model-derived distribution.

``Posterior`` is the shared base contract -- *inference produces posteriors; you draw from them
through one interface*. Every realization answers the same questions where they are defined::

    sample(rng)        -> a single draw
    samples(n, rng)    -> n draws (loop by default; vectorized override where cheaper)
    mean() / mode()    -> the posterior mean / MAP configuration
    marginals()        -> per-component marginals (e.g. the EM E-step responsibilities)
    entropy()          -> H[q]                       (the ELBO entropy term)
    interval(level)    -> a credible interval

This base lives in the compute layer next to the sampler contracts (``DistributionSampler`` /
``ConditionalSampler`` in :mod:`mixle.stats.compute.pdist`) so both ``mixle.stats`` and
``mixle.inference`` can build on it without a layering inversion. The richer realizations that need
inference machinery -- parameter posteriors (conjugate / MCMC) and the posterior-predictive, plus the
``posterior(model, ...)`` factory -- live in :mod:`mixle.inference.posterior`.

``LatentPosterior`` is the latent ``q(z | x)`` subtype implemented here: each latent model handles its
hidden variables implicitly inside EM (the E-step returns raw responsibility arrays); ``LatentPosterior``
makes ``q(z | x)`` a single object -- *exact* for mixtures/HMMs, *mean-field* for LDA/VMP -- so the EM
E-step (``marginals``), latent sampling (``sample``), and the ELBO entropy term are methods on it.
Mean-field realizations additionally provide ``expected_complete_ll(dist)`` / ``update(dist)`` /
``elbo(dist)``; for exact posteriors those are not needed.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import digamma, gammaln, logsumexp

from mixle.utils.optional_deps import HAS_PANDAS, pandas, require

# Canonical guarded softmax (all-(-inf) slices -> uniform). For the finite 1-D inputs used here this
# matches the previous local `_softmax`; the guard only changes the degenerate all-(-inf) case.
from mixle.utils.special import softmax as _softmax


def _as_rng(rng: Any) -> RandomState:
    return rng if isinstance(rng, RandomState) else RandomState(rng)


def _cat(rng: RandomState, p: np.ndarray) -> int:
    return int(np.searchsorted(np.cumsum(p), rng.random_sample() * p.sum()))


def _entropy(p: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(-np.sum(np.where(p > 0.0, p * np.log(p), 0.0)))


class Posterior(ABC):
    """A model-derived distribution exposing one uniform interface for draws and summaries.

    Only :meth:`sample` (a single draw) is required. :meth:`samples` loops it by default; vectorized
    subtypes override it. ``mean`` / ``mode`` / ``marginals`` / ``entropy`` / ``interval`` raise
    :class:`NotImplementedError` unless a subtype defines them, so each realization implements exactly
    the summaries that are meaningful for it.
    """

    @abstractmethod
    def sample(self, rng: Any = None) -> Any:
        """Draw a single sample from the posterior (``rng`` is a seed, ``RandomState``, or ``None``)."""

    def samples(self, n: int, rng: Any = None) -> Any:
        """Draw ``n`` samples; loops :meth:`sample` by default (override for a vectorized draw)."""
        rng = _as_rng(rng)
        return [self.sample(rng) for _ in range(int(n))]

    def mean(self) -> Any:
        """The posterior mean ``E[.]`` (not defined for every realization)."""
        raise NotImplementedError(f"{type(self).__name__} does not define mean()")

    def mode(self) -> Any:
        """The maximum-a-posteriori configuration (not defined for every realization)."""
        raise NotImplementedError(f"{type(self).__name__} does not define mode()")

    def marginals(self) -> Any:
        """Per-component marginals -- e.g. the EM E-step responsibilities (not always defined)."""
        raise NotImplementedError(f"{type(self).__name__} does not define marginals()")

    def entropy(self) -> Any:
        """The entropy ``H[q]`` (not always defined)."""
        raise NotImplementedError(f"{type(self).__name__} does not define entropy()")

    def interval(self, level: float = 0.9) -> Any:
        """A central credible interval at the given ``level`` (not always defined)."""
        raise NotImplementedError(f"{type(self).__name__} does not define interval()")


class LatentPosterior(Posterior):
    """The posterior ``q(z | x)`` over a model's latent variables (exact or mean-field)."""

    @abstractmethod
    def marginals(self) -> Any:
        """Per-latent marginal responsibilities -- the quantity the EM M-step consumes."""

    @abstractmethod
    def sample(self, rng: Any = None) -> Any:
        """Draw the latent variables ``z ~ q(z | x)`` (``rng`` is a seed, RandomState, or None)."""

    @abstractmethod
    def mode(self) -> Any:
        """The maximum-a-posteriori latent configuration."""

    @abstractmethod
    def entropy(self) -> Any:
        """The entropy ``H[q]`` (per latent / per observation)."""


class CategoricalLatentPosterior(LatentPosterior):
    """Independent categorical latents ``q(z) = prod_i Cat(z_i; r_i)``.

    The exact posterior for a finite mixture's component labels (and the per-token topic factor of an
    LDA document). ``responsibilities`` is the row-stochastic ``(N, K)`` matrix ``r_ik = q(z_i = k |
    x_i)``; ``support`` maps column ``k`` to its latent label (default ``0..K-1``).
    """

    def __init__(self, responsibilities: np.ndarray, support: Any = None) -> None:
        self.responsibilities = np.asarray(responsibilities, dtype=np.float64)
        if self.responsibilities.ndim != 2:
            raise ValueError("responsibilities must be a 2-D (N, K) matrix")
        self.n, self.k = self.responsibilities.shape
        self.support = np.arange(self.k) if support is None else np.asarray(list(support))

    def marginals(self) -> np.ndarray:
        """The ``(N, K)`` responsibility matrix."""
        return self.responsibilities

    def sample(self, rng: Any = None) -> np.ndarray:
        """Draw one latent label per observation; returns an ``(N,)`` array of support labels."""
        rng = _as_rng(rng)
        cdf = np.cumsum(self.responsibilities, axis=1)
        cdf[:, -1] = 1.0  # guard tiny round-off so every uniform draw lands in a bin
        u = rng.random_sample(self.n)[:, None]
        idx = (u < cdf).argmax(axis=1)
        return self.support[idx]

    def mode(self) -> np.ndarray:
        """The most-probable latent label per observation, ``(N,)``."""
        return self.support[np.argmax(self.responsibilities, axis=1)]

    def entropy(self) -> np.ndarray:
        """Per-observation entropy ``-sum_k r_ik log r_ik``, ``(N,)``."""
        r = self.responsibilities
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(r > 0.0, r * np.log(r), 0.0)
        return -terms.sum(axis=1)


class MarkovChainLatentPosterior(LatentPosterior):
    """Chain-structured latents ``q(z_1..z_T | x)`` for an HMM -- exact, via forward-backward.

    Built from the log initial distribution ``log_pi`` ``(K,)``, the log transition matrix ``log_A``
    ``(K, K)`` (row ``j`` -> column ``k``), and the per-position emission log-likelihoods ``log_b``
    ``(T, K)``. The latents are *coupled* (a Markov chain), so:

      marginals() -> the ``(T, K)`` forward-backward smoothing probabilities ``q(z_t = k | x)``
      sample(rng) -> a full state path ``(T,)`` by forward-filter / backward-sample (FFBS)
      mode()      -> the Viterbi (max-product) path ``(T,)``
      entropy()   -> the exact *scalar* chain entropy ``H[q(z_1..z_T | x)]``
    """

    def __init__(self, log_pi: np.ndarray, log_A: np.ndarray, log_b: np.ndarray) -> None:
        self.log_pi = np.asarray(log_pi, dtype=np.float64)
        self.log_A = np.asarray(log_A, dtype=np.float64)
        self.log_b = np.asarray(log_b, dtype=np.float64)
        self.t, self.k = self.log_b.shape
        self._log_alpha = self._forward()  # alpha_t(k) = log p(z_t=k, x_{1:t}), the FFBS filter

    def _forward(self) -> np.ndarray:
        la = np.empty((self.t, self.k))
        la[0] = self.log_pi + self.log_b[0]
        for t in range(1, self.t):
            la[t] = self.log_b[t] + logsumexp(la[t - 1][:, None] + self.log_A, axis=0)
        return la

    def log_likelihood(self) -> float:
        """The sequence log-likelihood ``log p(x)`` (the forward normalizer)."""
        return float(logsumexp(self._log_alpha[-1]))

    def _backward(self) -> np.ndarray:
        lb = np.zeros((self.t, self.k))
        for t in range(self.t - 2, -1, -1):
            lb[t] = logsumexp(self.log_A + (self.log_b[t + 1] + lb[t + 1])[None, :], axis=1)
        return lb

    def marginals(self) -> np.ndarray:
        """The ``(T, K)`` smoothing probabilities ``q(z_t = k | x)``."""
        log_gamma = self._log_alpha + self._backward()
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)

    def sample(self, rng: Any = None) -> np.ndarray:
        """Draw a state path ``z ~ q(z | x)`` via FFBS; returns ``(T,)`` state indices."""
        rng = _as_rng(rng)
        z = np.empty(self.t, dtype=int)
        z[-1] = _cat(rng, _softmax(self._log_alpha[-1]))  # z_T ~ filter at the last step (= smoother)
        for t in range(self.t - 2, -1, -1):
            # z_t ~ q(z_t | z_{t+1}, x_{1:t}) prop. alpha_t(.) * A(., z_{t+1})
            z[t] = _cat(rng, _softmax(self._log_alpha[t] + self.log_A[:, z[t + 1]]))
        return z

    def mode(self) -> np.ndarray:
        """The Viterbi (max-product) MAP path ``(T,)``."""
        v = np.empty((self.t, self.k))
        bp = np.zeros((self.t, self.k), dtype=int)
        v[0] = self.log_pi + self.log_b[0]
        for t in range(1, self.t):
            m = v[t - 1][:, None] + self.log_A
            bp[t] = np.argmax(m, axis=0)
            v[t] = self.log_b[t] + m.max(axis=0)
        z = np.empty(self.t, dtype=int)
        z[-1] = int(np.argmax(v[-1]))
        for t in range(self.t - 2, -1, -1):
            z[t] = bp[t + 1][z[t + 1]]
        return z

    def entropy(self) -> float:
        """Exact scalar chain entropy via the FFBS factorization ``q = q(z_T) prod_t q(z_t|z_{t+1})``."""
        gamma = self.marginals()
        h = _entropy(_softmax(self._log_alpha[-1]))
        for t in range(self.t - 1):
            logp = self._log_alpha[t][:, None] + self.log_A  # (j, k): unnormalized q(z_t=j, z_{t+1}=k)
            cond = np.exp(logp - logsumexp(logp, axis=0, keepdims=True))  # q(z_t=j | z_{t+1}=k)
            with np.errstate(divide="ignore", invalid="ignore"):
                h_k = -np.sum(np.where(cond > 0.0, cond * np.log(cond), 0.0), axis=0)  # (k,)
            h += float(np.sum(gamma[t + 1] * h_k))
        return h

    def to_dataframe(self) -> Any:
        """Return the per-position state posterior as a ``pandas.DataFrame``.

        One row per sequence position ``t`` with columns ``t`` (position), ``state`` (the Viterbi MAP
        state from :meth:`mode`), and one ``state_{k}_prob`` column per latent state holding the
        forward-backward smoothing probability from :meth:`marginals`. Deterministic: unlike
        :meth:`sample`, ``mode``/``marginals`` are exact closed-form quantities, not random draws.
        Requires the ``pandas`` extra (``pip install mixle[pandas]``).
        """
        if not HAS_PANDAS:
            require("pandas", "pandas")
        marginals = self.marginals()
        data = {"t": np.arange(self.t), "state": self.mode()}
        for k in range(self.k):
            data[f"state_{k}_prob"] = marginals[:, k]
        return pandas.DataFrame(data)

    def to_parquet(self, path: Any, **kwargs: Any) -> None:
        """Write the per-position state posterior to a Parquet file; see :meth:`to_dataframe`.

        ``kwargs`` forward to ``DataFrame.to_parquet`` (e.g. ``engine=``, ``compression=``). Needs a
        Parquet engine in addition to pandas -- ``pip install mixle[arrow]`` (pyarrow) or fastparquet.
        """
        self.to_dataframe().to_parquet(path, **kwargs)


class MeanFieldLDAPosterior(LatentPosterior):
    """Mean-field variational posterior for one LDA document: ``q(theta, z) = Dir(theta; gamma) prod_n Cat(z_n; phi_n)``.

    The Blei-Ng-Jordan variational factorization made into an object instead of loose ``gamma``/``phi``
    arrays. ``gamma`` ``(K,)`` is the document's variational Dirichlet parameter (``q(theta)``);
    ``phi`` ``(W, K)`` the per-*distinct*-word topic responsibilities (``q(z_n)``, rows sum to 1);
    ``counts`` ``(W,)`` the word counts. Note the latents are heterogeneous (continuous ``theta`` +
    discrete ``z``), so ``sample`` returns the pair ``(theta, z)`` and ``entropy`` is a scalar.
    """

    def __init__(self, gamma: np.ndarray, phi: np.ndarray, counts: np.ndarray) -> None:
        self.gamma = np.asarray(gamma, dtype=np.float64).ravel()
        self.phi = np.asarray(phi, dtype=np.float64)
        self.counts = np.asarray(counts)
        self.k = self.gamma.shape[0]

    def topic_proportions(self) -> np.ndarray:
        """The mean document-topic distribution ``E_q[theta] = gamma / sum(gamma)`` ``(K,)``."""
        return self.gamma / self.gamma.sum()

    def marginals(self) -> np.ndarray:
        """The ``(W, K)`` per-distinct-word topic responsibilities ``q(z_n)``."""
        return self.phi

    def sample(self, rng: Any = None) -> tuple[np.ndarray, np.ndarray]:
        """Draw the full latent ``(theta, z)``: ``theta ~ Dir(gamma)`` and per-*token* topics ``z`` from ``phi``."""
        rng = _as_rng(rng)
        theta = rng.dirichlet(self.gamma)
        cdf = np.cumsum(self.phi, axis=1)
        cdf[:, -1] = 1.0
        z = []
        for w, c in enumerate(self.counts):
            u = rng.random_sample(int(c))[:, None]
            z.extend((u < cdf[w]).argmax(axis=1).tolist())
        return theta, np.asarray(z, dtype=int)

    def mode(self) -> np.ndarray:
        """The MAP topic per distinct word, ``argmax_k phi_wk`` ``(W,)``."""
        return np.argmax(self.phi, axis=1)

    def entropy(self) -> float:
        """Mean-field entropy ``H[q(theta)] + sum_w count_w H[Cat(phi_w)]`` (scalar)."""
        g = self.gamma
        g0 = g.sum()
        h_theta = float(gammaln(g).sum() - gammaln(g0) + (g0 - self.k) * digamma(g0) - np.sum((g - 1.0) * digamma(g)))
        with np.errstate(divide="ignore", invalid="ignore"):
            h_z = -float(np.sum(self.counts[:, None] * np.where(self.phi > 0.0, self.phi * np.log(self.phi), 0.0)))
        return h_theta + h_z
