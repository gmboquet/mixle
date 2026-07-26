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
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import digamma, gammaln, logsumexp

from mixle.utils.optional_deps import HAS_PANDAS, pandas, require

# Canonical guarded softmax (all-(-inf) slices -> uniform). For the finite 1-D inputs used here this
# matches the previous local `_softmax`; the guard only changes the degenerate all-(-inf) case.
from mixle.utils.special import softmax as _softmax


def _float64_array(value: Any, label: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label} must be a real numeric array.") from exc
    if np.iscomplexobj(raw):
        raise TypeError(f"{label} must be real-valued.")
    try:
        return np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label} must be a real numeric array.") from exc


def _as_rng(rng: Any) -> RandomState:
    return rng if isinstance(rng, RandomState) else RandomState(rng)


def _cat(rng: RandomState, p: np.ndarray) -> int:
    cdf = np.cumsum(p)
    return int(np.searchsorted(cdf, rng.random_sample() * cdf[-1], side="right"))


def _entropy(p: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(-np.sum(np.where(p > 0.0, p * np.log(p), 0.0)))


class ImpossiblePosteriorError(ValueError):
    """Raised when an operation requires a posterior law but the evidence has zero probability."""


@dataclass(frozen=True, slots=True)
class ImpossiblePosteriorResult:
    """Typed explanation that a model/evidence pair does not define a conditional probability law."""

    reason: str
    log_evidence: float = float("-inf")


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
        probabilities = _float64_array(responsibilities, "responsibilities")
        if probabilities.ndim != 2:
            raise ValueError("responsibilities must be a 2-D (N, K) matrix")
        self.n, self.k = probabilities.shape
        if self.k == 0:
            raise ValueError("responsibilities must contain at least one latent category.")
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
            raise ValueError("responsibilities must contain finite non-negative probabilities.")
        row_sums = probabilities.sum(axis=1)
        if not np.allclose(row_sums, 1.0, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("each responsibility row must sum to one.")
        # Normalize only round-off-sized deviations already accepted above; never assign missing
        # probability mass to a particular category.
        self.responsibilities = np.array(probabilities / row_sums[:, None], copy=True)
        self.responsibilities.setflags(write=False)
        if support is None:
            self.support = np.arange(self.k)
        else:
            try:
                labels = list(support)
            except TypeError as exc:
                raise TypeError("support must be an iterable with one label per category.") from exc
            if len(labels) != self.k:
                raise ValueError(f"support must contain exactly {self.k} labels.")
            self.support = np.empty(self.k, dtype=object)
            for i, label in enumerate(labels):
                self.support[i] = label
        self.support.setflags(write=False)

    def marginals(self) -> np.ndarray:
        """The ``(N, K)`` responsibility matrix."""
        return self.responsibilities

    def sample(self, rng: Any = None) -> np.ndarray:
        """Draw one latent label per observation; returns an ``(N,)`` array of support labels."""
        rng = _as_rng(rng)
        idx = np.fromiter((_cat(rng, row) for row in self.responsibilities), dtype=int, count=self.n)
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
        log_pi = _float64_array(log_pi, "log_pi")
        log_A = _float64_array(log_A, "log_A")
        log_b = _float64_array(log_b, "log_b")
        if log_pi.ndim != 1 or log_pi.size == 0:
            raise ValueError("log_pi must be a non-empty (K,) vector.")
        k = log_pi.size
        if log_A.shape != (k, k):
            raise ValueError(f"log_A must have shape ({k}, {k}).")
        if log_b.ndim != 2 or log_b.shape[1] != k:
            raise ValueError(f"log_b must have shape (T, {k}).")
        for label, values in (("log_pi", log_pi), ("log_A", log_A), ("log_b", log_b)):
            if np.isnan(values).any() or np.isposinf(values).any():
                raise ValueError(f"{label} may contain finite values or -inf, but not NaN or +inf.")
        pi_norm = float(logsumexp(log_pi))
        transition_norms = logsumexp(log_A, axis=1)
        if not np.isfinite(pi_norm) or not np.isclose(pi_norm, 0.0, rtol=0.0, atol=1.0e-10):
            raise ValueError("log_pi must encode a normalized probability vector.")
        if not np.isfinite(transition_norms).all() or not np.allclose(
            transition_norms, 0.0, rtol=0.0, atol=1.0e-10
        ):
            raise ValueError("each log_A row must encode a normalized transition probability vector.")
        self.log_pi = np.array(log_pi - pi_norm, copy=True)
        self.log_A = np.array(log_A - transition_norms[:, None], copy=True)
        self.log_b = np.array(log_b, copy=True)
        self.log_pi.setflags(write=False)
        self.log_A.setflags(write=False)
        self.log_b.setflags(write=False)
        self.t, self.k = self.log_b.shape
        self._log_alpha = self._forward()  # alpha_t(k) = log p(z_t=k, x_{1:t}), the FFBS filter
        impossible_at = (
            None
            if self.t == 0
            else next((t for t in range(self.t) if np.isneginf(self._log_alpha[t]).all()), None)
        )
        self.impossibility = (
            None
            if impossible_at is None
            else ImpossiblePosteriorResult(
                f"the evidence has zero probability at sequence position {impossible_at}; "
                "no conditional latent-state law exists."
            )
        )

    def _forward(self) -> np.ndarray:
        la = np.empty((self.t, self.k))
        if self.t == 0:
            return la
        la[0] = self.log_pi + self.log_b[0]
        for t in range(1, self.t):
            la[t] = self.log_b[t] + logsumexp(la[t - 1][:, None] + self.log_A, axis=0)
        return la

    def log_likelihood(self) -> float:
        """The sequence log-likelihood ``log p(x)`` (the forward normalizer)."""
        if self.t == 0:
            return 0.0
        return float(logsumexp(self._log_alpha[-1]))

    @property
    def is_impossible(self) -> bool:
        """Whether the supplied evidence has zero probability under the chain model."""
        return self.impossibility is not None

    def _require_possible(self) -> None:
        if self.impossibility is not None:
            raise ImpossiblePosteriorError(self.impossibility.reason)

    def _backward(self) -> np.ndarray:
        lb = np.zeros((self.t, self.k))
        for t in range(self.t - 2, -1, -1):
            lb[t] = logsumexp(self.log_A + (self.log_b[t + 1] + lb[t + 1])[None, :], axis=1)
        return lb

    def marginals(self) -> np.ndarray:
        """The ``(T, K)`` smoothing probabilities ``q(z_t = k | x)``."""
        self._require_possible()
        if self.t == 0:
            return np.empty((0, self.k))
        log_gamma = self._log_alpha + self._backward()
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)

    def sample(self, rng: Any = None) -> np.ndarray:
        """Draw a state path ``z ~ q(z | x)`` via FFBS; returns ``(T,)`` state indices."""
        self._require_possible()
        if self.t == 0:
            return np.empty(0, dtype=int)
        rng = _as_rng(rng)
        z = np.empty(self.t, dtype=int)
        z[-1] = _cat(rng, _softmax(self._log_alpha[-1]))  # z_T ~ filter at the last step (= smoother)
        for t in range(self.t - 2, -1, -1):
            # z_t ~ q(z_t | z_{t+1}, x_{1:t}) prop. alpha_t(.) * A(., z_{t+1})
            z[t] = _cat(rng, _softmax(self._log_alpha[t] + self.log_A[:, z[t + 1]]))
        return z

    def mode(self) -> np.ndarray:
        """The Viterbi (max-product) MAP path ``(T,)``."""
        self._require_possible()
        if self.t == 0:
            return np.empty(0, dtype=int)
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
        self._require_possible()
        if self.t == 0:
            return 0.0
        gamma = self.marginals()
        h = _entropy(_softmax(self._log_alpha[-1]))
        for t in range(self.t - 1):
            logp = self._log_alpha[t][:, None] + self.log_A  # (j, k): unnormalized q(z_t=j, z_{t+1}=k)
            for k in np.flatnonzero(gamma[t + 1] > 0.0):
                conditional = _softmax(logp[:, k])
                h += float(gamma[t + 1, k]) * _entropy(conditional)
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
        gamma = _float64_array(gamma, "gamma")
        phi = _float64_array(phi, "phi")
        try:
            raw_counts = np.asarray(counts)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("counts must be a real numeric array.") from exc
        if np.iscomplexobj(raw_counts):
            raise TypeError("counts must be real-valued.")
        if gamma.ndim != 1 or gamma.size == 0 or not np.isfinite(gamma).all() or np.any(gamma <= 0.0):
            raise ValueError("gamma must be a non-empty vector of finite positive Dirichlet parameters.")
        with np.errstate(over="ignore"):
            gamma_total = gamma.sum()
        if not np.isfinite(gamma_total):
            raise ValueError("gamma parameters must have a finite positive total.")
        k = gamma.shape[0]
        if phi.ndim != 2 or phi.shape[1] != k:
            raise ValueError(f"phi must have shape (W, {k}).")
        if not np.isfinite(phi).all() or np.any(phi < 0.0):
            raise ValueError("phi must contain finite non-negative probabilities.")
        row_sums = phi.sum(axis=1)
        if not np.allclose(row_sums, 1.0, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("each phi row must sum to one.")
        if raw_counts.ndim != 1 or raw_counts.shape[0] != phi.shape[0]:
            raise ValueError("counts must be a (W,) vector aligned with phi rows.")
        if np.issubdtype(raw_counts.dtype, np.bool_) or any(
            isinstance(value, (bool, np.bool_)) for value in raw_counts
        ):
            raise TypeError("counts must contain exact non-negative integers, not booleans.")
        if np.issubdtype(raw_counts.dtype, np.integer):
            if np.any(raw_counts < 0) or (
                np.issubdtype(raw_counts.dtype, np.unsignedinteger)
                and np.any(raw_counts > np.iinfo(np.int64).max)
            ):
                raise ValueError("counts must contain exact non-negative integers representable as int64.")
            integer_counts = np.asarray(raw_counts, dtype=np.int64)
        else:
            try:
                numeric_counts = np.asarray(raw_counts, dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError("counts must contain exact non-negative integers.") from exc
            if (
                not np.isfinite(numeric_counts).all()
                or np.any(numeric_counts < 0.0)
                or np.any(numeric_counts != np.floor(numeric_counts))
                or np.any(numeric_counts >= float(2**63))
            ):
                raise ValueError("counts must contain exact non-negative integers representable as int64.")
            integer_counts = np.asarray(numeric_counts, dtype=np.int64)
        self.gamma = np.array(gamma, copy=True)
        self.phi = np.array(phi / row_sums[:, None], copy=True)
        self.counts = np.array(integer_counts, dtype=np.int64, copy=True)
        self.gamma.setflags(write=False)
        self.phi.setflags(write=False)
        self.counts.setflags(write=False)
        self.k = k

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
        z = []
        for w, c in enumerate(self.counts):
            z.extend(_cat(rng, self.phi[w]) for _ in range(int(c)))
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
