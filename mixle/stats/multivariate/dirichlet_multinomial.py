"""Dirichlet-multinomial (Polya) distribution -- an overdispersed multinomial.

The multivariate analogue of the beta-binomial: a multinomial whose category probabilities are
Dirichlet(alpha) distributed and integrated out. For a count vector ``x`` over ``K`` categories summing
to ``n``,

    P(x; alpha) = n!/prod_k x_k! * B(alpha + x) / B(alpha),    B(a) = prod_k Gamma(a_k) / Gamma(sum a),

which adds overdispersion (and category correlation) over a plain multinomial. The number of trials
``n`` is a fixed, known parameter; ``alpha`` is fit by Minka's maximum-likelihood fixed point, run from
a cumulative-count sufficient statistic so it converges inside a single ``estimate`` call.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import gammaln

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class DirichletMultinomialDistribution(SequenceEncodableProbabilityDistribution):
    """Dirichlet-multinomial over ``K``-category count vectors summing to ``n`` (concentration ``alpha``)."""

    def __init__(self, alpha: np.ndarray, n: int, name: str | None = None, keys: str | None = None) -> None:
        a = np.asarray(alpha, dtype=np.float64)
        if a.ndim != 1 or np.any(a <= 0.0) or not np.all(np.isfinite(a)):
            raise ValueError("alpha must be a 1-D vector of positive concentrations")
        if int(n) < 0:
            raise ValueError("n (number of trials) must be non-negative")
        self.alpha = a
        self.dim = a.shape[0]
        self.n = int(n)
        self.name = name
        self.keys = keys
        self._sum_alpha = float(a.sum())
        self._gammaln_alpha = gammaln(a)
        self._log_const = gammaln(self.n + 1) + gammaln(self._sum_alpha) - gammaln(self.n + self._sum_alpha)

    def __str__(self) -> str:
        return "DirichletMultinomialDistribution(%s, %s, name=%s, keys=%s)" % (
            repr(self.alpha.tolist()),
            repr(self.n),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: np.ndarray) -> float:
        """Return the probability mass at a single count vector ``x``."""
        return math.exp(self.log_density(x))

    def log_density(self, x: np.ndarray) -> float:
        """Return the log-mass at ``x`` (``-inf`` if any count is negative or the total is not ``n``)."""
        xx = np.asarray(x, dtype=np.float64)
        if xx.shape != (self.dim,) or np.any(xx < 0) or xx.sum() != self.n:
            return -np.inf
        term = gammaln(xx + self.alpha) - self._gammaln_alpha - gammaln(xx + 1.0)
        return float(self._log_const + term.sum())

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized log-mass for a stack of count vectors, shape ``(N, K)``."""
        xx = np.asarray(x, dtype=np.float64)
        term = gammaln(xx + self.alpha) - self._gammaln_alpha - gammaln(xx + 1.0)
        rv = self._log_const + term.sum(axis=1)
        bad = (xx < 0).any(axis=1) | (xx.sum(axis=1) != self.n)
        return np.where(bad, -np.inf, rv)

    def sampler(self, seed: int | None = None) -> "DirichletMultinomialSampler":
        """Return a sampler for drawing count vectors from this distribution."""
        return DirichletMultinomialSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "DirichletMultinomialEstimator":
        """Return a Minka fixed-point MLE estimator for ``alpha`` at the fixed number of trials ``n``."""
        return DirichletMultinomialEstimator(self.dim, self.n, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "DirichletMultinomialDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return DirichletMultinomialDataEncoder()


class DirichletMultinomialSampler(DistributionSampler):
    """Draw counts as ``p ~ Dirichlet(alpha)`` then ``x ~ Multinomial(n, p)``."""

    def __init__(self, dist: DirichletMultinomialDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> np.ndarray:
        """Draw one count vector or a stack of iid count vectors."""
        d = self.dist
        n_draws = 1 if size is None else int(size)
        p = self.rng.dirichlet(d.alpha, size=n_draws)
        out = np.array([self.rng.multinomial(d.n, pi) for pi in p])
        return out[0] if size is None else out


class DirichletMultinomialAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate cumulative counts ``c[k, j] = sum_i w_i 1{x_ik > j}`` (the Minka digamma-recurrence stat)."""

    def __init__(self, dim: int, n: int, name: str | None = None, keys: str | None = None) -> None:
        self.dim = dim
        self.n = n
        self.c = np.zeros((dim, max(n, 1)), dtype=np.float64)
        self.count = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: np.ndarray, weight: float, estimate: DirichletMultinomialDistribution | None) -> None:
        """Accumulate Minka recurrence statistics for one count vector."""
        xx = np.asarray(x, dtype=int)
        for k in range(self.dim):
            self.c[k, : xx[k]] += weight  # j = 0 .. x_k-1
        self.count += weight

    def initialize(self, x: np.ndarray, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one count vector."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any) -> None:
        """Accumulate Minka recurrence statistics from encoded count vectors."""
        xx = np.asarray(x, dtype=int)
        w = np.asarray(weights, dtype=np.float64)
        for k in range(self.dim):
            hist = np.bincount(xx[:, k], weights=w, minlength=self.n + 1)
            tail = np.cumsum(hist[::-1])[::-1]  # tail[v] = sum_{u>=v} hist[u]
            self.c[k, :] += tail[1 : self.n + 1]  # c[k,j] = sum_{v>j} hist[v]
        self.count += float(w.sum())

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded count vectors."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[np.ndarray, float]) -> "DirichletMultinomialAccumulator":
        """Merge another Dirichlet-multinomial sufficient-statistic tuple."""
        self.c += suff_stat[0]
        self.count += suff_stat[1]
        return self

    def value(self) -> tuple[np.ndarray, float]:
        """Return cumulative recurrence counts and total weight."""
        return self.c.copy(), self.count

    def from_value(self, x: tuple[np.ndarray, float]) -> "DirichletMultinomialAccumulator":
        """Replace accumulator contents from recurrence statistics."""
        self.c = np.asarray(x[0], dtype=np.float64).copy()
        self.count = float(x[1])
        self.dim, self.n = self.c.shape[0], self.c.shape[1]
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge keyed statistics into ``stats_dict`` when keys are configured."""
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator from keyed statistics when available."""
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> "DirichletMultinomialDataEncoder":
        """Return the encoder used by this accumulator."""
        return DirichletMultinomialDataEncoder()


class DirichletMultinomialAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for DirichletMultinomialAccumulator."""

    def __init__(self, dim: int, n: int, name: str | None = None, keys: str | None = None) -> None:
        self.dim = dim
        self.n = n
        self.name = name
        self.keys = keys

    def make(self) -> DirichletMultinomialAccumulator:
        """Create a fresh Dirichlet-multinomial accumulator."""
        return DirichletMultinomialAccumulator(self.dim, self.n, name=self.name, keys=self.keys)


class DirichletMultinomialEstimator(ParameterEstimator):
    """Minka fixed-point maximum-likelihood estimator for the Dirichlet-multinomial concentration."""

    def __init__(
        self,
        dim: int,
        n: int,
        max_iter: int = 500,
        tol: float = 1.0e-9,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.dim = dim
        self.n = n
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> DirichletMultinomialAccumulatorFactory:
        """Return an accumulator factory for Dirichlet-multinomial statistics."""
        return DirichletMultinomialAccumulatorFactory(self.dim, self.n, name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[np.ndarray, float]) -> DirichletMultinomialDistribution:
        """Estimate concentration parameters by Minka's fixed-point update."""
        c, count = suff_stat
        if count <= 0.0 or self.n == 0:
            return DirichletMultinomialDistribution(np.ones(self.dim), self.n, name=self.name, keys=self.keys)
        j = np.arange(self.n, dtype=np.float64)
        alpha = np.full(self.dim, 1.0, dtype=np.float64)
        for _ in range(self.max_iter):
            s = alpha.sum()
            # Minka: alpha_k <- alpha_k * [sum_j c[k,j]/(alpha_k+j)] / [N * sum_j 1/(s+j)]
            numer = (c / (alpha[:, None] + j[None, :])).sum(axis=1)
            denom = count * float((1.0 / (s + j)).sum())
            alpha_new = alpha * numer / denom
            if np.max(np.abs(alpha_new - alpha)) < self.tol:
                alpha = alpha_new
                break
            alpha = alpha_new
        return DirichletMultinomialDistribution(alpha, self.n, name=self.name, keys=self.keys)


class DirichletMultinomialDataEncoder(DataSequenceEncoder):
    """Encode a sequence of ``K``-category count vectors as an ``(N, K)`` array."""

    def __str__(self) -> str:
        return "DirichletMultinomialDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DirichletMultinomialDataEncoder)

    def seq_encode(self, x: Sequence[np.ndarray]) -> np.ndarray:
        """Encode count vectors as a floating-point matrix."""
        return np.asarray(x, dtype=np.float64)
