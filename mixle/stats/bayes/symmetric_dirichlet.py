"""Symmetric Dirichlet distribution on probability vectors with a single shared concentration alpha.

Observations are length-n sequences/arrays of non-negative reals summing to one (points on the
(n-1)-simplex), scored with one shared concentration parameter alpha. The log-density is

    log f(x; alpha) = sum_k (alpha - 1)*log(x_k) + gammaln(n*alpha) - n*gammaln(alpha),

where n = len(x) is inferred from each observation.

This is a parameter prior (the conjugate Dirichlet prior used by
:class:`~mixle.stats.univariate.discrete.integer_categorical.IntegerCategoricalDistribution` when a symmetric prior is desired). It
is scored on probability vectors, not fit from data by EM. Ported from mixle.bstats.symdirichlet.
"""

from typing import Any

import numpy as np

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
)
from mixle.utils.special import digamma, gammaln


class SymmetricDirichletDistribution(SequenceEncodableProbabilityDistribution):
    """Symmetric Dirichlet distribution with shared concentration alpha; the dimension is inferred
    from each observation (or fixed with dim for sampling)."""

    def __init__(self, alpha: float, dim: int | None = None, name: str | None = None) -> None:
        """Create a symmetric Dirichlet distribution.

        Args:
            alpha (float): Shared positive concentration parameter.
            dim (Optional[int]): Dimension of the probability vectors. Only required for sampling;
                log_density infers the dimension from each observation.
            name (Optional[str]): Name of object.

        """
        self.dim = dim
        self.alpha = float(alpha)
        self.name = name

    def __str__(self) -> str:
        return "SymmetricDirichletDistribution(%s, dim=%s, name=%s)" % (
            repr(self.alpha),
            repr(self.dim),
            repr(self.name),
        )

    def get_parameters(self) -> float:
        """Returns the shared concentration parameter alpha."""
        return self.alpha

    def set_parameters(self, params: float) -> None:
        """Set the shared concentration parameter alpha."""
        self.alpha = float(params)

    def density(self, x: np.ndarray | list[float]) -> float:
        """Density at the probability vector x (exp of log_density)."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: np.ndarray | list[float]) -> float:
        """Log-density of the symmetric Dirichlet at the probability vector x."""
        nc = len(x) * gammaln(self.alpha) - gammaln(len(x) * self.alpha)
        if self.alpha == 1:
            return float(-nc)
        else:
            return float(np.sum(np.log(x) * (self.alpha - 1)) - nc)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized log-density at sequence-encoded (m, n) array of probability vectors."""
        log_x = x
        if len(log_x) == 0:
            return np.zeros(0, dtype=float)
        n = log_x.shape[1]
        nc = n * gammaln(self.alpha) - gammaln(n * self.alpha)
        rv = np.zeros(log_x.shape[0]) - nc
        if self.alpha != 1:
            rv += log_x.sum(axis=1) * (self.alpha - 1)
        return rv

    def entropy(self) -> float:
        """Differential entropy in nats (requires dim to be set)."""
        n = self.dim
        if n is None:
            raise ValueError("SymmetricDirichletDistribution.entropy requires dim to be set.")
        a = np.ones(n) * self.alpha
        a0 = np.sum(a)
        return float(-((gammaln(a0) - np.sum(gammaln(a))) + np.dot(digamma(a) - digamma(a0), a - 1)))

    def sampler(self, seed: int | None = None) -> "SymmetricDirichletSampler":
        """Returns a SymmetricDirichletSampler for this distribution."""
        return SymmetricDirichletSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "ParameterEstimator":
        """SymmetricDirichlet is a parameter prior and is not fit from data by EM."""
        raise NotImplementedError("SymmetricDirichletDistribution is a parameter prior; it has no data estimator.")

    def dist_to_encoder(self) -> "SymmetricDirichletDataEncoder":
        """Returns a SymmetricDirichletDataEncoder for encoding probability vectors."""
        return SymmetricDirichletDataEncoder()


class SymmetricDirichletSampler(DistributionSampler):
    """Draws probability vectors from a SymmetricDirichletDistribution with a known dimension."""

    def __init__(self, dist: SymmetricDirichletDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray:
        """Draw symmetric-Dirichlet-distributed probability vectors (requires dist.dim)."""
        a = self.dist.alpha
        n = getattr(self.dist, "dim", None)
        if n is None:
            raise ValueError(
                "SymmetricDirichletSampler requires SymmetricDirichletDistribution(alpha, dim=...) "
                "with a specified dimension."
            )
        return self.rng.dirichlet(np.ones(n) * a, size=size)


class SymmetricDirichletDataEncoder(DataSequenceEncoder):
    """Encodes a sequence of probability vectors into an (m, n) float array of log values."""

    def __str__(self) -> str:
        return "SymmetricDirichletDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SymmetricDirichletDataEncoder)

    def seq_encode(self, x: Any) -> np.ndarray:
        """Encode simplex observations and their clipped log values."""
        import sys

        rv = np.asarray(x, dtype=float)
        rv2 = np.maximum(rv, sys.float_info.min)
        np.log(rv2, out=rv2)
        return rv2
