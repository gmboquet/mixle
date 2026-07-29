"""Symmetric Dirichlet distribution on probability vectors with a single shared concentration alpha.

Observations are length-n sequences/arrays of non-negative reals summing to one (points on the
(n-1)-simplex), scored with one shared concentration parameter alpha. The log-density is

    log f(x; alpha) = sum_k (alpha - 1)*log(x_k) + gammaln(n*alpha) - n*gammaln(alpha),

where n = len(x) is inferred from each observation.

This is a parameter prior (the conjugate Dirichlet prior used by
:class:`~mixle.stats.univariate.discrete.integer_categorical.IntegerCategoricalDistribution` when a symmetric prior is desired). It
is scored on probability vectors, not fit from data by EM. Ported from mixle.bstats.symdirichlet.
"""

import operator
from typing import Any

import numpy as np

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
)
from mixle.utils.special import digamma, gammaln

# Tolerance for the "does this vector sum to one" simplex check -- numpy's own np.isclose default
# (rtol=1e-05, atol=1e-08), not a bespoke float64-tuned bound. Real probability rows scored here are
# not always float64: e.g. IntegerMarkovChainDistribution.cond_dist (see
# mixle.stats.sequences.integer_markov_chain, which uses this class as its implicit row prior) is
# stored in float32, whose row sums land ~1e-8 to ~1e-7 away from 1.0 -- outside a naive 1e-10/1e-12
# bound (which would reject every row of a legitimately fitted float32 matrix as off-simplex) but
# comfortably inside this one, while still four-plus orders of magnitude tighter than any sum that
# would indicate a genuinely invalid input.
_SIMPLEX_SUM_RTOL = 1.0e-5
_SIMPLEX_SUM_ATOL = 1.0e-8


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
        self.alpha = self._validated_alpha(alpha)
        self.dim = self._validated_dimension(dim)
        self.name = name

    @staticmethod
    def _validated_alpha(value: Any) -> float:
        if np.ndim(value) != 0:
            raise ValueError("SymmetricDirichletDistribution alpha must be a scalar.")
        try:
            alpha = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("SymmetricDirichletDistribution requires a positive finite concentration alpha.") from exc
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("SymmetricDirichletDistribution requires a positive finite concentration alpha.")
        return alpha

    @staticmethod
    def _validated_dimension(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("SymmetricDirichletDistribution dim must be a positive integer.")
        try:
            dimension = operator.index(value)
        except TypeError as exc:
            raise TypeError("SymmetricDirichletDistribution dim must be a positive integer.") from exc
        if dimension <= 0:
            raise ValueError("SymmetricDirichletDistribution dim must be positive.")
        return int(dimension)

    def _validated_state(self) -> tuple[float, int | None]:
        return self._validated_alpha(self.alpha), self._validated_dimension(self.dim)

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
        self.alpha = self._validated_alpha(params)

    def density(self, x: np.ndarray | list[float]) -> float:
        """Density at the probability vector x (exp of log_density)."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: np.ndarray | list[float]) -> float:
        """Log-density of the symmetric Dirichlet at the probability vector x (``-inf`` off the
        simplex: a negative or non-finite entry, or a vector that doesn't sum to one)."""
        alpha, dimension = self._validated_state()
        try:
            xx = np.asarray(x, dtype=np.float64)
        except (TypeError, ValueError):
            return -np.inf
        if xx.ndim != 1 or xx.size == 0 or not np.all(np.isfinite(xx)) or np.any(xx < 0.0):
            return -np.inf
        if dimension is not None and xx.shape != (dimension,):
            return -np.inf
        if not np.isclose(float(xx.sum()), 1.0, rtol=_SIMPLEX_SUM_RTOL, atol=_SIMPLEX_SUM_ATOL):
            return -np.inf
        n = xx.shape[0]
        nc = n * gammaln(alpha) - gammaln(n * alpha)
        if alpha == 1:
            return float(-nc)
        else:
            with np.errstate(divide="ignore"):
                return float(np.sum(np.log(xx) * (alpha - 1)) - nc)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized log-density at a sequence-encoded (m, n) array of probability vectors (``-inf``
        rows that are off the simplex: a negative or non-finite entry, or a row that doesn't sum to
        one)."""
        alpha, dimension = self._validated_state()
        try:
            xx = np.asarray(x, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("SymmetricDirichletDistribution encoded observations must be numeric.") from exc
        if xx.ndim != 2:
            raise ValueError("SymmetricDirichletDistribution encoded observations must have exact shape (n, d).")
        if dimension is not None and xx.shape[1] != dimension:
            raise ValueError("SymmetricDirichletDistribution encoded observations must have %d columns." % dimension)
        if xx.shape[0] == 0:
            return np.zeros(0, dtype=float)
        n = xx.shape[1]
        if n == 0:
            raise ValueError("SymmetricDirichletDistribution observations cannot have zero dimension.")
        nc = n * gammaln(alpha) - gammaln(n * alpha)
        good = (
            np.all(np.isfinite(xx), axis=1)
            & np.all(xx >= 0.0, axis=1)
            & np.isclose(xx.sum(axis=1), 1.0, rtol=_SIMPLEX_SUM_RTOL, atol=_SIMPLEX_SUM_ATOL)
        )
        rv = np.full(xx.shape[0], -np.inf, dtype=float)
        if alpha == 1:
            rv[good] = -nc
        elif np.any(good):
            with np.errstate(divide="ignore"):
                log_xx = np.log(xx[good])
            rv[good] = log_xx.sum(axis=1) * (alpha - 1) - nc
        return rv

    def entropy(self) -> float:
        """Differential entropy in nats (requires dim to be set)."""
        alpha, n = self._validated_state()
        if n is None:
            raise ValueError("SymmetricDirichletDistribution.entropy requires dim to be set.")
        a = np.ones(n) * alpha
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
        return SymmetricDirichletDataEncoder(self.dim)


class SymmetricDirichletSampler(DistributionSampler):
    """Draws probability vectors from a SymmetricDirichletDistribution with a known dimension."""

    def __init__(self, dist: SymmetricDirichletDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray:
        """Draw symmetric-Dirichlet-distributed probability vectors (requires dist.dim)."""
        a, n = self.dist._validated_state()
        if n is None:
            raise ValueError(
                "SymmetricDirichletSampler requires SymmetricDirichletDistribution(alpha, dim=...) "
                "with a specified dimension."
            )
        if size is not None:
            if isinstance(size, (bool, np.bool_)):
                raise TypeError("SymmetricDirichletSampler size must be a non-negative integer.")
            try:
                size = operator.index(size)
            except TypeError as exc:
                raise TypeError("SymmetricDirichletSampler size must be a non-negative integer.") from exc
            if size < 0:
                raise ValueError("SymmetricDirichletSampler size must be non-negative.")
        return self.rng.dirichlet(np.ones(n) * a, size=size)


class SymmetricDirichletDataEncoder(DataSequenceEncoder):
    """Encodes a sequence of probability vectors into an (m, n) float array of log values."""

    def __init__(self, dimension: int | None = None) -> None:
        self.dimension = SymmetricDirichletDistribution._validated_dimension(dimension)

    def __str__(self) -> str:
        return "SymmetricDirichletDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SymmetricDirichletDataEncoder) and self.dimension == other.dimension

    def seq_encode(self, x: Any) -> np.ndarray:
        """Encode simplex observations as a raw (m, n) float array.

        Kept raw (not pre-logged/clipped) so ``seq_log_density`` can validate simplex membership --
        negative entries, non-finite entries, rows not summing to one -- from the actual values; a
        pre-clipped log representation would silently discard exactly the information needed to
        reject those.
        """
        try:
            values = np.asarray(x, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("SymmetricDirichletDataEncoder observations must be numeric.") from exc
        if values.ndim != 2:
            raise ValueError("SymmetricDirichletDataEncoder observations must have exact shape (n, d).")
        if values.shape[1] == 0:
            raise ValueError("SymmetricDirichletDataEncoder observations cannot have zero dimension.")
        if self.dimension is not None and values.shape[1] != self.dimension:
            raise ValueError("SymmetricDirichletDataEncoder observations must have %d columns." % self.dimension)
        return values
