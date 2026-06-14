"""Symmetric Dirichlet distribution on probability vectors: observations are
length-n sequences/arrays of non-negative reals summing to one (points on the
(n-1)-simplex), scored with a single shared concentration parameter alpha.

Data type: (Sequence[float]/np.ndarray): A SymmetricDirichletDistribution with
concentration alpha has log-density

        log f(x; alpha) = sum_k (alpha-1)*log(x_k)
                                          + gammaln(n*alpha) - n*gammaln(alpha),

where n = len(x) is inferred from each observation. Defines the
SymmetricDirichletDistribution and SymmetricDirichletSampler classes for use
with pysp.bstats. This distribution is primarily used as a prior on the weight
vectors of other bstats models (e.g. pysp.bstats.mixture); since the dimension
is inferred per observation, sampling requires the optional dim argument.
"""

import numpy as np
from scipy.special import gammaln

from pysp.bstats.pdist import SequenceEncodableDistribution


class SymmetricDirichletDistribution(SequenceEncodableDistribution):
    """Symmetric Dirichlet distribution with shared concentration alpha; the
    dimension is inferred from each observation (or fixed with dim)."""

    def __init__(self, alpha: float, dim: int | None = None):
        """SymmetricDirichletDistribution object.

        Args:
                alpha (float): Shared positive concentration parameter.
                dim (Optional[int]): Dimension of the probability vectors. Only
                        required for sampling; log_density infers the dimension from
                        each observation.
        """
        self.dim = dim
        self.set_parameters(alpha)

    def __str__(self):
        return "SymmetricDirichletDistribution(%s)" % (str(self.alpha))

    def get_parameters(self) -> float:
        """Returns the shared concentration parameter alpha."""
        return self.alpha

    def set_parameters(self, params: float) -> None:
        """Sets the shared concentration parameter alpha.

        Args:
                params (float): New positive concentration parameter.
        """
        self.alpha = params

    def density(self, x: float | np.ndarray | list[float]) -> float:
        """Density at the probability vector x (exp of log_density).

        Args:
                x: Length-n sequence of non-negative reals summing to one.

        Returns:
                Density value at x.
        """
        return np.exp(self.log_density(x))

    def log_density(self, x: float | np.ndarray | list[float]) -> float:
        """Log-density of the symmetric Dirichlet at the probability vector x.

        Args:
                x: Length-n sequence of non-negative reals summing to one.

        Returns:
                Log-density value at x.
        """
        nc = len(x) * gammaln(self.alpha) - gammaln(len(x) * self.alpha)

        if self.alpha == 1:
            return -nc
        else:
            return np.sum(np.log(x) * (self.alpha - 1)) - nc

    def sampler(self, seed: int | None = None):
        """Returns a SymmetricDirichletSampler for this distribution.

        Args:
                seed (Optional[int]): Seed for the random number generator.
        """
        return SymmetricDirichletSampler(self, seed)


class SymmetricDirichletSampler:
    """Draws probability vectors from a SymmetricDirichletDistribution with a
    known dimension (dist.dim must be set)."""

    def __init__(self, dist: SymmetricDirichletDistribution, seed: int | None = None):
        """SymmetricDirichletSampler object.

        Args:
                dist (SymmetricDirichletDistribution): Distribution to sample from.
                seed (Optional[int]): Seed for the random number generator.
        """
        self.dist = dist
        self.dir = np.random.RandomState(seed)

    def sample(self, size: int | None = None) -> float | np.ndarray:
        """Draw symmetric-Dirichlet-distributed probability vectors.

        Args:
                size (Optional[int]): Number of samples to draw.

        Returns:
                np.ndarray of shape (dim,) if size is None, else (size, dim).

        Raises:
                ValueError: If the distribution was created without a dim, since
                        the sample dimension is then unspecified.
        """
        a = self.dist.alpha
        n = getattr(self.dist, "dim", None)
        if n is None:
            raise ValueError(
                "SymmetricDirichletSampler requires SymmetricDirichletDistribution(alpha, dim=...) with a specified dimension."
            )
        return self.dir.dirichlet(np.ones(n) * a, size=size)
