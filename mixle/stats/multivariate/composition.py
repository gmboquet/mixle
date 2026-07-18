"""Compositional data analysis: Aitchison logratio transforms and the logratio-normal distribution.

Geochemistry (and many earth-science) measurements are *compositions* -- vectors of non-negative parts
that sum to a constant (element abundances, mineral fractions, isotope splits). Ordinary statistics on
them is wrong: they live on the simplex, not in real space. Aitchison's logratio transforms (clr/ilr) map
the simplex isometrically to real coordinates where standard multivariate-Gaussian modelling applies; the
isometric logratio (ilr) uses an orthonormal basis so distances/covariances are preserved.

``AitchisonNormalDistribution`` is the logratio-normal -- ``ilr(x) ~ N(mean, cov)`` -- as a first-class
mixle distribution: it follows the ``Distribution`` / ``Sampler`` / ``Estimator`` / ``Accumulator`` /
``DataEncoder`` contract by *delegating* the Gaussian to :class:`MultivariateGaussianDistribution` after
the ilr transform, so it composes with the rest of the library (mixtures, the unified ``estimate``/
``sample`` entry points, ...) like any other leaf.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.stats.compute.pdist import (
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    StatisticAccumulatorFactory,
)
from mixle.stats.multivariate.multivariate_gaussian import (
    MultivariateGaussianDistribution,
    MultivariateGaussianEstimator,
)

__all__ = [
    "closure",
    "clr",
    "clr_inv",
    "ilr",
    "ilr_inv",
    "ilr_basis",
    "AitchisonNormalDistribution",
    "AitchisonNormalEstimator",
]


def closure(x: np.ndarray, total: float = 1.0) -> np.ndarray:
    """Normalize each row to sum to ``total`` (project onto the simplex)."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    return total * x / x.sum(axis=1, keepdims=True)


def clr(x: np.ndarray) -> np.ndarray:
    """Centered logratio: ``clr(x)_i = log(x_i) - mean_j log(x_j)``. Maps the simplex to the zero-sum
    hyperplane in ``R^D`` (the parts stay labelled, but the result is singular -- use ilr for modelling)."""
    lx = np.log(np.atleast_2d(np.asarray(x, dtype=float)))
    return lx - lx.mean(axis=1, keepdims=True)


def clr_inv(y: np.ndarray) -> np.ndarray:
    """Inverse clr (softmax onto the simplex)."""
    y = np.atleast_2d(np.asarray(y, dtype=float))
    e = np.exp(y - y.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def ilr_basis(d: int) -> np.ndarray:
    """A ``(D, D-1)`` orthonormal contrast basis (Helmert) for the isometric logratio of ``D`` parts."""
    v = np.zeros((d, d - 1))
    for i in range(d - 1):
        n = i + 1
        v[:n, i] = 1.0 / n
        v[n, i] = -1.0
        v[:, i] *= np.sqrt(n / (n + 1.0))
    return v


def ilr(x: np.ndarray, basis: np.ndarray | None = None) -> np.ndarray:
    """Isometric logratio: ``D``-part composition -> ``D-1`` real coordinates (orthonormal, so Euclidean
    distance in ilr space equals Aitchison distance on the simplex)."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    v = ilr_basis(x.shape[1]) if basis is None else basis
    return clr(x) @ v


def ilr_inv(y: np.ndarray, basis: np.ndarray | None = None) -> np.ndarray:
    """Inverse isometric logratio: ``D-1`` real coordinates -> ``D``-part composition on the simplex."""
    y = np.atleast_2d(np.asarray(y, dtype=float))
    v = ilr_basis(y.shape[1] + 1) if basis is None else basis
    return clr_inv(y @ v.T)


class AitchisonNormalDistribution(SequenceEncodableProbabilityDistribution):
    """A logratio-normal distribution on the simplex: ``ilr(x) ~ N(mean, cov)``.

    The natural Gaussian for compositions -- modelled in the orthonormal ilr coordinates, interpreted on
    the simplex. ``mean`` (length ``D-1``) and ``cov`` (``(D-1, D-1)``) are the ilr-space parameters of a
    ``D``-part composition; everything Gaussian is delegated to a :class:`MultivariateGaussianDistribution`.
    """

    def __init__(self, mean: np.ndarray, cov: np.ndarray, name: str | None = None, keys: str | None = None):
        self.gaussian = MultivariateGaussianDistribution(np.asarray(mean, dtype=float), np.asarray(cov, dtype=float))
        self.n_parts = self.gaussian.dim + 1
        self.name = name
        self.keys = keys

    @property
    def mean(self) -> np.ndarray:
        """Return the Gaussian mean in ilr coordinates."""
        return self.gaussian.mu

    @property
    def cov(self) -> np.ndarray:
        """Return the Gaussian covariance in ilr coordinates."""
        return np.asarray(self.gaussian.covar)

    def __str__(self) -> str:
        return "AitchisonNormalDistribution(%r, %r)" % (list(self.gaussian.mu), [list(r) for r in self.cov])

    def density(self, x: np.ndarray) -> float:
        """Return the density of one composition under the ilr-space Gaussian."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: np.ndarray) -> float:
        """Log-density at a single composition (the ilr-space Gaussian log-density)."""
        return float(self.gaussian.log_density(ilr(x)[0]))

    def seq_log_density(self, x) -> np.ndarray:
        """Return vectorized log-densities for ilr-encoded compositions."""
        return self.gaussian.seq_log_density(x)  # x is already ilr-encoded by the AitchisonNormal encoder

    def mean_composition(self) -> np.ndarray:
        """The center of the distribution as a composition (the ilr-mean mapped back to the simplex)."""
        return ilr_inv(self.gaussian.mu)[0]

    def sampler(self, seed: int | None = None) -> AitchisonNormalSampler:
        """Return a sampler that draws in ilr space and maps back to the simplex."""
        return AitchisonNormalSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> AitchisonNormalEstimator:
        """Return an estimator that fits a Gaussian in ilr coordinates."""
        return AitchisonNormalEstimator(name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> AitchisonNormalDataEncoder:
        """Return the ilr-transform encoder used by vectorized methods."""
        return AitchisonNormalDataEncoder(self.gaussian.dist_to_encoder())


class AitchisonNormalSampler(DistributionSampler):
    """Sample compositions by drawing Gaussian ilr coordinates and inverting the transform."""

    def __init__(self, dist: AitchisonNormalDistribution, seed: int | None = None):
        self.dist = dist
        self.gaussian_sampler = dist.gaussian.sampler(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray:
        """Draw one composition or ``size`` iid compositions on the simplex."""
        y = self.gaussian_sampler.sample(size)
        return ilr_inv(np.atleast_2d(y))[0] if size is None else ilr_inv(np.asarray(y))


class AitchisonNormalDataEncoder:
    """Encode compositions by the ilr transform, then defer to the Gaussian encoder over ilr coordinates."""

    def __init__(self, gaussian_encoder):
        self.gaussian_encoder = gaussian_encoder

    def seq_encode(self, x):
        """Encode compositions as Gaussian ilr-coordinate observations."""
        return self.gaussian_encoder.seq_encode(ilr(np.asarray(x, dtype=float)))


class AitchisonNormalEstimator(ParameterEstimator):
    """Maximum-likelihood estimator: the Gaussian MLE in ilr coordinates (delegated to MVN)."""

    def __init__(self, name: str | None = None, keys: str | None = None):
        self.gaussian_estimator = MultivariateGaussianEstimator()
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> StatisticAccumulatorFactory:
        """Return an accumulator factory that delegates to the Gaussian estimator."""
        gaussian_factory = self.gaussian_estimator.accumulator_factory()

        class _Factory(StatisticAccumulatorFactory):
            def make(self):
                """Create an Aitchison-normal accumulator."""
                return AitchisonNormalAccumulator(gaussian_factory.make())

        return _Factory()

    def estimate(self, nobs, suff_stat) -> AitchisonNormalDistribution:
        """Estimate a logratio-normal distribution from ilr-space sufficient statistics."""
        g = self.gaussian_estimator.estimate(nobs, suff_stat)
        return AitchisonNormalDistribution(g.mu, g.covar, name=self.name, keys=self.keys)


class AitchisonNormalAccumulator:
    """Wrap a Gaussian accumulator; the data arrives already ilr-encoded, so the stats are delegated."""

    def __init__(self, gaussian_acc):
        self.gaussian_acc = gaussian_acc

    def update(self, x: np.ndarray, weight: float, estimate: AitchisonNormalDistribution | None) -> None:
        """Update delegated Gaussian statistics from one composition."""
        self.gaussian_acc.update(ilr(x)[0], weight, None if estimate is None else estimate.gaussian)

    def initialize(self, x: np.ndarray, weight: float, rng) -> None:
        """Initialize delegated Gaussian statistics from one composition."""
        self.gaussian_acc.initialize(ilr(x)[0], weight, rng)

    def seq_update(self, x, weights, estimate) -> None:
        """Update delegated Gaussian statistics from ilr-encoded compositions."""
        self.gaussian_acc.seq_update(x, weights, None if estimate is None else estimate.gaussian)

    def seq_initialize(self, x, weights, rng) -> None:
        """Initialize delegated Gaussian statistics from ilr-encoded compositions."""
        self.gaussian_acc.seq_initialize(x, weights, rng)

    def combine(self, suff_stat) -> AitchisonNormalAccumulator:
        """Merge delegated Gaussian sufficient statistics."""
        self.gaussian_acc.combine(suff_stat)
        return self

    def value(self) -> Any:
        """Return delegated Gaussian sufficient statistics."""
        return self.gaussian_acc.value()

    def from_value(self, x) -> AitchisonNormalAccumulator:
        """Restore delegated Gaussian sufficient statistics."""
        self.gaussian_acc.from_value(x)
        return self

    def acc_to_encoder(self) -> AitchisonNormalDataEncoder:
        """Return the composition encoder compatible with this accumulator."""
        return AitchisonNormalDataEncoder(self.gaussian_acc.acc_to_encoder())
