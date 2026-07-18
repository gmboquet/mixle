"""Clayton copula: an Archimedean copula with lower-tail dependence, on ``(0,1)^d``.

Where the Gaussian copula is symmetric and tail-independent, the Clayton copula concentrates dependence in
the LOWER tail: joint small values (a market crash where everything drops together) are far more likely than
its correlation alone would suggest. It is Archimedean with generator ``phi(t) = (t^{-theta} - 1)/theta``,
``theta > 0``; ``theta -> 0`` is independence, larger ``theta`` is stronger lower-tail dependence. The
``d``-dimensional density is

    c(u) = [prod_{k=1}^{d-1} (1 + k*theta)] * (prod_i u_i)^{-(1+theta)} * S^{-(d + 1/theta)},
    S = sum_i u_i^{-theta} - (d - 1),

exchangeable (one parameter for every pair). Fit by Kendall's-tau inversion: ``tau = theta / (theta + 2)`` so
``theta = 2*tau / (1 - tau)`` (averaged over pairs in ``d > 2``). Sampled by the Marshall-Olkin frailty
construction (a Gamma mixing variable shared across coordinates).

Reference: Nelsen, *An Introduction to Copulas* (2nd ed., Springer, 2006), ch. 4.
"""

from __future__ import annotations

import numpy as np
from numpy.random import RandomState

from mixle.stats.compute.pdist import (
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
)
from mixle.stats.multivariate._copula_common import (
    BufferedUScoreAccumulatorFactory,
    UScoreEncoder,
    weighted_kendall_tau,
)

_CLIP = 1.0e-12


class ClaytonCopulaDistribution(SequenceEncodableProbabilityDistribution):
    """Clayton copula on ``(0,1)^d`` with lower-tail dependence parameter ``theta > 0``."""

    def __init__(self, dim: int, theta: float, name: str | None = None, keys: str | None = None) -> None:
        if int(dim) < 2:
            raise ValueError("ClaytonCopulaDistribution needs dim >= 2; got %d" % dim)
        self.dim = int(dim)
        self.theta = max(float(theta), 1.0e-6)  # theta -> 0 is independence; keep strictly positive
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "ClaytonCopulaDistribution(dim=%d, theta=%.6g)" % (self.dim, self.theta)

    def log_density(self, u: np.ndarray) -> float:
        return float(self.seq_log_density(np.atleast_2d(np.asarray(u, dtype=np.float64)))[0])

    def seq_log_density(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=np.float64), _CLIP, 1.0 - _CLIP)
        th = self.theta
        const = float(np.sum(np.log1p(np.arange(1, self.dim) * th)))  # sum_{k=1}^{d-1} log(1 + k theta)
        log_prod = -(1.0 + th) * np.sum(np.log(u), axis=1)
        s = np.sum(u ** (-th), axis=1) - (self.dim - 1)
        return const + log_prod - (self.dim + 1.0 / th) * np.log(s)

    def sampler(self, seed: int | None = None) -> ClaytonCopulaSampler:
        return ClaytonCopulaSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> ClaytonCopulaEstimator:
        return ClaytonCopulaEstimator(self.dim, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> UScoreEncoder:
        return UScoreEncoder()


class ClaytonCopulaSampler(DistributionSampler):
    """Marshall-Olkin frailty: draw ``V ~ Gamma(1/theta, 1)``, ``E_i ~ Exp(1)``, ``u_i = (1 + E_i/V)^{-1/theta}``."""

    def __init__(self, dist: ClaytonCopulaDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray:
        n = 1 if size is None else int(size)
        th = self.dist.theta
        v = self.rng.gamma(shape=1.0 / th, scale=1.0, size=(n, 1))
        e = self.rng.exponential(scale=1.0, size=(n, self.dist.dim))
        u = (1.0 + e / v) ** (-1.0 / th)
        u = np.clip(u, _CLIP, 1.0 - _CLIP)
        return u[0] if size is None else u


class ClaytonCopulaEstimator(ParameterEstimator):
    """Kendall's-tau inversion: ``theta = 2*tau / (1 - tau)`` (pair-averaged for ``d > 2``)."""

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        self.dim = dim
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> BufferedUScoreAccumulatorFactory:
        return BufferedUScoreAccumulatorFactory(self.dim, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray]) -> ClaytonCopulaDistribution:
        u, w = suff_stat
        if len(u) < 2:
            return ClaytonCopulaDistribution(self.dim, 1.0e-6, name=self.name, keys=self.keys)
        taus = [weighted_kendall_tau(u[:, i], u[:, j], w) for i in range(self.dim) for j in range(i + 1, self.dim)]
        tau = float(np.clip(np.mean(taus), 1.0e-6, 0.999))  # tau<=0 -> near-independence; tau->1 -> huge theta
        theta = 2.0 * tau / (1.0 - tau)
        return ClaytonCopulaDistribution(self.dim, theta, name=self.name, keys=self.keys)
