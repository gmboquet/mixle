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
from scipy.special import logsumexp

from mixle.stats.compute.pdist import (
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
)
from mixle.stats.multivariate._copula_common import (
    BufferedUScoreAccumulatorFactory,
    UScoreEncoder,
    reject_unsupported_pseudo_count,
    u_score_batch,
    validated_buffered_statistic,
    validated_dimension,
    validated_finite_scalar,
    validated_sample_size,
    weighted_kendall_tau,
)

_CLIP = 1.0e-12


class ClaytonCopulaDistribution(SequenceEncodableProbabilityDistribution):
    """Clayton copula on ``(0,1)^d`` with ``theta >= 0``; zero is independence."""

    def __init__(self, dim: int, theta: float, name: str | None = None, keys: str | None = None) -> None:
        self.dim = validated_dimension(dim, label="Clayton copula dimension")
        self.theta = validated_finite_scalar(theta, label="Clayton theta")
        if self.theta < 0.0:
            raise ValueError("Clayton theta must be non-negative")
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "ClaytonCopulaDistribution(dim=%d, theta=%.6g)" % (self.dim, self.theta)

    def log_density(self, u: np.ndarray) -> float:
        return float(self.seq_log_density(np.atleast_2d(np.asarray(u, dtype=np.float64)))[0])

    def seq_log_density(self, u: np.ndarray) -> np.ndarray:
        u = u_score_batch(u, self.dim)
        th = self.theta
        if th == 0.0:
            return np.zeros(len(u), dtype=np.float64)
        const = float(np.sum(np.log1p(np.arange(1, self.dim) * th)))  # sum_{k=1}^{d-1} log(1 + k theta)
        log_prod = -(1.0 + th) * np.sum(np.log(u), axis=1)
        powers = -th * np.log(u)
        log_s = np.empty(len(u), dtype=np.float64)
        direct = np.max(powers, axis=1) < 500.0
        log_s[direct] = np.log1p(np.sum(np.expm1(powers[direct]), axis=1))
        if np.any(~direct):
            log_power_sum = logsumexp(powers[~direct], axis=1)
            correction = (self.dim - 1) * np.exp(-log_power_sum)
            log_s[~direct] = log_power_sum + np.log1p(-correction)
        result = const + log_prod - (self.dim + 1.0 / th) * log_s
        if np.any(np.isnan(result)):
            raise FloatingPointError("Clayton log-density was numerically indeterminate")
        return result

    def sampler(self, seed: int | None = None) -> ClaytonCopulaSampler:
        return ClaytonCopulaSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> ClaytonCopulaEstimator:
        reject_unsupported_pseudo_count(pseudo_count, family="Clayton copula")
        return ClaytonCopulaEstimator(self.dim, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> UScoreEncoder:
        return UScoreEncoder(self.dim)


class ClaytonCopulaSampler(DistributionSampler):
    """Marshall-Olkin frailty: draw ``V ~ Gamma(1/theta, 1)``, ``E_i ~ Exp(1)``, ``u_i = (1 + E_i/V)^{-1/theta}``."""

    def __init__(self, dist: ClaytonCopulaDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray:
        n = 1 if size is None else validated_sample_size(size)
        th = self.dist.theta
        if th == 0.0:
            u = self.rng.uniform(_CLIP, 1.0 - _CLIP, size=(n, self.dist.dim))
        else:
            v = self.rng.gamma(shape=1.0 / th, scale=1.0, size=(n, 1))
            e = self.rng.exponential(scale=1.0, size=(n, self.dist.dim))
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                u = (1.0 + e / v) ** (-1.0 / th)
            u = np.clip(u, _CLIP, 1.0 - _CLIP)
        return u[0] if size is None else u


class ClaytonCopulaEstimator(ParameterEstimator):
    """Kendall's-tau inversion: ``theta = 2*tau / (1 - tau)`` (pair-averaged for ``d > 2``)."""

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        self.dim = validated_dimension(dim, label="Clayton copula dimension")
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> BufferedUScoreAccumulatorFactory:
        return BufferedUScoreAccumulatorFactory(self.dim, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray]) -> ClaytonCopulaDistribution:
        u, w = validated_buffered_statistic(
            suff_stat, self.dim, minimum_rows=2, require_positive_weight=True
        )
        taus = [weighted_kendall_tau(u[:, i], u[:, j], w) for i in range(self.dim) for j in range(i + 1, self.dim)]
        tau = max(float(np.mean(taus)), 0.0)
        if tau >= 1.0 - 1.0e-12:
            raise ValueError("Clayton fit is on the comonotonic boundary and has no finite theta estimate")
        theta = 2.0 * tau / (1.0 - tau)
        return ClaytonCopulaDistribution(self.dim, theta, name=self.name, keys=self.keys)
