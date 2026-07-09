"""Gumbel copula: an Archimedean copula with UPPER-tail dependence, on ``(0,1)^2`` -- the complement to Clayton.

Clayton concentrates dependence in the lower tail (joint crashes); Gumbel is its mirror, concentrating it in
the UPPER tail (joint booms / simultaneous extremes on the high side, the tail that matters for insurance
maxima, flood peaks, or a portfolio's joint upside). It is Archimedean with generator ``phi(t) = (-log t)^theta``,
``theta >= 1``: ``theta = 1`` is independence, ``theta -> inf`` is comonotonicity. The bivariate density is

    C(u, v) = exp(-A),  A = (x^theta + y^theta)^{1/theta},  x = -log u,  y = -log v,
    c(u, v) = C(u, v) / (u v) * (x y)^{theta - 1} * (x^theta + y^theta)^{2/theta - 2} * (A + theta - 1).

Its upper-tail dependence coefficient is ``lambda_U = 2 - 2^{1/theta}`` (Clayton's is lower-tail). Fit by
Kendall's-tau inversion ``tau = 1 - 1/theta`` so ``theta = 1 / (1 - tau)``; sampled by the positive-stable
frailty construction (a totally-skewed stable mixing variable). Bivariate only, matching
:class:`~mixle.stats.multivariate.frank_copula.FrankCopulaDistribution` -- and exactly the pair-copula shape a
vine consumes.

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


class GumbelCopulaDistribution(SequenceEncodableProbabilityDistribution):
    """Gumbel copula on ``(0,1)^2`` with upper-tail dependence parameter ``theta >= 1``."""

    def __init__(self, dim: int, theta: float, name: str | None = None, keys: str | None = None) -> None:
        if int(dim) != 2:
            raise ValueError("GumbelCopulaDistribution is bivariate (dim == 2); got dim=%d" % dim)
        self.dim = 2
        self.theta = max(float(theta), 1.0)  # theta = 1 is independence; theta < 1 is not a valid Gumbel
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "GumbelCopulaDistribution(theta=%.6g)" % self.theta

    def log_density(self, u: np.ndarray) -> float:
        return float(self.seq_log_density(np.atleast_2d(np.asarray(u, dtype=np.float64)))[0])

    def seq_log_density(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=np.float64), _CLIP, 1.0 - _CLIP)
        th = self.theta
        if th <= 1.0 + 1.0e-12:
            return np.zeros(u.shape[0])  # independence copula
        x = -np.log(u[:, 0])
        y = -np.log(u[:, 1])
        sx = x**th + y**th
        a = sx ** (1.0 / th)  # A = (x^th + y^th)^{1/th}
        # log c = -A + (th-1)(log x + log y) + (1/th - 2) log(x^th+y^th) + log(A + th - 1) - log(u v)
        log_c = (
            -a
            + (th - 1.0) * (np.log(x) + np.log(y))
            + (1.0 / th - 2.0) * np.log(sx)
            + np.log(a + th - 1.0)
            - np.log(u[:, 0])
            - np.log(u[:, 1])
        )
        return log_c

    def sampler(self, seed: int | None = None) -> GumbelCopulaSampler:
        return GumbelCopulaSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> GumbelCopulaEstimator:
        return GumbelCopulaEstimator(name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> UScoreEncoder:
        return UScoreEncoder()


class GumbelCopulaSampler(DistributionSampler):
    """Positive-stable frailty: draw ``M ~ Stable(1/theta)``, ``E_i ~ Exp(1)``, ``u_i = exp(-(E_i / M)^{1/theta})``."""

    def __init__(self, dist: GumbelCopulaDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def _positive_stable(self, alpha: float, n: int) -> np.ndarray:
        """A totally-skewed positive stable with Laplace transform ``exp(-s^alpha)`` (Chambers-Mallows-Stuck)."""
        if alpha >= 1.0:  # alpha = 1 (theta = 1) is the degenerate point mass at 1 -> independence
            return np.ones(n)
        theta_u = self.rng.uniform(0.0, np.pi, size=n)
        w = self.rng.exponential(1.0, size=n)
        a = np.sin(alpha * theta_u) / (np.sin(theta_u) ** (1.0 / alpha))
        b = (np.sin((1.0 - alpha) * theta_u) / w) ** ((1.0 - alpha) / alpha)
        return a * b

    def sample(self, size: int | None = None) -> np.ndarray:
        n = 1 if size is None else int(size)
        th = self.dist.theta
        m = self._positive_stable(1.0 / th, n).reshape(n, 1)
        e = self.rng.exponential(1.0, size=(n, 2))
        u = np.exp(-((e / m) ** (1.0 / th)))
        u = np.clip(u, _CLIP, 1.0 - _CLIP)
        return u[0] if size is None else u


class GumbelCopulaEstimator(ParameterEstimator):
    """Kendall's-tau inversion: ``theta = 1 / (1 - tau)`` (clamped to ``theta >= 1``)."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> BufferedUScoreAccumulatorFactory:
        return BufferedUScoreAccumulatorFactory(2, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray]) -> GumbelCopulaDistribution:
        u, w = suff_stat
        if len(u) < 2:
            return GumbelCopulaDistribution(2, 1.0, name=self.name, keys=self.keys)
        tau = float(np.clip(weighted_kendall_tau(u[:, 0], u[:, 1], w), 0.0, 0.999))
        theta = 1.0 / (1.0 - tau)  # tau <= 0 -> theta = 1 (independence); tau -> 1 -> huge theta
        return GumbelCopulaDistribution(2, theta, name=self.name, keys=self.keys)
