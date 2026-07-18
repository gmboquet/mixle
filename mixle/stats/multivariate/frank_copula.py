"""Frank copula: a symmetric Archimedean copula with NO tail dependence, on ``(0,1)^2``.

The Frank copula is the symmetric Archimedean member: unlike Clayton (lower-tail) it is radially symmetric and
has no tail dependence, and unlike the Gaussian it can be fit by a single interpretable parameter. Crucially it
spans the FULL dependence range -- ``theta > 0`` is positive dependence, ``theta < 0`` NEGATIVE dependence,
``theta -> 0`` independence -- so it is the natural core when the coupling may be either sign. Bivariate density

    c(u, v) = theta * (1 - e^{-theta}) * e^{-theta (u + v)} / [ (1 - e^{-theta}) - (1 - e^{-theta u})(1 - e^{-theta v}) ]^2 .

Fit by 1-D maximum likelihood on ``theta`` (its Kendall's-tau relation involves the Debye function, so direct
MLE on the copula likelihood is both simpler and exact). Sampled by conditional inversion. Bivariate only:
the general-``d`` Frank density is not a clean closed form, so ``d != 2`` is rejected rather than approximated.

Reference: Nelsen, *An Introduction to Copulas* (2nd ed., Springer, 2006), example 4.5.1.
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
    maximize_1d,
)

_CLIP = 1.0e-12
_MIN_ABS_THETA = 1.0e-4  # |theta| below this is treated as independence (the density -> uniform)


class FrankCopulaDistribution(SequenceEncodableProbabilityDistribution):
    """Frank copula on ``(0,1)^2`` with dependence parameter ``theta`` (any sign; 0 = independence)."""

    def __init__(self, dim: int, theta: float, name: str | None = None, keys: str | None = None) -> None:
        if int(dim) != 2:
            raise ValueError("FrankCopulaDistribution is bivariate (dim == 2); got dim=%d" % dim)
        self.dim = 2
        self.theta = float(theta)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "FrankCopulaDistribution(theta=%.6g)" % self.theta

    def log_density(self, u: np.ndarray) -> float:
        return float(self.seq_log_density(np.atleast_2d(np.asarray(u, dtype=np.float64)))[0])

    def seq_log_density(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=np.float64), _CLIP, 1.0 - _CLIP)
        th = self.theta
        if abs(th) < _MIN_ABS_THETA:
            return np.zeros(u.shape[0])  # independence copula: c(u, v) = 1
        a, b = u[:, 0], u[:, 1]
        h1 = 1.0 - np.exp(-th)
        denom = h1 - (1.0 - np.exp(-th * a)) * (1.0 - np.exp(-th * b))
        return np.log(abs(th)) + np.log(abs(h1)) - th * (a + b) - 2.0 * np.log(np.abs(denom))

    def sampler(self, seed: int | None = None) -> FrankCopulaSampler:
        return FrankCopulaSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> FrankCopulaEstimator:
        return FrankCopulaEstimator(name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> UScoreEncoder:
        return UScoreEncoder()


class FrankCopulaSampler(DistributionSampler):
    """Conditional inversion: draw ``u1``, ``w`` uniform, solve ``v`` from the conditional ``C(v | u1) = w``."""

    def __init__(self, dist: FrankCopulaDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray:
        n = 1 if size is None else int(size)
        th = self.dist.theta
        u1 = self.rng.uniform(_CLIP, 1.0 - _CLIP, size=n)
        w = self.rng.uniform(_CLIP, 1.0 - _CLIP, size=n)
        if abs(th) < _MIN_ABS_THETA:
            u2 = w  # independence
        else:
            eu = np.exp(-th * u1)
            v = -np.log(1.0 + w * (1.0 - np.exp(-th)) / (w * (eu - 1.0) - eu)) / th
            u2 = np.clip(v, _CLIP, 1.0 - _CLIP)
        out = np.column_stack([u1, u2])
        return out[0] if size is None else out


class FrankCopulaEstimator(ParameterEstimator):
    """1-D maximum likelihood on ``theta`` over ``[-40, 40]`` (golden section on the copula log-likelihood)."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> BufferedUScoreAccumulatorFactory:
        return BufferedUScoreAccumulatorFactory(2, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray]) -> FrankCopulaDistribution:
        u, w = suff_stat
        if len(u) < 2:
            return FrankCopulaDistribution(2, 0.0, name=self.name, keys=self.keys)
        w = np.asarray(w, dtype=np.float64)

        def loglik(theta: float) -> float:
            return float(np.dot(w, FrankCopulaDistribution(2, theta).seq_log_density(u)))

        theta = maximize_1d(loglik, -40.0, 40.0)
        if abs(theta) < _MIN_ABS_THETA:
            theta = 0.0
        return FrankCopulaDistribution(2, theta, name=self.name, keys=self.keys)
