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
from scipy.special import logsumexp

from mixle.stats.compute.pdist import (
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
)
from mixle.stats.multivariate._copula_common import (
    BufferedUScoreAccumulatorFactory,
    UScoreEncoder,
    maximize_1d,
    reject_unsupported_pseudo_count,
    u_score_batch,
    validated_buffered_statistic,
    validated_dimension,
    validated_finite_scalar,
    validated_sample_size,
)

_CLIP = 1.0e-12
_MIN_ABS_THETA = 1.0e-4  # |theta| below this is treated as independence (the density -> uniform)


class FrankCopulaDistribution(SequenceEncodableProbabilityDistribution):
    """Frank copula on ``(0,1)^2`` with dependence parameter ``theta`` (any sign; 0 = independence)."""

    def __init__(self, dim: int, theta: float, name: str | None = None, keys: str | None = None) -> None:
        checked_dim = validated_dimension(dim, label="Frank copula dimension")
        if checked_dim != 2:
            raise ValueError("FrankCopulaDistribution is bivariate (dim == 2); got dim=%d" % checked_dim)
        self.dim = 2
        self.theta = validated_finite_scalar(theta, label="Frank theta")
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "FrankCopulaDistribution(theta=%.6g)" % self.theta

    def log_density(self, u: np.ndarray) -> float:
        return float(self.seq_log_density(np.atleast_2d(np.asarray(u, dtype=np.float64)))[0])

    def seq_log_density(self, u: np.ndarray) -> np.ndarray:
        u = u_score_batch(u, self.dim)
        th = self.theta
        if abs(th) < _MIN_ABS_THETA:
            return np.zeros(u.shape[0])  # independence copula: c(u, v) = 1
        q = abs(th)
        a = u[:, 0]
        b = 1.0 - u[:, 1] if th < 0.0 else u[:, 1]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        terms = np.column_stack(
            [
                np.zeros(len(u)),
                -q * (hi - lo),
                -q * hi,
                -q * (1.0 - lo),
            ]
        )
        log_bracket, sign = logsumexp(
            terms,
            b=np.asarray([1.0, 1.0, -1.0, -1.0]),
            axis=1,
            return_sign=True,
        )
        if np.any(sign <= 0.0):
            raise FloatingPointError("Frank log-density denominator was numerically indeterminate")
        result = np.log(q) + np.log(-np.expm1(-q)) - q * (hi - lo) - 2.0 * log_bracket
        if np.any(np.isnan(result)):
            raise FloatingPointError("Frank log-density was numerically indeterminate")
        return result

    def sampler(self, seed: int | None = None) -> FrankCopulaSampler:
        return FrankCopulaSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> FrankCopulaEstimator:
        reject_unsupported_pseudo_count(pseudo_count, family="Frank copula")
        return FrankCopulaEstimator(name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> UScoreEncoder:
        return UScoreEncoder(self.dim)


class FrankCopulaSampler(DistributionSampler):
    """Conditional inversion: draw ``u1``, ``w`` uniform, solve ``v`` from the conditional ``C(v | u1) = w``."""

    def __init__(self, dist: FrankCopulaDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray:
        n = 1 if size is None else validated_sample_size(size)
        th = self.dist.theta
        u1 = self.rng.uniform(_CLIP, 1.0 - _CLIP, size=n)
        w = self.rng.uniform(_CLIP, 1.0 - _CLIP, size=n)
        if abs(th) < _MIN_ABS_THETA:
            u2 = w  # independence
        else:
            q = abs(th)
            log_w = np.log(w)
            log_eu = -q * u1
            log_numerator = np.logaddexp(log_eu + np.log1p(-w), -q + log_w)
            log_denominator = np.logaddexp(
                log_w + np.log1p(-np.exp(log_eu)),
                log_eu,
            )
            u2 = np.clip((log_denominator - log_numerator) / q, _CLIP, 1.0 - _CLIP)
            if th < 0.0:
                u2 = 1.0 - u2
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
        u, w = validated_buffered_statistic(suff_stat, 2, minimum_rows=2, require_positive_weight=True)

        def loglik(theta: float) -> float:
            return float(np.dot(w, FrankCopulaDistribution(2, theta).seq_log_density(u)))

        theta = maximize_1d(loglik, -40.0, 40.0)
        if abs(theta) < _MIN_ABS_THETA:
            theta = 0.0
        return FrankCopulaDistribution(2, theta, name=self.name, keys=self.keys)
