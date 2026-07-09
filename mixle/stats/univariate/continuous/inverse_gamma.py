"""Inverse-gamma distributions over positive real values.

Observations are floats ``x > 0``. An inverse-gamma distribution with shape ``alpha > 0`` and
rate ``beta > 0`` has log-density

        log(f(x; alpha, beta)) = alpha*log(beta) - lgamma(alpha) - (alpha + 1)*log(x) - beta / x,

for x > 0 (it is the law of 1/Y when Y ~ Gamma(alpha, 1/beta)). It is widely used as the conjugate
prior for the variance of a Gaussian, so this class also exposes ``get_parameters`` / ``cross_entropy``
/ ``entropy`` for the Bayesian (conjugate-prior) estimation path, in addition to being a standalone
positive-support leaf.

It is a two-parameter exponential family with sufficient statistics ``(log x, 1/x)``; once the scalar
normalizer ``alpha*log(beta) - lgamma(alpha)`` is precomputed the per-row score is linear in the encoded
``log x`` and ``1/x`` fields, so the family gets generated NumPy, Torch, and Numba kernels.


Reference: Johnson, Kotz & Balakrishnan, *Continuous Univariate Distributions* (2nd ed., Wiley, 1994/95).
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.utils.special import digamma, gammaln, trigamma

_MIN_PARAM = 1.0e-12
_MAX_SHAPE = 1.0e12


class InverseGammaDistribution(SequenceEncodableProbabilityDistribution):
    """Inverse-gamma distribution with shape alpha > 0 and rate beta > 0 on x > 0."""

    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated inverse-gamma kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for inverse-gamma distributions."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ExponentialFamilySpec,
            ParameterSpec,
            StatisticSpec,
        )

        return DistributionDeclaration(
            name="inverse_gamma",
            distribution_type=cls,
            parameters=(
                ParameterSpec("alpha", constraint="positive"),
                ParameterSpec("beta", constraint="positive"),
                ParameterSpec("log_const", constraint="real", differentiable=False),
            ),
            statistics=(StatisticSpec("count"), StatisticSpec("sum_inv"), StatisticSpec("sum_neg_log")),
            support="positive_real",
            exponential_family=ExponentialFamilySpec(
                sufficient_statistics=cls.exp_family_sufficient_statistics,
                natural_parameters=cls.exp_family_natural_parameters,
                log_partition=cls.exp_family_log_partition,
                legacy_sufficient_statistics=cls.backend_legacy_sufficient_statistics,
            ),
        )

    @staticmethod
    def backend_legacy_sufficient_statistics(
        x: tuple[Any, Any], params: dict[str, Any], engine: Any
    ) -> tuple[Any, ...]:
        """Return per-row (count, 1/x, -log x) sufficient statistics in accumulator order."""
        log_x = engine.asarray(x[0])
        inv_x = engine.asarray(x[1])
        return inv_x * 0.0 + engine.asarray(1.0), inv_x, -log_x

    @staticmethod
    def exp_family_sufficient_statistics(x: tuple[Any, Any], engine: Any) -> tuple[Any, ...]:
        """Return inverse-gamma sufficient statistics ``T(x) = (log x, 1/x)``."""
        return engine.asarray(x[0]), engine.asarray(x[1])

    @staticmethod
    def exp_family_natural_parameters(params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return inverse-gamma natural parameters ``eta = (-(alpha + 1), -beta)``."""
        return -(params["alpha"] + engine.asarray(1.0)), -params["beta"]

    @staticmethod
    def exp_family_log_partition(params: dict[str, Any], engine: Any) -> Any:
        """Return inverse-gamma log partition ``A = lgamma(alpha) - alpha * log(beta)``."""
        alpha = params["alpha"]
        return engine.gammaln(alpha) - alpha * engine.log(params["beta"])

    @staticmethod
    def exp_family_from_natural(eta: Any) -> "InverseGammaDistribution":
        """Return the inverse-gamma with natural parameters ``eta = (-(alpha + 1), -beta)``."""
        alpha = -float(eta[0]) - 1.0
        beta = -float(eta[1])
        return InverseGammaDistribution(alpha, beta)

    @staticmethod
    def backend_log_density_from_params(
        log_x: Any, inv_x: Any, alpha: Any, beta: Any, log_const: Any, engine: Any
    ) -> Any:
        """Engine-neutral inverse-gamma log-density from explicit parameters (linear in log x and 1/x)."""
        return log_const - (alpha + engine.asarray(1.0)) * log_x - beta * inv_x

    def __init__(self, alpha: float, beta: float, name: str | None = None, keys: str | None = None) -> None:
        """InverseGammaDistribution for shape alpha and rate beta.

        Args:
            alpha (float): Positive shape parameter.
            beta (float): Positive rate (scale) parameter.
            name (Optional[str]): Assign a name to InverseGammaDistribution instance.
            keys (Optional[str]): Assign keys for merging sufficient statistics.

        Attributes:
            alpha (float): Shape parameter.
            beta (float): Rate parameter.
            log_const (float): Cached alpha*log(beta) - lgamma(alpha).

        """
        if alpha <= 0.0 or not np.isfinite(alpha):
            raise ValueError("InverseGammaDistribution requires alpha > 0.")
        if beta <= 0.0 or not np.isfinite(beta):
            raise ValueError("InverseGammaDistribution requires beta > 0.")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.log_const = self.alpha * math.log(self.beta) - gammaln(self.alpha)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return a constructor-style representation of the inverse-gamma distribution."""
        return "InverseGammaDistribution(%s, %s, name=%s, keys=%s)" % (
            repr(self.alpha),
            repr(self.beta),
            repr(self.name),
            repr(self.keys),
        )

    def get_parameters(self) -> tuple[float, float]:
        """Return the (shape alpha, rate beta) pair (so this can serve as a conjugate prior)."""
        return self.alpha, self.beta

    def cross_entropy(self, dist: "InverseGammaDistribution") -> float:
        """Cross entropy -E_self[log dist(x)] for an inverse-gamma argument (closed form)."""
        if isinstance(dist, InverseGammaDistribution):
            ao, bo = dist.alpha, dist.beta
            # E_self[log x] = log(beta) - digamma(alpha); E_self[1/x] = alpha / beta.
            e_log_x = math.log(self.beta) - digamma(self.alpha)
            e_inv_x = self.alpha / self.beta
            return float(-(ao * math.log(bo) - gammaln(ao) - (ao + 1.0) * e_log_x - bo * e_inv_x))
        raise NotImplementedError(
            "InverseGammaDistribution.cross_entropy is only implemented for inverse-gamma arguments (got %s)."
            % type(dist).__name__
        )

    def entropy(self) -> float:
        """Returns the differential entropy in nats."""
        return float(self.alpha + math.log(self.beta) + gammaln(self.alpha) - (1.0 + self.alpha) * digamma(self.alpha))

    def density(self, x: float) -> float:
        """Return the probability density at a single observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Return the log-density at a single observation (or -inf off support)."""
        try:
            xx = float(x)
        except (TypeError, ValueError):
            return -np.inf
        if not np.isfinite(xx) or xx <= 0.0:
            return -np.inf
        return self.log_const - (self.alpha + 1.0) * math.log(xx) - self.beta / xx

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded (log x, 1/x) observations."""
        log_x, inv_x = x
        return self.log_const - (self.alpha + 1.0) * log_x - self.beta * inv_x

    def backend_seq_log_density(self, x: tuple[Any, Any], engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        return self.backend_log_density_from_params(
            engine.asarray(x[0]),
            engine.asarray(x[1]),
            engine.asarray(self.alpha),
            engine.asarray(self.beta),
            engine.asarray(self.log_const),
            engine,
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["InverseGammaDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked parameters for a homogeneous mixture kernel."""
        return {
            "alpha": engine.asarray([d.alpha for d in dists]),
            "beta": engine.asarray([d.beta for d in dists]),
            "log_const": engine.asarray([d.log_const for d in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: tuple[Any, Any], params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of inverse-gamma log densities."""
        log_x = engine.asarray(x[0])[:, None]
        inv_x = engine.asarray(x[1])[:, None]
        return cls.backend_log_density_from_params(
            log_x, inv_x, params["alpha"][None, :], params["beta"][None, :], params["log_const"][None, :], engine
        )

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: tuple[Any, Any], weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any, Any]:
        """Return stacked sufficient statistics using engine-resident arrays."""
        log_x = engine.asarray(x[0])
        inv_x = engine.asarray(x[1])
        ww = engine.asarray(weights)
        return (
            engine.sum(ww, axis=0),
            engine.sum(ww * inv_x[:, None], axis=0),
            engine.sum(ww * (-log_x)[:, None], axis=0),
        )

    def cdf(self, x: float) -> float:
        """Cumulative distribution function P(X <= x) = Q(alpha, beta/x) (0 for x <= 0)."""
        from scipy.special import gammaincc

        x = float(x)
        return float(gammaincc(self.alpha, self.beta / x)) if x > 0.0 else 0.0

    def quantile(self, q: float) -> float:
        """Inverse CDF F^{-1}(q)."""
        from scipy.special import gammainccinv

        return float(self.beta / gammainccinv(self.alpha, float(q)))

    def sampler(self, seed: int | None = None) -> "InverseGammaSampler":
        """Return a sampler for drawing observations from this distribution."""
        return InverseGammaSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "InverseGammaEstimator":
        """Return an estimator for fitting this distribution from data."""
        if pseudo_count is None:
            return InverseGammaEstimator(name=self.name, keys=self.keys)
        # E[1/x] = alpha/beta, E[log(1/x)] = digamma(alpha) - log(beta).
        suff_stat = (self.alpha / self.beta, digamma(self.alpha) - math.log(self.beta))
        return InverseGammaEstimator(pseudo_count=pseudo_count, suff_stat=suff_stat, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "InverseGammaDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return InverseGammaDataEncoder()


class InverseGammaSampler(DistributionSampler):
    """Draw iid inverse-gamma observations as 1 / Gamma(alpha, 1/beta)."""

    def __init__(self, dist: InverseGammaDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist
        self.seed = seed

    def sample(self, size: int | None = None) -> float | np.ndarray:
        """Draw ``size`` iid observations (a float when ``size`` is None)."""
        return 1.0 / self.rng.gamma(shape=self.dist.alpha, scale=1.0 / self.dist.beta, size=size)


class InverseGammaAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted count, sum of reciprocals, and sum of negative logs for inverse-gamma estimation."""

    def __init__(self, keys: str | None = None) -> None:
        self.count = 0.0
        self.sum_inv = 0.0
        self.sum_neg_log = 0.0
        self.keys = keys

    def update(self, x: float, weight: float, estimate: InverseGammaDistribution | None) -> None:
        """Accumulate reciprocal and negative-log statistics for one observation."""
        if x <= 0.0 or not np.isfinite(x):
            raise ValueError("InverseGammaDistribution has support x > 0.")
        self.count += weight
        self.sum_inv += weight / x
        self.sum_neg_log += -math.log(x) * weight

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one observation."""
        self.update(x, weight, None)

    def seq_update(
        self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, estimate: InverseGammaDistribution | None
    ) -> None:
        """Accumulate transformed sufficient statistics from encoded data."""
        log_x, inv_x = x
        self.count += np.sum(weights, dtype=np.float64)
        self.sum_inv += np.dot(inv_x, weights)
        self.sum_neg_log += np.dot(-log_x, weights)

    def seq_initialize(self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float]) -> "InverseGammaAccumulator":
        """Merge another inverse-gamma sufficient-statistic tuple."""
        self.count += suff_stat[0]
        self.sum_inv += suff_stat[1]
        self.sum_neg_log += suff_stat[2]
        return self

    def value(self) -> tuple[float, float, float]:
        """Return count, reciprocal sum, and negative-log sum."""
        return self.count, self.sum_inv, self.sum_neg_log

    def from_value(self, x: tuple[float, float, float]) -> "InverseGammaAccumulator":
        """Replace accumulator contents from a sufficient-statistic tuple."""
        self.count, self.sum_inv, self.sum_neg_log = x
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

    def acc_to_encoder(self) -> "InverseGammaDataEncoder":
        """Return the encoder used by this accumulator."""
        return InverseGammaDataEncoder()


class InverseGammaAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for InverseGammaAccumulator."""

    def __init__(self, keys: str | None = None) -> None:
        self.keys = keys

    def make(self) -> InverseGammaAccumulator:
        """Create a fresh inverse-gamma accumulator."""
        return InverseGammaAccumulator(keys=self.keys)


class InverseGammaEstimator(ParameterEstimator):
    """Maximum-likelihood estimator for the inverse-gamma shape and rate.

    Fits the gamma distribution to the reciprocals ``y = 1/x`` (``y ~ Gamma(alpha, 1/beta)``) by the
    standard log-mean / mean shape equation, then maps back to ``(alpha, beta) = (k, 1/theta)``.
    """

    def __init__(
        self,
        pseudo_count: float | None = None,
        suff_stat: tuple[float, float] | None = None,
        threshold: float = 1.0e-8,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.threshold = threshold
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> InverseGammaAccumulatorFactory:
        """Return an accumulator factory for inverse-gamma statistics."""
        return InverseGammaAccumulatorFactory(keys=self.keys)

    @staticmethod
    def _estimate_shape(mean_y: float, mean_log_y: float, threshold: float) -> float:
        s = math.log(mean_y) - mean_log_y
        if not np.isfinite(s) or s <= 0.0:
            return _MAX_SHAPE if (np.isfinite(s) and s <= 0.0) else 1.0
        k = (3.0 - s + math.sqrt((s - 3.0) ** 2 + 24.0 * s)) / (12.0 * s)  # Minka initializer
        for _ in range(100):
            if k <= 0.0:
                return _MIN_PARAM
            g = math.log(k) - digamma(k) - s
            gp = 1.0 / k - float(trigamma(k))
            step = g / gp
            k_new = k - step
            if k_new <= 0.0:
                k_new = k / 2.0
            if abs(k_new - k) <= threshold * max(1.0, k):
                k = k_new
                break
            k = k_new
        return float(min(max(k, _MIN_PARAM), _MAX_SHAPE))

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> InverseGammaDistribution:
        """Estimate shape and rate from reciprocal and log-reciprocal statistics."""
        count, sum_inv, sum_neg_log = suff_stat
        if self.pseudo_count is not None and self.suff_stat is not None:
            inv0, neglog0 = self.suff_stat
            sum_inv += self.pseudo_count * inv0
            sum_neg_log += self.pseudo_count * neglog0
            count += self.pseudo_count
        if count <= 0.0 or sum_inv <= 0.0 or not np.isfinite(sum_inv):
            return InverseGammaDistribution(1.0, 1.0, name=self.name, keys=self.keys)

        mean_y = sum_inv / count  # E[1/x]
        mean_log_y = sum_neg_log / count  # E[log(1/x)]
        alpha = self._estimate_shape(mean_y, mean_log_y, self.threshold)
        beta = max(alpha / mean_y, _MIN_PARAM)  # theta_gamma = mean_y / alpha; beta = 1/theta
        return InverseGammaDistribution(alpha, beta, name=self.name, keys=self.keys)


class InverseGammaDataEncoder(DataSequenceEncoder):
    """Encode inverse-gamma observations as (log x, 1/x) pairs."""

    def __str__(self) -> str:
        return "InverseGammaDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, InverseGammaDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """Encode observations as log-values and reciprocal values."""
        rv = np.asarray(x, dtype=np.float64)
        if rv.size and (np.any(rv <= 0.0) or np.any(~np.isfinite(rv))):
            raise ValueError("InverseGammaDistribution has support x > 0.")
        with np.errstate(divide="ignore"):
            return np.log(rv), 1.0 / rv
