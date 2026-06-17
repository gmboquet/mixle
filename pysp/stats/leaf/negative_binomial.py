"""Create, estimate, enumerate, and sample from a negative binomial distribution.

The parameterization is the number of failures X before r successes, with
success probability p:

    P(X=x) = Gamma(x+r) / (Gamma(r) Gamma(x+1)) * p**r * (1-p)**x,
    x = 0, 1, 2, ...

The shape r is treated as fixed by the estimator; p has a closed-form M-step.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from pysp.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from pysp.utils.vector import gammaln


def _fisher_mean_var(dist):
    r = float(dist.r)
    p = float(dist.p)
    return r * (1.0 - p) / p, r * (1.0 - p) / (p * p)


def _fisher_encoded(enc_data):
    if isinstance(enc_data, tuple):
        return np.asarray(enc_data[0], dtype=np.float64)
    return np.asarray(enc_data, dtype=np.float64)


class NegativeBinomialDistribution(SequenceEncodableProbabilityDistribution):
    """Negative binomial distribution over non-negative integer counts."""

    @classmethod
    def compute_capabilities(cls):
        from pysp.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        from pysp.stats.compute.declarations import (
            DistributionDeclaration,
            ExponentialFamilySpec,
            ParameterSpec,
            StatisticSpec,
        )

        return DistributionDeclaration(
            name="negative_binomial",
            distribution_type=cls,
            parameters=(
                ParameterSpec("r", constraint="positive"),
                ParameterSpec("p", constraint="unit_interval"),
            ),
            statistics=(StatisticSpec("count"), StatisticSpec("sum")),
            support="non_negative_integer",
            exponential_family=ExponentialFamilySpec(
                sufficient_statistics=cls.exp_family_sufficient_statistics,
                natural_parameters=cls.exp_family_natural_parameters,
                log_partition=cls.exp_family_log_partition,
                base_measure_from_params=cls.exp_family_base_measure_from_params,
                legacy_sufficient_statistics=cls.backend_legacy_sufficient_statistics,
                # h(x) = lgamma(x+r) - lgamma(r) - log(x!) depends on the per-component shape r,
                # so the fixed-base stacked loop does not apply; stacked scoring uses the backend
                # hooks below while the scalar canonical map still uses the spec above.
                fixed_base=False,
            ),
        )

    @staticmethod
    def backend_legacy_sufficient_statistics(
        x: tuple[Any, Any], params: dict[str, Any], engine: Any
    ) -> tuple[Any, ...]:
        """Return per-row negative-binomial sufficient statistics in accumulator order."""
        vals = engine.asarray(x[0])
        return vals * 0.0 + engine.asarray(1.0), vals

    @staticmethod
    def exp_family_sufficient_statistics(x: tuple[Any, Any], engine: Any) -> tuple[Any, ...]:
        """Return the NegativeBinomial sufficient statistic ``T(x) = (x,)`` (``r`` fixed)."""
        return (engine.asarray(x[0]),)

    @staticmethod
    def exp_family_natural_parameters(params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return the NegativeBinomial natural parameter ``eta = log(1 - p)`` (``r`` fixed)."""
        return (engine.log(engine.asarray(1.0) - params["p"]),)

    @staticmethod
    def exp_family_log_partition(params: dict[str, Any], engine: Any) -> Any:
        """Return the NegativeBinomial log partition ``A = -r * log(p)`` (``r`` fixed)."""
        return -params["r"] * engine.log(params["p"])

    @staticmethod
    def exp_family_base_measure_from_params(x: tuple[Any, Any], params: dict[str, Any], engine: Any) -> Any:
        """Return the NegativeBinomial base measure ``log h(x) = lgamma(x+r) - lgamma(r) - log(x!)``.

        The base measure carries the binomial-coefficient term and depends on the fixed
        shape ``r``; invalid (non-integer / negative) counts are mapped to ``-inf``.
        """
        vals = engine.asarray(x[0])
        log_fact = engine.asarray(x[1])
        r = engine.asarray(params["r"])
        log_h = engine.gammaln(vals + r) - engine.gammaln(r) - log_fact
        good = (vals >= 0.0) & (engine.floor(vals) == vals)
        return engine.where(good, log_h, engine.asarray(-np.inf))

    def __init__(self, r: float, p: float, name: str | None = None, keys: str | None = None) -> None:
        if r <= 0.0 or not np.isfinite(r):
            raise ValueError("NegativeBinomialDistribution requires r > 0.")
        if p <= 0.0 or p >= 1.0:
            raise ValueError("NegativeBinomialDistribution requires p in (0, 1).")
        self.r = float(r)
        self.p = float(p)
        self.log_p = math.log(self.p)
        self.log_1p = math.log1p(-self.p)
        self.log_gamma_r = float(gammaln(self.r))
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "NegativeBinomialDistribution(%s, %s, name=%s, keys=%s)" % (
            repr(self.r),
            repr(self.p),
            repr(self.name),
            repr(self.keys),
        )

    @staticmethod
    def _valid_count(x: Any) -> bool:
        try:
            xx = float(x)
        except Exception:
            return False
        return np.isfinite(xx) and xx >= 0.0 and math.floor(xx) == xx

    def density(self, x: int) -> float:
        """Return the probability density or mass at a single observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        """Return the log-density or log-mass at a single observation."""
        if not self._valid_count(x):
            return -np.inf
        xx = float(x)
        return (
            float(gammaln(xx + self.r))
            - self.log_gamma_r
            - float(gammaln(xx + 1.0))
            + self.r * self.log_p
            + xx * self.log_1p
        )

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        xx, lgx1 = x
        return gammaln(xx + self.r) - self.log_gamma_r - lgx1 + self.r * self.log_p + xx * self.log_1p

    @staticmethod
    def backend_log_density_from_params(vals: Any, log_fact: Any, r: Any, p: Any, engine: Any) -> Any:
        """Engine-neutral negative-binomial log-density from explicit parameters."""
        rv = (
            engine.gammaln(vals + r)
            - engine.gammaln(r)
            - log_fact
            + r * engine.log(p)
            + vals * engine.log(engine.asarray(1.0) - p)
        )
        good = (vals >= 0.0) & (engine.floor(vals) == vals)
        return engine.where(good, rv, engine.asarray(-np.inf))

    def backend_seq_log_density(self, x: tuple[Any, Any], engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        vals = engine.asarray(x[0])
        log_fact = engine.asarray(x[1])
        return self.backend_log_density_from_params(
            vals, log_fact, engine.asarray(self.r), engine.asarray(self.p), engine
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["NegativeBinomialDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked negative-binomial parameters for a homogeneous mixture kernel."""
        return {
            "r": engine.asarray([d.r for d in dists]),
            "p": engine.asarray([d.p for d in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: tuple[Any, Any], params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of negative-binomial log densities."""
        vals = engine.asarray(x[0])
        log_fact = engine.asarray(x[1])
        return cls.backend_log_density_from_params(
            vals[:, None], log_fact[:, None], params["r"][None, :], params["p"][None, :], engine
        )

    def to_fisher(self, **kwargs):
        """Return the NegativeBinomial's count-family Fisher view."""
        from pysp.utils.fisher import CountFisherView, _count_data

        return CountFisherView(self, _fisher_mean_var, _count_data, _fisher_encoded)

    def sampler(self, seed: int | None = None) -> "NegativeBinomialSampler":
        """Return a sampler for drawing observations from this distribution."""
        return NegativeBinomialSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "NegativeBinomialEstimator":
        """Return an estimator for fitting this distribution from data."""
        if pseudo_count is None:
            return NegativeBinomialEstimator(r=self.r, name=self.name, keys=self.keys)
        return NegativeBinomialEstimator(
            r=self.r, pseudo_count=pseudo_count, suff_stat=self.p, name=self.name, keys=self.keys
        )

    def dist_to_encoder(self) -> "NegativeBinomialDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return NegativeBinomialDataEncoder()

    def enumerator(self) -> "NegativeBinomialEnumerator":
        """Return an enumerator over the distribution support when available."""
        return NegativeBinomialEnumerator(self)


class NegativeBinomialEnumerator(DistributionEnumerator):
    """Enumerate the infinite support in descending probability order."""

    def __init__(self, dist: NegativeBinomialDistribution) -> None:
        super().__init__(dist)
        mode = math.floor((dist.r - 1.0) * (1.0 - dist.p) / dist.p) if dist.r > 1.0 else 0
        self._mode = int(max(0, mode))
        self._left = self._mode - 1
        self._right = self._mode + 1
        self._started = False

    def __next__(self) -> tuple[int, float]:
        if not self._started:
            self._started = True
            return self._mode, self.dist.log_density(self._mode)
        lp_l = self.dist.log_density(self._left) if self._left >= 0 else -np.inf
        lp_r = self.dist.log_density(self._right)
        if lp_l >= lp_r:
            x, lp = self._left, lp_l
            self._left -= 1
        else:
            x, lp = self._right, lp_r
            self._right += 1
        return x, lp


class NegativeBinomialSampler(DistributionSampler):
    """Draw iid negative binomial observations."""

    def __init__(self, dist: NegativeBinomialDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> int | np.ndarray:
        scale = (1.0 - self.dist.p) / self.dist.p
        lam = self.rng.gamma(shape=self.dist.r, scale=scale, size=size)
        rv = self.rng.poisson(lam=lam)
        return int(rv) if size is None else rv


class NegativeBinomialAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted count and sum statistics."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.count = 0.0
        self.sum = 0.0
        self.name = name
        self.key = keys

    def update(self, x: int, weight: float, estimate: NegativeBinomialDistribution | None) -> None:
        if not NegativeBinomialDistribution._valid_count(x):
            raise ValueError("NegativeBinomialDistribution requires non-negative integer observations.")
        self.count += weight
        self.sum += float(x) * weight

    def initialize(self, x: int, weight: float, rng: RandomState | None) -> None:
        self.update(x, weight, None)

    def seq_update(
        self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, estimate: NegativeBinomialDistribution | None
    ) -> None:
        self.count += np.sum(weights, dtype=np.float64)
        self.sum += np.dot(x[0], weights)

    def seq_initialize(self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float]) -> "NegativeBinomialAccumulator":
        self.count += suff_stat[0]
        self.sum += suff_stat[1]
        return self

    def value(self) -> tuple[float, float]:
        return self.count, self.sum

    def from_value(self, x: tuple[float, float]) -> "NegativeBinomialAccumulator":
        self.count = x[0]
        self.sum = x[1]
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        if self.key is not None and self.key in stats_dict:
            self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> "NegativeBinomialDataEncoder":
        return NegativeBinomialDataEncoder()


class NegativeBinomialAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for NegativeBinomialAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> NegativeBinomialAccumulator:
        return NegativeBinomialAccumulator(name=self.name, keys=self.keys)


class NegativeBinomialEstimator(ParameterEstimator):
    """Estimate p for a negative binomial distribution with fixed r."""

    def __init__(
        self,
        r: float = 1.0,
        pseudo_count: float | None = None,
        suff_stat: float | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        if r <= 0.0 or not np.isfinite(r):
            raise ValueError("NegativeBinomialEstimator requires r > 0.")
        self.r = float(r)
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> NegativeBinomialAccumulatorFactory:
        return NegativeBinomialAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float]) -> NegativeBinomialDistribution:
        count, xsum = suff_stat
        if self.pseudo_count is not None:
            prior_p = 0.5 if self.suff_stat is None else float(self.suff_stat)
            prior_p = float(np.clip(prior_p, 1.0e-12, 1.0 - 1.0e-12))
            xsum += self.pseudo_count * self.r * (1.0 - prior_p) / prior_p
            count += self.pseudo_count
        p = (self.r * count) / (self.r * count + xsum) if count > 0.0 else 0.5
        p = float(np.clip(p, 1.0e-12, 1.0 - 1.0e-12))
        return NegativeBinomialDistribution(self.r, p, name=self.name, keys=self.keys)


class NegativeBinomialDataEncoder(DataSequenceEncoder):
    """Encode count observations with precomputed log-factorials."""

    def __str__(self) -> str:
        return "NegativeBinomialDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NegativeBinomialDataEncoder)

    def seq_encode(self, x: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        rv = np.asarray(x, dtype=np.float64)
        if rv.size and (np.any(rv < 0) or np.any(np.isnan(rv)) or np.any(np.floor(rv) != rv)):
            raise ValueError("NegativeBinomialDistribution requires non-negative integer observations.")
        return rv, gammaln(rv + 1.0)
