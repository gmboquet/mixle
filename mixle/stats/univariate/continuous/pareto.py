"""Pareto type-I distributions over heavy-tailed positive values.

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


class ParetoDistribution(SequenceEncodableProbabilityDistribution):
    """Pareto type-I distribution with scale xm > 0 and shape alpha > 0."""

    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated Pareto kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for Pareto distributions."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ExponentialFamilySpec,
            ParameterSpec,
            StatisticSpec,
        )

        return DistributionDeclaration(
            name="pareto",
            distribution_type=cls,
            parameters=(
                ParameterSpec("xm", constraint="positive"),
                ParameterSpec("alpha", constraint="positive"),
            ),
            statistics=(
                StatisticSpec("count"),
                StatisticSpec("sum_of_logs"),
                StatisticSpec("min_val", kind="support_bound", additive=False, scales=False),
            ),
            support="positive_tail",
            exponential_family=ExponentialFamilySpec(
                sufficient_statistics=cls.exp_family_sufficient_statistics,
                natural_parameters=cls.exp_family_natural_parameters,
                log_partition=cls.exp_family_log_partition,
                base_measure_from_params=cls.exp_family_base_measure_from_params,
                # h(x) = 1/x on the support [xm, inf) depends on the per-component scale xm, so the
                # fixed-base stacked loop does not apply; stacked scoring uses the backend hooks while
                # the scalar canonical map / to_exponential_family view still uses the spec above.
                fixed_base=False,
            ),
        )

    @staticmethod
    def exp_family_sufficient_statistics(x: tuple[Any, Any], engine: Any) -> tuple[Any, ...]:
        """Return the Pareto sufficient statistic ``T(x) = (log x,)`` (scale ``xm`` fixed)."""
        return (engine.asarray(x[1]),)

    @staticmethod
    def exp_family_natural_parameters(params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return the Pareto natural parameter ``eta = -alpha`` (scale ``xm`` fixed)."""
        return (-engine.asarray(params["alpha"]),)

    @staticmethod
    def exp_family_log_partition(params: dict[str, Any], engine: Any) -> Any:
        """Return the Pareto log partition ``A = -log(alpha) - alpha * log(xm)``."""
        alpha = engine.asarray(params["alpha"])
        xm = engine.asarray(params["xm"])
        return -engine.log(alpha) - alpha * engine.log(xm)

    @staticmethod
    def exp_family_base_measure_from_params(x: tuple[Any, Any], params: dict[str, Any], engine: Any) -> Any:
        """Return the Pareto base measure ``log h(x) = -log(x)`` on ``[xm, inf)`` (``-inf`` below ``xm``)."""
        vals = engine.asarray(x[0])
        log_vals = engine.asarray(x[1])
        xm = engine.asarray(params["xm"])
        return engine.where(vals >= xm, -log_vals, engine.asarray(-np.inf))

    def __init__(self, xm: float, alpha: float, name: str | None = None, keys: str | None = None) -> None:
        if xm <= 0.0 or alpha <= 0.0 or not np.isfinite(xm) or not np.isfinite(alpha):
            raise ValueError("ParetoDistribution requires xm > 0 and alpha > 0.")
        self.xm = float(xm)
        self.alpha = float(alpha)
        self.log_xm = math.log(self.xm)
        self.log_alpha = math.log(self.alpha)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "ParetoDistribution(%s, %s, name=%s, keys=%s)" % (
            repr(self.xm),
            repr(self.alpha),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: float) -> float:
        """Return the probability density or mass at a single observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Return the log-density or log-mass at a single observation."""
        try:
            xx = float(x)
        except Exception:  # noqa: BLE001
            return -np.inf
        if not np.isfinite(xx) or xx < self.xm:
            return -np.inf
        return self.log_alpha + self.alpha * self.log_xm - (self.alpha + 1.0) * math.log(xx)

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        xx, lx = x
        rv = self.log_alpha + self.alpha * self.log_xm - (self.alpha + 1.0) * lx
        return np.where(xx >= self.xm, rv, -np.inf)

    @staticmethod
    def backend_log_density_from_params(vals: Any, log_vals: Any, xm: Any, alpha: Any, engine: Any) -> Any:
        """Engine-neutral Pareto log-density from explicit parameters."""
        rv = engine.log(alpha) + alpha * engine.log(xm) - (alpha + engine.asarray(1.0)) * log_vals
        return engine.where(vals >= xm, rv, engine.asarray(-np.inf))

    def backend_seq_log_density(self, x: tuple[Any, Any], engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        return self.backend_log_density_from_params(
            engine.asarray(x[0]), engine.asarray(x[1]), engine.asarray(self.xm), engine.asarray(self.alpha), engine
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["ParetoDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked Pareto parameters for a homogeneous mixture kernel."""
        return {
            "xm": engine.asarray([d.xm for d in dists]),
            "alpha": engine.asarray([d.alpha for d in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: tuple[Any, Any], params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of Pareto log densities."""
        vals = engine.asarray(x[0])
        log_vals = engine.asarray(x[1])
        return cls.backend_log_density_from_params(
            vals[:, None], log_vals[:, None], params["xm"][None, :], params["alpha"][None, :], engine
        )

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: tuple[Any, Any], weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any, Any]:
        """Return stacked Pareto sufficient statistics using engine-resident arrays."""
        vals = engine.asarray(x[0])
        log_vals = engine.asarray(x[1])
        ww = engine.asarray(weights)
        mask = ww > 0.0
        count = engine.sum(ww, axis=0)
        sum_logs = engine.sum(ww * log_vals[:, None], axis=0)
        min_val = -engine.max(engine.where(mask, -vals[:, None], engine.asarray(-np.inf)), axis=0)
        return count, sum_logs, min_val

    def cdf(self, x: float) -> float:
        """Cumulative distribution function ``P(X <= x)`` (exact). The continuous 'index of' a value."""
        from scipy.stats import pareto as _sp

        return float(_sp.cdf(x, self.alpha, scale=self.xm))

    def quantile(self, q: float) -> float:
        """Inverse CDF ``F^{-1}(q)``: the value at cumulative-probability index ``q`` (continuous unranking)."""
        from scipy.stats import pareto as _sp

        return float(_sp.ppf(q, self.alpha, scale=self.xm))

    def mean(self) -> float:
        """Mean alpha*xm/(alpha-1) for alpha > 1, else inf."""
        return float(self.alpha * self.xm / (self.alpha - 1.0)) if self.alpha > 1.0 else float("inf")

    def variance(self) -> float:
        """Variance xm^2 alpha / ((alpha-1)^2 (alpha-2)) for alpha > 2, else inf."""
        a, xm = self.alpha, self.xm
        return float(xm * xm * a / ((a - 1.0) ** 2 * (a - 2.0))) if a > 2.0 else float("inf")

    def entropy(self) -> float:
        """Differential entropy log(xm/alpha) + 1/alpha + 1."""
        import math

        return float(math.log(self.xm / self.alpha) + 1.0 / self.alpha + 1.0)

    def mode(self) -> float:
        """Mode (the scale xm)."""
        return float(self.xm)

    def sampler(self, seed: int | None = None) -> "ParetoSampler":
        """Return a sampler for drawing observations from this distribution."""
        return ParetoSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "ParetoEstimator":
        """Return an estimator for fitting this distribution from data."""
        if pseudo_count is None:
            return ParetoEstimator(name=self.name, keys=self.keys)
        return ParetoEstimator(
            pseudo_count=pseudo_count, suff_stat=(self.xm, self.alpha), name=self.name, keys=self.keys
        )

    def dist_to_encoder(self) -> "ParetoDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return ParetoDataEncoder()


class ParetoSampler(DistributionSampler):
    """Draw iid Pareto observations."""

    def __init__(self, dist: ParetoDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> float | np.ndarray:
        """Draw one sample or an array of iid samples."""
        return self.dist.xm * (self.rng.pareto(self.dist.alpha, size=size) + 1.0)


class ParetoAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted support minimum and log-sum statistics."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.count = 0.0
        self.sum_of_logs = 0.0
        self.min_val = np.inf
        self.name = name
        self.keys = keys

    def update(self, x: float, weight: float, estimate: ParetoDistribution | None) -> None:
        """Accumulate weighted log and support-minimum statistics for one sample."""
        if x <= 0.0:
            raise ValueError("ParetoDistribution requires observations x > 0.")
        if weight > 0.0:
            self.count += weight
            self.sum_of_logs += math.log(x) * weight
            self.min_val = min(self.min_val, x)

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one observation."""
        self.update(x, weight, None)

    def seq_update(
        self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, estimate: ParetoDistribution | None
    ) -> None:
        """Accumulate weighted log and support-minimum statistics from encoded data."""
        xx, lx = x
        mask = weights > 0.0
        if np.any(mask):
            self.count += np.sum(weights[mask], dtype=np.float64)
            self.sum_of_logs += np.dot(lx[mask], weights[mask])  # mask consistently with count/min_val
            self.min_val = min(self.min_val, float(np.min(xx[mask])))

    def seq_update_engine(
        self, x: tuple[np.ndarray, np.ndarray], weights: Any, estimate: ParetoDistribution | None, engine: Any
    ) -> None:
        """Engine-resident accumulation of the count and log-sum statistics (numpy or torch)."""
        xx, lx = x
        weights_np = np.asarray(engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, dtype=np.float64)
        w = engine.asarray(weights_np)
        lx_e = engine.asarray(np.asarray(lx, dtype=np.float64))
        zero = engine.asarray(0.0)
        pos = w > zero
        self.count += float(engine.to_numpy(engine.sum(engine.where(pos, w, zero))))
        self.sum_of_logs += float(engine.to_numpy(engine.sum(lx_e * w)))
        mask_np = weights_np > 0.0
        if np.any(mask_np):
            self.min_val = min(self.min_val, float(np.min(np.asarray(xx)[mask_np])))

    def seq_initialize(self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float]) -> "ParetoAccumulator":
        """Merge another Pareto sufficient-statistic tuple."""
        self.count += suff_stat[0]
        self.sum_of_logs += suff_stat[1]
        self.min_val = min(self.min_val, suff_stat[2])
        return self

    def value(self) -> tuple[float, float, float]:
        """Return count, weighted log-sum, and observed support minimum."""
        return self.count, self.sum_of_logs, self.min_val

    def from_value(self, x: tuple[float, float, float]) -> "ParetoAccumulator":
        """Replace accumulator contents from a sufficient-statistic tuple."""
        self.count = x[0]
        self.sum_of_logs = x[1]
        self.min_val = x[2]
        return self

    def scale(self, c: float) -> "ParetoAccumulator":
        """Scale linear count/log-sum statistics while preserving support bound."""
        self.count *= c
        self.sum_of_logs *= c
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

    def acc_to_encoder(self) -> "ParetoDataEncoder":
        """Return the encoder used by this accumulator."""
        return ParetoDataEncoder()


class ParetoAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for ParetoAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> ParetoAccumulator:
        """Create a fresh Pareto accumulator."""
        return ParetoAccumulator(name=self.name, keys=self.keys)


class ParetoEstimator(ParameterEstimator):
    """MLE estimator for Pareto scale and shape."""

    def __init__(
        self,
        pseudo_count: float | None = None,
        suff_stat: tuple[float, float] | None = None,
        min_denom: float = 1.0e-12,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.min_denom = min_denom
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> ParetoAccumulatorFactory:
        """Return an accumulator factory for Pareto sufficient statistics."""
        return ParetoAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> ParetoDistribution:
        """Estimate Pareto scale and shape from support and log-sum statistics."""
        count, sum_logs, xm = suff_stat
        if count <= 0.0:
            xm, alpha = self.suff_stat if self.suff_stat is not None else (1.0, 1.0)
            return ParetoDistribution(xm, alpha, name=self.name, keys=self.keys)

        if self.pseudo_count is not None and self.suff_stat is not None:
            xm0, alpha0 = self.suff_stat
            xm = min(xm, xm0)
            sum_logs += self.pseudo_count * (math.log(xm0) + 1.0 / alpha0)
            count += self.pseudo_count

        denom = max(sum_logs - count * math.log(xm), self.min_denom)
        alpha = count / denom
        return ParetoDistribution(xm, alpha, name=self.name, keys=self.keys)


class ParetoDataEncoder(DataSequenceEncoder):
    """Encode Pareto observations with x and log(x)."""

    def __str__(self) -> str:
        return "ParetoDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ParetoDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """Encode observations as values and log-values."""
        rv = np.asarray(x, dtype=np.float64)
        if rv.size and (np.any(rv <= 0.0) or np.any(np.isnan(rv))):
            raise ValueError("ParetoDistribution requires observations x > 0.")
        return rv, np.log(rv)
