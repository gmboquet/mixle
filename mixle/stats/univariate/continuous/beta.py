"""Beta distributions over values in the unit interval.

Reference: Johnson, Kotz & Balakrishnan, *Continuous Univariate Distributions* (2nd ed., Wiley, 1994/95).
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.inference.fisher import FixedFisherView
from mixle.stats.bayes.dirichlet import dirichlet_param_solve
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.utils.special import digamma, gammaln, trigamma


class BetaFisherView(FixedFisherView):
    """Fisher view for Beta sufficient statistics."""

    def __init__(self, dist: Any) -> None:
        super().__init__(dist, [("count",), ("log_x",), ("log1m_x",)])

    @staticmethod
    def _matrix_from_values(x: Any, log_x: Any | None = None, log1m_x: Any | None = None) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float64).reshape(-1)
        lx = np.log(xx) if log_x is None else np.asarray(log_x, dtype=np.float64).reshape(-1)
        l1 = np.log1p(-xx) if log1m_x is None else np.asarray(log1m_x, dtype=np.float64).reshape(-1)
        return np.column_stack((np.ones_like(xx, dtype=np.float64), lx, l1))

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        return self._matrix_from_values(data)

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        if isinstance(enc_data, tuple):
            x = enc_data[2] if len(enc_data) > 2 else None
            if x is None:
                x = np.exp(enc_data[0])
            return self._matrix_from_values(x, enc_data[0], enc_data[1])
        return self._matrix_from_values(enc_data)

    def _model_mean(self) -> np.ndarray:
        a = float(self.dist.a)
        b = float(self.dist.b)
        ab = a + b
        return np.asarray([1.0, digamma(a) - digamma(ab), digamma(b) - digamma(ab)], dtype=np.float64)

    def _model_fisher(self) -> np.ndarray:
        a = float(self.dist.a)
        b = float(self.dist.b)
        tab = trigamma(a + b)
        out = np.zeros((3, 3), dtype=np.float64)
        out[1, 1] = trigamma(a) - tab
        out[1, 2] = -tab
        out[2, 1] = -tab
        out[2, 2] = trigamma(b) - tab
        return out


class BetaDistribution(SequenceEncodableProbabilityDistribution):
    """Beta distribution with positive shape parameters a and b."""

    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated Beta kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for Beta distributions."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ExponentialFamilySpec,
            ParameterSpec,
            StatisticSpec,
        )

        return DistributionDeclaration(
            name="beta",
            distribution_type=cls,
            parameters=(
                ParameterSpec("a", constraint="positive"),
                ParameterSpec("b", constraint="positive"),
            ),
            statistics=(
                StatisticSpec("count"),
                StatisticSpec("sum_of_logs"),
                StatisticSpec("sum_of_log1m"),
                StatisticSpec("sum"),
                StatisticSpec("sum2"),
            ),
            support="unit_interval_open",
            exponential_family=ExponentialFamilySpec(
                sufficient_statistics=cls.exp_family_sufficient_statistics,
                natural_parameters=cls.exp_family_natural_parameters,
                log_partition=cls.exp_family_log_partition,
                legacy_sufficient_statistics=cls.exp_family_legacy_sufficient_statistics,
            ),
        )

    @staticmethod
    def exp_family_sufficient_statistics(x: tuple[Any, Any, Any, Any], engine: Any) -> tuple[Any, ...]:
        """Return Beta sufficient statistics for generated scoring.

        Scoring needs only ``log_x`` and ``log1m_x`` (the two natural-parameter partners). The
        encoder emits the full ``(log_x, log1m_x, x, x**2)`` tuple for the M-step, while the
        symbolic generator supplies just the two scoring symbols from
        ``backend_log_density_from_params``; indexing the leading two works for both arities.
        """
        return engine.asarray(x[0]), engine.asarray(x[1])

    @staticmethod
    def exp_family_legacy_sufficient_statistics(
        x: tuple[Any, Any, Any, Any], params: dict[str, Any], engine: Any
    ) -> tuple[Any, ...]:
        """Return per-row Beta sufficient statistics in accumulator order."""
        log_x, log1m_x, xx, xx2 = x
        vals = engine.asarray(xx)
        return (
            vals * 0.0 + engine.asarray(1.0),
            engine.asarray(log_x),
            engine.asarray(log1m_x),
            vals,
            engine.asarray(xx2),
        )

    @staticmethod
    def exp_family_natural_parameters(params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return Beta natural parameters for generated scoring."""
        return params["a"] - engine.asarray(1.0), params["b"] - engine.asarray(1.0)

    @staticmethod
    def exp_family_log_partition(params: dict[str, Any], engine: Any) -> Any:
        """Return Beta log partition for generated scoring."""
        return engine.betaln(params["a"], params["b"])

    def __init__(self, a: float, b: float, name: str | None = None, keys: str | None = None) -> None:
        if a <= 0.0 or b <= 0.0 or not np.isfinite(a) or not np.isfinite(b):
            raise ValueError("BetaDistribution requires a > 0 and b > 0.")
        self.a = float(a)
        self.b = float(b)
        self.log_const = float(gammaln(self.a) + gammaln(self.b) - gammaln(self.a + self.b))
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "BetaDistribution(%s, %s, name=%s, keys=%s)" % (
            repr(self.a),
            repr(self.b),
            repr(self.name),
            repr(self.keys),
        )

    def get_parameters(self) -> tuple[float, float]:
        """Return the (a, b) shape pair.

        Lets a BetaDistribution serve as a conjugate prior (on a Bernoulli/Geometric/Binomial
        success probability) under the unified Bayesian estimation protocol.
        """
        return self.a, self.b

    def density(self, x: float) -> float:
        """Return the probability density or mass at a single observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Return the log-density or log-mass at a single observation."""
        try:
            xx = float(x)
        except Exception:
            return -np.inf
        if not np.isfinite(xx) or xx <= 0.0 or xx >= 1.0:
            return -np.inf
        return (self.a - 1.0) * math.log(xx) + (self.b - 1.0) * math.log1p(-xx) - self.log_const

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        lx, l1mx, _, _ = x
        return (self.a - 1.0) * lx + (self.b - 1.0) * l1mx - self.log_const

    @staticmethod
    def backend_log_density_from_params(log_x: Any, log1m_x: Any, a: Any, b: Any, engine: Any) -> Any:
        """Engine-neutral Beta log-density from encoded logs and parameters."""
        return (a - 1.0) * log_x + (b - 1.0) * log1m_x - engine.betaln(a, b)

    def backend_seq_log_density(self, x: tuple[Any, Any, Any, Any], engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        log_x, log1m_x, _, _ = x
        return self.backend_log_density_from_params(
            engine.asarray(log_x), engine.asarray(log1m_x), engine.asarray(self.a), engine.asarray(self.b), engine
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["BetaDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked Beta parameters for a homogeneous mixture kernel."""
        return {
            "a": engine.asarray([d.a for d in dists]),
            "b": engine.asarray([d.b for d in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: tuple[Any, Any, Any, Any], params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of Beta log densities."""
        log_x, log1m_x, _, _ = x
        return cls.backend_log_density_from_params(
            engine.asarray(log_x)[:, None],
            engine.asarray(log1m_x)[:, None],
            params["a"][None, :],
            params["b"][None, :],
            engine,
        )

    def cdf(self, x: float) -> float:
        """Cumulative distribution function ``P(X <= x)`` (exact). The continuous 'index of' a value."""
        from scipy.stats import beta as _sp

        return float(_sp.cdf(x, self.a, self.b))

    def quantile(self, q: float) -> float:
        """Inverse CDF ``F^{-1}(q)``: the value at cumulative-probability index ``q`` (continuous unranking)."""
        from scipy.stats import beta as _sp

        return float(_sp.ppf(q, self.a, self.b))

    def to_fisher(self, **kwargs):
        """Return this distribution's own Fisher view."""
        return BetaFisherView(self)

    def mean(self) -> float:
        """Mean E[X] of the distribution."""
        return float(self.a / (self.a + self.b))

    def variance(self) -> float:
        """Variance Var[X] of the distribution."""
        return float(self.a * self.b / ((self.a + self.b) ** 2 * (self.a + self.b + 1.0)))

    def entropy(self) -> float:
        """Differential entropy ln B(a,b) - (a-1)psi(a) - (b-1)psi(b) + (a+b-2)psi(a+b)."""
        from scipy.special import betaln, digamma

        a, b = self.a, self.b
        return float(betaln(a, b) - (a - 1.0) * digamma(a) - (b - 1.0) * digamma(b) + (a + b - 2.0) * digamma(a + b))

    def mode(self) -> float:
        """Mode (a-1)/(a+b-2) for a,b>1; boundary otherwise."""
        a, b = self.a, self.b
        if a > 1.0 and b > 1.0:
            return float((a - 1.0) / (a + b - 2.0))
        if a <= 1.0 < b:
            return 0.0
        if b <= 1.0 < a:
            return 1.0
        return 0.0

    def sampler(self, seed: int | None = None) -> "BetaSampler":
        """Return a sampler for drawing observations from this distribution."""
        return BetaSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "BetaEstimator":
        """Return an estimator for fitting this distribution from data."""
        if pseudo_count is None:
            return BetaEstimator(name=self.name, keys=self.keys)
        suff_stat = np.asarray(
            [
                digamma(self.a) - digamma(self.a + self.b),
                digamma(self.b) - digamma(self.a + self.b),
            ],
            dtype=np.float64,
        )
        return BetaEstimator(pseudo_count=pseudo_count, suff_stat=suff_stat, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "BetaDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return BetaDataEncoder()


class BetaSampler(DistributionSampler):
    """Draw iid beta observations."""

    def __init__(self, dist: BetaDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> float | np.ndarray:
        """Draw one sample or an array of iid samples."""
        return self.rng.beta(self.dist.a, self.dist.b, size=size)


class BetaAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate sufficient statistics for beta estimation."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.count = 0.0
        self.sum_of_logs = 0.0
        self.sum_of_log1m = 0.0
        self.sum = 0.0
        self.sum2 = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: float, weight: float, estimate: BetaDistribution | None) -> None:
        """Accumulate weighted statistics for one observation in ``(0, 1)``."""
        if x <= 0.0 or x >= 1.0:
            raise ValueError("BetaDistribution requires observations in (0, 1).")
        self.count += weight
        self.sum_of_logs += math.log(x) * weight
        self.sum_of_log1m += math.log1p(-x) * weight
        self.sum += x * weight
        self.sum2 += x * x * weight

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one observation."""
        self.update(x, weight, None)

    def seq_update(
        self,
        x: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        weights: np.ndarray,
        estimate: BetaDistribution | None,
    ) -> None:
        """Accumulate weighted statistics from encoded observations."""
        lx, l1mx, xx, xx2 = x
        self.count += np.sum(weights, dtype=np.float64)
        self.sum_of_logs += np.dot(lx, weights)
        self.sum_of_log1m += np.dot(l1mx, weights)
        self.sum += np.dot(xx, weights)
        self.sum2 += np.dot(xx2, weights)

    def seq_initialize(
        self, x: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState | None
    ) -> None:
        """Initialize statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float, float, float]) -> "BetaAccumulator":
        """Merge another Beta sufficient-statistic tuple."""
        self.count += suff_stat[0]
        self.sum_of_logs += suff_stat[1]
        self.sum_of_log1m += suff_stat[2]
        self.sum += suff_stat[3]
        self.sum2 += suff_stat[4]
        return self

    def value(self) -> tuple[float, float, float, float, float]:
        """Return the accumulated Beta sufficient statistics."""
        return self.count, self.sum_of_logs, self.sum_of_log1m, self.sum, self.sum2

    def from_value(self, x: tuple[float, float, float, float, float]) -> "BetaAccumulator":
        """Replace accumulator contents from a sufficient-statistic tuple."""
        self.count = x[0]
        self.sum_of_logs = x[1]
        self.sum_of_log1m = x[2]
        self.sum = x[3]
        self.sum2 = x[4]
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

    def acc_to_encoder(self) -> "BetaDataEncoder":
        """Return the encoder used by this accumulator."""
        return BetaDataEncoder()


class BetaAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for BetaAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> BetaAccumulator:
        """Create a fresh Beta accumulator."""
        return BetaAccumulator(name=self.name, keys=self.keys)


class BetaEstimator(ParameterEstimator):
    """Estimate beta shape parameters from weighted log-moment statistics."""

    def __init__(
        self,
        pseudo_count: float | None = None,
        suff_stat: Sequence[float] | None = None,
        delta: float = 1.0e-8,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.pseudo_count = pseudo_count
        self.suff_stat = np.asarray(suff_stat, dtype=np.float64) if suff_stat is not None else None
        self.delta = delta
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> BetaAccumulatorFactory:
        """Return an accumulator factory for Beta sufficient statistics."""
        return BetaAccumulatorFactory(name=self.name, keys=self.keys)

    @staticmethod
    def _moment_initial(count: float, sum_x: float, sum_x2: float) -> np.ndarray:
        if count <= 0.0:
            return np.array([1.0, 1.0], dtype=np.float64)
        mean = float(np.clip(sum_x / count, 1.0e-6, 1.0 - 1.0e-6))
        var = max(sum_x2 / count - mean * mean, 0.0)
        max_var = mean * (1.0 - mean)
        if var > 0.0 and var < max_var:
            common = max_var / var - 1.0
            if np.isfinite(common) and common > 0.0:
                return np.array([mean * common, (1.0 - mean) * common], dtype=np.float64)
        return np.array([1.0, 1.0], dtype=np.float64)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float, float, float]) -> BetaDistribution:
        """Estimate Beta shape parameters from weighted sufficient statistics."""
        count, sum_log_x, sum_log1m, sum_x, sum_x2 = suff_stat
        if count <= 0.0 and self.pseudo_count is None:
            return BetaDistribution(1.0, 1.0, name=self.name, keys=self.keys)

        logs = np.asarray([sum_log_x, sum_log1m], dtype=np.float64)
        denom = count
        if self.pseudo_count is not None:
            prior_logs = self.suff_stat
            if prior_logs is None:
                prior_logs = np.asarray([digamma(1.0) - digamma(2.0), digamma(1.0) - digamma(2.0)], dtype=np.float64)
            logs = logs + self.pseudo_count * prior_logs
            denom = denom + self.pseudo_count

        mean_logs = logs / denom
        initial = self._moment_initial(count, sum_x, sum_x2)
        alpha, _ = dirichlet_param_solve(initial, mean_logs, self.delta)
        alpha = np.maximum(alpha, 1.0e-12)
        return BetaDistribution(float(alpha[0]), float(alpha[1]), name=self.name, keys=self.keys)


class BetaDataEncoder(DataSequenceEncoder):
    """Encode beta observations with log x and log(1-x) columns."""

    def __str__(self) -> str:
        return "BetaDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BetaDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Encode observations as log and moment arrays for vectorized updates."""
        rv = np.asarray(x, dtype=np.float64)
        if rv.size and (np.any(rv <= 0.0) or np.any(rv >= 1.0) or np.any(np.isnan(rv))):
            raise ValueError("BetaDistribution requires observations in (0, 1).")
        return np.log(rv), np.log1p(-rv), rv, rv * rv
