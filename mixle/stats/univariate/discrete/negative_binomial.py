"""Negative binomial distributions over non-negative integer counts.

The parameterization is the number of failures X before r successes, with
success probability p:

    P(X=x) = Gamma(x+r) / (Gamma(r) Gamma(x+1)) * p**r * (1-p)**x,
    x = 0, 1, 2, ...

The dispersion ``r`` has no closed-form MLE; the estimator recovers it by a 1-D
numerical solve of the digamma score equation (with ``p`` profiled out), then ``p``
follows in closed form. Pass ``estimate_r=False`` to hold ``r`` fixed instead.


Reference: Johnson, Kemp & Kotz, *Univariate Discrete Distributions* (3rd ed., Wiley, 2005).
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.utils.special import digamma, valid_integer
from mixle.utils.vector import gammaln

_MIN_NB_SHAPE = 1.0e-8
_MAX_NB_SHAPE = 1.0e7


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
        """Describe backend support for generated negative-binomial kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for negative binomial distributions."""
        from mixle.stats.compute.declarations import (
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
            # The count histogram is needed for the dispersion (r) MLE; it is an additive,
            # linearly-scalable weighted-count map rather than a fixed-width moment.
            statistics=(StatisticSpec("count"), StatisticSpec("sum"), StatisticSpec("histogram", kind="histogram")),
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
        """Return per-row negative-binomial sufficient statistics in accumulator order.

        The accumulator's :meth:`NegativeBinomialAccumulator.value` returns
        ``(count, sum, histogram)``; the third row stat carries the per-row counts so the
        declaration's ``kind="histogram"`` reducer can fold them into the weighted count
        histogram the dispersion (``r``) solve needs.
        """
        vals = engine.asarray(x[0])
        return vals * 0.0 + engine.asarray(1.0), vals, vals

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

    def density(self, x: int) -> float:
        """Return the probability density or mass at a single observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        """Return the log-density or log-mass at a single observation."""
        if not valid_integer(x, nonneg=True):
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
        from mixle.inference.fisher import CountFisherView, _count_data

        return CountFisherView(self, _fisher_mean_var, _count_data, _fisher_encoded)

    def mean(self) -> float:
        """Mean E[X] (failures before r successes): r(1-p)/p."""
        return float(self.r * (1.0 - self.p) / self.p)

    def variance(self) -> float:
        """Variance Var[X]: r(1-p)/p^2."""
        return float(self.r * (1.0 - self.p) / (self.p * self.p))

    def entropy(self) -> float:
        """Shannon entropy in nats, by exact summation of the standard series.

        The negative-binomial entropy ``-sum_k p_k log p_k`` has no closed form (Johnson, Kemp &
        Kotz, *Univariate Discrete Distributions*, ch. 5). The series is summed over the support
        up to this distribution's own quantile at ``1 - 1e-16`` (plus a safety margin), beyond
        which the tail mass is far below double rounding -- the same effective-support truncation
        PoissonDistribution.entropy() uses, but driven by the exact CDF (via ``betainc``) rather
        than a Gaussian-tail heuristic, since the negative binomial can be heavy-tailed.
        """
        kmax = int(self.quantile(1.0 - 1.0e-16)) + 50
        k = np.arange(kmax + 1, dtype=np.float64)
        lp = gammaln(k + self.r) - self.log_gamma_r - gammaln(k + 1.0) + self.r * self.log_p + k * self.log_1p
        return float(-np.sum(np.exp(lp) * lp))

    def cdf(self, x: float) -> float:
        """Cumulative distribution function P(X <= x) = I_p(r, floor(x)+1)."""
        import math

        from scipy.special import betainc

        k = math.floor(float(x))
        return float(betainc(self.r, k + 1, self.p)) if k >= 0 else 0.0

    def quantile(self, q: float) -> float:
        """Inverse CDF F^{-1}(q) (via scipy nbinom)."""
        from scipy.stats import nbinom

        return float(nbinom.ppf(float(q), self.r, self.p))

    def sampler(self, seed: int | None = None) -> "NegativeBinomialSampler":
        """Return a sampler for drawing observations from this distribution."""
        return NegativeBinomialSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "NegativeBinomialEstimator":
        """Return an estimator for fitting this distribution from data."""
        if pseudo_count is None:
            return NegativeBinomialEstimator(r=self.r, name=self.name, keys=self.keys)
        # Conjugate-prior path: regularize p toward this distribution and hold r at its value.
        return NegativeBinomialEstimator(
            r=self.r,
            pseudo_count=pseudo_count,
            suff_stat=self.p,
            estimate_r=False,
            name=self.name,
            keys=self.keys,
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
        """Draw one sample or an array of iid samples."""
        scale = (1.0 - self.dist.p) / self.dist.p
        lam = self.rng.gamma(shape=self.dist.r, scale=scale, size=size)
        rv = self.rng.poisson(lam=lam)
        return int(rv) if size is None else rv


class NegativeBinomialAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted count and sum statistics plus the count histogram.

    The dispersion ``r`` has no finite-dimensional sufficient statistic: its MLE needs
    ``sum_i w_i digamma(x_i + r)``, which cannot be reduced to count/sum. We therefore
    accumulate the weighted histogram ``{x: weight}`` so :meth:`NegativeBinomialEstimator.estimate`
    can solve the score equation for ``r``.
    """

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.count = 0.0
        self.sum = 0.0
        self.histogram: dict[int, float] = {}
        self.name = name
        self.keys = keys

    def update(self, x: int, weight: float, estimate: NegativeBinomialDistribution | None) -> None:
        """Accumulate weighted statistics for one non-negative integer count."""
        if not valid_integer(x, nonneg=True):
            raise ValueError("NegativeBinomialDistribution requires non-negative integer observations.")
        xi = int(x)
        self.count += weight
        self.sum += float(xi) * weight
        self.histogram[xi] = self.histogram.get(xi, 0.0) + weight

    def initialize(self, x: int, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one observation."""
        self.update(x, weight, None)

    def seq_update(
        self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, estimate: NegativeBinomialDistribution | None
    ) -> None:
        """Accumulate weighted statistics from encoded observations."""
        weights = np.asarray(weights, dtype=np.float64)
        self.count += np.sum(weights, dtype=np.float64)
        self.sum += np.dot(x[0], weights)
        if x[0].size:
            vals = np.rint(x[0]).astype(np.int64)
            uniq, inv = np.unique(vals, return_inverse=True)
            wsum = np.zeros(uniq.shape[0], dtype=np.float64)
            np.add.at(wsum, inv, weights)
            for k, w in zip(uniq.tolist(), wsum.tolist()):
                self.histogram[k] = self.histogram.get(k, 0.0) + w

    def seq_initialize(self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, dict[int, float]]) -> "NegativeBinomialAccumulator":
        """Merge another negative-binomial sufficient-statistic tuple."""
        self.count += suff_stat[0]
        self.sum += suff_stat[1]
        if len(suff_stat) > 2 and suff_stat[2]:
            for k, w in suff_stat[2].items():
                self.histogram[int(k)] = self.histogram.get(int(k), 0.0) + w
        return self

    def value(self) -> tuple[float, float, dict[int, float]]:
        """Return count, sum, and histogram statistics."""
        return self.count, self.sum, self.histogram

    def from_value(self, x: tuple[float, float, dict[int, float]]) -> "NegativeBinomialAccumulator":
        """Replace accumulator contents from a sufficient-statistic tuple."""
        self.count = x[0]
        self.sum = x[1]
        self.histogram = {int(k): float(w) for k, w in x[2].items()} if len(x) > 2 and x[2] else {}
        return self

    def acc_to_encoder(self) -> "NegativeBinomialDataEncoder":
        """Return the encoder used by this accumulator."""
        return NegativeBinomialDataEncoder()


class NegativeBinomialAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for NegativeBinomialAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> NegativeBinomialAccumulator:
        """Create a fresh negative-binomial accumulator."""
        return NegativeBinomialAccumulator(name=self.name, keys=self.keys)


class NegativeBinomialEstimator(ParameterEstimator):
    """Estimate ``r`` and ``p`` for a negative binomial distribution.

    By default the dispersion ``r`` is recovered by a 1-D solve of the digamma score
    equation (``p`` profiled out), then ``p`` follows in closed form. Pass
    ``estimate_r=False`` to hold ``r`` fixed at the constructor value (the historical
    geometric/fixed-shape M-step). The ``r`` argument is the initial value / fallback
    when ``estimate_r=True``.
    """

    def __init__(
        self,
        r: float = 1.0,
        pseudo_count: float | None = None,
        suff_stat: float | None = None,
        estimate_r: bool = True,
        threshold: float = 1.0e-8,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        if r <= 0.0 or not np.isfinite(r):
            raise ValueError("NegativeBinomialEstimator requires r > 0.")
        self.r = float(r)
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.estimate_r = estimate_r
        self.threshold = threshold
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> NegativeBinomialAccumulatorFactory:
        """Return an accumulator factory for negative-binomial statistics."""
        return NegativeBinomialAccumulatorFactory(name=self.name, keys=self.keys)

    def resident_accumulation_supported(self) -> bool:
        """Dispersion estimation needs the full count histogram, not fixed-width resident stats."""
        return not self.estimate_r

    @staticmethod
    def estimate_dispersion(histogram: dict[int, float], r_init: float, threshold: float) -> float:
        """Solve the negative-binomial dispersion MLE from a weighted count histogram.

        With ``p`` profiled to its MLE ``p = r / (r + xbar)``, the score for ``r`` is

            g(r) = sum_k h(k) [digamma(k + r) - digamma(r)] - N * log(1 + xbar / r),

        which is strictly decreasing and crosses zero exactly when the data are
        over-dispersed (sample variance > mean). Under equi/under-dispersion the MLE
        runs off to the Poisson limit, so ``r`` is capped at ``_MAX_NB_SHAPE``.
        """
        if not histogram:
            return r_init
        keys = np.fromiter(histogram.keys(), dtype=np.float64, count=len(histogram))
        wts = np.fromiter(histogram.values(), dtype=np.float64, count=len(histogram))
        n = float(wts.sum())
        if n <= 0.0:
            return r_init
        xbar = float(np.dot(keys, wts) / n)
        if xbar <= 0.0:
            # All mass at zero: r is unidentified (p -> 1); keep the initial value.
            return r_init
        var = float(np.dot(keys * keys, wts) / n - xbar * xbar)
        if var <= xbar:
            # No over-dispersion -> Poisson limit, r -> infinity.
            return _MAX_NB_SHAPE

        def g(r: float) -> float:
            return float(np.dot(wts, digamma(keys + r) - digamma(r)) - n * math.log1p(xbar / r))

        lo = _MIN_NB_SHAPE
        if g(lo) <= 0.0:
            return lo
        hi = max(1.0, float(r_init))
        while g(hi) > 0.0 and hi < _MAX_NB_SHAPE:
            hi = min(_MAX_NB_SHAPE, hi * 2.0)
        if g(hi) > 0.0:
            return _MAX_NB_SHAPE

        threshold = max(float(threshold), 1.0e-12)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if g(mid) > 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo <= threshold * max(1.0, hi):
                break
        return min(_MAX_NB_SHAPE, max(_MIN_NB_SHAPE, 0.5 * (lo + hi)))

    def estimate(
        self, nobs: float | None, suff_stat: tuple[float, float, dict[int, float]]
    ) -> NegativeBinomialDistribution:
        """Estimate ``r`` and ``p`` from weighted count statistics."""
        count, xsum = suff_stat[0], suff_stat[1]
        histogram = suff_stat[2] if len(suff_stat) > 2 else None

        r = self.r
        if self.estimate_r and histogram:
            r = self.estimate_dispersion(histogram, self.r, self.threshold)

        if self.pseudo_count is not None:
            prior_p = 0.5 if self.suff_stat is None else float(self.suff_stat)
            prior_p = float(np.clip(prior_p, 1.0e-12, 1.0 - 1.0e-12))
            xsum += self.pseudo_count * r * (1.0 - prior_p) / prior_p
            count += self.pseudo_count
        p = (r * count) / (r * count + xsum) if count > 0.0 else 0.5
        p = float(np.clip(p, 1.0e-12, 1.0 - 1.0e-12))
        return NegativeBinomialDistribution(r, p, name=self.name, keys=self.keys)


class NegativeBinomialDataEncoder(DataSequenceEncoder):
    """Encode count observations with precomputed log-factorials."""

    def __str__(self) -> str:
        return "NegativeBinomialDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NegativeBinomialDataEncoder)

    def seq_encode(self, x: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Encode counts with precomputed ``log(x!)`` values."""
        rv = np.asarray(x, dtype=np.float64)
        if rv.size and (np.any(rv < 0) or np.any(np.isnan(rv)) or np.any(np.floor(rv) != rv)):
            raise ValueError("NegativeBinomialDistribution requires non-negative integer observations.")
        return rv, gammaln(rv + 1.0)
