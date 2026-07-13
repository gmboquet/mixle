"""Beta-binomial distribution -- an overdispersed binomial (Beta-binomial compound).

If the success probability of a binomial is itself Beta(a, b) distributed and integrated out, the
count ``k`` in ``n`` trials follows the beta-binomial:

    P(k; n, a, b) = C(n, k) B(k + a, n - k + b) / B(a, b),    k = 0, ..., n,

with mean ``n a/(a+b)`` and a variance inflated over the binomial by the intra-class correlation
``rho = 1/(a+b+1)``. It is the standard model for overdispersed bounded counts (clustered trials,
batch defect rates). The number of trials ``n`` is a fixed, known parameter; ``a`` and ``b`` are
estimated by moments -- ``rho`` from the dispersion and the mean fixing ``a/(a+b)``.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import betaln, gammaln

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class BetaBinomialDistribution(SequenceEncodableProbabilityDistribution):
    """Beta-binomial distribution over ``{0, ..., n}`` with shape parameters ``a, b > 0``."""

    def __init__(self, n: int, a: float, b: float, name: str | None = None, keys: str | None = None) -> None:
        if int(n) < 0 or a <= 0.0 or b <= 0.0 or not (np.isfinite(a) and np.isfinite(b)):
            raise ValueError("BetaBinomialDistribution requires n >= 0 and a, b > 0.")
        self.n = int(n)
        self.a = float(a)
        self.b = float(b)
        self.name = name
        self.keys = keys
        self._log_beta_ab = betaln(self.a, self.b)

    def __str__(self) -> str:
        return "BetaBinomialDistribution(%s, %s, %s, name=%s, keys=%s)" % (
            repr(self.n),
            repr(self.a),
            repr(self.b),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: int) -> float:
        """Return the probability mass at a single count ``x``."""
        return math.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        """Return the log probability mass at ``x`` (``-inf`` outside ``{0, ..., n}``)."""
        k = int(x)
        if k < 0 or k > self.n:
            return -np.inf
        log_choose = gammaln(self.n + 1) - gammaln(k + 1) - gammaln(self.n - k + 1)
        return float(log_choose + betaln(k + self.a, self.n - k + self.b) - self._log_beta_ab)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-mass for an array of counts."""
        k = np.asarray(x, dtype=np.float64)
        log_choose = gammaln(self.n + 1) - gammaln(k + 1) - gammaln(self.n - k + 1)
        rv = log_choose + betaln(k + self.a, self.n - k + self.b) - self._log_beta_ab
        return np.where((k < 0) | (k > self.n), -np.inf, rv)

    def support_size(self) -> int:
        """``n + 1`` outcomes ``{0, ..., n}``."""
        return int(self.n) + 1

    def _log_cdf_table(self) -> np.ndarray:
        """Log cumulative masses ``log P(X <= k)`` for k = 0..n by stable log-space accumulation."""
        return np.logaddexp.accumulate(self.seq_log_density(np.arange(self.n + 1, dtype=np.float64)))

    def cdf(self, x: float) -> float:
        """Cumulative distribution function P(X <= x): a log-space partial sum of the pmf over {0..n}."""
        k = math.floor(float(x))
        if k < 0:
            return 0.0
        if k >= self.n:
            return 1.0
        return float(min(np.exp(self._log_cdf_table()[k]), 1.0))

    def quantile(self, q: float) -> float:
        """Inverse CDF F^{-1}(q) over {0..n}: the smallest count whose cdf reaches ``q`` (search on the cdf)."""
        qq = float(q)
        if qq <= 0.0:
            return 0.0
        cum = np.exp(self._log_cdf_table())
        return float(min(int(np.searchsorted(cum, qq, side="left")), self.n))

    def enumerator(self) -> "BetaBinomialEnumerator":
        """Returns BetaBinomialEnumerator iterating the support in descending probability order."""
        return BetaBinomialEnumerator(self)

    def sampler(self, seed: int | None = None) -> "BetaBinomialSampler":
        """Return a sampler for drawing counts from this distribution."""
        return BetaBinomialSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "BetaBinomialEstimator":
        """Return a method-of-moments estimator for ``a, b`` at the fixed number of trials ``n``."""
        return BetaBinomialEstimator(self.n, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "BetaBinomialDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return BetaBinomialDataEncoder()


class BetaBinomialEnumerator(DistributionEnumerator):
    """Enumerate the bounded beta-binomial support {0, ..., n} in descending probability order.

    Unlike the binomial, the beta-binomial pmf is not always unimodal (it is U-shaped for
    ``a < 1, b < 1``), so a two-pointer walk from a mode is not safe here; the finite support is
    materialized and sorted instead (the CategoricalEnumerator pattern), with value-order tie-breaks.
    """

    def __init__(self, dist: BetaBinomialDistribution) -> None:
        super().__init__(dist)
        lp = dist.seq_log_density(np.arange(dist.n + 1, dtype=np.float64))
        order = np.argsort(-lp, kind="stable")
        self._entries = [(int(k), float(lp[k])) for k in order if np.isfinite(lp[k])]
        self._pos = 0

    def __next__(self) -> tuple[int, float]:
        if self._pos >= len(self._entries):
            raise StopIteration
        rv = self._entries[self._pos]
        self._pos += 1
        return rv


class BetaBinomialSampler(DistributionSampler):
    """Draw counts as ``p ~ Beta(a, b)`` then ``k ~ Binomial(n, p)``."""

    def __init__(self, dist: BetaBinomialDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> int | np.ndarray:
        """Draw one count or an array of iid counts."""
        d = self.dist
        p = self.rng.beta(d.a, d.b, size=size)
        return self.rng.binomial(d.n, p)


class BetaBinomialAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted first and second moments for beta-binomial estimation."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.sum = 0.0
        self.sum2 = 0.0
        self.count = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: int, weight: float, estimate: BetaBinomialDistribution | None) -> None:
        """Accumulate weighted first and second moments for one count."""
        self.sum += weight * x
        self.sum2 += weight * x * x
        self.count += weight

    def initialize(self, x: int, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one count."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: BetaBinomialDistribution | None) -> None:
        """Accumulate weighted moments from encoded counts."""
        xx = np.asarray(x, dtype=np.float64)
        self.sum += float(np.dot(xx, weights))
        self.sum2 += float(np.dot(xx * xx, weights))
        self.count += float(np.sum(weights))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded counts."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float]) -> "BetaBinomialAccumulator":
        """Merge another beta-binomial sufficient-statistic tuple."""
        self.sum += suff_stat[0]
        self.sum2 += suff_stat[1]
        self.count += suff_stat[2]
        return self

    def value(self) -> tuple[float, float, float]:
        """Return accumulated sum, second moment sum, and count."""
        return self.sum, self.sum2, self.count

    def from_value(self, x: tuple[float, float, float]) -> "BetaBinomialAccumulator":
        """Replace accumulator contents from a sufficient-statistic tuple."""
        self.sum, self.sum2, self.count = float(x[0]), float(x[1]), float(x[2])
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

    def acc_to_encoder(self) -> "BetaBinomialDataEncoder":
        """Return the encoder used by this accumulator."""
        return BetaBinomialDataEncoder()


class BetaBinomialAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for BetaBinomialAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> BetaBinomialAccumulator:
        """Create a fresh beta-binomial accumulator."""
        return BetaBinomialAccumulator(name=self.name, keys=self.keys)


class BetaBinomialEstimator(ParameterEstimator):
    """Method-of-moments estimator for the beta-binomial shape parameters."""

    def __init__(
        self,
        n: int,
        min_conc: float = 1.0e-6,
        max_conc: float = 1.0e8,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.n = int(n)
        self.min_conc = min_conc
        self.max_conc = max_conc
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> BetaBinomialAccumulatorFactory:
        """Return an accumulator factory for beta-binomial moments."""
        return BetaBinomialAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> BetaBinomialDistribution:
        """Estimate beta-binomial shape parameters from weighted moments."""
        sum_x, sum_x2, count = suff_stat
        n = self.n
        if count <= 0.0 or n == 0:
            return BetaBinomialDistribution(n, 1.0, 1.0, name=self.name, keys=self.keys)
        mean = sum_x / count
        var = sum_x2 / count - mean * mean
        pi = min(max(mean / n, 1.0e-12), 1.0 - 1.0e-12)  # success probability a/(a+b)
        binom_var = n * pi * (1.0 - pi)
        if n == 1 or binom_var <= 0.0:
            s = self.max_conc  # no overdispersion information -> binomial limit
        else:
            rho = (var / binom_var - 1.0) / (n - 1.0)  # intra-class correlation
            rho = min(max(rho, 1.0 / (self.max_conc + 1.0)), 1.0 - 1.0e-9)
            s = 1.0 / rho - 1.0  # a + b
        s = min(max(s, self.min_conc), self.max_conc)
        return BetaBinomialDistribution(n, pi * s, (1.0 - pi) * s, name=self.name, keys=self.keys)


class BetaBinomialDataEncoder(DataSequenceEncoder):
    """Encode beta-binomial counts as a float array."""

    def __str__(self) -> str:
        return "BetaBinomialDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BetaBinomialDataEncoder)

    def seq_encode(self, x: Sequence[int]) -> np.ndarray:
        """Encode counts as a floating-point array."""
        return np.asarray(x, dtype=np.float64)
