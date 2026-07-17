"""Binomial distributions, estimators, samplers, accumulators, and encoders.

The observation type is ``int``. The implementation supports ordinary
fixed-trial binomial data as well as shifted support through ``min_val``.

Reference: Johnson, Kemp & Kotz, *Univariate Discrete Distributions*
(3rd ed., Wiley, 2005).
"""

import math
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
from numpy.random import RandomState

from mixle.enumeration.algorithms import QuantizedCrossIndex, QuantizedEnumerationIndex
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.univariate.continuous.beta import BetaDistribution
from mixle.utils.special import digamma
from mixle.utils.vector import gammaln

E = tuple[np.ndarray, np.ndarray, np.ndarray, int, int]


def _fisher_mean_var(dist):
    shift = 0.0 if getattr(dist, "min_val", None) is None else float(dist.min_val)
    n = float(dist.n)
    p = float(dist.p)
    return shift + n * p, n * p * (1.0 - p)


def _fisher_encoded(enc_data):
    if isinstance(enc_data, tuple):
        if len(enc_data) >= 3:
            return np.asarray(enc_data[2], dtype=np.float64)
        return np.asarray(enc_data[0], dtype=np.float64)
    return np.asarray(enc_data, dtype=np.float64)


class BinomialDistribution(SequenceEncodableProbabilityDistribution):
    """Binomial distribution over ``min_val + {0, ..., n}`` with success probability ``p``."""

    @classmethod
    def compute_capabilities(cls):
        """Declare backend support for generated Binomial density kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the generated-compute declaration for the Binomial distribution."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ExponentialFamilySpec,
            ParameterSpec,
            StatisticSpec,
        )

        return DistributionDeclaration(
            name="binomial",
            distribution_type=cls,
            parameters=(
                ParameterSpec("p", constraint="unit_interval"),
                ParameterSpec("n", constraint="non_negative_integer", differentiable=False),
                ParameterSpec("min_val", constraint="optional_integer", differentiable=False),
            ),
            statistics=(
                StatisticSpec("count"),
                StatisticSpec("sum"),
                StatisticSpec("min_val", kind="support_bound", additive=False, scales=False),
                StatisticSpec("max_val", kind="support_bound", additive=False, scales=False),
            ),
            support="bounded_integer",
            exponential_family=ExponentialFamilySpec(
                sufficient_statistics=cls.exp_family_sufficient_statistics,
                natural_parameters=cls.exp_family_natural_parameters,
                log_partition=cls.exp_family_log_partition,
                base_measure=cls.exp_family_base_measure,
                sufficient_statistics_from_params=cls.exp_family_sufficient_statistics_from_params,
                base_measure_from_params=cls.exp_family_base_measure_from_params,
            ),
        )

    @staticmethod
    def _exp_family_shifted_values(x: E, params: dict[str, Any], engine: Any) -> Any:
        ux, ix, _, _, _ = x
        vals = engine.asarray(ux)
        min_val = params.get("min_val")
        if min_val is not None:
            vals = vals - engine.asarray(min_val)
        return vals[engine.asarray(ix)]

    @staticmethod
    def exp_family_sufficient_statistics(x: E, engine: Any) -> tuple[Any, ...]:
        """Return Binomial sufficient statistics from encoded observations.

        The parameter-dependent support shift is handled by
        ``exp_family_sufficient_statistics_from_params`` when generated scoring
        supplies the declaration-stacked parameter bundle.
        """
        _, _, vals, _, _ = x
        return (engine.asarray(vals),)

    @staticmethod
    def exp_family_sufficient_statistics_from_params(x: E, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return Binomial sufficient statistics for generated scoring."""
        return (BinomialDistribution._exp_family_shifted_values(x, params, engine),)

    @staticmethod
    def exp_family_natural_parameters(params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return Binomial natural parameters for generated scoring."""
        p = params["p"]
        return (engine.log(p) - engine.log(engine.asarray(1.0) - p),)

    @staticmethod
    def exp_family_log_partition(params: dict[str, Any], engine: Any) -> Any:
        """Return Binomial log partition for generated scoring."""
        p = params["p"]
        return -params["n"] * engine.log(engine.asarray(1.0) - p)

    @staticmethod
    def exp_family_base_measure(x: E, engine: Any) -> Any:
        """Return the observation-only Binomial base measure."""
        return engine.asarray(x[2]) * 0.0

    @staticmethod
    def exp_family_base_measure_from_params(x: E, params: dict[str, Any], engine: Any) -> Any:
        """Return Binomial support/base measure for generated scoring."""
        xx = BinomialDistribution._exp_family_shifted_values(x, params, engine)
        n = params["n"]
        one = engine.asarray(1.0)
        good = (xx >= 0.0) & (xx <= n) & (engine.floor(xx) == xx)
        base = engine.gammaln(n + one) - engine.gammaln(xx + one) - engine.gammaln(n - xx + one)
        return engine.where(good, base, engine.asarray(-np.inf))

    def __init__(
        self,
        p: float,
        n: int,
        min_val: int | None = None,
        name: str | None = None,
        keys: str | None = None,
        prior: SequenceEncodableProbabilityDistribution | None = None,
    ) -> None:
        """Create a shifted binomial distribution.

        The support is ``min_val + {0, ..., n}``, with ``min_val=0`` by
        default. The log probability of ``x`` is the ordinary binomial
        log-mass evaluated at ``x - min_val`` and is ``-inf`` outside support.

        Args:
            p: Success probability in ``(0, 1)``.
            n: Number of trials.
            min_val: Optional support shift.
            name: Optional distribution name.
            keys: Optional merge key for sufficient statistics.

        Attributes:
            p: Success probability.
            log_p: ``log(p)``.
            log_1p: ``log(1 - p)``.
            n: Number of trials.
            min_val: Optional support shift.
            name: Optional distribution name.
            keys: Optional merge key.
        """
        if p <= 0.0 or p >= 1.0:
            raise ValueError("Binomial distribution requires p in (0, 1).")
        else:
            self.p = float(p)

        if n < 0 or np.isinf(n) or int(n) != n:
            raise ValueError("Binomial distribution requires a non-negative integer n.")
        else:
            self.n = int(n)

        self.log_p = np.log(p)
        self.log_1p = np.log1p(-p)
        self.name = name
        self.keys = keys
        self.min_val = min_val
        self.set_prior(prior)

    def set_prior(self, prior: SequenceEncodableProbabilityDistribution | None) -> None:
        """Attach a Beta parameter prior and precompute conjugate-prior expectations.

        With a Beta(a, b) prior on the success probability ``p`` this caches
        ``(E[log p], E[log(1-p)]) = (digamma(a) - digamma(a+b), digamma(b) - digamma(a+b))``
        so that ``expected_log_density`` evaluates the variational Bayes expectation
        ``E_q[log p(x | p)]``. Any other prior (including ``None``) leaves the distribution a
        plain point model.
        """
        self.prior = prior
        if isinstance(prior, BetaDistribution):
            a, b = prior.get_parameters()
            self.expected_nparams = (digamma(a) - digamma(a + b), digamma(b) - digamma(a + b))
            self.has_conj_prior = True
        else:
            self.expected_nparams = None
            self.has_conj_prior = False

    def expected_log_density(self, x: int) -> float:
        """Variational expectation ``E_q[log p(x | p)]`` under the Beta prior.

        Uses the cached digamma expectations of ``log p`` and ``log(1-p)``; falls back to the
        plug-in ``log_density(x)`` when no conjugate prior is attached.
        """
        if not self.has_conj_prior:
            return self.log_density(x)
        n = self.n
        try:
            xx = float(x)
        except Exception:  # noqa: BLE001
            return -np.inf
        if self.min_val is not None:
            xx -= self.min_val
        if not np.isfinite(xx) or np.floor(xx) != xx or xx < 0 or xx > n:
            return -np.inf
        xx = int(xx)
        e1, e2 = self.expected_nparams
        cc = gammaln(n + 1) - gammaln(xx + 1) - gammaln(n - xx + 1)
        return cc + xx * e1 + (n - xx) * e2

    def seq_expected_log_density(self, x: E) -> np.ndarray:
        """Vectorized ``expected_log_density`` over sequence-encoded observations."""
        if not self.has_conj_prior:
            return self.seq_log_density(x)
        ux, ix, _, _, _ = x
        n = self.n
        gn = gammaln(n + 1)
        xx = ux - self.min_val if self.min_val is not None else ux
        good = np.isfinite(xx) & (np.floor(xx) == xx) & (xx >= 0) & (xx <= n)
        e1, e2 = self.expected_nparams
        cc = np.full_like(xx, -np.inf, dtype=np.float64)
        xg = xx[good]
        cc[good] = (gn - gammaln(xg + 1) - gammaln(n - xg + 1)) + xg * e1 + (n - xg) * e2
        return cc[ix]

    def __str__(self) -> str:
        """Get string representation of BinomialDistribution."""
        return "BinomialDistribution(p=%s, n=%s, min_val=%s, name=%s, keys=%s)" % (
            repr(self.p),
            repr(self.n),
            repr(self.min_val),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: int) -> float:
        """Returns the probability mass of integer value x.

        If x is not an integer between [0,n) or [min_val, n-1-min_val), density is 0.0.

        Args:
            x (int): Integer value for density evaluation.

        Returns:
            Probability mass of x for binomial(n,p) with min_val=min_val. 0.0 if x is not in support.
        """
        return np.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        """Returns the log-probability mass of integer value x.

        If x is not an integer between [0,n) or [min_val, n-1-min_val), log-density is -inf.

        Args:
            x (int): Integer value for density evaluation.

        Returns:
            Log-probability mass of x for binomial(n,p) with min_val=min_val. -inf if x is not in support.
        """
        n = self.n
        try:
            xx = float(x)
        except Exception:  # noqa: BLE001
            return -np.inf
        if self.min_val is not None:
            xx -= self.min_val

        if not np.isfinite(xx) or np.floor(xx) != xx or xx < 0 or xx > n:
            return -np.inf
        xx = int(xx)
        return (gammaln(n + 1) - gammaln(xx + 1) - gammaln(n - xx + 1)) + self.log_1p * (n - xx) + self.log_p * xx

    def seq_log_density(self, x: E) -> np.ndarray:
        """Vectorized evaluation of log-density for sequence encoded data.

        Input value x must be obtained from a call to BinomialDataEncoder.seq_encode(data). Returns numpy array
        of log-density evaluated at all observations contained in encoded data x.

        Args:
            x (Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]): containing unique values in x, indices of ux to
                reconstruct x, numpy array of x, min value of x, and max value of x.

        Returns:
            Numpy array of log-density evaluated at all observations contained in encoded data x.

        """
        ux, ix, _, _, _ = x
        n = self.n
        gn = gammaln(n + 1)

        if self.min_val is not None:
            xx = ux - self.min_val
        else:
            xx = ux

        good = np.isfinite(xx) & (np.floor(xx) == xx) & (xx >= 0) & (xx <= n)
        cc = np.full_like(xx, -np.inf, dtype=np.float64)
        xg = xx[good]
        cc[good] = (gn - gammaln(xg + 1) - gammaln(n - xg + 1)) + self.log_1p * (n - xg) + self.log_p * xg
        return cc[ix]

    @staticmethod
    def backend_log_density_from_params(vals: Any, n: Any, p: Any, min_val: int | None, engine: Any) -> Any:
        """Engine-neutral binomial log-density from explicit parameters."""
        xx = vals - engine.asarray(min_val) if min_val is not None else vals
        good = (xx >= 0.0) & (xx <= n) & (engine.floor(xx) == xx)
        one = engine.asarray(1.0)
        rv = (
            engine.gammaln(n + one)
            - engine.gammaln(xx + one)
            - engine.gammaln(n - xx + one)
            + (n - xx) * engine.log(one - p)
            + xx * engine.log(p)
        )
        return engine.where(good, rv, engine.asarray(-np.inf))

    def backend_seq_log_density(self, x: E, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        ux, ix, _, _, _ = x
        vals = engine.asarray(ux)
        scores = self.backend_log_density_from_params(
            vals, engine.asarray(float(self.n)), engine.asarray(self.p), self.min_val, engine
        )
        return scores[engine.asarray(ix)]

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["BinomialDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked binomial parameters for a homogeneous mixture kernel.

        A stacked binomial mixture requires components to share the same trial count
        and shifted support. Mixtures with heterogeneous supports fall back to the
        generic kernel.
        """
        n = dists[0].n
        min_val = dists[0].min_val
        if any(d.n != n or d.min_val != min_val for d in dists):
            raise ValueError("Stacked BinomialDistribution components require shared n and min_val.")
        return {
            "p": engine.asarray([d.p for d in dists]),
            "n": engine.asarray(float(n)),
            "min_val": min_val,
        }

    @classmethod
    def backend_stacked_log_density(cls, x: E, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of binomial log densities."""
        ux, ix, _, _, _ = x
        vals = engine.asarray(ux)
        scores = cls.backend_log_density_from_params(
            vals[:, None], params["n"], params["p"][None, :], params["min_val"], engine
        )
        return scores[engine.asarray(ix), :]

    @classmethod
    def backend_stacked_sufficient_statistics_with_estimator(
        cls, x: E, weights: Any, params: dict[str, Any], engine: Any, estimator: Any
    ) -> tuple[Any, Any, Any, Any]:
        """Return stacked Binomial sufficient statistics using estimator-owned support bounds."""
        _, _, xx, min_val, max_val = x
        vals = engine.asarray(xx)
        ww = engine.asarray(weights)
        count = engine.sum(ww, axis=0)
        obs_sum = engine.sum(ww * vals[:, None], axis=0)
        init_min, init_max = _binomial_initial_bounds(estimator, len(np.asarray(engine.to_numpy(count))))
        min_bounds = [min_val if lo is None else min(lo, min_val) for lo in init_min]
        max_bounds = [max_val if hi is None else max(hi, max_val) for hi in init_max]
        return count, obs_sum, engine.asarray(min_bounds), engine.asarray(max_bounds)

    def support_size(self) -> int:
        """``n + 1`` outcomes ``min_val + {0, ..., n}``."""
        return int(self.n) + 1

    def to_fisher(self, **kwargs):
        """Return the Binomial's count-family Fisher view."""
        from mixle.inference.fisher import CountFisherView, _count_data

        return CountFisherView(self, _fisher_mean_var, _count_data, _fisher_encoded)

    def mean(self) -> float:
        """Mean E[X] of the distribution."""
        return float(float(getattr(self, "min_val", 0) or 0) + self.n * self.p)

    def variance(self) -> float:
        """Variance Var[X] of the distribution."""
        return float(self.n * self.p * (1.0 - self.p))

    def cdf(self, x: float) -> float:
        """Cumulative distribution function P(X <= x) over min_val + {0..n}."""
        import math

        from scipy.special import betainc

        shift = float(getattr(self, "min_val", 0) or 0)
        k = math.floor(float(x) - shift)
        if k < 0:
            return 0.0
        if k >= self.n:
            return 1.0
        return float(betainc(self.n - k, k + 1, 1.0 - self.p))

    def skewness(self) -> float:
        """Skewness (1-2p)/sqrt(npq)."""
        import math

        p, n = self.p, self.n
        return float((1.0 - 2.0 * p) / math.sqrt(n * p * (1.0 - p)))

    def kurtosis(self) -> float:
        """Excess kurtosis (1-6pq)/(npq)."""
        p, n = self.p, self.n
        return float((1.0 - 6.0 * p * (1.0 - p)) / (n * p * (1.0 - p)))

    def quantile(self, q: float) -> float:
        """Inverse CDF F^{-1}(q) over min_val + {0..n} (via scipy binom)."""
        from scipy.stats import binom

        shift = float(getattr(self, "min_val", 0) or 0)
        return float(shift + binom.ppf(float(q), self.n, self.p))

    def mode(self) -> float:
        """Mode min_val + floor((n+1)p)."""
        import math

        return float(getattr(self, "min_val", 0) or 0) + math.floor((self.n + 1.0) * self.p)

    def sampler(self, seed: int | None = None) -> "BinomialSampler":
        """Returns BinomialSampler for generating samples from BinomialDistribution(n,p,min_val).

        Args:
            seed Optional[int]: Used to set seed on random number generator for sampling.

        Returns:
            BinomialSampler for BinomialDistribution with seed.
        """
        return BinomialSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "BinomialEstimator":
        """Creates a BinomialEstimator for estimating parameters of BinomialDistribution.

        Args:
            pseudo_count (Optional[float]): If set, inflates counts for currently set sufficient statistic (p).

        Returns:
            BinomialEstimator object.
        """
        if self.prior is not None:
            min_val = self.min_val if self.min_val is not None else 0
            return BinomialEstimator(
                max_val=self.n + min_val,
                min_val=min_val,
                name=self.name,
                keys=self.keys,
                prior=self.prior,
            )
        if pseudo_count is None:
            return BinomialEstimator(name=self.name, keys=self.keys)
        else:
            return BinomialEstimator(
                max_val=self.n,
                min_val=self.min_val,
                pseudo_count=pseudo_count,
                suff_stat=self.p * self.n * pseudo_count,
                name=self.name,
            )

    def dist_to_encoder(self) -> "BinomialDataEncoder":
        """Create the encoder for sequence-encoded binomial observations.

        Returns:
            BinomialDataEncoder object.
        """
        return BinomialDataEncoder()

    def enumerator(self) -> "BinomialEnumerator":
        """Returns BinomialEnumerator iterating the support in descending probability order."""
        return BinomialEnumerator(self)

    def quantized_index(self, max_bits: float, bin_width_bits: float = 1.0) -> QuantizedEnumerationIndex:
        """Build a bounded bit-quantized index by walking the binomial mode outward."""
        if max_bits < 0:
            raise ValueError("max_bits must be non-negative.")
        if bin_width_bits <= 0:
            raise ValueError("bin_width_bits must be positive.")

        shift = self.min_val if self.min_val is not None else 0
        mode = int(np.floor((self.n + 1) * self.p))
        mode = min(max(mode, 0), self.n)
        left = mode
        right = mode + 1
        limit_lp = -(float(max_bits) + 1.0e-12) * math.log(2.0)
        items: list[tuple[int, float]] = []

        while left >= 0 or right <= self.n:
            lp_l = self.log_density(shift + left) if left >= 0 else -np.inf
            lp_r = self.log_density(shift + right) if right <= self.n else -np.inf
            if lp_l < limit_lp and lp_r < limit_lp:
                break
            if lp_l >= lp_r:
                if lp_l >= limit_lp:
                    items.append((shift + left, float(lp_l)))
                left -= 1
            else:
                if lp_r >= limit_lp:
                    items.append((shift + right, float(lp_r)))
                right += 1

        return QuantizedEnumerationIndex.from_items(
            items,
            max_bits=max_bits,
            bin_width_bits=bin_width_bits,
            sorted_items=True,
            truncated=len(items) < self.n + 1,
        )

    def quantized_multi_cross_index(self, others, max_bits, bin_width_bits: float = 1.0) -> QuantizedCrossIndex:
        """Build an exact aligned cross-bin view over finite binomial supports."""
        dists = [self] + list(others)
        if any(not isinstance(dist, BinomialDistribution) for dist in dists):
            return super().quantized_multi_cross_index(others, max_bits=max_bits, bin_width_bits=bin_width_bits)

        lo = min((dist.min_val if dist.min_val is not None else 0) for dist in dists)
        hi = max((dist.min_val if dist.min_val is not None else 0) + dist.n for dist in dists)
        items = []
        for value in range(lo, hi + 1):
            items.append((value, tuple(float(dist.log_density(value)) for dist in dists)))
        return QuantizedCrossIndex.from_items(items, max_bits=max_bits, bin_width_bits=bin_width_bits)

    def quantized_cross_index(self, other, max_bits, bin_width_bits: float = 1.0) -> QuantizedCrossIndex:
        """Build an exact aligned cross-bin view over two binomial supports."""
        return self.quantized_multi_cross_index([other], max_bits=max_bits, bin_width_bits=bin_width_bits)


def _binomial_initial_bounds(estimator: Any, num_components: int) -> tuple[list[int | None], list[int | None]]:
    estimators = tuple(getattr(estimator, "estimators", ()))
    if len(estimators) != num_components:
        return [None] * num_components, [None] * num_components
    min_bounds = [getattr(est, "min_val", None) for est in estimators]
    max_bounds = [getattr(est, "max_val", None) for est in estimators]
    return min_bounds, max_bounds


class BinomialEnumerator(DistributionEnumerator):
    """Enumerate bounded Binomial support values in descending probability order."""

    def __init__(self, dist: BinomialDistribution) -> None:
        """Enumerates the support of a BinomialDistribution in descending probability order.

        The binomial pmf is unimodal, so enumeration starts at the mode floor((n+1)*p)
        and walks outward with two pointers, emitting the larger side each step.

        Args:
            dist (BinomialDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        self._shift = dist.min_val if dist.min_val is not None else 0
        mode = int(np.floor((dist.n + 1) * dist.p))
        self._mode = min(max(mode, 0), dist.n)
        self._left = self._mode - 1
        self._right = self._mode + 1
        self._started = False

    def _lp(self, i: int) -> float:
        return self.dist.log_density(self._shift + i)

    def __next__(self) -> tuple[int, float]:
        if not self._started:
            self._started = True
            return (self._shift + self._mode, self._lp(self._mode))
        while self._left >= 0 or self._right <= self.dist.n:
            lp_l = self._lp(self._left) if self._left >= 0 else -np.inf
            lp_r = self._lp(self._right) if self._right <= self.dist.n else -np.inf
            if lp_l >= lp_r:
                i, lp = self._left, lp_l
                self._left -= 1
            else:
                i, lp = self._right, lp_r
                self._right += 1
            if lp > -np.inf:
                return (self._shift + i, lp)
        raise StopIteration


class BinomialSampler(DistributionSampler):
    """Draw independent samples from a :class:`BinomialDistribution`."""

    def __init__(self, dist: BinomialDistribution, seed: int | None = None) -> None:
        """Create a sampler for a binomial distribution.

        Args:
            dist: Distribution to sample from.
            seed: Optional random seed.

        Attributes:
            dist (BinomialDistribution): BinomialDistribution to sample from.
            seed (Optional[int]): Seed for setting random number generator.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> int | list[int]:
        """Draw samples from BinomialSampler.

        Args:
            size (Optional[int]): Number of samples to draw from BinomialSampler (1 if size is None).

        Returns:
            An integer sample from BinomialDistribution(n,p,min_val), or List[int] of samples with length = size.

        """
        rv = self.rng.binomial(n=self.dist.n, p=self.dist.p, size=size)

        if size is None:
            if self.dist.min_val is not None:
                return int(rv) + self.dist.min_val
            else:
                return int(rv)
        else:
            if self.dist.min_val is not None:
                return list(rv + self.dist.min_val)
            else:
                return list(rv)


class BinomialAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted Binomial count, sum, and observed support bounds."""

    def __init__(
        self,
        max_val: int | None = None,
        min_val: int | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create an accumulator for binomial sufficient statistics.

        Sufficient statistics are ``(count, sum, min_val, max_val)``.

        Args:
            max_val (Optional[int]): Largest integer value encountered while accumulating sufficient statistics.
            min_val (Optional[int]): Smallest integer value encountered while accumulating sufficient statistics.
            name (Optional[str]): Assign a name to the instance of BinomialAccumulator.
            keys (Optional[str]): All BinomialAccumulators with same keys will have suff-stats merged.

        Attributes:
            sum (float): Aggregates the sum of all data observations.
            count (float): Aggregates the number of weighted-data observations used in accumulating sum.
            max_val (Optional[int]): Largest integer value encountered while accumulating sufficient statistics.
            min_val (Optional[int]): Smallest integer value encountered while accumulating sufficient statistics.
            name (Optional[str]): Assign a name to the instance of BinomialAccumulator.
            key (Optional[str]): All BinomialAccumulators with same key will have suff-stats merged.

        """
        self.sum = 0.0
        self.count = 0.0
        self.keys = keys
        self.name = name
        self.max_val = max_val
        self.min_val = min_val

    def update(self, x: int, weight: float, estimate: Optional["BinomialDistribution"]) -> None:
        """Accumulates Binomial sufficient statistics for weighted single observation.

        Add x*weight to attribute sum, and increase the count by weight.

        Args:
            x (int): Data observed.
            weight (float): Weight for observation.
            estimate (Optional[BinomialDistribution]): Previous estimate of BinomialDistribution obtained from
                prior data.

        Returns:
            None (updates BinomialAccumulator sufficient statistics.)

        """
        self.sum += x * weight
        self.count += weight

        if self.min_val is None:
            self.min_val = x
        else:
            self.min_val = min(self.min_val, x)

        if self.max_val is None:
            self.max_val = x
        else:
            self.max_val = max(self.max_val, x)

    def initialize(self, x: int, weight: float, rng: RandomState | None) -> None:
        """Initialize BinomialAccumulator sufficient statistics for one weighted observation.

        Args:
            x (int): Data observed.
            weight (float): Weight for observation.
            rng (Optional[RandomState]): RandomState not needed. No randomness in initialization.

        Returns:
            None (updates BinomialAccumulator sufficient statistics.)

        """
        self.update(x, weight, None)

    def seq_update(self, x: E, weights: np.ndarray, estimate: Optional["BinomialDistribution"]) -> None:
        """Accumulates Binomial sufficient statistics for encoded sequence.

        Args:
            x (E): Encoded sequence of observations.
            weights (np.ndarray): Numpy array of floats for weighting each observation.
            estimate (Optional[BinomialDistribution]): Previous estimate of BinomialDistribution obtained from
                prior data.:

        Returns:
            None

        """
        _, _, xx, min_val, max_val = x

        self.sum += np.sum(xx * weights)
        self.count += np.sum(weights)

        if self.min_val is not None:
            self.min_val = min(self.min_val, min_val)
        else:
            self.min_val = min_val

        if self.max_val is not None:
            self.max_val = max(self.max_val, max_val)
        else:
            self.max_val = max_val

    def seq_update_engine(self, x: E, weights: Any, estimate: Optional["BinomialDistribution"], engine: Any) -> None:
        """Engine-resident accumulation of count/sum statistics (numpy or torch).

        The weighted sum and count reductions run on the active engine; the scalar min/max
        support bounds remain host bookkeeping. Matches seq_update.
        """
        _, _, xx, min_val, max_val = x
        weights_np = np.asarray(engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, dtype=np.float64)
        w = engine.asarray(weights_np)
        xv = engine.asarray(np.asarray(xx, dtype=np.float64))

        self.sum += float(engine.to_numpy(engine.sum(xv * w)))
        self.count += float(engine.to_numpy(engine.sum(w)))

        if self.min_val is not None:
            self.min_val = min(self.min_val, min_val)
        else:
            self.min_val = min_val

        if self.max_val is not None:
            self.max_val = max(self.max_val, max_val)
        else:
            self.max_val = max_val

    def seq_initialize(self, x: E, weights: np.ndarray, rng: RandomState | None) -> None:
        """Vectorized initialization of BinomialAccumulator sufficient statistics with weights.

        Calls seq_update().

        Args:
            x (E): Encoded sequence of observations.
            weights (np.ndarray): Numpy array of floats for weighting each observation.
            rng (Optional[RandomState]): RandomState not needed. No randomness in initialization.

        Returns:
            None

        """
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, int | None, int | None]) -> "BinomialAccumulator":
        """Combine the sufficient statistics of BinomialAccumulator with suff_stat.

        Args:
            suff_stat (Tuple[float, float, Optional[int], Optional[int]]): Count, sum of observations, optional min_val
                observed, and optional max_val observed.

        Returns:
            None

        """
        self.sum += suff_stat[1]
        self.count += suff_stat[0]

        if self.min_val is None:
            self.min_val = suff_stat[2]
        elif self.min_val is not None and suff_stat[2] is not None:
            self.min_val = min(self.min_val, suff_stat[2])

        if self.max_val is None:
            self.max_val = suff_stat[3]
        elif self.max_val is not None and suff_stat[3] is not None:
            self.max_val = max(self.max_val, suff_stat[3])

        return self

    def value(self) -> tuple[float, float, int | None, int | None]:
        """Returns the sufficient statistics, and member variables min_val and max_val.

        Returns:
            Tuple[float,float, Optional[int], Optional[int]] containing suff stats count, sum and attributes min_val
                max_val if they are not None.

        """
        return self.count, self.sum, self.min_val, self.max_val

    def from_value(self, x: tuple[float, float, int | None, int | None]) -> "BinomialAccumulator":
        """Set BinomialAccumulator suff stats and member variables from suff_stat tuple defined in value().

        Takes tuple of (count, sum, min_val, max_val) for setting values of BinomialAccumulator.

        Args:
            x (Tuple[float,float, Optional[int], Optional[int]]): containing suff stats count, sum and attributes
                min_val max_val if they are not None.

        Returns:
            None, sets sufficient statistics and member variables.

        """
        self.count = x[0]
        self.sum = x[1]
        self.min_val = x[2]
        self.max_val = x[3]

        return self

    def scale(self, c: float) -> "BinomialAccumulator":
        """Scale linear count/sum statistics while preserving support bounds."""
        self.count *= c
        self.sum *= c
        return self

    def acc_to_encoder(self) -> "BinomialDataEncoder":
        """Create the encoder for binomial observations.

        Note: Used for seq_initialize.

        Returns:
            BinomialDataEncoder()

        """
        return BinomialDataEncoder()


class BinomialAccumulatorFactory(StatisticAccumulatorFactory):
    """Create Binomial accumulators with configured support-bound metadata."""

    def __init__(
        self,
        max_val: int | None = None,
        min_val: int | None = 0,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create a binomial accumulator factory.

        Args:
            max_val (Optional[int]): Max value for binomial observations.
            min_val (Optional[int]): min value for binomial observations.
            name (Optional[str]): Name the BinomialAccumulatorFactory.
            keys (Optional[str]): Declare BinomialAccumulatorFactory objects for merging suff_stats.

        Return:
             None
        """
        self.max_val = max_val
        self.min_val = min_val
        self.name = name
        self.keys = keys

    def make(self) -> "BinomialAccumulator":
        """Create a fresh binomial accumulator.

        Returns:
            BinomialAccumulator.

        """
        return BinomialAccumulator(self.max_val, self.min_val, self.name, self.keys)


class BinomialEstimator(ParameterEstimator):
    """Estimate Binomial probability and support parameters from sufficient statistics."""

    def __init__(
        self,
        max_val: int | None = None,
        min_val: int | None = 0,
        pseudo_count: float | None = None,
        suff_stat: float | None = None,
        name: str | None = None,
        keys: str | None = None,
        prior: SequenceEncodableProbabilityDistribution | None = None,
    ) -> None:
        """Create an estimator for a binomial distribution.

        Args:
            max_val (Optional[int]): Set max value encountered.
            min_val (Optional[int]): Set min value for BinomialDistribution.
            pseudo_count (Optional[float]): Inflate sufficient statistic (p).
            suff_stat (Optional[float]): Set p from prior observations.
            name (Optional[str]): Assign a name to the estimator.
            keys (Optional[str]): Assign key to BinomialEstimator designating all same key estimators to later be combined,
                in accumualtation.

        Attributes:
            max_val (Optional[int]): Set max value encountered.
            min_val (Optional[int]): Set min value for BinomialDistribution.
            pseudo_count (Optional[float]): Inflate sufficient statistic (p).
            suff_stat (Optional[float]): Set p from prior observations.
            name (Optional[str]): Assign a name to the estimator.
            keys (Optional[str]): Assign key to BinomialEstimator designating all same key estimators to later be combined,
                in accumualtation.

        """
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name
        self.min_val = min_val if min_val is not None else 0
        self.max_val = max_val
        self.prior = prior
        self.has_conj_prior = isinstance(prior, BetaDistribution)

    def accumulator_factory(self) -> BinomialAccumulatorFactory:
        """Create a BinomialAccumulatorFactory from estimator member variables.

        Returns:
            BinomialAccumulatorFactory

        """
        return BinomialAccumulatorFactory(self.max_val, self.min_val, self.name, self.keys)

    def model_log_density(self, model: "BinomialDistribution") -> float:
        """Log-density of the model's success probability under the Beta prior (ELBO global term)."""
        if self.has_conj_prior:
            return float(self.prior.log_density(model.p))
        return 0.0

    def _estimate_conjugate(self, suff_stat: tuple[float, float, int | None, int | None]) -> "BinomialDistribution":
        """Closed-form Beta conjugate posterior update returning the MAP estimate.

        The number of trials ``n`` is treated as known (recovered from the estimator's
        ``min_val``/``max_val`` bounds). With a Beta(a, b) prior and weighted counts of
        successes ``psum`` and failures ``fsum``, the posterior is Beta(a + psum, b + fsum)
        and the returned point estimate is the posterior mode ``(a' - 1)/(a' + b' - 2)`` when
        ``a', b' > 1``, falling back to the posterior mean ``a'/(a' + b')`` on the boundary;
        the posterior is carried forward as the fitted model's prior.
        """
        count, total, _, _ = suff_stat
        min_val = self.min_val if self.min_val is not None else 0
        n = (self.max_val if self.max_val is not None else 0) - min_val
        psum = total - min_val * count
        fsum = count * n - psum

        a, b = self.prior.get_parameters()
        new_a = a + psum
        new_b = b + fsum
        if new_a > 1.0 and new_b > 1.0:
            p = (new_a - 1.0) / (new_a + new_b - 2.0)
        else:
            p = new_a / (new_a + new_b)
        return BinomialDistribution(
            p, n, min_val=min_val, name=self.name, keys=self.keys, prior=BetaDistribution(new_a, new_b)
        )

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, int | None, int | None]):
        """Estimate a binomial distribution from accumulated sufficient statistics.

        ``nobs`` is accepted for estimator API consistency but is not used. The
        sufficient statistic is ``(count, sum, min_val, max_val)``. When a prior
        statistic and ``pseudo_count`` are present, they shrink the empirical
        success probability toward the estimator's prior probability.
        """
        if self.has_conj_prior:
            return self._estimate_conjugate(suff_stat)

        count, sum, min_val, max_val = suff_stat

        if min_val is not None:
            if self.min_val is not None:
                min_val = min(min_val, self.min_val)
        else:
            if self.min_val is not None:
                min_val = self.min_val
            else:
                min_val = 0

        if max_val is not None:
            if self.max_val is not None:
                max_val = max(max_val, self.max_val)
        else:
            if self.max_val is not None:
                max_val = self.max_val
            else:
                max_val = 0

        # Number of trials inferred from the observed support span. CAVEAT: this is data-driven --
        # if the true maximum (n successes) is never observed, ``max_val`` underestimates it and the
        # estimated ``n`` (and hence ``p``) is biased high. To pin the trial count, construct the
        # estimator with an explicit ``max_val`` (and/or ``min_val``); the observed span is then only
        # widened to include it, never used to shrink a known bound below the data.
        n = max_val - min_val

        if self.pseudo_count is not None and self.suff_stat is not None:
            pn = self.pseudo_count
            pp = self.suff_stat
            p = (sum - min_val * count + pp) / ((count + pn) * n)

        elif self.pseudo_count is not None and self.suff_stat is None:
            pn = self.pseudo_count
            pp = self.pseudo_count * 0.5 * n
            p = (sum - min_val * count + pp) / ((count + pn) * n)

        else:
            if count > 0 and n > 0:
                p = (sum - min_val * count) / (count * n)
            else:
                p = 0.5

        p = float(np.clip(p, 1.0e-12, 1.0 - 1.0e-12))
        return BinomialDistribution(p, max_val - min_val, min_val=min_val, name=self.name, keys=self.keys)


class BinomialDataEncoder(DataSequenceEncoder):
    """Data encoder for iid integer binomial observations."""

    def __str__(self) -> str:
        """Creates string name of BinomialDataEncoder.

        Returns:
            String name BinomialDataEncoder

        """
        return "BinomialDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Define equality for BinomialDataEncoder objects.

        Args:
            other (object): Any object to be compares to BinomialDataEncoder.

        Returns:
            True is other is BinomialDataEncoder, else False.

        """
        return isinstance(other, BinomialDataEncoder)

    def seq_encode(self, x: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        """Encode List[int] for vectorized seq calls in Accumulator and Distribution.

        Args:
            x (List[int]): List of integers.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, int, int] containing unique values in x, indices of ux to
                reconstruct x, numpy array of x, min value of x, and max value of x.

        """
        xx0 = np.asarray(x, dtype=np.float64)

        if np.any(xx0 < 0) or np.any(np.isnan(xx0)) or np.any(np.floor(xx0) != xx0):
            raise ValueError("BinomialDistribution requires non-negative integer values for x.")

        xx = np.asarray(xx0, dtype=np.int32)
        if xx.size == 0:
            return xx, np.asarray([], dtype=np.int64), xx, 0, 0
        ux, ix = np.unique(xx, return_inverse=True)
        min_val = np.min(ux)
        max_val = np.max(ux)

        return ux, ix, xx, min_val, max_val
