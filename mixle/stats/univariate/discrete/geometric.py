"""Geometric distributions, estimators, samplers, accumulators, and encoders.

The observation type is ``int`` on support ``{1, 2, ...}``. The log-density is
``(k - 1) * log(1 - p) + log(p)`` for ``k >= 1``.

Reference: Johnson, Kemp & Kotz, *Univariate Discrete Distributions*
(3rd ed., Wiley, 2005).
"""

import math
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
from numpy.random import RandomState

from mixle.engines.arithmetic import *
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


def _fisher_mean_var(dist):
    p = float(dist.p)
    return 1.0 / p, (1.0 - p) / (p * p)


class GeometricDistribution(SequenceEncodableProbabilityDistribution):
    """Geometric distribution on ``{1, 2, ...}`` with success probability ``p``."""

    @classmethod
    def compute_capabilities(cls):
        """Declare backend support for generated Geometric density kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch", "jax"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the generated-compute declaration for the Geometric distribution."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ExponentialFamilySpec,
            ParameterSpec,
            StatisticSpec,
        )

        return DistributionDeclaration(
            name="geometric",
            distribution_type=cls,
            parameters=(ParameterSpec("p", constraint="unit_interval"),),
            statistics=(StatisticSpec("count"), StatisticSpec("sum")),
            support="positive_integer",
            exponential_family=ExponentialFamilySpec(
                sufficient_statistics=cls.exp_family_sufficient_statistics,
                natural_parameters=cls.exp_family_natural_parameters,
                log_partition=cls.exp_family_log_partition,
                legacy_sufficient_statistics=cls.exp_family_legacy_sufficient_statistics,
            ),
        )

    @staticmethod
    def exp_family_legacy_sufficient_statistics(x: Any, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return per-row Geometric sufficient statistics in accumulator order."""
        xx = engine.asarray(x)
        return xx * 0.0 + engine.asarray(1.0), xx

    @staticmethod
    def exp_family_sufficient_statistics(x: Any, engine: Any) -> tuple[Any, ...]:
        """Return Geometric sufficient statistic ``T(x) = (x,)`` (support x = 1, 2, ...)."""
        return (engine.asarray(x),)

    @staticmethod
    def exp_family_natural_parameters(params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return Geometric natural parameter ``eta = log(1 - p)``."""
        return (engine.log(engine.asarray(1.0) - params["p"]),)

    @staticmethod
    def exp_family_log_partition(params: dict[str, Any], engine: Any) -> Any:
        """Return Geometric log partition ``A = log(1 - p) - log(p)``."""
        p = params["p"]
        return engine.log(engine.asarray(1.0) - p) - engine.log(p)

    @staticmethod
    def exp_family_from_natural(eta: Any) -> "GeometricDistribution":
        """Return the Geometric with natural parameter ``eta = log(1 - p)``."""
        import numpy as _np

        return GeometricDistribution(float(1.0 - _np.exp(float(eta[0]))))

    def __init__(
        self,
        p: float,
        name: str | None = None,
        prior: SequenceEncodableProbabilityDistribution | None = None,
    ) -> None:
        """Create a geometric distribution with success probability ``p``.

        The mean is ``1 / p`` and the variance is ``(1 - p) / p**2``.

        Args:
            p: Success probability in ``(0, 1]``.
            name: Optional distribution name.
            prior (Optional): Conjugate Beta prior on the success probability ``p``. A
                :class:`~mixle.stats.univariate.continuous.beta.BetaDistribution` enables the Bayesian/variational
                machinery (``expected_log_density`` and the conjugate posterior update);
                ``None`` (default) is a plain point model.

        Attributes:
            p: Success probability.
            log_p: ``log(p)``.
            log_1p: ``log(1 - p)``.
            name: Optional distribution name.
        """
        if p <= 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError("GeometricDistribution requires p in (0, 1].")
        self.p = float(p)
        self.log_p = np.log(self.p)
        self.log_1p = np.log1p(-self.p)
        self.name = name
        self.set_prior(prior)

    def __str__(self) -> str:
        """Return a constructor-style representation of the geometric distribution."""
        return "GeometricDistribution(%s, name=%s)" % (repr(self.p), repr(self.name))

    def set_prior(self, prior: SequenceEncodableProbabilityDistribution | None) -> None:
        """Attach a Beta parameter prior and precompute conjugate-prior expectations.

        With a Beta(a, b) prior on the success probability ``p`` this caches the digamma
        terms ``(digamma(a), digamma(b), digamma(a+b))`` so that ``expected_log_density``
        evaluates the variational Bayes expectation ``E_q[log p(x | p)]`` via
        ``E[log p] = digamma(a) - digamma(a+b)`` and ``E[log(1-p)] = digamma(b) - digamma(a+b)``.
        Any other prior (including ``None``) leaves the distribution a plain point model.
        """
        self.prior = prior
        if isinstance(prior, BetaDistribution):
            a, b = prior.get_parameters()
            self.conj_prior_params = (digamma(a), digamma(b), digamma(a + b))
            self.has_conj_prior = True
        else:
            self.conj_prior_params = (0, 0, 0)
            self.has_conj_prior = False

    def expected_log_density(self, x: int) -> float:
        """Variational expectation ``E_q[log p(x | p)]`` under the Beta prior.

        Uses the cached digamma expectations of ``log p`` and ``log(1-p)``; falls back to
        the plug-in ``log_density(x)`` when no conjugate prior is attached.
        """
        if self.has_conj_prior:
            ga, gb, gab = self.conj_prior_params
            if x < 1:
                return -np.inf
            return (gb - gab) * (x - 1) + (ga - gab)
        return self.log_density(x)

    def seq_expected_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized ``expected_log_density`` over sequence-encoded observations."""
        if self.has_conj_prior:
            ga, gb, gab = self.conj_prior_params
            rv = (x - 1) * (gb - gab) + (ga - gab)
            rv = np.where(x < 1, -np.inf, rv)
            return rv
        return self.seq_log_density(x)

    def density(self, x: int) -> float:
        """Density of geometric distribution evaluated at x.

            P(x=k) = (k-1)*log(1-p) + log(p), for x = 1,2,..., else 0.0.

        Args:
            x (int): Observed geometric value (1,2,3,....).


        Returns:
            Density of geometric distribution evaluated at x.

        """
        return exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        """Log-density of geometric distribution evaluated at x.

        See density() for details.

        Args:
            x (int): Must be natural number (1,2,3,....).

        Returns:
            Log-density of geometric distribution evaluated at x.

        """
        try:
            xx = float(x)
        except Exception:  # noqa: BLE001
            return -np.inf
        if not np.isfinite(xx) or xx < 1 or np.floor(xx) != xx:
            return -np.inf
        if self.p == 1.0:
            return 0.0 if xx == 1 else -np.inf
        return (xx - 1.0) * self.log_1p + self.log_p

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized log-density evaluated on sequence encoded x.

        Args:
            x (int): Numpy array of non-negative integers.

        Returns:
            Numpy array of log-density evaluated at each encoded observation value x.

        """
        xx = np.asarray(x, dtype=np.float64)
        good = np.isfinite(xx) & (xx >= 1) & (np.floor(xx) == xx)
        if self.p == 1.0:
            return np.where(good & (xx == 1), 0.0, -np.inf)
        rv = (xx - 1.0) * self.log_1p + self.log_p
        rv = np.where(good, rv, -np.inf)

        return rv

    @staticmethod
    def backend_log_density_from_params(x: Any, p: Any, engine: Any) -> Any:
        """Engine-neutral geometric log-density from explicit parameters."""
        one = engine.asarray(1.0)
        good = (x >= one) & (engine.floor(x) == x)
        log_p = engine.log(p)
        log_1p = engine.log(one - p)
        rv = (x - one) * log_1p + log_p
        at_one = engine.where((x == one) & good, engine.asarray(0.0), engine.asarray(-np.inf))
        rv = engine.where(p == one, at_one, rv)
        return engine.where(good, rv, engine.asarray(-np.inf))

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        xx = engine.asarray(x)
        return self.backend_log_density_from_params(xx, engine.asarray(self.p), engine)

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["GeometricDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked geometric parameters for a homogeneous mixture kernel."""
        return {"p": engine.asarray([d.p for d in dists])}

    @classmethod
    def backend_stacked_log_density(cls, x: Any, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of geometric log densities."""
        xx = engine.asarray(x)
        return cls.backend_log_density_from_params(xx[:, None], params["p"][None, :], engine)

    def to_fisher(self, **kwargs):
        """Return the Geometric's count-family Fisher view."""
        from mixle.inference.fisher import CountFisherView, _count_data, _identity_encoded

        return CountFisherView(self, _fisher_mean_var, _count_data, _identity_encoded)

    def mean(self) -> float:
        """Mean E[X] of the distribution."""
        return float(1.0 / self.p)

    def variance(self) -> float:
        """Variance Var[X] of the distribution."""
        return float((1.0 - self.p) / (self.p * self.p))

    def cdf(self, x: float) -> float:
        """Cumulative distribution function P(X <= x) = 1 - (1-p)^floor(x), support x >= 1."""
        import math

        k = math.floor(float(x))
        return float(1.0 - (1.0 - self.p) ** k) if k >= 1 else 0.0

    def quantile(self, q: float) -> float:
        """Inverse CDF F^{-1}(q), support >= 1 (via scipy geom)."""
        from scipy.stats import geom

        return float(geom.ppf(float(q), self.p))

    def entropy(self) -> float:
        """Shannon entropy (-(1-p) log(1-p) - p log p) / p (nats)."""
        import math

        p = self.p
        if p >= 1.0:
            return 0.0
        return float((-(1.0 - p) * math.log(1.0 - p) - p * math.log(p)) / p)

    def mode(self) -> float:
        """Mode (1 -- the minimum of the decreasing pmf)."""
        return 1.0

    def sampler(self, seed: int | None = None) -> "GeometricSampler":
        """Create a sampler from this geometric distribution.

        Args:
            seed (Optional[int]): Used to set seed on random number generator.

        Returns:
            GeometricSampler configured from this distribution.

        """
        return GeometricSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "GeometricEstimator":
        """Create an estimator for a geometric distribution.

        Args:
            pseudo_count (Optional[float]): Regularize empirical summary statistics.

        Returns:
            GeometricEstimator configured with this distribution's prior and name.

        """
        if pseudo_count is None:
            return GeometricEstimator(name=self.name, prior=self.prior)
        else:
            return GeometricEstimator(pseudo_count=pseudo_count, suff_stat=self.p, name=self.name, prior=self.prior)

    def dist_to_encoder(self) -> "GeometricDataEncoder":
        """Return the encoder for geometric observations."""
        return GeometricDataEncoder()

    def enumerator(self) -> "GeometricEnumerator":
        """Returns GeometricEnumerator iterating the support {1, 2, ...} in descending probability order."""
        return GeometricEnumerator(self)

    def quantized_index(self, max_bits: float, bin_width_bits: float = 1.0) -> QuantizedEnumerationIndex:
        """Build a bounded bit-quantized index directly from the geometric tail formula."""
        if max_bits < 0:
            raise ValueError("max_bits must be non-negative.")
        if bin_width_bits <= 0:
            raise ValueError("bin_width_bits must be positive.")

        if self.log_p == -np.inf:
            return QuantizedEnumerationIndex.from_items(
                [], max_bits=max_bits, bin_width_bits=bin_width_bits, truncated=False
            )

        if self.log_1p == -np.inf:
            return QuantizedEnumerationIndex.from_items(
                [(1, float(self.log_p))],
                max_bits=max_bits,
                bin_width_bits=bin_width_bits,
                sorted_items=True,
                truncated=False,
            )

        limit_nats = float(max_bits) * math.log(2.0)
        max_offset = int(math.floor((limit_nats + float(self.log_p)) / (-float(self.log_1p)) + 1.0e-12))
        if max_offset < 0:
            items = []
        else:
            items = [(x, float((x - 1) * self.log_1p + self.log_p)) for x in range(1, max_offset + 2)]

        return QuantizedEnumerationIndex.from_items(
            items, max_bits=max_bits, bin_width_bits=bin_width_bits, sorted_items=True, truncated=True
        )

    def quantized_multi_cross_index(self, others, max_bits, bin_width_bits: float = 1.0) -> QuantizedCrossIndex:
        """Build an exact aligned cross-bin view over bounded geometric prefixes."""
        dists = [self] + list(others)
        if any(not isinstance(dist, GeometricDistribution) for dist in dists):
            return super().quantized_multi_cross_index(others, max_bits=max_bits, bin_width_bits=bin_width_bits)
        if isinstance(max_bits, np.ndarray):
            max_bits_tuple = tuple(float(x) for x in max_bits.tolist())
        elif isinstance(max_bits, (list, tuple)):
            max_bits_tuple = tuple(float(x) for x in max_bits)
        else:
            max_bits_tuple = tuple([float(max_bits)] * len(dists))
        if len(max_bits_tuple) != len(dists):
            raise ValueError("max_bits length must match the number of distributions.")

        def max_value(dist: "GeometricDistribution", bit_bound: float) -> int:
            if bit_bound < 0.0 or dist.log_p == -np.inf:
                return 0
            if dist.log_1p == -np.inf:
                return 1 if -float(dist.log_p) / math.log(2.0) <= bit_bound + 1.0e-12 else 0
            limit_nats = float(bit_bound) * math.log(2.0)
            max_offset = int(math.floor((limit_nats + float(dist.log_p)) / (-float(dist.log_1p)) + 1.0e-12))
            return max(0, max_offset) + 1 if max_offset >= 0 else 0

        hi = max(max_value(dist, bit_bound) for dist, bit_bound in zip(dists, max_bits_tuple))
        items = [(value, tuple(float(dist.log_density(value)) for dist in dists)) for value in range(1, hi + 1)]
        return QuantizedCrossIndex.from_items(
            items, max_bits=max_bits_tuple, bin_width_bits=bin_width_bits, truncated=True
        )

    def quantized_cross_index(self, other, max_bits, bin_width_bits: float = 1.0) -> QuantizedCrossIndex:
        """Build an exact aligned cross-bin view over two bounded geometric prefixes."""
        return self.quantized_multi_cross_index([other], max_bits=max_bits, bin_width_bits=bin_width_bits)


class GeometricEnumerator(DistributionEnumerator):
    """Enumerate geometric support values in descending probability order."""

    def __init__(self, dist: GeometricDistribution) -> None:
        """Enumerates the support {1, 2, 3, ...} of a GeometricDistribution.

        The geometric pmf is strictly decreasing in x, so the natural order is already
        the descending-probability order. The iterator is infinite for p < 1.

        Args:
            dist (GeometricDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        self._x = 1

    def __next__(self) -> tuple[int, float]:
        x = self._x
        lp = (x - 1) * self.dist.log_1p + self.dist.log_p
        if lp == -np.inf:
            raise StopIteration
        self._x += 1
        return (x, lp)


class GeometricSampler(DistributionSampler):
    """Draw independent samples from a :class:`GeometricDistribution`."""

    def __init__(self, dist: GeometricDistribution, seed: int | None = None) -> None:
        """Create a sampler for a geometric distribution.

        Args:
            dist: Distribution to sample from.
            seed: Optional random seed.

        Attributes:
            rng (RandomState): RandomState with seed set for sampling.
            dist (GeometricDistribution): GeometricDistribution to sample from.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> int | np.ndarray:
        """Generate iid samples from geometric distribution.

        Generates a single geometric sample (int) if size is None, else a numpy array of integers of length size,
        iid samples, from the geometric distribution.

        Args:
            size (Optional[int]): Number of iid samples to draw. If None, assumed to be 1.

        Returns:
            If size is None, int, else size length numpy array of ints.

        """
        return self.rng.geometric(p=self.dist.p, size=size)


class GeometricAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted count and sum statistics for Geometric estimation."""

    def __init__(self, name: str | None = None, keys: str | None = None):
        """Create an accumulator for geometric sufficient statistics.

        Args:
            name (Optional[str]): Optional accumulator name.
            keys (Optional[str]): Accumulators with the same key merge sufficient statistics.

        Attributes:
            sum (float): Aggregate weighted sum of observations.
            count (float): Aggregate sum of weighted observation count.
            name (Optional[str]): Assigned from name arg.
            key (Optional[str]): Assigned from keys arg.

        """
        self.sum = 0.0
        self.count = 0.0
        self.keys = keys
        self.name = name

    def update(self, x: int, weight: float, estimate: Optional["GeometricDistribution"]) -> None:
        """Update sufficient statistics for GeometricAccumulator with one weighted observation.

        Args:
            x (int): Positive integer observation of geometric distribution.
            weight (float): Weight for observation.
            estimate (Optional[GeometricDistribution]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None

        """
        if x >= 1:
            self.sum += x * weight
            self.count += weight

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Optional["GeometricDistribution"]) -> None:
        """Vectorized update of sufficient statistics from encoded sequence x.

        sum increased by sum of weighted observations.
        count increased by sum of weights.

        Args:
            x (ndarray): Numpy array of positive integers.
            weights (ndarray): Numpy array of positive floats.
            estimate (Optional[GeometricDistribution]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.sum += np.dot(x, weights)
        self.count += np.sum(weights)

    def initialize(self, x: int, weight: float, rng: RandomState | None) -> None:
        """Initialize sufficient statistics of GeometricAccumulator with weighted observation.

        This delegates to :meth:`update`.

        Args:
            x (int): Positive integer observation of geometric distribution.
            weight (float): Positive real-valued weight for observation x.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.update(x, weight, None)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Vectorized initialization of GeometricAccumulator sufficient statistics with weighted observations.

        This delegates to :meth:`seq_update`.

        Args:
            x (ndarray): Numpy array of positive integers.
            weights (ndarray): Numpy array of positive floats.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float]) -> "GeometricAccumulator":
        """Combine aggregated sufficient statistics with sufficient statistics of GeometricAccumulator instance.

        Input suff_stat is Tuple[float, float] with:
            suff_stat[0] (float): sum of observation weights,
            suff_stat[1] (float): weighted sum of observations.

        Args:
            suff_stat (Tuple[float, float]): See above for details.

        Returns:
            GeometricAccumulator object.

        """
        self.sum += suff_stat[1]
        self.count += suff_stat[0]

        return self

    def value(self) -> tuple[float, float]:
        """Returns sufficient statistics Tuple[float, float] of GeometricAccumulator instance."""
        return self.count, self.sum

    def from_value(self, x: tuple[float, float]) -> "GeometricAccumulator":
        """Sets GeometricAccumulator instance sufficient statistic member variables to x.

        Args:
            x (Tuple[float, float]): Sum of observations weights and sum of weighted observations.

        Returns:
            GeometricAccumulator object.

        """
        self.count = x[0]
        self.sum = x[1]

        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge sufficient statistics from ``stats_dict`` when this accumulator's key is present.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to sufficient statistics.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                x0, x1 = stats_dict[self.keys]
                self.count += x0
                self.sum += x1
                # write the POOL back: without this, the dict keeps the FIRST site's stats and
                # key_replace hands every tied site that truncated pool -- later sites' data was
                # silently discarded (order-dependent wrong fits; found by the compiler review's
                # keyed-tying probe, present in 8 families vs the combine-into-dict families)
                stats_dict[self.keys] = (self.count, self.sum)

            else:
                stats_dict[self.keys] = (self.count, self.sum)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace sufficient statistics from ``stats_dict`` when this accumulator's key is present.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to sufficient statistics.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                self.count, self.sum = stats_dict[self.keys]

    def acc_to_encoder(self) -> "GeometricDataEncoder":
        """Return the encoder associated with this accumulator."""
        return GeometricDataEncoder()


class GeometricAccumulatorFactory(StatisticAccumulatorFactory):
    """Create Geometric accumulators with shared name and merge-key metadata."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        """Create a factory for geometric accumulators.

        Args:
            name (Optional[str]): Optional name assigned to created accumulators.
            keys (Optional[str]): GeometricAccumulator objects with same key merge sufficient statistics.

        Attributes:
            name (Optional[str]): Assigned from name arg.
            keys (Optional[str]): Assigned from keys arg.

        """
        self.name = name
        self.keys = keys

    def make(self) -> "GeometricAccumulator":
        """Return GeometricAccumulator with name and keys passed."""
        return GeometricAccumulator(name=self.name, keys=self.keys)


class GeometricEstimator(ParameterEstimator):
    """Estimate Geometric success probabilities from accumulated sufficient statistics."""

    def __init__(
        self,
        pseudo_count: float | None = None,
        suff_stat: float | None = None,
        name: str | None = None,
        keys: str | None = None,
        prior: SequenceEncodableProbabilityDistribution | None = None,
    ) -> None:
        """Estimator for a geometric distribution from aggregated sufficient statistics.

        Args:
            pseudo_count (Optional[float]): Float value for re-weighting suff_stat member variable.
            suff_stat (Optional[float]): Probability of success (value between (0,1)).
            name (Optional[str]): Optional name assigned to the estimated distribution.
            keys (Optional[str]): GeometricAccumulator objects with same key merge sufficient statistics.
            prior (Optional): Conjugate Beta prior on the success probability ``p``. When present,
                ``estimate`` performs the closed-form conjugate posterior update (returning the
                posterior-mode MAP estimate and carrying the posterior forward as the fitted
                model's prior) instead of the maximum-likelihood / pseudo-count update.

        Attributes:
            pseudo_count (Optional[float]): Assigned from pseudo_count arg.
            suff_stat (Optional[float]): Assigned from suff_stat arg (corrected for [0,1] constraint).
            name (Optional[str]): Assigned from name arg.
            keys (Optional[str]): Assigned from keys arg.

        """
        self.pseudo_count = pseudo_count
        self.suff_stat = max(min(suff_stat, 1.0), 0.0) if suff_stat is not None else None
        self.keys = keys
        self.name = name
        self.prior = prior
        self.has_conj_prior = isinstance(prior, BetaDistribution)

    def accumulator_factory(self) -> "GeometricAccumulatorFactory":
        """Create a geometric accumulator factory with this estimator's name and keys."""
        return GeometricAccumulatorFactory(name=self.name, keys=self.keys)

    def model_log_density(self, model: "GeometricDistribution") -> float:
        """Log-density of the model's success probability under the Beta prior (ELBO global term)."""
        if self.has_conj_prior:
            return float(self.prior.log_density(model.p))
        return 0.0

    def _estimate_conjugate(self, suff_stat: tuple[float, float]) -> "GeometricDistribution":
        """Closed-form Beta conjugate posterior update returning the MAP estimate.

        With a Beta(a, b) prior and statistics ``(count, sum)`` the posterior is
        Beta(a + count, b + sum - count) and the returned point estimate is the posterior
        mode ``(a' - 1) / (a' + b' - 2)``, clamped to 0, 1, or 1/2 on the boundary where
        the mode is undefined; the posterior is carried forward as the fitted model's prior.
        """
        ocnt, osum = suff_stat
        old_a, old_b = self.prior.get_parameters()
        a = old_a + ocnt
        b = old_b + osum - ocnt
        if a > 1 and b > 1:
            p = (a - 1) / (a + b - 2)
        elif a <= 1 and b > 1:
            p = 0.0
        elif a > 1 and b <= 1:
            p = 1.0
        else:
            p = 0.5
        return GeometricDistribution(p, name=self.name, prior=BetaDistribution(a, b))

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float]) -> "GeometricDistribution":
        """Estimate a geometric distribution from weighted count and sum statistics.

        ``nobs`` is accepted for estimator API consistency but is not used.
        ``suff_stat`` is ``(count, sum)``. Without a pseudo-count, the estimate is
        ``p = count / sum``; with a pseudo-count, the empirical statistic is
        shrunk toward the estimator's prior statistic when available.
        """
        if self.has_conj_prior:
            return self._estimate_conjugate(suff_stat)

        if self.pseudo_count is not None and self.suff_stat is not None:
            p = (suff_stat[0] + self.pseudo_count * self.suff_stat) / (suff_stat[1] + self.pseudo_count)
        elif self.pseudo_count is not None and self.suff_stat is None:
            p = (suff_stat[0] + self.pseudo_count) / (suff_stat[1] + self.pseudo_count)
        elif suff_stat[1] == 0.0:
            p = 0.5
        else:
            p = suff_stat[0] / suff_stat[1]

        p = float(np.clip(p, 1.0e-12, 1.0 - 1.0e-12))
        return GeometricDistribution(p, name=self.name)


class GeometricDataEncoder(DataSequenceEncoder):
    """Data encoder for iid positive-integer geometric observations."""

    def __str__(self) -> str:
        """Return the geometric encoder's display name."""
        return "GeometricDataEncoder"

    def __eq__(self, other) -> bool:
        """Return true when ``other`` is an equivalent geometric data encoder.

        Args:
            other (object): Object to be compared to self.

        Returns:
            True if other is GeometricDataEncoder instance, else False.

        """
        return isinstance(other, GeometricDataEncoder)

    def seq_encode(self, x: Sequence[int] | np.ndarray) -> np.ndarray:
        """Encode iid sequence of geometric observations for vectorized "seq_" function calls.

        Note: x should be list of numpy array of positive integers.

        Args:
            x (Union[Sequence[int], np.ndarray]): Positive integer geometric observations.

        Returns:
            Numpy array of positive integers.

        """
        rv = np.asarray(x, dtype=np.float64)
        if np.any(rv < 1) or np.any(np.isnan(rv)) or np.any(np.floor(rv) != rv):
            raise ValueError("GeometricDistribution requires positive integer values for x.")
        else:
            return rv
