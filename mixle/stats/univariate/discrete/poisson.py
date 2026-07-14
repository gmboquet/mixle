"""Poisson distributions, estimators, accumulators, samplers, and encoders.

For nonnegative integer counts, ``PoissonDistribution(lam)`` has rate
``lam > 0`` and log-density:

    log(p_mat(x_mat=x; lam)) = x*log(lam) - log(x!) - lam,

Values outside ``{0, 1, 2, ...}`` score ``-inf``.

Reference: Johnson, Kemp & Kotz, *Univariate Discrete Distributions* (3rd ed., Wiley, 2005).
"""

import math
from collections.abc import Sequence
from math import log
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
from mixle.stats.univariate.continuous.gamma import GammaDistribution
from mixle.utils.special import digamma
from mixle.utils.vector import gammaln


def _fisher_mean_var(dist):
    lam = float(dist.lam)
    return lam, lam


def _fisher_encoded(enc_data):
    return np.asarray(enc_data[0], dtype=np.float64)


class PoissonDistribution(SequenceEncodableProbabilityDistribution):
    """Poisson distribution over non-negative integer counts with rate ``lam``."""

    @classmethod
    def compute_capabilities(cls):
        """Declare backend support for generated Poisson density kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch", "jax"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the generated-compute declaration for the Poisson distribution."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ExponentialFamilySpec,
            ParameterSpec,
            StatisticSpec,
        )

        return DistributionDeclaration(
            name="poisson",
            distribution_type=cls,
            parameters=(ParameterSpec("lam", constraint="positive"),),
            statistics=(StatisticSpec("count"), StatisticSpec("sum")),
            support="non_negative_integer",
            exponential_family=ExponentialFamilySpec(
                sufficient_statistics=cls.exp_family_sufficient_statistics,
                natural_parameters=cls.exp_family_natural_parameters,
                log_partition=cls.exp_family_log_partition,
                base_measure=cls.exp_family_base_measure,
                legacy_sufficient_statistics=cls.exp_family_legacy_sufficient_statistics,
            ),
        )

    @staticmethod
    def exp_family_sufficient_statistics(x: tuple[Any, Any], engine: Any) -> tuple[Any, ...]:
        """Return Poisson sufficient statistics for generated scoring."""
        return (engine.asarray(x[0]),)

    @staticmethod
    def exp_family_legacy_sufficient_statistics(
        x: tuple[Any, Any], params: dict[str, Any], engine: Any
    ) -> tuple[Any, ...]:
        """Return per-row Poisson sufficient statistics in accumulator order."""
        vals = engine.asarray(x[0])
        return vals * 0.0 + engine.asarray(1.0), vals

    @staticmethod
    def exp_family_natural_parameters(params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return Poisson natural parameters for generated scoring."""
        return (engine.log(params["lam"]),)

    @staticmethod
    def exp_family_log_partition(params: dict[str, Any], engine: Any) -> Any:
        """Return Poisson log partition for generated scoring."""
        return params["lam"]

    @staticmethod
    def exp_family_base_measure(x: tuple[Any, Any], engine: Any) -> Any:
        """Return Poisson base measure for generated scoring."""
        vals = engine.asarray(x[0])
        log_fact = engine.asarray(x[1])
        good = (vals >= 0) & (engine.floor(vals) == vals)
        return engine.where(good, -log_fact, engine.asarray(-np.inf))

    def __init__(
        self,
        lam: float,
        name: str | None = None,
        prior: SequenceEncodableProbabilityDistribution | None = None,
    ) -> None:
        """Create a Poisson distribution.

        Args:
            lam: Positive finite rate and mean.
            name: Optional diagnostic name.
            prior (Optional): Conjugate parameter prior over the rate ``lam``. A
                :class:`~mixle.stats.univariate.continuous.gamma.GammaDistribution` enables the
                Bayesian/variational machinery (``expected_log_density`` and the
                conjugate posterior update); ``None`` (default) is a plain point model.

        Attributes:
            lam: Rate and mean of the Poisson distribution.
            name: Optional diagnostic name.
            log_lambda: Log rate used by scoring.
        """
        if lam <= 0.0 or not np.isfinite(lam):
            raise ValueError("PoissonDistribution requires lam > 0.")
        self.lam = float(lam)
        self.log_lambda = log(self.lam)
        self.name = name
        self.set_prior(prior)

    def __str__(self) -> str:
        """Return a readable distribution summary."""
        return "PoissonDistribution(%s, name=%s)" % (repr(self.lam), repr(self.name))

    def set_prior(self, prior: SequenceEncodableProbabilityDistribution | None) -> None:
        """Attach a parameter prior and cache the conjugate Gamma expectations.

        With a Gamma(k, theta) prior over the rate ``lam`` this caches (k, theta) so that
        ``expected_log_density(x) = (psi(k) + ln theta)*x - k*theta - gammaln(x+1)`` (the
        VB E-step term using E[ln lam] = psi(k) + ln theta and E[lam] = k*theta). Any other
        prior (including ``None``) leaves the distribution a plain point model.
        """
        self.prior = prior
        if isinstance(prior, GammaDistribution):
            self.conj_prior_params = prior.get_parameters()
            self.has_conj_prior = True
        else:
            self.conj_prior_params = None
            self.has_conj_prior = False

    def expected_log_density(self, x: float) -> float:
        """Variational expectation E_q[log p(x | lam)] under the Gamma prior.

        Falls back to the plug-in ``log_density(x)`` when no conjugate prior is attached.
        """
        if self.has_conj_prior:
            k, theta = self.conj_prior_params
            return (digamma(k) + np.log(theta)) * x - k * theta - gammaln(x + 1.0)
        return self.log_density(x)

    def seq_expected_log_density(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Vectorized ``expected_log_density`` over sequence-encoded observations."""
        if not self.has_conj_prior:
            return self.seq_log_density(x)
        vals, log_fact = x
        k, theta = self.conj_prior_params
        rv = (digamma(k) + np.log(theta)) * vals - k * theta - log_fact
        good = np.isfinite(vals) & (vals >= 0) & (np.floor(vals) == vals)
        return np.where(good, rv, -np.inf)

    def density(self, x: int) -> float:
        """Evaluate the density of Poisson distribution at observation x.

        Calls np.exp(log_density(x)). See log_density() for details.

        Args:
            x (int): Must be a non-negative integer value (0,1,2,....).

        Returns:
            Density of Poisson distribution evaluated at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        """Log-density of Poisson distribution evaluated at x.

        Log-density given by,
            log(p_mat(x_mat=x; lam) = x*log(lam) - log(x!) - lam, for x in {0,1,2,...}
        and -np.inf else.

        Note: log(Gamma(x+1.0)) = log(x!), where Gamma is the gamma function.

        Args:
            x (int): Must be a non-negative integer value (0,1,2,....).

        Returns:
            Log-density of Poisson distribution evaluated at x.

        """
        try:
            xx = float(x)
        except Exception:  # noqa: BLE001
            return -np.inf
        if not np.isfinite(xx) or xx < 0 or np.floor(xx) != xx:
            return -np.inf
        else:
            return xx * self.log_lambda - gammaln(xx + 1.0) - self.lam

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Vectorized log-density evaluated on sequence encoded x.

        Arg value x (Tuple[np.ndarray[int], np.ndarray[float]]) is seq_encoded Poisson data from
        PoissonDataEncoder.seq_encode(), containing
            x[0] (np.ndarray[int]): Non-negative integer valued Poisson iid observations,
            x[1] (np.ndarray[float]): np.log(Gamma(x[0]+1.0)), Gamma is the gamma function.

        Args:
            x: See above for details.

        Returns:
            Numpy array of log-density evaluated at each encoded observation value x.

        """
        vals, log_fact = x
        # out-of-place arithmetic keeps the autograd graph intact under torch
        rv = vals * self.log_lambda
        rv = rv - log_fact
        rv = rv - self.lam
        good = np.isfinite(vals) & (vals >= 0) & (np.floor(vals) == vals)
        rv = np.where(good, rv, -np.inf)
        return rv

    @staticmethod
    def backend_log_density_from_params(vals: Any, log_fact: Any, lam: Any, engine: Any) -> Any:
        """Engine-neutral Poisson log-density from explicit parameters."""
        rv = vals * engine.log(lam) - log_fact - lam
        good = (vals >= 0) & (engine.floor(vals) == vals)
        return engine.where(good, rv, engine.asarray(-np.inf))

    def backend_seq_log_density(self, x: tuple[Any, Any], engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        vals = engine.asarray(x[0])
        log_fact = engine.asarray(x[1])
        lam = engine.asarray(self.lam)
        return self.backend_log_density_from_params(vals, log_fact, lam, engine)

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["PoissonDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked Poisson parameters for a homogeneous mixture kernel."""
        return {"lam": engine.asarray([d.lam for d in dists])}

    @classmethod
    def backend_stacked_log_density(cls, x: tuple[Any, Any], params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of Poisson log densities."""
        vals = engine.asarray(x[0])
        log_fact = engine.asarray(x[1])
        return cls.backend_log_density_from_params(vals[:, None], log_fact[:, None], params["lam"][None, :], engine)

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: tuple[Any, Any], weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any]:
        """Return stacked Poisson sufficient statistics using engine-resident arrays."""
        vals = engine.asarray(x[0])
        ww = engine.asarray(weights)
        return engine.sum(ww, axis=0), engine.sum(ww * vals[:, None], axis=0)

    def to_fisher(self, **kwargs):
        """Return the Poisson's count-family Fisher view."""
        from mixle.inference.fisher import CountFisherView, _count_data

        return CountFisherView(self, _fisher_mean_var, _count_data, _fisher_encoded)

    def mean(self) -> float:
        """Mean E[X] of the distribution."""
        return float(self.lam)

    def variance(self) -> float:
        """Variance Var[X] of the distribution."""
        return float(self.lam)

    def cdf(self, x: float) -> float:
        """Cumulative distribution function P(X <= x) = Q(floor(x)+1, lam)."""
        import math

        from scipy.special import gammaincc

        k = math.floor(float(x))
        return float(gammaincc(k + 1, self.lam)) if k >= 0 else 0.0

    def skewness(self) -> float:
        """Skewness 1/sqrt(lambda)."""
        import math

        return float(1.0 / math.sqrt(self.lam))

    def kurtosis(self) -> float:
        """Excess kurtosis 1/lambda."""
        return float(1.0 / self.lam)

    def entropy(self) -> float:
        """Shannon entropy in nats, by exact summation of the standard series.

        The Poisson entropy ``-sum_k p_k log p_k`` has no closed form; the series is summed over
        the effective support ``k <= lam + 40 sqrt(lam) + 40``, beyond which the remaining terms
        are far below double rounding (the tail mass decays super-geometrically).
        """
        from scipy.special import gammaln

        kmax = int(math.ceil(self.lam + 40.0 * math.sqrt(self.lam) + 40.0))
        k = np.arange(kmax + 1, dtype=np.float64)
        log_p = k * math.log(self.lam) - self.lam - gammaln(k + 1.0)
        return float(-np.sum(np.exp(log_p) * log_p))

    def quantile(self, q: float) -> float:
        """Inverse CDF F^{-1}(q) (via scipy poisson)."""
        from scipy.stats import poisson

        return float(poisson.ppf(float(q), self.lam))

    def mode(self) -> float:
        """Mode floor(lambda)."""
        import math

        return float(math.floor(self.lam))

    def sampler(self, seed: int | None = None) -> "PoissonSampler":
        """Return a sampler for iid draws from this distribution.

        Args:
            seed: Optional seed for the sampler's random state.

        Returns:
            A configured ``PoissonSampler``.

        """
        return PoissonSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "PoissonEstimator":
        """Return an estimator initialized from this distribution's shape.

        Args:
            pseudo_count: Optional smoothing count applied to the current rate.

        Returns:
            A ``PoissonEstimator``.

        """
        if pseudo_count is None:
            return PoissonEstimator(name=self.name, prior=self.prior)
        else:
            return PoissonEstimator(pseudo_count=pseudo_count, suff_stat=self.lam, name=self.name, prior=self.prior)

    def dist_to_encoder(self) -> "PoissonDataEncoder":
        """Return an encoder for iid Poisson observations."""
        return PoissonDataEncoder()

    def enumerator(self) -> "PoissonEnumerator":
        """Return an enumerator over nonnegative counts in descending probability order."""
        return PoissonEnumerator(self)

    def quantized_index(self, max_bits: float, bin_width_bits: float = 1.0) -> QuantizedEnumerationIndex:
        """Build a bounded bit-quantized index by walking the Poisson mode outward."""
        if max_bits < 0:
            raise ValueError("max_bits must be non-negative.")
        if bin_width_bits <= 0:
            raise ValueError("bin_width_bits must be positive.")

        mode = int(np.floor(self.lam))
        left = mode
        right = mode + 1
        lp_left = self.log_density(left)
        lp_right = self.log_density(right)
        limit_lp = -(float(max_bits) + 1.0e-12) * math.log(2.0)
        items: list[tuple[int, float]] = []

        while (left >= 0 and lp_left >= limit_lp) or lp_right >= limit_lp:
            if left >= 0 and lp_left >= lp_right:
                items.append((left, float(lp_left)))
                left -= 1
                lp_left = self.log_density(left) if left >= 0 else -np.inf
            else:
                items.append((right, float(lp_right)))
                right += 1
                lp_right = self.log_density(right)

        return QuantizedEnumerationIndex.from_items(
            items, max_bits=max_bits, bin_width_bits=bin_width_bits, sorted_items=True, truncated=True
        )

    def quantized_multi_cross_index(self, others, max_bits, bin_width_bits: float = 1.0) -> QuantizedCrossIndex:
        """Build an aligned cross-bin view over bounded Poisson high-mass regions."""
        dists = [self] + list(others)
        if any(not isinstance(dist, PoissonDistribution) for dist in dists):
            return super().quantized_multi_cross_index(others, max_bits=max_bits, bin_width_bits=bin_width_bits)
        if isinstance(max_bits, np.ndarray):
            max_bits_tuple = tuple(float(x) for x in max_bits.tolist())
        elif isinstance(max_bits, (list, tuple)):
            max_bits_tuple = tuple(float(x) for x in max_bits)
        else:
            max_bits_tuple = tuple([float(max_bits)] * len(dists))
        if len(max_bits_tuple) != len(dists):
            raise ValueError("max_bits length must match the number of distributions.")

        values = set()
        for dist, bit_bound in zip(dists, max_bits_tuple):
            if bit_bound < 0.0:
                continue
            index = dist.quantized_index(max_bits=bit_bound, bin_width_bits=bin_width_bits)
            values.update(value for value, _ in index.iter_from())

        items = [(value, tuple(float(dist.log_density(value)) for dist in dists)) for value in sorted(values)]
        return QuantizedCrossIndex.from_items(
            items, max_bits=max_bits_tuple, bin_width_bits=bin_width_bits, truncated=True
        )

    def quantized_cross_index(self, other, max_bits, bin_width_bits: float = 1.0) -> QuantizedCrossIndex:
        """Build an aligned cross-bin view over two bounded Poisson high-mass regions."""
        return self.quantized_multi_cross_index([other], max_bits=max_bits, bin_width_bits=bin_width_bits)


class PoissonEnumerator(DistributionEnumerator):
    """Enumerate Poisson support values in descending probability order."""

    def __init__(self, dist: PoissonDistribution) -> None:
        """Enumerates the support {0, 1, 2, ...} of a PoissonDistribution.

        The Poisson pmf is unimodal with mode floor(lam), so enumeration starts at the
        mode and walks outward with two pointers (left bounded at 0, right unbounded),
        emitting the larger side each step. The iterator is infinite.

        Args:
            dist (PoissonDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        mode = int(np.floor(dist.lam))
        self._left = mode - 1
        self._right = mode + 1
        self._lp_left = dist.log_density(self._left) if self._left >= 0 else -np.inf
        self._lp_right = dist.log_density(self._right)
        self._head: tuple[int, float] | None = (mode, dist.log_density(mode))

    def __next__(self) -> tuple[int, float]:
        if self._head is not None:
            rv = self._head
            self._head = None
            return rv
        if self._lp_left >= self._lp_right and self._left >= 0:
            rv = (self._left, self._lp_left)
            self._left -= 1
            self._lp_left = self.dist.log_density(self._left) if self._left >= 0 else -np.inf
        else:
            rv = (self._right, self._lp_right)
            self._right += 1
            self._lp_right = self.dist.log_density(self._right)
        return rv


class PoissonSampler(DistributionSampler):
    """Draw independent samples from a :class:`PoissonDistribution`."""

    def __init__(self, dist: "PoissonDistribution", seed: int | None = None) -> None:
        """Create a sampler for a Poisson distribution.

        Args:
            dist: Distribution to sample from.
            seed: Optional random seed.

        Attributes:
            rng: Random state used for sampling.
            dist: Distribution to sample from.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> int | np.ndarray:
        """Draw iid samples from the Poisson distribution.

        Args:
            size: Number of iid samples to draw. ``None`` returns a scalar sample.

        Returns:
            A scalar draw when ``size`` is ``None``; otherwise an array of draws.

        """
        return self.rng.poisson(lam=self.dist.lam, size=size)


class PoissonAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted count and sum statistics for Poisson estimation."""

    def __init__(self, keys: str | None = None) -> None:
        """Create an accumulator for Poisson sufficient statistics.

        Args:
            keys: Optional key for merging sufficient statistics.

        Attributes:
            sum: Weighted sum of observations.
            count: Sum of observation weights.
            keys: Optional sufficient-statistic key.

        """
        self.sum = 0.0
        self.count = 0.0
        self.keys = keys

    def initialize(self, x: int, weight: float, rng: np.random.RandomState | None = None) -> None:
        """Initialize sufficient statistics with one weighted observation.

        This method delegates to ``update``.

        Args:
            x: Observation from a Poisson distribution.
            weight: Observation weight.
            rng: Unused; accepted for the accumulator interface.

        """
        self.update(x, weight, None)

    def seq_initialize(
        self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, rng: np.random.RandomState | None = None
    ) -> None:
        """Vectorized initialization of PoissonAccumulator sufficient statistics with weighted observations.

        This delegates to :meth:`seq_update`.

        Arg value x (Tuple[np.ndarray[int], np.ndarray[float]]) is seq_encoded Poisson data from
        PoissonDataEncoder.seq_encode(), containing
            x[0] (np.ndarray[int]): Non-negative integer valued Poisson iid observations,
            x[1] (np.ndarray[float]): np.log(Gamma(x[0]+1.0)), Gamma is the gamma function.

        Args:
            x: See above for details.
            weights (ndarray): Numpy array of positive floats.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.seq_update(x, weights, None)

    def update(self, x: int, weight: float, estimate: Optional["PoissonDistribution"] = None) -> None:
        """Update sufficient statistics for PoissonAccumulator with one weighted observation.

        Args:
            x (int): Observation from Poisson distribution.
            weight (float): Weight for observation.
            estimate (Optional[PoissonDistribution]): Kept for consistency with
                SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.sum += x * weight
        self.count += weight

    def seq_update(
        self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, estimate: Optional["PoissonDistribution"] = None
    ) -> None:
        """Vectorized update of PoissonAccumulator sufficient statistics with weighted observations.

        Arg value x (Tuple[np.ndarray[int], np.ndarray[float]]) is seq_encoded Poisson data from
        PoissonDataEncoder.seq_encode(), containing
            x[0] (np.ndarray[int]): Non-negative integer valued Poisson iid observations,
            x[1] (np.ndarray[float]): np.log(Gamma(x[0]+1.0)), Gamma is the gamma function.

        Args:
            x: See above for details.
            weights (ndarray): Numpy array of positive floats.
            estimate (Optional[PoissonDistribution]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.sum += np.dot(x[0], weights)
        self.count += weights.sum()

    def combine(self, suff_stat: tuple[float, float]) -> "PoissonAccumulator":
        """Merge aggregated Poisson sufficient statistics into this accumulator.

        The tuple is interpreted as ``(count, sum)``.

        Args:
            suff_stat: Aggregated Poisson sufficient statistics.

        Returns:
            This accumulator.

        """
        self.sum += suff_stat[1]
        self.count += suff_stat[0]

        return self

    def value(self) -> tuple[float, float]:
        """Return sufficient statistics as ``(count, sum)``."""
        return self.count, self.sum

    def from_value(self, x: tuple[float, float]) -> "PoissonAccumulator":
        """Replace this accumulator's sufficient statistics.

        Args:
            x: Aggregated Poisson sufficient statistics as ``(count, sum)``.

        Returns:
            This accumulator.

        """
        self.count = x[0]
        self.sum = x[1]

        return self

    def acc_to_encoder(self) -> "PoissonDataEncoder":
        """Return an encoder compatible with Poisson observations."""
        return PoissonDataEncoder()


class PoissonAccumulatorFactory(StatisticAccumulatorFactory):
    """Create Poisson accumulators with a shared optional merge key."""

    def __init__(self, keys: str | None = None) -> None:
        """Create an accumulator factory.

        Args:
            keys: Optional key for merging sufficient statistics.

        Attributes:
             keys: Optional sufficient-statistic key.

        """
        self.keys = keys

    def make(self) -> "PoissonAccumulator":
        """Return a fresh Poisson accumulator."""
        return PoissonAccumulator(keys=self.keys)


class PoissonEstimator(ParameterEstimator):
    """Estimate Poisson rate parameters from accumulated sufficient statistics."""

    def __init__(
        self,
        pseudo_count: float | None = None,
        suff_stat: float | None = None,
        name: str | None = None,
        keys: str | None = None,
        prior: SequenceEncodableProbabilityDistribution | None = None,
    ) -> None:
        """Create an estimator for Poisson sufficient statistics.

        Args:
            pseudo_count: Optional nonnegative smoothing count.
            suff_stat: Optional prior rate used for smoothing.
            name: Optional diagnostic name.
            keys: Optional key for merging sufficient statistics.
            prior (Optional): Conjugate Gamma prior over the rate ``lam``. When present,
                ``estimate`` performs the closed-form conjugate posterior update (returning the
                Gamma posterior mode and carrying the posterior forward as the fitted model's
                prior) instead of the maximum-likelihood / pseudo-count update.

        Attributes:
            pseudo_count: Smoothing count.
            suff_stat: Prior rate used for smoothing.
            name: Optional diagnostic name.
            keys: Optional sufficient-statistic key.

        """
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.name = name
        self.keys = keys
        self.prior = prior
        self.has_conj_prior = isinstance(prior, GammaDistribution)

    def accumulator_factory(self) -> "PoissonAccumulatorFactory":
        """Return an accumulator factory matching this estimator."""
        return PoissonAccumulatorFactory(self.keys)

    def model_log_density(self, model: "PoissonDistribution") -> float:
        """Log-density of the model's rate under the Gamma prior (ELBO global term)."""
        if self.has_conj_prior:
            return float(self.prior.log_density(model.lam))
        return 0.0

    def _estimate_conjugate(self, suff_stat: tuple[float, float]) -> "PoissonDistribution":
        """Closed-form Gamma conjugate posterior update returning the posterior-mode estimate."""
        nobs, psum = suff_stat
        k, theta = self.prior.get_parameters()

        new_k = k + psum
        new_theta = theta / (nobs * theta + 1.0)

        # posterior mode of Gamma(k, theta) is (k-1)*theta for k >= 1; fall back to the
        # posterior mean when the mode is at the boundary
        if new_k >= 1.0:
            posterior_mode = (new_k - 1.0) * new_theta
        else:
            posterior_mode = new_k * new_theta

        posterior_mode = max(posterior_mode, 1.0e-128)

        return PoissonDistribution(posterior_mode, name=self.name, prior=GammaDistribution(new_k, new_theta))

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float]) -> "PoissonDistribution":
        """Estimate a Poisson distribution from aggregated sufficient statistics.

        The tuple is interpreted as ``(count, sum)``.

        Args:
            nobs: Unused; accepted for the ``ParameterEstimator`` interface.
            suff_stat: Aggregated Poisson sufficient statistics.

        Returns:
            A fitted Poisson distribution.

        """
        if self.has_conj_prior:
            return self._estimate_conjugate(suff_stat)

        nobs, psum = suff_stat

        if self.pseudo_count is not None and self.suff_stat is not None:
            lam = (psum + self.suff_stat * self.pseudo_count) / (nobs + self.pseudo_count)
            return PoissonDistribution(max(float(lam), 1.0e-12), name=self.name)
        elif nobs == 0.0:
            return PoissonDistribution(1.0, name=self.name)
        else:
            return PoissonDistribution(max(float(psum / nobs), 1.0e-12), name=self.name)


class PoissonDataEncoder(DataSequenceEncoder):
    """Encode iid non-negative integer Poisson observations with log-factorials."""

    def __str__(self) -> str:
        """Return a readable encoder summary."""
        return "PoissonDataEncoder"

    def __eq__(self, other) -> bool:
        """Return whether ``other`` is a Poisson data encoder.

        Args:
            other: Object to compare.

        Returns:
            ``True`` when ``other`` is also a ``PoissonDataEncoder``.

        """
        return isinstance(other, PoissonDataEncoder)

    def seq_encode(self, x: np.ndarray | Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Encode iid sequence of Poisson observations for vectorized "seq_" function calls.

        Data type must be int. Values must be non-negative integers.
        Returns the integer observations and ``log(Gamma(x + 1))`` values used by the vectorized scorer.

        Args:
            x (Union[np.ndarray, Sequence[int]]): Sequence of iid non-negative integers valued Poisson observations.

        Returns:
            Tuple[ndarray[int], ndarray[float]].

        """
        rv1 = np.asarray(x)

        if np.any(rv1 < 0) or np.any(np.isnan(rv1)) or np.any(np.floor(rv1) != rv1):
            raise ValueError("Poisson requires non-negative integer values of x.")
        else:
            rv2 = gammaln(rv1 + 1.0)
            return rv1, rv2
