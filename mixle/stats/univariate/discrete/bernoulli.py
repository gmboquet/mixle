"""Bernoulli distributions over binary outcomes.

Data type: bool or values in {0, 1}. The distribution has success
probability p and log-density log(p) for True/1 and log(1-p) for False/0.


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
from mixle.stats.univariate.continuous.beta import BetaDistribution
from mixle.utils.special import digamma


def _fisher_mean_var(dist):
    p = float(dist.p)
    return p, p * (1.0 - p)


class BernoulliDistribution(SequenceEncodableProbabilityDistribution):
    """Bernoulli distribution over {False, True} with success probability p."""

    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated Bernoulli kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch", "jax"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for Bernoulli distributions."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ExponentialFamilySpec,
            ParameterSpec,
            StatisticSpec,
        )

        return DistributionDeclaration(
            name="bernoulli",
            distribution_type=cls,
            parameters=(ParameterSpec("p", constraint="unit_interval"),),
            statistics=(StatisticSpec("count"), StatisticSpec("sum")),
            support="boolean",
            exponential_family=ExponentialFamilySpec(
                sufficient_statistics=cls.exp_family_sufficient_statistics,
                natural_parameters=cls.exp_family_natural_parameters,
                log_partition=cls.exp_family_log_partition,
                legacy_sufficient_statistics=cls.exp_family_legacy_sufficient_statistics,
            ),
        )

    @staticmethod
    def exp_family_sufficient_statistics(x: Any, engine: Any) -> tuple[Any, ...]:
        """Return Bernoulli sufficient statistics for generated scoring."""
        return (engine.asarray(x) * engine.asarray(1.0),)

    @staticmethod
    def exp_family_legacy_sufficient_statistics(x: Any, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return per-row Bernoulli sufficient statistics in accumulator order."""
        xx = engine.asarray(x) * engine.asarray(1.0)
        return xx * 0.0 + engine.asarray(1.0), xx

    @staticmethod
    def exp_family_natural_parameters(params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return Bernoulli natural parameters for generated scoring."""
        p = params["p"]
        return (engine.log(p) - engine.log(engine.asarray(1.0) - p),)

    @staticmethod
    def exp_family_log_partition(params: dict[str, Any], engine: Any) -> Any:
        """Return Bernoulli log partition for generated scoring."""
        p = params["p"]
        return -engine.log(engine.asarray(1.0) - p)

    def __init__(
        self,
        p: float,
        name: str | None = None,
        keys: str | None = None,
        prior: SequenceEncodableProbabilityDistribution | None = None,
    ) -> None:
        if p <= 0.0 or p >= 1.0:
            raise ValueError("BernoulliDistribution requires p in (0, 1).")
        self.p = float(p)
        self.log_p = math.log(self.p)
        self.log_1p = math.log1p(-self.p)
        self.name = name
        self.keys = keys
        self.set_prior(prior)

    def __str__(self) -> str:
        return "BernoulliDistribution(%s, name=%s, keys=%s)" % (repr(self.p), repr(self.name), repr(self.keys))

    def set_prior(self, prior: SequenceEncodableProbabilityDistribution | None) -> None:
        """Attach a Beta parameter prior and precompute conjugate-prior expectations.

        With a Beta(a, b) prior on the success probability ``p`` this caches the
        digamma terms so that ``expected_log_density`` evaluates the variational Bayes
        expectation ``E_q[log p(x | p)]`` via ``E[log p] = digamma(a) - digamma(a+b)``
        and ``E[log(1-p)] = digamma(b) - digamma(a+b)``. Any other prior (including
        ``None``) leaves the distribution a plain point model.
        """
        self.prior = prior
        if isinstance(prior, BetaDistribution):
            a, b = prior.get_parameters()
            self.conj_prior_params = (digamma(a), digamma(b), digamma(a + b))
            self.has_conj_prior = True
        else:
            self.conj_prior_params = None
            self.has_conj_prior = False

    def expected_log_density(self, x: bool | int) -> float:
        """Variational expectation ``E_q[log p(x | p)]`` under the Beta prior.

        Falls back to the plug-in ``log_density(x)`` when no conjugate prior is attached.
        """
        if self.has_conj_prior:
            xx = self._as_bool(x)
            if xx is None:
                return -np.inf
            da, db, dab = self.conj_prior_params
            return da - dab if xx else db - dab
        return self.log_density(x)

    def seq_expected_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized ``expected_log_density`` over sequence-encoded observations."""
        if self.has_conj_prior:
            da, db, dab = self.conj_prior_params
            return np.where(x, da - dab, db - dab)
        return self.seq_log_density(x)

    @staticmethod
    def _as_bool(x: Any) -> bool | None:
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        try:
            if x == 1:
                return True
            if x == 0:
                return False
        except Exception:  # noqa: BLE001
            return None
        return None

    def density(self, x: bool | int) -> float:
        """Return the probability density or mass at a single observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: bool | int) -> float:
        """Return the log-density or log-mass at a single observation."""
        xx = self._as_bool(x)
        if xx is None:
            return -np.inf
        return self.log_p if xx else self.log_1p

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        return np.where(x, self.log_p, self.log_1p)

    @staticmethod
    def backend_log_density_from_params(x: Any, p: Any, engine: Any) -> Any:
        """Engine-neutral Bernoulli log-mass from explicit parameters."""
        return engine.where(x >= 0.5, engine.log(p), engine.log(engine.asarray(1.0) - p))

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        xx = engine.asarray(x)
        p = engine.asarray(self.p)
        return self.backend_log_density_from_params(xx, p, engine)

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["BernoulliDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked Bernoulli parameters for a homogeneous mixture kernel."""
        return {"p": engine.asarray([d.p for d in dists])}

    @classmethod
    def backend_stacked_log_density(cls, x: Any, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of Bernoulli log masses."""
        xx = engine.asarray(x)
        return cls.backend_log_density_from_params(xx[:, None], params["p"][None, :], engine)

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: Any, weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any]:
        """Return stacked Bernoulli sufficient statistics using engine-resident arrays."""
        xx = engine.asarray(x)
        ww = engine.asarray(weights)
        return engine.sum(ww, axis=0), engine.sum(ww * xx[:, None], axis=0)

    def support_size(self) -> int:
        """The two outcomes {0, 1}."""
        return 2

    def to_fisher(self, **kwargs):
        """Return the Bernoulli's count-family Fisher view."""
        from mixle.inference.fisher import CountFisherView, _count_data, _identity_encoded

        return CountFisherView(self, _fisher_mean_var, _count_data, _identity_encoded)

    def mean(self) -> float:
        """Mean E[X] of the distribution."""
        return float(self.p)

    def variance(self) -> float:
        """Variance Var[X] of the distribution."""
        return float(self.p * (1.0 - self.p))

    def cdf(self, x: float) -> float:
        """Cumulative distribution function P(X <= x) over {0, 1}."""
        xv = float(x)
        if xv < 0.0:
            return 0.0
        return float(1.0 - self.p) if xv < 1.0 else 1.0

    def skewness(self) -> float:
        """Skewness (1-2p)/sqrt(p(1-p))."""
        import math

        p = self.p
        return float((1.0 - 2.0 * p) / math.sqrt(p * (1.0 - p)))

    def kurtosis(self) -> float:
        """Excess kurtosis (1-6p(1-p))/(p(1-p))."""
        p = self.p
        return float((1.0 - 6.0 * p * (1.0 - p)) / (p * (1.0 - p)))

    def quantile(self, q: float) -> float:
        """Inverse CDF F^{-1}(q) over {0, 1}."""
        return 0.0 if float(q) <= 1.0 - self.p else 1.0

    def entropy(self) -> float:
        """Shannon entropy -p log p - (1-p) log(1-p) (nats)."""
        import math

        p = self.p
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return float(-p * math.log(p) - (1.0 - p) * math.log(1.0 - p))

    def mode(self) -> float:
        """Mode (0 if p<1/2 else 1)."""
        return 0.0 if self.p < 0.5 else 1.0

    def sampler(self, seed: int | None = None) -> "BernoulliSampler":
        """Return a sampler for drawing observations from this distribution."""
        return BernoulliSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "BernoulliEstimator":
        """Return an estimator for fitting this distribution from data."""
        if pseudo_count is None:
            return BernoulliEstimator(name=self.name, keys=self.keys, prior=self.prior)
        return BernoulliEstimator(
            pseudo_count=pseudo_count, suff_stat=self.p, name=self.name, keys=self.keys, prior=self.prior
        )

    def dist_to_encoder(self) -> "BernoulliDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return BernoulliDataEncoder()

    def enumerator(self) -> "BernoulliEnumerator":
        """Return an enumerator over the distribution support when available."""
        return BernoulliEnumerator(self)


class BernoulliEnumerator(DistributionEnumerator):
    """Enumerate False/True in descending probability order."""

    def __init__(self, dist: BernoulliDistribution) -> None:
        super().__init__(dist)
        self._entries = [(True, dist.log_p), (False, dist.log_1p)]
        self._entries.sort(key=lambda u: -u[1])
        self._pos = 0

    def __next__(self) -> tuple[bool, float]:
        if self._pos >= len(self._entries):
            raise StopIteration
        rv = self._entries[self._pos]
        self._pos += 1
        return rv


class BernoulliSampler(DistributionSampler):
    """Draw iid Bernoulli observations."""

    def __init__(self, dist: BernoulliDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> bool | Sequence[bool]:
        """Draw one sample or a list of iid samples."""
        rv = self.rng.rand() < self.dist.p if size is None else self.rng.rand(size) < self.dist.p
        return bool(rv) if size is None else rv.tolist()


class BernoulliAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted success and observation counts."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.sum = 0.0
        self.count = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: bool | int, weight: float, estimate: BernoulliDistribution | None) -> None:
        """Accumulate weighted success and count statistics for one observation."""
        xx = BernoulliDistribution._as_bool(x)
        if xx is None:
            raise ValueError("BernoulliDistribution requires observations in {False, True} or {0, 1}.")
        self.sum += float(xx) * weight
        self.count += weight

    def initialize(self, x: bool | int, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one observation."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: BernoulliDistribution | None) -> None:
        """Accumulate weighted statistics from encoded observations."""
        self.sum += np.dot(x.astype(np.float64), weights)
        self.count += np.sum(weights, dtype=np.float64)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float]) -> "BernoulliAccumulator":
        """Merge another Bernoulli sufficient-statistic tuple."""
        self.count += suff_stat[0]
        self.sum += suff_stat[1]
        return self

    def value(self) -> tuple[float, float]:
        """Return the accumulated count and success total."""
        return self.count, self.sum

    def from_value(self, x: tuple[float, float]) -> "BernoulliAccumulator":
        """Replace accumulator contents from a sufficient-statistic tuple."""
        self.count = x[0]
        self.sum = x[1]
        return self

    def acc_to_encoder(self) -> "BernoulliDataEncoder":
        """Return the encoder used by this accumulator."""
        return BernoulliDataEncoder()


class BernoulliAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for BernoulliAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> BernoulliAccumulator:
        """Create a fresh Bernoulli accumulator."""
        return BernoulliAccumulator(name=self.name, keys=self.keys)


class BernoulliEstimator(ParameterEstimator):
    """Estimate a Bernoulli distribution from weighted success counts."""

    def __init__(
        self,
        pseudo_count: float | None = None,
        suff_stat: float | None = None,
        name: str | None = None,
        keys: str | None = None,
        prior: SequenceEncodableProbabilityDistribution | None = None,
    ) -> None:
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.name = name
        self.keys = keys
        self.prior = prior
        self.has_conj_prior = isinstance(prior, BetaDistribution)

    def accumulator_factory(self) -> BernoulliAccumulatorFactory:
        """Return an accumulator factory for Bernoulli sufficient statistics."""
        return BernoulliAccumulatorFactory(name=self.name, keys=self.keys)

    def model_log_density(self, model: "BernoulliDistribution") -> float:
        """Log-density of the model's success probability under the Beta prior (ELBO global term)."""
        if self.has_conj_prior:
            return float(self.prior.log_density(model.p))
        return 0.0

    def _estimate_conjugate(self, suff_stat: tuple[float, float]) -> "BernoulliDistribution":
        """Closed-form Beta conjugate posterior update returning the MAP estimate.

        With a Beta(a, b) prior and weighted counts of successes ``psum`` and failures
        ``nsum``, the posterior is Beta(a + psum, b + nsum) and the returned point
        estimate is the posterior mode ``(psum + a - 1) / (psum + nsum + a + b - 2)``;
        the posterior is carried forward as the fitted model's prior. On the boundary
        where the mode is undefined (``new_a <= 1`` or ``new_b <= 1``, e.g. an empty
        Beta(1, 1) update) the posterior mean ``new_a / (new_a + new_b)`` is returned.
        """
        count, psum = suff_stat
        nsum = count - psum
        a, b = self.prior.get_parameters()
        new_a = a + psum
        new_b = b + nsum
        if new_a > 1.0 and new_b > 1.0:
            p = (psum + a - 1.0) / (psum + nsum + a + b - 2.0)
        else:
            p = new_a / (new_a + new_b)
        return BernoulliDistribution(p, name=self.name, keys=self.keys, prior=BetaDistribution(new_a, new_b))

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float]) -> BernoulliDistribution:
        """Estimate the Bernoulli success probability from weighted counts."""
        if self.has_conj_prior:
            return self._estimate_conjugate(suff_stat)
        count, psum = suff_stat
        if self.pseudo_count is not None:
            prior_p = 0.5 if self.suff_stat is None else self.suff_stat
            psum += self.pseudo_count * prior_p
            count += self.pseudo_count
        p = psum / count if count > 0.0 else 0.5
        p = float(np.clip(p, 1.0e-12, 1.0 - 1.0e-12))
        return BernoulliDistribution(p, name=self.name, keys=self.keys)


class BernoulliDataEncoder(DataSequenceEncoder):
    """Encode Bernoulli observations as a boolean numpy array."""

    def __str__(self) -> str:
        return "BernoulliDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BernoulliDataEncoder)

    def seq_encode(self, x: Sequence[bool | int]) -> np.ndarray:
        """Encode Bernoulli observations as a boolean array."""
        rv = np.asarray(x)
        valid = (rv == 0) | (rv == 1)
        if not np.all(valid):
            raise ValueError("BernoulliDistribution requires observations in {False, True} or {0, 1}.")
        return rv.astype(bool)
