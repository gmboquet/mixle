"""Spearman ranking distributions over full permutations.

Data type: List[int] (Component-wise rank of K dimensional observation vector)

The Spearman ranking distribution with dimension K, has probability function

    p_mat(x_k;rho, sigma) = exp(-rho * ||x_k-sigma||^2 ) / sum_{k=0}^{K-1} exp(-rho * ||x_k-sigma||^2 ), for k = 0,1,..,K-1

where x_k list of integers containing a permutation of the integers 0,1,2,...K-1. Note sigma is a list of floats with
dimension equal to K representing the mean of the rank variables, and rho is a correlation coefficient.

"""

import itertools
from collections.abc import Sequence
from functools import cache
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.enumeration.assignment import k_best_assignments
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


@cache
def _permutation_array(dim: int) -> np.ndarray:
    return np.asarray(list(itertools.permutations(range(dim))), dtype=np.float64)


def _squared_distances_to_sigma(sigma: np.ndarray) -> np.ndarray:
    perms = _permutation_array(len(sigma))
    diff = perms - sigma
    return np.sum(diff * diff, axis=1)


def _log_partition(sigma: np.ndarray, rho: float) -> float:
    d2 = _squared_distances_to_sigma(sigma)
    z = -float(rho) * d2
    m = float(np.max(z))
    return m + np.log(np.exp(z - m).sum())


def _expected_distance(distances: np.ndarray, rho: float) -> float:
    z = -float(rho) * distances
    m = float(np.max(z))
    weights = np.exp(z - m)
    return float(np.dot(weights, distances) / weights.sum())


def _estimate_rho_from_mean_distance(
    distances: np.ndarray, mean_distance: float, max_rho: float = 1.0e6, tol: float = 1.0e-12, max_iter: int = 100
) -> float:
    """Return the nonnegative MLE rho satisfying E_rho[D] = mean_distance."""
    if mean_distance <= tol:
        return float(max_rho)

    uniform_mean = float(np.mean(distances))
    if mean_distance >= uniform_mean - tol:
        return 0.0

    lo = 0.0
    hi = 1.0
    while hi < max_rho and _expected_distance(distances, hi) > mean_distance:
        lo = hi
        hi *= 2.0

    if hi >= max_rho and _expected_distance(distances, max_rho) > mean_distance:
        return float(max_rho)

    hi = min(hi, max_rho)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if hi - lo <= tol * max(1.0, hi):
            break
        if _expected_distance(distances, mid) > mean_distance:
            lo = mid
        else:
            hi = mid

    return float(0.5 * (lo + hi))


class SpearmanRankingDistribution(SequenceEncodableProbabilityDistribution):
    """Spearman ranking distribution over permutations of 0,...,K-1 with location sigma and decay rate rho.

    Data type: List[int] (a permutation of the integers 0,1,...,K-1).
    """

    @classmethod
    def compute_capabilities(cls):
        """Declare backend support for Spearman-ranking generated kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic")

    @classmethod
    def compute_declaration(cls):
        """Return the generated-compute declaration for the Spearman ranking distribution."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="spearman_ranking",
            distribution_type=cls,
            parameters=(
                ParameterSpec("sigma", constraint="real_vector"),
                ParameterSpec("rho"),
                ParameterSpec("log_const", constraint="real", differentiable=False),
            ),
            statistics=(
                StatisticSpec("count"),
                StatisticSpec("sum", kind="vector_moment"),
            ),
            support="permutation",
            differentiable=False,
            legacy_sufficient_statistics=cls.backend_legacy_sufficient_statistics,
        )

    @staticmethod
    def backend_legacy_sufficient_statistics(x: Any, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return row-wise legacy sufficient statistics for resident reductions."""
        xx = engine.asarray(x)
        one = engine.sum(xx * 0.0, axis=1) + engine.asarray(1.0)
        return one, xx

    @staticmethod
    def backend_log_density_from_params(x: Any, sigma: Any, rho: Any, log_const: Any, engine: Any) -> Any:
        """Engine-neutral Spearman ranking log-density from fitted parameters."""
        diff = engine.asarray(x) - sigma
        return -rho * engine.sum(diff * diff, axis=-1) - log_const

    def __init__(
        self,
        sigma: Sequence[float] | np.ndarray,
        rho: float = 1.0,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create a Spearman ranking distribution.

        Args:
            sigma (Union[Sequence[float], np.ndarray]): Numpy array of means for the rank variables.
            rho (float): Decay rate on variance of ranks.
            name (Optional[str]): Optional distribution name.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        Attributes:
            sigma (np.ndarray]): Numpy array of means for the rank variables.
            rho (float): Decay rate on variance of ranks.
            name (Optional[str]): Optional distribution name.
            dim (int): Dimension of the rank variable.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        """
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.rho = float(rho)
        self.name = name
        self.dim = len(sigma)
        self.keys = keys

        self.log_const = _log_partition(self.sigma, self.rho)

    def __str__(self) -> str:
        """Return a constructor-style representation of the Spearman ranking distribution."""
        return "SpearmanRankingDistribution(sigma=%s, rho=%s, name=%s, keys=%s)" % (
            repr(self.sigma),
            repr(self.rho),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: list[int]) -> float:
        """Density of Spearman ranking distribution at observation x.

        See log_density() for details.

        Args:
            x (List[int]): Permutation of the integers 0,1,...,K-1.

        Returns:
            Density at observation x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: list[int]) -> float:
        """Log-density of Spearman ranking distribution at observation x.

        The log-density is given by

            log(p(x; rho, sigma)) = -rho * ||x - sigma||^2 - log_const,

        where log_const normalizes over all K! permutations.

        Args:
            x (List[int]): Permutation of the integers 0,1,...,K-1.

        Returns:
            Log-density at observation x.

        """
        temp = np.subtract(x, self.sigma)
        return -self.rho * np.dot(temp, temp) - self.log_const

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of log-density at sequence encoded input x.

        Args:
            x (np.ndarray): 2-d numpy array of N permutations with K columns.

        Returns:
            Numpy array of log-density (float) of length N.

        """
        temp = x - self.sigma
        temp *= temp
        rv = np.sum(temp, axis=1) * -self.rho
        rv -= self.log_const
        return rv

    def backend_seq_log_density(self, x: np.ndarray, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded permutations."""
        return self.backend_log_density_from_params(
            engine.asarray(x),
            engine.asarray(self.sigma),
            engine.asarray(self.rho),
            engine.asarray(self.log_const),
            engine,
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["SpearmanRankingDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked parameters for equal-dimensional Spearman ranking mixtures."""
        dim = int(dists[0].dim)
        if any(int(dist.dim) != dim for dist in dists):
            raise ValueError("Stacked SpearmanRankingDistribution components require equal dimension.")
        return {
            "__pysp_component_axis__": {"sigma": 0, "rho": 0, "log_const": 0},
            "sigma": engine.asarray([dist.sigma for dist in dists]),
            "rho": engine.asarray([dist.rho for dist in dists]),
            "log_const": engine.asarray([dist.log_const for dist in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: np.ndarray, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of Spearman ranking component log densities."""
        diff = engine.asarray(x)[:, None, :] - params["sigma"][None, :, :]
        return -params["rho"][None, :] * engine.sum(diff * diff, axis=2) - params["log_const"][None, :]

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: np.ndarray, weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any]:
        """Return component-stacked legacy ``(count, weighted_rank_sum)`` statistics."""
        ww = engine.asarray(weights)
        xx = engine.asarray(x, dtype=getattr(ww, "dtype", None))
        return engine.sum(ww, axis=0), engine.matmul(ww.T, xx)

    def sampler(self, seed: int | None = None) -> "SpearmanRankingSampler":
        """Create a sampler from this Spearman ranking distribution.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            SpearmanRankingSampler configured from this distribution.

        """
        return SpearmanRankingSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "SpearmanRankingEstimator":
        """Create a SpearmanRankingEstimator with matching dimension and concentration rho.

        Args:
            pseudo_count (Optional[float]): Used to inflate sufficient statistics.

        Returns:
            SpearmanRankingEstimator object.

        """
        return SpearmanRankingEstimator(self.dim, rho=None, pseudo_count=pseudo_count, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "SpearmanRankingDataEncoder":
        """Return the encoder for Spearman ranking observations."""
        return SpearmanRankingDataEncoder()

    def enumerator(self) -> "SpearmanRankingEnumerator":
        """Returns SpearmanRankingEnumerator iterating permutations in descending probability order."""
        return SpearmanRankingEnumerator(self)


class SpearmanRankingEnumerator(DistributionEnumerator):
    """Enumerate permutations in descending Spearman probability order, lazily.

    The Spearman distance ``sum_i (x_i - sigma_i)^2`` is a linear assignment cost (assigning value j to
    position i costs ``(j - sigma_i)^2``), so descending probability is increasing assignment cost: Murty's
    k-best assignment streams the permutations in order without materializing the K! support.
    """

    def __init__(self, dist: SpearmanRankingDistribution) -> None:
        super().__init__(dist)
        n = dist.dim
        sigma = np.asarray(dist.sigma, dtype=float)
        # cost[i, j] = (j - sigma_i)^2; scaling by rho makes increasing cost == descending log-density for any
        # sign of rho (log p = -rho * distance - log_const)
        cost = dist.rho * (np.arange(n, dtype=float)[None, :] - sigma[:, None]) ** 2
        self._gen = k_best_assignments(cost)
        self._log_const = dist.log_const

    def __next__(self) -> tuple[list[int], float]:
        total, rows, cols = next(self._gen)  # StopIteration propagates at the end of the support
        x = [0] * self.dist.dim
        for r, c in zip(rows, cols):
            x[int(r)] = int(c)
        return x, float(-total - self._log_const)


class SpearmanRankingSampler(DistributionSampler):
    """Sampler for the SpearmanRankingDistribution. Draws permutations of 0,...,K-1."""

    def __init__(self, dist: SpearmanRankingDistribution, seed: int | None = None) -> None:
        """Create a sampler for Spearman ranking observations.

        Args:
            dist (SpearmanRankingDistribution): Distribution to sample from.
            seed (Optional[int]): Seed for random number generator.

        Attributes:
            rng (np.random.RandomState): Random number generator.
            dist (SpearmanRankingDistribution): Distribution to sample from.
            perms (List[List[int]]): All K! permutations of 0,...,K-1.
            probs (np.ndarray): Probability of each permutation under dist.

        """
        self.rng = np.random.RandomState(seed)
        self.dist = dist

        self.perms = list(map(list, itertools.permutations(range(dist.dim))))
        encoder = self.dist.dist_to_encoder()
        self.probs = np.exp(dist.seq_log_density(encoder.seq_encode(self.perms)))

    def sample(self, size: int | None = None) -> list[int] | Sequence[list[int]]:
        """Draw iid samples (permutations of 0,...,K-1) from the Spearman ranking distribution.

        Args:
            size (Optional[int]): Number of samples to draw. If None, a single permutation is returned.

        Returns:
            A single permutation (List[int]) if size is None, else a list of size permutations.

        """
        idx = self.rng.choice(len(self.perms), p=self.probs, replace=True, size=size)

        if size is None:
            return self.perms[idx]
        else:
            return [self.perms[u] for u in idx]


class SpearmanRankingAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for the SpearmanRankingDistribution. Tracks the weighted sum of ranks and total weight."""

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        """Create an accumulator for Spearman ranking sufficient statistics.

        Args:
            dim (int): Dimension K of the rank vectors.
            name (Optional[str]): Optional accumulator name.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        Attributes:
            sum (np.ndarray): Weighted component-wise sum of observed rank vectors.
            count (float): Sum of observation weights.
            key (Optional[str]): Optional key for merging sufficient statistics.
            name (Optional[str]): Optional accumulator name.

        """
        self.sum = np.zeros(dim, dtype=np.float64)
        self.count = 0.0
        self.keys = keys
        self.name = name

    def update(self, x: list[int] | np.ndarray, weight: float, estimate: SpearmanRankingDistribution | None) -> None:
        """Update sufficient statistics with a weighted observation.

        Args:
            x (Union[List[int], np.ndarray]): Permutation of the integers 0,1,...,K-1.
            weight (float): Weight for observation.
            estimate (Optional[SpearmanRankingDistribution]): Previous estimate (unused).

        """
        self.sum += np.multiply(x, weight)
        self.count += weight

    def initialize(self, x: list[int] | np.ndarray, weight: float, rng: RandomState) -> None:
        """Initialize sufficient statistics with a weighted observation.

        Args:
            x (Union[List[int], np.ndarray]): Permutation of the integers 0,1,...,K-1.
            weight (float): Weight for observation.
            rng (RandomState): Random number generator (unused).

        """
        if weight != 0:
            self.sum += np.multiply(x, weight)
            self.count += weight

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: SpearmanRankingDistribution | None) -> None:
        """Vectorized update of sufficient statistics from sequence encoded data.

        Args:
            x (np.ndarray): 2-d numpy array of N permutations with K columns.
            weights (np.ndarray): Weights for each of the N observations.
            estimate (Optional[SpearmanRankingDistribution]): Previous estimate (unused).

        """
        self.sum += np.dot(x.T, weights)
        self.count += weights.sum()

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState) -> None:
        """Vectorized initialization of sufficient statistics from sequence encoded data.

        Args:
            x (np.ndarray): 2-d numpy array of N permutations with K columns.
            weights (np.ndarray): Weights for each of the N observations.
            rng (RandomState): Random number generator (unused).

        """
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, np.ndarray]) -> "SpearmanRankingAccumulator":
        """Combine sufficient statistics from another accumulator into this one.

        Args:
            suff_stat (Tuple[float, np.ndarray]): Tuple of count and component-wise rank sums.

        Returns:
            Self, with aggregated sufficient statistics.

        """
        self.sum += suff_stat[1]
        self.count += suff_stat[0]
        return self

    def value(self) -> tuple[float, np.ndarray]:
        """Returns sufficient statistics as a Tuple of count and component-wise rank sums."""
        return self.count, self.sum

    def from_value(self, x: tuple[float, np.ndarray]) -> "SpearmanRankingAccumulator":
        """Set sufficient statistics of accumulator from value x.

        Args:
            x (Tuple[float, np.ndarray]): Tuple of count and component-wise rank sums.

        Returns:
            Self, with sufficient statistics set to x.

        """
        self.sum = x[1]
        self.count = x[0]
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge sufficient statistics from ``stats_dict`` when this accumulator's key is present.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to shared sufficient statistics.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                vals = stats_dict[self.keys]
                stats_dict[self.keys] = (vals[0] + self.count, vals[1] + self.sum)
            else:
                stats_dict[self.keys] = (self.count, self.sum)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace sufficient statistics from ``stats_dict`` when this accumulator's key is present.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to shared sufficient statistics.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                vals = stats_dict[self.keys]
                self.count = vals[0]
                self.sum = vals[1]

    def acc_to_encoder(self) -> "SpearmanRankingDataEncoder":
        """Return the encoder associated with this accumulator."""
        return SpearmanRankingDataEncoder()


class SpearmanRankingAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for Spearman ranking accumulators."""

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        """Create a factory for Spearman ranking accumulators.

        Args:
            dim (int): Dimension K of the rank vectors.
            name (Optional[str]): Optional name assigned to created accumulators.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        """
        self.keys = keys
        self.name = name
        self.dim = dim

    def make(self) -> "SpearmanRankingAccumulator":
        """Return a fresh Spearman ranking accumulator."""
        return SpearmanRankingAccumulator(dim=self.dim, name=self.name, keys=self.keys)


class SpearmanRankingEstimator(ParameterEstimator):
    """Estimator for the SpearmanRankingDistribution from aggregated sufficient statistics.

    The consensus ranking sigma and, by default, the concentration rho are estimated by
    maximum likelihood. Pass a numeric rho to hold the concentration fixed.
    """

    def __init__(
        self,
        dim: int,
        rho: float | None = None,
        pseudo_count: float | None = None,
        suff_stat: tuple[float, np.ndarray] | None = None,
        name: str | None = None,
        keys: str | None = None,
        max_rho: float = 1.0e6,
    ) -> None:
        """Create an estimator for Spearman ranking parameters.

        Args:
            dim (int): Dimension K of the rank vectors.
            rho (Optional[float]): Fixed concentration for the estimated distribution. If None, estimate rho by MLE.
            pseudo_count (Optional[float]): Used to inflate sufficient statistics.
            suff_stat (Optional[Tuple[float, np.ndarray]]): Tuple of count and component-wise rank sums.
            name (Optional[str]): Optional name assigned to the estimated distribution.
            keys (Optional[str]): Optional key for merging sufficient statistics.
            max_rho (float): Finite cap used when the MLE is at rho = infinity.

        """
        if rho is not None and rho < 0:
            raise ValueError("SpearmanRankingEstimator requires rho >= 0 or None (got %s)." % repr(rho))
        if max_rho <= 0:
            raise ValueError("SpearmanRankingEstimator requires max_rho > 0 (got %s)." % repr(max_rho))

        self.rho = None if rho is None else float(rho)
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name
        self.dim = dim
        self.max_rho = float(max_rho)

    def accumulator_factory(self) -> "SpearmanRankingAccumulatorFactory":
        """Return a factory for Spearman ranking accumulators."""
        return SpearmanRankingAccumulatorFactory(self.dim, self.name, self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, np.ndarray]) -> "SpearmanRankingDistribution":
        """Estimate a SpearmanRankingDistribution from sufficient statistics.

        The consensus ranking sigma is the maximum likelihood estimate, given by the rank order
        (argsort) of the component-wise rank sums. When rho is None, the concentration is the
        nonnegative MLE satisfying E_rho[||X-sigma||^2] = observed mean squared distance. If no
        data was observed, rho is set to 0.0.

        Args:
            nobs (Optional[float]): Number of observations (unused).
            suff_stat (Tuple[float, np.ndarray]): Tuple of count and component-wise rank sums.

        Returns:
            SpearmanRankingDistribution object.

        """
        count, vsum = suff_stat
        count = float(count)
        vsum = np.asarray(vsum, dtype=np.float64)

        if self.pseudo_count is not None and self.suff_stat is not None:
            pcount, psum = self.suff_stat
            count += float(self.pseudo_count) * float(pcount)
            vsum = vsum + float(self.pseudo_count) * np.asarray(psum, dtype=np.float64)

        if count > 0:
            # Observations are rank vectors: x[j] is the rank assigned to item j.
            # argsort(vsum) returns item order, so rank the item order once more.
            sigma = np.argsort(np.argsort(vsum))
            if self.rho is None:
                sigma_float = np.asarray(sigma, dtype=np.float64)
                rank_norm2 = float(np.dot(np.arange(self.dim, dtype=np.float64), np.arange(self.dim, dtype=np.float64)))
                total_distance = 2.0 * count * rank_norm2 - 2.0 * float(np.dot(vsum, sigma_float))
                mean_distance = max(0.0, total_distance / count)
                distances = _squared_distances_to_sigma(sigma_float)
                rho = _estimate_rho_from_mean_distance(distances, mean_distance, max_rho=self.max_rho)
            else:
                rho = self.rho
        else:
            sigma = np.arange(self.dim)  # no data: the identity permutation (rho=0 is uniform regardless)
            rho = 0.0

        return SpearmanRankingDistribution(sigma, rho, name=self.name, keys=self.keys)


class SpearmanRankingDataEncoder(DataSequenceEncoder):
    """Data encoder for sequences of rank vector (permutation) observations."""

    def __str__(self) -> str:
        """Return the Spearman ranking encoder's display name."""
        return "SpearmanRankingDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return true when ``other`` is a Spearman ranking data encoder.

        Args:
            other (object): Object to compare against.

        Returns:
            True if other is a SpearmanRankingDataEncoder instance, else False.

        """
        return isinstance(other, SpearmanRankingDataEncoder)

    def seq_encode(self, x: Sequence[list[int]]) -> np.ndarray:
        """Encode a sequence of N rank vectors for vectorized functions.

        Args:
            x (Sequence[List[int]]): Sequence of N permutations of 0,1,...,K-1.

        Returns:
            2-d numpy array with N rows and K columns.

        """
        rv = np.asarray(x, dtype=int)  # rank vectors are integer permutations
        if rv.ndim != 2:
            raise ValueError("SpearmanRankingDataEncoder expects a 2-d array of equal-length rank vectors.")
        return rv
