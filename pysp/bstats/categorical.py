"""Categorical distribution over an arbitrary discrete set of values with a
(dictionary) Dirichlet prior on the category probabilities.

Data type: hashable values. A CategoricalDistribution with probability map
{v_k: p_k} has log-density

        log f(x) = log p_x - log(1 + default_value),

where default_value is the unnormalized probability assigned to values not in
the map. Defines the CategoricalDistribution, CategoricalSampler,
CategoricalEstimatorAccumulator, CategoricalEstimatorAccumulatorFactory, and
CategoricalEstimator classes for use with pysparkplug. With a
DictDirichletDistribution prior, estimation is MAP (counts + alpha - 1,
clamped at the simplex boundary, posterior mean when degenerate) and the
posterior Dirichlet is carried forward as the new prior.
"""

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.special import digamma

from pysp.arithmetic import *
from pysp.bstats.catdirichlet import DictDirichletDistribution
from pysp.bstats.pdist import (
    DataFrameEncodableAccumulator,
    ParameterEstimator,
    ProbabilityDistribution,
    SequenceEncodableAccumulator,
)

default_prior = DictDirichletDistribution(1.0 + 1.0e-12)


class CategoricalDistribution(ProbabilityDistribution):
    """Categorical distribution over the keys of a probability map."""

    def __init__(
        self,
        prob_map: dict[Any, float],
        default_value: float = 0.0,
        name: str | None = None,
        prior: ProbabilityDistribution | None = default_prior,
    ):
        """Create a categorical distribution.

        Args:
                prob_map (Dict[Any, float]): Map from value to probability.
                default_value (float): Unnormalized probability assigned to
                        values not present in prob_map.
                name (Optional[str]): Name of the distribution.
                prior (Optional[ProbabilityDistribution]): Prior on the category
                        probabilities (DictDirichletDistribution for conjugacy).
        """

        with np.errstate(divide="ignore"):
            self.prob_map = prob_map
            # self.prob_vec = np.asarray(u[1] for u in prob_list)
            self.name = name

            self.default_value = default_value
            self.log_default_value = np.log(default_value)
            self.log1p_default_value = np.log1p(default_value)
            self.set_prior(prior)

    def __str__(self):
        return "CategoricalDistribution(%s, default_value=%s, name=%s, prior=%s)" % (
            str(self.prob_map),
            str(self.default_value),
            str(self.name),
            str(self.prior),
        )

    def get_parameters(self):
        """Return the probability map."""
        return self.prob_map

    def set_parameters(self, params):
        """Set the probability map.

        Args:
                params (Dict[Any, float]): Map from value to probability.
        """
        self.prob_map = params

    def get_prior(self) -> ProbabilityDistribution:
        """Return the prior on the category probabilities."""
        return self.prior

    def set_prior(self, prior: ProbabilityDistribution):
        """Set the prior on the category probabilities and cache digamma
        expectations when the prior is conjugate.

        Args:
                prior (ProbabilityDistribution): New prior distribution.
        """
        self.prior = prior

        if isinstance(prior, DictDirichletDistribution):
            a = self.prior.get_parameters()
            n = len(self.prob_map)
            if isinstance(a, float):
                bb = digamma(a) - digamma(n * a)
                b = {k: bb for k in self.prob_map.keys()}
            else:
                b = digamma(sum(a.values()))
                b = {k: digamma(v) - b for k, v in a.items()}
            self.conj_prior_params = a
            self.expected_nparams = b
            self.has_conj_prior = True

        else:
            self.conj_prior_params = None
            self.expected_nparams = None
            self.has_conj_prior = False

    def entropy(self) -> float:
        """Return the entropy of the categorical distribution in nats.

        The stored probabilities are normalized by ``1 + default_value`` to
        match log_density().  When default_value is positive it is treated as
        the aggregate mass of the out-of-map default bucket.
        """
        rv = 0.0
        z = 1.0 + self.default_value
        for v in self.prob_map.values():
            if v > 0:
                p = v / z
                rv -= np.log(p) * p
        if self.default_value > 0:
            p = self.default_value / z
            rv -= np.log(p) * p
        return rv

    def cross_entropy(self, dist: ProbabilityDistribution) -> float:
        """Cross entropy -E_self[log dist(X)] over the categorical support.

        For another CategoricalDistribution, the default_value of each
        distribution is treated as a single aggregate default bucket, matching
        the entropy convention used here.  For non-categorical distributions,
        this is implemented only when this distribution has no default bucket.
        """
        z = 1.0 + self.default_value
        rv = 0.0
        for k, v in self.prob_map.items():
            if v > 0:
                rv -= (v / z) * dist.log_density(k)

        if self.default_value > 0:
            if isinstance(dist, CategoricalDistribution):
                if dist.default_value <= 0.0:
                    return np.inf
                rv -= (self.default_value / z) * (dist.log_default_value - dist.log1p_default_value)
            else:
                raise NotImplementedError(
                    "CategoricalDistribution.cross_entropy with default_value > 0 requires "
                    "a CategoricalDistribution argument."
                )
        return rv

    def log_density(self, x) -> float:
        """Log-density at observation x.

        Args:
                x: Observed value (any hashable).

        Returns:
                log p_x for mapped values, log(default_value) otherwise, both
                normalized by log(1 + default_value).
        """
        p = self.prob_map.get(x, self.default_value)
        return -np.inf if p <= 0.0 else np.log(p) - self.log1p_default_value

    def expected_log_density(self, x) -> float:
        """Prior-expected log-density E[log p_x] at observation x.

        Falls back to log_density when no conjugate prior is set.

        Args:
                x: Observed value (any hashable).

        Returns:
                Expected log-density (float) at x.
        """

        if not self.has_conj_prior:
            return self.log_density(x)

        if x not in self.prob_map:
            return self.log_default_value - self.log1p_default_value

        if self.has_conj_prior:
            return self.expected_nparams[x] - self.log1p_default_value

    def seq_log_density(self, x) -> float:
        """Vectorized log-density at sequence-encoded input x.

        Args:
                x: Encoded data from seq_encode().

        Returns:
                Numpy array of log-densities, one entry per observation.
        """
        xs, val_map_inv = x
        mapped_probs = np.asarray([self.prob_map.get(u, self.default_value) for u in val_map_inv], dtype=np.float64)
        mapped_log_probs = np.full(len(mapped_probs), -np.inf, dtype=np.float64)
        good = mapped_probs > 0.0
        mapped_log_probs[good] = np.log(mapped_probs[good]) - self.log1p_default_value

        return mapped_log_probs[xs]

    def seq_expected_log_density(self, x):
        """Vectorized expected log-density at sequence-encoded input x.

        Args:
                x: Encoded data from seq_encode().

        Returns:
                Numpy array of expected log-densities, one entry per observation.
        """
        xs, val_map_inv = x
        rv = np.asarray([self.expected_log_density(u) for u in val_map_inv])

        return rv[xs]

    def seq_encode(self, x):
        """Encode a sequence of observations for vectorized evaluation.

        Args:
                x: Iterable of observed values.

        Returns:
                Tuple (index array, unique value array).
        """
        val_map_inv, xs = np.unique(x, return_inverse=True)
        return xs, val_map_inv

    def sampler(self, seed=None):
        """Return a CategoricalSampler for this distribution.

        Args:
                seed (Optional[int]): Seed for the random number generator.
        """
        return CategoricalSampler(self, seed)

    def estimator(self):
        """Return a CategoricalEstimator matching this distribution."""
        return CategoricalEstimator(name=self.name, prior=self.prior)


class CategoricalSampler:
    """Draws observations from a CategoricalDistribution."""

    def __init__(self, dist, seed=None):
        """Create a sampler for a CategoricalDistribution.

        Args:
                dist (CategoricalDistribution): Distribution to sample from.
                seed (Optional[int]): Seed for the random number generator.
        """
        self.rng = np.random.RandomState(seed)

        temp = dist.prob_map.items()
        self.levels = [u[0] for u in temp]
        self.probs = [u[1] for u in temp]
        self.num_levels = len(self.levels)

    def sample(self, size=None):
        """Draw size samples (or one sample when size is None).

        Args:
                size (Optional[int]): Number of samples to draw.

        Returns:
                A single value when size is None, otherwise a list of values.
        """

        if size is None:
            return self.rng.choice(self.levels, p=self.probs, size=size)

        else:
            levels = self.levels
            rv = self.rng.choice(self.num_levels, p=self.probs, size=size)
            return [levels[i] for i in rv]


class CategoricalEstimatorAccumulator(SequenceEncodableAccumulator, DataFrameEncodableAccumulator):
    """Accumulates per-value counts for categorical estimation."""

    def __init__(self, name, keys):
        """Create a categorical accumulator.

        Args:
                name: Column name used by the DataFrame update path.
                keys: Tuple whose first entry keys this accumulator for
                        statistic sharing.
        """
        self.name = name
        self.key = keys[0]
        self.count_map = defaultdict(float)
        self.count_sum = 0.0

    def update(self, x, weight, estimate):
        """Accumulate one weighted observation.

        Args:
                x: Observed value.
                weight (float): Observation weight.
                estimate: Unused (kept for protocol consistency).
        """
        self.count_map[x] += weight
        self.count_sum += weight

    def initialize(self, x, weight, rng):
        """Initialize with one weighted observation (delegates to update)."""
        self.update(x, weight, None)

    def seq_initialize(self, x, weights, rng):
        """Vectorized initialization from sequence-encoded data.

        Args:
                x: Encoded data from CategoricalDistribution.seq_encode().
                weights (np.ndarray): Observation weights.
                rng: Unused (kept for protocol consistency).
        """
        inv_key_map = x[1]
        bcnt = np.bincount(x[0], weights=weights)
        self.count_sum += np.sum(bcnt)
        for i in range(0, len(bcnt)):
            self.count_map[inv_key_map[i]] += bcnt[i]

    def seq_update(self, x, weights, estimate):
        """Vectorized update from sequence-encoded data.

        Args:
                x: Encoded data from CategoricalDistribution.seq_encode().
                weights (np.ndarray): Observation weights.
                estimate: Unused (kept for protocol consistency).
        """
        inv_key_map = x[1]
        bcnt = np.bincount(x[0], weights=weights)
        self.count_sum += np.sum(bcnt)
        for i in range(0, len(bcnt)):
            self.count_map[inv_key_map[i]] += bcnt[i]

    def df_initialize(self, df, weights, rng):
        """Initialize from a DataFrame column (delegates to df_update)."""
        self.df_update(df, weights, None)

    def df_update(self, df, weights, estimate):
        """Accumulate weighted counts from the DataFrame column self.name.

        Args:
                df (pd.DataFrame): DataFrame containing the column self.name.
                weights: Per-row observation weights (indexable by position).
                estimate: Unused (kept for protocol consistency).
        """
        weights = np.asarray(weights)
        gb = df.groupby([self.name])
        for k, idx in gb.indices.items():
            loc_sum = np.sum(weights[idx])
            self.count_map[k] += loc_sum
            self.count_sum += loc_sum

    def combine(self, suff_stat):
        """Merge another accumulator's value() into this one.

        Args:
                suff_stat: Tuple (count map, count sum).

        Returns:
                This accumulator.
        """
        self.count_sum += suff_stat[1]
        for item in suff_stat[0].items():
            self.count_map[item[0]] = self.count_map.get(item[0], 0.0) + item[1]
        return self

    def value(self):
        """Return (count map, count sum)."""
        return self.count_map, self.count_sum

    def from_value(self, x):
        """Set this accumulator's state from a value() tuple.

        Args:
                x: Tuple (count map, count sum).

        Returns:
                This accumulator.
        """
        self.count_map = x[0]
        self.count_sum = x[1]
        return self


class CategoricalEstimatorAccumulatorFactory:
    """Factory for creating CategoricalEstimatorAccumulator objects."""

    def __init__(self, name, keys):
        """Create a categorical accumulator factory.

        Args:
                name: Column name passed to the accumulators.
                keys: Key tuple passed to the accumulators.
        """
        self.name = name
        self.keys = keys

    def make(self):
        """Return a new CategoricalEstimatorAccumulator."""
        return CategoricalEstimatorAccumulator(self.name, self.keys)


class CategoricalEstimator(ParameterEstimator):
    """Estimates a CategoricalDistribution from accumulated counts, using
    Dirichlet MAP probabilities when a conjugate prior is set."""

    def __init__(self, default_value: float = 0.0, name=None, prior=default_prior, keys=(None,)):
        """Create a categorical estimator.

        Args:
                default_value (float): Unnormalized probability for unseen values
                        in the estimated distribution.
                name (Optional[str]): Name of the estimated distribution.
                prior: Prior on the category probabilities.
                keys: Key tuple for sharing statistics.
        """
        self.keys = keys
        self.name = name
        self.prior = prior
        self.default_value = default_value

    def accumulator_factory(self):
        """Return a CategoricalEstimatorAccumulatorFactory for this
        estimator."""
        return CategoricalEstimatorAccumulatorFactory(self.name, self.keys)

    def get_prior(self):
        """Return the prior on the category probabilities."""
        return self.prior

    def set_prior(self, prior):
        """Set the prior on the category probabilities.

        Args:
                prior (ProbabilityDistribution): New prior distribution.
        """
        self.prior = prior

    def estimate(self, suff_stat):
        """Estimate a CategoricalDistribution from sufficient statistics.

        Args:
                suff_stat: Tuple (count map, count sum) as returned by
                        CategoricalEstimatorAccumulator.value().

        Returns:
                CategoricalDistribution with MAP probabilities (under a
                DictDirichletDistribution prior) or relative frequencies.
        """

        count_map, stats_sum = suff_stat
        stats_sum = sum(count_map.values())

        # if self.default_value:
        # if stats_sum > 0:
        # default_value = 1.0/stats_sum
        # default_value *= default_value
        # else:
        # default_value = 0.5
        # else:
        # default_value = 0.0
        default_value = self.default_value

        if isinstance(self.prior, DictDirichletDistribution):
            conj_prior_params = self.prior.get_parameters()

            if isinstance(conj_prior_params, float):
                alpha = conj_prior_params

                keys = count_map.keys()
                # Dirichlet MAP sits on the boundary when alpha_k + n_k < 1
                num = {k: max((alpha - 1) + count_map[k], 0.0) for k in keys}
                cpp = {k: (alpha + count_map[k]) for k in keys}
            else:
                alpha_sum = sum(conj_prior_params.values())

                keys = set(conj_prior_params.keys()).union(count_map.keys())
                num = {k: max((conj_prior_params.get(k, 0.0) - 1) + count_map.get(k, 0.0), 0.0) for k in keys}
                cpp = {k: (conj_prior_params.get(k, 0.0) + count_map.get(k, 0.0)) for k in keys}

            norm_const = sum(num.values())

            if norm_const > 0:
                pMap = {k: v / norm_const for k, v in num.items()}
            else:
                # fall back to the posterior mean when the MAP is degenerate
                cpp_sum = sum(cpp.values())
                pMap = {k: v / cpp_sum for k, v in cpp.items()}

            return CategoricalDistribution(
                pMap, default_value=default_value, name=self.name, prior=DictDirichletDistribution(cpp)
            )

        else:
            nobs_loc = stats_sum

            if nobs_loc == 0:
                pMap = {k: 1.0 / float(len(count_map)) for k in count_map.keys()}
            else:
                pMap = {k: v / nobs_loc for k, v in count_map.items()}

            return CategoricalDistribution(pMap, default_value=default_value, name=self.name)


# --- API naming aliases (notes/distribution_api_naming_accounting.md) ---
CategoricalAccumulator = CategoricalEstimatorAccumulator
CategoricalAccumulatorFactory = CategoricalEstimatorAccumulatorFactory
