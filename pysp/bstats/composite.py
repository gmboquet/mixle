"""Composite distribution: a product of independent component distributions
evaluated on the entries of a tuple.

Data type: Tuple[X_1, ..., X_K], where X_i is the data type of the i-th
component distribution. The log-density of an observation x is

        log f(x) = sum_i log f_i(x_i).

Defines the CompositeDistribution, CompositeSampler,
CompositeEstimatorAccumulator, CompositeAccumulatorFactory, and
CompositeEstimator classes for use with pysparkplug. Priors compose: the
joint prior is a CompositeDistribution of the component priors.
"""

from numpy.random import RandomState

from pysp.arithmetic import maxint
from pysp.bstats.pdist import (
    DataFrameEncodableAccumulator,
    ParameterEstimator,
    ProbabilityDistribution,
    SequenceEncodableAccumulator,
)


class CompositeDistribution(ProbabilityDistribution):
    """Product distribution over tuples with independent components."""

    def __init__(self, dists, name: str | None = None, keys: str | None = None):
        """Create a composite distribution.

        Args:
                dists: Sequence of component ProbabilityDistribution objects.
                name (Optional[str]): Name of the distribution.
                keys (Optional[str]): Key for sharing sufficient statistics.
        """
        self.dists = dists
        self.count = len(dists)
        self.keys = keys
        self.set_name(name)

        # self.parents = []
        # for d in dists:
        # d.add_parent(self)

    def __str__(self):
        return "CompositeDistribution((%s))" % (",".join(map(str, self.dists)))

    def get_name(self):
        """Return the comma-joined names of the component distributions."""
        return ",".join([u.get_name() for u in self.dists])

    def get_parameters(self):
        """Return a tuple of the component parameters."""
        return tuple([d.get_parameters() for d in self.dists])

    def set_parameters(self, params):
        """Set the component parameters.

        Args:
                params: Sequence of per-component parameter values.
        """
        for d, p in zip(self.dists, params):
            d.set_parameters(p)

    def get_prior(self):
        """Return the joint prior as a CompositeDistribution of the
        component priors."""
        return CompositeDistribution([d.get_prior() for d in self.dists])

    def set_prior(self, prior):
        """Set the joint prior from a CompositeDistribution of component
        priors (the inverse of get_prior).

        Args:
                prior (CompositeDistribution): Composite of one prior per
                        component, in component order.

        Raises:
                ValueError: If the number of priors does not match the number
                        of components.
        """
        if len(prior.dists) != self.count:
            raise ValueError(
                "CompositeDistribution.set_prior expected %d priors but got %d." % (self.count, len(prior.dists))
            )
        for d, p in zip(self.dists, prior.dists):
            d.set_prior(p)

    def cross_entropy(self, dist):
        """Cross entropy H(self, dist), summed over the components.

        Args:
                dist: CompositeDistribution (matched component-wise) or a
                        single distribution evaluated against every component.

        Returns:
                Sum of component cross entropies.
        """
        if isinstance(dist, CompositeDistribution):
            # no name checking right now...
            rv = 0
            for u, v in zip(self.dists, dist.dists):
                rv += u.cross_entropy(v)
            return rv
        else:
            rv = 0
            for u in self.dists:
                rv += u.cross_entropy(dist)
            return rv

    def entropy(self):
        """Return the sum of the component entropies."""
        rv = 0
        for u in self.dists:
            rv += u.entropy()
        return rv

    def log_density(self, x):
        """Log-density sum_i log f_i(x_i) at observation x.

        Args:
                x: Tuple of component observations.

        Returns:
                Log-density (float) at x.
        """
        rv = self.dists[0].log_density(x[0])

        for i in range(1, self.count):
            rv += self.dists[i].log_density(x[i])

        return rv

    def expected_log_density(self, x):
        """Prior-expected log-density at observation x.

        Args:
                x: Tuple of component observations.

        Returns:
                Sum of component expected log-densities.
        """
        rv = self.dists[0].expected_log_density(x[0])

        for i in range(1, self.count):
            rv += self.dists[i].expected_log_density(x[i])

        return rv

    def seq_encode(self, x):
        """Encode a sequence of tuples for vectorized evaluation.

        Args:
                x: Iterable of tuples of component observations.

        Returns:
                Tuple of per-component encodings.
        """
        return tuple([self.dists[i].seq_encode([u[i] for u in x]) for i in range(self.count)])

    def seq_log_density(self, x):
        """Vectorized log-density at sequence-encoded input x.

        Args:
                x: Encoded data from seq_encode().

        Returns:
                Numpy array of log-densities, one entry per observation.
        """
        rv = self.dists[0].seq_log_density(x[0])
        for i in range(1, self.count):
            rv += self.dists[i].seq_log_density(x[i])

        return rv

    def seq_expected_log_density(self, x):
        """Vectorized expected log-density at sequence-encoded input x.

        Args:
                x: Encoded data from seq_encode().

        Returns:
                Numpy array of expected log-densities, one entry per
                observation.
        """
        rv = self.dists[0].seq_expected_log_density(x[0])
        for i in range(1, self.count):
            rv += self.dists[i].seq_expected_log_density(x[i])

        return rv

    def df_log_density(self, df):
        """Log-density evaluated on the columns of a DataFrame.

        Args:
                df (pd.DataFrame): DataFrame with one column per component.

        Returns:
                Per-row sum of component log-densities.
        """
        rv = self.dists[0].df_log_density(df)
        for i in range(1, self.count):
            rv += self.dists[i].df_log_density(df)

        return rv

    def sampler(self, seed=None):
        """Return a CompositeSampler for this distribution.

        Args:
                seed (Optional[int]): Seed for the random number generator.
        """
        return CompositeSampler(self, seed)

    def estimator(self):
        """Return a CompositeEstimator matching this distribution."""
        return CompositeEstimator([d.estimator() for d in self.dists])


class CompositeSampler:
    """Draws tuples of observations from a CompositeDistribution."""

    def __init__(self, dist, seed=None):
        """Create a sampler for a CompositeDistribution.

        Args:
                dist (CompositeDistribution): Distribution to sample from.
                seed (Optional[int]): Seed for the random number generator.
        """
        self.dist = dist
        self.rng = RandomState(seed)
        self.distSamplers = [d.sampler(seed=self.rng.randint(maxint)) for d in dist.dists]

    def sample(self, size=None):
        """Draw size tuples (or one tuple when size is None).

        Args:
                size (Optional[int]): Number of samples to draw.

        Returns:
                A tuple when size is None, otherwise a list of tuples.
        """

        if size is None:
            return tuple([d.sample(size=size) for d in self.distSamplers])
        else:
            return list(zip(*[d.sample(size=size) for d in self.distSamplers]))


class CompositeEstimatorAccumulator(SequenceEncodableAccumulator, DataFrameEncodableAccumulator):
    """Accumulates per-component sufficient statistics for composite
    estimation."""

    def __init__(self, accumulators, keys=None):
        """Create a composite accumulator.

        Args:
                accumulators: List of per-component StatisticAccumulator
                        objects.
                keys (Optional[str]): Key for sharing statistics.
        """
        self.accumulators = accumulators
        self.count = len(accumulators)
        self.key = keys

    def update(self, x, weight, estimate):
        """Accumulate one weighted tuple observation.

        Args:
                x: Tuple of component observations.
                weight (float): Observation weight.
                estimate (Optional[CompositeDistribution]): Current model
                        estimate or None.
        """
        if estimate is not None:
            for i in range(0, self.count):
                self.accumulators[i].update(x[i], weight, estimate.dists[i])
        else:
            for i in range(0, self.count):
                self.accumulators[i].update(x[i], weight, None)

    def initialize(self, x, weight, rng):
        """Initialize with one weighted tuple observation.

        Args:
                x: Tuple of component observations.
                weight (float): Observation weight.
                rng (RandomState): Random number generator.
        """
        for i in range(0, self.count):
            self.accumulators[i].initialize(x[i], weight, rng)

    def seq_initialize(self, x, weights, rng):
        """Vectorized initialization from sequence-encoded data.

        Args:
                x: Encoded data from CompositeDistribution.seq_encode().
                weights (np.ndarray): Observation weights.
                rng (RandomState): Random number generator.
        """
        for i in range(self.count):
            self.accumulators[i].seq_initialize(x[i], weights, rng)

    def seq_update(self, x, weights, estimate):
        """Vectorized update from sequence-encoded data.

        Args:
                x: Encoded data from CompositeDistribution.seq_encode().
                weights (np.ndarray): Observation weights.
                estimate (CompositeDistribution): Current model estimate.
        """
        for i in range(self.count):
            self.accumulators[i].seq_update(x[i], weights, estimate.dists[i])

    def df_initialize(self, df, weights, rng):
        """Initialize from a DataFrame (delegates to the components).

        Args:
                df (pd.DataFrame): DataFrame with one column per component.
                weights: Per-row observation weights.
                rng (RandomState): Random number generator.
        """
        for i in range(self.count):
            self.accumulators[i].df_initialize(df, weights, rng)

    def df_update(self, df, weights, estimate):
        """Accumulate from a DataFrame (delegates to the components).

        Args:
                df (pd.DataFrame): DataFrame with one column per component.
                weights: Per-row observation weights.
                estimate (Optional[CompositeDistribution]): Current model
                        estimate or None.
        """
        if estimate is None:
            for i in range(self.count):
                self.accumulators[i].df_update(df, weights, None)
        else:
            for i in range(self.count):
                self.accumulators[i].df_update(df, weights, estimate.dists[i])

    def combine(self, suff_stat):
        """Merge another accumulator's value() into this one.

        Args:
                suff_stat: Tuple of per-component sufficient statistics.

        Returns:
                This accumulator.
        """
        for i in range(0, self.count):
            self.accumulators[i].combine(suff_stat[i])
        return self

    def value(self):
        """Return a tuple of the per-component sufficient statistics."""
        return tuple([x.value() for x in self.accumulators])

    def from_value(self, x):
        """Set this accumulator's state from a value() tuple.

        Args:
                x: Tuple of per-component sufficient statistics.

        Returns:
                This accumulator.
        """
        self.accumulators = [self.accumulators[i].from_value(x[i]) for i in range(len(x))]
        self.count = len(x)
        return self

    def key_merge(self, stats_dict):
        """Merge keyed statistics into stats_dict.

        Args:
                stats_dict: Mapping from key to shared statistics.
        """

        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

        for u in self.accumulators:
            u.key_merge(stats_dict)

    def key_replace(self, stats_dict):
        """Replace this accumulator's statistics with keyed entries from
        stats_dict.

        Args:
                stats_dict: Mapping from key to shared statistics.
        """

        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

        for u in self.accumulators:
            u.key_replace(stats_dict)


class CompositeAccumulatorFactory:
    """Factory for creating CompositeEstimatorAccumulator objects."""

    def __init__(self, factories, keys):
        """Create a composite accumulator factory.

        Args:
                factories: List of per-component accumulator factories.
                keys (Optional[str]): Key passed to the accumulators.
        """
        self.factories = factories
        self.keys = keys

    def make(self) -> CompositeEstimatorAccumulator:
        """Return a new CompositeEstimatorAccumulator."""
        return CompositeEstimatorAccumulator([f.make() for f in self.factories], keys=self.keys)


class CompositeEstimator(ParameterEstimator):
    """Estimates a CompositeDistribution by estimating each component from
    its own sufficient statistics."""

    def __init__(self, estimators, name: str | None = None, keys: str | None = None):
        """Create a composite estimator.

        Args:
                estimators: List of per-component ParameterEstimator objects.
                name (Optional[str]): Name of the estimated distribution.
                keys (Optional[str]): Key for sharing statistics.
        """

        self.estimators = estimators
        self.count = len(estimators)
        self.keys = keys
        self.name = name

    def get_prior(self):
        """Return the joint prior as a CompositeDistribution of the
        component estimators' priors."""
        return CompositeDistribution([d.get_prior() for d in self.estimators], name=self.keys)

    def set_prior(self, params):
        """Set the joint prior from a CompositeDistribution of component
        priors (the inverse of get_prior).

        Args:
                params (CompositeDistribution): Composite of one prior per
                        component estimator, in estimator order.

        Raises:
                ValueError: If the number of priors does not match the number
                        of component estimators.
        """
        if len(params.dists) != self.count:
            raise ValueError(
                "CompositeEstimator.set_prior expected %d priors but got %d." % (self.count, len(params.dists))
            )
        for d, p in zip(self.estimators, params.dists):
            d.set_prior(p)

    def accumulator_factory(self):
        """Return a CompositeAccumulatorFactory for this estimator."""
        return CompositeAccumulatorFactory([x.accumulator_factory() for x in self.estimators], self.keys)

    def model_log_density(self, model: CompositeDistribution) -> float:
        """Log density of the model parameters under this estimator's prior.

        Args:
                model (CompositeDistribution): Model to evaluate.

        Returns:
                Prior log-density at the model parameters.
        """
        return self.get_prior().log_density(model.get_parameters())

    def scale_suff_stat(self, suff_stat, c):
        """Scale per-field sufficient statistics through child estimators."""
        return tuple(est.scale_suff_stat(ss, c) for est, ss in zip(self.estimators, suff_stat))

    def estimate(self, suff_stat):
        """Estimate a CompositeDistribution from sufficient statistics.

        Args:
                suff_stat: Tuple of per-component sufficient statistics as
                        returned by CompositeEstimatorAccumulator.value().

        Returns:
                CompositeDistribution of the per-component estimates.
        """
        return CompositeDistribution(
            tuple([est.estimate(ss) for est, ss in zip(self.estimators, suff_stat)]), name=self.name, keys=self.keys
        )


# --- API naming aliases (notes/distribution_api_naming_accounting.md) ---
CompositeAccumulator = CompositeEstimatorAccumulator
