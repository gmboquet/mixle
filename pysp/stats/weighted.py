"""Evaluate, estimate, and sample from a weighted wrapper around a base distribution.

Defines the WeightedDistribution, WeightedSampler, WeightedAccumulator, WeightedAccumulatorFactory,
WeightedEstimator, and the WeightedDataEncoder classes for use with pysparkplug.

Data type: Tuple[D, float]: An observation is a pair (value, weight) where value has the data type D of
the base distribution and weight is a non-negative score attached to the observation. The weight does not
enter the likelihood; it only scales the observation's contribution to the sufficient statistics during
estimation. Likelihood evaluations delegate to the base distribution on the value alone, i.e.

    P((x, w)) = P_base(x).

"""
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder, \
    DistributionEnumerator, child_enumerator
from numpy.random import RandomState
import numpy as np
from typing import Dict, Any, Optional, Tuple, Sequence, TypeVar, Union

D = TypeVar('D')
E = TypeVar('E')
SS = TypeVar('SS')


class WeightedDistribution(SequenceEncodableProbabilityDistribution):
    """WeightedDistribution object that attaches observation weights to a base distribution.

    Args:
        dist (SequenceEncodableProbabilityDistribution): Base distribution for the observed values.
        name (Optional[str]): Set name for object instance.

    Attributes:
        dist (SequenceEncodableProbabilityDistribution): Base distribution for the observed values.
        name (Optional[str]): Name for object instance.

    """

    def __init__(self, dist: SequenceEncodableProbabilityDistribution, name: Optional[str] = None):
        self.dist = dist
        self.name = name

    def __str__(self) -> str:
        """Returns string representation of WeightedDistribution object."""
        return 'WeightedDistribution(dist=%s, name=%s)' % (repr(self.dist), repr(self.name))

    def density(self, x: D) -> float:
        """Density of the base distribution at observation value x.

        Args:
            x (D): Observation value (weight excluded).

        Returns:
            Density of the base distribution at x.

        """
        return self.dist.density(x)

    def log_density(self, x: D) -> float:
        """Log-density of the base distribution at observation value x.

        The observation weight does not enter the likelihood, so this is simply the
        base distribution's log-density evaluated on the value.

        Args:
            x (D): Observation value (weight excluded).

        Returns:
            Log-density of the base distribution at x.

        """
        return self.dist.log_density(x)

    def seq_log_density(self, x: Tuple[E, np.ndarray]) -> np.ndarray:
        """Vectorized log-density of the base distribution on encoded values.

        Args:
            x (Tuple[E, np.ndarray]): Sequence encoded values and weights from WeightedDataEncoder.

        Returns:
            Numpy array of base-distribution log-densities.

        """
        return self.dist.seq_log_density(x[0])

    def dist_to_encoder(self) -> 'WeightedDataEncoder':
        """Returns a WeightedDataEncoder for encoding sequences of (value, weight) observations."""
        return WeightedDataEncoder(encoder=self.dist.dist_to_encoder())

    def estimator(self, pseudo_count: Optional[float] = None) -> 'WeightedEstimator':
        """Create a WeightedEstimator wrapping the base distribution's estimator.

        Args:
            pseudo_count (Optional[float]): Passed through to the base distribution's estimator.

        Returns:
            WeightedEstimator object.

        """
        if pseudo_count is not None:
            return WeightedEstimator(estimator=self.dist.estimator(pseudo_count=pseudo_count), name=self.name)
        else:
            return WeightedEstimator(estimator=self.dist.estimator(), name=self.name)

    def sampler(self, seed: Optional[int] = None) -> 'WeightedSampler':
        """Create a WeightedSampler producing (value, weight) pairs.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            WeightedSampler object.

        """
        return WeightedSampler(self, seed)

    def enumerator(self) -> 'DistributionEnumerator':
        """Delegates to the base distribution's enumerator (log_density is pure delegation)."""
        return child_enumerator(self.dist, 'WeightedDistribution.dist')


class WeightedSampler(DistributionSampler):
    """WeightedSampler object for drawing (value, weight) observations from a WeightedDistribution.

    The likelihood does not model the weight, so samples carry the neutral weight 1.0: accumulating
    (value, 1.0) is equivalent to accumulating the bare value with the base distribution. Values are
    drawn from the base distribution's sampler.

    Args:
        dist (WeightedDistribution): WeightedDistribution to draw samples from.
        seed (Optional[int]): Seed to set for sampling with RandomState.

    Attributes:
        dist (WeightedDistribution): WeightedDistribution to draw samples from.
        rng (RandomState): Seeded RandomState for sampling.
        dist_sampler (DistributionSampler): Sampler for the base distribution.

    """

    def __init__(self, dist: WeightedDistribution, seed: Optional[int] = None) -> None:
        super().__init__(dist, seed)
        self.dist_sampler = dist.dist.sampler(seed=self.new_seed())

    def sample(self, size: Optional[int] = None) -> Union[Tuple[Any, float], Sequence[Tuple[Any, float]]]:
        """Draw iid (value, weight) samples, each with weight 1.0.

        Args:
            size (Optional[int]): Number of iid samples to draw.

        Returns:
            A single (value, 1.0) tuple if size is None, else a list of size such tuples.

        """
        if size is None:
            return self.dist_sampler.sample(), 1.0
        else:
            return [(v, 1.0) for v in self.dist_sampler.sample(size=size)]


class WeightedAccumulator(SequenceEncodableStatisticAccumulator):
    """WeightedAccumulator object that scales each observation's weight by its attached score.

    Args:
        accumulator (SequenceEncodableStatisticAccumulator): Accumulator for the base distribution.
        name (Optional[str]): Set name for object instance.

    Attributes:
        accumulator (SequenceEncodableStatisticAccumulator): Accumulator for the base distribution.
        name (Optional[str]): Name for object instance.

    """

    def __init__(self, accumulator: SequenceEncodableStatisticAccumulator, name: Optional[str] = None):
        self.accumulator = accumulator
        self.name = name

    def initialize(self, x: Tuple[D, float], weight: float, rng: np.random.RandomState) -> None:
        """Initialize the base accumulator with observation x[0] weighted by weight*x[1].

        Args:
            x (Tuple[D, float]): Observation (value, weight) pair.
            weight (float): External weight on the observation.
            rng (RandomState): Random number generator for initialization.

        """
        self.accumulator.initialize(x[0], weight*x[1], rng)

    def update(self, x: Tuple[D, float], weight: float, estimate: WeightedDistribution) -> None:
        """Update the base accumulator with observation x[0] weighted by weight*x[1].

        Args:
            x (Tuple[D, float]): Observation (value, weight) pair.
            weight (float): External weight on the observation.
            estimate (WeightedDistribution): Previous estimate of the weighted distribution.

        """
        self.accumulator.update(x[0], weight*x[1], estimate.dist)

    def seq_update(self, x, weights: np.ndarray, estimate: WeightedDistribution) -> None:
        """Vectorized update of the base accumulator with weights scaled by the observation weights.

        Args:
            x (Tuple[E, np.ndarray]): Sequence encoded values and weights from WeightedDataEncoder.
            weights (np.ndarray): External weights on the observations.
            estimate (WeightedDistribution): Previous estimate of the weighted distribution.

        """
        self.accumulator.seq_update(x[0], weights*x[1], estimate.dist)

    def seq_initialize(self, x: Tuple[E, np.ndarray], weights: np.ndarray, rng: np.random.RandomState) -> None:
        """Vectorized initialization of the base accumulator with scaled weights.

        Args:
            x (Tuple[E, np.ndarray]): Sequence encoded values and weights from WeightedDataEncoder.
            weights (np.ndarray): External weights on the observations.
            rng (RandomState): Random number generator for initialization.

        """
        self.accumulator.seq_initialize(x[0], weights*x[1], rng)

    def combine(self, suff_stat: SS) -> 'WeightedAccumulator':
        """Combine the base accumulator's sufficient statistics with suff_stat.

        Args:
            suff_stat (SS): Sufficient statistics of the base accumulator.

        Returns:
            This WeightedAccumulator.

        """
        self.accumulator.combine(suff_stat)
        return self

    def from_value(self, x: SS) -> 'WeightedAccumulator':
        """Set the base accumulator's sufficient statistics from x.

        Args:
            x (SS): Sufficient statistics of the base accumulator.

        Returns:
            This WeightedAccumulator.

        """
        self.accumulator.from_value(x)

        return self

    def value(self) -> Any:
        """Returns the base accumulator's sufficient statistics."""
        return self.accumulator.value()

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge keyed sufficient statistics of the base accumulator into stats_dict."""
        self.accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace keyed sufficient statistics of the base accumulator from stats_dict."""
        self.accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'WeightedDataEncoder':
        """Returns a WeightedDataEncoder for encoding sequences of (value, weight) observations."""
        return WeightedDataEncoder(encoder=self.accumulator.acc_to_encoder())

class WeightedAccumulatorFactory(StatisticAccumulatorFactory):
    """WeightedAccumulatorFactory object for creating WeightedAccumulator objects.

    Args:
        factory (StatisticAccumulatorFactory): Accumulator factory for the base distribution.
        name (Optional[str]): Set name for object instance.

    Attributes:
        factory (StatisticAccumulatorFactory): Accumulator factory for the base distribution.
        name (Optional[str]): Name for object instance.

    """

    def __init__(self, factory: StatisticAccumulatorFactory, name: Optional[str] = None):
        self.factory = factory
        self.name = name

    def make(self) -> 'WeightedAccumulator':
        """Returns a new WeightedAccumulator wrapping a fresh base accumulator."""
        return WeightedAccumulator(accumulator=self.factory.make(), name=self.name)


class WeightedEstimator(ParameterEstimator):
    """WeightedEstimator object for estimating a WeightedDistribution from weighted observations.

    Args:
        estimator (ParameterEstimator): Estimator for the base distribution.
        name (Optional[str]): Set name for object instance.

    Attributes:
        estimator (ParameterEstimator): Estimator for the base distribution.
        name (Optional[str]): Name for object instance.

    """

    def __init__(self, estimator: ParameterEstimator, name: Optional[str] = None):
        self.estimator = estimator
        self.name = name

    def accumulator_factory(self) -> 'WeightedAccumulatorFactory':
        """Returns a WeightedAccumulatorFactory wrapping the base estimator's factory."""
        return WeightedAccumulatorFactory(factory=self.estimator.accumulator_factory(), name=self.name)

    def estimate(self, nobs: Optional[float], suff_stat: SS) -> 'WeightedDistribution':
        """Estimate a WeightedDistribution from the base distribution's sufficient statistics.

        Args:
            nobs (Optional[float]): Weighted number of observations.
            suff_stat (SS): Sufficient statistics of the base accumulator.

        Returns:
            WeightedDistribution wrapping the estimated base distribution.

        """
        return WeightedDistribution(dist=self.estimator.estimate(nobs, suff_stat), name=self.name)


class WeightedDataEncoder(DataSequenceEncoder):
    """WeightedDataEncoder object for encoding sequences of iid (value, weight) observations.

    Args:
        encoder (DataSequenceEncoder): Encoder for the base distribution's values.

    Attributes:
        encoder (DataSequenceEncoder): Encoder for the base distribution's values.

    """

    def __init__(self, encoder: DataSequenceEncoder) -> None:
        self.encoder = encoder

    def __str__(self) -> str:
        """Returns string representation of WeightedDataEncoder object."""
        return 'WeightedDataEncoder(encoder=%s)' % (repr(self.encoder))

    def __eq__(self, other: object) -> bool:
        """Return True if other is a WeightedDataEncoder with an equal base encoder."""
        if isinstance(other, WeightedDataEncoder):
            return other.encoder == self.encoder
        else:
            return False

    def seq_encode(self, x: Sequence[Tuple[D, float]]) -> Tuple[Any, np.ndarray]:
        """Encode a sequence of (value, weight) observations for vectorized use.

        Args:
            x (Sequence[Tuple[D, float]]): Sequence of iid (value, weight) observations.

        Returns:
            Tuple of base-encoded values and a numpy array of weights.

        """
        return self.encoder.seq_encode([xx[0] for xx in x]), np.asarray([xx[1] for xx in x], dtype=float)


