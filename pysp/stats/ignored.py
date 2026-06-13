"""Create, estimate, and sample from an IgnoredDistribution.

Defines the IgnoredDistribution, IgnoredSampler, IgnoredAccumulatorFactory, IgnoredAccumulator, IgnoredEstimator,
and the IgnoredDataEncoder classes for use with pysparkplug.

Ignored distribution is simply a distribution that is ignored in estimation and treated as fixed.

"""
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator, DataSequenceEncoder, DistributionSampler, StatisticAccumulatorFactory
from numpy.random import RandomState
import numpy as np
from pysp.stats.null_dist import NullDistribution, NullDataEncoder, NullSampler
from typing import Dict, Any, Sequence, TypeVar, Optional, Union

T = TypeVar('T')
E = TypeVar('E')


class IgnoredDistribution(SequenceEncodableProbabilityDistribution):

    """Distribution wrapper that assigns zero log-density while preserving an estimator interface."""

    def compute_capabilities(self):
        from pysp.stats.capabilities import capabilities_for
        return capabilities_for(self.dist)

    def compute_declaration(self):
        from pysp.stats.declarations import DistributionDeclaration, StatisticSpec, declaration_for
        child = declaration_for(self.dist)
        children = () if child is None else (child,)
        return DistributionDeclaration(
            name='ignored',
            distribution_type=type(self),
            parameters=(),
            statistics=(StatisticSpec('ignored', kind='none', additive=False, scales=False),),
            support='delegated',
            children=children,
            child_roles=('ignored',) if child is not None else (),
            differentiable=False,
        )

    def __init__(self, dist: Optional[SequenceEncodableProbabilityDistribution], name: Optional[str] = None):
        """IgnoredDistribution object for using IgnoredDistributions in estimation.

        Args:
            dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution to be ignored.
            name (Optional[str]): Set name for object instance.
        """
        self.dist = dist if dist is not None else NullDistribution()
        self.name = name

    def __str__(self) -> str:
        return 'IgnoredDistribution(%s)' % (str(self.dist))

    def density(self, x: T) -> float:
        """Evaluate the density of the IgnoredDistribution at x.

        Args:
            x (T): Type corresponding to attribute 'dist'.

        Returns:
            Density of attribute 'dist' at x

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: T):
        """Evaluate the log-density of the IgnoredDistribution at x.

        Args:
            x (T): Type corresponding to attribute 'dist'.

        Returns:
            log-density of attribute 'dist' at x.

        """
        return self.dist.log_density(x)

    def seq_log_density(self, x: E) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        rv = self.dist.seq_log_density(x)
        return rv

    def backend_seq_log_density(self, x: E, engine: Any) -> Any:
        """Engine-neutral vectorized log-density delegated to the wrapped distribution."""
        from pysp.stats.backend import backend_seq_log_density
        return backend_seq_log_density(self.dist, x, engine)

    @classmethod
    def backend_stacked_params(cls, dists: Sequence['IgnoredDistribution'], engine: Any) -> Dict[str, Any]:
        """Return stacked child parameters for homogeneous ignored-wrapper mixtures."""
        from pysp.stats.stacked import stacked_component_params
        child_dists = [dist.dist for dist in dists]
        if all(isinstance(dist, NullDistribution) for dist in child_dists):
            return {'child_route': None, 'num_components': len(dists)}
        try:
            child_route = stacked_component_params(child_dists, engine)
        except ValueError as exc:
            raise ValueError('Ignored child %s is not stackable: %s' %
                             (type(child_dists[0]).__name__, exc))
        return {'child_route': child_route, 'num_components': len(dists)}

    @classmethod
    def backend_stacked_log_density(cls, x: E, params: Dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of delegated child log densities."""
        from pysp.stats.stacked import stacked_component_log_density
        child_route = params['child_route']
        if child_route is None:
            return engine.zeros((int(x), int(params['num_components'])))
        return stacked_component_log_density(x, child_route, engine)

    @classmethod
    def backend_stacked_sufficient_statistics(cls, x: E, weights: Any,
                                             params: Dict[str, Any], engine: Any) -> Any:
        """Return empty legacy statistics for each ignored component."""
        return tuple(None for _ in range(int(params['num_components'])))

    def sampler(self, seed: Optional[int] = None) -> 'IgnoredSampler':
        """Return a sampler for drawing observations from this distribution."""
        return IgnoredSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'IgnoredEstimator':
        """Return an estimator for fitting this distribution from data."""
        return IgnoredEstimator(dist=self.dist, name=self.name)

    def dist_to_encoder(self) -> 'IgnoredDataEncoder':
        """Return the data encoder used by this distribution for vectorized methods."""
        return IgnoredDataEncoder(encoder=self.dist.dist_to_encoder())


class IgnoredSampler(DistributionSampler):

    def __init__(self, dist: IgnoredDistribution, seed: Optional[int] = None) -> None:
        self.dist_sampler = dist.dist.sampler(seed)
        self.null_sampler = isinstance(self.dist_sampler, NullSampler)

    def sample(self, size: Optional[int] = None):
        if self.null_sampler:
            if size is None:
                return None
            else:
                return [None]*size
        else:
            return self.dist_sampler.sample(size=size)


class IgnoredAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, encoder: Optional[DataSequenceEncoder] = NullDataEncoder(), name: Optional[str] = None) -> None:
        self.encoder = encoder if encoder is not None else NullDataEncoder()
        self.name = name

    def update(self, x: T, weight: float, estimate: Optional[IgnoredDistribution]) -> None:
        pass

    def seq_update(self, x: E, weights: np.ndarray, estimate: Optional[IgnoredDistribution]) -> None:
        pass

    def initialize(self, x: T, weight: float, rng: Optional[RandomState]) -> None:
        pass

    def seq_initialize(self, x: E, weight: np.ndarray, rng: Optional[RandomState]) -> None:
        pass

    def combine(self, suff_stat: Any) -> 'IgnoredAccumulator':
        return self

    def value(self) -> None:
        return None

    def from_value(self, x: Any) -> 'IgnoredAccumulator':
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        pass

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        pass

    def acc_to_encoder(self) -> 'IgnoredDataEncoder':
        return IgnoredDataEncoder(encoder=self.encoder)


class IgnoredAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, encoder: Optional[DataSequenceEncoder] = NullDataEncoder(), name: Optional[str] = None):
        self.encoder = encoder if encoder is not None else NullDataEncoder()
        self.name = name

    def make(self) -> 'IgnoredAccumulator':
        return IgnoredAccumulator(encoder=self.encoder, name=self.name)


class IgnoredEstimator(ParameterEstimator):

    def __init__(self, dist: Optional[SequenceEncodableProbabilityDistribution] = NullDistribution(),
                 pseudo_count: Optional[float] = None, suff_stat: Optional[Any] = None,
                 keys: Optional[str] = None,
                 name: Optional[str] = None) -> None:
        """IgnoredEstimator object for consistency in estimation step.

        Args:
            dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution to be ignored.
            pseudo_count (Optional[float]): Place holder for consistency.
            suff_stat (Optional[Any]): Place holder for consistency.
            keys (Optional[str]): Place holder for consistency.
            name (Optional[str]): Set name for object instance.

        Args:
            dist (SequenceEncodableProbabilityDistribution): Distribution to be ignored.
            pseudo_count (Optional[float]): Place holder for consistency.
            suff_stat (Optional[Any]): Place holder for consistency.
            keys (Optional[str]): Place holder for consistency.
            name (Optional[str]): Set name for object instance.

        """
        self.dist = dist if dist is not None else NullDistribution
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name

    def accumulator_factory(self):
        return IgnoredAccumulatorFactory(self.dist.dist_to_encoder(), name=self.name)

    def estimate(self, nobs: Optional[float], suff_stat: Any) -> IgnoredDistribution:
        return IgnoredDistribution(self.dist, name=self.name)


class IgnoredDataEncoder(DataSequenceEncoder):

    def __init__(self, encoder: Optional[DataSequenceEncoder] = NullDataEncoder()) -> None:
        self.encoder = encoder if encoder is not None else NullDataEncoder()
        self.null = isinstance(self.encoder, NullDataEncoder)

    def __str__(self) -> str:
        return 'IgnoredDataEncoder(dist=' + str(self.encoder) + ')'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, IgnoredDataEncoder):
            return other.encoder == self.encoder
        else:
            return False

    def seq_encode(self, x: Sequence[T]) -> Any:
        enc_data = self.encoder.seq_encode(x)
        return enc_data



