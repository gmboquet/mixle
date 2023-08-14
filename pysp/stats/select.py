"""Create, estimate, and sample from a select distribution.

Defines the SelectDistribution, SelectSampler, SelectAccumulatorFactory, SelectAccumulator,
SelectEstimator, and the SelectDataEncoder classes for use with pysparkplug.

The SelectDistribution samples from a set of SequenceEncodableProbabilityDistribution objects. The a choice function
maps an observation a distribution from the set of distributions.

"""
import numpy as np
from numpy.random import RandomState
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator, DistributionSampler, DataSequenceEncoder, StatisticAccumulatorFactory

from typing import Callable, Dict, Tuple, Any, Optional, Sequence, TypeVar, List

T = TypeVar('T')


class SelectDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, dists: Sequence[SequenceEncodableProbabilityDistribution],
                 choice_function: Callable[[T], int]) -> None:
        self.dists = dists
        self.choice_function = choice_function
        self.count = len(dists)

    def __str__(self):
        return 'SelectDistribution(' + ','.join([str(u) for u in self.dists]) + ')'

    def density(self, x: T) -> float:
        idx = self.choice_function(x)
        return self.dists[idx].density(x)

    def log_density(self, x: T) -> float:
        idx = self.choice_function(x)
        return self.dists[idx].log_density(x)

    def seq_log_density(self, x: Sequence[T]) -> np.ndarray:
        xi, idx, enc_tuple = x
        rv = np.zeros(len(xi))
        for i in range(len(idx)):
            rv[xi[i]] = self.dists[i].seq_log_density(enc_tuple[i])
        return rv

    def sampler(self, seed: Optional[int] = None) -> 'SelectSampler':
        return SelectSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'SelectEstimator':
        return SelectEstimator([d.estimator(pseudo_count=pseudo_count) for d in self.dists], self.choice_function)

    def dist_to_encoder(self) -> 'SelectDataEncoder':
        encoders = [d.dist_to_encoder() for d in self.dists]
        return SelectDataEncoder(encoders=encoders, choice_function=self.choice_function)


class SelectSampler(DistributionSampler):
    def __init__(self, dist: SelectDistribution, seed: Optional[int] = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)
        self.dist_samplers = [d.sampler(seed=self.rng.randint(maxint)) for d in dist.dists]

    def sample(self, size: Optional[int] = None):

        if size is None:
            return tuple([d.sample(size=size) for d in self.dist_samplers])
        else:
            return zip(*[d.sample(size=size) for d in self.dist_samplers])


class SelectEstimatorAccumulator(SequenceEncodableStatisticAccumulator):
    def __init__(self, accumulators: Sequence[SequenceEncodableStatisticAccumulator],
                 choice_function: Callable[[T], int]) -> None:
        self.accumulators = accumulators
        self.choice_function = choice_function
        self.weights = [zero] * len(accumulators)
        self.count = len(accumulators)

        self._rng_init = False
        self._acc_rng: Optional[List[RandomState]] = None

    def update(self, x: T, weight: float, estimate: Optional[SelectDistribution]) -> None:
        # cf  = pickle.loads(self.choice_function)
        idx = self.choice_function(x)
        self.accumulators[idx].update(x, weight, None)
        self.weights[idx] += weight

    def _rng_initialize(self, rng: RandomState) -> None:
        self._acc_rng = [RandomState(seed=rng.randint(0, maxrandint)) for xx in range(self.count)]
        self._rng_init = True

    def initialize(self, x: T, weight: float, rng: RandomState) -> None:
        if not self._rng_init:
            self._rng_initialize(rng)

        idx = self.choice_function(x)
        self.accumulators[idx].initialize(x, weight, self._acc_rng[idx])
        self.weights[idx] += weight

    def seq_update(self, x, weights: np.ndarray, estimate: SelectDistribution) -> None:
        xi, idx, enc_tuple = x
        for i in range(len(idx)):
            w = weights[xi[i]]
            self.accumulators[i].seq_update(enc_tuple[i], w, estimate.dists[i])
            self.weights[i] += np.sum(w)

    def seq_initialize(self, x, weights: np.ndarray, rng: RandomState) -> None:
        if not self._rng_init:
            self._rng_initialize(rng)

        xi, idx, enc_tuple = x
        for i in range(len(idx)):
            w = weights[xi[i]]
            self.accumulators[i].seq_initialize(enc_tuple[i], w, self._acc_rng[i])
            self.weights[i] += np.sum(w)

    def combine(self, suff_stat) -> 'SelectEstimatorAccumulator':
        for i in range(0, self.count):
            self.weights[i] += suff_stat[i][0]
            self.accumulators[i].combine(suff_stat[i][1])

        return self

    def value(self):
        return zip(self.weights, [x.value() for x in self.accumulators])

    def from_value(self, x) -> 'SelectEstimatorAccumulator':
        for i, u in enumerate(x):
            self.weights[i] = u[0]
            self.accumulators[i].from_value(u[1])

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        pass

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        pass

    def acc_to_encoder(self) -> 'SelectDataEncoder':
        encoders = [acc.acc_to_encoder() for acc in self.accumulators]
        return SelectDataEncoder(encoders=encoders, choice_function=self.choice_function)


class SelectEstimatorAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, estimators, choice_function):
        self.estimators = estimators
        self.choice_function = choice_function

    def make(self):
        return SelectEstimatorAccumulator([x.accumulatorFactory().make() for x in self.estimators],
                                          self.choice_function)

class SelectEstimator(ParameterEstimator):
    def __init__(self, estimators, choice_function):
        self.estimators = estimators
        self.choice_function = choice_function
        self.count = len(estimators)

    def accumulator_factory(self):
        return SelectEstimatorAccumulatorFactory(self.estimators, self.choice_function)

    def estimate(self, nobs, suff_stat):
        return (SelectDistribution([est.estimate(ss[0], ss[1]) for est, ss in zip(self.estimators, suff_stat)],
                                   self.choice_function))


class SelectDataEncoder(DataSequenceEncoder):

    def __init__(self, encoders: Sequence[DataSequenceEncoder], choice_function: Callable[[T], int]) -> None:
        self.encoders = encoders
        self.choice_function = choice_function

    def __eq__(self, other: object) -> bool:
        ### Asssumes that the choice functions of each encoder are equal
        if isinstance(other, SelectDataEncoder):
            for i, encoder in enumerate(self.encoders):
                if other.encoders[i] != encoder:
                    return False

            return True

        else:
            return False

    def seq_encode(self, x: Sequence[T]) -> Tuple[Tuple[np.ndarray, ...], Tuple[int, ...], Tuple[Any, ...]]:
        cnt = 0
        idx_dict = dict()

        for i, xx in enumerate(x):
            idx = self.choice_function(xx)
            if idx not in idx_dict:
                idx_dict[idx] = [[], []]
            idx_dict[idx][1].append(xx)
            idx_dict[idx][0].append(i)
            cnt += 1

        idx_keys = []
        idx_xi = []
        idx_enc_vals = []

        for keys, vals in idx_dict.items():
            idx_keys.append(keys)
            idx_xi.append(np.asarray(vals[0]))
            idx_enc_vals.append(self.encoders[keys].seq_encode(vals[1]))

        return tuple(idx_xi), tuple(idx_keys), tuple(idx_enc_vals)