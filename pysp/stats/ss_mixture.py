"""Create, estimate, and sample from a semi-supervised mixture distribution.

Defines the SemiSupervisedMixtureDistribution, SemiSupervisedMixtureSampler, SemiSupervisedMixtureAccumulatorFactory,
SemiSupervisedMixtureEstimatorAccumulator, SemiSupervisedMixtureEstimator, and the SemiSupervisedMixtureDataEncoder
classes for use with pysparkplug.

Data type (Tuple[T, Optional[Sequence[Tuple[int, float]]]): T is the data type of the mixture components. The optional
Sequence of tuples contain labels for the observations coming from the component (0,1,2,...num_components-1) and an
associated probability for the label.

The likelihood for an observation x = (y, prior) is simply a mixture distribution with the weights of the mixture
re-weighted to account for the prior knowledge that x was observaed from components in prior with probs in prior as well.

If no prior is provided, the likelihood is simply a mixture.

Note: seq_initialize() is not well implemented.

"""
import numpy as np
from numpy.random import RandomState

import pysp.utils.vector as vec
from pysp.arithmetic import *
from pysp.arithmetic import maxrandint
from pysp.stats.null_dist import NullDistribution, NullAccumulator, NullEstimator, NullDataEncoder, \
    NullAccumulatorFactory
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator, DistributionSampler, DataSequenceEncoder, StatisticAccumulatorFactory

from typing import Sequence, Tuple, List, Dict, Any, Optional, TypeVar, Union

T0 = TypeVar('T0')  # Data type
T1 = TypeVar('T1')  # Prior type

E0 = TypeVar('E0')  # Encoded data type components
E1 = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]  # Encoded prior type
E = Tuple[int, E0, Tuple[E1, np.ndarray, np.ndarray], Sequence[Tuple[T0, Optional[Sequence[Tuple[int, T1]]]]]]

SS0 = TypeVar('SS0')  # Suff-stat type from components


class SemiSupervisedMixtureDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, components: Sequence[SequenceEncodableProbabilityDistribution],
                 w: Union[List[float], np.ndarray], name: Optional[str] = None) -> None:
        """Create SemiSupervisedMixtureDistribution object.

        Args:
            components (Sequence[SequenceEncodableProbabilityDistribution]): Mixture components.
            w ( Union[List[float], np.ndarray]): Mixture weights. Should sum to 1.0
            name (Optional[str]): Set name for object.

        Attributes:
            components (Sequence[SequenceEncodableProbabilityDistribution]): Mixture components.
            num_components (int): Number of mixture components.
            zw (np.ndarray): Bool numpy array, True where weights are 0.0.
            log_w (np.ndarray): Log of weights. Set to -np.inf where weights are 0.
            w (np.ndarray): Mixture weights. Should sum to 1.0.
            name (Optional[str]): Set name for object.

        """
        self.components = components
        self.num_components = len(components)
        self.w = np.asarray(w)
        self.zw = (self.w == 0.0)
        self.log_w = np.log(w + self.zw)
        self.log_w[self.zw] = -np.inf
        self.name = name

    def __str__(self) -> str:
        return 'SemiSupervisedMixtureDistribution([%s], [%s], name=%s)' % (
            ','.join([str(u) for u in self.components]), ','.join(map(str, self.w)), ','.join(repr(self.name)))

    def density(self, x: Tuple[T0, Optional[Sequence[Tuple[int, T1]]]]) -> float:
        return exp(self.log_density(x))

    def log_density(self, x: Tuple[T0, Optional[Sequence[Tuple[int, T1]]]]) -> float:

        datum, prior = x
        if prior is None:
            return vec.log_sum(np.asarray([u.log_density(datum) for u in self.components]) + self.log_w)
        else:
            w_loc = np.zeros(self.num_components)
            h_loc = np.zeros(self.num_components, dtype=bool)
            i_loc = np.zeros(self.num_components, dtype=int)

            for idx, val in prior:
                w_loc[idx] += np.log(val)
                h_loc[idx] = True
                i_loc[idx] = idx

            w_loc[h_loc] += self.log_w[h_loc]
            w_loc = vec.log_posterior(w_loc[h_loc])

            return vec.log_sum(
                np.asarray([self.components[i].log_density(datum) for i in np.flatnonzero(h_loc)]) + w_loc)

    def posterior(self, x: Tuple[T0, Optional[Sequence[Tuple[int, T1]]]]) -> np.ndarray:
        datum, prior = x

        if prior is None:
            rv = vec.posterior(np.asarray([u.log_density(datum) for u in self.components]) + self.log_w)
        else:

            w_loc = np.zeros(self.num_components)
            h_loc = np.zeros(self.num_components, dtype=bool)

            for idx, val in prior:
                w_loc[idx] += np.log(val)
                h_loc[idx] = True

            w_loc[h_loc] += self.log_w[h_loc]
            for i in np.flatnonzero(h_loc):
                w_loc[i] += self.components[i].log_density(datum)

            w_loc[h_loc] = vec.posterior(w_loc[h_loc])
            rv = w_loc

        return rv

    def seq_log_density(self, x: E) -> np.ndarray:

        sz, enc_data, (enc_prior, enc_prior_sum, enc_prior_flag), _ = x
        ll_mat = np.zeros((sz, self.num_components))
        ll_mat.fill(-np.inf)

        norm_const = np.bincount(enc_prior[0], weights=(enc_prior[2] * self.w[enc_prior[1]]), minlength=sz)
        norm_const = np.log(norm_const[enc_prior_flag])

        ll_mat[~enc_prior_flag, :] = self.log_w
        ll_mat[enc_prior[0], enc_prior[1]] = enc_prior[3] + self.log_w[enc_prior[1]]

        for i in range(self.num_components):
            if not self.zw[i]:
                ll_mat[:, i] += self.components[i].seq_log_density(enc_data)
                ll_mat[enc_prior_flag, i] -= norm_const

        ll_max = ll_mat.max(axis=1, keepdims=True)
        good_rows = np.isfinite(ll_max.flatten())

        if np.all(good_rows):
            ll_mat -= ll_max

            np.exp(ll_mat, out=ll_mat)
            ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
            np.log(ll_sum, out=ll_sum)
            ll_sum += ll_max

            return ll_sum.flatten()

        else:
            ll_mat = ll_mat[good_rows, :]
            ll_max = ll_max[good_rows]

            ll_mat -= ll_max
            np.exp(ll_mat, out=ll_mat)
            ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
            np.log(ll_sum, out=ll_sum)
            ll_sum += ll_max
            rv = np.zeros(good_rows.shape, dtype=float)
            rv[good_rows] = ll_sum.flatten()
            rv[~good_rows] = -np.inf

            return rv

    def seq_posterior(self, x: E) -> np.ndarray:

        sz, enc_data, (enc_prior, enc_prior_sum, enc_prior_flag), _ = x
        ll_mat = np.zeros((sz, self.num_components))
        ll_mat.fill(-np.inf)

        norm_const = np.bincount(enc_prior[0], weights=(enc_prior[2] * self.w[enc_prior[1]]), minlength=sz)
        norm_const = np.log(norm_const[enc_prior_flag])

        ll_mat[~enc_prior_flag, :] = self.log_w
        ll_mat[enc_prior[0], enc_prior[1]] = enc_prior[3] + self.log_w[enc_prior[1]]

        for i in range(self.num_components):
            if not self.zw[i]:
                ll_mat[:, i] += self.components[i].seq_log_density(enc_data)
                ll_mat[enc_prior_flag, i] -= norm_const

        ll_max = ll_mat.max(axis=1, keepdims=True)

        bad_rows = np.isinf(ll_max.flatten())

        ll_mat[bad_rows, :] = self.log_w
        ll_max[bad_rows] = np.max(self.log_w)

        ll_mat -= ll_max

        np.exp(ll_mat, out=ll_mat)
        ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
        ll_mat /= ll_sum

        return ll_mat

    def sampler(self, seed: Optional[int] = None) -> 'SemiSupervisedMixtureSampler':
        return SemiSupervisedMixtureSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'SemiSupervisedMixtureEstimator':
        if pseudo_count is not None:
            return SemiSupervisedMixtureEstimator(
                [u.estimator(pseudo_count=1.0 / self.num_components) for u in self.components],
                pseudo_count=pseudo_count, name=self.name)
        else:
            return SemiSupervisedMixtureEstimator([u.estimator() for u in self.components], name=self.name)

    def dist_to_encoder(self) -> 'SemiSupervisedMixtureDataEncoder':
        return SemiSupervisedMixtureDataEncoder(encoder=self.components[0].dist_to_encoder())


class SemiSupervisedMixtureSampler(DistributionSampler):

    def __init__(self, dist: SemiSupervisedMixtureDistribution, seed: Optional[int] = None) -> None:
        rng_loc = RandomState(seed)
        self.rng = RandomState(rng_loc.randint(0, maxrandint))
        self.dist = dist
        self.comp_samplers = [d.sampler(seed=rng_loc.randint(0, maxrandint)) for d in self.dist.components]

    def sample(self, size: Optional[int] = None) -> Union[Sequence[Any], Any]:
        comp_state = self.rng.choice(range(0, self.dist.num_components), size=size, replace=True, p=self.dist.w)

        if size is None:
            return self.comp_samplers[comp_state].sample()
        else:
            return [self.comp_samplers[i].sample() for i in comp_state]


class SemiSupervisedMixtureEstimatorAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, accumulators: Sequence[SequenceEncodableStatisticAccumulator],
                 keys: Optional[Tuple[Optional[str], Optional[str]]] = (None, None),
                 name: Optional[str] = None) -> None:
        self.accumulators = accumulators
        self.num_components = len(accumulators)
        self.comp_counts = np.zeros(self.num_components, dtype=float)
        self.weight_key, self.comp_key = keys if keys is not None else (None, None)
        self.name = name

        self._init_rng = False
        self._acc_rng = None
        self._w_rng = None

    def update(self, x: Tuple[T0, Optional[Sequence[Tuple[int, T1]]]], weight: float,
               estimate: SemiSupervisedMixtureDistribution) -> None:

        likelihood = estimate.posterior(x)
        datum, prior = x

        self.comp_counts += likelihood * weight

        for i in range(self.num_components):
            self.accumulators[i].update(datum, likelihood[i] * weight, estimate.components[i])

    def _rng_initialize(self, rng: RandomState) -> None:
        if not self._init_rng:

            self._w_rng = RandomState(seed=rng.randint(maxrandint))
            self._prior_rng = RandomState(seed=rng.randint(maxrandint))

            seeds = rng.randint(maxrandint, size=self.num_components)
            self._acc_rng = [RandomState(seed=seeds[i]) for i in range(self.num_components)]

            self._init_rng = True

    def initialize(self, x: Tuple[T0, Optional[Sequence[Tuple[int, T1]]]], weight: float, rng: RandomState) -> None:
        datum, prior = x

        if not self._init_rng:
            self._rng_initialize(rng)

        if prior is None:
            idx = self._prior_rng.choice(self.num_components)
            wc0 = 0.001
            wc1 = wc0 / max((float(self.num_components) - 1.0), 1.0)
            wc2 = 1.0 - wc0

            for i in range(self.num_components):
                w = weight * wc2 if i == idx else wc1
                self.accumulators[i].initialize(datum, w, self._acc_rng[i])
                self.comp_counts[i] += w

        else:
            for i, w in prior:
                ww = weight * w
                self.accumulators[i].initialize(datum, ww, self._acc_rng[i])
                self.comp_counts[i] += ww

    def seq_initialize(self, x: E, weights: np.ndarray, rng: RandomState) -> None:
        sz, enc_data, (enc_prior, enc_prior_sum, enc_prior_flag), xx = x
        for i in range(len(xx)):
            self.initialize(xx[i], weights[i], rng=rng)

    def seq_update(self, x: E, weights: np.ndarray, estimate: SemiSupervisedMixtureDistribution) -> None:

        sz, enc_data, (enc_prior, enc_prior_sum, enc_prior_flag), _ = x
        ll_mat = np.zeros((sz, estimate.num_components))
        ll_mat.fill(-np.inf)

        norm_const = np.bincount(enc_prior[0], weights=(enc_prior[2] * estimate.w[enc_prior[1]]), minlength=sz)
        norm_const = np.log(norm_const[enc_prior_flag])

        ll_mat[~enc_prior_flag, :] = estimate.log_w
        ll_mat[enc_prior[0], enc_prior[1]] = enc_prior[3] + estimate.log_w[enc_prior[1]]

        for i in range(self.num_components):
            ll_mat[:, i] += estimate.components[i].seq_log_density(enc_data)
            ll_mat[enc_prior_flag, i] -= norm_const

        ll_max = ll_mat.max(axis=1, keepdims=True)

        bad_rows = np.isinf(ll_max.flatten())

        ll_mat[bad_rows, :] = estimate.log_w
        ll_max[bad_rows] = np.max(estimate.log_w)

        ll_mat -= ll_max
        np.exp(ll_mat, out=ll_mat)
        ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
        ll_mat /= ll_sum

        for i in range(self.num_components):
            w_loc = ll_mat[:, i] * weights
            self.comp_counts[i] += w_loc.sum()
            self.accumulators[i].seq_update(enc_data, w_loc, estimate.components[i])

    def combine(self, suff_stat: Tuple[np.ndarray, Tuple[SS0, ...]]) -> 'SemiSupervisedMixtureEstimatorAccumulator':

        self.comp_counts += suff_stat[0]
        for i in range(self.num_components):
            self.accumulators[i].combine(suff_stat[1][i])

        return self

    def value(self) -> Tuple[np.ndarray, Tuple[Any, ...]]:
        return self.comp_counts, tuple([u.value() for u in self.accumulators])

    def from_value(self, x: Tuple[np.ndarray, Tuple[SS0, ...]]) -> 'SemiSupervisedMixtureEstimatorAccumulator':
        self.comp_counts = x[0]
        for i in range(self.num_components):
            self.accumulators[i].from_value(x[1][i])
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:

        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                stats_dict[self.weight_key] += self.comp_counts
            else:
                stats_dict[self.weight_key] = self.comp_counts

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                acc = stats_dict[self.comp_key]
                for i in range(len(acc)):
                    acc[i] = acc[i].combine(self.accumulators[i].value())
            else:
                stats_dict[self.comp_key] = self.accumulators

        for u in self.accumulators:
            u.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:

        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                self.comp_counts = stats_dict[self.weight_key]

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                acc = stats_dict[self.comp_key]
                self.accumulators = acc

        for u in self.accumulators:
            u.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'SemiSupervisedMixtureDataEncoder':
        return SemiSupervisedMixtureDataEncoder(encoder=self.accumulators[0].acc_to_encoder())


class SemiSupervisedMixtureEstimatorAccumulatorFactory(StatisticAccumulatorFactory):
    def __init__(self, factories: Sequence[StatisticAccumulatorFactory], dim: int,
                 keys: Optional[Tuple[Optional[str], Optional[str]]] = (None, None),
                 name: Optional[str] = None):
        self.factories = factories
        self.dim = dim
        self.keys = keys if keys is not None else (None, None)
        self.name = name

    def make(self) -> 'SemiSupervisedMixtureEstimatorAccumulator':
        return SemiSupervisedMixtureEstimatorAccumulator([self.factories[i].make() for i in range(self.dim)], self.keys,
                                                         self.name)


class SemiSupervisedMixtureEstimator(ParameterEstimator):
    def __init__(self, estimators: Sequence[ParameterEstimator],
                 suff_stat: Optional[np.ndarray] = None,
                 pseudo_count: Optional[float] = None,
                 keys: Optional[Tuple[Optional[str], Optional[str]]] = (None, None),
                 name: Optional[str] = None) -> None:
        """SemiSupervisedMixtureEstimator object for estimating SemiSupervisedMixtureDistribution from aggregated
            sufficient statistics.

        Args:
            estimators (Sequence[ParameterEstimator]): Sequence of ParameterEstimators objects for the components of
                the mixture. All must be of the same class compatible with data type T.
            suff_stat (Optional[np.ndarray]): Mixture weights for components obtained from prev estimation or for
                regularization.
            pseudo_count (Optional[float]): Re-weight sufficient statistics, i.e. penalize sufficient statistics.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Set keys for the weights and components.
            name (Optional[str]): Set name for object.

        Attributes:
            estimators (Sequence[ParameterEstimator]): Sequence of ParameterEstimators objects for the components of
                the mixture. All must be of the same class compatible with data type T.
            suff_stat (Optional[np.ndarray]): Mixture weights for components obtained from prev estimation or for
                regularization.
            pseudo_count (Optional[float]): Re-weight sufficient statistics, i.e. penalize sufficient statistics.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Set keys for the weights and components.
            name (Optional[str]): Set name for object.

        """
        self.num_components = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys if keys is not None else (None, None)
        self.name = name

    def accumulator_factory(self) -> 'SemiSupervisedMixtureEstimatorAccumulatorFactory':
        est_factories = [u.accumulator_factory() for u in self.estimators]
        return SemiSupervisedMixtureEstimatorAccumulatorFactory(est_factories, self.num_components, self.keys,
                                                                self.name)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[np.ndarray, Tuple[SS0, ...]]) \
            -> 'SemiSupervisedMixtureDistribution':
        num_components = self.num_components
        counts, comp_suff_stats = suff_stat

        components = [self.estimators[i].estimate(counts[i], comp_suff_stats[i]) for i in range(num_components)]

        if self.pseudo_count is not None and self.suff_stat is None:
            p = self.pseudo_count / num_components
            w = counts + p
            w /= w.sum()

        elif self.pseudo_count is not None and self.suff_stat is not None:
            w = (counts + self.suff_stat * self.pseudo_count) / (counts.sum() + self.pseudo_count)
        else:

            nobs_loc = counts.sum()

            if nobs_loc == 0:
                w = np.ones(num_components) / float(num_components)
            else:
                w = counts / counts.sum()

        return SemiSupervisedMixtureDistribution(components, w)

class SemiSupervisedMixtureDataEncoder(DataSequenceEncoder):

    def __init__(self, encoder: DataSequenceEncoder):
        self.encoder = encoder

    def __str__(self) -> str:
        return 'SemiSupervisedMixtureDataEncoder(encoder=' + str(self.encoder) + ')'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SemiSupervisedMixtureDataEncoder):
            return self.encoder == other.encoder
        else:
            return False

    def seq_encode(self, x: Sequence[Tuple[T0, Optional[Sequence[Tuple[int, T1]]]]]) \
            -> Tuple[int, Any, Tuple[E1, np.ndarray, np.ndarray],
                     Sequence[Tuple[T0, Optional[Sequence[Tuple[int, T1]]]]]]:

        prior_comp = []
        prior_idx = []
        prior_val = []
        data = []

        for i, xi in enumerate(x):
            datum, prior = xi
            data.append(datum)
            if prior is not None:
                for prior_entry in prior:
                    prior_idx.append(i)
                    prior_comp.append(prior_entry[0])
                    prior_val.append(prior_entry[1])

        prior_comp = np.asarray(prior_comp, dtype=int)
        prior_idx = np.asarray(prior_idx, dtype=int)
        prior_val = np.asarray(prior_val, dtype=float)

        prior_mat = (prior_idx, prior_comp, prior_val, np.log(prior_val))

        prior_sum = np.bincount(prior_idx, weights=prior_val, minlength=len(x))
        has_prior = prior_sum != 0

        return len(x), self.encoder.seq_encode(data), (prior_mat, prior_sum, has_prior), x
