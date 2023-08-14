"""Create, estimate, and sample from a mixture over a dirac delta at v and a length distribution.

Defines the DiracLengthMixtureDistribution, DiracLengthMixtureSampler, DiracLengthMixtureAccumulatorFactory,
DiracLengthMixtureAccumulator, DiracLengthMixtureEstimator, and the DiracLengthMixtureDataEncoder classes for use with
pysparkplug.

The DiracLengthMixtureDistribution is defined by the density of the form,

P(Y) = p*P_1(Y) + (1-p)*Delta_{v}(Y),

where P_1() is a length distribution with support on non-negative integers, or a subset of them, and Delta_{v}(x) = 1
if x = v, else 0.

"""
from typing import List, Union, Tuple, Any, Optional, TypeVar, Sequence, Dict

import numpy as np
from numpy.random import RandomState

from pysp.arithmetic import maxrandint
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder


E0 = TypeVar('E0')  # Type of encoded data.
E = Tuple[int, np.ndarray, np.ndarray, E0]
SS0 = TypeVar('SS0')  # Type of component suff_stat
key_type = Union[Tuple[str, str], Tuple[None, None]]


class DiracLengthMixtureDistribution(SequenceEncodableProbabilityDistribution):
    """DiracLengthMixtureDistribution object defined by a length distribution, choice of dirac value, and p.

    Args:
        p (float): Probability of being drawn from length distribution. Must be between 0 and 1.
        len_dist (SequenceEncodableProbabilityDistribution): Distribution with support on non-negative integers.
        name (Optional[str]): Set name for object instance.

    Attributes:
        p (float): Probability of being drawn from length distribution. Must be between 0 and 1.
        len_dist (SequenceEncodableProbabilityDistribution): Distribution with support on non-negative integers.
        name (Optional[str]): Name for object instance.

    """

    def __init__(self, len_dist: SequenceEncodableProbabilityDistribution, p: float, v: int = 0,
                 name: Optional[str] = None):
        if not 0 < p <= 1:
            raise Exception('p must be between (0,1].')
        with np.errstate(divide='ignore'):
            self.p = p
            self.v = v
            self.log_p = np.log(p)
            self.log_1p = np.log1p(-p)
            self.len_dist = len_dist
            self.name = name

    def __str__(self) -> str:
        s1 = repr(self.len_dist)
        s2 = repr(self.p)
        s3 = repr(self.v)
        s4 = repr(self.name)

        return 'LengthDiracMixtureDistribution(len_dist=%s, p=%s, v=%s, name=%s)' % (s1, s2, s3, s4)

    def density(self, x: int) -> float:
        """Evaluate density of length Dirac mixture distribution at observation x.

        See log_density() for details.

        Args:
            x (int): Integer value.

        Returns:
            Density at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        """Evaluate the log-density of length Dirac mixture distribution at observation x.

        log(P(x)) = log( p*P_1(x) + (1-p)*Delta_{v}(x) ),

        Args:
            x (int): Integer value.

        Returns:
            log-density at x.

        """
        rv0 = self.log_p + self.len_dist.log_density(x)

        if x == self.v:
            c1 = self.log_1p
            if c1 > rv0:
                rv = np.log1p(np.exp(rv0-c1)) + c1
            else:
                rv = np.log1p(np.exp(c1-rv0)) + rv0
        else:
            rv = rv0

        return rv

    def component_log_density(self, x: int) -> np.ndarray:
        rv = np.zeros(2, dtype=np.float64)
        rv[0] = self.log_p + self.len_dist.log_density(x)
        if x == self.v:
            rv[1] = -np.inf
        return rv

    def posterior(self, x: int) -> np.ndarray:
        comp_log_density = self.component_log_density(x)
        if comp_log_density[1] == -np.inf:
            return np.array([1, 0], dtype=np.float64)
        else:
            comp_log_density[0] += self.log_p
            comp_log_density[1] += self.log_1p

        max_val = np.max(comp_log_density)

        comp_log_density -= max_val
        np.exp(comp_log_density, out=comp_log_density)
        comp_log_density /= comp_log_density.sum()

        return comp_log_density

    def seq_component_log_density(self, x: E) -> np.ndarray:
        sz, idx_v, idx_nv, enc_x = x
        ll_mat = np.zeros((sz, 2), dtype=np.float64)

        ll_mat[:, 0] += self.len_dist.seq_log_density(enc_x)
        ll_mat[idx_nv, 1] = -np.inf

        return ll_mat

    def seq_log_density(self, x: E) -> np.ndarray:
        sz, idx_v, idx_nv, enc_x = x
        ll_mat = np.zeros((sz, 2), dtype=np.float64)

        ll_mat[:, 0] += self.len_dist.seq_log_density(enc_x) + self.log_p
        ll_mat[idx_nv, 1] = -np.inf
        ll_mat[idx_v, 1] += self.log_1p

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
        sz, idx_v, idx_nv, enc_x = x
        rv = np.zeros((sz, 2), dtype=np.float64)

        if len(idx_v) == 0:
            rv[:, 0] += 1.0

        else:
            rv[idx_nv, 0] += 1.0
            ll_mat = rv[idx_v, :]

            ll_mat[:, 1] += self.log_1p
            ll_mat[:, 0] += self.len_dist.seq_log_density(enc_x) + self.log_p

            ll_max = ll_mat.max(axis=1, keepdims=True)
            bad_rows = np.isinf(ll_max.flatten())

            ll_mat[bad_rows, :] = np.array([self.log_p, self.log_1p], dtype=np.float64)
            ll_max[bad_rows] = np.max(np.asarray([self.log_p, self.log_1p]))
            ll_mat -= ll_max

            np.exp(ll_mat, out=ll_mat)
            np.sum(ll_mat, axis=1, keepdims=True, out=ll_max)
            ll_mat /= ll_max

            rv[idx_v, :] = ll_mat

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'DiracLengthMixtureSampler':
        return DiracLengthMixtureSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'DiracLengthMixtureEstimator':

        if pseudo_count is not None:
            est = self.len_dist.estimator(pseudo_count)
            return DiracLengthMixtureEstimator(estimator=est, v=self.v, pseudo_count=pseudo_count,
                                               suff_stat = self.p,
                                               name=self.name)
        else:
            est = self.len_dist.estimator()
            return DiracLengthMixtureEstimator(estimator=est, v=self.v, name=self.name)

    def dist_to_encoder(self) -> 'DiracLengthMixtureDataEncoder':
        """Returns a MixtureDataEncoder object for encoding sequences of iid observations from MixtureDistribution."""
        len_dist_encoder = self.len_dist.dist_to_encoder()
        return DiracLengthMixtureDataEncoder(encoder=len_dist_encoder, v=self.v)


class DiracLengthMixtureSampler(DistributionSampler):

    def __init__(self, dist: DiracLengthMixtureDistribution, seed: Optional[int] = None) -> None:
        """DiracLengthMixtureSampler used to generate samples.

        Args:
            dist (DiracMixtureDistribution): Assign DiracLengthMixtureDistribution to draw samples from.
            seed (Optional[int]): Seed to set for sampling with RandomState.

        Attributes:
            rng (RandomState): Seeded RandomState for sampling.
            p (np.ndarray): Prob of drawing from length distribution.
            len_dist_sampler (DistributionSampler): Sampler for the length distribution.
            v (int): Dirac location.
.
        """
        rng_loc = np.random.RandomState(seed)
        self.rng = np.random.RandomState(rng_loc.randint(0, maxrandint))
        self.p = np.exp(dist.log_p)
        self.len_dist_sampler = dist.len_dist.sampler(seed=self.rng.randint(maxrandint))
        self.v = dist.v

    def sample(self, size: Optional[int] = None) -> Union[List[int], int]:
        """Draw iid samples from a DiracLengthMixture distribution.

        Args:
            size (Optional[int]): Number of iid samples to draw.

        Returns:
            Int or List[int] depending on size = None or size (int).

        """
        comp_state = self.rng.binomial(n=1, size=size, p=self.p)

        if size is None:
            if comp_state == 0:
                return self.v
            else:
                return self.len_dist_sampler.sample()
        else:
            rv = np.zeros(size, dtype=np.int32)
            rv.fill(self.v)

            idx = np.flatnonzero(comp_state == 1)
            if len(idx) > 0:
                rv[idx] = np.asarray(self.len_dist_sampler.sample(size=len(idx)), dtype=np.int32)
            return list(rv)


class DiracLengthMixtureAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, accumulator: SequenceEncodableStatisticAccumulator, v: int = 0,
                 keys: Tuple[Optional[str], Optional[str]] = (None, None), name: Optional[str] = None):
        self.accumulator = accumulator
        self.comp_counts = np.zeros(2, dtype=float)
        self.weight_key = keys[0]
        self.comp_key = keys[1]
        self.v = v
        self.name = name

        ### Initializer seeds
        self._init_rng: bool = False
        self._acc_rng: Optional[RandomState] = None
        self._w_rng: Optional[RandomState] = None

    def seq_update(self, x: E, weights: np.ndarray, estimate: 'DiracLengthMixtureDistribution'):
        sz, idx_v, idx_nv, enc_x = x
        ll_mat = np.zeros((sz, 2), dtype=np.float64)

        if len(idx_v) == 0:
            ll_mat[:, 0] += weights

        else:
            ll_mat[:, 0] += estimate.len_dist.seq_log_density(enc_x) + estimate.log_p
            ll_mat[idx_nv, 0] = weights[idx_nv].copy()

            rv = ll_mat[idx_v, :]
            rv[:, 1] += estimate.log_1p

            rv_max = rv.max(axis=1, keepdims=True)
            bad_rows = np.isinf(rv.flatten())

            if np.any(bad_rows):
                rv[bad_rows, :] = np.array([estimate.log_p, estimate.log_1p], dtype=np.float64)
                rv_max[bad_rows] = np.max(np.asarray([estimate.log_p, estimate.log_1p]))
            rv -= rv_max

            np.exp(rv, out=rv)
            np.sum(rv, axis=1, keepdims=True, out=rv_max)
            np.divide(weights[idx_v, None], rv_max, out=rv_max)
            rv *= rv_max

            ll_mat[idx_v, :] = rv

        self.comp_counts += ll_mat.sum(axis=0)
        self.accumulator.seq_update(enc_x, ll_mat[:, 0], estimate.len_dist)

    def update(self, x: int, weight: float, estimate: 'DiracLengthMixtureDistribution') -> None:
        posterior = estimate.posterior(x)
        posterior *= weight
        self.comp_counts += posterior

        self.accumulator.update(x, posterior[0], estimate.len_dist)

    def _rng_initialize(self, rng: RandomState):
        seeds = rng.randint(2 ** 31, size=2)
        self._acc_rng = RandomState(seed=seeds[0])
        self._w_rng = RandomState(seed=rng.randint(maxrandint))
        self._init_rng = True

    def initialize(self, x: int, weight: float, rng: np.random.RandomState):
        if not self._init_rng:
            self._rng_initialize(rng)

        if x == self.v:
            ww = self._w_rng.dirichlet(np.ones(2)/4)
            self.accumulator.initialize(x, weight*ww[0], rng=self._acc_rng)
            self.comp_counts += ww
        else:
            self.accumulator.initialize(x, weight, rng=self._acc_rng)
            self.comp_counts[0] += weight

    def seq_initialize(self, x: E, weights: np.ndarray, rng: np.random.RandomState) -> None:

        sz, xi_v, xi_nv, enc_x = x

        if not self._init_rng:
            self._rng_initialize(rng)

        sz = len(weights)
        keep_len = len(xi_v)
        ww = np.ones((sz, 2))

        if keep_len > 0:
            ww[xi_v, :] = self._w_rng.dirichlet(alpha=np.ones(2) / 4, size=keep_len)

        ww *= np.reshape(weights, (sz, 1))

        self.accumulator.seq_initialize(enc_x, weights=ww[:, 0], rng=self._acc_rng)
        self.comp_counts[0] += np.sum(ww[:, 0])
        self.comp_counts[1] += np.sum(ww[xi_v, 1])

    def combine(self, suff_stat: Tuple[np.ndarray, SS0]) -> 'DiracLengthMixtureAccumulator':
        self.comp_counts += suff_stat[0]
        self.accumulator.combine(suff_stat[1])

        return self

    def value(self) -> Tuple[np.ndarray, Any]:
        return self.comp_counts, self.accumulator.value()

    def from_value(self, x: Tuple[np.ndarray, SS0]) -> 'DiracLengthMixtureAccumulator':
        self.comp_counts = x[0]
        self.accumulator.from_value(x[1])

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                stats_dict[self.weight_key] += self.comp_counts
            else:
                stats_dict[self.weight_key] = self.comp_counts

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                stats_dict[self.comp_key].combine(self.accumulator.value())
            else:
                stats_dict[self.comp_key] = self.accumulator

        self.accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                self.comp_counts = stats_dict[self.weight_key]

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                acc = stats_dict[self.comp_key]
                self.accumulator = acc

        self.accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'DiracLengthMixtureDataEncoder':
        acc_encoder = self.accumulator.acc_to_encoder()
        return DiracLengthMixtureDataEncoder(encoder=acc_encoder, v=self.v)


class DiracLengthMixtureAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, factory: StatisticAccumulatorFactory, v: int = 0,
                 keys: Tuple[Optional[str], Optional[str]] = (None, None), name: Optional[str] = None) -> None:
        self.factory = factory
        self.v = v
        self.keys = keys
        self.name = name

    def make(self) -> 'DiracLengthMixtureAccumulator':
        return DiracLengthMixtureAccumulator(accumulator=self.factory.make(), v=self.v, keys=self.keys, name=self.name)


class DiracLengthMixtureEstimator(ParameterEstimator):

    def __init__(self, estimator: ParameterEstimator, v: int = 0, fixed_p: Optional[int] = None,
                 suff_stat: Optional[float] = None, pseudo_count: Optional[float] = None,
                 name: Optional[str] = None, keys: Tuple[Optional[str], Optional[str]] = (None, None)):
        self.estimator = estimator
        self.v = v
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name
        self.fixed_p_vec = np.asarray([fixed_p, 1-fixed_p]) if fixed_p is not None and 0 < fixed_p <= 1 else None

    def accumulator_factory(self) -> 'DiracLengthMixtureAccumulatorFactory':
        factory = self.estimator.accumulator_factory()
        return DiracLengthMixtureAccumulatorFactory(factory=factory, v=self.v, keys=self.keys, name=self.name)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[np.ndarray, SS0]) -> 'DiracLengthMixtureDistribution':
        counts, comp_suff_stats = suff_stat

        len_dist = self.estimator.estimate(counts[0], comp_suff_stats)

        if self.fixed_p_vec is not None:
            p = self.fixed_p_vec[0]

        elif self.pseudo_count is not None and self.suff_stat is None:
            w = counts + self.pseudo_count / 2
            w /= w.sum()
            p = w[0]

        elif self.pseudo_count is not None and self.suff_stat is not None:
            ss = np.array([self.suff_stat, 1-self.suff_stat])
            w = (counts + ss*self.pseudo_count) / (counts.sum() + self.pseudo_count)
            p = w[0]

        else:
            nobs_loc = counts.sum()

            if nobs_loc == 0:
                p = 0.5
            else:
                w = counts / counts.sum()
                p = w[0]

        return DiracLengthMixtureDistribution(len_dist=len_dist, p=p, v=self.v, name=self.name)


class DiracLengthMixtureDataEncoder(DataSequenceEncoder):

    def __init__(self, encoder: DataSequenceEncoder, v: int = 0) -> None:
        self.encoder = encoder
        self.v = v

    def __str__(self) -> str:
        return 'DiracMixtureDataEncoder(encoder=%s, v=%s)' % (repr(self.encoder), repr(self.v))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DiracLengthMixtureDataEncoder):
            if other.encoder == self.encoder:
                return other.v == self.v
            else:
                return False
        else:
            return False

    def seq_encode(self, x: Sequence[int]) -> Tuple[int, np.ndarray, np.ndarray, Any]:
        x = np.asarray(x, dtype=np.int32)
        xi_v = np.flatnonzero(x == self.v).astype(np.int32)
        xi_nv = np.flatnonzero(x != self.v).astype(np.int32)

        return len(x), xi_v, xi_nv, self.encoder.seq_encode(x)
