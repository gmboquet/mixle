
import numpy as np
from numpy.random import RandomState

from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator, DistributionSampler, DataSequenceEncoder, StatisticAccumulatorFactory
from pysp.stats.null_dist import NullDistribution, NullAccumulator, NullEstimator, NullDataEncoder, \
    NullAccumulatorFactory
from typing import Sequence, Optional, Union, Any, Tuple, List, TypeVar, Dict

T = Tuple[Union[Sequence[int], np.ndarray], Union[Sequence[int], np.ndarray]]
E1 = TypeVar('E1') ## encoded type for init
E = Tuple[int, np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray], Optional[E1]]
SS1 = TypeVar('SS1') ## suff-stat of init_dist


class IntegerStepBernoulliEditDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, log_edit_pmat: Union[Sequence[Tuple[float, float]], np.ndarray],
                 init_dist: Optional[SequenceEncodableProbabilityDistribution] = NullDistribution,
                 name: Optional[str] = None) -> None:
        num_vals = len(log_edit_pmat)
        self.name = name
        self.init_dist = init_dist if init_dist is not None else NullDistribution()
        self.num_vals = num_vals

        pmat = np.asarray(log_edit_pmat, dtype=np.float64).copy()
        if pmat.shape[1] == 2:
            log_pmat = np.zeros((num_vals, 4), dtype=np.float64)
            log_pmat[:, 0] = np.log1p(-np.exp(pmat[:, 0]))  # p_mat(missing | missing) = 1 - p_mat(present | missing)
            log_pmat[:, 1] = np.log1p(-np.exp(pmat[:, 1]))  # p_mat(missing | present) = 1 - p_mat(present | present)
            log_pmat[:, 2] = pmat[:, 0]  # p_mat(present | missing)
            log_pmat[:, 3] = pmat[:, 1]  # p_mat(present | present)
        else:
            log_pmat = pmat

        self.orig_log_edit_pmat = pmat
        self.log_edit_pmat = log_pmat
        self.log_nsum = self.log_edit_pmat[
            np.isfinite(self.log_edit_pmat[:, 0]), 0].sum()  # sum [ln p_mat(missing | missing)]
        self.log_dvec = self.log_edit_pmat[:, 1:] - self.log_edit_pmat[:, 0,
                                                    None]  # ln p_mat (?? | ??) - ln p_mat(missing | missing)

    def __str__(self) -> str:
        s1 = repr(list(map(list, self.orig_log_edit_pmat)))
        s2 = repr(self.init_dist)
        s3 = repr(self.name)
        return 'IntegerStepBernoulliEditDistribution(%s, init_dist=%s, name=%s)' % (s1, s2, s3)

    def density(self, x: T) -> float:
        return exp(self.log_density(x))

    def log_density(self, x: T) -> float:
        xx0 = np.asarray(x[0], dtype=int)
        xx1 = np.asarray(x[1], dtype=int)

        in10 = np.isin(xx1, xx0, invert=False)  # xx0 \cap xx1
        in01 = np.isin(xx0, xx1, invert=True)  # xx0 \cap xx1

        yy = np.ones(len(xx1), dtype=int)
        yy[in10] = 2
        rv = self.log_nsum  # ln p_mat(missing | missing) for the empty set
        rv += np.sum(self.log_dvec[xx1[in10], 2])  # ln p_mat(present | present) same stuff that was there
        rv += np.sum(self.log_dvec[xx1[~in10], 1])  # ln p_mat(present | missing) new additions
        rv += np.sum(self.log_dvec[xx0[in01], 0])  # ln p_mat(missing | present) stuff to remove
        # rv = ln p_mat(x[1] | x[0])

        # rv = ln p_mat(x[1] | x[0]) + ln(p_mat(x[0]) = ln p_mat(x[0], x[1])
        rv += self.init_dist.log_density(x[0])

        return rv

    def seq_log_density(self, x: E) -> np.ndarray:
        sz, idx, xs, ys, ym, init_enc = x
        rv = np.bincount(idx, weights=self.log_dvec[xs, ys], minlength=sz)
        rv += self.log_nsum

        rv += self.init_dist.seq_log_density(init_enc)

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'IntegerStepBernoulliEditSampler':
        return IntegerStepBernoulliEditSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'IntegerStepBernoulliEditEstimator':
        return IntegerStepBernoulliEditEstimator(self.num_vals, pseudo_count=pseudo_count, name=self.name)

    def dist_to_encoder(self) -> 'IntegerStepBernoulliEditDataEncoder':
        return IntegerStepBernoulliEditDataEncoder(init_encoder=self.init_dist.dist_to_encoder())

class IntegerStepBernoulliEditSampler(DistributionSampler):

    def __init__(self, dist: IntegerStepBernoulliEditDistribution, seed: Optional[int] = None) -> None:
        self.rng = np.random.RandomState(seed)
        self.dist = dist
        self.init_rng = dist.init_dist.sampler(self.rng.randint(0, maxrandint))
        self.next_rng = np.random.RandomState(self.rng.randint(0, maxrandint))

    def sample(self, size: Optional[int] = None) \
            -> Union[List[Tuple[List[int], List[int]]], Tuple[List[int], List[int]]]:
        if size is None:

            temp = np.log(self.rng.rand(self.dist.num_vals))
            rv = np.zeros(self.dist.num_vals, dtype=bool)
            prev_ob = np.asarray(self.init_rng.sample(), dtype=int)

            rv[temp <= self.dist.log_edit_pmat[:, 2]] = True
            rv[prev_ob] = temp[prev_ob] <= self.dist.log_edit_pmat[prev_ob, 3]

            return list(prev_ob), list(np.flatnonzero(rv))
        else:
            rv = []
            for i in range(size):
                rv.append(self.sample())
            return rv

    def sample_given(self, x: Sequence[Sequence[int]]) -> Sequence[int]:
        temp = np.log(self.rng.rand(self.dist.num_vals))
        rv = np.zeros(self.dist.num_vals, dtype=bool)
        prev_ob = np.asarray(x[-1], dtype=int)

        rv[temp <= self.dist.log_edit_pmat[:, 2]] = True
        rv[prev_ob] = temp[prev_ob] <= self.dist.log_edit_pmat[prev_ob, 3]

        return list(np.flatnonzero(rv))


class IntegerStepBernoulliEditAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, num_vals: int, init_acc: Optional[SequenceEncodableStatisticAccumulator] = NullAccumulator(),
                 keys: Optional[str] = None) -> None:
        self.pcnt = np.zeros((num_vals, 3), dtype=np.float64)
        self.key = keys
        self.num_vals = num_vals
        self.init_acc = init_acc if init_acc is not None else NullAccumulator()
        self.tot_sum = 0.0

        self._acc_rng = None
        self._init_rng = False

    def update(self, x: T, weight: float, estimate: Optional[IntegerStepBernoulliEditDistribution]) -> None:

        xx0 = np.asarray(x[0], dtype=int)
        xx1 = np.asarray(x[1], dtype=int)

        to_add = np.isin(xx1, xx0, invert=False)
        to_rem = np.isin(xx0, xx1, invert=True)

        self.pcnt[xx0[to_rem], 0] += weight
        self.pcnt[xx1[~to_add], 1] += weight
        self.pcnt[xx1[to_add], 2] += weight

        self.tot_sum += weight

        if self.init_acc is not None:
            if estimate is not None:
                self.init_acc.update(x[0], weight, estimate.init_dist)
            else:
                self.init_acc.update(x[0], weight, None)

    def _rng_initialize(self, rng: RandomState) -> None:
        if not self._init_rng:
            self._acc_rng = RandomState(seed=rng.randint(maxrandint))
            self._init_rng = True

    def initialize(self, x: T, weight: float, rng: RandomState) -> None:
        if not self._init_rng:
            self._rng_initialize(rng)

        xx0 = np.asarray(x[0], dtype=int)
        xx1 = np.asarray(x[1], dtype=int)

        to_add = np.isin(xx1, xx0, invert=False)
        to_rem = np.isin(xx0, xx1, invert=True)

        self.pcnt[xx0[to_rem], 0] += weight
        self.pcnt[xx1[~to_add], 1] += weight
        self.pcnt[xx1[to_add], 2] += weight

        self.tot_sum += weight
        self.init_acc.initialize(x[0], weight, rng)

    def seq_update(self, x: E, weights: np.ndarray, estimate: Optional[IntegerStepBernoulliEditDistribution]) -> None:

        sz, idx, xs, ys, ym, init_enc = x

        agg_cnt0 = np.bincount(xs[ym[0]], weights=weights[idx[ym[0]]])
        agg_cnt1 = np.bincount(xs[ym[1]], weights=weights[idx[ym[1]]])
        agg_cnt2 = np.bincount(xs[ym[2]], weights=weights[idx[ym[2]]])

        self.pcnt[:len(agg_cnt0), 0] += agg_cnt0
        self.pcnt[:len(agg_cnt1), 1] += agg_cnt1
        self.pcnt[:len(agg_cnt2), 2] += agg_cnt2
        self.tot_sum += weights.sum()

        self.init_acc.seq_update(init_enc, weights, estimate.init_dist)

    def seq_initialize(self, x: E, weights: np.ndarray, rng: RandomState) -> None:
        sz, idx, xs, ys, ym, init_enc = x

        if not self._init_rng:
            self._rng_initialize(rng)

        agg_cnt0 = np.bincount(xs[ym[0]], weights=weights[idx[ym[0]]])
        agg_cnt1 = np.bincount(xs[ym[1]], weights=weights[idx[ym[1]]])
        agg_cnt2 = np.bincount(xs[ym[2]], weights=weights[idx[ym[2]]])

        self.pcnt[:len(agg_cnt0), 0] += agg_cnt0
        self.pcnt[:len(agg_cnt1), 1] += agg_cnt1
        self.pcnt[:len(agg_cnt2), 2] += agg_cnt2
        self.tot_sum += weights.sum()

        self.init_acc.seq_initialize(init_enc, weights, rng)

    def combine(self, suff_stat: Tuple[np.ndarray, float, Optional[SS1]]) -> 'IntegerStepBernoulliEditAccumulator':
        self.pcnt += suff_stat[0]
        self.tot_sum += suff_stat[1]
        self.init_acc.combine(suff_stat[2])

        return self

    def value(self) -> Tuple[np.ndarray, float, Optional[Any]]:
        return self.pcnt, self.tot_sum, self.init_acc.value()

    def from_value(self, x: Tuple[np.ndarray, float, Optional[SS1]]) -> 'IntegerStepBernoulliEditAccumulator':
        self.pcnt = x[0]
        self.tot_sum = x[1]
        self.init_acc.from_value(x[2])

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                temp = stats_dict[self.key]
                stats_dict[self.key] = (temp[0] + self.pcnt, temp[1] + self.tot_sum)
            else:
                stats_dict[self.key] = (self.pcnt, self.tot_sum)

        self.init_acc.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                self.pcnt, self.tot_sum = stats_dict[self.key]

        self.init_acc.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'IntegerStepBernoulliEditDataEncoder':
        return IntegerStepBernoulliEditDataEncoder(init_encoder=self.init_acc.acc_to_encoder())

class IntegerStepBernoulliEditAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, num_vals: int, init_factory: Optional[StatisticAccumulatorFactory] = NullAccumulatorFactory,
                 keys: Optional[str] = None) -> None:
        self.keys = keys
        self.init_factory = init_factory if init_factory is not None else NullAccumulatorFactory()
        self.num_vals = num_vals

    def make(self) -> 'IntegerStepBernoulliEditAccumulator':
        return IntegerStepBernoulliEditAccumulator(self.num_vals, init_acc=self.init_factory.make(), keys=self.keys)


class IntegerStepBernoulliEditEstimator(ParameterEstimator):

    def __init__(self, num_vals: int, init_estimator: Optional[ParameterEstimator] = NullEstimator(),
                 min_prob: float = 1.0e-128, pseudo_count: Optional[float] = None,
                 suff_stat: Optional[np.ndarray] = None, name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        self.num_vals = num_vals
        self.keys = keys
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.name = name
        self.min_prob = min_prob
        self.init_est = init_estimator if init_estimator is not None else NullEstimator()

    def accumulator_factory(self) -> 'IntegerStepBernoulliEditAccumulatorFactory':
        init_factory = self.init_est.accumulator_factory()
        return IntegerStepBernoulliEditAccumulatorFactory(self.num_vals, init_factory, self.keys)

    def __get_pqk(self, obs_counts: np.ndarray, n: int) -> np.ndarray:
        sidx = np.argsort(-obs_counts)
        obs_counts = obs_counts[sidx]
        N = len(obs_counts)

        max_ll = -np.inf
        max_params = None
        for i in range(N):
            k = i + 1
            p = obs_counts[:k].sum() / (n * k)
            if p == 1:
                v1 = (obs_counts[:k]).sum() * np.log(p)
            else:
                v1 = (n - obs_counts[:k]).sum() * np.log1p(-p) + (obs_counts[:k]).sum() * np.log(p)
            if k < N:
                q = obs_counts[k:].sum() / (n * (N - k))
                if q == 1:
                    v2 = (obs_counts[k:]).sum() * np.log(q)
                else:
                    v2 = (n - obs_counts[k:]).sum() * np.log1p(-q) + (obs_counts[k:]).sum() * np.log(q)
            else:
                q = 0.0
                v2 = 0.0
            ll = v1 + v2
            # print((i, ll, p, q))
            if ll > max_ll:
                max_params = (p, q, k - 1)
                max_ll = ll

        p, q, k = max_params

        arr = np.zeros(len(sidx))
        arr[sidx[:k + 1]] = p
        arr[sidx[k + 1:]] = q
        return arr

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[np.ndarray, float, Optional[SS1]]) \
            -> 'IntegerStepBernoulliEditDistribution':

        init_dist = self.init_est.estimate(None, suff_stat[2])
        count_mat, tot_sum, _ = suff_stat

        if self.pseudo_count is not None and self.suff_stat is not None:

            p = self.pseudo_count
            s = self.suff_stat

            s1 = count_mat[:, 0] + count_mat[:, 2]
            s0 = (tot_sum - s1)

            log_s1 = np.log(s1 + p * (s[:, 1] + s[:, 3]))
            log_s0 = np.log(s0 + p * (s[:, 0] + s[:, 2]))

            log_pmat = np.empty((self.num_vals, 4), dtype=np.float64)

            # print('hello')
            log_pmat[:, 0] = np.log((s0 - count_mat[:, 1]) + p * s[:, 0]) - log_s0
            log_pmat[:, 1] = np.log(count_mat[:, 0] + p * s[:, 1]) - log_s1
            log_pmat[:, 2] = np.log(count_mat[:, 1] + p * s[:, 2]) - log_s0
            log_pmat[:, 3] = np.log(count_mat[:, 2] + p * s[:, 3]) - log_s1

        elif self.pseudo_count is not None and self.suff_stat is None:
            p = self.pseudo_count

            s1 = count_mat[:, 0] + count_mat[:, 2]
            s0 = tot_sum - s1

            log_s1 = np.log(s1 + p / 2.0)
            log_s0 = np.log(s0 + p / 2.0)

            log_pmat = np.empty((self.num_vals, 4), dtype=np.float64)

            log_pmat[:, 2] = np.log(count_mat[:, 1] + (p / 4.0)) - log_s0
            log_pmat[:, 3] = np.log(count_mat[:, 2] + (p / 4.0)) - log_s1
            log_pmat[:, 0] = np.log((s0 - count_mat[:, 1]) + (p / 4.0)) - log_s0
            log_pmat[:, 1] = np.log(count_mat[:, 0] + (p / 4.0)) - log_s1

        else:
            if suff_stat[1] == 0:
                log_pmat = np.zeros((self.num_vals, 4), dtype=np.float64) + np.log(0.5)

            elif (self.min_prob is not None) and (self.min_prob > 0):

                s1 = count_mat[:, 0] + count_mat[:, 2]
                s0 = tot_sum - s1

                log_pmat = np.empty((self.num_vals, 4), dtype=np.float64)
                log_pmat.fill(np.log(self.min_prob))

                if s0 != 0:
                    log_pmat[:, 0] = np.log(np.maximum((s0 - count_mat[:, 1]) / s0, self.min_prob))
                    log_pmat[:, 2] = np.log(np.maximum(count_mat[:, 1] / s0, self.min_prob))

                if s1 != 0:
                    log_pmat[:, 1] = np.log(np.maximum(count_mat[:, 0] / s1, self.min_prob))
                    log_pmat[:, 3] = np.log(np.maximum(count_mat[:, 2] / s1, self.min_prob))

            else:

                s1 = count_mat[:, 0] + count_mat[:, 2]
                s0 = tot_sum - s1

                log_pmat = np.empty((self.num_vals, 4), dtype=np.float64)
                log_pmat[:, 0] = np.log((s0 - count_mat[:, 1]) / s0)
                log_pmat[:, 1] = np.log(count_mat[:, 0] / s1)
                log_pmat[:, 2] = np.log(count_mat[:, 1] / s0)
                log_pmat[:, 3] = np.log(count_mat[:, 2] / s1)

        obs_counts = np.exp(log_pmat[:, 1])
        n = 1
        arr1 = self.__get_pqk(obs_counts, n)

        obs_counts = np.exp(log_pmat[:, 2])
        n = 1
        arr2 = self.__get_pqk(obs_counts, n)

        log_pmat[:, 2] = np.log(arr2)
        log_pmat[:, 0] = np.log(1 - arr2)
        log_pmat[:, 1] = np.log(arr1)
        log_pmat[:, 3] = np.log(1 - arr1)

        return IntegerStepBernoulliEditDistribution(log_pmat, init_dist=init_dist, name=self.name)

class IntegerStepBernoulliEditDataEncoder(DataSequenceEncoder):

    def __init__(self, init_encoder: DataSequenceEncoder) -> None:
        self.init_encoder = init_encoder

    def __str__(self) -> str:
        return 'IntegerBernoulliEditDataEncoder(init_encoder=' + str(self.init_encoder) + ')'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, IntegerStepBernoulliEditDataEncoder):
            return other.init_encoder == self.init_encoder
        else:
            return False

    def seq_encode(self, x: Sequence[T]) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray,
                                                  Tuple[np.ndarray, np.ndarray, np.ndarray], Optional[Any]]:
        idx = []
        xs = []
        ys = []
        pre = []

        for i, xx in enumerate(x):
            pre.append(xx[0])

            xx0 = np.asarray(xx[0], dtype=int)
            xx1 = np.asarray(xx[1], dtype=int)

            to_add = np.isin(xx1, xx0, invert=False)
            to_rem = np.isin(xx0, xx1, invert=True)

            new_x = np.concatenate([xx0[to_rem], xx1[~to_add], xx1[to_add]])
            new_i = np.concatenate([[0] * np.sum(to_rem), [1] * np.sum(~to_add), [2] * np.sum(to_add)])

            idx.extend([i] * len(new_x))
            xs.extend(list(new_x))
            ys.extend(list(new_i))

        idx = np.asarray(idx, dtype=np.int32)
        xs = np.asarray(xs, dtype=np.int32)
        ys = np.asarray(ys, dtype=np.int32)
        ym = (np.flatnonzero(ys == 0), np.flatnonzero(ys == 1), np.flatnonzero(ys == 2))

        init_enc = self.init_encoder.seq_encode(pre)

        return len(x), idx, xs, ys, ym, init_enc
