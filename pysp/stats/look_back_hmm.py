import math

import numba
import numpy as np
from numpy.random import RandomState

import pysp.utils.vector as vec
from pysp.arithmetic import *
from pysp.arithmetic import maxrandint
from pysp.stats.markovchain import MarkovChainDistribution
from pysp.stats.null_dist import NullDistribution, NullAccumulator, NullEstimator, NullDataEncoder, \
    NullAccumulatorFactory
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator, DistributionSampler, DataSequenceEncoder, StatisticAccumulatorFactory

from typing import Optional, Dict, Union, List, Sequence, TypeVar, Tuple

T = TypeVar('T')
E0 = TypeVar('E0')
E1 = TypeVar('E1')


class LookbackHiddenMarkovDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, topics: Sequence[SequenceEncodableProbabilityDistribution], w: np.ndarray,
                 transitions, lag: int = 0,
                 init_dist: Optional[Sequence[SequenceEncodableProbabilityDistribution]] = None,
                 len_dist: Optional[SequenceEncodableProbabilityDistribution] = NullDistribution(),
                 name: Optional[str] = None) -> None:
        with np.errstate(divide='ignore'):
            self.topics = topics
            self.init_dist = init_dist if init_dist is not None else [NullDistribution()]*lag
            self.lag = lag
            self.num_topics = len(topics)
            self.num_states = len(w)
            self.w = vec.make(w)
            self.log_w = log(self.w)
            self.transitions = np.reshape(transitions, (self.num_states, self.num_states))
            self.len_dist = len_dist if len_dist is not None else NullDistribution()
            self.name = name

    def __str__(self) -> str:
        s1 = ','.join(map(str, self.topics))
        s2 = repr(list(self.w))
        s3 = repr([list(u) for u in self.transitions])
        s4 = repr(self.lag)
        s5 = ','.join(map(str, self.init_dist))
        s6 = str(self.len_dist)
        s7 = repr(self.name)

        return 'LookbackHiddenMarkovDistribution([%s], %s, %s, lag=%s, init_dist=[%s], len_dist=%s, name=%s)' % (
        s1, s2, s3, s4, s5, s6, s7)

    def density(self, x):
        return exp(self.log_density(x))

    def log_density(self, x):

        if x is None or len(x) == 0:
            if self.len_dist is not None:
                return self.len_dist(0)
            else:
                return 0.0

        log_w = self.log_w
        num_states = self.num_states
        comps = self.topics
        lag = self.lag
        init_comps = self.init_dist

        obs_log_likelihood = np.zeros(num_states, dtype=np.float64)
        obs_log_likelihood += log_w
        for i in range(num_states):
            obs_log_likelihood[i] += init_comps[i].log_density(x[:lag])

        if np.max(obs_log_likelihood) == -np.inf:
            return -np.inf

        max_ll = obs_log_likelihood.max()
        obs_log_likelihood -= max_ll
        np.exp(obs_log_likelihood, out=obs_log_likelihood)
        sum_ll = np.sum(obs_log_likelihood)
        retval = np.log(sum_ll) + max_ll

        for k in range(lag, len(x)):

            #  P(Z(t+1) | Z(t) = i) P(Z(t) = i | X(t), X(t-1), ...)
            np.dot(self.transitions.T, obs_log_likelihood, out=obs_log_likelihood)
            obs_log_likelihood /= obs_log_likelihood.sum()

            # log P(Z(t+1) | X(t), X(t-1), ...)
            np.log(obs_log_likelihood, out=obs_log_likelihood)

            # log P(X(t+1) | X(t), ..., Z(t+1)=i) + log P(Z(t+1)=i | X(t), X(t-1), ...)
            for i in range(num_states):
                obs_log_likelihood[i] += comps[i].log_density(x[(k - lag):(k + 1)])

            # P(X(t+1) | X(t), X(t-1), ...)  [prevent underflow]
            max_ll = obs_log_likelihood.max()
            obs_log_likelihood -= max_ll
            np.exp(obs_log_likelihood, out=obs_log_likelihood)
            sum_ll = np.sum(obs_log_likelihood)

            # P(X(t+1), X(t), ...)
            retval += np.log(sum_ll) + max_ll

        if self.len_dist is not None:
            retval += self.len_dist.log_density(len(x) - lag + 1)

        return retval

    def viterbi_sequence(self, x):

        obs_cnt = len(x) - self.lag + 1
        log_w = self.log_w
        log_t = np.log(self.transitions)
        num_states = self.num_states
        comps = self.topics
        lag = self.lag
        init_comps = self.init_dist

        rv = np.zeros(obs_cnt, dtype=int)
        max_mat = np.zeros((num_states, obs_cnt), dtype=int)
        obs_mat = np.zeros((num_states, obs_cnt), dtype=float)

        obs_mat[:, 0] += log_w
        for i in range(num_states):
            obs_mat[i, 0] += init_comps[i].log_density(x[:lag])

        for idx, k in enumerate(range(lag, len(x))):
            for i in range(num_states):
                obs_ll = comps[i].log_density(x[(k - lag):(k + 1)])
                temp_ll = obs_mat[:, idx] + log_t[:, i] + obs_ll
                max_idx = np.argmax(temp_ll)
                max_mat[i, idx + 1] = max_idx
                obs_mat[i, idx + 1] = temp_ll[max_idx]

        rv[obs_cnt - 1] = np.argmax(obs_mat[:, obs_cnt - 1])
        for idx in range(obs_cnt - 1, 0, -1):
            rv[idx - 1] = max_mat[rv[idx], idx]

        return rv

    def seq_log_density(self, x):

        num_states = self.num_states

        (ids, idi, ims, imi, sz, enc_sdata, enc_idata), len_enc = x

        w = self.w
        A = self.transitions
        tot_cnt = len(ids) + len(idi)
        num_seq = len(sz)

        pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)
        ll_ret = np.zeros(num_seq, dtype=np.float64)
        tz = np.concatenate([[0], sz]).cumsum().astype(dtype=np.int32)

        # Compute state likelihood vectors and scale the max to one
        for i in range(num_states):
            pr_obs[imi, i] = self.init_dist[i].seq_log_density(enc_idata).astype(np.float64)
            pr_obs[ims, i] = self.topics[i].seq_log_density(enc_sdata).astype(np.float64)

        pr_max0 = pr_obs.max(axis=1)
        pr_obs -= pr_max0[:, None]
        np.exp(pr_obs, out=pr_obs)

        alpha_buff = np.zeros((num_seq, num_states), dtype=np.float64)
        next_alpha = np.zeros((num_seq, num_states), dtype=np.float64)

        numba_seq_log_density(num_states, tz, pr_obs, w, A, pr_max0, next_alpha, alpha_buff, ll_ret)

        ll_ret += self.len_dist.seq_log_density(len_enc)

        return ll_ret

    def seq_posterior(self, x):

        (ids, idi, ims, imi, sz, enc_sdata, enc_idata), len_enc = x

        tot_cnt = len(ids) + len(idi)
        seq_cnt = len(sz)
        num_states = self.num_states
        pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)
        weights = np.ones(seq_cnt, dtype=np.float64)

        max_len = sz.max()
        tz = np.concatenate([[0], sz]).cumsum().astype(dtype=np.int32)

        init_pvec = self.w
        tran_mat = self.transitions

        # Compute state likelihood vectors and scale the max to one
        for i in range(num_states):
            pr_obs[imi, i] = self.init_dist[i].seq_log_density(enc_idata)
            pr_obs[ims, i] = self.topics[i].seq_log_density(enc_sdata)

        pr_max = pr_obs.max(axis=1, keepdims=True)
        pr_obs -= pr_max
        np.exp(pr_obs, out=pr_obs)

        alphas = np.zeros((tot_cnt, num_states), dtype=np.float64)
        xi_acc = np.zeros((seq_cnt, num_states, num_states), dtype=np.float64)
        pi_acc = np.zeros((seq_cnt, num_states), dtype=np.float64)
        numba_baum_welch_alphas(num_states, tz, pr_obs, init_pvec, tran_mat, weights, alphas, xi_acc, pi_acc)

        return [alphas[tz[i]:tz[i + 1], :] for i in range(len(tz) - 1)]

    def sampler(self, seed: Optional[int] = None) -> 'LookbackHiddenMarkovSampler':
        return LookbackHiddenMarkovSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'LookbackHiddenMarkovEstimator':
        len_est = None if self.len_dist is None else self.len_dist.estimator(pseudo_count=pseudo_count)
        comp_ests = [u.estimator(pseudo_count=pseudo_count) for u in self.topics]
        return LookbackHiddenMarkovEstimator(comp_ests, pseudo_count=(pseudo_count, pseudo_count),
                                             len_estimator=len_est)

    def dist_to_encoder(self) -> 'LookbackHiddenMarkovDataEncoder':
        encoder = self.topics[0].dist_to_encoder()
        len_encoder = self.len_dist.dist_to_encoder()
        init_encoder = self.init_dist.dist_to_encoder()

        return LookbackHiddenMarkovDataEncoder(encoder=encoder, len_encoder=len_encoder, init_encoder=init_encoder,
                                               lag=self.lag)



class LookbackHiddenMarkovSampler(DistributionSampler):

    def __init__(self, dist: LookbackHiddenMarkovDistribution, seed: Optional[int] = None) -> None:
        self.num_states = dist.num_states
        self.dist = dist
        self.rng = RandomState(seed)

        self.init_samplers = [dist.init_dist[i].sampler(seed=self.rng.randint(0, maxrandint)) for i in
                              range(dist.num_states)]
        self.obs_samplers = [dist.topics[i].sampler(seed=self.rng.randint(0, maxrandint)) for i in
                             range(dist.num_states)]
        self.len_sampler = dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

        t_map = {i: {k: dist.transitions[i, k] for k in range(dist.num_states)} for i in range(dist.num_states)}
        p_map = {i: dist.w[i] for i in range(dist.num_states)}

        self.state_sampler = MarkovChainDistribution(p_map, t_map).sampler(seed=self.rng.randint(0, maxrandint))

    def sample(self, size: Optional[int] = None):

        if size is None:
            lag = self.dist.lag
            n = self.len_sampler.sample()
            state_seq = self.state_sampler.sample_seq(n)

            rv = list(self.init_samplers[state_seq[0]].sample())  # [v_1, ..., v_lag]
            for i in range(1, n):
                rv.append(self.obs_samplers[state_seq[i]].sample_given(rv[-lag:]))
            return rv
        else:
            return [self.sample() for i in range(size)]


class LookbackHiddenMarkovEstimatorAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, seq_accumulators, init_accumulators=None, lag=0, len_accumulator=None, keys=(None, None, None)):
        self.seq_accumulators = seq_accumulators
        self.init_accumulators = init_accumulators
        self.num_states = len(seq_accumulators)
        self.init_counts = vec.zeros(self.num_states)
        self.trans_counts = vec.zeros((self.num_states, self.num_states))
        self.state_counts = vec.zeros(self.num_states)
        self.len_accumulator = len_accumulator
        self.lag = lag

        self.init_key = keys[0]
        self.trans_key = keys[1]
        self.state_key = keys[2]

    def update(self, x, weight, estimate):
        self.seq_update(estimate.seq_encode([x]), np.asarray([weight]), estimate)

    def initialize(self, x, weight, rng):

        n = len(x) - self.lag + 1
        lag = self.lag

        if self.len_accumulator is not None:
            self.len_accumulator.initialize(n, weight, rng)

        if n > 0:
            w = rng.dirichlet(np.ones(self.num_states) / (self.num_states ** 2), size=n) * weight

            self.init_counts += w[0, :]
            self.state_counts += w.sum(axis=0)

            for j in range(self.num_states):
                self.init_accumulators[j].initialize(x[:lag], w[0, j], rng)

            for k, i in enumerate(range(lag, len(x))):
                self.trans_counts += np.outer(w[k, :], w[k + 1, :])

                for j in range(self.num_states):
                    self.seq_accumulators[j].initialize(x[(i - lag):(i + 1)], w[k + 1, j], rng)

    def seq_update(self, x, weights, estimate):

        (ids, idi, ims, imi, sz, enc_sdata, enc_idata), len_enc = x

        tot_cnt = len(ids) + len(idi)
        seq_cnt = len(sz)
        num_states = estimate.num_states
        pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)

        max_len = sz.max()
        tz = np.concatenate([[0], sz]).cumsum().astype(dtype=np.int32)

        init_pvec = estimate.w
        tran_mat = estimate.transitions

        # Compute state likelihood vectors and scale the max to one
        for i in range(num_states):
            pr_obs[imi, i] = estimate.init_dist[i].seq_log_density(enc_idata)
            pr_obs[ims, i] = estimate.topics[i].seq_log_density(enc_sdata)

        pr_max = pr_obs.max(axis=1, keepdims=True)
        pr_obs -= pr_max
        np.exp(pr_obs, out=pr_obs)

        alphas = np.zeros((tot_cnt, num_states), dtype=np.float64)
        xi_acc = np.zeros((seq_cnt, num_states, num_states), dtype=np.float64)
        pi_acc = np.zeros((seq_cnt, num_states), dtype=np.float64)
        numba_baum_welch2(num_states, tz, pr_obs, init_pvec, tran_mat, weights, alphas, xi_acc, pi_acc)
        self.init_counts += pi_acc.sum(axis=0)
        self.trans_counts += xi_acc.sum(axis=0)

        # numba_baum_welch2.parallel_diagnostics(level=4)

        for i in range(num_states):
            self.init_accumulators[i].seq_update(enc_idata, alphas[imi, i], estimate.init_dist[i])
            self.seq_accumulators[i].seq_update(enc_sdata, alphas[ims, i], estimate.topics[i])

        self.state_counts += alphas.sum(axis=0)

        if self.len_accumulator is not None:
            self.len_accumulator.seq_update(len_enc, weights, estimate.len_dist)

    def combine(self, suff_stat):

        lag, num_states, init_counts, state_counts, trans_counts, seq_accumulators, init_accumulators, len_acc = suff_stat

        self.init_counts += init_counts
        self.state_counts += state_counts
        self.trans_counts += trans_counts

        for i in range(self.num_states):
            self.init_accumulators[i].combine(init_accumulators[i])
            self.seq_accumulators[i].combine(seq_accumulators[i])

        if self.len_accumulator is not None and len_acc is not None:
            self.len_accumulator.combine(len_acc)

        return self

    def value(self):

        if self.len_accumulator is not None:
            len_val = self.len_accumulator.value()
        else:
            len_val = None

        return self.lag, self.num_states, self.init_counts, self.state_counts, self.trans_counts, tuple(
            [u.value() for u in self.seq_accumulators]), tuple([u.value() for u in self.init_accumulators]), len_val

    def from_value(self, x):
        lag, num_states, init_counts, state_counts, trans_counts, seq_accumulators, init_accumulators, len_acc = x

        self.lag = lag
        self.num_states = num_states
        self.init_counts = init_counts
        self.state_counts = state_counts
        self.trans_counts = trans_counts

        for i, v in enumerate(init_accumulators):
            self.init_accumulators[i].from_value(v)

        for i, v in enumerate(seq_accumulators):
            self.seq_accumulators[i].from_value(v)

        if self.len_accumulator is not None:
            self.len_accumulator.from_value(len_acc)

        return self

    def key_merge(self, stats_dict):

        if self.init_key is not None:
            if self.init_key in stats_dict:
                stats_dict[self.init_key] += self.init_counts
            else:
                stats_dict[self.init_key] = self.init_counts

        if self.trans_key is not None:
            if self.trans_key in stats_dict:
                stats_dict[self.trans_key] += self.trans_counts
            else:
                stats_dict[self.trans_key] = self.trans_counts

        if self.state_key is not None:
            if self.state_key in stats_dict:
                acc = stats_dict[self.state_key]
                for i in range(len(acc)):
                    acc[i] = acc[i].combine(self.seq_accumulators[i].value())
            else:
                stats_dict[self.state_key] = self.seq_accumulators

        for u in self.init_accumulators:
            u.key_merge(stats_dict)

        for u in self.seq_accumulators:
            u.key_merge(stats_dict)

        if self.len_accumulator is not None:
            self.len_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict):

        if self.init_key is not None:
            if self.init_key in stats_dict:
                self.init_counts = stats_dict[self.init_key]

        if self.trans_key is not None:
            if self.trans_key in stats_dict:
                self.trans_counts = stats_dict[self.trans_key]

        if self.state_key is not None:
            if self.state_key in stats_dict:
                self.seq_accumulators = stats_dict[self.state_key]

        for u in self.init_accumulators:
            u.key_replace(stats_dict)

        for u in self.seq_accumulators:
            u.key_replace(stats_dict)

        if self.len_accumulator is not None:
            self.len_accumulator.key_replace(stats_dict)


class LookbackHiddenMarkovEstimatorAccumulatorFactory(object):

    def __init__(self, lag: int, seq_factories: Sequence[StatisticAccumulatorFactory],
                 init_factories: Optional[Sequence[StatisticAccumulatorFactory]] = None,
                 len_factory: Optional[StatisticAccumulatorFactory] = NullAccumulatorFactory(),
                 keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (None, None, None)):
        self.seq_factories = seq_factories
        self.keys = keys if keys is not None else (None, None, None)
        self.len_factory = len_factory if len_factory is not None else NullAccumulatorFactory()
        self.lag = lag

        if init_factories is None:
            self.init_factories = [NullAccumulatorFactory() for j in range(len(seq_factories))]
        else:
            self.init_factories = init_factories

    def make(self) -> 'LookbackHiddenMarkovEstimatorAccumulator':
        len_acc = self.len_factory.make() if self.len_factory is not None else None
        seq_acc = [self.seq_factories[i].make() for i in range(len(self.seq_factories))]
        init_acc = [self.init_factories[i].make() for i in range(len(self.init_factories))]
        return LookbackHiddenMarkovEstimatorAccumulator(seq_acc, lag=self.lag, init_accumulators=init_acc,
                                                        len_accumulator=len_acc, keys=self.keys)


class LookbackHiddenMarkovEstimator(ParameterEstimator):

    def __init__(self, estimators: Sequence[ParameterEstimator], lag: int = 0,
                 init_estimators: Optional[Sequence[ParameterEstimator]] = None,
                 len_estimator: Optional[ParameterEstimator] = NullEstimator(),
                 suff_stat=None,
                 pseudo_count: Optional[Tuple[Optional[float], Optional[float]]] = (None, None),
                 name: Optional[str] = None,
                 keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (None, None, None)):
        self.num_states = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count if pseudo_count is not None else (None, None)
        self.suff_stat = suff_stat
        self.keys = keys if keys is not None else (None, None, None)
        self.len_estimator = len_estimator
        self.name = name
        self.lag = lag

        if init_estimators is None:
            self.init_estimators = [NullEstimator() for xx in range(self.num_states)]
        else:
            self.init_estimators = init_estimators

    def accumulator_factory(self):
        est_factories = [u.accumulator_factory() for u in self.estimators]
        iest_factories = [u.accumulator_factory() for u in self.init_estimators]

        len_factory = self.len_estimator.accumulator_factory()
        return LookbackHiddenMarkovEstimatorAccumulatorFactory(self.lag, est_factories, iest_factories, len_factory,
                                                               self.keys)

    def estimate(self, nobs: Optional[float], suff_stat):

        lag, num_states, init_counts, state_counts, trans_counts, topic_ss, init_ss, len_ss = suff_stat

        len_dist = self.len_estimator.estimate(nobs, len_ss)

        topics = [self.estimators[i].estimate(state_counts[i], topic_ss[i]) for i in range(num_states)]
        init_dist = [self.init_estimators[i].estimate(init_counts[i], init_ss[i]) for i in range(num_states)]

        if self.pseudo_count[0] is not None:
            p1 = self.pseudo_count[0] / float(num_states)
            w = init_counts + p1
            w /= w.sum()
        else:
            w = init_counts / init_counts.sum()

        if self.pseudo_count[1] is not None:
            p2 = self.pseudo_count[1] / float(num_states * num_states)
            transitions = trans_counts + p2
            row_sum = transitions.sum(axis=1, keepdims=True)
            transitions /= row_sum
        else:
            row_sum = trans_counts.sum(axis=1, keepdims=True)
            transitions = trans_counts / row_sum

        return LookbackHiddenMarkovDistribution(topics, w, transitions, lag=lag, init_dist=init_dist, len_dist=len_dist,
                                                name=self.name)


class LookbackHiddenMarkovDataEncoder(DataSequenceEncoder):

    def __init__(self, encoder: DataSequenceEncoder, lag: int,
                 len_encoder: Optional[DataSequenceEncoder] = NullDataEncoder(),
                 init_encoder: Optional[DataSequenceEncoder] = NullDataEncoder()) -> None:
        self.encoder = encoder
        self.lag = lag
        self.len_encoder = len_encoder if len_encoder is not None else NullDataEncoder()
        self.init_encoder = init_encoder if init_encoder is not None else NullDataEncoder()

    def __str__(self) -> str:
        s = 'LookbackHiddenMarkovDataEncoder(encoder=' + str(self.encoder) +',lag=' + str(self.lag)
        s += ',len_encoder=' + str(self.len_encoder) + ',init_encoder=' + str(self.init_encoder) + ')'
        return s

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LookbackHiddenMarkovDataEncoder):
            c0 = self.len_encoder == other.len_encoder
            c1 = self.init_encoder == other.init_encoder
            c2 = self.lag == other.lag

            return c0 and c1 and c2

        else:
            return False

    def seq_encode(self, x):

        ids = []
        idi = []
        xss = []
        sz = []
        xsi = []
        imi = []
        ims = []

        lag = self.lag
        cnt = 0
        for i in range(len(x)):
            xxi = x[i][:lag]
            xxs = [x[i][(j - lag):(j + 1)] for j in range(lag, len(x[i]))]
            xsi.append(xxi)
            idi.append(i)
            ids.extend([i] * len(xxs))
            xss.extend(xxs)
            sz.append(len(x[i]) - lag + 1)

            imi.append(cnt)
            ims.extend(range(cnt + 1, cnt + 1 + (len(x[i]) - lag)))
            cnt += len(x[i]) - lag + 1

        len_enc = self.len_encoder.seq_encode(sz)

        ids = np.asarray(ids, dtype=np.int32)
        idi = np.asarray(idi, dtype=np.int32)
        ims = np.asarray(ims, dtype=np.int32)
        imi = np.asarray(imi, dtype=np.int32)
        sz = np.asarray(sz, dtype=np.int32)
        xss = self.encoder.seq_encode(xss)
        xsi = self.init_encoder.seq_encode(xsi)

        return (ids, idi, ims, imi, sz, xss, xsi), len_enc

@numba.njit(
    'void(int32, int32[:], float64[:,:], float64[:], float64[:,:], float64[:], float64[:,:], float64[:,:], float64[:])',
    parallel=True, fastmath=True)
def numba_seq_log_density(num_states, tz, prob_mat, init_pvec, tran_mat, max_ll, next_alpha_mat, alpha_buff_mat, out):
    for n in numba.prange(len(tz) - 1):

        s0 = tz[n]
        s1 = tz[n + 1]

        if s0 == s1:
            out[n] = 0
            continue

        next_alpha = next_alpha_mat[n, :]
        alpha_buff = alpha_buff_mat[n, :]

        llsum = 0
        alpha_sum = 0
        for i in range(num_states):
            temp = init_pvec[i] * prob_mat[s0, i]
            next_alpha[i] = temp
            alpha_sum += temp

        llsum += math.log(alpha_sum)
        llsum += max_ll[s0]

        for s in range(s0 + 1, s1):

            for i in range(num_states):
                alpha_buff[i] = next_alpha[i] / alpha_sum

            alpha_sum = 0
            for i in range(num_states):
                temp = 0.0
                for j in range(num_states):
                    temp += tran_mat[j, i] * alpha_buff[j]
                temp *= prob_mat[s, i]
                next_alpha[i] = temp
                alpha_sum += temp

            llsum += math.log(alpha_sum)
            llsum += max_ll[s]

        out[n] = llsum


@numba.njit(
    'void(int32, int32[:], float64[:,:], float64[:], float64[:,:], float64[:], float64[:,:], float64[:,:], float64[:], float64[:], float64[:,:])')
def numba_baum_welch(num_states, tz, prob_mat, init_pvec, tran_mat, weights, alpha_loc, xi_acc, pi_acc, beta_buff,
                     xi_buff):
    for n in range(len(tz) - 1):

        s0 = tz[n]
        s1 = tz[n + 1]

        if s0 == s1:
            continue

        weight_loc = weights[n]
        alpha_sum = 0
        for i in range(num_states):
            temp = init_pvec[i] * prob_mat[s0, i]
            alpha_loc[s0, i] = temp
            alpha_sum += temp
            # alpha_sum = temp if temp > alpha_sum else alpha_sum
        for i in range(num_states):
            alpha_loc[s0, i] /= alpha_sum

        for s in range(s0 + 1, s1):

            sm1 = s - 1
            alpha_sum = 0
            for i in range(num_states):
                temp = 0.0
                for j in range(num_states):
                    temp += tran_mat[j, i] * alpha_loc[sm1, j]
                temp *= prob_mat[s, i]
                alpha_loc[s, i] = temp
                alpha_sum += temp
                # alpha_sum = temp if temp > alpha_sum else alpha_sum

            for i in range(num_states):
                alpha_loc[s, i] /= alpha_sum

        for i in range(num_states):
            alpha_loc[s1 - 1, i] *= weight_loc

        beta_sum = 1
        # beta_sum = 1/num_states
        prev_beta = np.empty(num_states, dtype=np.float64)
        prev_beta.fill(1 / num_states)

        for s in range(s1 - 2, s0 - 1, -1):

            sp1 = s + 1

            for j in range(num_states):
                beta_buff[j] = prev_beta[j] * prob_mat[sp1, j] / beta_sum

            xi_buff_sum = 0
            gamma_buff = 0
            beta_sum = 0
            for i in range(num_states):

                temp_beta = 0
                for j in range(num_states):
                    temp = tran_mat[i, j] * beta_buff[j]
                    temp_beta += temp
                    temp *= alpha_loc[s, i]
                    xi_buff[i, j] = temp
                    xi_buff_sum += temp

                prev_beta[i] = temp_beta
                alpha_loc[s, i] *= temp_beta
                gamma_buff += alpha_loc[s, i]
                beta_sum += temp_beta
                # beta_sum = temp_beta if temp_beta > beta_sum else beta_sum

            if gamma_buff > 0:
                gamma_buff = weight_loc / gamma_buff

            if xi_buff_sum > 0:
                xi_buff_sum = weight_loc / xi_buff_sum

            for i in range(num_states):
                alpha_loc[s, i] *= gamma_buff
                for j in range(num_states):
                    xi_acc[i, j] += xi_buff[i, j] * xi_buff_sum

        for i in range(num_states):
            pi_acc[i] += alpha_loc[s0, i]


@numba.njit(
    'void(int64, int32[:], float64[:,:], float64[:], float64[:,:], float64[:], float64[:,:], float64[:,:,:], float64[:,:])',
    parallel=True, fastmath=True)
def numba_baum_welch2(num_states, tz, prob_mat, init_pvec, tran_mat, weights, alpha_loc, xi_acc, pi_acc):
    for n in numba.prange(len(tz) - 1):

        s0 = tz[n]
        s1 = tz[n + 1]

        if s0 == s1:
            continue

        beta_buff = np.zeros(num_states, dtype=np.float64)
        xi_buff = np.zeros((num_states, num_states), dtype=np.float64)

        weight_loc = weights[n]
        alpha_sum = 0
        for i in range(num_states):
            temp = init_pvec[i] * prob_mat[s0, i]
            alpha_loc[s0, i] = temp
            alpha_sum += temp
            # alpha_sum = temp if temp > alpha_sum else alpha_sum
        for i in range(num_states):
            alpha_loc[s0, i] /= alpha_sum

        for s in range(s0 + 1, s1):

            sm1 = s - 1
            alpha_sum = 0
            for i in range(num_states):
                temp = 0.0
                for j in range(num_states):
                    temp += tran_mat[j, i] * alpha_loc[sm1, j]
                temp *= prob_mat[s, i]
                alpha_loc[s, i] = temp
                alpha_sum += temp
                # alpha_sum = temp if temp > alpha_sum else alpha_sum

            for i in range(num_states):
                alpha_loc[s, i] /= alpha_sum

        for i in range(num_states):
            alpha_loc[s1 - 1, i] *= weight_loc

        beta_sum = 1
        # beta_sum = 1/num_states
        prev_beta = np.empty(num_states, dtype=np.float64)
        prev_beta.fill(1 / num_states)

        for s in range(s1 - 2, s0 - 1, -1):

            sp1 = s + 1

            for j in range(num_states):
                beta_buff[j] = prev_beta[j] * prob_mat[sp1, j] / beta_sum

            xi_buff_sum = 0
            gamma_buff = 0
            beta_sum = 0
            for i in range(num_states):

                temp_beta = 0
                for j in range(num_states):
                    temp = tran_mat[i, j] * beta_buff[j]
                    temp_beta += temp
                    temp *= alpha_loc[s, i]
                    xi_buff[i, j] = temp
                    xi_buff_sum += temp

                prev_beta[i] = temp_beta
                alpha_loc[s, i] *= temp_beta
                gamma_buff += alpha_loc[s, i]
                beta_sum += temp_beta
                # beta_sum = temp_beta if temp_beta > beta_sum else beta_sum

            if gamma_buff > 0:
                gamma_buff = weight_loc / gamma_buff

            if xi_buff_sum > 0:
                xi_buff_sum = weight_loc / xi_buff_sum

            for i in range(num_states):
                alpha_loc[s, i] *= gamma_buff
                for j in range(num_states):
                    xi_acc[n, i, j] += xi_buff[i, j] * xi_buff_sum

        for i in range(num_states):
            pi_acc[n, i] += alpha_loc[s0, i]


@numba.njit(
    'void(int64, int32[:], float64[:,:], float64[:], float64[:,:], float64[:], float64[:,:], float64[:,:,:], float64[:,:])',
    parallel=True, fastmath=True)
def numba_baum_welch_alphas(num_states, tz, prob_mat, init_pvec, tran_mat, weights, alpha_loc, xi_acc, pi_acc):
    for n in numba.prange(len(tz) - 1):

        s0 = tz[n]
        s1 = tz[n + 1]

        if s0 == s1:
            continue

        beta_buff = np.zeros(num_states, dtype=np.float64)
        xi_buff = np.zeros((num_states, num_states), dtype=np.float64)

        weight_loc = weights[n]
        alpha_sum = 0
        for i in range(num_states):
            temp = init_pvec[i] * prob_mat[s0, i]
            alpha_loc[s0, i] = temp
            alpha_sum += temp
            # alpha_sum = temp if temp > alpha_sum else alpha_sum
        for i in range(num_states):
            alpha_loc[s0, i] /= alpha_sum

        for s in range(s0 + 1, s1):

            sm1 = s - 1
            alpha_sum = 0
            for i in range(num_states):
                temp = 0.0
                for j in range(num_states):
                    temp += tran_mat[j, i] * alpha_loc[sm1, j]
                temp *= prob_mat[s, i]
                alpha_loc[s, i] = temp
                alpha_sum += temp
                # alpha_sum = temp if temp > alpha_sum else alpha_sum

            for i in range(num_states):
                alpha_loc[s, i] /= alpha_sum
