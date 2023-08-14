""""Create, estimate, and sample from a hidden markov model with K emission distributions (i.e. K states).

Defines the HierarchicalMixtureDistribution, HierarchicalMixtureSampler, HierarchicalMixtureEstimatorAccumulatorFactory,
HierarchicalMixtureEstimatorAccumulator, HierarchicalMixtureEstimator, and the HierarchicalMixtureDataEncoder classes
for use with pysparkplug.

Data type: Sequence[T] (determined by emission distributions).

Consider an observation x = (x_1, x_2, ..., x_T) where x_i is of data type T. Assume Z = (Z_1, ..., Z_T) is an
unobserved sequence of hidden states taking on values {1,2,..,K}. A K state hidden markov model can be written as
hierarchical model as follows:

For t = 1,2,..,T, the emission distributions are given by
    (1) P_1(X_t = x_t | Z_t = k), for k = {1,2,...,K}.

The state transitions are given by the K by K matrix formed from
    (2) p_mat(Z_t = i | Z_{t-1} = j), for i, j = {2,3,..,K}.

The initial state distribution is given by weights
    (3) p_mat(Z_1=k) = pi_k, for k = {1,2,...,K}, where sum_k pi_k = 1.0

If included, the length of the hidden markov model sequences is modeled through
    (4) P_len(T), where P_len() is a distribution with support on non-negative integers.

Note that P_1() in (1) must be a distribution compatible with type T data. p_mat() in (2) is a 2-d numpy array of 2-d
list of floats where the rows sum to 1.0. (3) is represented by a numpy array of list of floats that sum to 1.

"""

import numba
import numpy as np
import math
from numpy.random import RandomState
import pysp.utils.vector as vec
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator, DataSequenceEncoder, DistributionSampler, StatisticAccumulatorFactory
from pysp.stats.markovchain import MarkovChainDistribution
from pysp.stats.mixture import MixtureDistribution
from pysp.stats.null_dist import NullDistribution, NullAccumulatorFactory, NullEstimator, NullDataEncoder, \
    NullAccumulator
from pysp.arithmetic import maxrandint

from typing import List, Any, Tuple, Sequence, Union, Optional, TypeVar, Set, Dict

T = TypeVar('T')
T1 = TypeVar('T1')  # Emission suff-stat type
T2 = TypeVar('T2')  # Len suff-stat type
E1 = Tuple[Tuple[int, List[Tuple[int, int]], List[np.ndarray], np.ndarray, np.ndarray, np.ndarray, Any], Any,
           Optional[Any]]
E2 = Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Optional[Any]]


class HiddenMarkovModelDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, topics: Sequence[SequenceEncodableProbabilityDistribution],
                 w: Union[Sequence[float], np.ndarray],
                 transitions: Union[List[List[float]], np.ndarray],
                 taus: Optional[Union[List[List[float]], np.ndarray]] = None,
                 len_dist: Optional[SequenceEncodableProbabilityDistribution] = NullDistribution(),
                 name: Optional[str] = None,
                 terminal_values: Optional[Set[T]] = None,
                 use_numba: bool = False) -> None:
        """HiddenMarkovModelDistribution object defining HMM compatible with data type T.

        Defines an HMM with emission distributions in 'topics' (all must have the same data type T). If a length
        distribution for the length of HMM sequence is included, it must have data type int with support of non-negative
        integers.


        Args:
            topics (Sequence[SequenceEncodableProbabilityDistribution]): Emission distributions all having type T.
            w (Union[Sequence[float], np.ndarray]): Initial state probabilities.
            transitions (Union[List[List[float]], np.ndarray]): 2-d array of hidden state transition probabilities.
            taus (Optional[Union[Sequence[float], np.ndarray]]): Emission distributions are a Mixture over topics.
                Hidden states govern transitions between mixture weights.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]):
            name (Optional[str]): Set name to object instance.
            terminal_values (Optional[Set[T]]): Define terminating emission outputs of the HMM.
            use_numba (bool): If True, use numba package for encoding and vectorized operations.

        Attributes:
            topics (Sequence[SequenceEncodableProbabilityDistribution]): Emission distributions all having type T.
            n_topics (int): Number of emission distributions.
            n_states (int): Number of hidden states.
            w (np.ndarray): Initial state probabilities.
            log_w (np.ndarray): Initial state log-probabilities.
            transitions (np.ndarray): 2-d Numpy array of hidden state transition probabilities. (n_states by n_states).
            log_transitions (np.ndarray): Log of above.
            taus (Optional[np.ndarray]): Emission distributions are a Mixture over topics. Hidden states govern
                transitions between mixture weights.
            log_taus (Optional[np.ndarray]): Log probabilties of taus above.
            has_topics (bool): True if taus is passed.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]):
            name (Optional[str]): Set name to object instance.
            terminal_values (Optional[Set[T]]): Define terminating emission outputs of the HMM.
            use_numba (bool): If True, use numba package for encoding and vectorized operations.

        """
        self.use_numba = use_numba

        with np.errstate(divide='ignore'):

            self.topics = topics
            self.n_topics = len(topics)
            self.n_states = len(w)
            self.w = vec.make(w)
            self.log_w = np.log(self.w)

            if not isinstance(transitions, np.ndarray):
                transitions = np.asarray(transitions, dtype=float)

            self.transitions = np.reshape(transitions, (self.n_states, self.n_states))
            self.log_transitions = np.log(self.transitions)
            self.terminal_values = terminal_values
            self.name = name
            self.len_dist = len_dist if len_dist is not None else NullDistribution()

        if taus is not None:
            self.taus = vec.make(taus)
            self.log_taus = log(self.taus)
            self.has_topics = True
        else:
            self.taus = None
            self.has_topics = False

    def __str__(self) -> str:
        """Returns string representation of HiddenMarkovDistribution instance."""
        s1 = ','.join(map(str, self.topics))
        s2 = repr(list(self.w))
        s3 = repr([list(u) for u in self.transitions])
        if self.taus is None:
            s4 = repr(self.taus)
        else:
            s4 = repr([list(u) for u in self.taus])
        s5 = str(self.len_dist)
        s6 = repr(self.name)
        s7 = repr(self.terminal_values)
        s8 = repr(self.use_numba)

        return 'HiddenMarkovModelDistribution([%s], %s, %s, %s, len_dist=%s, name=%s, terminal_values=%s, ' \
               'use_numba=%s)' % (s1, s2, s3, s4, s5, s6, s7, s8)

    def density(self, x: List[T]) -> float:
        """Returns the density of HMM for an observed sequence x.

        See 'HiddenMarkovDistribution.log_density()' for details.

        Args:
            x (List[T]): Observed sequence of HMM emissions.

        Returns:
            Density of HMM for observed sequence x.

        """
        return exp(self.log_density(x))

    def log_density(self, x: List[T]) -> float:
        """Returns the log-density of HMM for observed sequence x.

        Density for a sequence of length N is given by recursively evaluating the conditional density,

            p_mat(x_mat(0),x_mat(1),....,x_mat(t)) = p_mat(x_mat(t)|x_mat(0),...,x_mat(t-1)) = p_mat(x_mat(t)|Z(t))*p_mat(Z(t)|Z(t-1))*p_mat(Z(t-1)|x_mat(0),....,x_mat(t-1))

        for t = 1,2,...,N-1. p_mat(Z(0)) is given by 'w', p_mat(x_mat(t)|Z(t)) is given by emission distribution 'topics' for
        t = 0,1,...,N-1.

        The returned density is given by

            p_mat(x_mat) = p_mat(x_mat(0),x_mat(1),....,x_mat(t))*P_len(N).

        where P_len(N) is the length distribution 'len_dist', if assigned.
        Note: All calculations are done on the log scale with log-sum-exp used to prevent numerical underflow.

        If 'has_topics' is true, 'weighed_log_sum_exp' and 'log_sum' calls from pysp.utils.vector are used to handle
        the emission distributions being treated as mixture distributions with weights 'log_taus'.

        Args:
            x (List[T]): Observed sequence of HMM emissions.

        Returns:
            Log-density of observed HMM sequence x.

        """
        if x is None or len(x) == 0:
            return self.len_dist.log_density(0)  # this will return 0.0 if NullDistribution()

        if not self.has_topics:
            log_w = self.log_w
            num_states = self.n_states
            comps = self.topics

            obs_log_likelihood = np.zeros(num_states, dtype=np.float64)
            obs_log_likelihood += log_w
            for i in range(num_states):
                obs_log_likelihood[i] += comps[i].log_density(x[0])

            if np.max(obs_log_likelihood) == -np.inf:
                return -np.inf

            max_ll = obs_log_likelihood.max()
            obs_log_likelihood -= max_ll
            np.exp(obs_log_likelihood, out=obs_log_likelihood)
            sum_ll = np.sum(obs_log_likelihood)
            retval = np.log(sum_ll) + max_ll

            for k in range(1, len(x)):
                #  p_mat(Z(t) | Z(t-1) = i) p_mat(Z(t-1) = i | x_mat(0), ..., x_mat(t-1))
                np.dot(self.transitions.T, obs_log_likelihood, out=obs_log_likelihood)
                obs_log_likelihood /= obs_log_likelihood.sum()

                # log p_mat(Z(t-1) | x_mat(0), ..., x_mat(t-1))
                np.log(obs_log_likelihood, out=obs_log_likelihood)

                # log p_mat(x_mat(t) | Z(t)=i) + log p_mat(Z(t-1)=i | x_mat(0), ..., x_mat(t-1))
                for i in range(num_states):
                    obs_log_likelihood[i] += comps[i].log_density(x[k])

                # p_mat(x_mat(t) | x_mat(0), ..., x_mat(t-1))  [prevent underflow]
                max_ll = obs_log_likelihood.max()
                obs_log_likelihood -= max_ll
                np.exp(obs_log_likelihood, out=obs_log_likelihood)
                sum_ll = np.sum(obs_log_likelihood)

                # p_mat(x_mat(0), ..., x_mat(t-1), x_mat(t))
                retval += np.log(sum_ll) + max_ll

            retval += self.len_dist.log_density(len(x))

            return retval

        else:
            x_iter = iter(x)
            log_w = self.log_w
            log_taus = self.log_taus
            n_states = self.n_states
            x0 = next(x_iter)

            obs_log_density_by_topic = np.asarray([u.log_density(x0) for u in self.topics])
            log_likelihood_by_state = np.asarray([log_w[i] + vec.weighted_log_sum(
                obs_log_density_by_topic, log_taus[i, :]) for i in range(n_states)])

            for x in x_iter:
                obs_log_density_by_topic = np.asarray([u.log_density(x) for u in self.topics])
                log_likelihood_by_state = [
                    vec.weighted_log_sum(obs_log_density_by_topic, log_taus[:, i]) + vec.weighted_log_sum(
                        obs_log_density_by_topic, log_taus[i, :]) for i in range(n_states)]

            rv = vec.log_sum(log_likelihood_by_state)
            rv += self.len_dist.log_density(len(x))

            return rv

    def seq_log_density(self, x: Union[E1, E2]) -> 'np.ndarray':

        x0, x1 = x
        if x1 is None:

            num_states = self.n_states
            (tot_cnt, idx_bands, has_next, len_vec, idx_mat, idx_vec, enc_data), _, len_enc = x0
            w = self.w
            a_mat = self.transitions

            max_len = len(idx_bands)
            num_seq = idx_mat.shape[0]

            good = idx_mat >= 0

            pr_obs = np.zeros((tot_cnt, num_states))
            ll_ret = np.zeros(num_seq)

            # Compute state likelihood vectors and scale the max to one
            for i in range(num_states):
                pr_obs[:, i] = self.topics[i].seq_log_density(enc_data)

            pr_max0 = pr_obs.max(axis=1, keepdims=True)
            pr_obs -= pr_max0
            np.exp(pr_obs, out=pr_obs)

            # Vectorized alpha pass
            band = idx_bands[0]
            alphas_prev = np.multiply(pr_obs[band[0]:band[1], :], w)
            temp = alphas_prev.sum(axis=1, keepdims=True)
            # temp2 = temp.copy()
            # temp2[temp2 == 0] = 1.0
            alphas_prev /= temp

            np.log(temp, out=temp)
            temp2 = pr_max0[band[0]:band[1], 0]
            ll_ret[good[:, 0]] += temp[:, 0] + temp2

            for i in range(1, max_len):
                band = idx_bands[i]
                has_next_loc = has_next[i - 1]

                alphas_next = np.dot(alphas_prev[has_next_loc, :], a_mat)
                alphas_next *= pr_obs[band[0]:band[1], :]
                pr_max = alphas_next.sum(axis=1, keepdims=True)
                # pr_max2 = pr_max.copy()
                # pr_max2[pr_max2 == 0] = 1.0
                alphas_next /= pr_max
                alphas_prev = alphas_next

                np.log(pr_max, out=pr_max)
                temp2 = pr_max0[band[0]:band[1], 0]
                ll_ret[good[:, i]] += pr_max[:, 0] + temp2

            # nz = len_vec != 0
            # ll_ret[nz] /= len_vec[nz]

            ll_ret[np.isnan(ll_ret)] = -np.inf

            if self.len_dist is not None:
                ll_ret += self.len_dist.seq_log_density(len_enc)

            return ll_ret

        else:

            num_states = self.n_states
            (idx, sz, enc_data), len_enc = x1

            w = self.w
            a_mat = self.transitions
            tot_cnt = len(idx)
            num_seq = len(sz)

            pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)
            ll_ret = np.zeros(num_seq, dtype=np.float64)
            tz = np.concatenate([[0], sz]).cumsum().astype(dtype=np.int32)

            # Compute state likelihood vectors and scale the max to one
            for i in range(num_states):
                pr_obs[:, i] = self.topics[i].seq_log_density(enc_data)

            pr_max0 = pr_obs.max(axis=1)
            pr_obs -= pr_max0[:, None]
            np.exp(pr_obs, out=pr_obs)

            alpha_buff = np.zeros((num_seq, num_states), dtype=np.float64)
            next_alpha = np.zeros((num_seq, num_states), dtype=np.float64)

            numba_seq_log_density(num_states, tz, pr_obs, w, a_mat, pr_max0, next_alpha, alpha_buff, ll_ret)

            if self.len_dist is not None:
                ll_ret += self.len_dist.seq_log_density(len_enc)

            return ll_ret

    def seq_posterior(self, x: E2) -> Optional[List[np.ndarray]]:

        if not self.use_numba:
            return None

        x0, x1 = x

        (idx, sz, enc_data), len_enc = x1

        tot_cnt = len(idx)
        seq_cnt = len(sz)
        num_states = self.n_states
        pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)
        weights = np.ones(seq_cnt, dtype=np.float64)
        max_len = sz.max()
        tz = np.concatenate([[0], sz]).cumsum().astype(dtype=np.int32)

        init_pvec = self.w
        tran_mat = self.transitions

        # Compute state likelihood vectors and scale the max to one
        for i in range(num_states):
            pr_obs[:, i] = self.topics[i].seq_log_density(enc_data)

        pr_max = pr_obs.max(axis=1, keepdims=True)
        pr_obs -= pr_max
        np.exp(pr_obs, out=pr_obs)

        alphas = np.zeros((tot_cnt, num_states), dtype=np.float64)
        xi_acc = np.zeros((seq_cnt, num_states, num_states), dtype=np.float64)
        pi_acc = np.zeros((seq_cnt, num_states), dtype=np.float64)
        numba_baum_welch_alphas(num_states, tz, pr_obs, init_pvec, tran_mat, weights, alphas, xi_acc, pi_acc)

        return [alphas[tz[i]:tz[i + 1], :] for i in range(len(tz) - 1)]

    def viterbi(self, x: List[T]) -> np.ndarray:
        nn = len(x)
        num_states = self.n_states

        v = np.zeros((nn, num_states), dtype=np.float64)
        ptr = np.zeros(nn, dtype=np.int32)
        pr_obs = np.zeros((nn, num_states), dtype=np.float64)
        enc_x = self.topics[0].dist_to_encoder().seq_encode(x)

        for i in range(num_states):
            pr_obs[:, i] = self.topics[i].seq_log_density(enc_x)

        v[0, :] += pr_obs[0, :] + self.log_w

        for t in range(1, nn):
            temp = np.zeros((num_states, num_states), dtype=np.float64)
            temp += np.reshape(v[t-1, :], (num_states,1))
            temp += self.log_transitions
            temp += np.reshape(pr_obs[t, :], (1, num_states))
            v[t, :] += temp.max(axis=0, keepdims=False)

        for t in range(nn-1, -1, -1):
            ptr[t] = np.argmax(v[t, :])

        return ptr

    def seq_viterbi(self, x: E2):

        x0, x1 = x
        if x1 is None:

            num_states = self.n_states
            (tot_cnt, idx_bands, has_next, len_vec, idx_mat, idx_vec, enc_data), _, len_enc = x0
            log_w = self.log_w
            log_a_mat = self.log_transitions

            max_len = len(idx_bands)
            num_seq = idx_mat.shape[0]

            good = idx_mat >= 0

            pr_obs = np.zeros((tot_cnt, num_states))
            v = np.zeros((tot_cnt, num_states), dtype=np.float64)
            ptr = np.zeros(tot_cnt, dtype=np.int32)

            # Compute state likelihood vectors and scale the max to one
            for i in range(num_states):
                pr_obs[:, i] = self.topics[i].seq_log_density(enc_data)

            # Vectorized alpha pass
            prev_band_idx = np.arange(idx_bands[0][0], idx_bands[0][1])
            v[prev_band_idx, :] += pr_obs[prev_band_idx, :] + log_w

            for i in range(1, max_len):
                nxt_band_idx = np.arange(idx_bands[i][0], idx_bands[i][1])
                has_next_loc = has_next[i - 1]

                temp = np.zeros((len(has_next_loc), num_states, num_states), dtype=np.float64)
                temp += np.reshape(v[prev_band_idx[has_next_loc], :], (-1, num_states, 1)) + log_a_mat
                temp += np.reshape(pr_obs[nxt_band_idx, :], (-1, 1, num_states))

                v[nxt_band_idx, :] += np.max(temp, axis=1)

                prev_band_idx = nxt_band_idx.copy()

            for i in range(max_len-1, -1, -1):
                prev_band_idx = np.arange(idx_bands[i][0], idx_bands[i][1])
                ptr[prev_band_idx] += np.argmax(v[prev_band_idx, :], axis=1)

            return ptr

    def sampler(self, seed: Optional[int] = None) -> 'HiddenMarkovSampler':
        """Create a HiddenMarkovSampler object with seed passed.

        Note: Throws exception if 'len_dist'and 'terminal_values' are not set.

        If len_dist is set, it should be a SequenceEncodableProbabilityDistribution with data type int and support on
        non-negative integers.

        Args:
            seed (Optional[int]): Set seed for random sampling.

        Returns:
            HiddenMarkovSampler object.

        """
        if isinstance(self.len_dist, NullDistribution) and self.terminal_values is None:
            raise Exception('HiddenMarkovSampler requires len_dist with support on non-negative integers, or terminal_'
                            'values to be set.')

        return HiddenMarkovSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'HiddenMarkovEstimator':
        """Create HiddenMarkovEstimator for estimating HiddenMarkovDistribution objects from aggregated sufficient
            statistics.

        Args:
            pseudo_count (Optional[float]): Used to re-weight sufficient statistics of HiddenMarkovDistribution object
                instance.

        Returns:
            HiddenMarkovEstimator object.

        """
        len_est = None if self.len_dist is None else self.len_dist.estimator(pseudo_count=pseudo_count)
        comp_ests = [u.estimator(pseudo_count=pseudo_count) for u in self.topics]
        return HiddenMarkovEstimator(comp_ests, pseudo_count=(pseudo_count, pseudo_count), len_estimator=len_est,
                                     name=self.name)

    def dist_to_encoder(self) -> 'HiddenMarkovDataEncoder':
        """Returns HiddenMarkovDataEncoder object for encoding sequences of iid HMM observations."""
        emission_encoder = self.topics[0].dist_to_encoder()
        len_encoder = self.len_dist.dist_to_encoder()

        return HiddenMarkovDataEncoder(emission_encoder=emission_encoder, len_encoder=len_encoder,
                                       use_numba=self.use_numba)

class HiddenMarkovSampler(DistributionSampler):

    def __init__(self, dist: 'HiddenMarkovModelDistribution', seed: Optional[int] = None) -> None:
        """HiddenMarkovSampler object for sampling from HMM.

        If 'dist.len_dist' is set, samples HMM sequences with sequence lengths generated from 'len_dist'. If
        'dist.len_dist' is NullDistribution, 'dist.terminal_values' is must be set. Samples are generated until
        a terminal value is reached.

        Args:
            dist (HiddenMarkovModelDistribution): HiddenMarkovModelDistribution object instance to sample from.
            seed (Optional[int]): Set seed on random number generator for sampling.

        Attributes:
            num_states (int): Number of hidden states in 'dist' object.
            dist (HiddenMarkovModelDistribution): HiddenMarkovModelDistribution object instance to sample from.
            rng (RandomState): RandomState object with seed set for sampling.
            obs_samplers (List[DistributionSampler]): List of DistributionSampler objects corresponding to the emission
                distributions of 'dist'. Taken to be MixtureSampler objects if 'dist.has_topics' is True.
            len_sampler (Optional[DistributionSampler]): DistributionSampler object with data type int and support on
                non-negative integers for sampling HMM observation sequence lengths.
            terminal_set (Optional[Set[T]]): Set of values to terminate HMM sampling when calling 'sample_seq()'.
            state_sampler (MarkovChainSampler): MarkovChainSampler for sampling states of HMM.

        """
        self.num_states = dist.n_states
        self.dist = dist
        self.rng = RandomState(seed)

        if dist.has_topics:
            self.obs_samplers = [
                MixtureDistribution(dist.topics, dist.taus[i, :]).sampler(seed=self.rng.randint(0, maxrandint)) for i in
                range(dist.n_states)]
        else:
            self.obs_samplers = [dist.topics[i].sampler(seed=self.rng.randint(0, maxrandint)) for i in
                                 range(dist.n_states)]

        if dist.len_dist is not None:
            self.len_sampler = dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))
        else:
            self.len_sampler = None

        if dist.terminal_values is None:
            self.terminal_set = None
        else:
            self.terminal_set = set(dist.terminal_values)

        t_map = {i: {k: dist.transitions[i, k] for k in range(dist.n_states)} for i in range(dist.n_states)}
        p_map = {i: dist.w[i] for i in range(dist.n_states)}

        self.state_sampler = MarkovChainDistribution(p_map, t_map).sampler(seed=self.rng.randint(0, maxrandint))

    def sample_seq(self, size: Optional[int] = None) -> Union[List[Any], List[List[Any]]]:
        """Sample iid HMM sequences.

        If size is None, 1 sample is drawn and a List[T] is returned. If size > 0, 'size' samples are drawn and a List
        of length 'size' with HMM sequences (List[T]) is returned.

        Args:
            size (Optional[int]): Number of iid HMM sequences to sample.

        Returns:
            List[T] or List[List[T]] depending on size arg.

        """
        if size is None:
            n = self.len_sampler.sample()
            state_seq = self.state_sampler.sample_seq(n)
            obs_seq = [self.obs_samplers[state_seq[i]].sample() for i in range(n)]

            return obs_seq

        else:
            n = self.len_sampler.sample(size=size)
            state_seq = [self.state_sampler.sample_seq(size=nn) for nn in n]
            obs_seq = [[self.obs_samplers[j].sample() for j in nn] for nn in state_seq]

            return obs_seq

    def sample_terminal(self, terminal_set: Set[T]) -> List[T]:
        """Sample an HMM sequence, until a terminal value is samples from the emission distribution.

        Args:
            terminal_set (Set[T]): Set values to terminate the HMM sequence.

        Returns:
            List[T] with length determined by samples to reach the first terminating value.

        """
        z = self.state_sampler.sample_seq()
        rv = [self.obs_samplers[z].sample()]

        while rv[-1] not in terminal_set:
            z = self.state_sampler.sample_seq(v0=z)
            rv.append(self.obs_samplers[z].sample())

        return rv

    def sample(self, size: Optional[int] = None):
        """Draw iid samples from HMM.

        If a 'len_sampler' is set, call 'sample_seq()' (See HiddenMarkovSampler.sample_seq() for details).
        If 'len_sampler' is the NullDistributionSampler(), 'sample_terminal()' is called. (See
        HiddenMarkovSampler.sample_terminal() for details).

        Args:
            size (Optional[int]): Number of iid HMM sequences to sample.

        Returns:
            List[T] or List[List[T]] depending on arg size.

        """
        if self.len_sampler is not None:
            return self.sample_seq(size=size)

        elif self.terminal_set is not None:
            if size is None:
                return self.sample_terminal(self.terminal_set)
            else:
                return [self.sample_terminal(self.terminal_set) for i in range(size)]

        else:
            raise RuntimeError('HiddenMarkovSampler requires either a length distribution or terminal value set.')


class HiddenMarkovAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, accumulators: Sequence[SequenceEncodableStatisticAccumulator],
                 len_accumulator: Optional[SequenceEncodableStatisticAccumulator] = NullAccumulator(),
                 use_numba: Optional[bool] = False,
                 keys: Tuple[Optional[str], Optional[str], Optional[str]] = (None, None, None),
                 name: Optional[str] = None) -> None:
        """HiddenMarkovAccumulator object for aggregating sufficient statistics from HMM observations.

        Args:
            accumulators (Sequence[SequenceEncodableStatisticAccumulator]): SequenceEncodableStatisticAccumulator
                objects for the emission distributions.
            len_accumulator (Optional[SequenceEncodableStatisticAccumulator]): SequenceEncodableStatisticAccumulator
                object for the length distribution.
            use_numba (bool): True if sequence encodings are for use with numba.
            keys (Tuple[Optional[str], Optional[str], Optional[str]]): Set keys for initial states, transition counts,
                and emission accumulators.
            name (Optional[str]): Name for object.

        Attributes:
            accumulators (Sequence[SequenceEncodableStatisticAccumulator]): SequenceEncodableStatisticAccumulator
                objects for the emission distributions.
            num_states (int): Total number of hidden states.
            init_counts (ndarray): Track gamma_i(0), or first time point gamma for each component in Baum-Welch.
            trans_counts (ndarray): 2-d matrix tracking transition updates from Baum-Welch
                (sum_t psi_ij(t) / sum_t gamma_i(t)).
            state_counts (ndarray): Expected number of times state is observed in sequence from t=0 to t=T-2.
            len_accumulator (SequenceEncodableStatisticAccumulator): SequenceEncodableStatisticAccumulator
                object for the length distribution. Set to NullAccumulator is None is passed.
            use_numba (bool): True if sequence encodings are for use with numba.
            init_key (Optional[str]): Key for initial states.
            trans_key (Optional[str]): Key for state transitions.
            state_key (Optional[str]): Key for emission accumulators..
            name (Optional[str]): Name for object.

            _init_rng (bool): True if RandomState objects have been initialized
            _len_rng (Optional[RandomState]): RandomState for initializing length accumulator.
            _acc_rng (Optional[List[RandomState]): List of RandomState objects for initializing emission accumulators.
            _idx_rng (Optional[RandomState]): RandomState for initializing initial state draws.

        """
        self.accumulators = accumulators
        self.num_states = len(accumulators)
        self.init_counts = vec.zeros(self.num_states)
        self.trans_counts = vec.zeros((self.num_states, self.num_states))
        self.state_counts = vec.zeros(self.num_states)
        self.len_accumulator = len_accumulator if len_accumulator is not None else NullAccumulator()

        self.init_key = keys[0]
        self.trans_key = keys[1]
        self.state_key = keys[2]

        self.use_numba = use_numba
        self.name = name

        # protected for initialization.
        self._init_rng: bool = False
        self._len_rng: Optional[RandomState] = None
        self._acc_rng: Optional[List[RandomState]] = None
        self._idx_rng: Optional[RandomState] = None

    def update(self, x: List[T], weight: float, estimate: HiddenMarkovModelDistribution) -> None:
        """Update sufficient statistics of HiddenMarkovAccumulator with one observation.

        Note: Note efficient. Should use seq_encode() for fully encoded sequence instead.

        Args:
            x (List[T]): HMM observation sequence.
            weight (float): Weight for observation.
            estimate (HiddenMarkovModelDistribution): Previous estimate of HMM.

        Returns:
            None.

        """
        enc_x = estimate.dist_to_encoder().seq_encode([x])
        self.seq_update(enc_x, np.asarray([weight]), estimate)

    def _rng_initialize(self, rng: RandomState) -> None:
        """Set RandomState member variables for initialize and seq_initialize consistency.

        Args:
            rng (RandomState): RandomState object used to set member RandomState objects.

        Returns:
            None.

        """
        rng_seeds = rng.randint(maxrandint, size=2+self.num_states)
        self._idx_rng = RandomState(seed=rng_seeds[0])
        self._len_rng = RandomState(seed=rng_seeds[1])
        self._acc_rng = [RandomState(seed=rng_seeds[2+i]) for i in range(self.num_states)]
        self._init_rng = True

    def initialize(self, x: List[T], weight: float, rng: RandomState) -> None:
        """Initialize HiddenMarkovAccumulator object with HMM sequence x.

        Args:
            x (List[T]): HMM observation sequence.
            weight (float): Weight for observation.
            rng (RandomState): Sets RandomState member values if not already set.

        Returns:
            None.

        """
        if not self._init_rng:
            self._rng_initialize(rng)

        n = len(x)

        self.len_accumulator.initialize(n, weight, self._len_rng)

        if n > 0:

            idx = self._idx_rng.choice(self.num_states, size=n)

            self.init_counts[idx[0]] += weight
            self.state_counts[idx[0]] += weight

            for i in range(n):
                for j in range(self.num_states):
                    w = weight if j == idx[i] else 0.0
                    self.accumulators[j].initialize(x[i], w, self._acc_rng[j])

            if n > 1:
                for i in range(1, n):
                    self.trans_counts[idx[i-1], idx[i]] += weight
                    self.state_counts[idx[i]] += weight

    def seq_initialize(self, x, weights: np.ndarray, rng: np.random.RandomState) -> None:
        """Vectorized initialization of HiddenMarkovAccumulator.

        Note: Initialization method depends on sequence encoding for Numba or baseline numpy. Both methods do
        not call numba for initialization.

        If _init_rng is False, protected RandomState members are set from rng for the accumulators. This ensures
        initialize() method produces a consistent initialization for the same datasets.

        The input 'x' is a sequence encoded HMM sequence of iid observations produced by
        'HiddenMarkovDataEncoder.seq_encode()'. Arg x is either Tuple[None, enc] or Tuple[None, enc_numba].

        For the first case, enc is Tuple[Tuple[....], T_topic, T_len], where the first tuple is given by a Tuple of
            enc[0][0] (int): Total number of observed emissions from all HMM sequences.
            enc[0][1] (List[Tuple[int, int]]): Contains bands for t^th observation in HMM sequences stored in 'seq_x'.
            enc[0][2] (List[ndarray[int]]): List of numpy array on sequence indices that have a next observed emission.
            enc[0][3] (np.ndarray[int]): Numpy array of sequence lengths.
            enc[0][4] (np.ndarray[int]): 2-d matrix with rv[0][0] rows, and column length equal to the length of the
                largest HMM sequence. This is used to store the index of seq_x corresponding to emission x[i][t]. A -1
                is stored if the sequence length has already been met.
            enc[0][5] (ndarray): Numpy array containing lists index 'i' corresponding to x[i][t] block of 'seq_x'.
            enc[0][6] (T_topic): Sequence encoded value of 'seq_x'.
        The next two entries of the Tuple are,
            enc[1] (T_topic): Sequence encoded observation values in order. Just for seq_init consistency.
            enc[1] (Optional[T_len]): Sequence encoded value of lengths of HMM distribution. None if len_encoder is
                the NullDataEncoder.

        The first entry of enc_numba is a Tuple of length-3,
            enc_numba[0][0] (ndarray[int]): Sequence id's for observed values.
            enc_numba[0][1] (ndarray[int]): Sequence lengths for each observed HMM sequence.
            enc_numba[0][2] (T_topic): Sequence encoded observation values.
        The second entry is,
            enc_numba[1] (Optional[T_len]): Sequence encoded values of sequence lengths. None if len_encoder is
                NullDataEncoder.

        Args:
            x: See above for details.
            weights (np.ndarray): Numpy array of weights for observations.
            rng (RandomState): Used to set seed on random initialization.

        Returns:
            None.

        """
        x0, x1 = x

        if x1 is None:
            (tot_cnt, idx_bands, has_next, len_vec, idx_mat, idx_vec, enc_data), xs_enc, len_enc = x0

            if not self._init_rng:
                self._rng_initialize(rng)

            self.len_accumulator.seq_initialize(len_enc, weights, self._len_rng)

            non_zero_len = len_vec != 0
            weights_nz = weights[non_zero_len]

            idx = self._idx_rng.choice(self.num_states, size=tot_cnt)

            seq_i = []
            for i in range(len(len_vec[non_zero_len])):
                seq_i.extend([i]*len_vec[non_zero_len][i])

            seq_i = np.asarray(seq_i, dtype=int)

            x_idx_i, x_group_i, x_len_i = np.unique(seq_i, return_index=True, return_counts=True)

            self.init_counts += np.bincount(idx[x_group_i], weights_nz[x_idx_i], minlength=self.num_states)
            self.state_counts += np.bincount(idx, weights_nz[seq_i], minlength=self.num_states)

            sz_next = len_vec[non_zero_len].copy() - 1
            steps = np.zeros(len(sz_next), dtype=int)
            cond = steps < sz_next

            while np.any(cond):
                prev_state = idx[x_group_i[cond]+steps[cond]]
                next_state = idx[x_group_i[cond]+steps[cond]+1]
                temp = np.bincount(prev_state * self.num_states + next_state, weights_nz[cond],
                                   minlength=self.num_states ** 2)
                self.trans_counts += np.reshape(temp, (self.num_states, self.num_states))

                steps[cond] += 1
                cond = steps < sz_next

            for j in range(self.num_states):
                w = weights[idx_vec]
                w[idx != j] = 0.0
                self.accumulators[j].seq_initialize(xs_enc, w.flatten(), self._acc_rng[j])

        else:
            (idx, sz, xs), len_enc = x1

            if not self._init_rng:
                self._rng_initialize(rng)

            self.len_accumulator.seq_initialize(len_enc, weights, self._len_rng)

            tot_cnt = np.sum(sz)
            states = self._idx_rng.choice(self.num_states, size=tot_cnt)
            nz_idx, nz_idx_group, nz_idx_rep = np.unique(idx, return_index=True, return_inverse=True)
            weights_nz = weights[nz_idx]

            for j in range(self.num_states):
                w = weights_nz[idx].copy()
                w[states != j] = 0
                self.accumulators[j].seq_initialize(xs, w, self._acc_rng[j])

            sz_next = sz.copy()[nz_idx] - 1
            steps = np.zeros(len(sz_next), dtype=int)
            cond = steps < sz_next

            while np.any(cond):
                prev_state = states[nz_idx_group[cond] + steps[cond]]
                next_state = states[nz_idx_group[cond]+steps[cond]+1]
                temp = np.bincount(prev_state * self.num_states + next_state, weights_nz[cond],
                                   minlength=self.num_states ** 2)
                self.trans_counts += np.reshape(temp, (self.num_states, self.num_states))

                steps[cond] += 1
                cond = steps < sz_next

            self.state_counts += np.bincount(states, weights[idx], minlength=self.num_states)
            self.init_counts += np.bincount(states[nz_idx_group], weights[nz_idx], minlength=self.num_states)

    def seq_update(self, x, weights: np.ndarray, estimate: HiddenMarkovModelDistribution) -> None:
        """Vectorized update for HiddenMarkovAccumulator object from encoded sequence of observations.

        This is a vectorized implementation of the Baum-Welch algorithm. If use_numba, Numba functions are called
        for the alpha and beta pass. Else, a vectorized Numpy implementation of Baum-Welch is used.

        The input 'x' is a sequence encoded HMM sequence of iid observations produced by
        'HiddenMarkovDataEncoder.seq_encode()'. Arg x is either Tuple[None, enc] or Tuple[None, enc_numba].

        For the first case, enc is Tuple[Tuple[....], T_topic, T_len], where the first tuple is given by a Tuple of
            enc[0][0] (int): Total number of observed emissions from all HMM sequences.
            enc[0][1] (List[Tuple[int, int]]): Contains bands for t^th observation in HMM sequences stored in 'seq_x'.
            enc[0][2] (List[ndarray[int]]): List of numpy array on sequence indices that have a next observed emission.
            enc[0][3] (np.ndarray[int]): Numpy array of sequence lengths.
            enc[0][4] (np.ndarray[int]): 2-d matrix with rv[0][0] rows, and column length equal to the length of the
                largest HMM sequence. This is used to store the index of seq_x corresponding to emission x[i][t]. A -1
                is stored if the sequence length has already been met.
            enc[0][5] (ndarray): Numpy array containing lists index 'i' corresponding to x[i][t] block of 'seq_x'.
            enc[0][6] (T_topic): Sequence encoded value of 'seq_x'.
        The next two entries of the Tuple is,
            enc[1] (T_topic): Sequence encoded observation values in order. Just for seq_init consistency.
            enc[2] (Optional[T_len]): Sequence encoded value of lengths of HMM distribution. None if len_encoder is
                the NullDataEncoder.

        The first entry of enc_numba is a Tuple of length-3,
            enc_numba[0][0] (ndarray[int]): Sequence id's for observed values.
            enc_numba[0][1] (ndarray[int]): Sequence lengths for each observed HMM sequence.
            enc_numba[0][2] (T_topic): Sequence encoded observation values.
        The second entry is,
            enc_numba[1] (Optional[T_len]): Sequence encoded values of sequence lengths. None if len_encoder is
                NullDataEncoder.

        Args:
            x: See above for details.
            weights (np.ndarray): Numpy array of weights for observation.
            estimate (HiddenMarkovModelDistribution): Previous EM estimate of HMM model.

        Returns:
            None.

        """
        x0, x1 = x

        if x1 is None:

            num_states = self.num_states
            (tot_cnt, idx_bands, has_next, len_vec, idx_mat, idx_vec, enc_data), _, len_enc = x0
            w = estimate.w
            a_mat = estimate.transitions

            max_len = len(idx_bands)
            num_seq = idx_mat.shape[0]

            good = idx_mat >= 0

            pr_obs = np.zeros((tot_cnt, num_states))
            alphas = np.zeros((tot_cnt, num_states))

            # Compute state likelihood vectors and scale the max to one
            for i in range(num_states):
                pr_obs[:, i] = estimate.topics[i].seq_log_density(enc_data)

            pr_max = pr_obs.max(axis=1, keepdims=True)
            pr_obs -= pr_max
            np.exp(pr_obs, out=pr_obs)

            # Vectorized alpha pass
            band = idx_bands[0]
            alphas_prev = alphas[band[0]:band[1], :]
            np.multiply(pr_obs[band[0]:band[1], :], w, out=alphas_prev)
            pr_sum = alphas_prev.sum(axis=1, keepdims=True)
            pr_sum[pr_sum == 0] = 1.0
            alphas_prev /= pr_sum

            for i in range(1, max_len):
                band = idx_bands[i]
                has_next_loc = has_next[i - 1]
                alphas_next = alphas[band[0]:band[1], :]
                np.dot(alphas_prev[has_next_loc, :], a_mat, out=alphas_next)
                alphas_next *= pr_obs[band[0]:band[1], :]
                pr_max = alphas_next.sum(axis=1, keepdims=True)

                pr_max[pr_max == 0] = 1.0

                alphas_next /= pr_max
                alphas_prev = alphas_next

            band2 = idx_bands[-1]
            prev_beta = np.ones((band2[1] - band2[0], num_states))
            alphas[band2[0]:band2[1], :] /= alphas[band2[0]:band2[1], :].sum(axis=1, keepdims=True)

            # Vectorized beta pass
            for i in range(max_len - 2, -1, -1):
                band1 = idx_bands[i]
                band2 = idx_bands[i + 1]
                has_next_loc = has_next[i]

                next_b = pr_obs[band2[0]:band2[1], :]
                prev_a = alphas[band1[0]:band1[1], :]
                prev_a = prev_a[has_next_loc, :]

                prev_beta *= next_b

                prev_a = np.reshape(prev_a, (prev_a.shape[0], prev_a.shape[1], 1))
                next_beta2 = np.reshape(prev_beta, (prev_beta.shape[0], 1, prev_beta.shape[1]))
                xi_loc = next_beta2 * a_mat
                next_beta = xi_loc.sum(axis=2)
                next_beta_max = next_beta.max(axis=1, keepdims=True)
                next_beta_max[next_beta_max == 0] = 1.0
                next_beta /= next_beta_max

                prev_beta = np.ones((band1[1] - band1[0], num_states))
                prev_beta[has_next_loc, :] = next_beta

                xi_loc *= prev_a
                # xi_loc = np.einsum('Bi,ij,Bj->Bij', prev_a, A, next_beta)
                xi_loc_sum = xi_loc.sum(axis=1, keepdims=True).sum(axis=2, keepdims=True)
                len_vec_loc = np.reshape(len_vec[good[:, i + 1]], (-1, 1, 1)) - 1
                weights_loc = np.reshape(weights[good[:, i + 1]], (-1, 1, 1))
                # xi_loc *= weights_loc/(len_vec_loc*xi_loc_sum)

                xi_loc_sum[xi_loc_sum == 0] = 1.0

                xi_loc *= weights_loc / xi_loc_sum

                temp = xi_loc.sum(axis=2)
                temp_sum = temp.sum(axis=1, keepdims=True)
                temp_sum[temp_sum == 0] = 1.0
                temp /= temp_sum

                alphas[band1[0] + has_next_loc, :] = temp

                self.trans_counts += xi_loc.sum(axis=0)

            # Aggregate sufficient statistics
            for i in range(num_states):
                # alphas[:,i] *= weights[idx_vec]/np.maximum(len_vec[idx_vec], 1.0)
                alphas[:, i] *= weights[idx_vec]
                self.accumulators[i].seq_update(enc_data, alphas[:, i], estimate.topics[i])

            self.state_counts += alphas.sum(axis=0)

            band1 = idx_bands[0]
            temp = alphas[band1[0]:band1[1], :].sum(axis=1, keepdims=True)
            temp[temp == 0] = 1.0
            alphas[band1[0]:band1[1], :] *= np.reshape(weights[good[:, 0]], (-1, 1)) / temp

            self.init_counts += alphas[band1[0]:band1[1], :].sum(axis=0)

            if self.len_accumulator is not None:
                self.len_accumulator.seq_update(len_enc, weights, estimate.len_dist)

        else:
            (idx, sz, enc_data), len_enc = x1

            tot_cnt = len(idx)
            seq_cnt = len(sz)
            num_states = estimate.n_states
            pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)

            max_len = sz.max()
            tz = np.concatenate([[0], sz]).cumsum().astype(dtype=np.int32)

            init_pvec = estimate.w
            tran_mat = estimate.transitions

            # Compute state likelihood vectors and scale the max to one
            for i in range(num_states):
                pr_obs[:, i] = estimate.topics[i].seq_log_density(enc_data)

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
                self.accumulators[i].seq_update(enc_data, alphas[:, i], estimate.topics[i])

            self.state_counts += alphas.sum(axis=0)

            if self.len_accumulator is not None:
                self.len_accumulator.seq_update(len_enc, weights, estimate.len_dist)

    def combine(self, suff_stat: Tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[T1], Optional[T2]]) \
            -> 'HiddenMarkovAccumulator':
        """Combine the sufficient statistics of HiddenMarkovAccumulator with suff_stat arg.

        Sufficient statistics in suff_stat are a Tuple containing:
            suff_stat[0] (int): Number of hidden states.
            suff_stat[1] (np.ndarray): Initial state counts.
            suff_stat[2] (np.ndarray): State counts.
            suff_stat[3] (np.ndarayy): State transition counts.
            suff_stat[4] (Sequence[T1]): Emission distribution accumulators.
            suff_stat[5] (Optional[T2]): Optional sufficient statistics of the length distribution.

        Note: T1 is the assumed type for the emission accumulator sufficient statistics. T2 is the assumed type for the
        length accumulator sufficient statistics.

        Args:
            suff_stat: See above for details.

        Returns:
            HiddenMarkovAccumulator object.

        """
        num_states, init_counts, state_counts, trans_counts, acc_values, len_acc_value = suff_stat

        self.init_counts += init_counts
        self.state_counts += state_counts
        self.trans_counts += trans_counts

        for i in range(self.num_states):
            self.accumulators[i].combine(acc_values[i])

        if len_acc_value is not None:
            self.len_accumulator.combine(len_acc_value)

        return self

    def value(self) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[Any],
                             Optional[Any]]:
        """Returns sufficient statistics of HiddenMarkovAccumulator object instance.

        Returned value rv is a Tuple containing:
            rv[0] (int): Number of hidden states.
            rv[1] (np.ndarray): Initial state counts.
            rv[2] (np.ndarray): State counts.
            rv[3] (np.ndarray): State transition counts.
            rv[4] (Sequence[T1]): Emission distribution accumulator sufficient statistics (type T1).
            rv[5] (Optional[T2]): Optional sufficient statistics of the length distribution (type T2).

        Note: T1 is the assumed type for the emission accumulator sufficient statistics. T2 is the assumed type for the
        length accumulator sufficient statistics.

        Returns:
            Tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[T1], Optional[T2]].

        """
        len_val = self.len_accumulator.value()

        return self.num_states, self.init_counts, self.state_counts, self.trans_counts, tuple(
            [u.value() for u in self.accumulators]), len_val

    def from_value(self, x: Tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[T1], Optional[T2]])\
            -> 'HiddenMarkovAccumulator':
        """Set the sufficient statistics of HiddenMarkovAccumulator object instance to value x.

        Returned value x is a Tuple containing:
            x[0] (int): Number of hidden states.
            x[1] (np.ndarray): Initial state counts.
            x[2] (np.ndarray): State counts.
            x[3] (np.ndarayy): State transition counts.
            x[4] (List[T1]): Emission distribution accumulators.
            x[5] (Optional[T2]): Optional sufficient statistics of the length distribution.

        Note: T1 is the assumed type for the emission accumulator sufficient statistics. T2 is the assumed type for the
        length accumulator sufficient statistics.

        Args:
            x: See above for details.

        Returns:
            HiddenMarkovAccumulator object.

        """
        num_states, init_counts, state_counts, trans_counts, accumulators, len_acc = x
        self.num_states = num_states
        self.init_counts = init_counts
        self.state_counts = state_counts
        self.trans_counts = trans_counts

        for i, v in enumerate(accumulators):
            self.accumulators[i].from_value(v)

        if self.len_accumulator is not None:
            self.len_accumulator.from_value(len_acc)

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge the sufficient statistics of object instance with sufficient statistics in suff_stat that have
            matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dictionary containing sufficient statistics for corresponding keys.

        Returns:
            None.

        """
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
                    acc[i] = acc[i].combine(self.accumulators[i].value())
            else:
                stats_dict[self.state_key] = self.accumulators

        for u in self.accumulators:
            u.key_merge(stats_dict)

        if self.len_accumulator is not None:
            self.len_accumulator.key_merge(stats_dict)

        return None

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace the sufficient statistics of HiddenMarkovAccumulator object with matching sufficient statistics in
            arg suff_stat that have matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to sufficient statistics.

        Returns:
            None.

        """
        if self.init_key is not None:
            if self.init_key in stats_dict:
                self.init_counts = stats_dict[self.init_key]

        if self.trans_key is not None:
            if self.trans_key in stats_dict:
                self.trans_counts = stats_dict[self.trans_key]

        if self.state_key is not None:
            if self.state_key in stats_dict:
                self.accumulators = stats_dict[self.state_key]

        for u in self.accumulators:
            u.key_replace(stats_dict)

        if self.len_accumulator is not None:
            self.len_accumulator.key_replace(stats_dict)

        return None

    def acc_to_encoder(self) -> 'HiddenMarkovDataEncoder':
        """Returns HiddenMarkovDataEncoder object for encoding sequences of iid HMM observations."""
        emission_encoder = self.accumulators[0].acc_to_encoder()
        len_encoder = self.len_accumulator.acc_to_encoder()

        return HiddenMarkovDataEncoder(emission_encoder=emission_encoder, len_encoder=len_encoder,
                                       use_numba=self.use_numba)


class HiddenMarkovAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, factories: Sequence[StatisticAccumulatorFactory],
                 len_factory: StatisticAccumulatorFactory = NullAccumulatorFactory(),
                 use_numba: bool = False,
                 keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (None, None, None),
                 name: Optional[str] = None) -> None:
        """HiddenMarkovAccumulatorFactory object for creating HiddenMarkovEstimatorAccumulator objects.

        Args:
            factories (Sequence[StatisticAccumulatorFactory]): StatisticAccumulatorFactory object for the emission
                distributions.
            len_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for the length distribution.
            use_numba (bool): Default to True.
            keys (Optional[Tuple[Optional[str],Optional[str], Optional[str]]]): Set keys for initial states, state
                transitions, and the emission distributions.
            name (Optional[str]): Name for object.

        Attributes:
            factories (Sequence[StatisticAccumulatorFactory]): StatisticAccumulatorFactory object for the emission
                distributions.
            len_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for the length distribution. Defaults
                to NullAccumulatorFactory().
            use_numba (bool): Default to True. Indicated if Numbda is to be used for 'seq_' calls.
            keys (Tuple[Optional[str],Optional[str], Optional[str]]): Set keys for initial states, state
                transitions, and the emission distributions.
            name (Optional[str]): Name for object.


        """
        self.factories = factories
        self.use_numba = use_numba
        self.keys = keys if keys is None else (None, None, None)
        self.len_factory = len_factory
        self.name = name

    def make(self) -> 'HiddenMarkovAccumulator':
        """Returns a HiddenMarkovAccumulator object. """
        len_acc = self.len_factory.make() if self.len_factory is not None else None
        return HiddenMarkovAccumulator([self.factories[i].make() for i in range(len(self.factories))],
                                       len_accumulator=len_acc, use_numba=self.use_numba, keys=self.keys,
                                       name=self.name)

class HiddenMarkovEstimator(ParameterEstimator):

    def __init__(self, estimators: List[ParameterEstimator],
                 len_estimator: Optional[ParameterEstimator] = NullEstimator(),
                 pseudo_count: Optional[Tuple[Optional[float], Optional[float]]] = (None, None),
                 name: Optional[str] = None,
                 keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (None, None, None),
                 use_numba: bool = False) -> None:
        """HiddenMarkovEstimator object for estimating HiddenMarkovDistribution for aggregated sufficient statistics.

        Args:
            estimators (List[ParameterEstimator]): Set ParameterEstimator objects for emission distributions.
            len_estimator (Optional[ParameterEstimator]): Optional ParameterEstimator object for length distribution.
            pseudo_count (Optional[Tuple[Optional[float], Optional[float]]]): Pseudo count for initial states and
                state transitions.
            name (Optional[str]): Set name to object.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set keys for initial states,
                transitions counts, and emission distributions.
            use_numba (bool): If True, Numba is used for sequence encoding and vectorized functions.

        Attributes:
            estimators (List[ParameterEstimator]): Set ParameterEstimator objects for emission distributions.
            len_estimator (ParameterEstimator): ParameterEstimator object for length distribution, set to NullEstimator
                if None was passed.
            pseudo_count (Tuple[Optional[float], Optional[float]]): Pseudo count for initial states and
                state transitions. Defaults to Tuple of (None, None) if None was passed.
            name (Optional[str]): Name for object instance.
            keys (Tuple[Optional[str], Optional[str], Optional[str]]): Keys for initial states, transitions counts, and
                emission distributions. Defaults to Tuple of (None, None, None).
            use_numba (bool): If True, Numba is used for sequence encoding and vectorized functions.

        """
        self.num_states = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count if pseudo_count is not None else (None, None)
        self.keys = keys if keys is not None else (None, None, None)
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.name = name
        self.use_numba = use_numba

    def accumulator_factory(self):
        """Returns an HiddenMarkovAccumulatorFactory object."""
        est_factories = [u.accumulator_factory() for u in self.estimators]
        len_factory = self.len_estimator.accumulator_factory()
        return HiddenMarkovAccumulatorFactory(est_factories, len_factory, self.use_numba, self.keys, self.name)

    def estimate(self, nobs: Optional[float],
                 suff_stat: Tuple[int, np.ndarray, np.ndarray, np.ndarray, List[T1], Optional[T2]])\
            -> 'HiddenMarkovModelDistribution':
        """Estimate HiddenMarkovModel from aggregated sufficient statistics contained in arg 'suff_stat'.

        Sufficient statistics in arg 'suff_stat' are a Tuple containing:
            suff_stat[0] (int): Number of hidden states.
            suff_stat[1] (np.ndarray): Initial state counts.
            suff_stat[2] (np.ndarray): State counts.
            suff_stat[3] (np.ndarayy): State transition counts.
            suff_stat[4] (List[T1]): List of Sufficient statistics for the emission distribution accumulators.
                Each having type S0.
            suff_stat[5] (Optional[T2]): Optional sufficient statistics of the length distribution.

        Note: T1 is the type for the sufficient statistics of the emission accumulators. T2 is the type for the
        length accumulator.

        If pseudo_count[0] is not None, the initial counts in 'suff_stat' is re-weighted in estimation.
        If pseudo_count[1] is not None, the transition counts in 'suff_stat' are re-weighted in estimation.


        Args:
            nobs (Optional[float]): Number of observations used in estimation.
            suff_stat: See above for details.

        Returns:
            HiddenMarkovModelDistribution object.

        """
        num_states, init_counts, state_counts, trans_counts, topic_ss, len_ss = suff_stat

        len_dist = self.len_estimator.estimate(nobs, len_ss)
        topics = [self.estimators[i].estimate(state_counts[i], topic_ss[i]) for i in range(num_states)]

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

            bad_rows = row_sum.flatten() == 0.0

            if np.any(bad_rows):
                good_rows = ~bad_rows
                transitions = np.zeros_like(trans_counts, dtype=np.float64)
                transitions[good_rows, :] += trans_counts[good_rows, :] / row_sum[good_rows]
            else:
                transitions = trans_counts / row_sum

        return HiddenMarkovModelDistribution(topics=topics, w=w, transitions=transitions, taus=None, len_dist=len_dist,
                                             name=self.name, terminal_values=None, use_numba=self.use_numba)

class HiddenMarkovDataEncoder(DataSequenceEncoder):

    def __init__(self, emission_encoder: DataSequenceEncoder,
                 len_encoder: Optional[DataSequenceEncoder] = NullDataEncoder(), use_numba: bool = False) -> None:
        """HiddenMarkovDataEncoder object for encoding sequences of iid HMM observations.

        Args:
            emission_encoder (DataSequenceEncoder): DataSequenceEncoder object of type T for the observed
                emission distribution values.
            len_encoder (Optional[DataSequenceEncoder]): Optional DataSequenceEncoder object for the length
                of sequences. Should have support of non-negative integers.
            use_numba (bool): If True, sequence encode for Numba.

        Attributes:
            emission_encoder (DataSequenceEncoder): DataSequenceEncoder object of type T for the observed
                emission distribution values.
            len_encoder (DataSequenceEncoder): DataSequenceEncoder object for the length of sequences.
                Should have support of non-negative integers. Set to NullDataEncoder if None.
            use_numba (bool): If True, sequence encode for Numba.

        """
        self.emission_encoder = emission_encoder
        self.len_encoder = len_encoder if len_encoder is not None else NullDataEncoder()
        self.use_numba = use_numba

    def __str__(self) -> str:
        """Returns string representation of HiddenMarkovDataEncoder object instance."""
        s = 'HiddenMarkovDataEncoder(emission_encoder=' + str(self.emission_encoder) + ','
        s += 'len_encoder=' + str(self.len_encoder) + ","
        s += 'use_numba=' + str(self.use_numba) + ')'
        return s

    def __eq__(self, other: object) -> bool:
        """Check if other is equivalent to HiddenMarkovDataEncoder object instance.

        Args:
            other (Object): Object to compare to HiddenMarkovDataEncoder object instance.

        Returns:
            True if other is HiddenMarkovDataEncoder with equivalent 'len_encoder' and 'use_numba', else False.

        """
        if isinstance(other, HiddenMarkovDataEncoder):
            if self.use_numba == other.use_numba:
                if self.len_encoder == other.len_encoder:
                    return True
        else:
            return False

    def _seq_encode(self, x: List[List[T]]) -> Tuple[E1, None]:
        """Sequence encoding for iid HMM sequence for vectorized numpy functions that do not use numba.

        Encoding  x: List[List[T]) where x[i] the ith HMM sequence of length n_i, s.t. x[i] = [x[i][0],...,x[i][n_i]].
        Call the t^th observation in the ith HMM sequence x[i][t].

        Blocks observations of each HMM sequence into blocks of same 't' value. I.e.
            seq_x = [ x[0][0],...,x[cnt][0], x[0][1],x[1][1],...,x[cnt][1],...]
        Note: That seq_x chunks will include x[i][t] values only if the sequence x[i] is length >= t.

        The returned value rv is a Tuple[Tuple[....], T_topic, T_len], where the first tuple is given by a Tuple of
            rv[0][0] (int): Total number of observed emissions from all HMM sequences.
            rv[0][1] (List[Tuple[int, int]]): Contains bands for t^th observation in HMM sequences stored in 'seq_x'.
            rv[0][2] (List[ndarray[int]]): List of numpy array on sequence indices that have a next observed emission.
            rv[0][3] (np.ndarray[int]): Numpy array of sequence lengths.
            rv[0][4] (np.ndarray[int]): 2-d matrix with rv[0][0] rows, and column length equal to the length of the
                largest HMM sequence. This is used to store the index of seq_x corresponding to emission x[i][t]. A -1
                is stored if the sequence length has already been met.
            rv[0][5] (ndarray): Numpy array containing lists index 'i' corresponding to x[i][t] block of 'seq_x'.
            rv[0][6] (T_topic): Sequence encoded value of 'seq_x'.

        The second entry of 'rv' is given by,
            rv[1] (T_topic): Sequence encoded observation values in order. Just for seq_init consistency.
            rv[2] (Optional[T_len]): Sequence encoded value of lengths of HMM distribution. None if len_encoder is
                the NullDataEncoder.

        Args:
            x(List[List[T]]): A sequence of iid observations from an HMM distribution of type T.

        Returns:
            Tuple[rv, None].

        """
        cnt = len(x)
        len_vec = [len(u) for u in x]
        len_enc = self.len_encoder.seq_encode(len_vec)

        len_vec = np.asarray(len_vec)
        max_len = len_vec.max()
        # len_cnt = np.bincount(len_vec)

        seq_x = []
        idx_loc = 0
        idx_mat = np.zeros((cnt, max_len), dtype=int) - 1
        idx_bands = []
        has_next = []
        idx_vec = []

        for i in range(max_len):
            i0 = idx_loc
            has_next_loc = []
            for j in range(cnt):
                if i < len_vec[j]:
                    if i < (len_vec[j] - 1):
                        has_next_loc.append(idx_loc - i0)
                    idx_vec.append(j)
                    seq_x.append(x[j][i])
                    idx_mat[j, i] = idx_loc
                    idx_loc += 1

            has_next.append(np.asarray(has_next_loc))
            idx_bands.append((i0, idx_loc))

        tot_cnt = len(seq_x)
        enc_data = self.emission_encoder.seq_encode(seq_x)
        idx_vec = np.asarray(idx_vec)

        xs = []
        for xx in x:
            if len(xx) > 0:
                xs.extend(xx)
        xs_enc = self.emission_encoder.seq_encode(xs)

        rv = ((tot_cnt, idx_bands, has_next, len_vec, idx_mat, idx_vec, enc_data), xs_enc, len_enc)
        return rv, None

    def seq_encode(self, x: List[List[T]]) -> Tuple[Optional[E1], Optional[E2]]:
        """Sequence encode sequences of iid HMM observations.

        Numba sequence encoding: Return type Tuple[Tuple[np.ndarray, np.ndarray, T_topic], Optional[T_len]] where
        T_topicis the type for 'emission_encoder.seq_encode()' and T_len is the type for 'len_encoder.seq_encode()'.
        The first entry of the returned value (rv_numba) is a Tuple of length-3,

            rv_numba[0][0] (ndarray[int]): Sequence id's for observed values.
            rv_numba[0][1] (ndarray[int]): Sequence lengths for each observed HMM sequence.
            rv_numba[0][2] (T_topic): Sequence encoded observation values.
            rv_numba[1] (Optional[T_len]): Sequence encoded values of sequence lengths. None if len_encoder is
                NullDataEncoder.

        If use_numba is False, calls HiddenMarkovDataEncoder._seq_encode(x). (See '_seq_encode' for details).


        Args:
            x (List[List[T]]): A sequence of iid observations from an HMM distribution of type T.

        Returns:
            Tuple[None, rv_numba] if use_numba, else Tuple[rv, None].

        """
        if not self.use_numba:
            return self._seq_encode(x)

        idx = []
        xs = []
        sz = []
        seq_x = []

        for i, xx in enumerate(x):
            idx.extend([i] * len(xx))
            xs.extend(xx)
            sz.append(len(xx))
            if sz[-1] > 0:
                seq_x.extend([xx[j] for j in range(sz[-1])])

        len_enc = self.len_encoder.seq_encode(sz)

        idx = np.asarray(idx, dtype=np.int32)
        sz = np.asarray(sz, dtype=np.int32)
        xs = self.emission_encoder.seq_encode(xs)

        return None, ((idx, sz, xs), len_enc)

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
    'void(int32, int32[:], float64[:,:], float64[:], float64[:,:], float64[:], float64[:,:], float64[:,:], float64[:], '
    'float64[:], float64[:,:])')
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
    'void(int64, int32[:], float64[:,:], float64[:], float64[:,:], float64[:], float64[:,:], float64[:,:,:], '
    'float64[:,:])',
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
    'void(int64, int32[:], float64[:,:], float64[:], float64[:,:], float64[:], float64[:,:], float64[:,:,:], '
    'float64[:,:])',
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


@numba.njit('float64[:,:](int32[:], float64[:,:], float64[:,:])')
def vec_bincount1(x, w, out):
    """Numba bincount on the rows of matrix w for groups x.

    Args:
        x (np.ndarray[np.float64]): Group ids of rows
        w (np.ndarray[np.float64]): N by S numpy array with rows corresponding to x
        out (np.ndarray[np.float64]): Unique values in support of x by S.

    Returns:
        Numpy 2-d array.

    """
    for i in range(len(x)):
        out[x[i], :] += w[i, :]
    return out


@numba.njit('float64[:,:](int32[:], float64[:,:], float64[:,:])')
def vec_bincount2(x, w, out):
    """Numba bincount on the rows of matrix w for groups x.

    N = len(x)
    S = number of states.
    U = unique values in x can take on.

    Args:
        x (np.ndarray[np.float64]): Group ids of columns of w.
        w (np.ndarray[np.float64]): S by N numpy array with cols corresponding to x
        out (np.ndarray[np.float64]): S by U matrix.

    Returns:
        Numpy 2-d array.

    """
    for j in range(len(x)):
        out[:, x[j]] += w[:, j]
    return out
