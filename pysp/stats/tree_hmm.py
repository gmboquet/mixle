import math
from typing import List, Any, Tuple, Sequence, Union, Optional, TypeVar, Dict
import itertools
import numba
import numpy as np
from numpy.random import RandomState
import pysp.utils.vector as vec
from pysp.arithmetic import *
from pysp.arithmetic import maxrandint
from pysp.stats.null_dist import NullDistribution, NullAccumulatorFactory, NullEstimator, NullDataEncoder, \
    NullAccumulator
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator, DataSequenceEncoder, DistributionSampler, StatisticAccumulatorFactory

D = Tuple[int, Optional[int]]
T = TypeVar('T')  # Type for emissions
SS0 = TypeVar('SS0')  # Type for suff stat of emissions
SS1 = TypeVar('SS1')  # Type for suff-stat of length dist

E1 = Tuple[int, np.ndarray, np.ndarray, np.ndarray]
E2 = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
E3 = TypeVar('E3')  # Encoded emissions
E4 = TypeVar('E4')  # encoded lengths of children
E5 = Tuple[np.ndarray, np.ndarray, np.ndarray]
E6 = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray],
           List[np.ndarray], np.ndarray]
E01 = Tuple[np.ndarray, E1, E2, E3, Optional[Tuple[np.ndarray, E4]]]
E02 = Tuple[int, np.ndarray, E5, E6, E3, Optional[Tuple[np.ndarray, E4]]]
E = Tuple[Optional[E01], Optional[E02]]


# def get_combos(u: Union[List[int], np.ndarray]) -> Tuple[List[int], List[int]]:
#     """Get
#
#     Args:
#         u:
#
#     Returns:
#
#     """
#     v = np.asarray(u, dtype=np.int32)
#     nv = len(v) - 1
#     combs = itertools.combinations(v, nv)
#     singles = [v[i] for i in range(nv, -1, -1) for j in range(nv)]
#     return [v for x in list(combs) for v in x], singles


def find_level(parents: np.ndarray) -> List[int]:
    """Find the level in the tree for nodes, given an array of parents.

    Args:
        parents (np.ndarray): Numpy array of integers with first entry -1.

    Returns:
        Level of each node in the free excluding the first entry which is the root (level = 0).

    """
    n = len(parents)
    if n == 1:
        return []
    out = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        out[i] = out[parents[i]] + 1
    return list(out[1:])


class TreeHiddenMarkovModelDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, topics: Sequence[SequenceEncodableProbabilityDistribution],
                 w: Union[Sequence[float], np.ndarray],
                 transitions: Union[List[List[float]], np.ndarray],
                 len_dist: Optional[SequenceEncodableProbabilityDistribution] = NullDistribution(),
                 terminal_level: int = 10,
                 name: Optional[str] = None,
                 use_numba: bool = False) -> None:
        """TreeHiddenMarkovModelDistribution for specifying an HMM on a rooted tree.

        Args:
            topics (Sequence[SequenceEncodableProbabilityDistribution]): Emission distributions having type T.
            w (Union[Sequence[float], np.ndarray]): Initial state weights. Must sum to 1 and have same length as topics.
            transitions (Union[List[List[float]], np.ndarray]): Define the TPM for HMM. Dim is len(topics) by
                len(topics).
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution for the number of children
                a node in the tree will have. Must have support on non-negative integers.
            terminal_level (int): Level of tree to terminate sampling. Default to 10.
            name (Optional[str]): Assign a name to object instance.
            use_numba (bool): If true Numba is used for vectorized calculations.

        Attributes:
            topics (Sequence[SequenceEncodableProbabilityDistribution]): Emission distributions having type T.
            num_states (int): Number of states in HMM.
            w (np.ndarray): Initial state distribution. Sums to 1.
            log_w (np.ndarray): Log of above.
            transitions (np.ndarray): TPM with dimensions num_states by num_states.
            log_transitions (np.ndarray): Log of TPM.
            len_dist (SequenceEncodableProbabilityDistribution): Distribution for number of children for a node.
                Defaults to NullDistribution.
            terminal_level (int): Level in tree to terminate sampling.
            use_numba (bool): If true Numba used for computations.

        """

        with np.errstate(divide='ignore'):
            self.topics = topics
            self.num_states = len(w)
            self.w = vec.make(w)
            self.log_w = np.log(self.w)

            if not isinstance(transitions, np.ndarray):
                transitions = np.asarray(transitions, dtype=float)

            self.transitions = np.reshape(transitions, (self.num_states, self.num_states))
            self.log_transitions = np.log(self.transitions)
            self.name = name
            self.len_dist = len_dist if len_dist is not None else NullDistribution()
            self.terminal_level = terminal_level
            self.use_numba = use_numba

    def __str__(self) -> str:
        s1 = ','.join(map(str, self.topics))
        s2 = repr(list(self.w))
        s3 = repr([list(u) for u in self.transitions])
        s4 = str(self.len_dist)
        s5 = repr(self.name)
        s6 = repr(self.use_numba)

        return 'TreeHiddenMarkovModelDistribution(topics=[%s], w=%s, transitions=%s, len_dist=%s, name=%s, ' \
               'use_numba=%s)' % (s1, s2, s3, s4, s5, s6)

    def density(self, x: Sequence[Tuple[D, T]]) -> float:
        return exp(self.log_density(x))

    def log_density(self, x: Sequence[Tuple[D, T]]) -> float:
        enc_x = self.dist_to_encoder().seq_encode([x])
        return self.seq_log_density(enc_x)[0]

    def seq_log_density(self, x: E) -> np.ndarray:

        if self.use_numba:
            tz, (max_level, xln, xlnl, tlnz), (xbi, xp, xc, xl, txz, tp, tpz), enc_x, len_enc = x[0]

            num_states = self.num_states
            w = self.w
            a_mat = self.transitions
            tot_cnt = tz[-1]
            num_trees = len(tz)-1

            p_level = np.zeros((max_level+1, num_states), dtype=np.float64)
            level_state_prob(max_level+1, num_states, a_mat, w, p_level)

            pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)
            ll_ret = np.zeros(num_trees, dtype=np.float64)

            # Compute state likelihood vectors and scale the max to one
            for i in range(num_states):
                pr_obs[:, i] = self.topics[i].seq_log_density(enc_x)

            pr_max0 = pr_obs.max(axis=1)
            pr_obs -= pr_max0[:, None]
            np.exp(pr_obs, out=pr_obs)

            betas = np.ones_like(pr_obs, dtype=np.float64)
            etas = np.zeros((len(xbi), num_states), dtype=np.float64)

            numba_seq_log_density(num_states, tz, txz, tp, tpz, tlnz, xp, xc, xl, xbi, xln, xlnl, pr_obs, p_level, a_mat,
                                  pr_max0, betas, etas, ll_ret)

            # if len_enc is not None:
            #     ret_len = np.zeros(num_trees, dtype=np.float64)
            #     ll_ret += vec_bincount(len_enc[0], self.len_dist.seq_log_density(len_enc[1]), ret_len)

            return ll_ret

        else:

            cnt, tz, (xln, xlnl, xlni), (idx, xbi, xp, xc, level_idx, p_nxt, eta_p, i_nxt, _), enc_x, len_enc = x[1]

            num_states = self.num_states
            max_level = len(level_idx)
            a_mat = self.transitions
            w = self.w
            num_trees = len(tz)-1

            betas = np.ones((cnt, num_states), dtype=np.float64)
            etas = np.zeros((len(xbi), num_states), dtype=np.float64)

            p_level = np.zeros((max_level+1, num_states), dtype=np.float64)
            p_level[0, :] += w

            for level in range(1, max_level+1):
                p_level[level, :] += np.matmul(p_level[level-1, :], a_mat)

            pr_obs = np.zeros((cnt, num_states), dtype=np.float64)
            ll_ret = np.zeros(num_trees, dtype=np.float64)

            # Compute state likelihood vectors and scale the max to one
            for i in range(num_states):
                pr_obs[:, i] = self.topics[i].seq_log_density(enc_x)

            pr_max0 = pr_obs.max(axis=1)
            pr_obs -= pr_max0[:, None]
            np.exp(pr_obs, out=pr_obs)

            #  set the leaf nodes
            betas[xln, :] *= pr_obs[xln, :] * p_level[xlnl, :]
            betas_sum = np.sum(betas[xln, :], axis=1, keepdims=True)
            betas[xln, :] /= betas_sum

            ll_ret += np.bincount(xlni, weights=np.log(betas_sum.flatten()) + pr_max0[xln], minlength=num_trees)

            #  upward pass on betas
            for level in range(len(level_idx)-1, -1, -1):

                lidx = level_idx[level]
                idxs, xbis, xps, xcs = idx[lidx], xbi[lidx], xp[lidx], xc[lidx]

                #  Get etas
                temp = np.reshape(betas[xcs, :], (-1, num_states, 1))
                temp /= np.reshape(p_level[level+1, :], (1, num_states, 1))
                temp = np.sum(a_mat.T * temp, axis=1)
                etas[xbis, :] += temp

                temp = np.zeros((len(xbis)+1, num_states), dtype=np.float64)
                temp[1:, :] += np.log(etas[xbis, :])
                log_etas = np.cumsum(temp, axis=0)
                log_etas = log_etas[eta_p[level][1:], :] - log_etas[eta_p[level][:-1], :]

                betas[p_nxt[level], :] *= np.exp(log_etas) * pr_obs[p_nxt[level], :]
                betas[p_nxt[level], :] *= p_level[level, :]
                betas_sum = np.sum(betas[p_nxt[level], :], axis=1, keepdims=True)

                betas[p_nxt[level], :] /= betas_sum

                ll_ret += np.bincount(i_nxt[level], weights=np.log(betas_sum.flatten())+pr_max0[p_nxt[level]],
                                      minlength=num_trees)

            # if len_enc is not None:
            #     ret_len = np.zeros(num_trees, dtype=np.float64)
            #     ll_ret += vec_bincount(len_enc[0], self.len_dist.seq_log_density(len_enc[1]), ret_len)

            return ll_ret

    def seq_posterior(self, x: E) -> Optional[List[np.ndarray]]:

        if self.use_numba:
            tz, (max_level, xln, xlnl, tlnz), (xbi, xp, xc, xl, txz, tp, tpz), enc_x, _ = x[0]

            num_states = self.num_states
            w = self.w
            a_mat = self.transitions
            tot_cnt = tz[-1]

            p_level = np.zeros((max_level + 1, num_states), dtype=np.float64)
            level_state_prob(max_level + 1, num_states, a_mat, w, p_level)

            pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)

            # Compute state likelihood vectors and scale the max to one
            for i in range(num_states):
                pr_obs[:, i] = self.topics[i].seq_log_density(enc_x)

            pr_max0 = pr_obs.max(axis=1)
            pr_obs -= pr_max0[:, None]
            np.exp(pr_obs, out=pr_obs)

            betas = np.zeros_like(pr_obs, dtype=np.float64)
            etas = np.zeros((len(xbi), num_states), dtype=np.float64)

            ### Need to do upward and downward, then read back the gammas
            numba_posteriors(num_states, tz, txz, tp, tpz, tlnz, xp, xc, xl, xbi, xln, xlnl, pr_obs, p_level, a_mat,
                             betas, etas)

            return [betas[tz[i]:tz[i + 1], :] for i in range(len(tz) - 1)]

        else:
            cnt, tz, (xln, xlnl, xlni), (idx, xbi, xp, xc, level_idx, p_nxt, eta_p, i_nxt, _), enc_x, len_enc = x[1]

            num_states = self.num_states
            max_level = len(level_idx)
            a_mat = self.transitions
            w = self.w
            num_trees = len(tz) - 1

            betas = np.ones((cnt, num_states), dtype=np.float64)
            etas = np.zeros((len(xbi), num_states), dtype=np.float64)

            p_level = np.zeros((max_level + 1, num_states), dtype=np.float64)
            p_level[0, :] += w

            for level in range(1, max_level + 1):
                p_level[level, :] += np.matmul(p_level[level - 1, :], a_mat)

            pr_obs = np.zeros((cnt, num_states), dtype=np.float64)

            # Compute state likelihood vectors and scale the max to one
            for i in range(num_states):
                pr_obs[:, i] = self.topics[i].seq_log_density(enc_x)

            pr_max0 = pr_obs.max(axis=1)
            pr_obs -= pr_max0[:, None]
            np.exp(pr_obs, out=pr_obs)

            #  set the leaf nodes
            betas[xln, :] *= pr_obs[xln, :] * p_level[xlnl, :]
            betas_sum = np.sum(betas[xln, :], axis=1, keepdims=True)
            betas[xln, :] /= betas_sum

            #  upward pass on betas
            for level in range(len(level_idx) - 1, -1, -1):
                lidx = level_idx[level]
                idxs, xbis, xps, xcs = idx[lidx], xbi[lidx], xp[lidx], xc[lidx]

                #  Get etas
                temp = np.reshape(betas[xcs, :], (-1, num_states, 1))
                temp /= np.reshape(p_level[level + 1, :], (1, num_states, 1))
                temp = np.sum(a_mat.T * temp, axis=1)
                etas[xbis, :] += temp

                temp = np.zeros((len(xbis) + 1, num_states), dtype=np.float64)
                temp[1:, :] += np.log(etas[xbis, :])
                log_etas = np.cumsum(temp, axis=0)
                log_etas = log_etas[eta_p[level][1:], :] - log_etas[eta_p[level][:-1], :]

                betas[p_nxt[level], :] *= np.exp(log_etas) * pr_obs[p_nxt[level], :]
                betas[p_nxt[level], :] *= p_level[level, :]
                betas_sum = np.sum(betas[p_nxt[level], :], axis=1, keepdims=True)

                betas[p_nxt[level], :] /= betas_sum

            #  Return betas by observed sequence need tz
            return [betas[tz[i]:tz[i + 1], :] for i in range(len(tz) - 1)]

    def viterbi(self, x: Sequence[Tuple[D, T]]) -> np.ndarray:
        enc_x = self.dist_to_encoder().seq_encode([x])
        return self.seq_viterbi(enc_x)[0]

    def seq_viterbi(self, x: E) -> List[np.ndarray]:
        if self.use_numba:
            tz, (max_level, xln, xlnl, tlnz), (xbi, xp, xc, xl, txz, tp, tpz), enc_x, _ = x[0]

            num_states = self.num_states
            log_w = self.log_w
            log_a_mat = self.log_transitions
            tot_cnt = tz[-1]

            log_pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)

            for i in range(num_states):
                log_pr_obs[:, i] = self.topics[i].seq_log_density(enc_x)

            betas = np.ones_like(log_pr_obs, dtype=np.float64)
            etas = np.ones((len(xbi), num_states), dtype=np.float64)
            out = np.zeros(tot_cnt, dtype=np.int32)

            numba_viterbi(num_states, tz, txz, tp, tpz, tlnz, xp, xc, xl, xbi, xln, log_pr_obs, log_w, log_a_mat,
                          betas, etas, out)

            return [out[tz[i]:tz[i + 1]] for i in range(len(tz) - 1)]

        else:
            cnt, tz, (xln, xlnl, xlni), (idx, xbi, xp, xc, level_idx, p_nxt, eta_p, i_nxt, rns), enc_x, _ = x[1]

            num_states = self.num_states
            max_level = len(level_idx)
            log_a_mat = self.log_transitions
            log_w = self.log_w

            log_delta = np.ones((cnt, num_states), dtype=np.float64)
            log_eta = np.zeros((len(xbi), num_states), dtype=np.float64)
            state_tracker = np.zeros(cnt, dtype=np.int32)

            # Compute state likelihood vectors, and initialize the deltas for each state
            for i in range(num_states):
                log_delta[:, i] += self.topics[i].seq_log_density(enc_x)

            state_tracker[xln] += np.argmax(log_delta[xln, :], axis=1).flatten()

            #  upward pass on deltas
            for level in range(max_level-1, -1, -1):
                lidx = level_idx[level]
                idxs, xbis, xps, xcs = idx[lidx], xbi[lidx], xp[lidx], xc[lidx]

                #  Get log_etas
                log_eta[xbis, :] += np.max(np.reshape(log_delta[xcs, :], (-1, 1, num_states)) + log_a_mat, axis=2)
                temp = np.zeros((len(xbis)+1, num_states), dtype=np.float64)
                temp[1:, :] += np.cumsum(log_eta[xbis, :], axis=0)
                temp = temp[eta_p[level][1:], :] - temp[eta_p[level][:-1], :]
                log_delta[p_nxt[level], :] += temp
                state_tracker[p_nxt[level]] += np.argmax(log_delta[p_nxt[level], :], axis=1, keepdims=False)

            #  Set the init for leaf nodes
            log_delta[rns, :] += log_w
            state_tracker[rns] += np.argmax(log_delta[rns, :], axis=1).flatten()

            return [state_tracker[tz[i]:tz[i + 1]] for i in range(len(tz) - 1)]

    def sampler(self, seed: Optional[int] = None) -> 'TreeHiddenMarkovSampler':
        if isinstance(self.len_dist, NullDistribution):
            raise Exception('TreeHiddenMarkovSampler requires len_dist with support on non-negative integers')
        return TreeHiddenMarkovSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'TreeHiddenMarkovEstimator':
        len_est = None if self.len_dist is None else self.len_dist.estimator(pseudo_count=pseudo_count)
        comp_ests = [u.estimator(pseudo_count=pseudo_count) for u in self.topics]
        return TreeHiddenMarkovEstimator(comp_ests, pseudo_count=(pseudo_count, pseudo_count), len_estimator=len_est,
                                         name=self.name)

    def dist_to_encoder(self) -> 'TreeHiddenMarkovDataEncoder':
        """Returns TreeHiddenMarkovDataEncoder object for encoding sequences of iid Tree HMM observations."""
        emission_encoder = self.topics[0].dist_to_encoder()
        len_encoder = self.len_dist.dist_to_encoder()

        return TreeHiddenMarkovDataEncoder(emission_encoder=emission_encoder, len_encoder=len_encoder,
                                           use_numba=self.use_numba)


class TreeHiddenMarkovSampler(DistributionSampler):

    def __init__(self, dist: 'TreeHiddenMarkovModelDistribution', seed: Optional[int] = None) -> None:
        self.num_states = dist.num_states
        self.dist = dist
        self.rng = RandomState(seed)
        self.obs_samplers = [topic.sampler(seed=self.rng.randint(maxrandint)) for topic in dist.topics]
        self.init_w = dist.w
        self.transitions = dist.transitions

        if dist.len_dist is not None:
            self.len_sampler = dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))
        else:
            self.len_sampler = None

    def sample_state(self, given_state: int, size: Optional[int] = None) -> Union[int, np.ndarray]:
        return self.rng.choice(self.num_states, p=self.transitions[given_state, :], replace=True, size=size)

    def sample_tree(self, size: Optional[int] = None):
        if size is None:

            seq = []
            xi = 0
            zi = self.rng.choice(self.num_states, p=self.init_w)
            ni = self.len_sampler.sample()
            nodes = [(xi, zi, ni)]
            y0 = self.obs_samplers[zi].sample()

            seq.append(((0, -1), y0))
            iter_cond = True if ni > 0 else False

            cnt = 1
            lvl_cnt = 0

            while iter_cond and lvl_cnt < self.dist.terminal_level:
                nodes_next = []
                for node in nodes:
                    xi, zi, ni = node

                    zj = self.sample_state(given_state=zi, size=ni)
                    nj = self.len_sampler.sample(size=ni)

                    for j in range(ni):
                        if nj[j] > 0:
                            nodes_next.append((cnt + j, zj[j], nj[j]))
                        seq.append(((cnt + j, xi), self.obs_samplers[zj[j]].sample()))
                    cnt += ni
                if len(nodes_next) == 0:
                    iter_cond = False
                else:
                    nodes = [xx for xx in nodes_next]

                lvl_cnt += 1

            return seq

        else:
            return [self.sample_tree() for xx in range(size)]

    def sample(self, size: Optional[int] = None):
        if self.len_sampler is not None:
            return self.sample_tree(size=size)
        else:
            raise RuntimeError('TreeHiddenMarkovSampler requires either a length distribution for number of children.')


class TreeHiddenMarkovAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, accumulators: Sequence[SequenceEncodableStatisticAccumulator],
                 len_accumulator: Optional[SequenceEncodableStatisticAccumulator] = NullAccumulator(),
                 keys: Tuple[Optional[str], Optional[str], Optional[str]] = (None, None, None),
                 name: Optional[str] = None, use_numba: bool = True) -> None:

        self.accumulators = accumulators
        self.num_states = len(accumulators)
        self.init_counts = np.zeros(self.num_states, dtype=np.float64)
        self.trans_counts = np.zeros((self.num_states, self.num_states), dtype=np.float64)
        self.state_counts = np.zeros(self.num_states, dtype=np.float64)
        self.len_accumulator = len_accumulator if len_accumulator is not None else NullAccumulator()

        self.init_key = keys[0]
        self.trans_key = keys[1]
        self.state_key = keys[2]

        self.name = name
        self.use_numba = use_numba

        # protected for initialization.
        self._init_rng: bool = False
        self._len_rng: Optional[RandomState] = None
        self._acc_rng: Optional[List[RandomState]] = None
        self._idx_rng: Optional[RandomState] = None

    def update(self, x: Sequence[Tuple[D, T]], weight: float, estimate: TreeHiddenMarkovModelDistribution) -> None:
        enc_x = estimate.dist_to_encoder().seq_encode([x])
        self.seq_update(enc_x, np.asarray([weight]), estimate)

    def _rng_initialize(self, rng: RandomState) -> None:
        rng_seeds = rng.randint(maxrandint, size=2 + self.num_states)
        self._idx_rng = RandomState(seed=rng_seeds[0])
        self._len_rng = RandomState(seed=rng_seeds[1])
        self._acc_rng = [RandomState(seed=rng_seeds[2 + i]) for i in range(self.num_states)]
        self._w_rng = RandomState(seed=rng.randint(2 ** 30))
        self._init_rng = True

    def initialize(self, x: Sequence[Tuple[D, T]], weight: float, rng: RandomState) -> None:
        if not self._init_rng:
            self._rng_initialize(rng)

        enc_x = self.accumulators[0].acc_to_encoder().seq_encode([x])
        self.seq_initialize(enc_x, weights=np.asarray([weight]), rng=rng)

    def seq_initialize(self, x: E, weights: np.ndarray, rng: np.random.RandomState) -> None:

        if not self._init_rng:
            self._rng_initialize(rng)

        if self.use_numba:

            tz, _, (xbi, xp, xc, xl, txz, tp, tpz), enc_x, len_enc = x[0]

            states = self._idx_rng.choice(self.num_states, replace=True, size=tz[-1])

            numba_initialize(tz, txz, tp, tpz, xp, xc, states, weights, self.init_counts, self.state_counts,
                             self.trans_counts)

            idx = len_enc[0]
            nz_idx, nz_idx_group, nz_idx_rep = np.unique(idx, return_index=True, return_inverse=True)
            weights_nz = weights[nz_idx]

            for i in range(self.num_states):
                w = weights_nz[idx].copy()
                w[states == i] = 0.0
                self.accumulators[i].seq_initialize(enc_x, w, self._acc_rng[i])

            if len_enc is not None:
                self.len_accumulator.seq_initialize(len_enc[1], weights[len_enc[0]], self._len_rng)

        else:
            cnt, tz, _, (idx, xbi, xp, xc, level_idx, p_nxt, eta_p, i_nxt, rns), enc_x, len_enc = x[1]

            num_states = self.num_states
            states = self._idx_rng.choice(self.num_states, replace=True, size=cnt)

            #  Get root node states
            root_states = np.bincount(states[rns], weights=weights[i_nxt[0]], minlength=num_states)
            self.init_counts += root_states
            self.state_counts += root_states

            # count state transitions by the levels
            ns2 = num_states ** 2
            for level in range(len(level_idx)-1, -1, -1):
                lidx = level_idx[level]
                idxs, xps, xcs = idx[lidx], xp[lidx], xc[lidx]

                _, xps_cnt = np.unique(xps, return_counts=True)
                bin_weights = []
                bin_weights.extend([weights[kk] for kk in idxs])

                arr = np.asarray([states[xps], states[xcs]], dtype=np.int32)
                multi_idx = np.ravel_multi_index(arr, (num_states, num_states))

                trans_cnts = np.bincount(multi_idx, weights=bin_weights, minlength=ns2)
                self.trans_counts += np.reshape(trans_cnts, (num_states, num_states))

            obs_idx = len_enc[0]
            nz_idx, nz_idx_group, nz_idx_rep = np.unique(obs_idx, return_index=True, return_inverse=True)
            weights_nz = weights[nz_idx]

            for i in range(self.num_states):
                w = weights_nz[obs_idx].copy()
                w[states == i] = 0.0
                self.accumulators[i].seq_initialize(enc_x, w, self._acc_rng[i])

            if len_enc is not None:
                self.len_accumulator.seq_initialize(len_enc[1], weights[len_enc[0]], self._len_rng)

    def seq_update(self, x: E, weights: np.ndarray, estimate: TreeHiddenMarkovModelDistribution) -> None:

        if self.use_numba:
            tz, (max_level, xln, xlnl, tlnz), (xbi, xp, xc, xl, txz, tp, tpz), enc_x, len_enc = x[0]

            tot_cnt = tz[-1]
            num_states = estimate.num_states
            w = estimate.w
            a_mat = estimate.transitions
            num_trees = len(tz)-1

            p_level = np.zeros((max_level+1, num_states), dtype=np.float64)

            level_state_prob(max_level+1, num_states, a_mat, w, p_level)
            pr_obs = np.zeros((tot_cnt, num_states), dtype=np.float64)

            # Compute state likelihood vectors and scale the max to one
            for i in range(num_states):
                pr_obs[:, i] = estimate.topics[i].seq_log_density(enc_x)

            pr_max0 = pr_obs.max(axis=1)
            pr_obs -= pr_max0[:, None]
            np.exp(pr_obs, out=pr_obs)

            betas = np.zeros((tot_cnt, num_states), dtype=np.float64)
            etas = np.zeros((len(xbi), num_states), dtype=np.float64)
            alphas = np.zeros((tot_cnt, num_states), dtype=np.float64)
            xi_acc = np.zeros((num_trees, num_states, num_states), dtype=np.float64)
            pi_acc = np.zeros((num_trees, num_states), dtype=np.float64)

            numba_baum_welch(num_states, tz, txz, tp, tpz, tlnz, xp, xc, xl, xbi, xln, xlnl, pr_obs, p_level, a_mat,
                             weights, betas, etas, alphas, xi_acc, pi_acc)

            self.init_counts += pi_acc.sum(axis=0)
            self.trans_counts += xi_acc.sum(axis=0)

            for i in range(num_states):
                self.accumulators[i].seq_update(enc_x, alphas[:, i], estimate.topics[i])

            self.state_counts += alphas.sum(axis=0)

            if len_enc is not None:
                self.len_accumulator.seq_update(len_enc[1], weights[len_enc[0]], estimate.len_dist)

        else:
            ## numpy calculation from encoding
            cnt, tz, (xln, xlnl, xlni), (idx, xbi, xp, xc, level_idx, p_nxt, eta_p, i_nxt, rns), enc_x, len_enc = x[1]

            num_states = estimate.num_states
            max_level = len(level_idx)
            a_mat = estimate.transitions
            w = estimate.w
            num_trees = len(tz) - 1

            betas = np.ones((cnt, num_states), dtype=np.float64)
            etas = np.zeros((len(xbi), num_states), dtype=np.float64)
            alphas = np.zeros((cnt, num_states), dtype=np.float64)

            p_level = np.zeros((max_level + 1, num_states), dtype=np.float64)
            p_level[0, :] += w

            for level in range(1, max_level + 1):
                p_level[level, :] += np.matmul(p_level[level - 1, :], a_mat)

            pr_obs = np.zeros((cnt, num_states), dtype=np.float64)

            # Compute state likelihood vectors and scale the max to one
            for i in range(num_states):
                pr_obs[:, i] = estimate.topics[i].seq_log_density(enc_x)

            pr_max0 = pr_obs.max(axis=1)
            pr_obs -= pr_max0[:, None]
            np.exp(pr_obs, out=pr_obs)

            #  set the leaf nodes
            betas[xln, :] *= pr_obs[xln, :] * p_level[xlnl, :]
            betas_sum = np.sum(betas[xln, :], axis=1, keepdims=True)
            betas[xln, :] /= betas_sum

            #  upward pass on betas
            for level in range(len(level_idx) - 1, -1, -1):
                lidx = level_idx[level]
                idxs, xbis, xps, xcs = idx[lidx], xbi[lidx], xp[lidx], xc[lidx]

                #  Get etas
                temp = np.reshape(betas[xcs, :], (-1, num_states, 1))
                temp /= np.reshape(p_level[level + 1, :], (1, num_states, 1))
                temp = np.sum(a_mat.T * temp, axis=1)
                etas[xbis, :] += temp

                temp = np.zeros((len(xbis) + 1, num_states), dtype=np.float64)
                temp[1:, :] += np.log(etas[xbis, :])
                log_etas = np.cumsum(temp, axis=0)
                log_etas = log_etas[eta_p[level][1:], :] - log_etas[eta_p[level][:-1], :]

                betas[p_nxt[level], :] *= np.exp(log_etas) * pr_obs[p_nxt[level], :]
                betas[p_nxt[level], :] *= p_level[level, :]
                betas_sum = np.sum(betas[p_nxt[level], :], axis=1, keepdims=True)

                betas[p_nxt[level], :] /= betas_sum

            ## alpha (upward pass) set the root nodes
            alphas[rns, :] += betas[rns, :]

            for level in range(len(level_idx)):
                lidx = level_idx[level]
                idxs, xbis, xps, xcs = idx[lidx], xbi[lidx], xp[lidx], xc[lidx]
                weights_loc = np.reshape(weights[idxs], (-1, 1, 1))

                xi0 = np.reshape(alphas[xps, :] / etas[xbis, :], (-1, num_states, 1))*a_mat
                xi1 = np.reshape(betas[xcs, :] / p_level[level+1, :], (-1, 1, num_states))
                xi_loc = xi0*xi1

                xi_loc_sum = xi_loc.sum(axis=1, keepdims=True).sum(axis=2, keepdims=True)
                xi_loc_sum[xi_loc_sum == 0] = 1.0

                temp = xi_loc.sum(axis=1)
                temp_sum = temp.sum(axis=1, keepdims=True)
                temp_sum[temp_sum == 0] = 1.0
                temp /= temp_sum

                xi_loc *= weights_loc / xi_loc_sum

                self.trans_counts += xi_loc.sum(axis=0)
                alphas[xcs, :] += temp

            self.init_counts += np.sum(alphas[rns, :], axis=0)
            self.state_counts += alphas.sum(axis=0)

            for i in range(num_states):
                alphas[:, i] *= weights[len_enc[0]]
                self.accumulators[i].seq_update(enc_x, alphas[:, i], estimate.topics[i])

            if len_enc is not None:
                self.len_accumulator.seq_update(len_enc[1], weights[len_enc[0]], estimate.len_dist)

    def combine(self, suff_stat: Tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[SS0], Optional[SS1]]) \
            -> 'TreeHiddenMarkovAccumulator':
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
        len_val = self.len_accumulator.value()

        return self.num_states, self.init_counts, self.state_counts, self.trans_counts, tuple(
            [u.value() for u in self.accumulators]), len_val

    def from_value(self, x: Tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[SS0], Optional[SS1]]) \
            -> 'TreeHiddenMarkovAccumulator':
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

    def acc_to_encoder(self) -> 'TreeHiddenMarkovDataEncoder':
        emission_encoder = self.accumulators[0].acc_to_encoder()
        len_encoder = self.len_accumulator.acc_to_encoder()

        return TreeHiddenMarkovDataEncoder(emission_encoder=emission_encoder, len_encoder=len_encoder,
                                           use_numba=self.use_numba)


class TreeHiddenMarkovAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, factories: Sequence[StatisticAccumulatorFactory],
                 len_factory: StatisticAccumulatorFactory = NullAccumulatorFactory(),
                 keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (None, None, None),
                 name: Optional[str] = None,
                 use_numba: bool = True) -> None:
        self.factories = factories
        self.keys = keys if keys is None else (None, None, None)
        self.len_factory = len_factory
        self.name = name
        self.use_numba = use_numba

    def make(self) -> 'TreeHiddenMarkovAccumulator':
        len_acc = self.len_factory.make() if self.len_factory is not None else None
        return TreeHiddenMarkovAccumulator([self.factories[i].make() for i in range(len(self.factories))],
                                           len_accumulator=len_acc, keys=self.keys, name=self.name,
                                           use_numba=self.use_numba)


class TreeHiddenMarkovEstimator(ParameterEstimator):

    def __init__(self, estimators: List[ParameterEstimator],
                 len_estimator: Optional[ParameterEstimator] = NullEstimator(),
                 pseudo_count: Optional[Tuple[Optional[float], Optional[float]]] = (None, None),
                 name: Optional[str] = None,
                 keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (None, None, None),
                 use_numba: bool = True) -> None:

        self.num_states = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count if pseudo_count is not None else (None, None)
        self.keys = keys if keys is not None else (None, None, None)
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.name = name
        self.use_numba = use_numba

    def accumulator_factory(self) -> TreeHiddenMarkovAccumulatorFactory:
        est_factories = [u.accumulator_factory() for u in self.estimators]
        len_factory = self.len_estimator.accumulator_factory()
        return TreeHiddenMarkovAccumulatorFactory(factories=est_factories, len_factory=len_factory, keys=self.keys,
                                                  name=self.name, use_numba=self.use_numba)

    def estimate(self, nobs: Optional[float],
                 suff_stat: Tuple[int, np.ndarray, np.ndarray, np.ndarray, Sequence[SS0], Optional[SS1]]) \
            -> 'TreeHiddenMarkovModelDistribution':
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

        return TreeHiddenMarkovModelDistribution(topics=topics, w=w, transitions=transitions, len_dist=len_dist,
                                                 name=self.name, use_numba=self.use_numba)


class TreeHiddenMarkovDataEncoder(DataSequenceEncoder):

    def __init__(self, emission_encoder: DataSequenceEncoder,
                 len_encoder: Optional[DataSequenceEncoder] = NullDataEncoder(),
                 use_numba: bool = True) -> None:
        self.emission_encoder = emission_encoder
        self.len_encoder = len_encoder if len_encoder is not None else NullDataEncoder()
        self.use_numba = use_numba

    def __str__(self) -> str:
        s1 = repr(self.emission_encoder)
        s2 = repr(self.len_encoder)
        s3 = repr(self.use_numba)
        return 'TreeHiddenMarkovDataEncoder(emission_encoder=%s, len_encoder=%s, use_numba=%s)' % (s1, s2, s3)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TreeHiddenMarkovDataEncoder):
            if self.len_encoder == other.len_encoder:
                return True
        else:
            return False

    def _seq_encode(self, x: Sequence[Sequence[Tuple[D, T]]]) -> Tuple[int, np.ndarray, E5, E6, Any, Optional[Any]]:

        xs = []  # flattened values of nodes in order encoded
        obs_idx = [] #  tree seq idx for observed flattened nodes
        idx = []  # idx for node observation by tree in seq used in betas
        tz = [0] #  Track entries in beta by observation.
        #  Encodings for the beta pass
        xln = []  # leaf nodes
        xlnl = []  # levels for the leaf nodes
        xlni = []
        root_id = []

        xbi = []  # Use this to track beta_j(p(u), u)
        xp = []  # parents, repeated for each child
        xl = []  # level of xc below
        xc = []  # children of xp

        nc = []  # number of children for a given node.

        cnt = 0
        eta_cnt = 0
        for i, xx in enumerate(x):

            n = len(xx)
            tz.append(n)
            if n > 0:
                root_id.append(i)

            xi0 = np.asarray([v[0][0] for v in xx], dtype=np.int32)
            xp0 = np.asarray([v[0][1] for v in xx], dtype=np.int32)

            p_sort = np.argsort(xp0)

            xc0 = np.asarray([xx[i][0][0] for i in p_sort[1:]], dtype=np.int32)
            ## relabel entries to be 0,1,2,3,....,n-1
            xi0 = xi0[p_sort] + cnt
            xp0 = xp0[p_sort]

            xs.extend([xx[i][1] for i in p_sort])

            u0, u1 = np.unique(xp0[1:], return_counts=True)

            #  beta parent/child combos
            if len(u1) > 0:
                for j in range(len(u1)):
                    xp.extend([u0[j] + cnt] * u1[j])
                    xc.extend(cnt + xc0[np.flatnonzero(xp0[1:] == u0[j])])

            if len(xp0) > 1:
                xbi.extend([kk + eta_cnt for kk in range(len(xp0) - 1)])
                eta_cnt += len(xp0)-1

                xl_temp = find_level(xp0)
                xl.extend(xl_temp)
                xln_temp = np.delete(np.arange(n), u0)
                xlnl.extend([xl_temp[np.flatnonzero(xc0 == x)[0]] for x in xln_temp])
                xlni.extend([i]*len(xln_temp))
                xln.extend(xln_temp + cnt)
                idx.extend([i] * len(xl_temp))

            #  Length distribution
            nc_temp = np.zeros(n, dtype=np.int32)
            nc_temp[u0] = u1
            nc.extend(nc_temp)
            obs_idx.extend([i]*n)

            cnt += n

        idx = np.asarray(idx, dtype=np.int32)
        xbi = np.asarray(xbi, dtype=np.int32)
        xp = np.asarray(xp, dtype=np.int32)
        xc = np.asarray(xc, dtype=np.int32)
        xl = np.asarray(xl, dtype=np.int32)
        xln = np.asarray(xln, dtype=np.int32)
        xlnl = np.asarray(xlnl, dtype=np.int32)
        xlni = np.asarray(xlni, dtype=np.int32)
        root_idx = np.asarray(root_id, dtype=np.int32)

        level_idx = []
        eta_p = []
        p_nxt = []
        i_nxt = [root_idx]

        for level in range(1, np.max(xl)+1):

            level_idx.append(np.flatnonzero(xl == level))
            u0, u1 = np.unique(xp[level_idx[-1]], return_counts=True)
            eta_p.append(np.cumsum(np.append([0], u1)))
            p_nxt.append(u0)
            i_nxt.append(idx[level_idx[-1]])

        rns = np.unique(xp[level_idx[0]]) # root nodes

        enc_x = self.emission_encoder.seq_encode(xs)
        len_enc = self.len_encoder.seq_encode(nc)

        tz = np.cumsum(tz).astype(np.int32)
        obs_idx = np.asarray(obs_idx, dtype=np.int32)

        if len_enc is not None:
            return cnt, tz, (xln, xlnl, xlni), (idx, xbi, xp, xc, level_idx, p_nxt, eta_p, i_nxt, rns), enc_x, \
                   (obs_idx, len_enc)
        else:
            return cnt, tz, (xln, xlnl, xlni), (idx, xbi, xp, xc, level_idx, p_nxt, eta_p, i_nxt, rns), enc_x, None

    def seq_encode(self, x: Sequence[Sequence[Tuple[D, T]]]) -> Tuple[Optional[Tuple[np.ndarray, E1, E2, Any,
                                                                      Optional[Tuple[np.ndarray, Any]]]],
                                                                      Optional[Tuple[int, np.ndarray, E5, E6, Any,
                                                                                     Optional[Tuple[np.ndarray,
                                                                                                    Any]]]]]:
        if self.use_numba:
            xs = []  # flattened values of nodes in order encoded
            idx = []  # idx corresponding to weight
            tz = [0]  # slice entries for a given observed tree

            #  Encodings for the beta pass
            xln = []  # leaf nodes
            xlnl = []  # levels for the leaf nodes
            tlnz = [0]  # slice leaf nodes for given tree observation
            xbi = []  # Use this to track beta_j(p(u), u)
            xp = []  # parents, repeated for each child
            xl = []  # level of xc below
            xc = []  # children of xp
            txz = [0]  # slice xp, xc, and xl for observed tree
            tp = []  # partition couples of (p, c) for all of a parents children.
            tpz = [0]  # slice tp for an observed tree

            nc = []  # number of children for a given node.

            for i, xx in enumerate(x):

                n = len(xx)

                xi0 = np.asarray([v[0][0] for v in xx], dtype=np.int32)
                xp0 = np.asarray([v[0][1] for v in xx], dtype=np.int32)

                p_sort = np.argsort(xp0)

                xc0 = np.asarray([xx[i][0][0] for i in p_sort[1:]], dtype=np.int32)
                #  relabel entries to be 0,1,2,3,....,n-1
                xi0 = xi0[p_sort]
                xp0 = xp0[p_sort]
                xs.extend([xx[i][1] for i in p_sort])

                u0, u1 = np.unique(xp0[1:], return_counts=True)

                #  beta parent/child combos
                if len(u1) > 0:
                    for j in range(len(u1)):
                        xp.extend([u0[j]] * u1[j])
                        xc.extend(xc0[np.flatnonzero(xp0[1:] == u0[j])])

                    txz.append(np.sum(u1))
                    tp.extend(np.cumsum([0] + list(u1)))
                    tpz.append(len(u1) + 1)

                else:
                    txz.append(0)
                    tp.append(0)
                    tpz.append(1)

                if len(xp0) > 1:
                    xbi.extend([kk for kk in range(len(xp0) - 1)])

                    xl_temp = find_level(xp0)
                    xl.extend(xl_temp)
                    xln_temp = [yy for yy in np.delete(np.arange(n), u0)]
                    xlnl.extend([xl_temp[np.flatnonzero(xc0 == x)[0]] for x in xln_temp])
                    xln.extend(xln_temp)

                    tlnz.append(len(xln_temp))
                else:
                    tlnz.append(0)

                tz.append(n)

                #  Length distribution
                idx.extend([i] * n)

                nc_temp = np.zeros(n, dtype=np.int32)
                nc_temp[u0] = u1
                nc.extend(nc_temp)

            tz = np.cumsum(tz).astype(np.int32)

            xln = np.asarray(xln, dtype=np.int32)
            xlnl = np.asarray(xlnl, dtype=np.int32)
            tlnz = np.cumsum(tlnz).astype(np.int32)

            xbi = np.asarray(xbi, dtype=np.int32)
            xp = np.asarray(xp, dtype=np.int32)
            xc = np.asarray(xc, dtype=np.int32)
            xl = np.asarray(xl, dtype=np.int32)
            txz = np.cumsum(txz).astype(np.int32)
            tp = np.asarray(tp, dtype=np.int32)
            tpz = np.cumsum(tpz).astype(np.int32)

            enc_x = self.emission_encoder.seq_encode(xs)
            len_enc = self.len_encoder.seq_encode(nc)

            #if len_enc is not None:
            len_enc = tuple([np.asarray(idx, np.int32), len_enc])

            return (tz, (np.max(xln), xln, xlnl, tlnz), (xbi, xp, xc, xl, txz, tp, tpz), enc_x, len_enc), None

        else:
            return None, self._seq_encode(x)

@numba.njit(
    'void(int32, int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], '
    'int32[:], float64[:,:], float64[:, :], float64[:, :], float64[:], float64[:,:], float64[:,:], float64[:])',
    fastmath=True, parallel=True)
def numba_seq_log_density(num_states, tz, txz, tp, tpz, tlnz, xp, xc, xl, xbi, xln, xlnl, pr_obs, p_level, tr_mat,
                          pr_max0, betas, etas, out):
    for n in numba.prange(len(tz)-1):
        #  Observed value slice (xs)
        s0, s1 = tz[n], tz[n+1]

        if s0 == s1:
            out[n] = 0
            continue

        #  Slice the upward pass
        i0, i1 = txz[n], txz[n + 1]
        if i0 == i1:
            #  Only root node in tree
            beta_sum = 0
            for i in range(num_states):
                temp = pr_obs[s0, i]*p_level[0, i]
                beta_sum += temp
            out[n] = math.log(beta_sum) + pr_max0[s0]

        ll_sum = 0.0
        beta_mat = betas[s0:s1, :]
        eta_mat = etas[i0:i1, :]
        b = pr_obs[s0:s1, :]
        b_max = pr_max0[s0:s1]

        #  Start with the leaf nodes (non-parent-nodes).
        j0, j1 = tlnz[n], tlnz[n+1]
        xlns = xln[j0:j1]
        xlnls = xlnl[j0:j1]

        for k in range(len(xlns)):
            leaf_node = xlns[k]
            leaf_level = xlnls[k]
            beta_sum = 0
            for i in range(num_states):
                temp = b[leaf_node, i]*p_level[leaf_level, i]
                beta_mat[leaf_node, i] *= temp
                beta_sum += temp

            ll_sum += math.log(beta_sum) + b_max[leaf_node]

            for i in range(num_states):
                beta_mat[leaf_node, i] /= beta_sum

        #  Slice the upward pass
        xps = xp[i0:i1]
        xcs = xc[i0:i1]
        xls = xl[i0:i1]
        xbis = xbi[i0:i1]

        #  Partitions for the groupings on the betas
        tps = tp[tpz[n]:tpz[n+1]]

        for nn in range(len(tps)-2, -1, -1):
            t0, t1 = tps[nn], tps[nn+1]
            p, level = xps[t0], xls[t0]

            #  Get eta(p, u)_i and sum then get beta_i(p)
            beta_sum = 0
            for i in range(num_states):
                beta_mat[p, i] *= b[p, i]*p_level[level-1, i]

                for k in range(t0, t1):
                    c = xcs[k]
                    eta_idx = xbis[k]
                    eta_sum = 0

                    for j in range(num_states):
                        eta_sum += beta_mat[c, j] * tr_mat[i, j] / p_level[level,  j]

                    eta_mat[eta_idx, i] += eta_sum
                    beta_mat[p, i] *= eta_sum

                beta_sum += beta_mat[p, i]

            ll_sum += math.log(beta_sum) + b_max[p]

            for i in range(num_states):
                beta_mat[p, i] /= beta_sum

        out[n] = ll_sum


@numba.njit(
    'void(int32, int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], '
    'int32[:], float64[:,:], float64[:, :], float64[:, :], float64[:], float64[:,:], float64[:,:], float64[:,:], '
    'float64[:,:, :], float64[:,:])', parallel=True)
def numba_baum_welch(num_states, tz, txz, tp, tpz, tlnz, xp, xc, xl, xbi, xln, xlnl, pr_obs, p_level, tr_mat,
                     weights, betas, etas, alphas, xi_acc, pi_acc):
    for n in numba.prange(len(tz)-1):

        #  Observed value slice (xs)
        s0, s1 = tz[n], tz[n+1]
        weight_loc = weights[n]

        if s0 == s1:
            continue

        #  Slice the upward pass
        i0, i1 = txz[n], txz[n + 1]

        if i0 == i1:
            #  Only one node with no children, need to handle this. No transition updates just pi_acc
            alpha_sum = 0
            for i in range(num_states):
                temp = pr_obs[s0, i]*p_level[0, i]

                alphas[s0, i] = temp * weight_loc
                alpha_sum += temp

            for i in range(num_states):
                alphas[s0, i] /= alpha_sum
                pi_acc[n, i] += alphas[s0, i]

            continue

        beta_mat = betas[s0:s1, :]
        eta_mat = etas[i0:i1, :]
        b = pr_obs[s0:s1, :]

        #  Start with the leaf nodes (non-parent-nodes).
        j0, j1 = tlnz[n], tlnz[n+1]
        xlns = xln[j0:j1]
        xlnls = xlnl[j0:j1]

        for k in range(len(xlns)):
            leaf_node = xlns[k]
            leaf_level = xlnls[k]
            beta_sum = 0
            for i in range(num_states):
                temp = b[leaf_node, i]*p_level[leaf_level, i]
                beta_mat[leaf_node, i] = temp
                beta_sum += temp

            for i in range(num_states):
                beta_mat[leaf_node, i] /= beta_sum

        #  Slice the upward pass
        xps = xp[i0:i1]
        xcs = xc[i0:i1]
        xls = xl[i0:i1]
        xbis = xbi[i0:i1]

        #  Partitions for the groupings on the betas
        tps = tp[tpz[n]:tpz[n+1]]

        for nn in range(len(tps)-2, -1, -1):
            t0, t1 = tps[nn], tps[nn+1]
            p, level = xps[t0], xls[t0]

            #  Get eta(p, u)_i and sum then get beta_i(p)
            beta_sum = 0
            for i in range(num_states):
                beta_mat[p, i] = b[p, i]*p_level[level-1, i]

                for k in range(t0, t1):
                    c = xcs[k]
                    eta_idx = xbis[k]
                    eta_sum = 0

                    for j in range(num_states):
                        eta_sum += beta_mat[c, j] * tr_mat[i, j] / p_level[level,  j]

                    eta_mat[eta_idx, i] = eta_sum
                    beta_mat[p, i] *= eta_sum

                beta_sum += beta_mat[p, i]

            for i in range(num_states):
                beta_mat[p, i] /= beta_sum

        ### do the alpha pass
        alpha_mat = alphas[s0:s1, :]
        xi_buff = np.zeros((num_states, num_states), dtype=np.float64)

        #  set the root
        for i in range(num_states):
            alpha_mat[0, i] += beta_mat[0, i]*weight_loc

        for nn in range(0, len(tps)-1):
            t0, t1 = tps[nn], tps[nn+1]
            p, level = xps[t0], xls[t0]

            for k in range(t0, t1):
                c, eta_idx = xcs[k], xbis[k]
                xi_buff_sum = 0

                gamma_sum = 0
                for i in range(num_states):
                    alpha_sum = 0
                    for j in range(num_states):
                        temp = tr_mat[j, i] * alpha_mat[p, j] / eta_mat[eta_idx, j]
                        alpha_sum += temp

                        temp *= beta_mat[c, i]
                        temp /= p_level[level, i]

                        xi_buff_sum += temp
                        xi_buff[j, i] = temp

                    alpha_sum *= beta_mat[c, i]
                    alpha_sum /= p_level[level, i]

                    alpha_mat[c, i] += alpha_sum
                    gamma_sum += alpha_sum

                if gamma_sum > 0:
                    gamma_sum = weight_loc / gamma_sum
                if xi_buff_sum > 0:
                    xi_buff_sum = weight_loc / xi_buff_sum
                for i in range(num_states):
                    alpha_mat[c, i] *= gamma_sum
                    for j in range(num_states):
                        xi_acc[n, i, j] += xi_buff[i, j] * xi_buff_sum

        for i in range(num_states):
            pi_acc[n, i] += alpha_mat[0, i]


@numba.njit(
    'void(int32, int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], '
    'int32[:], float64[:,:], float64[:, :], float64[:,:], float64[:,:], float64[:,:])', fastmath=True, parallel=True)
def numba_posteriors(num_states, tz, txz, tp, tpz, tlnz, xp, xc, xl, xbi, xln, xlnl, pr_obs, p_level, tr_mat,
                     betas, etas):
    for n in numba.prange(len(tz)-1):

        #  Observed value slice (xs)
        s0, s1 = tz[n], tz[n+1]

        if s0 == s1:
            continue

        #  Slice the upward pass
        i0, i1 = txz[n], txz[n + 1]

        if i0 == i1:
            #  Only one node with no children, need to handle this. No transition updates just pi_acc
            beta_sum = 0
            for i in range(num_states):
                temp = pr_obs[s0, i]*p_level[0, i]

                betas[s0, i] += temp
                beta_sum += temp

            for i in range(num_states):
                betas[s0, i] /= beta_sum

        beta_mat = betas[s0:s1, :]
        eta_mat = etas[i0:i1, :]
        b = pr_obs[s0:s1, :]

        #  Start with the leaf nodes (non-parent-nodes).
        j0, j1 = tlnz[n], tlnz[n+1]
        xlns = xln[j0:j1]
        xlnls = xlnl[j0:j1]

        for k in range(len(xlns)):
            leaf_node = xlns[k]
            leaf_level = xlnls[k]
            beta_sum = 0
            for i in range(num_states):
                temp = b[leaf_node, i]*p_level[leaf_level, i]
                beta_mat[leaf_node, i] = temp
                beta_sum += temp

            for i in range(num_states):
                beta_mat[leaf_node, i] /= beta_sum

        #  Slice the upward pass
        xps = xp[i0:i1]
        xcs = xc[i0:i1]
        xls = xl[i0:i1]
        xbis = xbi[i0:i1]

        #  Partitions for the groupings on the betas
        tps = tp[tpz[n]:tpz[n+1]]

        for nn in range(len(tps)-2, -1, -1):
            t0, t1 = tps[nn], tps[nn+1]
            p, level = xps[t0], xls[t0]

            #  Get eta(p, u)_i and sum then get beta_i(p)
            beta_sum = 0
            for i in range(num_states):
                beta_mat[p, i] = b[p, i]*p_level[level-1, i]

                for k in range(t0, t1):
                    c = xcs[k]
                    eta_idx = xbis[k]
                    eta_sum = 0

                    for j in range(num_states):
                        eta_sum += beta_mat[c, j] * tr_mat[i, j] / p_level[level,  j]

                    eta_mat[eta_idx, i] = eta_sum
                    beta_mat[p, i] *= eta_sum

                beta_sum += beta_mat[p, i]

            for i in range(num_states):
                beta_mat[p, i] /= beta_sum


@numba.jit('void(int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int64[:], float64[:], float64[:], '
           'float64[:], float64[:,:])', parallel=True, nopython=True)
def numba_initialize(tz, txz, tp, tpz, xp, xc, states, weights, init_counts, state_counts, trans_counts):
    for n in numba.prange(len(tz) - 1):
        s0, s1 = tz[n], tz[n+1]

        if s0 == s1:
            continue

        weight_loc = weights[n]
        ss = states[s0:s1]
        init_counts[ss[0]] += weight_loc
        state_counts[ss[0]] += weight_loc

        i0, i1 = txz[n], txz[n+1]

        if i0 == i1:
            continue

        xps = xp[i0:i1]
        xcs = xc[i0:i1]
        tps = tp[tpz[n]:tpz[n + 1]]

        for nn in range(len(tps)-1):
            j0, j1 = tps[nn], tps[nn+1]
            p = ss[xps[j0]]
            for k in range(j0, j1):
                c = ss[xcs[k]]
                trans_counts[p, c] += weight_loc
                state_counts[c] += weight_loc


@numba.njit(
    'void(int32, int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], '
    'int32[:], float64[:,:], float64[:], float64[:,:], float64[:,:], float64[:,:], int32[:])', parallel=True)
def numba_viterbi(num_states, tz, txz, tp, tpz, tlnz, xp, xc, xl, xbi, xln, log_pr_obs, log_init_p, log_tr_mat,
                     betas, etas, out):
    for n in numba.prange(len(tz)-1):

        #  Observed value slice (xs)
        s0, s1 = tz[n], tz[n+1]

        if s0 == s1:
            continue

        #  Slice the upward pass
        i0, i1 = txz[n], txz[n + 1]
        outs = out[s0:s1]

        if i0 == i1:
            #  Only one node with no children, need to handle this. No transition updates just pi_acc
            beta_max = None
            beta_max_i = 0
            for i in range(num_states):
                temp = log_pr_obs[s0, i] + log_init_p[i]
                if beta_max is None:
                    beta_max = temp
                    beta_max_i = i
                else:
                    if beta_max < temp:
                        beta_max = temp
                        beta_max_i = i

            outs[0] = beta_max_i

        beta_mat = betas[s0:s1, :]
        eta_mat = etas[i0:i1, :]
        log_b = log_pr_obs[s0:s1, :]

        #  Start with the leaf nodes (non-parent-nodes).
        j0, j1 = tlnz[n], tlnz[n+1]
        xlns = xln[j0:j1]

        for k in range(len(xlns)):
            leaf_node = xlns[k]
            temp = log_b[leaf_node, 0]
            beta_mat[leaf_node, 0] += temp
            max_leaf_v = temp
            max_leaf_i = 0
            for i in range(1, num_states):
                temp = log_b[leaf_node, i]
                beta_mat[leaf_node, i] += temp

                if max_leaf_v < temp:
                    max_leaf_v = temp
                    max_leaf_i = i

            outs[leaf_node] = max_leaf_i

        #  Slice the upward pass
        xps = xp[i0:i1]
        xcs = xc[i0:i1]
        xls = xl[i0:i1]
        xbis = xbi[i0:i1]

        #  Partitions for the groupings on the betas
        tps = tp[tpz[n]:tpz[n+1]]

        for nn in range(len(tps)-2, -1, -1):
            t0, t1 = tps[nn], tps[nn+1]
            p, level = xps[t0], xls[t0]
            beta_max_v = None
            beta_max_i = None
            #  Get eta(p, u)_i and sum then get beta_i(p)
            for i in range(0, num_states):

                for k in range(t0, t1):
                    c = xcs[k]
                    eta_idx = xbis[k]
                    eta_max = beta_mat[c, 0] + log_tr_mat[i, 0]

                    for j in range(1, num_states):
                        temp = beta_mat[c, j] + log_tr_mat[i, j]
                        eta_max = max(eta_max, temp)

                    eta_mat[eta_idx, i] += eta_max
                    beta_mat[p, i] += log_b[p, i]
                    if beta_max_v is None:
                        beta_max_v = beta_mat[p, i]
                        beta_max_i = i
                    else:
                        if beta_max_v < beta_mat[p, i]:
                            beta_max_v = beta_mat[p, i]
                            beta_max_i = i

            outs[p] = beta_max_i


@numba.njit('float64[:](int32[:], float64[:], float64[:])', parallel=True)
def vec_bincount(idx, ll, out):
    for i in numba.prange(len(idx)):
        out[idx[i]] += ll[i]
    return out

@numba.njit('void(int32, int32, float64[:, :], float64[:], float64[:, :])')
def level_state_prob(levels, num_states, tr_mat, init_prob, out):

    for i in range(num_states):
        out[0, i] = init_prob[i]

    for k in range(1, levels):
        for i in range(num_states):
            for j in range(num_states):
                out[k, i] += out[k-1, i]*tr_mat[i, j]


