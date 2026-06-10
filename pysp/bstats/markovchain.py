"""First-order Markov chain over integer states {0..S-1} with conjugate
Dirichlet priors: one Dirichlet on the initial-state probabilities and an
independent Dirichlet on each row of the transition matrix.

Observations are sequences of integers. Sequence lengths are treated as
exogenous unless a len_dist is supplied, in which case its log density of the
sequence length is added and its parameters are estimated alongside.

Estimation is MAP (counts + alpha - 1, clamped at the simplex boundary,
falling back to the posterior mean when degenerate), carrying the posterior
Dirichlets as the new prior. expected_log_density uses the standard digamma
expectations E[ln p_k] = psi(alpha_k) - psi(sum alpha).
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.random import RandomState
from scipy.special import digamma

from pysp.bstats.pdist import ProbabilityDistribution, StatisticAccumulator, ParameterEstimator
from pysp.bstats.dirichlet import DirichletDistribution
from pysp.bstats.composite import CompositeDistribution
from pysp.bstats.nulldist import null_dist, NullDistribution, NullEstimator, NullAccumulator


def default_prior(num_states: int):
    return (DirichletDistribution(np.ones(num_states)),
            [DirichletDistribution(np.ones(num_states)) for _ in range(num_states)])


def _map_probs(counts: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Dirichlet MAP with boundary clamp; posterior mean when degenerate."""
    num = np.maximum(counts + alpha - 1.0, 0.0)
    tot = num.sum()
    if tot > 0:
        return num/tot
    cpp = counts + alpha
    return cpp/cpp.sum()


def _unpack_chain_prior(prior):
    """Accept the (init_prior, row_priors) tuple form or the composable
    CompositeDistribution((init_prior, CompositeDistribution(row_priors), ...))
    form and return (init_prior, row_priors, extra_dists)."""
    if isinstance(prior, CompositeDistribution):
        init_prior = prior.dists[0]
        row_priors = list(prior.dists[1].dists)
        extra = list(prior.dists[2:])
        return init_prior, row_priors, extra
    init_prior, row_priors = prior[0], list(prior[1])
    extra = list(prior[2:]) if len(prior) > 2 else []
    return init_prior, row_priors, extra


class MarkovChainDistribution(ProbabilityDistribution):

    def __init__(self, init_prob_vec, transition_mat, name: Optional[str] = None,
                 prior=None, len_dist: ProbabilityDistribution = null_dist):

        self.name = name
        self.len_dist = len_dist
        self.set_parameters((init_prob_vec, transition_mat))
        self.set_prior(prior if prior is not None else default_prior(self.num_states))

    def __str__(self):
        pstr = ','.join(map(str, self.init_prob_vec.tolist()))
        tstr = ','.join(map(str, self.transition_mat.flatten().tolist()))
        return 'MarkovChainDistribution([%s], [%s], name=%s, len_dist=%s)' % (
            pstr, tstr, self.name, str(self.len_dist))

    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.init_prob_vec, self.transition_mat

    def set_parameters(self, params) -> None:
        init_prob_vec, transition_mat = params

        with np.errstate(divide='ignore'):
            self.init_prob_vec = np.asarray(init_prob_vec, dtype=float)
            self.transition_mat = np.asarray(transition_mat, dtype=float)
            self.num_states = len(self.init_prob_vec)
            self.log_init = np.log(self.init_prob_vec)
            self.log_trans = np.log(self.transition_mat)

    def get_prior(self):
        return CompositeDistribution((self.init_prior, CompositeDistribution(self.row_priors)))

    def set_prior(self, prior) -> None:
        self.init_prior, self.row_priors, _ = _unpack_chain_prior(prior)

        if isinstance(self.init_prior, DirichletDistribution) and \
                all(isinstance(u, DirichletDistribution) for u in self.row_priors):
            a0 = np.asarray(self.init_prior.get_parameters(), dtype=float)
            self.e_log_init = digamma(a0) - digamma(a0.sum())

            self.e_log_trans = np.zeros((self.num_states, self.num_states))
            for i, row_prior in enumerate(self.row_priors):
                ai = np.asarray(row_prior.get_parameters(), dtype=float)
                self.e_log_trans[i, :] = digamma(ai) - digamma(ai.sum())
            self.has_conj_prior = True
        else:
            self.e_log_init = None
            self.e_log_trans = None
            self.has_conj_prior = False

    def density(self, x) -> float:
        return np.exp(self.log_density(x))

    def log_density(self, x) -> float:
        return self._chain_log_density(x, self.log_init, self.log_trans) + self._len_term(x)

    def expected_log_density(self, x) -> float:
        if self.has_conj_prior:
            return self._chain_log_density(x, self.e_log_init, self.e_log_trans) + self._len_term(x)
        else:
            return self.log_density(x)

    def _len_term(self, x) -> float:
        if isinstance(self.len_dist, NullDistribution) or self.len_dist is None:
            return 0.0
        return self.len_dist.log_density(len(x))

    @staticmethod
    def _chain_log_density(x, log_init, log_trans) -> float:
        if len(x) == 0:
            return 0.0
        rv = log_init[x[0]]
        for i in range(1, len(x)):
            rv += log_trans[x[i - 1], x[i]]
        return float(rv)

    def seq_encode(self, x: Sequence[Sequence[int]]):
        lengths = np.asarray([len(u) for u in x], dtype=int)
        init_states = np.asarray([u[0] for u in x], dtype=int)

        pair_seq_idx = np.repeat(np.arange(len(x)), np.maximum(lengths - 1, 0))
        prev_states = np.concatenate([np.asarray(u[:-1], dtype=int) for u in x]) if len(x) > 0 else np.zeros(0, dtype=int)
        next_states = np.concatenate([np.asarray(u[1:], dtype=int) for u in x]) if len(x) > 0 else np.zeros(0, dtype=int)

        if isinstance(self.len_dist, NullDistribution) or self.len_dist is None:
            len_enc = None
        else:
            len_enc = self.len_dist.seq_encode(lengths)

        return init_states, pair_seq_idx, prev_states, next_states, lengths, len_enc

    def seq_log_density(self, x) -> np.ndarray:
        return self._seq_chain_log_density(x, self.log_init, self.log_trans)

    def seq_expected_log_density(self, x) -> np.ndarray:
        if self.has_conj_prior:
            return self._seq_chain_log_density(x, self.e_log_init, self.e_log_trans)
        else:
            return self.seq_log_density(x)

    def _seq_chain_log_density(self, x, log_init, log_trans) -> np.ndarray:
        init_states, pair_seq_idx, prev_states, next_states, lengths, len_enc = x

        rv = log_init[init_states].astype(float)
        np.add.at(rv, pair_seq_idx, log_trans[prev_states, next_states])

        if len_enc is not None:
            rv += self.len_dist.seq_log_density(len_enc)

        return rv

    def sampler(self, seed: Optional[int] = None):
        return MarkovChainSampler(self, seed)

    def estimator(self):
        len_est = NullEstimator() if isinstance(self.len_dist, NullDistribution) else self.len_dist.estimator()
        return MarkovChainEstimator(self.num_states, name=self.name,
                                    prior=(self.init_prior, self.row_priors), len_estimator=len_est)


class MarkovChainSampler(object):

    def __init__(self, dist: MarkovChainDistribution, seed: Optional[int] = None):
        rng = RandomState(seed)
        self.rng = RandomState(rng.randint(0, 2**31 - 1))
        self.dist = dist
        if isinstance(dist.len_dist, NullDistribution) or dist.len_dist is None:
            self.len_sampler = None
        else:
            self.len_sampler = dist.len_dist.sampler(seed=rng.randint(0, 2**31 - 1))

    def sample_seq(self, n: Optional[int] = None) -> List[int]:
        if n is None:
            if self.len_sampler is None:
                raise Exception('MarkovChainSampler requires a len_dist (or explicit n) to sample sequences.')
            n = int(self.len_sampler.sample())

        if n == 0:
            return []

        rv = [int(self.rng.choice(self.dist.num_states, p=self.dist.init_prob_vec))]
        for _ in range(n - 1):
            rv.append(int(self.rng.choice(self.dist.num_states, p=self.dist.transition_mat[rv[-1], :])))
        return rv

    def sample(self, size=None):
        if size is None:
            return self.sample_seq()
        return [self.sample_seq() for _ in range(size)]


class MarkovChainAccumulator(StatisticAccumulator):

    def __init__(self, num_states: int, len_accumulator=NullAccumulator(), name=None, keys=None):
        self.num_states = num_states
        self.name = name
        self.key = keys
        self.init_counts = np.zeros(num_states)
        self.trans_counts = np.zeros((num_states, num_states))
        self.len_accumulator = len_accumulator

    def initialize(self, x, weight, rng):
        self.update(x, weight, None)

    def seq_initialize(self, x, weights, rng):
        self.seq_update(x, weights, None)

    def update(self, x, weight, estimate):
        if len(x) == 0:
            return
        self.init_counts[x[0]] += weight
        for i in range(1, len(x)):
            self.trans_counts[x[i - 1], x[i]] += weight
        if not isinstance(self.len_accumulator, NullAccumulator):
            self.len_accumulator.update(len(x), weight, None)

    def seq_update(self, x, weights, estimate):
        init_states, pair_seq_idx, prev_states, next_states, lengths, len_enc = x

        np.add.at(self.init_counts, init_states, weights)
        np.add.at(self.trans_counts, (prev_states, next_states), weights[pair_seq_idx])

        if len_enc is not None and not isinstance(self.len_accumulator, NullAccumulator):
            self.len_accumulator.seq_update(len_enc, weights, None)

    def combine(self, suff_stat):
        self.init_counts += suff_stat[0]
        self.trans_counts += suff_stat[1]
        if suff_stat[2] is not None and not isinstance(self.len_accumulator, NullAccumulator):
            self.len_accumulator.combine(suff_stat[2])
        return self

    def value(self):
        len_val = None if isinstance(self.len_accumulator, NullAccumulator) else self.len_accumulator.value()
        return self.init_counts, self.trans_counts, len_val

    def from_value(self, x):
        self.init_counts = x[0]
        self.trans_counts = x[1]
        if x[2] is not None and not isinstance(self.len_accumulator, NullAccumulator):
            self.len_accumulator.from_value(x[2])
        return self

    def key_merge(self, stats_dict):
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict):
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())


class MarkovChainAccumulatorFactory(object):

    def __init__(self, num_states, len_factory, name, keys):
        self.num_states = num_states
        self.len_factory = len_factory
        self.name = name
        self.keys = keys

    def make(self):
        len_acc = NullAccumulator() if self.len_factory is None else self.len_factory.make()
        return MarkovChainAccumulator(self.num_states, len_accumulator=len_acc, name=self.name, keys=self.keys)


class MarkovChainEstimator(ParameterEstimator):

    def __init__(self, num_states: int, name: Optional[str] = None, keys: Optional[str] = None,
                 prior=None, len_estimator: ParameterEstimator = NullEstimator()):

        self.num_states = int(num_states)
        self.name = name
        self.keys = keys
        self.len_estimator = len_estimator
        self.set_prior(prior if prior is not None else default_prior(self.num_states))

    def accumulator_factory(self):
        len_factory = None if isinstance(self.len_estimator, NullEstimator) else self.len_estimator.accumulator_factory()
        return MarkovChainAccumulatorFactory(self.num_states, len_factory, self.name, self.keys)

    def get_prior(self):
        return CompositeDistribution((self.init_prior, CompositeDistribution(self.row_priors)))

    def set_prior(self, prior):
        self.init_prior, self.row_priors, _ = _unpack_chain_prior(prior)
        self.has_conj_prior = isinstance(self.init_prior, DirichletDistribution) and \
            all(isinstance(u, DirichletDistribution) for u in self.row_priors)

    def model_log_density(self, model) -> float:
        if not self.has_conj_prior:
            return 0.0
        # floor at tiny to avoid 0*log(0) = nan when a MAP probability sits on
        # the simplex boundary with a unit prior count
        tiny = 1.0e-300
        rv = float(self.init_prior.log_density(np.maximum(model.init_prob_vec, tiny)))
        for i, row_prior in enumerate(self.row_priors):
            rv += float(row_prior.log_density(np.maximum(model.transition_mat[i, :], tiny)))
        return rv

    def estimate(self, suff_stat) -> MarkovChainDistribution:

        init_counts, trans_counts, len_val = suff_stat
        s = self.num_states

        if isinstance(self.len_estimator, NullEstimator) or len_val is None:
            len_dist = null_dist
        else:
            len_dist = self.len_estimator.estimate(len_val)

        if self.has_conj_prior:

            a0 = np.asarray(self.init_prior.get_parameters(), dtype=float)
            init_probs = _map_probs(init_counts, a0)
            init_posterior = DirichletDistribution(init_counts + a0)

            trans_mat = np.zeros((s, s))
            row_posteriors = []
            for i in range(s):
                ai = np.asarray(self.row_priors[i].get_parameters(), dtype=float)
                trans_mat[i, :] = _map_probs(trans_counts[i, :], ai)
                row_posteriors.append(DirichletDistribution(trans_counts[i, :] + ai))

            return MarkovChainDistribution(init_probs, trans_mat, name=self.name,
                                           prior=(init_posterior, row_posteriors), len_dist=len_dist)

        else:

            init_probs = init_counts/init_counts.sum() if init_counts.sum() > 0 else np.ones(s)/s
            row_sums = trans_counts.sum(axis=1, keepdims=True)
            trans_mat = np.where(row_sums > 0, trans_counts/np.maximum(row_sums, 1.0), 1.0/s)

            return MarkovChainDistribution(init_probs, trans_mat, name=self.name, len_dist=len_dist)
