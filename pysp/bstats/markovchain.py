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
    """Returns the default (init_prior, row_priors) pair of unit-parameter
    Dirichlet distributions for a chain with num_states states.

    Args:
        num_states (int): Number of states S.

    Returns:
        Tuple (DirichletDistribution, list of S DirichletDistribution).

    """
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
    """First-order Markov chain over integer states with initial-state
    probabilities and a transition matrix, optionally carrying conjugate
    Dirichlet priors on each."""

    def __init__(self, init_prob_vec, transition_mat, name: Optional[str] = None,
                 prior=None, len_dist: ProbabilityDistribution = null_dist):
        """MarkovChainDistribution object.

        Args:
            init_prob_vec: Length-S vector of initial-state probabilities.
            transition_mat: (S, S) row-stochastic transition matrix.
            name (Optional[str]): Name of object.
            prior: (init_prior, row_priors) tuple or composable
                CompositeDistribution form (see set_prior()); defaults to
                unit-parameter Dirichlets.
            len_dist (ProbabilityDistribution): Distribution of the sequence
                length; null_dist treats lengths as exogenous.

        """
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
        """Returns the parameter tuple (init_prob_vec, transition_mat)."""
        return self.init_prob_vec, self.transition_mat

    def set_parameters(self, params) -> None:
        """Set the parameters and refresh the cached log-probabilities.

        Args:
            params: Tuple (init_prob_vec, transition_mat).

        """
        init_prob_vec, transition_mat = params

        with np.errstate(divide='ignore'):
            self.init_prob_vec = np.asarray(init_prob_vec, dtype=float)
            self.transition_mat = np.asarray(transition_mat, dtype=float)
            self.num_states = len(self.init_prob_vec)
            self.log_init = np.log(self.init_prob_vec)
            self.log_trans = np.log(self.transition_mat)

    def get_prior(self):
        """Returns the priors in composable form:
        CompositeDistribution((init_prior, CompositeDistribution(row_priors)))."""
        return CompositeDistribution((self.init_prior, CompositeDistribution(self.row_priors)))

    def set_prior(self, prior) -> None:
        """Set the priors and precompute conjugate-prior expectations.

        Accepts the (init_prior, row_priors) tuple form or the composable
        form returned by get_prior(). When the initial-state prior and all
        row priors are Dirichlet, this caches the digamma expectations
        E[ln p_k] = psi(alpha_k) - psi(sum alpha) for the initial and
        transition probabilities (used by expected_log_density) and sets
        has_conj_prior accordingly.

        Args:
            prior: (init_prior, row_priors) tuple or CompositeDistribution.

        """
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
        """Density of the Markov chain at sequence x; see log_density().

        Args:
            x: Sequence of integer states.

        Returns:
            Density at observation x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x) -> float:
        """Log-density of a state sequence (initial-state term plus
        transition terms, plus the len_dist term when present).

        Args:
            x: Sequence of integer states.

        Returns:
            Log-density at observation x.

        """
        return self._chain_log_density(x, self.log_init, self.log_trans) + self._len_term(x)

    def expected_log_density(self, x) -> float:
        """Variational expectation E_q[log p(x)] under the Dirichlet priors.

        Replaces the log-probabilities with their digamma expectations;
        falls back to the plug-in log_density(x) without a conjugate prior.

        Args:
            x: Sequence of integer states.

        Returns:
            Expected log-density at observation x.

        """
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
        """Encode sequences into flat initial-state and transition-pair arrays.

        Args:
            x (Sequence[Sequence[int]]): Iterable of state sequences.

        Returns:
            Tuple (init_states, pair_seq_idx, prev_states, next_states,
            lengths, len_enc) for use with seq_ methods.

        """
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
        """Vectorized log-density at sequence-encoded input x.

        Args:
            x: Encoded sequences from seq_encode().

        Returns:
            Numpy array of log-densities, one per sequence.

        """
        return self._seq_chain_log_density(x, self.log_init, self.log_trans)

    def seq_expected_log_density(self, x) -> np.ndarray:
        """Vectorized expected_log_density() at sequence-encoded input x.

        Args:
            x: Encoded sequences from seq_encode().

        Returns:
            Numpy array of expected log-densities, one per sequence.

        """
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
        """Create a MarkovChainSampler for this distribution.

        Args:
            seed (Optional[int]): Seed for the random number generator.

        Returns:
            MarkovChainSampler object.

        """
        return MarkovChainSampler(self, seed)

    def estimator(self):
        """Create a MarkovChainEstimator with this distribution's state count,
        name, priors, and length estimator.

        Returns:
            MarkovChainEstimator object.

        """
        len_est = NullEstimator() if isinstance(self.len_dist, NullDistribution) else self.len_dist.estimator()
        return MarkovChainEstimator(self.num_states, name=self.name,
                                    prior=(self.init_prior, self.row_priors), len_estimator=len_est)


class MarkovChainSampler(object):
    """Draws state sequences from a MarkovChainDistribution."""

    def __init__(self, dist: MarkovChainDistribution, seed: Optional[int] = None):
        """MarkovChainSampler object.

        Args:
            dist (MarkovChainDistribution): Distribution to sample from.
            seed (Optional[int]): Seed for the random number generator.

        """
        rng = RandomState(seed)
        self.rng = RandomState(rng.randint(0, 2**31 - 1))
        self.dist = dist
        if isinstance(dist.len_dist, NullDistribution) or dist.len_dist is None:
            self.len_sampler = None
        else:
            self.len_sampler = dist.len_dist.sampler(seed=rng.randint(0, 2**31 - 1))

    def sample_seq(self, n: Optional[int] = None) -> List[int]:
        """Draw a single state sequence.

        Args:
            n (Optional[int]): Sequence length; drawn from the len_dist
                sampler when None (which then must exist).

        Returns:
            List of n integer states.

        """
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
        """Draw size sequences (a single sequence when size is None).

        Args:
            size (Optional[int]): Number of sequences to draw.

        Returns:
            A state sequence if size is None, else a list of size sequences.

        """
        if size is None:
            return self.sample_seq()
        return [self.sample_seq() for _ in range(size)]


class MarkovChainAccumulator(StatisticAccumulator):
    """Accumulates Markov chain sufficient statistics (weighted initial-state
    counts and transition-pair counts, plus length statistics)."""

    def __init__(self, num_states: int, len_accumulator=NullAccumulator(), name=None, keys=None):
        """MarkovChainAccumulator object.

        Args:
            num_states (int): Number of states S.
            len_accumulator: Accumulator for the sequence lengths.
            name (Optional[str]): Name of the accumulator.
            keys (Optional[str]): Key for sharing sufficient statistics.

        """
        self.num_states = num_states
        self.name = name
        self.key = keys
        self.init_counts = np.zeros(num_states)
        self.trans_counts = np.zeros((num_states, num_states))
        self.len_accumulator = len_accumulator

    def initialize(self, x, weight, rng):
        """Initialize the accumulator with sequence x (delegates to update).

        Args:
            x: Sequence of integer states.
            weight (float): Weight of the observation.
            rng: Random number generator (unused).

        """
        self.update(x, weight, None)

    def seq_initialize(self, x, weights, rng):
        """Vectorized initialize() on sequence-encoded data (delegates to seq_update).

        Args:
            x: Encoded sequences.
            weights (np.ndarray): Weight per sequence.
            rng: Random number generator (unused).

        """
        self.seq_update(x, weights, None)

    def update(self, x, weight, estimate):
        """Accumulate the weighted state and transition counts of sequence x.

        Args:
            x: Sequence of integer states.
            weight (float): Weight of the observation.
            estimate: Current distribution estimate (unused).

        """
        if len(x) == 0:
            return
        self.init_counts[x[0]] += weight
        for i in range(1, len(x)):
            self.trans_counts[x[i - 1], x[i]] += weight
        if not isinstance(self.len_accumulator, NullAccumulator):
            self.len_accumulator.update(len(x), weight, None)

    def seq_update(self, x, weights, estimate):
        """Vectorized update() on sequence-encoded data.

        Args:
            x: Encoded sequences from seq_encode().
            weights (np.ndarray): Weight per sequence.
            estimate: Current distribution estimate (unused).

        """
        init_states, pair_seq_idx, prev_states, next_states, lengths, len_enc = x

        np.add.at(self.init_counts, init_states, weights)
        np.add.at(self.trans_counts, (prev_states, next_states), weights[pair_seq_idx])

        if len_enc is not None and not isinstance(self.len_accumulator, NullAccumulator):
            self.len_accumulator.seq_update(len_enc, weights, None)

    def combine(self, suff_stat):
        """Add another accumulator's sufficient-statistic value into this one.

        Args:
            suff_stat: Tuple as returned by value().

        Returns:
            This accumulator.

        """
        self.init_counts += suff_stat[0]
        self.trans_counts += suff_stat[1]
        if suff_stat[2] is not None and not isinstance(self.len_accumulator, NullAccumulator):
            self.len_accumulator.combine(suff_stat[2])
        return self

    def value(self):
        """Returns the sufficient statistics (init_counts, trans_counts, len_value)."""
        len_val = None if isinstance(self.len_accumulator, NullAccumulator) else self.len_accumulator.value()
        return self.init_counts, self.trans_counts, len_val

    def from_value(self, x):
        """Set the sufficient statistics from a value() tuple.

        Args:
            x: Tuple as returned by value().

        Returns:
            This accumulator.

        """
        self.init_counts = x[0]
        self.trans_counts = x[1]
        if x[2] is not None and not isinstance(self.len_accumulator, NullAccumulator):
            self.len_accumulator.from_value(x[2])
        return self

    def key_merge(self, stats_dict):
        """Merge this accumulator's keyed statistics into a shared dict.

        Args:
            stats_dict (dict): Shared key-to-statistics dictionary.

        """
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict):
        """Replace this accumulator's statistics with the pooled keyed values.

        Args:
            stats_dict (dict): Shared key-to-statistics dictionary.

        """
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())


class MarkovChainAccumulatorFactory(object):
    """Factory that creates MarkovChainAccumulator objects."""

    def __init__(self, num_states, len_factory, name, keys):
        """MarkovChainAccumulatorFactory object.

        Args:
            num_states (int): Number of states passed to created accumulators.
            len_factory: Factory for the length accumulators (None for none).
            name (Optional[str]): Name passed to created accumulators.
            keys (Optional[str]): Key passed to created accumulators.

        """
        self.num_states = num_states
        self.len_factory = len_factory
        self.name = name
        self.keys = keys

    def make(self):
        """Returns a new MarkovChainAccumulator."""
        len_acc = NullAccumulator() if self.len_factory is None else self.len_factory.make()
        return MarkovChainAccumulator(self.num_states, len_accumulator=len_acc, name=self.name, keys=self.keys)


class MarkovChainEstimator(ParameterEstimator):
    """Estimates a MarkovChainDistribution from sufficient statistics, using
    clamped Dirichlet MAP updates when the priors allow it."""

    def __init__(self, num_states: int, name: Optional[str] = None, keys: Optional[str] = None,
                 prior=None, len_estimator: ParameterEstimator = NullEstimator()):
        """MarkovChainEstimator object.

        Args:
            num_states (int): Number of states S.
            name (Optional[str]): Name of the estimated distribution.
            keys (Optional[str]): Key for sharing sufficient statistics.
            prior: (init_prior, row_priors) tuple or composable form;
                Dirichlets enable the conjugate update. Defaults to
                unit-parameter Dirichlets.
            len_estimator (ParameterEstimator): Estimator for the sequence
                lengths (NullEstimator treats lengths as exogenous).

        """
        self.num_states = int(num_states)
        self.name = name
        self.keys = keys
        self.len_estimator = len_estimator
        self.set_prior(prior if prior is not None else default_prior(self.num_states))

    def accumulator_factory(self):
        """Returns a MarkovChainAccumulatorFactory for this estimator."""
        len_factory = None if isinstance(self.len_estimator, NullEstimator) else self.len_estimator.accumulator_factory()
        return MarkovChainAccumulatorFactory(self.num_states, len_factory, self.name, self.keys)

    def get_prior(self):
        """Returns the priors in composable form:
        CompositeDistribution((init_prior, CompositeDistribution(row_priors)))."""
        return CompositeDistribution((self.init_prior, CompositeDistribution(self.row_priors)))

    def set_prior(self, prior):
        """Set the priors and flag whether they admit the conjugate update.

        Args:
            prior: (init_prior, row_priors) tuple or CompositeDistribution
                form; has_conj_prior is set when all priors are Dirichlet.

        """
        self.init_prior, self.row_priors, _ = _unpack_chain_prior(prior)
        self.has_conj_prior = isinstance(self.init_prior, DirichletDistribution) and \
            all(isinstance(u, DirichletDistribution) for u in self.row_priors)

    def model_log_density(self, model) -> float:
        """Log-density of the model's probabilities under the Dirichlet priors.

        Probabilities are floored at a tiny constant so MAP estimates that
        sit exactly on the simplex boundary score finitely.

        Args:
            model (MarkovChainDistribution): Model to score.

        Returns:
            Prior log-density of the model parameters (0 without a
            conjugate prior).

        """
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
        """Estimate a MarkovChainDistribution from sufficient statistics.

        With Dirichlet priors the initial-state and per-row transition
        probabilities are the clamped Dirichlet MAP (counts + alpha - 1,
        floored at zero and renormalized; posterior mean when the MAP is
        degenerate), and the posterior Dirichlets (counts + alpha) are
        carried as the new prior. Otherwise the maximum likelihood
        estimates are returned with uniform fallbacks for empty counts.

        Args:
            suff_stat: Tuple (init_counts, trans_counts, len_value) as
                returned by MarkovChainAccumulator.value().

        Returns:
            MarkovChainDistribution object.

        """
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
