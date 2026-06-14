"""Hidden Markov model with conjugate Dirichlet priors.

States are latent; emissions come from arbitrary pysp.bstats component
distributions ("topics"), each carrying its own prior. The initial-state
probabilities have a Dirichlet prior and each transition-matrix row has an
independent Dirichlet prior.

Estimation is MAP-EM: the E-step runs a scaled forward-backward recursion at
the current point estimates, and the M-step applies the clamped Dirichlet MAP
updates (falling back to the posterior mean on the simplex boundary) for the
initial/transition probabilities and each topic's conjugate update for the
emissions. Combined with bestimation.optimize this maximizes the penalized
log-likelihood log p(data | theta) + log p(theta), which is monotone over
iterations.

Sequence lengths are exogenous unless len_dist is given (then its log density
is added and its parameters are estimated alongside).
"""

from collections.abc import Sequence

import numpy as np
from numpy.random import RandomState
from scipy.special import digamma

from pysp.arithmetic import maxint
from pysp.bstats.composite import CompositeDistribution
from pysp.bstats.dirichlet import DirichletDistribution
from pysp.bstats.markov_chain import _map_probs, _unpack_chain_prior, default_prior
from pysp.bstats.nulldist import NullAccumulator, NullDistribution, NullEstimator, null_dist
from pysp.bstats.pdist import ParameterEstimator, ProbabilityDistribution, StatisticAccumulator


class HiddenMarkovModelDistribution(ProbabilityDistribution):
    """Hidden Markov model with latent integer states, emission ("topic")
    distributions, and conjugate Dirichlet priors on the initial-state and
    transition probabilities."""

    def __init__(
        self,
        topics,
        w,
        transitions,
        name: str | None = None,
        prior=None,
        len_dist: ProbabilityDistribution = null_dist,
    ):
        """HiddenMarkovModelDistribution object.

        Args:
            topics: List of S emission distributions, one per state (each
                carrying its own prior).
            w: Length-S vector of initial-state probabilities.
            transitions: (S, S) row-stochastic transition matrix.
            name (Optional[str]): Name of object.
            prior: (init_prior, row_priors) tuple or composable
                CompositeDistribution form (see set_prior()); defaults to
                unit-parameter Dirichlets.
            len_dist (ProbabilityDistribution): Distribution of the sequence
                length; null_dist treats lengths as exogenous.

        """
        self.name = name
        self.topics = topics
        self.num_states = len(topics)
        self.len_dist = len_dist
        self.set_parameters((np.asarray(w, dtype=float), np.asarray(transitions, dtype=float)))
        self.set_prior(prior if prior is not None else default_prior(self.num_states))

    def __str__(self):
        tstr = ",".join(str(u) for u in self.topics)
        wstr = ",".join(map(str, self.w.tolist()))
        astr = ",".join(map(str, self.transitions.flatten().tolist()))
        return "HiddenMarkovModelDistribution([%s], [%s], [%s], name=%s, len_dist=%s)" % (
            tstr,
            wstr,
            astr,
            self.name,
            str(self.len_dist),
        )

    def get_parameters(self):
        """Returns the parameter tuple (w, transitions, topic parameters)."""
        return self.w, self.transitions, [u.get_parameters() for u in self.topics]

    def set_parameters(self, params) -> None:
        """Set the parameters and refresh the cached log-probabilities.

        Args:
            params: Tuple (w, transitions) or (w, transitions,
                topic parameters); topic parameters, when present, are
                pushed down into the topics.

        """
        if len(params) == 3:
            w, transitions, topic_params = params
            for topic, tp in zip(self.topics, topic_params):
                topic.set_parameters(tp)
        else:
            w, transitions = params

        with np.errstate(divide="ignore"):
            self.w = np.asarray(w, dtype=float)
            self.transitions = np.asarray(transitions, dtype=float)
            self.log_w = np.log(self.w)
            self.log_trans = np.log(self.transitions)

    def get_prior(self):
        """Returns the priors in composable form: CompositeDistribution of
        (init_prior, row priors, topic priors)."""
        # composable form: nesting machinery (DPM/mixtures) can take
        # cross-entropies and entropies of this against another HMM prior
        return CompositeDistribution(
            (
                self.init_prior,
                CompositeDistribution(self.row_priors),
                CompositeDistribution([t.get_prior() for t in self.topics]),
            )
        )

    def set_prior(self, prior) -> None:
        """Set the priors and precompute conjugate-prior expectations.

        Accepts the (init_prior, row_priors) tuple form or the composable
        form returned by get_prior() (whose topic priors, when present, are
        pushed down into the topics). When the initial-state prior and all
        row priors are Dirichlet, this caches the digamma expectations
        E[ln p_k] = psi(alpha_k) - psi(sum alpha) used by
        expected_log_density and sets has_conj_prior accordingly.

        Args:
            prior: (init_prior, row_priors) tuple or CompositeDistribution.

        """
        self.init_prior, self.row_priors, extra = _unpack_chain_prior(prior)

        if len(extra) > 0:
            for topic, tp in zip(self.topics, extra[0].dists):
                topic.set_prior(tp)

        if isinstance(self.init_prior, DirichletDistribution) and all(
            isinstance(u, DirichletDistribution) for u in self.row_priors
        ):
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
        """Density of the HMM at observation sequence x; see log_density().

        Args:
            x: Sequence of emissions accepted by the topics.

        Returns:
            Density at observation x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x) -> float:
        """Marginal log-likelihood of an emission sequence via the scaled
        forward recursion (plus the len_dist term when present).

        Args:
            x: Sequence of emissions accepted by the topics.

        Returns:
            Log-density at observation x.

        """
        if len(x) == 0:
            return self._len_term(x)
        log_b = np.asarray([[topic.log_density(u) for topic in self.topics] for u in x])
        return self._forward_ll(log_b, self.log_w, self.log_trans) + self._len_term(x)

    def expected_log_density(self, x) -> float:
        """Forward log-likelihood with digamma-expected initial/transition
        log-probabilities and the topics' expected_log_density emissions.

        Falls back to the plug-in log_density(x) without a conjugate prior.

        Args:
            x: Sequence of emissions accepted by the topics.

        Returns:
            Expected log-density at observation x.

        """
        if not self.has_conj_prior:
            return self.log_density(x)
        if len(x) == 0:
            return self._len_term(x)
        log_b = np.asarray([[topic.expected_log_density(u) for topic in self.topics] for u in x])
        return self._forward_ll(log_b, self.e_log_init, self.e_log_trans) + self._len_term(x)

    def _len_term(self, x) -> float:
        if isinstance(self.len_dist, NullDistribution) or self.len_dist is None:
            return 0.0
        return self.len_dist.log_density(len(x))

    @staticmethod
    def _forward_ll(log_b: np.ndarray, log_init: np.ndarray, log_trans: np.ndarray) -> float:
        """Scaled forward recursion; returns the sequence log-likelihood."""
        b_max = log_b.max(axis=1, keepdims=True)
        b = np.exp(log_b - b_max)
        a_mat = np.exp(log_trans)

        alpha = np.exp(log_init) * b[0, :]
        c = alpha.sum()
        ll = np.log(c) if c > 0 else -np.inf
        alpha = alpha / c if c > 0 else alpha

        for t in range(1, log_b.shape[0]):
            alpha = np.dot(alpha, a_mat) * b[t, :]
            c = alpha.sum()
            if c <= 0:
                return -np.inf
            ll += np.log(c)
            alpha /= c

        return float(ll + b_max.sum())

    def seq_encode(self, x: Sequence[Sequence]):
        """Encode emission sequences into a flat topic encoding with offsets.

        Args:
            x (Sequence[Sequence]): Iterable of emission sequences.

        Returns:
            Tuple (lengths, offsets, flat_enc, len_enc) for use with
            seq_ methods.

        """
        lengths = np.asarray([len(u) for u in x], dtype=int)
        offsets = np.concatenate([[0], np.cumsum(lengths)])

        flat = []
        for u in x:
            flat.extend(u)
        flat_enc = self.topics[0].seq_encode(flat)

        if isinstance(self.len_dist, NullDistribution) or self.len_dist is None:
            len_enc = None
        else:
            len_enc = self.len_dist.seq_encode(lengths)

        return lengths, offsets, flat_enc, len_enc

    def _emission_log_densities(self, flat_enc) -> np.ndarray:
        return np.asarray([topic.seq_log_density(flat_enc) for topic in self.topics]).T

    def seq_log_density(self, x) -> np.ndarray:
        """Vectorized log-density at sequence-encoded input x.

        Args:
            x: Encoded sequences from seq_encode().

        Returns:
            Numpy array of log-densities, one per sequence.

        """
        lengths, offsets, flat_enc, len_enc = x
        log_b_all = self._emission_log_densities(flat_enc)

        rv = np.zeros(len(lengths))
        for i in range(len(lengths)):
            if lengths[i] == 0:
                continue
            log_b = log_b_all[offsets[i] : offsets[i + 1], :]
            rv[i] = self._forward_ll(log_b, self.log_w, self.log_trans)

        if len_enc is not None:
            rv += self.len_dist.seq_log_density(len_enc)

        return rv

    def seq_expected_log_density(self, x) -> np.ndarray:
        """Vectorized expected_log_density() at sequence-encoded input x.

        Args:
            x: Encoded sequences from seq_encode().

        Returns:
            Numpy array of expected log-densities, one per sequence.

        """
        if not self.has_conj_prior:
            return self.seq_log_density(x)

        lengths, offsets, flat_enc, len_enc = x
        log_b_all = np.asarray([topic.seq_expected_log_density(flat_enc) for topic in self.topics]).T

        rv = np.zeros(len(lengths))
        for i in range(len(lengths)):
            if lengths[i] == 0:
                continue
            log_b = log_b_all[offsets[i] : offsets[i + 1], :]
            rv[i] = self._forward_ll(log_b, self.e_log_init, self.e_log_trans)

        if len_enc is not None:
            rv += self.len_dist.seq_log_density(len_enc)

        return rv

    def viterbi(self, x) -> list[int]:
        """Most likely state sequence for a single observation sequence.

        Args:
            x: Sequence of emissions accepted by the topics.

        Returns:
            List of len(x) integer states maximizing the joint probability.

        """
        if len(x) == 0:
            return []
        log_b = np.asarray([[topic.log_density(u) for topic in self.topics] for u in x])

        delta = self.log_w + log_b[0, :]
        back = np.zeros((len(x), self.num_states), dtype=int)

        for t in range(1, len(x)):
            cand = delta[:, None] + self.log_trans
            back[t, :] = np.argmax(cand, axis=0)
            delta = cand[back[t, :], np.arange(self.num_states)] + log_b[t, :]

        states = [int(np.argmax(delta))]
        for t in range(len(x) - 1, 0, -1):
            states.append(int(back[t, states[-1]]))
        return states[::-1]

    def sampler(self, seed: int | None = None):
        """Create a HiddenMarkovModelSampler for this distribution.

        Args:
            seed (Optional[int]): Seed for the random number generator.

        Returns:
            HiddenMarkovModelSampler object.

        """
        return HiddenMarkovModelSampler(self, seed)

    def estimator(self):
        """Create a HiddenMarkovModelEstimator from this distribution's
        topics, priors, and length estimator.

        Returns:
            HiddenMarkovModelEstimator object.

        """
        len_est = NullEstimator() if isinstance(self.len_dist, NullDistribution) else self.len_dist.estimator()
        return HiddenMarkovModelEstimator(
            [u.estimator() for u in self.topics],
            name=self.name,
            prior=(self.init_prior, self.row_priors),
            len_estimator=len_est,
        )


class HiddenMarkovModelSampler:
    """Draws emission sequences from a HiddenMarkovModelDistribution."""

    def __init__(self, dist: HiddenMarkovModelDistribution, seed: int | None = None):
        """HiddenMarkovModelSampler object.

        Args:
            dist (HiddenMarkovModelDistribution): Distribution to sample from.
            seed (Optional[int]): Seed for the random number generator.

        """
        rng = RandomState(seed)
        self.rng = RandomState(rng.randint(0, maxint))
        self.dist = dist
        self.topic_samplers = [u.sampler(seed=rng.randint(0, maxint)) for u in dist.topics]
        if isinstance(dist.len_dist, NullDistribution) or dist.len_dist is None:
            self.len_sampler = None
        else:
            self.len_sampler = dist.len_dist.sampler(seed=rng.randint(0, maxint))

    def sample_seq(self, n: int | None = None):
        """Draw a single emission sequence (states are latent).

        Args:
            n (Optional[int]): Sequence length; drawn from the len_dist
                sampler when None (which then must exist).

        Returns:
            List of n emissions.

        """
        if n is None:
            if self.len_sampler is None:
                raise Exception("HiddenMarkovModelSampler requires a len_dist (or explicit n) to sample sequences.")
            n = int(self.len_sampler.sample())

        if n == 0:
            return []

        state = self.rng.choice(self.dist.num_states, p=self.dist.w)
        rv = [self.topic_samplers[state].sample()]
        for _ in range(n - 1):
            state = self.rng.choice(self.dist.num_states, p=self.dist.transitions[state, :])
            rv.append(self.topic_samplers[state].sample())
        return rv

    def sample(self, size=None):
        """Draw size sequences (a single sequence when size is None).

        Args:
            size (Optional[int]): Number of sequences to draw.

        Returns:
            An emission sequence if size is None, else a list of size sequences.

        """
        if size is None:
            return self.sample_seq()
        return [self.sample_seq() for _ in range(size)]


class HiddenMarkovModelAccumulator(StatisticAccumulator):
    """Accumulates HMM sufficient statistics via forward-backward: expected
    initial-state counts, expected transition counts, and posterior-weighted
    emission statistics per topic."""

    def __init__(self, accumulators, len_accumulator=NullAccumulator(), name=None, keys=None):
        """HiddenMarkovModelAccumulator object.

        Args:
            accumulators: List of S topic accumulators.
            len_accumulator: Accumulator for the sequence lengths.
            name (Optional[str]): Name of the accumulator.
            keys (Optional[str]): Key for sharing sufficient statistics.

        """
        self.accumulators = accumulators
        self.num_states = len(accumulators)
        self.name = name
        self.key = keys
        self.init_counts = np.zeros(self.num_states)
        self.trans_counts = np.zeros((self.num_states, self.num_states))
        self.len_accumulator = len_accumulator

    def initialize(self, x, weight, rng):
        """Initialize with random Dirichlet state assignments for sequence x.

        Args:
            x: Sequence of emissions accepted by the topics.
            weight (float): Weight of the observation.
            rng (RandomState): Random number generator for the assignments.

        """
        if len(x) == 0:
            return

        # random state assignments with Markov-consistent pair counts
        prev = None
        for u in x:
            p = rng.dirichlet(np.ones(self.num_states))
            if prev is None:
                self.init_counts += p * weight
            else:
                self.trans_counts += np.outer(prev, p) * weight
            for k in range(self.num_states):
                self.accumulators[k].initialize(u, p[k] * weight, rng)
            prev = p

        if not isinstance(self.len_accumulator, NullAccumulator):
            self.len_accumulator.update(len(x), weight, None)

    def update(self, x, weight, estimate):
        """Accumulate the E-step statistics for one sequence (delegates to
        seq_update on a singleton encoding).

        Args:
            x: Sequence of emissions accepted by the topics.
            weight (float): Weight of the observation.
            estimate (HiddenMarkovModelDistribution): Current estimate used
                for the forward-backward recursion.

        """
        if len(x) == 0:
            return
        enc = estimate.seq_encode([x])
        self.seq_update(enc, np.asarray([weight]), estimate)

    def seq_update(self, x, weights, estimate):
        """Forward-backward E-step on sequence-encoded data.

        Runs the scaled forward-backward recursion per sequence at the
        current point estimates, accumulating expected initial-state counts,
        expected transition counts (xi), and pushing the state posteriors
        (gamma) into the topic accumulators.

        Args:
            x: Encoded sequences from seq_encode().
            weights (np.ndarray): Weight per sequence.
            estimate (HiddenMarkovModelDistribution): Current estimate used
                for the recursion.

        """
        lengths, offsets, flat_enc, len_enc = x

        log_b_all = estimate._emission_log_densities(flat_enc)
        a_mat = estimate.transitions
        pi = estimate.w

        gammas = np.zeros_like(log_b_all)

        for i in range(len(lengths)):
            n = lengths[i]
            if n == 0:
                continue

            log_b = log_b_all[offsets[i] : offsets[i + 1], :]
            b_max = log_b.max(axis=1, keepdims=True)
            b = np.exp(log_b - b_max)

            # scaled forward pass
            alphas = np.zeros((n, self.num_states))
            scale = np.zeros(n)
            alphas[0, :] = pi * b[0, :]
            scale[0] = alphas[0, :].sum()
            if scale[0] <= 0:
                scale[0] = 1.0
            alphas[0, :] /= scale[0]

            for t in range(1, n):
                alphas[t, :] = np.dot(alphas[t - 1, :], a_mat) * b[t, :]
                scale[t] = alphas[t, :].sum()
                if scale[t] <= 0:
                    scale[t] = 1.0
                alphas[t, :] /= scale[t]

            # scaled backward pass with posterior accumulation
            w_i = weights[i]
            beta = np.ones(self.num_states)
            gam = alphas[n - 1, :] * beta
            gammas[offsets[i] + n - 1, :] = gam * w_i

            for t in range(n - 2, -1, -1):
                bb = b[t + 1, :] * beta
                xi = (alphas[t, :][:, None] * a_mat * bb[None, :]) / scale[t + 1]
                self.trans_counts += xi * w_i

                beta = np.dot(a_mat, bb) / scale[t + 1]
                gam = alphas[t, :] * beta
                gam_sum = gam.sum()
                if gam_sum > 0:
                    gam /= gam_sum
                gammas[offsets[i] + t, :] = gam * w_i

            self.init_counts += gammas[offsets[i], :]

        for k in range(self.num_states):
            self.accumulators[k].seq_update(flat_enc, gammas[:, k], estimate.topics[k])

        if len_enc is not None and not isinstance(self.len_accumulator, NullAccumulator):
            self.len_accumulator.seq_update(len_enc, weights, None)

    def seq_initialize(self, x, weights, rng):
        """Vectorized initialize() with random Dirichlet state assignments.

        Args:
            x: Encoded sequences from seq_encode().
            weights (np.ndarray): Weight per sequence.
            rng (RandomState): Random number generator for the assignments.

        """
        lengths, offsets, flat_enc, len_enc = x
        tot = int(lengths.sum())

        gammas = rng.dirichlet(np.ones(self.num_states), size=tot)

        for i in range(len(lengths)):
            n = lengths[i]
            if n == 0:
                continue
            w_i = weights[i]
            g = gammas[offsets[i] : offsets[i + 1], :]
            self.init_counts += g[0, :] * w_i
            for t in range(1, n):
                self.trans_counts += np.outer(g[t - 1, :], g[t, :]) * w_i

        seq_w = np.repeat(weights, lengths)
        for k in range(self.num_states):
            self.accumulators[k].seq_initialize(flat_enc, gammas[:, k] * seq_w, rng)

        if len_enc is not None and not isinstance(self.len_accumulator, NullAccumulator):
            self.len_accumulator.seq_initialize(len_enc, weights, rng)

    def combine(self, suff_stat):
        """Add another accumulator's sufficient-statistic value into this one.

        Args:
            suff_stat: Tuple as returned by value().

        Returns:
            This accumulator.

        """
        self.init_counts += suff_stat[0]
        self.trans_counts += suff_stat[1]
        for k in range(self.num_states):
            self.accumulators[k].combine(suff_stat[2][k])
        if suff_stat[3] is not None and not isinstance(self.len_accumulator, NullAccumulator):
            self.len_accumulator.combine(suff_stat[3])
        return self

    def value(self):
        """Returns (init_counts, trans_counts, topic values, len_value)."""
        len_val = None if isinstance(self.len_accumulator, NullAccumulator) else self.len_accumulator.value()
        return self.init_counts, self.trans_counts, tuple(u.value() for u in self.accumulators), len_val

    def from_value(self, x):
        """Set the sufficient statistics from a value() tuple.

        Args:
            x: Tuple as returned by value().

        Returns:
            This accumulator.

        """
        self.init_counts = x[0]
        self.trans_counts = x[1]
        for k in range(self.num_states):
            self.accumulators[k].from_value(x[2][k])
        if x[3] is not None and not isinstance(self.len_accumulator, NullAccumulator):
            self.len_accumulator.from_value(x[3])
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
        for u in self.accumulators:
            u.key_merge(stats_dict)

    def key_replace(self, stats_dict):
        """Replace this accumulator's statistics with the pooled keyed values.

        Args:
            stats_dict (dict): Shared key-to-statistics dictionary.

        """
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())
        for u in self.accumulators:
            u.key_replace(stats_dict)


class HiddenMarkovModelAccumulatorFactory:
    """Factory that creates HiddenMarkovModelAccumulator objects."""

    def __init__(self, factories, len_factory, name, keys):
        """HiddenMarkovModelAccumulatorFactory object.

        Args:
            factories: List of S topic accumulator factories.
            len_factory: Factory for the length accumulators (None for none).
            name (Optional[str]): Name passed to created accumulators.
            keys (Optional[str]): Key passed to created accumulators.

        """
        self.factories = factories
        self.len_factory = len_factory
        self.name = name
        self.keys = keys

    def make(self):
        """Returns a new HiddenMarkovModelAccumulator."""
        len_acc = NullAccumulator() if self.len_factory is None else self.len_factory.make()
        return HiddenMarkovModelAccumulator(
            [f.make() for f in self.factories], len_accumulator=len_acc, name=self.name, keys=self.keys
        )


class HiddenMarkovModelEstimator(ParameterEstimator):
    """Estimates a HiddenMarkovModelDistribution by MAP-EM, using clamped
    Dirichlet MAP updates for the chain and each topic's conjugate update
    for the emissions."""

    def __init__(
        self,
        estimators,
        name: str | None = None,
        keys: str | None = None,
        prior=None,
        len_estimator: ParameterEstimator = NullEstimator(),
    ):
        """HiddenMarkovModelEstimator object.

        Args:
            estimators: List of S topic estimators.
            name (Optional[str]): Name of the estimated distribution.
            keys (Optional[str]): Key for sharing sufficient statistics.
            prior: (init_prior, row_priors) tuple or composable form (with
                optional topic priors pushed into the topic estimators);
                Dirichlets enable the conjugate update. Defaults to
                unit-parameter Dirichlets.
            len_estimator (ParameterEstimator): Estimator for the sequence
                lengths (NullEstimator treats lengths as exogenous).

        """
        self.estimators = estimators
        self.num_states = len(estimators)
        self.name = name
        self.keys = keys
        self.len_estimator = len_estimator
        self.set_prior(prior if prior is not None else default_prior(self.num_states))

    def accumulator_factory(self):
        """Returns a HiddenMarkovModelAccumulatorFactory for this estimator."""
        len_factory = (
            None if isinstance(self.len_estimator, NullEstimator) else self.len_estimator.accumulator_factory()
        )
        return HiddenMarkovModelAccumulatorFactory(
            [u.accumulator_factory() for u in self.estimators], len_factory, self.name, self.keys
        )

    def get_prior(self):
        """Returns the priors in composable form: CompositeDistribution of
        (init_prior, row priors, topic priors)."""
        return CompositeDistribution(
            (
                self.init_prior,
                CompositeDistribution(self.row_priors),
                CompositeDistribution([e.get_prior() for e in self.estimators]),
            )
        )

    def set_prior(self, prior):
        """Set the priors and flag whether they admit the conjugate update.

        Topic priors, when present in the composable form, are pushed down
        into the topic estimators.

        Args:
            prior: (init_prior, row_priors) tuple or CompositeDistribution
                form; has_conj_prior is set when the chain priors are all
                Dirichlet.

        """
        self.init_prior, self.row_priors, extra = _unpack_chain_prior(prior)

        if len(extra) > 0:
            for est, p in zip(self.estimators, extra[0].dists):
                est.set_prior(p)

        self.has_conj_prior = isinstance(self.init_prior, DirichletDistribution) and all(
            isinstance(u, DirichletDistribution) for u in self.row_priors
        )

    def model_log_density(self, model) -> float:
        """Log-density of the model parameters under the priors.

        Sums the Dirichlet log-densities of the initial-state and
        transition probabilities (floored at a tiny constant so boundary
        MAP estimates score finitely) and each topic estimator's
        model_log_density of its topic.

        Args:
            model (HiddenMarkovModelDistribution): Model to score.

        Returns:
            Prior log-density of the model parameters.

        """
        rv = 0.0
        if self.has_conj_prior:
            tiny = 1.0e-300
            rv += float(self.init_prior.log_density(np.maximum(model.w, tiny)))
            for i, row_prior in enumerate(self.row_priors):
                rv += float(row_prior.log_density(np.maximum(model.transitions[i, :], tiny)))
        for est, topic in zip(self.estimators, model.topics):
            rv += est.model_log_density(topic)
        return rv

    def scale_suff_stat(self, suff_stat, c):
        """Scale HMM sufficient statistics, delegating topics and length."""
        init_counts, trans_counts, topic_stats, len_val = suff_stat
        scaled_topics = tuple(est.scale_suff_stat(ss, c) for est, ss in zip(self.estimators, topic_stats))
        if isinstance(self.len_estimator, NullEstimator) or len_val is None:
            scaled_len = len_val
        else:
            scaled_len = self.len_estimator.scale_suff_stat(len_val, c)
        return init_counts * c, trans_counts * c, scaled_topics, scaled_len

    def estimate(self, suff_stat) -> HiddenMarkovModelDistribution:
        """Estimate a HiddenMarkovModelDistribution from sufficient statistics.

        Each topic is re-estimated with its own estimator (whose conjugate
        update carries its posterior forward as its prior). With Dirichlet
        chain priors the initial-state and per-row transition probabilities
        are the clamped Dirichlet MAP (posterior mean when degenerate) and
        the posterior Dirichlets are carried as the new prior; otherwise the
        maximum likelihood estimates are returned with uniform fallbacks.

        Args:
            suff_stat: Tuple (init_counts, trans_counts, topic stats,
                len_value) as returned by HiddenMarkovModelAccumulator.value().

        Returns:
            HiddenMarkovModelDistribution object.

        """
        init_counts, trans_counts, topic_stats, len_val = suff_stat
        s = self.num_states

        topics = [self.estimators[k].estimate(topic_stats[k]) for k in range(s)]

        if isinstance(self.len_estimator, NullEstimator) or len_val is None:
            len_dist = null_dist
        else:
            len_dist = self.len_estimator.estimate(len_val)

        if self.has_conj_prior:
            a0 = np.asarray(self.init_prior.get_parameters(), dtype=float)
            w = _map_probs(init_counts, a0)
            init_posterior = DirichletDistribution(init_counts + a0)

            trans_mat = np.zeros((s, s))
            row_posteriors = []
            for i in range(s):
                ai = np.asarray(self.row_priors[i].get_parameters(), dtype=float)
                trans_mat[i, :] = _map_probs(trans_counts[i, :], ai)
                row_posteriors.append(DirichletDistribution(trans_counts[i, :] + ai))

            return HiddenMarkovModelDistribution(
                topics, w, trans_mat, name=self.name, prior=(init_posterior, row_posteriors), len_dist=len_dist
            )

        else:
            w = init_counts / init_counts.sum() if init_counts.sum() > 0 else np.ones(s) / s
            row_sums = trans_counts.sum(axis=1, keepdims=True)
            trans_mat = np.where(row_sums > 0, trans_counts / np.maximum(row_sums, 1.0), 1.0 / s)

            return HiddenMarkovModelDistribution(topics, w, trans_mat, name=self.name, len_dist=len_dist)
