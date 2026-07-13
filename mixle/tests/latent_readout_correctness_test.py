"""Correctness tests for the latent-model read-out APIs and EM updates (review findings L-1..L-12).

Each test pins a defect found in the `stats/latent` review against an independent brute-force
reference on a tiny model:

  * Viterbi backtracking (not per-position argmax) for ``viterbi``/``seq_viterbi``  [L-1]
  * hierarchical-mixture EM monotonicity on variable-length documents             [L-2]
  * taus/topic-mixture emission scoring, scalar == vectorized == brute force      [L-3, L-4]
  * StructuredHMM ``fit(fast=True)`` honoring ``final_states``                    [L-5]
  * heterogeneous-emission HMM read-outs (viterbi/posterior variants)             [L-6]
  * HMM encoder equality comparing emission encoders (mixtures of HMMs)           [L-7]
  * ``seq_posterior`` returning smoothing (not filtered) marginals                [L-8]
  * engine forward-backward impossible-observation guard                          [L-9]
  * LDA corpora whose LAST document is empty                                      [L-10]
  * ``fit_chunked`` log-likelihood including the emission-max terms               [L-11]
  * StructuredHMM-family zero-mass / empty-batch guards                           [L-12]
"""

import itertools
import unittest

import numpy as np
from scipy.special import logsumexp

from mixle.engines import NUMPY_ENGINE
from mixle.stats import (
    CategoricalDistribution,
    CategoricalEstimator,
    ExponentialDistribution,
    GaussianDistribution,
    HiddenMarkovModelDistribution,
    HierarchicalMixtureDistribution,
    HierarchicalMixtureEstimator,
    LDADistribution,
    MixtureDistribution,
)
from mixle.stats.latent.hidden_markov import hmm_engine_forward_backward, hmm_pad_log_emissions
from mixle.stats.latent.structured_hmm import DenseTransition, InputOutputHMM, StructuredHMM, fit_chunked


def _emission_log_density(hmm, state, obs):
    """State ``state``'s log emission density: the topic itself, or the taus-weighted topic mixture."""
    if getattr(hmm, "has_topics", False):
        return logsumexp([hmm.log_taus[state, j] + hmm.topics[j].log_density(obs) for j in range(hmm.n_topics)])
    return hmm.topics[state].log_density(obs)


def _path_log_prob(hmm, x, path):
    """Joint log-probability of one hidden path and the observations (no length term)."""
    lp = np.log(hmm.w[path[0]]) + _emission_log_density(hmm, path[0], x[0])
    for t in range(1, len(x)):
        lp += np.log(hmm.transitions[path[t - 1], path[t]]) + _emission_log_density(hmm, path[t], x[t])
    return lp


def _brute_force_paths(hmm, x):
    """All ``(path, log_prob)`` pairs by exhaustive path enumeration."""
    return [(path, _path_log_prob(hmm, x, path)) for path in itertools.product(range(hmm.n_states), repeat=len(x))]


def _brute_force_log_density(hmm, x):
    """Sequence log-likelihood by exhaustive path enumeration (no length term)."""
    return logsumexp([lp for _, lp in _brute_force_paths(hmm, x)])


def _brute_force_viterbi(hmm, x):
    """The exact max-probability hidden path and its log-probability."""
    best_path, best_lp = max(_brute_force_paths(hmm, x), key=lambda pair: pair[1])
    return list(best_path), best_lp


def _brute_force_gamma(hmm, x):
    """Exact smoothing marginals P(z_t = k | x) by exhaustive path enumeration."""
    gamma = np.zeros((len(x), hmm.n_states))
    for path, lp in _brute_force_paths(hmm, x):
        p = np.exp(lp)
        for t, state in enumerate(path):
            gamma[t, state] += p
    return gamma / gamma.sum(axis=1, keepdims=True)


def _sticky_hmm(use_numba):
    """The ledger's L-1 repro: a sticky 2-state HMM whose per-position argmax is NOT the Viterbi path."""
    topics = [
        CategoricalDistribution(pmap={"a": 0.9, "b": 0.1}),
        CategoricalDistribution(pmap={"a": 0.2, "b": 0.8}),
    ]
    return HiddenMarkovModelDistribution(
        topics,
        w=[0.8, 0.2],
        transitions=[[0.999, 0.001], [0.001, 0.999]],
        use_numba=use_numba,
    )


def _heterogeneous_hmm(use_numba):
    """A two-state HMM whose states emit from DIFFERENT families (PR #275 heterogeneous emissions).

    Gaussian + Categorical is the ledger's L-6 repro: their encoders produce structurally different
    encodings, so any read-out that encodes with ``topics[0]``'s encoder alone cannot score state 1.
    """
    topics = [GaussianDistribution(mu=0.0, sigma2=1.0), CategoricalDistribution(pmap={1.0: 0.7, 2.0: 0.3})]
    return HiddenMarkovModelDistribution(
        topics,
        w=[0.6, 0.4],
        transitions=[[0.8, 0.2], [0.3, 0.7]],
        use_numba=use_numba,
    )


class ViterbiBacktrackingTest(unittest.TestCase):
    """L-1: viterbi()/seq_viterbi() must return the max-likelihood PATH, not per-position argmaxes."""

    x = ["a", "b", "b", "b", "b"]

    def test_viterbi_matches_brute_force(self):
        hmm = _sticky_hmm(use_numba=False)
        true_path, true_lp = _brute_force_viterbi(hmm, self.x)
        path = list(hmm.viterbi(self.x))
        self.assertEqual(path, true_path)
        self.assertAlmostEqual(_path_log_prob(hmm, self.x, path), true_lp, places=9)

    def test_seq_viterbi_matches_brute_force_blocked_encoding(self):
        hmm = _sticky_hmm(use_numba=False)
        seqs = [self.x, ["a", "b", "b"]]
        enc = hmm.dist_to_encoder().seq_encode(seqs)
        flat = hmm.seq_viterbi(enc)
        (_, _, _, len_vec, idx_mat, _, _), _, _ = enc[0]
        for s, seq in enumerate(seqs):
            path = [int(flat[idx_mat[s, t]]) for t in range(int(len_vec[s]))]
            true_path, _ = _brute_force_viterbi(hmm, seq)
            self.assertEqual(path, true_path, "sequence %d" % s)

    def test_seq_viterbi_matches_brute_force_numba_encoding(self):
        hmm = _sticky_hmm(use_numba=True)
        seqs = [self.x, ["a", "b", "b"]]
        enc = hmm.dist_to_encoder().seq_encode(seqs)
        flat = hmm.seq_viterbi(enc)
        self.assertIsNotNone(flat)
        (_, sz, _), _ = enc[1]
        tz = np.concatenate([[0], np.asarray(sz)]).cumsum()
        for s, seq in enumerate(seqs):
            path = [int(v) for v in flat[tz[s] : tz[s + 1]]]
            true_path, _ = _brute_force_viterbi(hmm, seq)
            self.assertEqual(path, true_path, "sequence %d" % s)


class HierarchicalMixtureMonotoneEMTest(unittest.TestCase):
    """L-2: EM on variable-length documents must not decrease the log-likelihood."""

    def _em_trace(self, seed, iterations=10):
        rng = np.random.RandomState(seed)
        symbols = ["a", "b", "c"]
        docs = []
        for _ in range(12):
            length = int(rng.randint(1, 8))
            probs = rng.dirichlet(np.ones(len(symbols)))
            docs.append([symbols[i] for i in rng.choice(len(symbols), size=length, p=probs)])

        def random_topic():
            probs = rng.dirichlet(np.ones(len(symbols)))
            return CategoricalDistribution(pmap=dict(zip(symbols, probs)))

        model = HierarchicalMixtureDistribution(
            [random_topic(), random_topic()],
            rng.dirichlet(np.ones(2)),
            rng.dirichlet(np.ones(2), size=2),
        )
        estimator = HierarchicalMixtureEstimator([CategoricalEstimator(), CategoricalEstimator()], num_mixtures=2)
        enc = model.dist_to_encoder().seq_encode(docs)
        weights = np.ones(len(docs))
        trace = []
        for _ in range(iterations):
            trace.append(float(np.sum(model.seq_log_density(enc))))
            accumulator = estimator.accumulator_factory().make()
            accumulator.seq_update(enc, weights, model)
            model = estimator.estimate(None, accumulator.value())
        trace.append(float(np.sum(model.seq_log_density(enc))))
        return np.asarray(trace)

    def test_em_log_likelihood_is_monotone(self):
        # Seeds 13/15/17 produce genuine LL decreases (up to ~1e-2/iteration) under the token-level
        # outer-weight M-step; 0/1/2 are unremarkable controls.
        for seed in (13, 15, 17, 0, 1, 2):
            with self.subTest(seed=seed):
                trace = self._em_trace(seed)
                deltas = np.diff(trace)
                self.assertGreaterEqual(
                    float(deltas.min()), -1.0e-9, "seed %d: EM decreased the log-likelihood: %s" % (seed, trace)
                )


class TausMixtureEmissionTest(unittest.TestCase):
    """L-3 + L-4: the taus/topic-mixture parameterization must score correctly and consistently."""

    def _taus_hmm(self, use_numba):
        topics = [
            CategoricalDistribution(pmap={"a": 0.8, "b": 0.2}),
            CategoricalDistribution(pmap={"a": 0.1, "b": 0.9}),
        ]
        return HiddenMarkovModelDistribution(
            topics,
            w=[0.6, 0.4],
            transitions=[[0.9, 0.1], [0.3, 0.7]],
            taus=[[0.7, 0.3], [0.2, 0.8]],
            use_numba=use_numba,
        )

    def test_scalar_and_seq_log_density_match_brute_force(self):
        seqs = [["a", "b", "a"], ["b"], ["b", "b", "a", "a"]]
        for use_numba in (False, True):
            hmm = self._taus_hmm(use_numba=use_numba)
            enc = hmm.dist_to_encoder().seq_encode(seqs)
            seq_ll = np.asarray(hmm.seq_log_density(enc))
            for s, seq in enumerate(seqs):
                expected = _brute_force_log_density(hmm, seq)
                with self.subTest(use_numba=use_numba, seq=s):
                    self.assertAlmostEqual(hmm.log_density(seq), expected, places=9)
                    self.assertAlmostEqual(float(seq_ll[s]), expected, places=9)


class StructuredHMMFinalStatesFastFitTest(unittest.TestCase):
    """L-5: fit(fast=True) must optimize the SAME objective as fast=False on final_states models."""

    def _model(self):
        return StructuredHMM(
            [GaussianDistribution(mu=-3.0, sigma2=1.0), GaussianDistribution(mu=3.0, sigma2=1.0)],
            [0.6, 0.4],
            DenseTransition(np.array([[0.7, 0.3], [0.4, 0.6]])),
            final_states={1},
        )

    def test_fast_fit_matches_slow_fit_objective(self):
        rng = np.random.RandomState(0)
        seqs = [[float(v) for v in rng.normal(0.0, 3.0, size=6)] for _ in range(4)]
        reference_ll = float(np.sum(self._model().seq_log_density(seqs)))
        _, fast_trace = self._model().fit(seqs, max_its=1, fast=True)
        _, slow_trace = self._model().fit(seqs, max_its=1, fast=False)
        self.assertAlmostEqual(fast_trace[0], slow_trace[0], places=9)
        self.assertAlmostEqual(fast_trace[0], reference_ll, places=9)


class HeterogeneousEmissionReadoutTest(unittest.TestCase):
    """L-6: viterbi/latent_posterior/seq_posterior/seq_viterbi must support heterogeneous emissions."""

    x = [0.1, 1.0, 2.0, 0.3]

    def test_viterbi_matches_brute_force(self):
        hmm = _heterogeneous_hmm(use_numba=True)
        true_path, _ = _brute_force_viterbi(hmm, self.x)
        self.assertEqual(list(hmm.viterbi(self.x)), true_path)

    def test_latent_posterior_marginals_match_brute_force(self):
        hmm = _heterogeneous_hmm(use_numba=True)
        np.testing.assert_allclose(hmm.latent_posterior(self.x).marginals(), _brute_force_gamma(hmm, self.x), atol=1e-9)

    def test_seq_posterior_matches_brute_force(self):
        hmm = _heterogeneous_hmm(use_numba=True)
        seqs = [self.x, [0.2, 1.0]]
        enc = hmm.dist_to_encoder().seq_encode(seqs)
        posteriors = hmm.seq_posterior(enc)
        for s, seq in enumerate(seqs):
            np.testing.assert_allclose(
                posteriors[s], _brute_force_gamma(hmm, seq), atol=1e-9, err_msg="sequence %d" % s
            )

    def test_seq_viterbi_matches_brute_force(self):
        hmm = _heterogeneous_hmm(use_numba=True)
        seqs = [self.x, [0.2, 1.0]]
        enc = hmm.dist_to_encoder().seq_encode(seqs)
        flat = hmm.seq_viterbi(enc)
        self.assertIsNotNone(flat)
        (_, sz, _), _ = enc[1]
        tz = np.concatenate([[0], np.asarray(sz)]).cumsum()
        for s, seq in enumerate(seqs):
            true_path, _ = _brute_force_viterbi(hmm, seq)
            self.assertEqual([int(v) for v in flat[tz[s] : tz[s + 1]]], true_path, "sequence %d" % s)

    def test_seq_viterbi_blocked_encoding_matches_brute_force(self):
        hmm = _heterogeneous_hmm(use_numba=False)
        enc = hmm.dist_to_encoder().seq_encode([self.x])
        flat = hmm.seq_viterbi(enc)
        (_, _, _, len_vec, idx_mat, _, _), _, _ = enc[0]
        path = [int(flat[idx_mat[0, t]]) for t in range(int(len_vec[0]))]
        true_path, _ = _brute_force_viterbi(hmm, self.x)
        self.assertEqual(path, true_path)


class MixtureOfHeterogeneousHMMsTest(unittest.TestCase):
    """L-7: HMM encoder equality must compare emission encoders (mixtures of unlike HMMs)."""

    def _mixture(self):
        hmm_gaussian = HiddenMarkovModelDistribution(
            [GaussianDistribution(mu=-1.0, sigma2=1.0), GaussianDistribution(mu=1.0, sigma2=1.0)],
            w=[0.5, 0.5],
            transitions=[[0.8, 0.2], [0.2, 0.8]],
        )
        hmm_categorical = HiddenMarkovModelDistribution(
            [CategoricalDistribution(pmap={1.0: 0.7, 2.0: 0.3}), CategoricalDistribution(pmap={1.0: 0.2, 2.0: 0.8})],
            w=[0.5, 0.5],
            transitions=[[0.6, 0.4], [0.4, 0.6]],
        )
        return MixtureDistribution([hmm_gaussian, hmm_categorical], [0.5, 0.5])

    def test_encoder_not_declared_homogeneous(self):
        mixture = self._mixture()
        encoders = [component.dist_to_encoder() for component in mixture.components]
        self.assertFalse(encoders[0] == encoders[1])
        self.assertFalse(mixture.dist_to_encoder().homogeneous)

    def test_seq_log_density_matches_scalar(self):
        mixture = self._mixture()
        seqs = [[1.0, 2.0, 1.0], [0.5, 1.0]]
        enc = mixture.dist_to_encoder().seq_encode(seqs)
        seq_ll = np.asarray(mixture.seq_log_density(enc))
        scalar_ll = np.asarray([mixture.log_density(seq) for seq in seqs])
        np.testing.assert_allclose(seq_ll, scalar_ll, atol=1e-9)

    def test_encoder_equality_returns_bool(self):
        mixture = self._mixture()
        encoders = [component.dist_to_encoder() for component in mixture.components]
        self.assertIs(encoders[0] == encoders[1], False)  # no None fall-through


class SeqPosteriorSmoothingTest(unittest.TestCase):
    """L-8: seq_posterior must return SMOOTHING marginals P(z_t | x_1..T), not filtered ones."""

    def test_seq_posterior_matches_brute_force_gamma(self):
        hmm = _sticky_hmm(use_numba=True)
        seqs = [["a", "b", "b", "b", "b"], ["b", "a"]]
        enc = hmm.dist_to_encoder().seq_encode(seqs)
        posteriors = hmm.seq_posterior(enc)
        for s, seq in enumerate(seqs):
            np.testing.assert_allclose(
                posteriors[s], _brute_force_gamma(hmm, seq), atol=1e-9, err_msg="sequence %d" % s
            )


class EngineForwardBackwardImpossibleObservationTest(unittest.TestCase):
    """L-9: the engine E-step must guard impossible observations like the host paths do."""

    def _model(self):
        topics = [
            CategoricalDistribution(pmap={"a": 0.7, "b": 0.3}),
            CategoricalDistribution(pmap={"a": 0.2, "b": 0.8}),
        ]
        return HiddenMarkovModelDistribution(topics, w=[0.6, 0.4], transitions=[[0.7, 0.3], [0.4, 0.6]])

    def test_kernel_statistics_are_finite(self):
        dist = self._model()
        data = [["a", "b", "a"], ["a", "z", "b"], ["b", "b"]]  # "z" is out of support in every state
        _, ((idx, sz, xs), _) = dist.dist_to_encoder().seq_encode(data)
        pr_obs = np.empty((int(np.sum(sz)), dist.n_states), dtype=np.float64)
        for i in range(dist.n_states):
            pr_obs[:, i] = dist.topics[i].seq_log_density(xs)
        padded, mask, _ = hmm_pad_log_emissions(pr_obs, sz)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_w = np.log(dist.w)
            log_a = np.log(dist.transitions)
            _, gamma, xi_sum, pi = hmm_engine_forward_backward(NUMPY_ENGINE, padded, log_w, log_a, mask)
        self.assertTrue(np.all(np.isfinite(gamma)), "gamma contains non-finite values")
        self.assertTrue(np.all(np.isfinite(xi_sum)), "xi contains non-finite values")
        self.assertTrue(np.all(np.isfinite(pi)), "pi contains non-finite values")

    def test_engine_estep_matches_host(self):
        dist = self._model()
        data = [["a", "b", "a"], ["a", "z", "b"], ["b", "b"]]
        enc = dist.dist_to_encoder().seq_encode(data)
        weights = np.ones(len(data))
        estimator = dist.estimator()

        host = estimator.accumulator_factory().make()
        host.seq_update(enc, weights, dist)
        engine_acc = estimator.accumulator_factory().make()
        with np.errstate(divide="ignore", invalid="ignore"):
            engine_acc.seq_update_engine(enc, weights, dist, NUMPY_ENGINE)

        np.testing.assert_allclose(engine_acc.init_counts, host.init_counts, atol=1e-9)
        np.testing.assert_allclose(engine_acc.state_counts, host.state_counts, atol=1e-9)
        np.testing.assert_allclose(engine_acc.trans_counts, host.trans_counts, atol=1e-9)
        self.assertTrue(np.all(np.isfinite(engine_acc.init_counts)))
        self.assertTrue(np.all(np.isfinite(engine_acc.state_counts)))
        self.assertTrue(np.all(np.isfinite(engine_acc.trans_counts)))


class LDAEmptyLastDocumentTest(unittest.TestCase):
    """L-10: a corpus whose LAST document is empty must encode and score without crashing."""

    def _model(self):
        topics = [
            CategoricalDistribution(pmap={"a": 0.6, "b": 0.3, "c": 0.1}),
            CategoricalDistribution(pmap={"a": 0.1, "b": 0.3, "c": 0.6}),
        ]
        return LDADistribution(topics, alpha=[0.7, 0.4])

    def test_seq_log_density_and_seq_posterior_with_empty_last_document(self):
        model = self._model()
        docs = [[("a", 2.0), ("b", 1.0)], [("c", 3.0)], []]
        enc = model.dist_to_encoder().seq_encode(docs)
        ll = np.asarray(model.seq_log_density(enc))
        self.assertEqual(len(ll), len(docs))
        self.assertTrue(np.all(np.isfinite(ll[:2])))
        posterior = np.asarray(model.seq_posterior(enc))
        self.assertEqual(posterior.shape[0], len(docs))


class FitChunkedLogLikelihoodTest(unittest.TestCase):
    """L-11: fit_chunked's ll_trace must include the per-position emission-max terms."""

    def _model(self):
        return StructuredHMM(
            [GaussianDistribution(mu=-2.0, sigma2=1.0), GaussianDistribution(mu=2.0, sigma2=1.0)],
            [0.5, 0.5],
            DenseTransition(np.array([[0.8, 0.2], [0.3, 0.7]])),
        )

    def test_single_chunk_ll_matches_full_fit(self):
        rng = np.random.RandomState(1)
        seqs = [[float(v) for v in rng.normal(0.0, 2.0, size=8)] for _ in range(3)]
        _, full_trace = self._model().fit(seqs, max_its=1, fast=False)
        _, chunked_trace = fit_chunked(self._model(), seqs, chunk=8, overlap=0, max_its=1)
        self.assertAlmostEqual(chunked_trace[0], full_trace[0], places=9)


class StructuredHMMZeroMassGuardsTest(unittest.TestCase):
    """L-12: out-of-support observations and empty batches must degrade cleanly, not to NaN."""

    def _model(self, **kwargs):
        return StructuredHMM(
            [ExponentialDistribution(1.0), ExponentialDistribution(3.0)],
            [0.5, 0.5],
            DenseTransition(np.array([[0.8, 0.2], [0.3, 0.7]])),
            **kwargs,
        )

    def test_impossible_observation_scores_neg_inf_not_nan(self):
        hmm = self._model()
        with np.errstate(divide="ignore", invalid="ignore"):
            ll = hmm.seq_log_density([[1.0, -1.0, 2.0]])  # -1.0 is out of support in every state
        self.assertEqual(float(ll[0]), -np.inf)
        self.assertFalse(np.isnan(ll[0]))

    def test_impossible_observation_posteriors_are_finite(self):
        hmm = self._model()
        with np.errstate(divide="ignore", invalid="ignore"):
            gamma = hmm.state_posteriors([1.0, -1.0, 2.0])
        self.assertTrue(np.all(np.isfinite(gamma)), "state posteriors contain non-finite values")

    def test_fit_on_empty_batch_keeps_parameters_finite(self):
        hmm = self._model()
        with np.errstate(divide="ignore", invalid="ignore"):
            fitted, _ = hmm.fit([[]], max_its=1, fast=False)
        self.assertTrue(np.all(np.isfinite(fitted.pi)), "pi became non-finite on an empty batch")

    def test_iohmm_impossible_first_observation_is_neg_inf_not_nan(self):
        iohmm = InputOutputHMM(
            [ExponentialDistribution(1.0), ExponentialDistribution(3.0)],
            [0.5, 0.5],
            [DenseTransition(np.array([[0.8, 0.2], [0.3, 0.7]]))],
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            ll = iohmm.log_density([(-1.0, 0), (1.0, 0)])
        self.assertEqual(float(ll), -np.inf)
        self.assertFalse(np.isnan(ll))


if __name__ == "__main__":
    unittest.main()
