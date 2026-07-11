"""Structured HMMs: a composable transition operator (dense / low-rank) + forward-backward + EM."""

import unittest

import numpy as np

import mixle.stats as S
from mixle.stats.latent.structured_hmm import (
    BlockDiagonalTransition,
    DenseTransition,
    KroneckerTransition,
    LowRankTransition,
    StructuredHMM,
    _row_normalize,
    chunked_state_posteriors,
    fit_chunked,
)


class TransitionOperatorTest(unittest.TestCase):
    def test_low_rank_is_consistent_and_low_rank(self):
        rng = np.random.RandomState(0)
        K, r = 6, 2
        lr = LowRankTransition(_row_normalize(rng.rand(K, r)), _row_normalize(rng.rand(r, K)))
        a = lr.as_matrix()
        alpha, v = rng.rand(K), rng.rand(K)
        self.assertTrue(np.allclose(a.sum(axis=1), 1.0))  # row-stochastic
        self.assertLessEqual(np.linalg.matrix_rank(a), r)  # genuinely low rank
        self.assertTrue(np.allclose(lr.forward(alpha), alpha @ a))  # forward == alpha @ A
        self.assertTrue(np.allclose(lr.backward(v), a @ v))  # backward == A @ v


class StructuredHMMTest(unittest.TestCase):
    def test_dense_forward_ll_matches_builtin_hmm(self):
        rng = np.random.RandomState(0)
        emis = [S.GaussianDistribution(-3, 1), S.GaussianDistribution(3, 1)]
        a = np.array([[0.9, 0.1], [0.2, 0.8]])
        pi = [0.6, 0.4]
        sh = StructuredHMM(emis, pi, DenseTransition(a))
        bi = S.HiddenMarkovModelDistribution(emis, w=pi, transitions=a.tolist())
        seq = [float(x) for x in rng.normal(0, 3, 25)]
        ours = sh.seq_log_density([seq])[0]
        ref = float(bi.seq_log_density(bi.dist_to_encoder().seq_encode([seq]))[0])
        self.assertAlmostEqual(ours, ref, places=6)

    def test_low_rank_em_recovers_chain_and_is_monotone(self):
        rng = np.random.RandomState(0)
        K, r = 8, 2
        gen = StructuredHMM(
            [S.GaussianDistribution(3.0 * k, 1.0) for k in range(K)],
            np.ones(K) / K,
            LowRankTransition(_row_normalize(rng.rand(K, r)), _row_normalize(rng.rand(r, K))),
        )
        seqs = [gen.sampler(seed=s).sample(50) for s in range(50)]
        init = StructuredHMM(
            [S.GaussianDistribution(3.0 * k + rng.uniform(-1, 1), 1.0) for k in range(K)],
            np.ones(K) / K,
            LowRankTransition(_row_normalize(rng.rand(K, r)), _row_normalize(rng.rand(r, K))),
        )
        _, trace = init.fit(seqs, max_its=30)
        self.assertTrue(np.all(np.diff(trace) >= -1e-6))  # EM log-likelihood non-decreasing
        means = sorted(e.mu for e in init.emissions)
        truth = [3.0 * k for k in range(K)]
        self.assertLess(max(abs(m - t) for m, t in zip(means, truth)), 1.0)

    def test_low_rank_has_fewer_parameters(self):
        K, r = 40, 2
        lr = LowRankTransition(_row_normalize(np.ones((K, r))), _row_normalize(np.ones((r, K))))
        dense_params = K * K
        low_rank_params = lr.g.size + lr.phi.size
        self.assertLess(low_rank_params, dense_params // 5)  # >5x fewer at K=40, r=2


class CombinatorTest(unittest.TestCase):
    def test_kronecker_matches_explicit_factorial_matrix(self):
        rng = np.random.RandomState(0)
        a1, a2 = _row_normalize(rng.rand(3, 3)), _row_normalize(rng.rand(4, 4))
        kt = KroneckerTransition(DenseTransition(a1), DenseTransition(a2))
        a = np.kron(a1, a2)
        alpha, v = rng.rand(12), rng.rand(12)
        self.assertEqual(kt.n_states, 12)
        self.assertTrue(np.allclose(kt.forward(alpha), alpha @ a))  # alpha @ (A1 (x) A2)
        self.assertTrue(np.allclose(kt.backward(v), a @ v))
        self.assertTrue(np.allclose(kt.as_matrix().sum(axis=1), 1.0))

    def test_factorial_hmm_em_is_monotone(self):
        rng = np.random.RandomState(0)
        a1, a2 = _row_normalize(rng.rand(3, 3)), _row_normalize(rng.rand(4, 4))
        gen = StructuredHMM(
            [S.GaussianDistribution(5 * k, 1) for k in range(12)],
            np.ones(12) / 12,
            KroneckerTransition(DenseTransition(a1), DenseTransition(a2)),
        )
        seqs = [gen.sampler(seed=s).sample(50) for s in range(40)]
        init = StructuredHMM(
            [S.GaussianDistribution(5 * k + rng.uniform(-1, 1), 1) for k in range(12)],
            np.ones(12) / 12,
            KroneckerTransition(
                DenseTransition(_row_normalize(rng.rand(3, 3))), DenseTransition(_row_normalize(rng.rand(4, 4)))
            ),
        )
        _, trace = init.fit(seqs, max_its=25)
        self.assertTrue(np.all(np.diff(trace) >= -1e-6))
        self.assertGreater(trace[-1], trace[0])

    def test_block_diagonal_is_block_structured(self):
        bd = BlockDiagonalTransition(
            [DenseTransition(np.array([[0.9, 0.1], [0.2, 0.8]])), DenseTransition(np.array([[0.7, 0.3], [0.4, 0.6]]))]
        )
        m = bd.as_matrix()
        self.assertEqual(bd.n_states, 4)
        self.assertTrue(m[0, 2] == 0 and m[2, 0] == 0)  # no cross-block transitions
        self.assertTrue(np.allclose(m.sum(axis=1), 1.0))

    def test_combinators_compose_dense_and_low_rank(self):
        # the point: build rich structure -- a factorial HMM with one dense and one low-rank factor
        rng = np.random.RandomState(1)
        kt = KroneckerTransition(
            DenseTransition(_row_normalize(rng.rand(3, 3))),
            LowRankTransition(_row_normalize(rng.rand(5, 2)), _row_normalize(rng.rand(2, 5))),
        )
        self.assertEqual(kt.n_states, 15)
        self.assertTrue(np.allclose(kt.as_matrix().sum(axis=1), 1.0))


class ContractTest(unittest.TestCase):
    def test_state_count_mismatch_raises(self):
        with self.assertRaises(ValueError):  # len(pi) != n_emissions != transition.n_states
            StructuredHMM([S.GaussianDistribution(0, 1)] * 3, [0.5, 0.5], DenseTransition(np.eye(3)))

    def test_optimize_fits_a_structured_hmm(self):
        from mixle.inference import optimize

        rng = np.random.RandomState(0)
        k, r = 6, 2
        gen = StructuredHMM(
            [S.GaussianDistribution(4.0 * i, 1) for i in range(k)],
            np.ones(k) / k,
            LowRankTransition(_row_normalize(rng.rand(k, r)), _row_normalize(rng.rand(r, k))),
        )
        seqs = [gen.sampler(seed=s).sample(30) for s in range(24)]
        proto = StructuredHMM(
            [S.GaussianDistribution(4.0 * i + rng.uniform(-1, 1), 1) for i in range(k)],
            np.ones(k) / k,
            LowRankTransition(_row_normalize(rng.rand(k, r)), _row_normalize(rng.rand(r, k))),
        )
        fit = optimize(seqs, proto.estimator(), prev_estimate=proto, max_its=20, out=None)
        means = sorted(e.mu for e in fit.emissions)
        truth = [4.0 * i for i in range(k)]
        self.assertLess(max(abs(m - t) for m, t in zip(means, truth)), 1.0)

    def test_transition_key_ties_counts(self):
        from mixle.stats.latent.structured_hmm import StructuredHMMAccumulator

        def emit():
            return [S.GaussianDistribution(0, 1).estimator().accumulator_factory().make() for _ in range(2)]

        a1 = StructuredHMMAccumulator(emit(), DenseTransition(np.eye(2)), keys=(None, "T"))
        a2 = StructuredHMMAccumulator(emit(), DenseTransition(np.eye(2)), keys=(None, "T"))
        a1.trans_acc = np.array([[2.0, 1.0], [0.0, 3.0]])
        a2.trans_acc = np.array([[1.0, 0.0], [1.0, 1.0]])
        store = {}
        a1.key_merge(store)
        a2.key_merge(store)
        a1.key_replace(store)
        a2.key_replace(store)
        self.assertTrue(np.array_equal(a1.trans_acc, [[3.0, 1.0], [1.0, 4.0]]))  # pooled
        self.assertTrue(np.array_equal(a1.trans_acc, a2.trans_acc))  # both share the tied pool


class ForgettingParallelTest(unittest.TestCase):
    def _ergodic_hmm(self):
        a = np.array([[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]])
        emis = [S.GaussianDistribution(-4, 1), S.GaussianDistribution(0, 1), S.GaussianDistribution(4, 1)]
        return StructuredHMM(emis, np.array([1 / 3, 1 / 3, 1 / 3]), DenseTransition(a))

    def test_chunked_posteriors_converge_to_exact_with_overlap(self):
        hmm = self._ergodic_hmm()
        seq = hmm.sampler(seed=1).sample(1200)
        exact = hmm._forward_backward(hmm._log_b(seq))[4]
        err0 = np.max(np.abs(chunked_state_posteriors(hmm, seq, chunk=150, overlap=0) - exact))
        err20 = np.max(np.abs(chunked_state_posteriors(hmm, seq, chunk=150, overlap=20) - exact))
        self.assertGreater(err0, 1e-3)  # no overlap -> visible boundary error
        self.assertLess(err20, 1e-9)  # forgetting absorbs it -> ~exact

    def test_chunked_em_recovers_chain_and_parallel_matches_serial(self):
        rng = np.random.RandomState(0)
        hmm = self._ergodic_hmm()
        seqs = [hmm.sampler(seed=s).sample(200) for s in range(10)]

        def init():
            return StructuredHMM(
                [S.GaussianDistribution(m + rng.uniform(-1, 1), 1) for m in (-4, 0, 4)],
                np.ones(3) / 3,
                DenseTransition(_row_normalize(rng.rand(3, 3) + np.eye(3))),
            )

        h_serial = init()
        fit_chunked(h_serial, seqs, chunk=80, overlap=25, max_its=20, workers=0)
        h_par = init()  # same init stream consumed identically
        fit_chunked(h_par, seqs, chunk=80, overlap=25, max_its=20, workers=4)
        means = sorted(e.mu for e in h_serial.emissions)
        self.assertLess(max(abs(m - t) for m, t in zip(means, [-4, 0, 4])), 0.5)
        self.assertTrue(
            np.allclose(means, sorted(e.mu for e in h_par.emissions), atol=1e-9)
        )  # chunks are independent -> parallel is exact


if __name__ == "__main__":
    unittest.main()


class DecodingTest(unittest.TestCase):
    def _two_state(self):
        a = np.array([[0.92, 0.08], [0.08, 0.92]])
        return StructuredHMM(
            [S.GaussianDistribution(-5, 0.5), S.GaussianDistribution(5, 0.5)], [0.5, 0.5], DenseTransition(a)
        )

    def _gen(self, hmm, n=200, seed=0):
        rng = np.random.RandomState(seed)
        a = hmm.transition.as_matrix()
        s = 0
        states, obs = [], []
        for _ in range(n):
            states.append(s)
            obs.append(float(rng.normal([-5, 5][s], 0.5)))
            s = rng.choice(2, p=a[s])
        return np.array(states), obs

    def test_viterbi_and_posterior_decode_recover_states(self):
        hmm = self._two_state()
        states, obs = self._gen(hmm)
        self.assertGreater((hmm.viterbi(obs) == states).mean(), 0.95)
        self.assertGreater((hmm.posterior_decode(obs) == states).mean(), 0.95)
        self.assertEqual(hmm.state_posteriors(obs).shape, (len(obs), 2))


class SparseAndPriorTest(unittest.TestCase):
    def test_left_to_right_structure_is_preserved(self):
        from mixle.stats.latent.structured_hmm import SparseTransition, left_to_right_edges

        rng = np.random.RandomState(0)
        sp = SparseTransition(4, left_to_right_edges(4, skip=1))
        a = sp.as_matrix()
        self.assertTrue(np.allclose(np.tril(a, -1), 0))  # left-to-right: no backward transitions
        self.assertTrue(np.allclose(a.sum(axis=1), 1.0))
        v = rng.rand(4)
        self.assertTrue(np.allclose(sp.forward(v), v @ a))  # forward == v @ A
        acc = sp.new_accumulator()
        for _ in range(50):
            sp.accumulate(acc, rng.rand(4), rng.rand(4), 1.0)
        self.assertTrue(np.allclose(np.tril(sp.estimate(acc).as_matrix(), -1), 0))  # structure survives EM

    def test_sticky_prior_raises_self_transition(self):
        from mixle.stats.latent.structured_hmm import sticky_transition

        rng = np.random.RandomState(0)
        st = sticky_transition(np.full((3, 3), 1 / 3), kappa=20.0)
        acc = st.new_accumulator()
        for _ in range(30):
            st.accumulate(acc, rng.rand(3), rng.rand(3), 1.0)
        self.assertTrue(np.all(np.diag(st.estimate(acc).as_matrix()) > 0.4))

    def test_kron_initial_is_factorized(self):
        from mixle.stats.latent.structured_hmm import kron_initial

        pi = kron_initial([0.6, 0.4], [0.3, 0.3, 0.4])
        self.assertEqual(len(pi), 6)
        self.assertAlmostEqual(pi.sum(), 1.0)
        self.assertAlmostEqual(pi[0], 0.6 * 0.3)


class StreamingTest(unittest.TestCase):
    def test_streaming_estimator_drives_online_baum_welch(self):
        from mixle.inference import StreamingEstimator

        rng = np.random.RandomState(0)
        gen = StructuredHMM(
            [S.GaussianDistribution(-4, 1), S.GaussianDistribution(4, 1)],
            [0.5, 0.5],
            DenseTransition(np.array([[0.9, 0.1], [0.1, 0.9]])),
        )
        proto = StructuredHMM(
            [S.GaussianDistribution(-2, 1), S.GaussianDistribution(2, 1)],
            [0.5, 0.5],
            DenseTransition(_row_normalize(rng.rand(2, 2) + np.eye(2))),
        )
        stream = StreamingEstimator(proto.estimator(), model=proto)
        model = None
        for b in range(15):
            batch = [gen.sampler(seed=b * 10 + i).sample(40) for i in range(8)]
            model = stream.update(batch)
        means = sorted(e.mu for e in model.emissions)
        self.assertLess(max(abs(m - t) for m, t in zip(means, [-4, 4])), 0.5)


class InputOutputHMMTest(unittest.TestCase):
    def test_iohmm_recovers_input_dependent_transitions(self):
        from mixle.stats.latent.structured_hmm import InputOutputHMM

        rng = np.random.RandomState(0)
        a0, a1 = np.array([[0.95, 0.05], [0.05, 0.95]]), np.array([[0.05, 0.95], [0.95, 0.05]])
        gen = InputOutputHMM(
            [S.GaussianDistribution(-5, 0.5), S.GaussianDistribution(5, 0.5)],
            [0.5, 0.5],
            [DenseTransition(a0), DenseTransition(a1)],
        )

        def gen_seq(seed):
            r = np.random.RandomState(seed)
            s, obs, inp = 0, [], []
            for _ in range(60):
                obs.append(float(r.normal([-5, 5][s], 0.5)))
                m = r.randint(2)
                inp.append(m)
                s = r.choice(2, p=gen.transitions[m].as_matrix()[s])
            return obs, inp

        data = [gen_seq(s) for s in range(40)]
        init = InputOutputHMM(
            [S.GaussianDistribution(-2, 1), S.GaussianDistribution(2, 1)],
            [0.5, 0.5],
            [DenseTransition(_row_normalize(rng.rand(2, 2) + np.eye(2))) for _ in range(2)],
        )
        _, trace = init.fit([d[0] for d in data], [d[1] for d in data], max_its=30)
        self.assertTrue(np.all(np.diff(trace) >= -1e-6))
        self.assertGreater(init.transitions[0].as_matrix()[0, 0], 0.8)  # input 0 = sticky
        self.assertGreater(init.transitions[1].as_matrix()[0, 1], 0.8)  # input 1 = flip


class ExplicitDurationHMMTest(unittest.TestCase):
    def test_forward_matches_brute_force_segmentation(self):

        from mixle.stats.latent.structured_hmm import ExplicitDurationHMM, _logsumexp

        rng = np.random.RandomState(0)
        K, D = 2, 3
        m = ExplicitDurationHMM(
            [S.GaussianDistribution(-2, 1), S.GaussianDistribution(2, 1)],
            [0.6, 0.4],
            np.array([[0, 1.0], [1.0, 0]]),
            np.array([[0.2, 0.5, 0.3], [0.5, 0.3, 0.2]]),
            D,
        )
        seq = [float(x) for x in rng.normal(0, 2, 6)]

        def brute(seq):
            t_len, log_b = len(seq), m._log_b(seq)
            logd, loga, logpi = np.log(m.dur + 1e-300), np.log(m.a + 1e-300), np.log(m.pi + 1e-300)
            total = []

            def rec(t, prev, lp):
                if t == t_len:
                    total.append(lp)
                    return
                for j in range(K):
                    if prev is not None and m.a[prev, j] == 0:
                        continue
                    trans = logpi[j] if prev is None else loga[prev, j]
                    for d in range(1, min(D, t_len - t) + 1):
                        seg = sum(log_b[t + s, j] for s in range(d))
                        rec(t + d, j, lp + trans + logd[j, d - 1] + seg)

            rec(0, None, 0.0)
            return _logsumexp(total)

        self.assertAlmostEqual(m.forward_loglik(seq), brute(seq), places=8)

    def test_em_recovers_durations(self):
        from mixle.stats.latent.structured_hmm import ExplicitDurationHMM

        D = 5
        dur_true = np.array([[0.0, 0.1, 0.2, 0.5, 0.2], [0.4, 0.5, 0.1, 0.0, 0.0]])
        gen = ExplicitDurationHMM(
            [S.GaussianDistribution(-5, 0.6), S.GaussianDistribution(5, 0.6)],
            [0.5, 0.5],
            np.array([[0, 1.0], [1.0, 0]]),
            dur_true,
            D,
        )
        seqs = [gen.sampler(seed=s).sample(80) for s in range(50)]
        init = ExplicitDurationHMM(
            [S.GaussianDistribution(-2, 1), S.GaussianDistribution(2, 1)],
            [0.5, 0.5],
            np.array([[0, 1.0], [1.0, 0]]),
            np.ones((2, D)) / D,
            D,
        )
        _, trace = init.fit(seqs, max_its=30)
        self.assertTrue(np.all(np.diff(trace) >= -1e-6))
        d0 = float((np.arange(1, D + 1) * init.dur[0]).sum())  # mean dwell time, state 0
        self.assertAlmostEqual(d0, 3.8, delta=0.4)
        self.assertLess(max(abs(m - t) for m, t in zip(sorted(e.mu for e in init.emissions), [-5, 5])), 0.5)


class JitForwardTest(unittest.TestCase):
    def test_jit_forward_matches_numpy(self):
        try:
            import jax  # noqa: F401
        except Exception:  # noqa: BLE001
            self.skipTest("jax not installed")
        from mixle.stats.latent.structured_hmm import jit_forward_loglik

        rng = np.random.RandomState(0)
        k = 6
        hmm = StructuredHMM(
            [S.GaussianDistribution(3 * i, 1) for i in range(k)],
            np.ones(k) / k,
            DenseTransition(_row_normalize(rng.rand(k, k) + np.eye(k))),
        )
        seq = [float(x) for x in rng.normal(0, 6, 1500)]
        self.assertAlmostEqual(jit_forward_loglik(hmm)(seq), hmm.seq_log_density([seq])[0], places=5)


class EnumerationTest(unittest.TestCase):
    def _hmm(self, transition):
        emis = [
            S.CategoricalDistribution({0: 0.7, 1: 0.2, 2: 0.1}),
            S.CategoricalDistribution({0: 0.1, 1: 0.2, 2: 0.7}),
        ]
        ld = S.CategoricalDistribution({2: 0.5, 3: 0.5})
        return StructuredHMM(emis, [0.6, 0.4], transition, len_dist=ld), ld

    def test_top_k_descending_and_prob_matches_density(self):
        hmm, ld = self._hmm(DenseTransition(np.array([[0.8, 0.2], [0.3, 0.7]])))
        tk = hmm.enumerator().top_k(8)
        logps = [lp for _, lp in tk]
        self.assertTrue(all(a >= b for a, b in zip(logps, logps[1:])))  # descending probability
        for seq, logp in tk:  # enumerated prob == StructuredHMM density * len_prob
            own = float(hmm.seq_log_density([seq])[0]) + float(ld.log_density(len(seq)))
            self.assertAlmostEqual(logp, own, places=9)

    def test_rank_seek_nucleus(self):
        hmm, _ = self._hmm(DenseTransition(np.array([[0.8, 0.2], [0.3, 0.7]])))
        en = hmm.enumerator()
        top = en.top_k(1)[0][0]
        self.assertEqual(en.rank(top).rank, 0)
        self.assertTrue(en.rank(top).exact)
        self.assertGreaterEqual(en.nucleus_size(0.95).covered_mass, 0.95)

    def test_low_rank_operator_enumerates_via_as_matrix(self):
        hmm, _ = self._hmm(
            LowRankTransition(
                _row_normalize(np.array([[0.6, 0.4], [0.3, 0.7]])), _row_normalize(np.array([[0.5, 0.5], [0.2, 0.8]]))
            )
        )
        self.assertEqual(len(hmm.enumerator().top_k(3)), 3)

    def test_no_len_dist_raises_enumeration_error(self):
        from mixle.enumeration import EnumerationError

        hmm = StructuredHMM([S.CategoricalDistribution({0: 0.5, 1: 0.5})] * 2, [0.5, 0.5], DenseTransition(np.eye(2)))
        with self.assertRaises(EnumerationError):
            hmm.enumerator()


class TerminalStateTest(unittest.TestCase):
    def _model(self, transition_cls=DenseTransition):
        a = _row_normalize(np.array([[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.0, 0.0, 1.0]]))
        emis = [S.GaussianDistribution(-3, 1), S.GaussianDistribution(0, 1), S.GaussianDistribution(3, 1)]
        return StructuredHMM(emis, np.array([0.7, 0.3, 0.0]), DenseTransition(a), terminal_states={2}), a

    def test_terminal_loglik_matches_dense_reference(self):
        from mixle.stats.latent.hidden_markov import terminal_forward_loglik

        hmm, a = self._model()
        rng = np.random.RandomState(0)
        seq = [float(x) for x in rng.normal(0, 3, 8)]
        log_b = hmm._log_b(seq)
        ref = terminal_forward_loglik(
            np.log(hmm.pi + 1e-300), np.log(a + 1e-300), log_b, np.array([False, False, True])
        )
        self.assertAlmostEqual(hmm._forward_backward(log_b)[5], ref, places=8)

    def test_sampler_stops_at_terminal_state(self):
        hmm, _ = self._model()
        lengths = [len(hmm.sampler(seed=s).sample(50)) for s in range(40)]
        self.assertLess(max(lengths), 50)  # terminal stopping -> not the full requested length
        self.assertGreaterEqual(min(lengths), 1)

    def test_em_increases_likelihood_and_recovers(self):
        from mixle.inference import optimize

        gen, _ = self._model()
        seqs = [gen.sampler(seed=s).sample(50) for s in range(200)]
        init = StructuredHMM(
            gen.emissions,
            gen.pi,
            DenseTransition(_row_normalize(np.array([[0.5, 0.3, 0.2], [0.3, 0.5, 0.2], [0, 0, 1.0]]))),
            terminal_states={2},
        )
        fit = optimize(seqs, init.estimator(), prev_estimate=init, max_its=15, out=None)
        self.assertEqual(fit.terminal_states, {2})  # retained through the contract
        self.assertLess(max(abs(m - t) for m, t in zip(sorted(e.mu for e in fit.emissions), [-3, 0, 3])), 0.4)


class IOHMMTerminalTest(unittest.TestCase):
    def test_iohmm_terminal_loglik_matches_input_aware_reference(self):
        from scipy.special import logsumexp

        from mixle.stats.latent.structured_hmm import InputOutputHMM

        rng = np.random.RandomState(0)
        a0 = _row_normalize(np.array([[0.6, 0.3, 0.1], [0.3, 0.5, 0.2], [0, 0, 1.0]]))
        a1 = _row_normalize(np.array([[0.4, 0.4, 0.2], [0.2, 0.6, 0.2], [0, 0, 1.0]]))
        emis = [S.GaussianDistribution(-3, 1), S.GaussianDistribution(0, 1), S.GaussianDistribution(3, 1)]
        pi = np.array([0.7, 0.3, 0.0])
        io = InputOutputHMM(emis, pi, [DenseTransition(a0), DenseTransition(a1)], terminal_states={2})
        obs = [float(x) for x in rng.normal(0, 3, 7)]
        inp = [int(rng.randint(2)) for _ in range(7)]
        log_b = io._log_b(obs)
        term, nonterm = np.array([False, False, True]), np.array([True, True, False])
        la = np.log(pi + 1e-300) + log_b[0]
        for t in range(1, len(obs)):
            a = [a0, a1][inp[t - 1]]
            prev = np.where(nonterm, la, -np.inf)
            la = log_b[t] + logsumexp(prev[:, None] + np.log(a + 1e-300), axis=0)
        ref = float(logsumexp(la[term]))
        self.assertAlmostEqual(io._forward_backward(log_b, inp)[5], ref, places=8)


class EDHMMDecodingTest(unittest.TestCase):
    def test_viterbi_segments_recovers_planted_segmentation(self):
        from mixle.stats.latent.structured_hmm import ExplicitDurationHMM

        D = 6
        dur = np.zeros((2, D))
        dur[0, 3] = 1.0  # state 0 always duration 4
        dur[1, 1] = 1.0  # state 1 always duration 2
        m = ExplicitDurationHMM(
            [S.GaussianDistribution(-6, 0.3), S.GaussianDistribution(6, 0.3)],
            [1.0, 0.0],
            np.array([[0, 1.0], [1.0, 0]]),
            dur,
            D,
        )
        rng = np.random.RandomState(0)
        true_states = [0] * 4 + [1] * 2 + [0] * 4 + [1] * 2
        seq = [float(rng.normal([-6, 6][s], 0.3)) for s in true_states]
        segs = m.viterbi_segments(seq)
        self.assertEqual([d for _, _, d in segs], [4, 2, 4, 2])  # durations recovered
        recovered = []
        for st, _, d in segs:
            recovered += [st] * d
        self.assertEqual(recovered[: len(true_states)], true_states)  # exact state path


class EDHMMPosteriorTest(unittest.TestCase):
    def test_state_posteriors_and_decode(self):
        from mixle.stats.latent.structured_hmm import ExplicitDurationHMM

        D = 6
        dur = np.zeros((2, D))
        dur[0, 3] = 1.0
        dur[1, 1] = 1.0
        m = ExplicitDurationHMM(
            [S.GaussianDistribution(-6, 0.4), S.GaussianDistribution(6, 0.4)],
            [1.0, 0.0],
            np.array([[0, 1.0], [1.0, 0]]),
            dur,
            D,
        )
        rng = np.random.RandomState(0)
        true = [0] * 4 + [1] * 2 + [0] * 4 + [1] * 2
        seq = [float(rng.normal([-6, 6][s], 0.4)) for s in true]
        g = m.state_posteriors(seq)
        self.assertTrue(np.allclose(g.sum(axis=1), 1.0))
        self.assertTrue(np.array_equal(m.posterior_decode(seq), np.array(true)))


class NumbaFastFitTest(unittest.TestCase):
    def test_fast_dense_fit_equals_numpy_fit(self):
        try:
            from mixle.utils.optional_deps import HAS_NUMBA
        except Exception:  # noqa: BLE001
            HAS_NUMBA = False
        if not HAS_NUMBA:
            self.skipTest("numba not installed")
        rng = np.random.RandomState(0)
        k = 5
        emis = [S.GaussianDistribution(4 * i, 1) for i in range(k)]
        gen = StructuredHMM(emis, np.ones(k) / k, DenseTransition(_row_normalize(rng.rand(k, k) + 2 * np.eye(k))))
        seqs = [gen.sampler(seed=s).sample(150) for s in range(30)]

        def init():
            return StructuredHMM(
                [S.GaussianDistribution(4 * i + rng.uniform(-1, 1), 1) for i in range(k)],
                np.ones(k) / k,
                DenseTransition(_row_normalize(rng.rand(k, k) + np.eye(k))),
            )

        rng = np.random.RandomState(1)
        hf = init()
        rng = np.random.RandomState(1)
        hs = init()
        hf.fit(seqs, max_its=20, fast=True)
        hs.fit(seqs, max_its=20, fast=False)
        self.assertTrue(
            np.allclose(sorted(e.mu for e in hf.emissions), sorted(e.mu for e in hs.emissions), atol=1e-6)
        )  # numba fast path is the SAME EM, just faster


class HSMMExpansionTest(unittest.TestCase):
    def _edhmm(self):
        from mixle.stats.latent.structured_hmm import ExplicitDurationHMM

        dur = np.array([[0.1, 0.3, 0.4, 0.2], [0.5, 0.3, 0.1, 0.1]])
        return ExplicitDurationHMM(
            [S.GaussianDistribution(-2, 1), S.GaussianDistribution(2, 1)],
            [0.6, 0.4],
            np.array([[0, 1.0], [1.0, 0]]),
            dur,
            4,
        )

    def test_expansion_likelihood_matches_edhmm_exactly(self):
        m = self._edhmm()
        rng = np.random.RandomState(0)
        seq = [float(x) for x in rng.normal(0, 2, 9)]
        exp = m.to_structured_hmm()
        self.assertEqual(exp.K, 2 * 4)  # K*D sub-states
        self.assertAlmostEqual(m.forward_loglik(seq), float(exp.seq_log_density([seq])[0]), places=9)

    def test_expansion_enables_viterbi_over_substates(self):
        m = self._edhmm()
        rng = np.random.RandomState(1)
        seq = [float(x) for x in rng.normal(0, 2, 9)]
        exp = m.to_structured_hmm()
        base_path = [int(v // m.D) for v in exp.viterbi(seq)]  # sub-state -> base state
        self.assertEqual(len(base_path), len(seq))
        self.assertTrue(all(0 <= s < m.K for s in base_path))

    def test_final_states_restrict_the_ending(self):
        # a StructuredHMM with final_states must end in one of them; loglik <= the unrestricted loglik
        a = _row_normalize(np.array([[0.6, 0.4], [0.4, 0.6]]))
        emis = [S.GaussianDistribution(-2, 1), S.GaussianDistribution(2, 1)]
        free = StructuredHMM(emis, [0.5, 0.5], DenseTransition(a))
        restricted = StructuredHMM(emis, [0.5, 0.5], DenseTransition(a), final_states={0})
        rng = np.random.RandomState(0)
        seq = [float(x) for x in rng.normal(0, 2, 6)]
        self.assertLessEqual(restricted.seq_log_density([seq])[0], free.seq_log_density([seq])[0] + 1e-9)


class HSMMEnumerationTest(unittest.TestCase):
    def _edhmm(self):
        from mixle.stats.latent.structured_hmm import ExplicitDurationHMM

        dur = np.array([[0.3, 0.5, 0.2], [0.6, 0.3, 0.1]])
        emis = [S.CategoricalDistribution({0: 0.7, 1: 0.3}), S.CategoricalDistribution({0: 0.2, 1: 0.8})]
        return ExplicitDurationHMM(emis, [0.6, 0.4], np.array([[0, 1.0], [1.0, 0]]), dur, 3)

    def test_enumeration_matches_brute_force(self):
        import itertools

        m = self._edhmm()
        ld = S.CategoricalDistribution({2: 0.5, 3: 0.5})
        hmm = m.to_structured_hmm(len_dist=ld)
        enum = m.enumerator(ld).top_k(12)

        brute = []
        for length in (2, 3):
            for seq in itertools.product([0, 1], repeat=length):
                lp = float(hmm.seq_log_density([list(seq)])[0]) + float(ld.log_density(length))
                brute.append((list(seq), lp))
        brute.sort(key=lambda x: -x[1])

        for i in range(min(len(enum), len(brute))):
            self.assertEqual(enum[i][0], brute[i][0])  # same order
            self.assertAlmostEqual(enum[i][1], brute[i][1], places=9)  # same probability
        self.assertTrue(all(enum[i][1] >= enum[i + 1][1] - 1e-12 for i in range(len(enum) - 1)))  # descending

    def test_enumerated_prob_equals_forward_loglik_plus_len(self):
        m = self._edhmm()
        ld = S.CategoricalDistribution({2: 0.5, 3: 0.5})
        seq, logp = m.enumerator(ld).top_k(1)[0]
        self.assertAlmostEqual(logp, m.forward_loglik(seq) + float(ld.log_density(len(seq))), places=9)


class IOHMMContractTest(unittest.TestCase):
    def test_optimize_fits_iohmm_via_paired_records(self):
        from mixle.inference import optimize
        from mixle.stats.latent.structured_hmm import InputOutputHMM

        rng = np.random.RandomState(0)
        a0, a1 = np.array([[0.95, 0.05], [0.05, 0.95]]), np.array([[0.05, 0.95], [0.95, 0.05]])
        gen = InputOutputHMM(
            [S.GaussianDistribution(-5, 0.5), S.GaussianDistribution(5, 0.5)],
            [0.5, 0.5],
            [DenseTransition(a0), DenseTransition(a1)],
        )

        def gen_seq(seed):
            r = np.random.RandomState(seed)
            s, rec = 0, []
            for _ in range(60):
                m = int(r.randint(2))
                rec.append((float(r.normal([-5, 5][s], 0.5)), m))  # one record = list of (obs, input) pairs
                s = r.choice(2, p=gen.transitions[m].as_matrix()[s])
            return rec

        data = [gen_seq(s) for s in range(40)]
        proto = InputOutputHMM(
            [S.GaussianDistribution(-2, 1), S.GaussianDistribution(2, 1)],
            [0.5, 0.5],
            [DenseTransition(_row_normalize(rng.rand(2, 2) + np.eye(2))) for _ in range(2)],
        )
        fit = optimize(data, proto.estimator(), prev_estimate=proto, max_its=25, out=None)
        self.assertLess(max(abs(m - t) for m, t in zip(sorted(e.mu for e in fit.emissions), [-5, 5])), 0.5)
        self.assertGreater(fit.transitions[0].as_matrix()[0, 0], 0.8)  # input 0 = sticky
        self.assertGreater(fit.transitions[1].as_matrix()[0, 1], 0.8)  # input 1 = flip


class EDHMMContractTest(unittest.TestCase):
    def test_optimize_fits_edhmm_and_recovers_durations(self):
        from mixle.inference import optimize
        from mixle.stats.latent.structured_hmm import ExplicitDurationHMM

        D = 5
        dur_true = np.array([[0.0, 0.1, 0.2, 0.5, 0.2], [0.4, 0.5, 0.1, 0.0, 0.0]])
        gen = ExplicitDurationHMM(
            [S.GaussianDistribution(-5, 0.6), S.GaussianDistribution(5, 0.6)],
            [0.5, 0.5],
            np.array([[0, 1.0], [1.0, 0]]),
            dur_true,
            D,
        )
        seqs = [gen.sampler(seed=s).sample(55) for s in range(20)]
        proto = ExplicitDurationHMM(
            [S.GaussianDistribution(-2, 1), S.GaussianDistribution(2, 1)],
            [0.5, 0.5],
            np.array([[0, 1.0], [1.0, 0]]),
            np.ones((2, D)) / D,
            D,
        )
        fit = optimize(seqs, proto.estimator(), prev_estimate=proto, max_its=18, out=None)
        self.assertLess(max(abs(m - t) for m, t in zip(sorted(e.mu for e in fit.emissions), [-5, 5])), 0.5)
        d0 = float((np.arange(1, D + 1) * fit.dur[0]).sum())  # mean dwell, state 0
        self.assertAlmostEqual(d0, 3.8, delta=0.4)
