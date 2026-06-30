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
        seqs = [gen.sampler(seed=s).sample(60) for s in range(60)]
        init = StructuredHMM(
            [S.GaussianDistribution(3.0 * k + rng.uniform(-1, 1), 1.0) for k in range(K)],
            np.ones(K) / K,
            LowRankTransition(_row_normalize(rng.rand(K, r)), _row_normalize(rng.rand(r, K))),
        )
        _, trace = init.fit(seqs, max_its=40)
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
        seqs = [gen.sampler(seed=s).sample(50) for s in range(60)]
        proto = StructuredHMM(
            [S.GaussianDistribution(4.0 * i + rng.uniform(-1, 1), 1) for i in range(k)],
            np.ones(k) / k,
            LowRankTransition(_row_normalize(rng.rand(k, r)), _row_normalize(rng.rand(r, k))),
        )
        fit = optimize(seqs, proto.estimator(), prev_estimate=proto, max_its=40, out=None)
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
        seqs = [hmm.sampler(seed=s).sample(400) for s in range(20)]

        def init():
            return StructuredHMM(
                [S.GaussianDistribution(m + rng.uniform(-1, 1), 1) for m in (-4, 0, 4)],
                np.ones(3) / 3,
                DenseTransition(_row_normalize(rng.rand(3, 3) + np.eye(3))),
            )

        h_serial = init()
        fit_chunked(h_serial, seqs, chunk=120, overlap=40, max_its=25, workers=0)
        h_par = init()  # same init stream consumed identically
        fit_chunked(h_par, seqs, chunk=120, overlap=40, max_its=25, workers=4)
        means = sorted(e.mu for e in h_serial.emissions)
        self.assertLess(max(abs(m - t) for m, t in zip(means, [-4, 0, 4])), 0.5)
        self.assertTrue(
            np.allclose(means, sorted(e.mu for e in h_par.emissions), atol=1e-9)
        )  # chunks are independent -> parallel is exact


if __name__ == "__main__":
    unittest.main()
