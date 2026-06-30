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


if __name__ == "__main__":
    unittest.main()
