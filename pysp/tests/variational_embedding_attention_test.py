"""Variational-EM attention with tied latent embeddings (pysp.stats.latent)."""

import io
import unittest

import numpy as np

from pysp.inference import optimize
from pysp.stats.latent.responsibility_attention import ResponsibilityAttentionEstimator
from pysp.stats.latent.variational_embedding_attention import (
    VariationalEmbeddingAttentionDistribution,
    VariationalEmbeddingAttentionEstimator,
)

C, K, N, D = 6, 2, 4, 8
S, T = C * K, C
CONCEPT = np.repeat(np.arange(C), K)
SBC = [np.where(CONCEPT == c)[0] for c in range(C)]


def _make(rng, n, f_map, query_syn):
    out = []
    for _ in range(n):
        cs = rng.choice(C, size=N, replace=False)
        ctx = np.array([rng.choice(SBC[c]) for c in cs])
        j = rng.randint(N)
        qc = cs[j]
        qsym = int(SBC[qc][rng.choice(query_syn)])
        out.append((ctx, qsym, int(f_map[qc])))
    return out


def _est(seed=0):
    return VariationalEmbeddingAttentionEstimator(
        num_symbols=S, context_length=N, embed_dim=D, num_targets=T, sigma2=0.5, lr=0.08, mc=5, seed=seed
    )


class MechanicsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.d = VariationalEmbeddingAttentionDistribution(
            rng.randn(S, D), np.full((S, D), np.log(0.3)), np.full((S, T), 1.0 / T), np.ones(N) / N, sigma2=0.5
        )
        self.data = _make(rng, 8, np.arange(C), [0, 1])

    def test_log_density_matches_seq(self):
        enc = self.d.dist_to_encoder().seq_encode(self.data)
        np.testing.assert_allclose(self.d.seq_log_density(enc), [self.d.log_density(x) for x in self.data], atol=1e-10)

    def test_encoder_roundtrip(self):
        ctx, q, t = self.d.dist_to_encoder().seq_encode(self.data)
        self.assertEqual(ctx.shape, (8, N))
        self.assertEqual(q.shape, (8,))
        self.assertEqual(t.shape, (8,))

    def test_predict_proba_normalized(self):
        ctx = np.array([x[0] for x in self.data])
        q = np.array([x[1] for x in self.data])
        p = self.d.predict_proba(ctx, q)
        self.assertEqual(p.shape, (8, T))
        np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-9)

    def test_sampler(self):
        s = self.d.sampler(0).sample(4)
        self.assertEqual(len(s), 4)
        ctx, q, t = s[0]
        self.assertEqual(len(ctx), N)
        self.assertIsInstance(q, int)
        self.assertIsInstance(t, int)

    def test_embeddings_shape(self):
        self.assertEqual(self.d.embeddings().shape, (S, D))

    def test_accumulator_additive(self):
        enc = self.d.dist_to_encoder().seq_encode(self.data)
        est = _est()
        acc = est.accumulator_factory().make()
        acc.seq_update(enc, np.ones(8), self.d)
        v = acc.value()
        acc.combine(v)  # doubling
        np.testing.assert_allclose(acc.value()[2], 2 * v[2])  # emission_count additive


class LearningTest(unittest.TestCase):
    def test_transfers_to_held_out_queries_and_beats_untied_lookup(self):
        rng = np.random.RandomState(1)
        f_map = rng.permutation(C)
        # train queries only ever use synonym index 0; test queries use synonym index 1 (held out)
        tr = _make(rng, 1500, f_map, [0])
        te = _make(rng, 1000, f_map, [1])
        model = optimize(tr, _est(seed=0), max_its=250, delta=None, rng=np.random.RandomState(2), out=io.StringIO())
        ctx = np.array([x[0] for x in te])
        q = np.array([x[1] for x in te])
        t = np.array([x[2] for x in te])
        acc = float(np.mean(model.predict_proba(ctx, q).argmax(1) == t))
        # tying transfers the embedding from the key role to the never-queried synonym
        self.assertGreater(acc, 0.85)
        # embeddings cluster by concept
        from scipy.spatial.distance import cdist

        Dm = cdist(model.mean, model.mean)
        np.fill_diagonal(Dm, np.inf)
        nn_same = float(np.mean(CONCEPT[Dm.argmin(1)] == CONCEPT))
        self.assertGreater(nn_same, 0.8)

        # untied lookup baseline (one-hot query, learned keys) cannot transfer to held-out queries
        eye = np.eye(S)
        tr_lk = [(c, eye[qi], ti) for c, qi, ti in tr]
        te_lk = [(c, eye[qi], ti) for c, qi, ti in te]
        lk = optimize(
            tr_lk,
            ResponsibilityAttentionEstimator(num_symbols=S, context_length=N, query_dim=S, num_targets=T, sigma2=0.5),
            max_its=40,
            rng=np.random.RandomState(3),
            out=io.StringIO(),
        )
        lk_acc = float(
            np.mean(lk.predict_proba(np.array([x[0] for x in te_lk]), np.array([x[1] for x in te_lk])).argmax(1) == t)
        )
        self.assertLess(lk_acc, 0.5)  # near chance
        self.assertGreater(acc, lk_acc + 0.3)  # tied embeddings decisively win

    def test_data_loglik_increases(self):
        rng = np.random.RandomState(4)
        tr = _make(rng, 1200, rng.permutation(C), [0, 1])
        est = _est(seed=1)
        enc = est.accumulator_factory().make().acc_to_encoder().seq_encode(tr)
        acc = est.accumulator_factory().make()
        acc.seq_initialize(enc, np.ones(len(tr)), np.random.RandomState(5))
        model = est.estimate(None, acc.value())
        lls = []
        for _ in range(60):
            a = est.accumulator_factory().make()
            a.seq_update(enc, np.ones(len(tr)), model)
            model = est.estimate(None, a.value())
            lls.append(a.ll)
        # the (Monte-Carlo) data log-likelihood trends strongly upward
        self.assertGreater(np.mean(lls[-10:]), np.mean(lls[:10]) + 100)


if __name__ == "__main__":
    unittest.main()
