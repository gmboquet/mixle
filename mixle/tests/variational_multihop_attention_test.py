"""Variational multi-hop attention: 2-hop chain over tied latent embeddings (mixle.stats.latent)."""

import io
import unittest

import numpy as np

from mixle.inference import optimize
from mixle.stats.latent.variational_multihop_attention import (
    VariationalMultiHopAttentionDistribution,
    VariationalMultiHopAttentionEstimator,
)

S, Mr, Dm = 18, 3, 8
N = 2 * Mr


def _transitive(rng, n):
    out = []
    for _ in range(n):
        a = rng.choice(S, Mr, replace=False)
        b = rng.choice(S, Mr, replace=False)
        c = rng.choice(S, Mr, replace=False)
        pb = rng.permutation(Mr)
        i = rng.randint(Mr)
        j = int(np.where(b[pb] == b[i])[0][0])
        out.append((np.concatenate([a, b[pb]]), np.concatenate([b, c]), int(a[i]), int(c[j])))
    return out


def _est(seed=0, mc=4, anneal_iters=100):
    return VariationalMultiHopAttentionEstimator(
        num_symbols=S,
        embed_dim=Dm,
        num_targets=S,
        sigma2=0.3,
        lr=0.07,
        mc=mc,
        prior_strength=0.05,
        anneal_iters=anneal_iters,
        seed=seed,
    )


class MechanicsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.d = VariationalMultiHopAttentionDistribution(
            rng.randn(S, Dm), np.full((S, Dm), np.log(0.3)), np.full((S, S), 1.0 / S), sigma2=0.3
        )
        self.data = _transitive(rng, 6)

    def test_log_density_matches_seq(self):
        enc = self.d.dist_to_encoder().seq_encode(self.data)
        np.testing.assert_allclose(self.d.seq_log_density(enc), [self.d.log_density(x) for x in self.data], atol=1e-9)

    def test_encoder_roundtrip(self):
        keys, vals, q, t = self.d.dist_to_encoder().seq_encode(self.data)
        self.assertEqual(keys.shape, (6, N))
        self.assertEqual(q.shape, (6,))

    def test_predict_proba_normalized(self):
        ck = np.array([x[0] for x in self.data])
        cv = np.array([x[1] for x in self.data])
        q = np.array([x[2] for x in self.data])
        p = self.d.predict_proba(ck, cv, q)
        self.assertEqual(p.shape, (6, S))
        np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-9)

    def test_sampler_and_embeddings(self):
        s = self.d.sampler(0).sample(3)
        self.assertEqual(len(s), 3)
        self.assertEqual(self.d.embeddings().shape, (S, Dm))

    def test_accumulator_additive(self):
        enc = self.d.dist_to_encoder().seq_encode(self.data)
        acc = _est().accumulator_factory().make()
        acc.seq_update(enc, np.ones(6), self.d)
        v = acc.value()
        acc.combine(v)
        np.testing.assert_allclose(acc.value()[2], 2 * v[2])  # emission_count additive


class LearningTest(unittest.TestCase):
    def test_variational_two_hop_solves_transitive_without_collapse(self):
        rng = np.random.RandomState(1)
        tr = _transitive(rng, 800)
        te = _transitive(rng, 600)
        model = optimize(
            tr,
            _est(seed=0, mc=2, anneal_iters=80),
            max_its=140,
            delta=None,
            rng=np.random.RandomState(2),
            out=io.StringIO(),
        )
        ck = np.array([x[0] for x in te])
        cv = np.array([x[1] for x in te])
        q = np.array([x[2] for x in te])
        t = np.array([x[3] for x in te])
        acc = float(np.mean(model.predict_proba(ck, cv, q).argmax(1) == t))
        spread = float(np.mean(np.sqrt(((model.mean[:, None] - model.mean[None]) ** 2).sum(2))))
        self.assertGreater(acc, 0.7)  # two latent-embedding hops solve the transitive lookup
        self.assertGreater(spread, 2.0)  # annealing prevented the prior collapse


if __name__ == "__main__":
    unittest.main()
