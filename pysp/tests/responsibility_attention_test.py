"""Responsibility-attention head: EM-able mixture over context positions (pysp.stats.latent)."""

import io
import unittest

import numpy as np

from pysp.inference import optimize
from pysp.stats import seq_encode, seq_log_density_sum
from pysp.stats.latent.mixture import MixtureEstimator
from pysp.stats.latent.responsibility_attention import (
    ResponsibilityAttentionDistribution,
    ResponsibilityAttentionEstimator,
)

S, N, D, T = 16, 6, 16, 16
EYE = np.eye(S)


def _recall_data(rng, n, f_map, noise=0.3):
    """Associative recall: a noisy query points at one of N distinct context symbols; target = f(it)."""
    out = []
    for _ in range(n):
        syms = rng.choice(S, size=N, replace=False)
        j = rng.randint(N)
        s = syms[j]
        out.append((syms, EYE[s] + noise * rng.randn(D), int(f_map[s])))
    return out


def _est(sigma2=0.5):
    return ResponsibilityAttentionEstimator(num_symbols=S, context_length=N, query_dim=D, num_targets=T, sigma2=sigma2)


class MechanicsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.d = ResponsibilityAttentionDistribution(
            rng.randn(S, D), np.full((S, T), 1.0 / T), position_prior=np.ones(N) / N, sigma2=0.5
        )
        self.data = _recall_data(rng, 8, np.arange(S))

    def test_log_density_matches_seq(self):
        enc = self.d.dist_to_encoder().seq_encode(self.data)
        seq = self.d.seq_log_density(enc)
        ind = np.array([self.d.log_density(x) for x in self.data])
        np.testing.assert_allclose(seq, ind, atol=1e-10)

    def test_encoder_roundtrip_shapes(self):
        ctx, y, t = self.d.dist_to_encoder().seq_encode(self.data)
        self.assertEqual(ctx.shape, (8, N))
        self.assertEqual(y.shape, (8, D))
        self.assertEqual(t.shape, (8,))

    def test_sampler(self):
        s = self.d.sampler(0).sample(5)
        self.assertEqual(len(s), 5)
        ctx, y, t = s[0]
        self.assertEqual(len(ctx), N)
        self.assertEqual(len(y), D)
        self.assertIsInstance(t, int)

    def test_accumulator_value_roundtrip(self):
        enc = self.d.dist_to_encoder().seq_encode(self.data)
        est = _est()
        acc = est.accumulator_factory().make()
        acc.seq_update(enc, np.ones(8), self.d)
        v = acc.value()
        acc2 = est.accumulator_factory().make().from_value(v)
        for a, b in zip(acc.value(), acc2.value()):
            np.testing.assert_allclose(a, b)
        # combine doubles the statistics
        acc.combine(v)
        np.testing.assert_allclose(acc.value()[1], 2 * v[1])

    def test_predict_proba_rows_sum_to_one(self):
        ctx = np.array([x[0] for x in self.data])
        y = np.array([x[1] for x in self.data])
        p = self.d.predict_proba(ctx, y)
        self.assertEqual(p.shape, (8, T))
        np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-9)


class LearningTest(unittest.TestCase):
    def test_em_recovers_content_addressing(self):
        rng = np.random.RandomState(1)
        f_map = rng.permutation(S)
        tr = _recall_data(rng, 4000, f_map)
        te = _recall_data(rng, 1500, f_map)
        model = optimize(tr, _est(), max_its=40, rng=np.random.RandomState(2), out=io.StringIO())
        ctx = np.array([x[0] for x in te])
        y = np.array([x[1] for x in te])
        t = np.array([x[2] for x in te])
        acc = float(np.mean(model.predict_proba(ctx, y).argmax(1) == t))
        self.assertGreater(acc, 0.85)
        # structure recovered: emission == f, keys peak at the right symbol
        self.assertEqual(float(np.mean(model.emission.argmax(1) == f_map)), 1.0)
        self.assertEqual(float(np.mean(model.key_means.argmax(1) == np.arange(S))), 1.0)

    def test_beats_position_only_baseline(self):
        # a query-independent (position-only) predictor cannot content-address -> ~chance (1/N)
        rng = np.random.RandomState(3)
        f_map = rng.permutation(S)
        te = _recall_data(rng, 2000, f_map)
        model = optimize(
            _recall_data(rng, 3000, f_map), _est(), max_its=30, rng=np.random.RandomState(4), out=io.StringIO()
        )
        ctx = np.array([x[0] for x in te])
        y = np.array([x[1] for x in te])
        t = np.array([x[2] for x in te])
        acc = float(np.mean(model.predict_proba(ctx, y).argmax(1) == t))
        # marginal baseline: ignore the query, predict the most likely target over context positions
        base_pred = np.einsum("i,nit->nt", model.position_prior, model.emission[ctx]).argmax(1)
        base_acc = float(np.mean(base_pred == t))
        self.assertGreater(acc, 0.85)
        self.assertGreater(acc, 3 * base_acc)

    def test_em_monotone(self):
        rng = np.random.RandomState(5)
        tr = _recall_data(rng, 1500, rng.permutation(S))
        est = _est()
        enc = est.accumulator_factory().make().acc_to_encoder().seq_encode(tr)
        acc = est.accumulator_factory().make()
        acc.seq_initialize(enc, np.ones(len(tr)), np.random.RandomState(6))
        model = est.estimate(None, acc.value())
        lls = []
        for _ in range(15):
            a = est.accumulator_factory().make()
            a.seq_update(enc, np.ones(len(tr)), model)
            model = est.estimate(None, a.value())
            lls.append(float(model.seq_log_density(enc).sum()))
        self.assertTrue(all(lls[i + 1] >= lls[i] - 1e-3 for i in range(len(lls) - 1)))


class CompositionTest(unittest.TestCase):
    def test_fits_inside_a_mixture(self):
        # the headline claim: a responsibility-attention head composes with other pysp models
        rng = np.random.RandomState(7)
        f_map = rng.permutation(S)
        tr = _recall_data(rng, 2000, f_map)
        mest = MixtureEstimator([_est(), _est()])
        mix = optimize(tr, mest, max_its=10, rng=np.random.RandomState(8), out=io.StringIO())
        _, ll = seq_log_density_sum(seq_encode(tr, model=mix), mix)
        self.assertTrue(np.isfinite(ll))
        self.assertAlmostEqual(float(np.sum(mix.w)), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
