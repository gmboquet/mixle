"""Tests for the extended torch engine families (pysp.stats.torch_engine).

Covers Gamma, LogGaussian, Binomial, DiagonalGaussian, Optional and Ignored
on cpu float64. For each family: seq_log_density parity with the legacy seq_*
path, em_step parameter parity with seq_estimate, gradient MLE likelihood
improvement on a small mixture, fit_map at prior_strength=0 == fit_mle, and a
composite fusing several new leaves end-to-end.
"""
import io
import importlib
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec('torch') is not None

from pysp.stats import (
    BinomialDistribution, BinomialEstimator,
    CategoricalDistribution, CategoricalEstimator,
    CompositeDistribution, CompositeEstimator,
    DiagonalGaussianDistribution, DiagonalGaussianEstimator,
    GammaDistribution, GammaEstimator,
    GaussianDistribution, GaussianEstimator,
    IgnoredDistribution, IgnoredEstimator,
    LogGaussianDistribution, LogGaussianEstimator,
    MixtureDistribution, MixtureEstimator,
    OptionalDistribution, OptionalEstimator,
    seq_encode, seq_estimate,
)
if HAS_TORCH:
    from pysp.stats.torch_engine import TorchMixture
else:
    TorchMixture = None
from pysp.tests.kernels_ext_test import dist_params


def _cat(k):
    return CategoricalDistribution({'a': 0.6 - 0.2 * k, 'b': 0.3, 'c': 0.1 + 0.2 * k})


@unittest.skipUnless(HAS_TORCH, 'torch is not installed')
class TorchExtBase(unittest.TestCase):

    RTOL = 1.0e-8
    ATOL = 1.0e-10

    def _assert_close(self, a, b, msg='', rtol=None, atol=None):
        if isinstance(a, list):
            self.assertEqual(len(a), len(b), msg)
            for j, (x, y) in enumerate(zip(a, b)):
                self._assert_close(x, y, msg + '[%d]' % j, rtol, atol)
        elif isinstance(a, str):
            self.assertEqual(a, b, msg)
        else:
            np.testing.assert_allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                                       rtol=rtol or self.RTOL, atol=atol or self.ATOL,
                                       err_msg=msg)

    def check_ld_parity(self, model, data):
        """seq_log_density (mixture and per-component) vs the legacy seq path."""
        tm = TorchMixture(model)
        enc = tm.encode(data)
        ll_t = tm.seq_log_density(enc)
        ll_l = model.seq_log_density(model.dist_to_encoder().seq_encode(data))
        self.assertTrue(np.allclose(ll_t, ll_l, atol=1.0e-8),
                        'mixture ll max err %g' % np.nanmax(np.abs(ll_t - ll_l)))
        ll_tk = tm.seq_component_log_density(enc)
        for k, comp in enumerate(model.components):
            cl = comp.seq_log_density(comp.dist_to_encoder().seq_encode(data))
            self.assertTrue(np.allclose(ll_tk[:, k], cl, atol=1.0e-8),
                            'component %d max err %g' % (k, np.nanmax(np.abs(ll_tk[:, k] - cl))))
        return tm, enc

    def check_em_parity(self, model, est, data, steps=2):
        """em_step (torch) vs seq_estimate (legacy) parameter trajectories."""
        tm = TorchMixture(model)
        enc = tm.encode(data)
        chunked = seq_encode(data, model=model)
        m_torch = m_legacy = model
        for it in range(steps):
            m_torch = tm.em_step(enc, est, model=m_torch)
            m_legacy = seq_estimate(chunked, est, m_legacy)
            self._assert_close(dist_params(m_torch), dist_params(m_legacy),
                               'em step %d ' % (it + 1))

    def check_mle_and_map(self, start, data, max_its=400, lr=0.05):
        """fit_mle improves the likelihood; fit_map at strength 0 matches it."""
        tm = TorchMixture(start)
        enc = tm.encode(data)
        ll_start = tm.seq_log_density(enc).sum()
        m_mle, ll_mle = tm.fit_mle(enc, max_its=max_its, lr=lr, out=io.StringIO())
        self.assertTrue(np.isfinite(ll_mle))
        self.assertGreater(ll_mle, ll_start)
        # fitted model round-trips through the legacy scorer
        ll_check = m_mle.seq_log_density(m_mle.dist_to_encoder().seq_encode(data))
        self.assertTrue(np.all(np.isfinite(ll_check)))
        # zero-strength MAP optimizes the identical objective from the same start
        m_map, lp_map = tm.fit_map(enc, prior_strength=0.0, max_its=max_its, lr=lr,
                                   out=io.StringIO())
        self.assertLess(abs(ll_mle - lp_map), 1.0e-6 * max(1.0, abs(ll_mle)))
        self._assert_close(dist_params(m_map), dist_params(m_mle), 'map-vs-mle ',
                           rtol=1.0e-5, atol=1.0e-7)
        return m_mle, ll_mle


class GammaTorchTest(TorchExtBase):

    def setUp(self):
        comps = [GammaDistribution(2.0, 1.5), GammaDistribution(5.0, 0.6),
                 GammaDistribution(1.0, 3.0)]
        self.model = MixtureDistribution(comps, [0.4, 0.35, 0.25])
        self.data = self.model.sampler(seed=11).sample(size=300)
        self.est = MixtureEstimator([GammaEstimator()] * 3)

    def test_log_density_parity(self):
        self.check_ld_parity(self.model, self.data)

    def test_em_parity(self):
        self.check_em_parity(self.model, self.est, self.data)

    def test_mle_and_map(self):
        start = MixtureDistribution([GammaDistribution(1.5, 2.5), GammaDistribution(4.0, 1.0),
                                     GammaDistribution(1.2, 2.0)], [1 / 3] * 3)
        self.check_mle_and_map(start, self.data)


class LogGaussianTorchTest(TorchExtBase):

    def setUp(self):
        comps = [LogGaussianDistribution(0.0, 0.5), LogGaussianDistribution(1.5, 0.2),
                 LogGaussianDistribution(-1.0, 1.0)]
        self.model = MixtureDistribution(comps, [0.3, 0.4, 0.3])
        self.data = self.model.sampler(seed=21).sample(size=300)
        self.est = MixtureEstimator([LogGaussianEstimator()] * 3)

    def test_log_density_parity(self):
        self.check_ld_parity(self.model, self.data)

    def test_em_parity(self):
        self.check_em_parity(self.model, self.est, self.data)

    def test_mle_and_map(self):
        start = MixtureDistribution([LogGaussianDistribution(0.4, 1.0),
                                     LogGaussianDistribution(1.0, 0.6),
                                     LogGaussianDistribution(-0.4, 1.5)], [1 / 3] * 3)
        self.check_mle_and_map(start, self.data)


class BinomialTorchTest(TorchExtBase):

    def setUp(self):
        comps = [BinomialDistribution(0.2, 12), BinomialDistribution(0.5, 12),
                 BinomialDistribution(0.8, 12)]
        self.model = MixtureDistribution(comps, [0.35, 0.3, 0.35])
        self.data = self.model.sampler(seed=31).sample(size=400)
        self.est = MixtureEstimator([BinomialEstimator()] * 3)

    def test_log_density_parity(self):
        self.check_ld_parity(self.model, self.data)

    def test_em_parity(self):
        # 3 steps: the first M-step re-derives (n, min_val) from data,
        # exercising the inline torch.lgamma fallback away from the
        # precomputed coefficient column
        self.check_em_parity(self.model, self.est, self.data, steps=3)

    def test_min_val_shift_parity(self):
        comps = [BinomialDistribution(0.3 + 0.3 * k, 10, min_val=2) for k in range(2)]
        model = MixtureDistribution(comps, [0.5, 0.5])
        data = model.sampler(seed=32).sample(size=300)
        self.check_ld_parity(model, data)
        self.check_em_parity(model, MixtureEstimator([BinomialEstimator()] * 2), data)

    def test_mle_and_map(self):
        start = MixtureDistribution([BinomialDistribution(0.35, 12), BinomialDistribution(0.5, 12),
                                     BinomialDistribution(0.65, 12)], [1 / 3] * 3)
        self.check_mle_and_map(start, self.data)


class DiagonalGaussianTorchTest(TorchExtBase):

    def setUp(self):
        comps = [DiagonalGaussianDistribution([0.0, 1.0, -2.0], [1.0, 0.5, 2.0]),
                 DiagonalGaussianDistribution([4.0, -3.0, 1.0], [0.7, 1.5, 0.4]),
                 DiagonalGaussianDistribution([-4.0, 0.0, 5.0], [2.0, 1.0, 1.0])]
        self.model = MixtureDistribution(comps, [0.3, 0.3, 0.4])
        self.data = self.model.sampler(seed=41).sample(size=300)
        self.est = MixtureEstimator([DiagonalGaussianEstimator(dim=3)] * 3)

    def test_log_density_parity(self):
        self.check_ld_parity(self.model, self.data)

    def test_em_parity(self):
        self.check_em_parity(self.model, self.est, self.data)

    def test_mle_and_map(self):
        start = MixtureDistribution(
            [DiagonalGaussianDistribution([0.5, 0.5, -1.0], [2.0, 1.0, 2.0]),
             DiagonalGaussianDistribution([3.0, -2.0, 0.5], [1.0, 2.0, 1.0]),
             DiagonalGaussianDistribution([-3.0, 0.5, 4.0], [1.0, 1.0, 2.0])], [1 / 3] * 3)
        self.check_mle_and_map(start, self.data)


class OptionalTorchTest(TorchExtBase):

    def setUp(self):
        # ~30% missing Gaussians
        comps = [OptionalDistribution(GaussianDistribution(-3.0 + 3.0 * k, 1.0 + 0.5 * k),
                                      p=0.2 + 0.1 * k)
                 for k in range(3)]
        self.model = MixtureDistribution(comps, [0.4, 0.3, 0.3])
        self.data = self.model.sampler(seed=51).sample(size=400)
        self.est = MixtureEstimator([OptionalEstimator(GaussianEstimator(), est_prob=True)] * 3)

    def test_log_density_parity(self):
        self.check_ld_parity(self.model, self.data)

    def test_no_p_parity(self):
        # degenerate legacy mode: no missing probability given
        comps = [OptionalDistribution(GaussianDistribution(-2.0 + 4.0 * k, 1.0)) for k in range(2)]
        model = MixtureDistribution(comps, [0.5, 0.5])
        data = list(model.components[0].dist.sampler(seed=53).sample(size=100))
        data[::7] = [None] * len(data[::7])
        self.check_ld_parity(model, data)

    def test_em_parity(self):
        self.check_em_parity(self.model, self.est, self.data)

    def test_mle_and_map(self):
        start = MixtureDistribution(
            [OptionalDistribution(GaussianDistribution(-2.0 + 2.0 * k, 2.0), p=0.5)
             for k in range(3)], [1 / 3] * 3)
        self.check_mle_and_map(start, self.data)


class IgnoredTorchTest(TorchExtBase):

    def setUp(self):
        self.fixed = GaussianDistribution(1.0, 4.0)
        comps = [CompositeDistribution((GaussianDistribution(-5.0 + 5.0 * k, 1.0),
                                        IgnoredDistribution(self.fixed)))
                 for k in range(3)]
        self.model = MixtureDistribution(comps, [0.3, 0.3, 0.4])
        self.data = self.model.sampler(seed=61).sample(size=300)
        self.est = MixtureEstimator([CompositeEstimator((GaussianEstimator(),
                                                         IgnoredEstimator(dist=self.fixed)))] * 3)

    def test_log_density_parity(self):
        self.check_ld_parity(self.model, self.data)

    def test_em_parity(self):
        self.check_em_parity(self.model, self.est, self.data)

    def test_mle_and_map(self):
        start = MixtureDistribution(
            [CompositeDistribution((GaussianDistribution(-3.0 + 3.0 * k, 2.0),
                                    IgnoredDistribution(self.fixed)))
             for k in range(3)], [1 / 3] * 3)
        m_mle, _ = self.check_mle_and_map(start, self.data)
        # the wrapped dist is fixed: it must come through the fit untouched
        for c in m_mle.components:
            self.assertEqual(str(c.dists[1].dist), str(self.fixed))

    def test_changed_wrapped_dist_rejected(self):
        tm = TorchMixture(self.model)
        enc = tm.encode(self.data)
        other = MixtureDistribution(
            [CompositeDistribution((GaussianDistribution(-5.0 + 5.0 * k, 1.0),
                                    IgnoredDistribution(GaussianDistribution(9.0, 1.0))))
             for k in range(3)], [0.3, 0.3, 0.4])
        with self.assertRaises(ValueError):
            tm.seq_log_density(enc, model=other)


class MixedExtCompositeTorchTest(TorchExtBase):
    """All new leaf families plus Optional/Ignored fused into one composite."""

    def setUp(self):
        self.fixed = GaussianDistribution(0.5, 2.0)
        comps = []
        for k in range(2):
            comps.append(CompositeDistribution((
                GammaDistribution(2.0 + k, 1.0 + 0.5 * k),
                LogGaussianDistribution(0.5 * k, 0.4),
                BinomialDistribution(0.3 + 0.3 * k, 11),
                DiagonalGaussianDistribution([2.0 * k, -2.0 * k], [1.0, 0.5 + k]),
                OptionalDistribution(GaussianDistribution(3.0 * k, 1.0), p=0.3),
                IgnoredDistribution(self.fixed),
                _cat(k),
            )))
        self.model = MixtureDistribution(comps, [0.45, 0.55])
        self.data = self.model.sampler(seed=71).sample(size=300)
        comp_est = CompositeEstimator((
            GammaEstimator(),
            LogGaussianEstimator(),
            BinomialEstimator(),
            DiagonalGaussianEstimator(dim=2),
            OptionalEstimator(GaussianEstimator(), est_prob=True),
            IgnoredEstimator(dist=self.fixed),
            CategoricalEstimator(),
        ))
        self.est = MixtureEstimator([comp_est] * 2)

    def test_log_density_parity(self):
        self.check_ld_parity(self.model, self.data)

    def test_em_parity(self):
        self.check_em_parity(self.model, self.est, self.data)

    def test_mle_and_map(self):
        comps = []
        for k in range(2):
            comps.append(CompositeDistribution((
                GammaDistribution(1.5 + 0.5 * k, 1.5),
                LogGaussianDistribution(0.25 * k + 0.1, 0.7),
                BinomialDistribution(0.4 + 0.1 * k, 11),
                DiagonalGaussianDistribution([1.0 * k, -1.0 * k], [1.5, 1.0]),
                OptionalDistribution(GaussianDistribution(1.5 * k + 0.5, 2.0), p=0.5),
                IgnoredDistribution(self.fixed),
                _cat(k),
            )))
        start = MixtureDistribution(comps, [0.5, 0.5])
        self.check_mle_and_map(start, self.data, max_its=300)

    def test_fit_map_default_priors_run(self):
        # default priors must compose over all the new families at strength > 0
        tm = TorchMixture(self.model)
        enc = tm.encode(self.data)
        fitted, lp = tm.fit_map(enc, prior_strength=1.0, max_its=150, lr=0.05,
                                out=io.StringIO())
        self.assertTrue(np.isfinite(lp))
        ll = fitted.seq_log_density(fitted.dist_to_encoder().seq_encode(self.data))
        self.assertTrue(np.all(np.isfinite(ll)))


if __name__ == '__main__':
    unittest.main()
