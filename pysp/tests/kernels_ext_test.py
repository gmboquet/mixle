"""Tests for the extended fused numba kernel families (pysp.stats.compute.fused_kernels).

Covers Gamma, LogGaussian, Binomial, DiagonalGaussian, Optional and Ignored.
For each family a small mixture is checked against the legacy vectorized
seq_* path: encode + seq_log_density parity, em_step vs seq_estimate parameter
parity, and a weighted_suff_stats round-trip through the legacy
estimator.estimate.
"""

import unittest

import numpy as np

from pysp.stats import (
    BinomialDistribution,
    BinomialEstimator,
    CategoricalDistribution,
    CategoricalEstimator,
    CompositeDistribution,
    CompositeEstimator,
    DiagonalGaussianDistribution,
    DiagonalGaussianEstimator,
    GammaDistribution,
    GammaEstimator,
    GaussianDistribution,
    GaussianEstimator,
    IgnoredDistribution,
    IgnoredEstimator,
    LogGaussianDistribution,
    LogGaussianEstimator,
    MixtureDistribution,
    MixtureEstimator,
    OptionalDistribution,
    OptionalEstimator,
    seq_encode,
    seq_estimate,
)
from pysp.stats.compute.fused_kernels import CompiledMixture


def _cat(k):
    return CategoricalDistribution({"a": 0.6 - 0.2 * k, "b": 0.3, "c": 0.1 + 0.2 * k})


def dist_params(d):
    """Nested parameter extraction for cross-path model comparison."""
    if isinstance(d, MixtureDistribution):
        return [np.asarray(d.w, dtype=float)] + [dist_params(c) for c in d.components]
    if isinstance(d, CompositeDistribution):
        return [dist_params(c) for c in d.dists]
    if isinstance(d, OptionalDistribution):
        return [float(d.p), dist_params(d.dist)]
    if isinstance(d, IgnoredDistribution):
        return [str(d.dist)]
    if isinstance(d, (GaussianDistribution, LogGaussianDistribution)):
        return [d.mu, d.sigma2]
    if isinstance(d, GammaDistribution):
        return [d.k, d.theta]
    if isinstance(d, BinomialDistribution):
        return [d.p, float(d.n), float(d.min_val if d.min_val is not None else 0)]
    if isinstance(d, DiagonalGaussianDistribution):
        return [d.mu, d.covar]
    if isinstance(d, CategoricalDistribution):
        return [sorted(d.pmap.keys()), np.array([d.pmap[v] for v in sorted(d.pmap.keys())])]
    raise TypeError("no parameter extractor for %s" % type(d).__name__)


class KernelsExtBase(unittest.TestCase):
    RTOL = 1.0e-8
    ATOL = 1.0e-10

    def _assert_close(self, a, b, msg=""):
        if isinstance(a, list):
            self.assertEqual(len(a), len(b), msg)
            for j, (x, y) in enumerate(zip(a, b)):
                self._assert_close(x, y, msg + "[%d]" % j)
        elif isinstance(a, str):
            self.assertEqual(a, b, msg)
        else:
            np.testing.assert_allclose(
                np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=self.RTOL, atol=self.ATOL, err_msg=msg
            )

    def check_seq_parity(self, model, data):
        """encode + seq_log_density (mixture and per-component) vs the legacy path."""
        cm = CompiledMixture(model)
        enc = cm.encode(data)
        ll_k = cm.seq_log_density(enc)
        ll_l = model.seq_log_density(model.dist_to_encoder().seq_encode(data))
        self.assertTrue(
            np.allclose(ll_k, ll_l, rtol=1.0e-10, atol=1.0e-10),
            "mixture ll max err %g" % np.nanmax(np.abs(ll_k - ll_l)),
        )
        ll_ck = cm.seq_component_log_density(enc)
        for k, comp in enumerate(model.components):
            cl = comp.seq_log_density(comp.dist_to_encoder().seq_encode(data))
            self.assertTrue(
                np.allclose(ll_ck[:, k], cl, rtol=1.0e-10, atol=1.0e-10),
                "component %d max err %g" % (k, np.nanmax(np.abs(ll_ck[:, k] - cl))),
            )
        return cm, enc

    def check_em_parity(self, model, est, data, steps=2):
        """em_step (fused kernels) vs seq_estimate (legacy) parameter trajectories."""
        cm = CompiledMixture(model)
        enc = cm.encode(data)
        chunked = seq_encode(data, model=model)
        m_kernel = m_legacy = model
        for it in range(steps):
            m_kernel = cm.em_step(enc, est, model=m_kernel)
            m_legacy = seq_estimate(chunked, est, m_legacy)
            self._assert_close(dist_params(m_kernel), dist_params(m_legacy), "em step %d " % (it + 1))

    def check_suff_stat_roundtrip(self, dist, est, data, seed=7):
        """weighted_suff_stats -> legacy estimator.estimate vs legacy accumulator path."""
        cm = CompiledMixture(dist)
        enc = cm.encode(data)
        n = len(data)
        w = np.random.RandomState(seed).rand(n) + 0.05
        ss_kernel = cm.weighted_suff_stats(enc, w.reshape(-1, 1))
        fit_kernel = est.estimate(n, ss_kernel)

        acc = est.accumulator_factory().make()
        acc.seq_update(dist.dist_to_encoder().seq_encode(data), w, dist)
        fit_legacy = est.estimate(n, acc.value())
        self._assert_close(dist_params(fit_kernel), dist_params(fit_legacy), "roundtrip ")


class GammaKernelTest(KernelsExtBase):
    def setUp(self):
        comps = [GammaDistribution(2.0, 1.5), GammaDistribution(5.0, 0.6), GammaDistribution(1.0, 3.0)]
        self.model = MixtureDistribution(comps, [0.4, 0.35, 0.25])
        self.data = self.model.sampler(seed=11).sample(size=300)
        self.est = MixtureEstimator([GammaEstimator()] * 3)

    def test_standalone(self):
        self.check_seq_parity(self.model, self.data)
        self.check_em_parity(self.model, self.est, self.data)

    def test_in_composite(self):
        comps = [CompositeDistribution((GammaDistribution(2.0 + k, 1.0 + 0.5 * k), _cat(k))) for k in range(2)]
        model = MixtureDistribution(comps, [0.5, 0.5])
        data = model.sampler(seed=12).sample(size=300)
        est = MixtureEstimator([CompositeEstimator((GammaEstimator(), CategoricalEstimator()))] * 2)
        self.check_seq_parity(model, data)
        self.check_em_parity(model, est, data)

    def test_suff_stat_roundtrip(self):
        d = GammaDistribution(2.5, 1.2)
        self.check_suff_stat_roundtrip(d, GammaEstimator(), d.sampler(seed=13).sample(size=250))


class LogGaussianKernelTest(KernelsExtBase):
    def setUp(self):
        comps = [
            LogGaussianDistribution(0.0, 0.5),
            LogGaussianDistribution(1.5, 0.2),
            LogGaussianDistribution(-1.0, 1.0),
        ]
        self.model = MixtureDistribution(comps, [0.3, 0.4, 0.3])
        self.data = self.model.sampler(seed=21).sample(size=300)
        self.est = MixtureEstimator([LogGaussianEstimator()] * 3)

    def test_standalone(self):
        self.check_seq_parity(self.model, self.data)
        self.check_em_parity(self.model, self.est, self.data)

    def test_in_composite(self):
        comps = [CompositeDistribution((LogGaussianDistribution(0.5 * k, 0.4 + 0.2 * k), _cat(k))) for k in range(2)]
        model = MixtureDistribution(comps, [0.6, 0.4])
        data = model.sampler(seed=22).sample(size=300)
        est = MixtureEstimator([CompositeEstimator((LogGaussianEstimator(), CategoricalEstimator()))] * 2)
        self.check_seq_parity(model, data)
        self.check_em_parity(model, est, data)

    def test_suff_stat_roundtrip(self):
        d = LogGaussianDistribution(0.8, 0.3)
        self.check_suff_stat_roundtrip(d, LogGaussianEstimator(), d.sampler(seed=23).sample(size=250))


class BinomialKernelTest(KernelsExtBase):
    def setUp(self):
        comps = [BinomialDistribution(0.2, 12), BinomialDistribution(0.5, 12), BinomialDistribution(0.8, 12)]
        self.model = MixtureDistribution(comps, [0.35, 0.3, 0.35])
        self.data = self.model.sampler(seed=31).sample(size=400)
        self.est = MixtureEstimator([BinomialEstimator()] * 3)

    def test_standalone(self):
        self.check_seq_parity(self.model, self.data)
        # 3 steps: the first M-step re-derives (n, min_val) from data, exercising
        # the inline-lgamma fallback away from the precomputed coefficient column
        self.check_em_parity(self.model, self.est, self.data, steps=3)

    def test_min_val_shift_parity(self):
        comps = [BinomialDistribution(0.3 + 0.3 * k, 10, min_val=2) for k in range(2)]
        model = MixtureDistribution(comps, [0.5, 0.5])
        data = model.sampler(seed=32).sample(size=300)
        self.check_seq_parity(model, data)
        self.check_em_parity(model, MixtureEstimator([BinomialEstimator()] * 2), data)

    def test_in_composite(self):
        comps = [CompositeDistribution((BinomialDistribution(0.25 + 0.4 * k, 15), _cat(k))) for k in range(2)]
        model = MixtureDistribution(comps, [0.5, 0.5])
        data = model.sampler(seed=33).sample(size=300)
        est = MixtureEstimator([CompositeEstimator((BinomialEstimator(), CategoricalEstimator()))] * 2)
        self.check_seq_parity(model, data)
        self.check_em_parity(model, est, data)

    def test_suff_stat_roundtrip(self):
        d = BinomialDistribution(0.4, 9)
        self.check_suff_stat_roundtrip(d, BinomialEstimator(), d.sampler(seed=34).sample(size=250))


class DiagonalGaussianKernelTest(KernelsExtBase):
    def setUp(self):
        comps = [
            DiagonalGaussianDistribution([0.0, 1.0, -2.0], [1.0, 0.5, 2.0]),
            DiagonalGaussianDistribution([4.0, -3.0, 1.0], [0.7, 1.5, 0.4]),
            DiagonalGaussianDistribution([-4.0, 0.0, 5.0], [2.0, 1.0, 1.0]),
        ]
        self.model = MixtureDistribution(comps, [0.3, 0.3, 0.4])
        self.data = self.model.sampler(seed=41).sample(size=300)
        self.est = MixtureEstimator([DiagonalGaussianEstimator(dim=3)] * 3)

    def test_standalone(self):
        self.check_seq_parity(self.model, self.data)
        self.check_em_parity(self.model, self.est, self.data)

    def test_in_composite(self):
        comps = [
            CompositeDistribution((DiagonalGaussianDistribution([2.0 * k, -2.0 * k], [1.0, 0.5 + k]), _cat(k)))
            for k in range(2)
        ]
        model = MixtureDistribution(comps, [0.5, 0.5])
        data = model.sampler(seed=42).sample(size=300)
        est = MixtureEstimator([CompositeEstimator((DiagonalGaussianEstimator(dim=2), CategoricalEstimator()))] * 2)
        self.check_seq_parity(model, data)
        self.check_em_parity(model, est, data)

    def test_suff_stat_roundtrip(self):
        d = DiagonalGaussianDistribution([1.0, -1.0, 0.5], [0.8, 1.2, 0.5])
        self.check_suff_stat_roundtrip(d, DiagonalGaussianEstimator(dim=3), d.sampler(seed=43).sample(size=250))


class OptionalKernelTest(KernelsExtBase):
    def setUp(self):
        # ~30% missing Gaussians
        comps = [
            OptionalDistribution(GaussianDistribution(-3.0 + 3.0 * k, 1.0 + 0.5 * k), p=0.2 + 0.1 * k) for k in range(3)
        ]
        self.model = MixtureDistribution(comps, [0.4, 0.3, 0.3])
        self.data = self.model.sampler(seed=51).sample(size=400)
        self.est = MixtureEstimator([OptionalEstimator(GaussianEstimator(), est_prob=True)] * 3)

    def test_has_missing(self):
        n_missing = sum(1 for x in self.data if x is None)
        self.assertGreater(n_missing, 0.15 * len(self.data))
        self.assertLess(n_missing, 0.5 * len(self.data))

    def test_standalone(self):
        self.check_seq_parity(self.model, self.data)
        self.check_em_parity(self.model, self.est, self.data)

    def test_in_composite(self):
        comps = [
            CompositeDistribution((OptionalDistribution(GaussianDistribution(4.0 * k, 1.0), p=0.3), _cat(k)))
            for k in range(2)
        ]
        model = MixtureDistribution(comps, [0.5, 0.5])
        data = model.sampler(seed=52).sample(size=300)
        est = MixtureEstimator(
            [CompositeEstimator((OptionalEstimator(GaussianEstimator(), est_prob=True), CategoricalEstimator()))] * 2
        )
        self.check_seq_parity(model, data)
        self.check_em_parity(model, est, data)

    def test_no_p_parity(self):
        # degenerate legacy mode: no missing probability given
        comps = [OptionalDistribution(GaussianDistribution(-2.0 + 4.0 * k, 1.0)) for k in range(2)]
        model = MixtureDistribution(comps, [0.5, 0.5])
        data = list(model.components[0].dist.sampler(seed=53).sample(size=100))
        data[::7] = [None] * len(data[::7])
        self.check_seq_parity(model, data)

    def test_suff_stat_roundtrip(self):
        d = OptionalDistribution(GaussianDistribution(1.0, 2.0), p=0.3)
        data = d.sampler(seed=54).sample(size=250)
        self.check_suff_stat_roundtrip(d, OptionalEstimator(GaussianEstimator(), est_prob=True), data)


class IgnoredKernelTest(KernelsExtBase):
    def setUp(self):
        self.fixed = GaussianDistribution(1.0, 4.0)
        comps = [
            CompositeDistribution((GaussianDistribution(-5.0 + 5.0 * k, 1.0), IgnoredDistribution(self.fixed)))
            for k in range(3)
        ]
        self.model = MixtureDistribution(comps, [0.3, 0.3, 0.4])
        self.data = self.model.sampler(seed=61).sample(size=300)
        self.est = MixtureEstimator([CompositeEstimator((GaussianEstimator(), IgnoredEstimator(dist=self.fixed)))] * 3)

    def test_in_composite(self):
        self.check_seq_parity(self.model, self.data)
        self.check_em_parity(self.model, self.est, self.data, steps=2)

    def test_suff_stat_roundtrip(self):
        d = IgnoredDistribution(self.fixed)
        data = d.sampler(seed=62).sample(size=200)
        cm = CompiledMixture(d)
        enc = cm.encode(data)
        w = np.random.RandomState(63).rand(len(data)) + 0.05
        ss = cm.weighted_suff_stats(enc, w.reshape(-1, 1))
        self.assertIsNone(ss)  # IgnoredAccumulator.value() is None
        est = IgnoredEstimator(dist=self.fixed)
        fit = est.estimate(len(data), ss)
        self.assertEqual(str(fit.dist), str(self.fixed))

    def test_changed_wrapped_dist_rejected(self):
        cm = CompiledMixture(self.model)
        enc = cm.encode(self.data)
        other = MixtureDistribution(
            [
                CompositeDistribution(
                    (GaussianDistribution(-5.0 + 5.0 * k, 1.0), IgnoredDistribution(GaussianDistribution(9.0, 1.0)))
                )
                for k in range(3)
            ],
            [0.3, 0.3, 0.4],
        )
        with self.assertRaises(ValueError):
            cm.seq_log_density(enc, model=other)


class MixedExtCompositeTest(KernelsExtBase):
    """All new leaf families plus Optional/Ignored fused into one composite."""

    def setUp(self):
        self.fixed = GaussianDistribution(0.5, 2.0)
        comps = []
        for k in range(2):
            comps.append(
                CompositeDistribution(
                    (
                        GammaDistribution(2.0 + k, 1.0 + 0.5 * k),
                        LogGaussianDistribution(0.5 * k, 0.4),
                        BinomialDistribution(0.3 + 0.3 * k, 11),
                        DiagonalGaussianDistribution([2.0 * k, -2.0 * k], [1.0, 0.5 + k]),
                        OptionalDistribution(GaussianDistribution(3.0 * k, 1.0), p=0.3),
                        IgnoredDistribution(self.fixed),
                        _cat(k),
                    )
                )
            )
        self.model = MixtureDistribution(comps, [0.45, 0.55])
        self.data = self.model.sampler(seed=71).sample(size=300)
        comp_est = CompositeEstimator(
            (
                GammaEstimator(),
                LogGaussianEstimator(),
                BinomialEstimator(),
                DiagonalGaussianEstimator(dim=2),
                OptionalEstimator(GaussianEstimator(), est_prob=True),
                IgnoredEstimator(dist=self.fixed),
                CategoricalEstimator(),
            )
        )
        self.est = MixtureEstimator([comp_est] * 2)

    def test_seq_and_em_parity(self):
        self.check_seq_parity(self.model, self.data)
        self.check_em_parity(self.model, self.est, self.data, steps=2)


if __name__ == "__main__":
    unittest.main()
