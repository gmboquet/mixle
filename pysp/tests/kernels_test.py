"""Tests for the fused numba kernel estimation path (pysp.stats.kernels).

Every result is checked against the legacy vectorized seq_* path: component
log-densities, mixture log-densities, posteriors, and full EM trajectories
must agree to floating-point tolerance.
"""

import time
import unittest

import numpy as np

from pysp.stats import (
    CategoricalDistribution,
    CategoricalEstimator,
    CompositeDistribution,
    CompositeEstimator,
    ExponentialDistribution,
    GaussianDistribution,
    GaussianEstimator,
    GeometricDistribution,
    IntegerCategoricalDistribution,
    IntegerCategoricalEstimator,
    MixtureDistribution,
    MixtureEstimator,
    PoissonDistribution,
    PoissonEstimator,
    SequenceDistribution,
    SequenceEstimator,
    seq_encode,
    seq_estimate,
)
from pysp.stats.kernels import CompiledMixture, build_kernel


def make_mixture(K=3):
    comps = []
    for k in range(K):
        mu = -10.0 + 10.0 * k
        cat = {"a": 0.7 - 0.2 * k, "b": 0.2, "c": 0.1 + 0.2 * k}
        ip = np.roll([0.6, 0.2, 0.1, 0.1], k)
        comps.append(
            CompositeDistribution(
                (
                    GaussianDistribution(mu, 1.0 + 0.5 * k),
                    CategoricalDistribution(cat),
                    PoissonDistribution(3.0 + 4.0 * k),
                    IntegerCategoricalDistribution(0, list(ip)),
                    SequenceDistribution(
                        IntegerCategoricalDistribution(0, list(np.roll([0.5, 0.3, 0.1, 0.1], k))),
                        len_dist=CategoricalDistribution({2: 0.5, 3: 0.3, 4: 0.2}),
                    ),
                )
            )
        )
    return MixtureDistribution(comps, [1.0 / K] * K)


def make_estimator(K=3):
    comp = CompositeEstimator(
        (
            GaussianEstimator(),
            CategoricalEstimator(),
            PoissonEstimator(),
            IntegerCategoricalEstimator(min_val=0, max_val=3),
            SequenceEstimator(IntegerCategoricalEstimator(min_val=0, max_val=3), len_estimator=CategoricalEstimator()),
        )
    )
    return MixtureEstimator([comp] * K)


class KernelsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = make_mixture()
        cls.data = cls.model.sampler(seed=1).sample(size=400)
        cls.compiled = CompiledMixture(cls.model)
        cls.enc = cls.compiled.encode(cls.data)
        cls.legacy_enc = cls.model.dist_to_encoder().seq_encode(cls.data)

    # -- scoring parity ---------------------------------------------------

    def test_component_log_density_parity(self):
        ll_k = self.compiled.seq_component_log_density(self.enc)
        for k, comp in enumerate(self.model.components):
            comp_enc = comp.dist_to_encoder().seq_encode(self.data)
            ll_legacy = comp.seq_log_density(comp_enc)
            self.assertTrue(
                np.allclose(ll_k[:, k], ll_legacy, atol=1.0e-10),
                "component %d max err %g" % (k, np.abs(ll_k[:, k] - ll_legacy).max()),
            )

    def test_mixture_log_density_parity(self):
        ll = self.compiled.seq_log_density(self.enc)
        ll_legacy = self.model.seq_log_density(self.legacy_enc)
        self.assertTrue(np.allclose(ll, ll_legacy, atol=1.0e-10), "max err %g" % np.abs(ll - ll_legacy).max())

    def test_posterior_parity(self):
        gam = self.compiled.posteriors(self.enc)
        gam_legacy = self.model.seq_posterior(self.legacy_enc)
        self.assertTrue(np.allclose(gam, gam_legacy, atol=1.0e-10))

    def test_single_distribution_parity(self):
        comp = self.model.components[0]
        cm = CompiledMixture(comp)
        enc = cm.encode(self.data)
        ll = cm.seq_log_density(enc)
        ll_legacy = comp.seq_log_density(comp.dist_to_encoder().seq_encode(self.data))
        self.assertTrue(np.allclose(ll, ll_legacy, atol=1.0e-10))

    def test_exponential_geometric_parity(self):
        comps = [
            CompositeDistribution((ExponentialDistribution(1.0 + k), GeometricDistribution(0.2 + 0.3 * k)))
            for k in range(2)
        ]
        model = MixtureDistribution(comps, [0.5, 0.5])
        data = model.sampler(seed=2).sample(size=300)
        cm = CompiledMixture(model)
        ll = cm.seq_log_density(cm.encode(data))
        ll_legacy = model.seq_log_density(model.dist_to_encoder().seq_encode(data))
        self.assertTrue(np.allclose(ll, ll_legacy, atol=1.0e-10))

    # -- estimation parity --------------------------------------------------

    def test_em_trajectory_matches_legacy(self):
        est = make_estimator()
        chunked = seq_encode(self.data, model=self.model)

        m_kernel = self.model
        m_legacy = self.model
        for _ in range(3):
            m_kernel = self.compiled.em_step(self.enc, est, model=m_kernel)
            m_legacy = seq_estimate(chunked, est, m_legacy)

            ll_k = self.compiled.seq_log_density(self.enc, model=m_kernel)
            ll_l = m_legacy.seq_log_density(self.legacy_enc)
            self.assertTrue(
                np.allclose(ll_k, ll_l, atol=1.0e-8), "EM step diverged: max err %g" % np.abs(ll_k - ll_l).max()
            )
            self.assertTrue(
                np.allclose(np.asarray(m_kernel.w, dtype=float), np.asarray(m_legacy.w, dtype=float), atol=1.0e-10)
            )

    def test_fit_converges(self):
        est = make_estimator()
        model, ll = self.compiled.fit(self.enc, est, max_its=60, delta=1.0e-7, rng=np.random.RandomState(5), init_p=1.0)
        ll0 = self.compiled.seq_log_density(self.enc, model=self.model).sum()
        self.assertTrue(np.isfinite(ll))
        # fit from random init should reach at least near the truth's likelihood
        self.assertGreater(ll, ll0 - 0.05 * abs(ll0))

    def test_weighted_em_step(self):
        est = make_estimator()
        w = np.random.RandomState(0).rand(len(self.data))
        m = self.compiled.em_step(self.enc, est, weights=w)
        ll = self.compiled.seq_log_density(self.enc, model=m)
        self.assertTrue(np.all(np.isfinite(ll)))

    # -- guards ------------------------------------------------------------

    def test_mixed_structure_rejected(self):
        with self.assertRaises(ValueError):
            build_kernel([GaussianDistribution(0.0, 1.0), PoissonDistribution(2.0)])

    def test_unsupported_type_rejected(self):
        from pysp.stats import DirichletDistribution

        with self.assertRaises(ValueError):
            build_kernel([DirichletDistribution([1.0, 2.0])])


class KernelsBenchmark(unittest.TestCase):
    def test_benchmark_vs_legacy(self):
        model = make_mixture()
        data = model.sampler(seed=3).sample(size=20000)

        cm = CompiledMixture(model)
        t0 = time.time()
        enc = cm.encode(data)
        t_enc_k = time.time() - t0
        t0 = time.time()
        legacy_enc = model.dist_to_encoder().seq_encode(data)
        t_enc_l = time.time() - t0

        cm.seq_log_density(enc)  # warm-up compile
        t0 = time.time()
        ll_k = cm.seq_log_density(enc)
        t_k = time.time() - t0
        t0 = time.time()
        ll_l = model.seq_log_density(legacy_enc)
        t_l = time.time() - t0
        self.assertTrue(np.allclose(ll_k, ll_l, atol=1.0e-9))

        est = make_estimator()
        chunked = [(len(data), legacy_enc)]
        cm.em_step(enc, est)  # warm-up
        t0 = time.time()
        cm.em_step(enc, est)
        t_em_k = time.time() - t0
        t0 = time.time()
        seq_estimate(chunked, est, model)
        t_em_l = time.time() - t0

        print("\n[kernels benchmark n=%d, K=3, 5-field composite w/ sequences]" % len(data))
        print("  encode:          kernel %.3fs   legacy %.3fs" % (t_enc_k, t_enc_l))
        print("  seq_log_density: kernel %.4fs   legacy %.4fs   (%.1fx)" % (t_k, t_l, t_l / max(t_k, 1e-9)))
        print("  em_step:         kernel %.4fs   legacy %.4fs   (%.1fx)" % (t_em_k, t_em_l, t_em_l / max(t_em_k, 1e-9)))

    def test_benchmark_sequence_heavy(self):
        # topic-model-like data: mixture of token sequences (~30 tokens/doc).
        # this is where kernel fusion wins decisively over the K-pass legacy path
        K, V, n = 5, 50, 20000
        rng = np.random.RandomState(0)
        comps = [
            SequenceDistribution(
                IntegerCategoricalDistribution(0, list(rng.dirichlet(np.ones(V) * 0.3))),
                len_dist=CategoricalDistribution({20: 0.3, 30: 0.4, 40: 0.3}),
            )
            for _ in range(K)
        ]
        model = MixtureDistribution(comps, [1.0 / K] * K)
        data = model.sampler(seed=1).sample(size=n)
        est = MixtureEstimator(
            [
                SequenceEstimator(
                    IntegerCategoricalEstimator(min_val=0, max_val=V - 1), len_estimator=CategoricalEstimator()
                )
            ]
            * K
        )

        cm = CompiledMixture(model)
        enc = cm.encode(data)
        lenc = model.dist_to_encoder().seq_encode(data)
        cm.seq_log_density(enc)
        cm.em_step(enc, est)  # warm-up

        t0 = time.time()
        ll_k = cm.seq_log_density(enc)
        t_k = time.time() - t0
        t0 = time.time()
        ll_l = model.seq_log_density(lenc)
        t_l = time.time() - t0
        self.assertTrue(np.allclose(ll_k, ll_l, atol=1.0e-9))
        t0 = time.time()
        cm.em_step(enc, est)
        t_em_k = time.time() - t0
        t0 = time.time()
        seq_estimate([(n, lenc)], est, model)
        t_em_l = time.time() - t0

        print("\n[kernels benchmark: sequence-heavy n=%d K=%d ~30 tokens/doc]" % (n, K))
        print("  seq_log_density: kernel %.4fs   legacy %.4fs   (%.1fx)" % (t_k, t_l, t_l / max(t_k, 1e-9)))
        print("  em_step:         kernel %.4fs   legacy %.4fs   (%.1fx)" % (t_em_k, t_em_l, t_em_l / max(t_em_k, 1e-9)))


if __name__ == "__main__":
    unittest.main()
