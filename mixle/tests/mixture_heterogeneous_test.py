"""Heterogeneous-family MixtureDistribution support.

A ``MixtureDistribution`` whose components belong to *different* distribution families (e.g. a
Gaussian and a Gamma) needs each component fed the sequence encoding its own family expects. The
mixture encoder detects this and carries per-component encodings; homogeneous mixtures still encode
once and share, bit-identically to the legacy single-encoder path.

These tests pin both halves:
  * the heterogeneous EM recovers a Gaussian + Gamma mixture, and density/posterior/encoding all
    work end to end;
  * a same-family (two-Gaussian) mixture is byte-identical to encoding with a single component
    encoder, so the fast path is untouched.
"""

import io
import unittest

import numpy as np

from mixle.inference.estimation import optimize
from mixle.stats import (
    GammaDistribution,
    GammaEstimator,
    GaussianDistribution,
    GaussianEstimator,
    MixtureDistribution,
    seq_encode,
    seq_log_density_sum,
)
from mixle.stats.compute.sequence import seq_estimate
from mixle.stats.latent.mixture import (
    MixtureEstimator,
    _HeteroMixtureEncoded,
    _SharedMixtureEncoded,
)


class HeterogeneousMixtureTestCase(unittest.TestCase):
    def _gauss_gamma_data(self, n=2000, seed=2):
        rng = np.random.RandomState(seed)
        z = rng.rand(n) < 0.6
        # well-separated so the modes are identifiable: Gaussian(8, 1) vs Gamma(k=2, theta=1).
        return np.where(z, rng.normal(8.0, 1.0, n), rng.gamma(2.0, 1.0, n)).tolist()

    def test_heterogeneous_em_recovers_components(self):
        data = self._gauss_gamma_data()
        est = MixtureEstimator([GaussianEstimator(), GammaEstimator()])
        model = optimize(data, est, max_its=60, rng=np.random.RandomState(3), out=io.StringIO())

        gauss, gamma = model.components
        # component families are preserved and parameters land near truth
        self.assertIsInstance(gauss, GaussianDistribution)
        self.assertIsInstance(gamma, GammaDistribution)
        self.assertAlmostEqual(gauss.mu, 8.0, delta=0.4)
        self.assertAlmostEqual(gauss.sigma2, 1.0, delta=0.4)
        self.assertAlmostEqual(gamma.k, 2.0, delta=0.6)
        self.assertAlmostEqual(gamma.theta, 1.0, delta=0.5)
        self.assertAlmostEqual(model.w[0], 0.6, delta=0.1)

    def test_heterogeneous_encoder_roundtrip(self):
        data = self._gauss_gamma_data(n=200)
        model = MixtureDistribution([GaussianDistribution(8.0, 1.0), GammaDistribution(2.0, 1.0)], [0.6, 0.4])
        enc = model.dist_to_encoder()
        self.assertFalse(enc.homogeneous)

        encoded = enc.seq_encode(data)
        self.assertIsInstance(encoded, _HeteroMixtureEncoded)
        self.assertEqual(len(encoded.encodings), 2)

        # vectorized density matches the scalar per-observation log_density
        seq_ll = model.seq_log_density(encoded)
        scalar_ll = np.array([model.log_density(x) for x in data])
        np.testing.assert_allclose(seq_ll, scalar_ll, rtol=1e-9, atol=1e-9)

        # posterior responsibilities are valid probability rows
        post = model.seq_posterior(encoded)
        self.assertEqual(post.shape, (len(data), 2))
        np.testing.assert_allclose(post.sum(axis=1), np.ones(len(data)), rtol=1e-9)

    def test_homogeneous_encoding_unchanged(self):
        rng = np.random.RandomState(5)
        data = (rng.randn(300) + np.where(rng.rand(300) < 0.5, 0.0, 6.0)).tolist()
        model = MixtureDistribution([GaussianDistribution(0.0, 1.0), GaussianDistribution(6.0, 1.0)], [0.5, 0.5])
        enc = model.dist_to_encoder()
        self.assertTrue(enc.homogeneous)

        encoded = enc.seq_encode(data)
        # same-family mixture must encode exactly as a single shared component encoder would
        single = GaussianDistribution(0.0, 1.0).dist_to_encoder().seq_encode(data)
        self.assertNotIsInstance(encoded, _HeteroMixtureEncoded)
        np.testing.assert_array_equal(np.asarray(encoded), np.asarray(single))

    def test_homogeneous_em_matches_single_encoder_path(self):
        # The fast homogeneous path should still optimize normally (sanity that the per-component
        # branch did not perturb the common case).
        rng = np.random.RandomState(11)
        data = (rng.randn(400) + np.where(rng.rand(400) < 0.5, 0.0, 6.0)).tolist()
        est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        model = optimize(data, est, max_its=30, rng=np.random.RandomState(13), out=io.StringIO())
        _, ll = seq_log_density_sum(seq_encode(data, model=model), model)
        self.assertTrue(np.isfinite(ll))
        self.assertTrue(model.dist_to_encoder().homogeneous)

    def test_nested_homogeneous_mixture_preserves_heterogeneous_encoding_depth(self):
        rng = np.random.RandomState(29)
        data = rng.gamma(3.0, 1.2, 300).tolist()
        outer = MixtureDistribution(
            [
                MixtureDistribution(
                    [GaussianDistribution(float(center), 1.5), GammaDistribution(2.0, 1.0)],
                    [0.6, 0.4],
                )
                for center in (1.0, 2.0, 3.0, 4.0)
            ],
            [0.25] * 4,
        )
        encoder = outer.dist_to_encoder()
        self.assertTrue(encoder.homogeneous)
        encoded = encoder.seq_encode(data)
        self.assertIsInstance(encoded, _SharedMixtureEncoded)

        seq_values = outer.seq_log_density(encoded)
        scalar_values = np.asarray([outer.log_density(value) for value in data])
        np.testing.assert_allclose(seq_values, scalar_values, rtol=1e-11, atol=1e-11)
        candidate = seq_estimate([(len(data), encoded)], outer.estimator(), outer)
        self.assertTrue(np.isfinite(candidate.seq_log_density(encoded)).all())


if __name__ == "__main__":
    unittest.main()
