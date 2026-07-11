"""Tests for mixle.stats.latent.gaussian_mixture (GaussianMixtureDistribution and friends).

Covers: sample -> estimate round-trip on a small 2-component 2-d problem, agreement between
scalar and vectorized (seq_) updates, and sampler output shapes. Also smoke-tests the
docstring-pass modules mvn, dmvn, and dirichlet via sample/log-density round trips.
"""

import unittest

import numpy as np
from numpy.random import RandomState

from mixle.stats.bayes.dirichlet import DirichletDistribution
from mixle.stats.latent.gaussian_mixture import (
    GaussianMixtureDataEncoder,
    GaussianMixtureDistribution,
    GaussianMixtureEstimator,
)
from mixle.stats.multivariate.diagonal_gaussian import DiagonalGaussianDistribution
from mixle.stats.multivariate.multivariate_gaussian import (
    MultivariateGaussianDistribution,
    MultivariateGaussianEstimator,
)


def make_dist():
    mu = [[-3.0, -3.0], [3.0, 3.0]]
    sig2 = [[[1.0, 0.3], [0.3, 1.0]], [[0.5, -0.1], [-0.1, 0.8]]]
    w = [0.4, 0.6]
    return GaussianMixtureDistribution(mu, sig2, w, name="gm")


class GaussianMixtureTestCase(unittest.TestCase):
    def test_sampler_shapes(self):
        dist = make_dist()
        sampler = dist.sampler(seed=1)

        single = sampler.sample()
        self.assertIsInstance(single, np.ndarray)
        self.assertEqual(single.shape, (2,))

        many = sampler.sample(size=25)
        self.assertIsInstance(many, np.ndarray)
        self.assertEqual(many.shape, (25, 2))

    def test_log_density_matches_manual(self):
        dist = make_dist()
        x = np.array([0.5, -0.25])

        comp_ll = np.array([c.log_density(x) for c in dist.components])
        manual = np.log(np.dot(np.exp(comp_ll), dist.w))

        self.assertAlmostEqual(dist.log_density(x), manual, places=10)

    def test_seq_log_density_matches_scalar(self):
        dist = make_dist()
        data = dist.sampler(seed=2).sample(size=50)

        enc = dist.dist_to_encoder().seq_encode(data)
        seq_ll = dist.seq_log_density(enc)
        scalar_ll = np.array([dist.log_density(u) for u in data])

        self.assertEqual(len(seq_ll), 50)
        self.assertTrue(np.allclose(seq_ll, scalar_ll))

    def test_seq_posterior_matches_scalar(self):
        dist = make_dist()
        data = dist.sampler(seed=3).sample(size=40)

        enc = dist.dist_to_encoder().seq_encode(data)
        seq_post = dist.seq_posterior(enc)
        scalar_post = np.array([dist.posterior(u) for u in data])

        self.assertEqual(seq_post.shape, (40, 2))
        self.assertTrue(np.allclose(seq_post, scalar_post))
        self.assertTrue(np.allclose(seq_post.sum(axis=1), 1.0))

    def test_update_and_seq_update_agree(self):
        dist = make_dist()
        data = dist.sampler(seed=4).sample(size=30)
        est = dist.estimator()

        acc_scalar = est.accumulator_factory().make()
        for u in data:
            acc_scalar.update(u, 1.0, dist)

        acc_seq = est.accumulator_factory().make()
        enc = acc_seq.acc_to_encoder().seq_encode(data)
        acc_seq.seq_update(enc, np.ones(len(data)), dist)

        v1 = acc_scalar.value()
        v2 = acc_seq.value()

        self.assertTrue(np.allclose(v1[0], v2[0]))
        for c1, c2 in zip(v1[1], v2[1]):
            self.assertTrue(np.allclose(c1[0], c2[0]))
            self.assertTrue(np.allclose(c1[1], c2[1]))
            self.assertAlmostEqual(c1[2], c2[2], places=8)

    def test_sample_estimate_round_trip(self):
        dist = make_dist()
        data = dist.sampler(seed=5).sample(size=500)

        est = GaussianMixtureEstimator([MultivariateGaussianEstimator(), MultivariateGaussianEstimator()])

        # Start EM from a rough (perturbed) model and run a few seq_update/estimate steps.
        model = GaussianMixtureDistribution([[-1.0, -1.0], [1.0, 1.0]], [4.0 * np.eye(2), 4.0 * np.eye(2)], [0.5, 0.5])
        enc = est.accumulator_factory().make().acc_to_encoder().seq_encode(data)

        for _ in range(10):
            acc = est.accumulator_factory().make()
            acc.seq_update(enc, np.ones(len(data)), model)
            model = est.estimate(None, acc.value())

        # Sort components by first mean coordinate to align with the truth.
        order = np.argsort(model.mu[:, 0])
        w_hat = model.w[order]
        mu_hat = model.mu[order]
        sig2_hat = model.sig2[order]

        self.assertTrue(np.allclose(w_hat, [0.4, 0.6], atol=0.1))
        self.assertTrue(np.allclose(mu_hat, [[-3.0, -3.0], [3.0, 3.0]], atol=0.35))
        self.assertTrue(np.allclose(sig2_hat[0], [[1.0, 0.3], [0.3, 1.0]], atol=0.35))
        self.assertTrue(np.allclose(sig2_hat[1], [[0.5, -0.1], [-0.1, 0.8]], atol=0.35))

        # Likelihood of the fit should be close to the truth's.
        ll_fit = model.seq_log_density(enc).mean()
        ll_true = dist.seq_log_density(enc).mean()
        self.assertLess(abs(ll_fit - ll_true), 0.1)

    def test_seq_initialize_produces_valid_model(self):
        dist = make_dist()
        data = dist.sampler(seed=9).sample(size=200)
        est = dist.estimator()

        rng = RandomState(10)
        acc = est.accumulator_factory().make()
        enc = acc.acc_to_encoder().seq_encode(data)
        acc.seq_initialize(enc, np.ones(len(data)), rng)
        model = est.estimate(None, acc.value())

        self.assertAlmostEqual(model.w.sum(), 1.0, places=10)
        self.assertEqual(model.mu.shape, (2, 2))
        self.assertEqual(model.sig2.shape, (2, 2, 2))
        self.assertTrue(np.all(np.isfinite(model.seq_log_density(enc))))

    def test_initialize_and_seq_initialize_agree(self):
        dist = make_dist()
        data = dist.sampler(seed=11).sample(size=20)
        est = dist.estimator()

        acc_scalar = est.accumulator_factory().make()
        for u in data:
            acc_scalar.initialize(u, 1.0, RandomState(12))

        acc_seq = est.accumulator_factory().make()
        enc = acc_seq.acc_to_encoder().seq_encode(data)
        acc_seq.seq_initialize(enc, np.ones(len(data)), RandomState(12))

        v1 = acc_scalar.value()
        v2 = acc_seq.value()

        self.assertTrue(np.allclose(v1[0], v2[0]))
        for c1, c2 in zip(v1[1], v2[1]):
            self.assertTrue(np.allclose(c1[0], c2[0]))
            self.assertTrue(np.allclose(c1[1], c2[1]))
            self.assertAlmostEqual(c1[2], c2[2], places=8)

    def test_diagonal_sig2_input(self):
        # A (K, d) sig2 arg is interpreted as per-component diagonal variances.
        dist = GaussianMixtureDistribution([[0.0, 0.0], [5.0, 5.0]], [[1.0, 2.0], [0.5, 0.25]], [0.5, 0.5])
        self.assertEqual(dist.sig2.shape, (2, 2, 2))
        self.assertTrue(np.allclose(dist.sig2[0], [[1.0, 0.0], [0.0, 2.0]]))
        self.assertTrue(np.allclose(dist.sig2[1], [[0.5, 0.0], [0.0, 0.25]]))

        x = np.array([0.1, -0.2])
        comp_ll = np.array([c.log_density(x) for c in dist.components])
        manual = np.log(np.dot(np.exp(comp_ll), dist.w))
        self.assertAlmostEqual(dist.log_density(x), manual, places=10)

    def test_str_eval_round_trip(self):
        dist = make_dist()
        dist2 = eval(str(dist))
        self.assertTrue(np.allclose(dist.mu, dist2.mu))
        self.assertTrue(np.allclose(dist.sig2, dist2.sig2))
        self.assertTrue(np.allclose(dist.w, dist2.w))
        self.assertEqual(dist.name, dist2.name)

    def test_estimator_from_distribution(self):
        dist = make_dist()
        est = dist.estimator()
        self.assertIsInstance(est, GaussianMixtureEstimator)
        self.assertEqual(est.num_components, 2)

        # Legacy camelCase alias still works (and warns: it is deprecated).
        fac1 = est.accumulator_factory()
        with self.assertWarns(DeprecationWarning):
            fac2 = est.accumulatorFactory()
        self.assertEqual(type(fac1), type(fac2))

    def test_encoder_eq_and_str(self):
        dist = make_dist()
        enc1 = dist.dist_to_encoder()
        acc = dist.estimator().accumulator_factory().make()
        enc3 = acc.acc_to_encoder()

        self.assertTrue(isinstance(str(enc1), str))
        self.assertEqual(enc1, dist.dist_to_encoder())
        self.assertIsInstance(enc3, GaussianMixtureDataEncoder)
        self.assertNotEqual(enc1, "not an encoder")

        data = dist.sampler(seed=13).sample(size=5)
        self.assertEqual(enc1.seq_encode(data).shape, (5, 2))
        self.assertEqual(GaussianMixtureDataEncoder().seq_encode(data).shape, (5, 2))


class DocstringModuleSmokeTestCase(unittest.TestCase):
    """Light behavioral checks that mvn, dmvn, and dirichlet still work."""

    def test_mvn_round_trip(self):
        dist = MultivariateGaussianDistribution([1.0, -1.0], [[2.0, 0.4], [0.4, 1.0]])
        data = dist.sampler(seed=6).sample(size=400)

        est = dist.estimator()
        acc = est.accumulator_factory().make()
        enc = acc.acc_to_encoder().seq_encode(data)
        acc.seq_update(enc, np.ones(len(data)), None)
        fit = est.estimate(None, acc.value())

        self.assertTrue(np.allclose(fit.mu, dist.mu, atol=0.25))
        self.assertTrue(np.allclose(fit.covar, dist.covar, atol=0.35))

        x = np.array([0.0, 0.0])
        seq_ll = dist.seq_log_density(enc)
        self.assertAlmostEqual(dist.log_density(data[0]), seq_ll[0], places=8)
        self.assertTrue(np.isfinite(dist.log_density(x)))

    def test_dmvn_round_trip(self):
        dist = DiagonalGaussianDistribution([0.5, -0.5, 2.0], [1.0, 0.5, 2.0])
        data = np.asarray(dist.sampler(seed=7).sample(size=400))

        est = dist.estimator()
        acc = est.accumulator_factory().make()
        enc = acc.acc_to_encoder().seq_encode(data)
        acc.seq_update(enc, np.ones(len(data)), None)
        fit = est.estimate(None, acc.value())

        self.assertTrue(np.allclose(fit.mu, dist.mu, atol=0.25))
        self.assertTrue(np.allclose(fit.covar, dist.covar, atol=0.4))

        seq_ll = dist.seq_log_density(enc)
        self.assertAlmostEqual(dist.log_density(data[0]), seq_ll[0], places=8)

    def test_dirichlet_round_trip(self):
        dist = DirichletDistribution([2.0, 3.0, 5.0])
        data = dist.sampler(seed=8).sample(size=400)

        est = dist.estimator()
        acc = est.accumulator_factory().make()
        enc = acc.acc_to_encoder().seq_encode(data)
        acc.seq_update(enc, np.ones(len(data)), None)
        fit = est.estimate(None, acc.value())

        self.assertTrue(np.allclose(fit.alpha, dist.alpha, rtol=0.25))

        seq_ll = dist.seq_log_density(enc)
        self.assertAlmostEqual(dist.log_density(data[0]), seq_ll[0], places=8)


if __name__ == "__main__":
    unittest.main()
