"""Compositional data analysis: Aitchison logratio transforms + logratio-normal distribution (Phase 6)."""

import unittest

import numpy as np
from scipy.integrate import quad

from mixle.inference import estimate, optimize
from mixle.stats.multivariate.composition import AitchisonNormalDistribution as AitchisonNormal
from mixle.stats.multivariate.composition import closure, clr, clr_inv, ilr, ilr_basis, ilr_inv


class LogratioTransformTest(unittest.TestCase):
    def setUp(self):
        self.x = closure(np.random.RandomState(0).gamma(2.0, size=(50, 4)))

    def test_ilr_round_trip(self):
        np.testing.assert_allclose(ilr_inv(ilr(self.x)), self.x, atol=1e-12)

    def test_clr_round_trip(self):
        np.testing.assert_allclose(clr_inv(clr(self.x)), self.x, atol=1e-12)

    def test_ilr_basis_is_orthonormal(self):
        v = ilr_basis(5)
        np.testing.assert_allclose(v.T @ v, np.eye(4), atol=1e-12)

    def test_ilr_is_isometric(self):
        a, b = self.x[0:1], self.x[1:2]
        aitchison = np.linalg.norm(clr(a) - clr(b))  # Aitchison distance on the simplex
        euclidean = np.linalg.norm(ilr(a) - ilr(b))  # Euclidean distance in ilr space
        self.assertAlmostEqual(aitchison, euclidean, places=10)

    def test_closure_projects_to_simplex(self):
        c = closure(np.array([[2.0, 3.0, 5.0]]))
        np.testing.assert_allclose(c.sum(axis=1), 1.0)

    def test_transform_domains_and_basis_are_strict(self):
        invalid_compositions = (
            [0.0, 1.0],
            [-1.0, 2.0],
            [np.nan, 1.0],
            [np.inf, 1.0],
            [],
        )
        for value in invalid_compositions:
            for transform in (closure, clr, ilr):
                with self.subTest(value=value, transform=transform.__name__), self.assertRaises(ValueError):
                    transform(value)
        for total in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(total=total), self.assertRaises(ValueError):
                closure([1.0, 1.0], total=total)
        with self.assertRaises(ValueError):
            ilr([1.0, 1.0, 1.0], basis=np.eye(3))
        with self.assertRaises(ValueError):
            ilr([1.0, 1.0, 1.0], basis=np.ones((3, 2)))
        with self.assertRaises(ValueError):
            ilr_inv([0.0, 1.0], basis=ilr_basis(4))
        with self.assertRaises((TypeError, ValueError)):
            ilr_basis(3.5)


class AitchisonNormalTest(unittest.TestCase):
    def setUp(self):
        self.true = AitchisonNormal(mean=np.array([0.5, -1.0, 0.3]), cov=np.diag([0.4, 0.6, 0.5]))

    def test_samples_lie_on_the_simplex(self):
        s = self.true.sampler(seed=1).sample(5000)
        np.testing.assert_allclose(s.sum(axis=1), 1.0, atol=1e-12)
        self.assertTrue((s > 0).all())
        self.assertEqual(s.shape[1], 4)  # D-1=3 ilr coords -> D=4 parts

    def test_estimate_recovers_parameters(self):
        s = self.true.sampler(seed=2).sample(20000)
        fit = estimate([row for row in s], self.true.estimator())  # the mixle estimator/accumulator contract
        self.assertIsInstance(fit, AitchisonNormal)
        np.testing.assert_allclose(fit.mean, self.true.mean, atol=0.04)
        np.testing.assert_allclose(fit.cov, self.true.cov, atol=0.06)

    def test_optimize_works_despite_the_accumulator_not_declaring_its_base_class(self):
        # AitchisonNormalAccumulator/DataEncoder didn't inherit SequenceEncodableStatisticAccumulator/
        # DataSequenceEncoder despite fully implementing their interface -- optimize() (unlike the
        # lower-level estimate() the other test above uses) raised AttributeError on the resulting
        # model, since machinery elsewhere assumes every accumulator has the base class's key_merge/
        # key_replace (a concrete default, not something this class needed to reimplement).
        s = self.true.sampler(seed=4).sample(2000)
        fit = optimize([row for row in s], self.true.estimator())
        self.assertIsInstance(fit, AitchisonNormal)
        np.testing.assert_allclose(fit.mean, self.true.mean, atol=0.1)

    def test_log_density_peaks_at_the_center(self):
        center = self.true.mean_composition()
        edge = closure(np.array([0.9, 0.05, 0.03, 0.02]))[0]
        self.assertGreater(self.true.log_density(center), self.true.log_density(edge))

    def test_mean_composition_is_on_the_simplex(self):
        self.assertAlmostEqual(self.true.mean_composition().sum(), 1.0, places=10)

    def test_seq_log_density_matches_scalar(self):
        s = self.true.sampler(seed=3).sample(4)
        enc = self.true.dist_to_encoder().seq_encode([s[i] for i in range(4)])
        batch = self.true.seq_log_density(enc)
        self.assertEqual(batch.shape, (4,))
        self.assertAlmostEqual(self.true.log_density(s[0]), batch[0], places=10)

    def test_two_part_density_integrates_to_one_in_ordinary_coordinates(self):
        distribution = AitchisonNormal(mean=np.array([0.0]), cov=np.array([[1.0]]))
        integral, error = quad(
            lambda first: distribution.density(np.array([first, 1.0 - first])),
            0.0,
            1.0,
            epsabs=1.0e-10,
        )
        self.assertLess(error, 1.0e-8)
        self.assertAlmostEqual(integral, 1.0, places=8)

    def test_distribution_and_encoder_enforce_simplex_support_and_width(self):
        encoder = self.true.dist_to_encoder()
        invalid = (
            [0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.20],
            [0.25, 0.25, 0.25, 0.0],
            [0.25, 0.25, 0.25, np.nan],
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.true.log_density(value)
            with self.subTest(encoded=value), self.assertRaises(ValueError):
                encoder.seq_encode([value])
        self.assertEqual(encoder, self.true.dist_to_encoder())
        self.assertIn("n_parts=4", str(encoder))

    def test_smoothing_and_keyed_pooling_are_propagated(self):
        keyed = AitchisonNormal(
            mean=np.array([0.5, -1.0, 0.3]),
            cov=np.diag([0.4, 0.6, 0.5]),
            keys="composition",
        )
        estimator = keyed.estimator(pseudo_count=2.0)
        self.assertEqual(estimator.pseudo_count, 2.0)
        self.assertEqual(estimator.gaussian_estimator.pseudo_count, (2.0, 2.0))
        first = estimator.accumulator_factory().make()
        second = estimator.accumulator_factory().make()
        self.assertEqual(first.keys, "composition")
        self.assertEqual(first.gaussian_acc.keys, "composition")
        first.update(keyed.mean_composition(), 1.0, None)
        second.update(keyed.sampler(seed=8).sample(), 1.0, None)
        pooled = {}
        first.key_merge(pooled)
        second.key_merge(pooled)
        first.key_replace(pooled)
        second.key_replace(pooled)
        self.assertEqual(first.value()[2], 2.0)
        self.assertEqual(second.value()[2], 2.0)

    def test_encoded_jacobian_alignment_is_checked(self):
        encoded = self.true.dist_to_encoder().seq_encode(self.true.sampler(seed=9).sample(2))
        with self.assertRaises(ValueError):
            self.true.seq_log_density((encoded[0], encoded[1][:1]))


if __name__ == "__main__":
    unittest.main()
