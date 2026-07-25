"""Regression tests for PPL parameter and composite-structure contracts."""

import unittest

import numpy as np

from mixle.ppl import (
    LDA,
    MVN,
    Bernoulli,
    Binomial,
    Categorical,
    DiagGaussian,
    Dirichlet,
    Gamma,
    Markov,
    Mix,
    Normal,
    SemiMix,
    Uniform,
    free,
)
from mixle.ppl.core import _rv_reconstruct


class FixedParameterValidationTest(unittest.TestCase):
    def test_scalar_support_and_order_validation_is_eager(self):
        invalid = (
            lambda: Normal(0.0, 0.0),
            lambda: Gamma(-1.0, 1.0),
            lambda: Bernoulli(float("nan")),
            lambda: Uniform(2.0, 1.0),
            lambda: Dirichlet([1.0, 0.0]),
            lambda: Categorical([0.6, 0.6]),
        )
        for constructor in invalid:
            with self.subTest(constructor=constructor):
                with self.assertRaises(ValueError):
                    constructor()

    def test_validation_is_reapplied_during_lowering_and_reconstruction(self):
        alpha = [1.0, 2.0]
        model = Dirichlet(alpha)
        alpha[0] = -1.0
        with self.assertRaises(ValueError):
            _ = model.dist
        with self.assertRaises(ValueError):
            _rv_reconstruct("sample", "Normal", (0.0, -1.0), None, None, None, "shared")

    def test_binomial_trial_count_is_exact_fixed_and_nonnegative(self):
        for n in (free, -1, 2.0, 2.5, True):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    Binomial(n, 0.5)
        self.assertEqual(Binomial(np.int64(3), 0.5).dist.n, 3)


class StructuralValidationTest(unittest.TestCase):
    def test_dimensions_are_exact_positive_integers(self):
        invalid = (
            lambda: MVN(2.0),
            lambda: DiagGaussian(0),
            lambda: LDA(0, 5),
            lambda: LDA(2, 3.5),
            lambda: Markov(Normal(0.0, 1.0), states=-1),
            lambda: free(2.5),
            lambda: free(0),
        )
        for constructor in invalid:
            with self.subTest(constructor=constructor):
                with self.assertRaises(ValueError):
                    constructor()

    def test_multivariate_arrays_are_shape_checked_and_preserved(self):
        mean = [3.0, 4.0]
        cov = [[2.0, 0.25], [0.25, 1.0]]
        mvn = MVN(2, mean=mean, cov=cov).dist
        self.assertTrue(np.array_equal(mvn.mu, mean))
        self.assertTrue(np.array_equal(mvn.covar, cov))

        diag = DiagGaussian(2, mean=mean, var=[2.0, 1.0]).dist
        self.assertTrue(np.array_equal(diag.mu, mean))
        self.assertTrue(np.array_equal(diag.covar, [2.0, 1.0]))

        with self.assertRaises(ValueError):
            MVN(2, mean=[1.0])
        with self.assertRaises(ValueError):
            MVN(2, cov=[[1.0, 2.0], [2.0, 1.0]])
        with self.assertRaises(ValueError):
            DiagGaussian(2, var=[1.0, 0.0])

    def test_composite_shapes_and_simplexes_are_validated(self):
        component = Normal(0.0, 1.0)
        for constructor in (
            lambda: Mix([]),
            lambda: SemiMix([]),
            lambda: Mix([component, component], [1.0]),
            lambda: SemiMix([component, component], [0.8, 0.8]),
            lambda: Markov(component, states=2, transitions=[[1.0, 0.0], [0.2, 0.2]]),
            lambda: Markov(component, states=2, initial=[1.0, -0.0, 0.0]),
            lambda: Markov([component, component], states=3),
        ):
            with self.subTest(constructor=constructor):
                with self.assertRaises(ValueError):
                    constructor()


class FixedEstimatorContractTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(8)
        self.x = rng.multivariate_normal([2.0, -1.0], [[1.5, 0.2], [0.2, 0.7]], size=300)

    def test_mvn_and_diagonal_fits_preserve_fixed_slots(self):
        fixed_mean = np.array([0.0, 0.0])
        mvn_mean = MVN(2, mean=fixed_mean).fit(self.x, max_its=2)
        self.assertTrue(np.array_equal(mvn_mean.dist.mu, fixed_mean))

        fixed_cov = np.array([[3.0, 0.4], [0.4, 2.0]])
        mvn_cov = MVN(2, cov=fixed_cov).fit(self.x, max_its=2)
        self.assertTrue(np.array_equal(mvn_cov.dist.covar, fixed_cov))

        fixed_var = np.array([3.0, 2.0])
        diag = DiagGaussian(2, var=fixed_var).fit(self.x, max_its=2)
        self.assertTrue(np.array_equal(diag.dist.covar, fixed_var))

    def test_semimix_fit_preserves_explicit_weights(self):
        rng = np.random.RandomState(3)
        data = [(float(x), None) for x in np.r_[rng.normal(-2.0, 0.5, 40), rng.normal(2.0, 0.5, 40)]]
        weights = np.array([0.8, 0.2])
        fitted = SemiMix([Normal(free, free), Normal(free, free)], weights).fit(
            data, max_its=2, rng=np.random.RandomState(4)
        )
        self.assertTrue(np.array_equal(fitted.dist.w, weights))

    def test_hmm_fit_and_seed_preserve_explicit_chain_structure(self):
        transitions = np.array([[0.95, 0.05], [0.2, 0.8]])
        initial = np.array([0.7, 0.3])
        rng = np.random.RandomState(5)
        sequences = [list(rng.normal(0.0, 1.0, 8)) for _ in range(20)]
        fitted = Markov(
            Normal(free, free),
            states=2,
            transitions=transitions,
            initial=initial,
        ).fit(sequences, max_its=2, rng=np.random.RandomState(6))
        self.assertTrue(np.array_equal(fitted.dist.transitions, transitions))
        self.assertTrue(np.array_equal(fitted.dist.w, initial))


if __name__ == "__main__":
    unittest.main()
