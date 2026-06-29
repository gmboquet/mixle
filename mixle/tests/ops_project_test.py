"""WS-5: ops.project -- sample-based variational (forward-KL / M-) projection onto a target family."""

import unittest

import mixle
from mixle.capability import CapabilityError, HasMoments
from mixle.ops import mixture, project
from mixle.stats.univariate.continuous.gamma import GammaDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution


class OpsProjectTest(unittest.TestCase):
    def test_mixture_projected_onto_gaussian_matches_overall_moments(self):
        # 50/50 mixture of N(0,1) and N(4,1): overall mean 2, variance 0.5*(1)+0.5*(1)+0.5*0.5*(4-0)^2 = 5
        src = mixture([GaussianDistribution(0.0, 1.0), GaussianDistribution(4.0, 1.0)], [0.5, 0.5])
        proj = project(src, GaussianDistribution(0.0, 1.0), n_samples=40_000, seed=0)
        self.assertIsInstance(proj, GaussianDistribution)
        self.assertAlmostEqual(proj.mean(), 2.0, delta=0.1)
        self.assertAlmostEqual(proj.variance(), 5.0, delta=0.3)

    def test_gamma_projected_onto_gaussian_matches_moments(self):
        proj = project(GammaDistribution(3.0, 0.5), GaussianDistribution(0.0, 1.0), n_samples=40_000, seed=1)
        self.assertAlmostEqual(proj.mean(), 1.5, delta=0.05)  # Gamma(k=3, theta=0.5): mean k*theta
        self.assertAlmostEqual(proj.variance(), 0.75, delta=0.05)  # var k*theta^2

    def test_target_may_be_an_estimator(self):
        proj = project(
            GammaDistribution(3.0, 0.5), GaussianDistribution(0.0, 1.0).estimator(), n_samples=20_000, seed=2
        )
        self.assertIsInstance(proj, GaussianDistribution)

    def test_projected_model_carries_target_capabilities(self):
        proj = project(GammaDistribution(3.0, 0.5), GaussianDistribution(0.0, 1.0), n_samples=10_000, seed=3)
        self.assertTrue(mixle.supports(proj, HasMoments))

    def test_non_sampleable_source_raises(self):
        with self.assertRaises(CapabilityError):
            project(object(), GaussianDistribution(0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
