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

    def test_sample_budget_is_exact_not_truncated(self):
        # MXR-080-1727: n_samples went through int(), so 2.9 silently became a two-draw
        # "projection" and 0 or a negative budget reached the fitter as an empty-data error about
        # something else entirely. The budget is the accuracy of an empirical projection; a
        # nonsensical one is refused where it was given.
        for bad in (2.9, 0, -5, True, "20000"):
            with self.subTest(n_samples=repr(bad)):
                with self.assertRaises(ValueError):
                    project(GaussianDistribution(0.0, 1.0), GaussianDistribution(0.0, 1.0), n_samples=bad)
        with self.assertRaises(ValueError):
            project(GaussianDistribution(0.0, 1.0), GaussianDistribution(0.0, 1.0), n_samples=100, max_its=0)

    def test_docstring_does_not_claim_an_exact_kl_projection(self):
        # MXR-080-1727: a finite Monte Carlo draw fitted by a capped, locally-optimal EM is not
        # "exactly" the forward-KL projection, and nothing here estimates the gap.
        doc = project.__doc__ or ""
        self.assertNotIn("exactly the projection", doc)
        self.assertIn("empirical M-projection, not the exact one", doc)


if __name__ == "__main__":
    unittest.main()
