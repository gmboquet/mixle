"""B3 escalation ladder: how='posterior' climbs to the cheapest route that yields a posterior.

Unlike how='auto' (which stops at MAP -- a point estimate -- for a non-conjugate prior), how='posterior'
escalates conjugate (exact) -> Laplace (Gaussian at the MAP) -> MCMC, and explain_fit reports the rung.
"""

import unittest

import numpy as np

from mixle.ppl import Beta, Gamma, Mix, Normal, Poisson, free


class PosteriorLadderTestCase(unittest.TestCase):
    def test_explain_reports_the_rung(self):
        self.assertEqual(Poisson(Gamma(2, 1, name="lam")).explain_fit(how="posterior")["route"], "conjugate")
        self.assertEqual(Normal(Beta(2, 2, name="m"), 1.0).explain_fit(how="posterior")["route"], "laplace")
        self.assertEqual(Normal(free, free).explain_fit(how="posterior")["route"], "conjugate")  # NIG
        self.assertEqual(
            Mix([Normal(-2, 1), Normal(2, 1)], [0.5, 0.5]).explain_fit(how="posterior")["route"], "mcmc"
        )

    def test_conjugate_rung_runs(self):
        rng = np.random.RandomState(0)
        m = Poisson(Gamma(2, 1, name="lam")).fit(list(rng.poisson(3.0, 600).astype(float)), how="posterior")
        self.assertAlmostEqual(float(m.result.mean("lam")), 3.0, delta=0.3)

    def test_laplace_rung_returns_a_posterior_not_a_point_estimate(self):
        # non-conjugate prior: auto would give MAP (point estimate); posterior climbs to Laplace
        rng = np.random.RandomState(1)
        m = Normal(Beta(2, 2, name="m"), 1.0).fit(list(rng.normal(0.4, 1.0, 600)), how="posterior")
        self.assertTrue(hasattr(m.result, "samples"))  # a posterior object, not a bare point estimate
        self.assertAlmostEqual(float(m.dist.mu), 0.4, delta=0.2)


if __name__ == "__main__":
    unittest.main()
