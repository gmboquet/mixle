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
        self.assertEqual(Mix([Normal(-2, 1), Normal(2, 1)], [0.5, 0.5]).explain_fit(how="posterior")["route"], "mcmc")

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

    def test_grouped_prior_escalates_to_hierarchical_not_a_silent_flat_pool(self):
        # how='auto' checks `grouped` before anything else; the posterior ladder used not to check it
        # at all, so a .each() group prior fell through to the flat conjugate/Laplace/MCMC checks and
        # was silently fit as if it had no group structure -- pooling every group into one flat estimate.
        model = Normal(Normal(0, 100).each(by="school"), free)
        self.assertEqual(model.explain_fit(how="posterior")["route"], "hierarchical")

        rng = np.random.RandomState(2)
        G = 5
        theta_true = rng.normal(5.0, 4.0, G)
        labels, y = [], []
        for g in range(G):
            n = rng.randint(20, 40)
            labels += [g] * n
            y += list(rng.normal(theta_true[g], 1.0, n))
        labels, y = np.array(labels), np.array(y)
        fit = model.fit(y, given={"school": labels}, how="posterior")
        group_means = np.asarray(fit.result.summary()["group_means"])
        self.assertEqual(group_means.shape, (G,))  # per-group latents, not one pooled estimate
        self.assertLess(float(np.max(np.abs(group_means - theta_true))), 1.0)


if __name__ == "__main__":
    unittest.main()
