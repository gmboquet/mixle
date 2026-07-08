"""mixle.ppl.guide: user-declared structured variational inference (Guide + structured_vi).

A Guide declares the mean-field variational approximation (per-latent exponential-family q-factors =
the variational projection); structured_vi runs VMP/CAVI over the underlying factor graph and returns a
posterior over the named latents. These cover the conjugate-exponential shapes the VMP engine supports
(Gaussian/Gamma/Dirichlet, hierarchies, shared latents) and the projection-constraint checks.
"""

import unittest

import numpy as np

from mixle.ppl import Categorical, Dirichlet, Gamma, Guide, Normal, structured_vi
from mixle.ppl.guide import StructuredVIPosterior


class StructuredVITestCase(unittest.TestCase):
    def test_hierarchical_normal_gamma_recovers_mean_and_precision(self):
        rng = np.random.RandomState(0)
        data = list(rng.normal(5.0, 2.0, 3000))  # true mean 5, precision 1/4 = 0.25
        mu, tau = Normal(0, 10), Gamma(1, 1)
        post = structured_vi([(Normal(mu, tau), data)], Guide(mu=(mu, "gaussian"), tau=(tau, "gamma")))
        self.assertIsInstance(post, StructuredVIPosterior)
        self.assertAlmostEqual(post.mean("mu"), 5.0, delta=0.15)
        self.assertAlmostEqual(post.mean("tau"), 0.25, delta=0.05)
        # the q(mu) factor exposes its Gaussian hyperparameters
        self.assertIn("sd", post.posterior("mu"))
        # ELBO is monotone non-decreasing (a correctness signal for CAVI)
        self.assertTrue(np.all(np.diff(post.elbo_trace) >= -1e-6))

    def test_shared_latent_combines_evidence_across_factors(self):
        rng = np.random.RandomState(1)
        mu = Normal(0, 10)  # ONE handle reused in two factors -> shared latent
        a = list(rng.normal(3.0, 1.0, 1500))
        b = list(rng.normal(3.0, 2.0, 1500))
        post = structured_vi([(Normal(mu, 1.0), a), (Normal(mu, 2.0), b)], Guide(mu=mu))
        self.assertAlmostEqual(post.mean("mu"), 3.0, delta=0.15)

    def test_dirichlet_categorical_and_single_factor_tuple(self):
        rng = np.random.RandomState(2)
        pi = Dirichlet([1.0, 1.0, 1.0])
        cats = list(rng.choice(3, size=4000, p=[0.2, 0.3, 0.5]))
        post = structured_vi((Categorical(pi), cats), Guide(pi=(pi, "dirichlet")))  # bare (model, data)
        self.assertTrue(np.allclose(post.mean("pi"), [0.2, 0.3, 0.5], atol=0.05))
        self.assertEqual(post.samples("pi", n=7).shape, (7, 3))

    def test_projection_constraint_family_mismatch_is_rejected(self):
        rng = np.random.RandomState(3)
        data = list(rng.normal(0.0, 1.0, 500))
        mu, tau = Normal(0, 10), Gamma(1, 1)
        with self.assertRaises(ValueError):  # mu's conjugate factor is gaussian, not gamma
            structured_vi([(Normal(mu, tau), data)], Guide(mu=(mu, "gamma")))

    def test_latent_not_in_model_is_rejected(self):
        rng = np.random.RandomState(4)
        data = list(rng.normal(0.0, 1.0, 500))
        mu, tau = Normal(0, 10), Gamma(1, 1)
        stray = Normal(0, 1)  # never used in any factor
        with self.assertRaises(ValueError):
            structured_vi([(Normal(mu, tau), data)], Guide(ghost=stray))

    def test_unknown_family_and_non_handle_rejected_at_guide_build(self):
        mu = Normal(0, 10)
        with self.assertRaises(ValueError):
            Guide(mu=(mu, "studentt"))  # not a supported q-family
        with self.assertRaises(TypeError):
            Guide(mu=3.0)  # not a RandomVariable handle

    def test_admixture_lda_recovers_topics_without_lda_distribution(self):
        # LDA via the guide surface: declare topic Dirichlet factors, fit by mean-field VI from
        # primitives (no LDADistribution). Topics are recovered up to the usual label permutation.
        import mixle.stats as S
        from mixle.ppl import Dirichlet, admixture

        V = 6
        true = [
            {0: 0.5, 1: 0.3, 2: 0.15, 3: 0.03, 4: 0.01, 5: 0.01},
            {0: 0.01, 1: 0.01, 2: 0.03, 3: 0.15, 4: 0.3, 5: 0.5},
        ]
        gen = S.LDADistribution(
            [S.CategoricalDistribution(t) for t in true],
            alpha=[0.3, 0.3],
            len_dist=S.CategoricalDistribution({20: 1.0}),
        )
        docs = [list(d) for d in gen.sampler(seed=1).sample(150)]
        topics = [Dirichlet([0.1] * V, name=f"topic{k}") for k in range(2)]
        # A smaller corpus (150 docs x 20 words) with fewer CAVI iterations still recovers the
        # topics comfortably within the assertion's tolerance (checked across 10 seeds: worst-case
        # max deviation ~0.033, well under the 0.07 threshold below) at a fraction of the runtime.
        post = admixture(docs, topics, alpha=0.3, max_its=30, inner_its=20)

        recovered = np.array([post.posterior(t)["mean"] for t in topics])  # (2, V)
        truth = np.array([[t.get(v, 0.0) for v in range(V)] for t in true])
        # best topic-label alignment (topics are unidentifiable up to permutation)
        aligned = min(
            max(np.max(np.abs(recovered - truth)), 0.0),
            np.max(np.abs(recovered[::-1] - truth)),
        )
        self.assertLess(aligned, 0.07)
        self.assertEqual(post.topics().shape, (2, V))
        self.assertTrue(np.all(np.diff(post.log_likelihood_trace) >= -1e-6))  # LL non-decreasing

    def test_admixture_rejects_non_dirichlet_topics(self):
        from mixle.ppl import Normal, admixture

        with self.assertRaises(TypeError):
            admixture([[0, 1, 2]], [Normal(0, 1)], alpha=0.3)

    def test_summary_lists_named_latents_and_elbo(self):
        rng = np.random.RandomState(5)
        data = list(rng.normal(2.0, 1.0, 1000))
        mu, tau = Normal(0, 10), Gamma(1, 1)
        post = structured_vi([(Normal(mu, tau), data)], Guide(mu=mu, tau=tau))
        s = post.summary()
        self.assertEqual(set(s) - {"elbo", "iterations"}, {"mu", "tau"})
        self.assertIn("elbo", s)


if __name__ == "__main__":
    unittest.main()
