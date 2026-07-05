"""Black-box Laplace posterior (mixle.inference.blackbox): a parameter posterior for ANY model.

Treats a fitted model's parameters as the latent, flattens them to an unconstrained vector, and fits a
Gaussian from a finite-difference Hessian of the model's own seq_log_density -- conjugate or not, no
autograd, no per-model inference. Covers the scalar exp-family leaves + Composite + Mixture (recursively).
"""

import unittest

import numpy as np

import mixle.stats as S
from mixle.inference import laplace_posterior, optimize
from mixle.inference.blackbox import _flatten


class BlackboxLaplaceTest(unittest.TestCase):
    def test_flatten_round_trip_is_identity(self):
        comp = S.CompositeDistribution((S.GaussianDistribution(1.5, 4.0), S.PoissonDistribution(3.0)))
        u0, rebuild = _flatten(comp)
        back, _ = rebuild(u0)
        self.assertTrue(np.allclose([back.dists[0].mu, back.dists[0].sigma2, back.dists[1].lam], [1.5, 4.0, 3.0]))

    def test_calibrated_on_gaussian_mean(self):
        rng = np.random.RandomState(0)
        data = list(rng.normal(5.0, 2.0, 400))
        fit = optimize(data, S.GaussianDistribution(0, 1).estimator(), max_its=20, out=None)
        post = laplace_posterior(fit, data)
        draws = np.array([m.mu for m in post.sample(4000, rng=np.random.RandomState(1))])
        self.assertAlmostEqual(draws.std(), 2.0 / np.sqrt(400), delta=0.02)  # ~ sigma/sqrt(n)
        self.assertAlmostEqual(draws.mean(), fit.mu, delta=0.03)

    def test_works_for_heterogeneous_composite(self):
        # neither conjugate nor an autograd target -> the general path still gives a posterior
        rng = np.random.RandomState(2)
        data = [(float(rng.normal(2, 1)), int(rng.poisson(4))) for _ in range(500)]
        proto = S.CompositeDistribution((S.GaussianDistribution(0, 1), S.PoissonDistribution(1.0)))
        fit = optimize(data, proto.estimator(), max_its=20, out=None)
        ms = laplace_posterior(fit, data).sample(2000, rng=np.random.RandomState(3))
        self.assertAlmostEqual(np.mean([m.dists[0].mu for m in ms]), 2.0, delta=0.2)
        self.assertAlmostEqual(np.mean([m.dists[1].lam for m in ms]), 4.0, delta=0.4)

    def test_works_for_gamma_mixture(self):
        rng = np.random.RandomState(4)
        data = list(np.concatenate([rng.gamma(2, 1, 400), rng.gamma(8, 1, 400)]))
        proto = S.MixtureDistribution([S.GammaDistribution(2, 1), S.GammaDistribution(6, 1)], [0.5, 0.5])
        fit = optimize(data, proto.estimator(), prev_estimate=proto, max_its=60, out=None)
        m = laplace_posterior(fit, data).sample(1, rng=np.random.RandomState(5))
        self.assertEqual(len(m.components), 2)  # a valid posterior draw rebuilt into a fitted mixture

    def test_bayesian_network_gets_a_parameter_posterior(self):
        # a learned heterogeneous DAG (categorical marginal + conditional-linear-Gaussian edges) is now
        # flattenable, so uq(model, data) / create(..., quantify_uq=True) attach a real posterior, not None.
        from mixle.inference.bayesian_network import learn_bayesian_network

        rng = np.random.RandomState(0)
        plan_spend_records = []
        for _ in range(400):
            plan = rng.choice(["free", "pro", "enterprise"], p=[0.5, 0.3, 0.2])
            base = {"free": 10.0, "pro": 50.0, "enterprise": 200.0}[plan]
            spend = base + rng.normal(0, 5)
            seats = 0.5 * spend + rng.normal(0, 3)
            plan_spend_records.append((plan, float(spend), float(seats)))

        bn = learn_bayesian_network(plan_spend_records, max_parents=2)

        # flatten round-trips exactly: same joint log-likelihood after params -> vector -> params
        u0, rebuild = _flatten(bn)
        self.assertGreater(len(u0), 0)
        back, _ = rebuild(u0)
        enc0 = bn.dist_to_encoder().seq_encode(plan_spend_records)
        enc1 = back.dist_to_encoder().seq_encode(plan_spend_records)
        self.assertAlmostEqual(
            float(np.sum(bn.seq_log_density(enc0))), float(np.sum(back.seq_log_density(enc1))), places=6
        )

        post = laplace_posterior(bn, plan_spend_records)
        self.assertIsNotNone(post)  # the uq handle a BN used to get None for
        draw = post.sample(1, rng=np.random.RandomState(1))
        self.assertEqual(len(draw.factors), len(bn.factors))  # a valid draw rebuilt into a network

    def test_unsupported_structure_raises_clearly(self):
        hmm = S.HiddenMarkovModelDistribution(
            [S.GaussianDistribution(-1, 1), S.GaussianDistribution(1, 1)], [0.5, 0.5], [[0.7, 0.3], [0.3, 0.7]]
        )
        with self.assertRaises(NotImplementedError):
            _flatten(hmm)


if __name__ == "__main__":
    unittest.main()
