"""EstimationCertificate: per-block method + guarantee ladder, the why-not-ADAM audit."""

import unittest

import numpy as np

import mixle.stats as st
from mixle.inference import Guarantee, certify, optimize, plan_estimation

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class GuaranteeLadderTest(unittest.TestCase):
    def test_ladder_is_ordered(self):
        self.assertLess(Guarantee.HEURISTIC, Guarantee.STATIONARY)
        self.assertLess(Guarantee.STATIONARY, Guarantee.STATIONARY_ESCAPE_TESTED)
        self.assertLess(Guarantee.STATIONARY_ESCAPE_TESTED, Guarantee.GLOBAL)
        self.assertLess(Guarantee.GLOBAL, Guarantee.GLOBAL_UNIQUE)


class ClosedFormCertificateTest(unittest.TestCase):
    def test_exp_family_composite_is_global_unique_with_no_gradient(self):
        rows = [(float(np.random.RandomState(i).randn()), int(np.random.RandomState(i).poisson(3))) for i in range(300)]
        model = optimize(rows, st.CompositeEstimator((st.GaussianEstimator(), st.PoissonEstimator())), out=None)
        cert = certify(model)
        self.assertEqual(cert.guarantee, Guarantee.GLOBAL_UNIQUE)
        self.assertEqual(len(cert.blocks), 2)
        self.assertEqual(cert.gradient_blocks, [])
        self.assertIn("No gradient descent", cert.why_not_adam())

    def test_single_exp_family_leaf(self):
        model = optimize(
            [float(np.random.RandomState(i).randn()) for i in range(200)], st.GaussianEstimator(), out=None
        )
        cert = certify(model)
        self.assertEqual(cert.guarantee, Guarantee.GLOBAL_UNIQUE)
        self.assertEqual(len(cert.blocks), 1)

    def test_discovered_bayesian_network_is_closed_form(self):
        def recs(n, seed):
            r = np.random.RandomState(seed)
            out = []
            for _ in range(n):
                plan = ["free", "pro"][r.randint(0, 2)]
                usage = float({"free": 5.0, "pro": 30.0}[plan] + 3.0 * r.randn())
                out.append((plan, usage))
            return out

        bn = optimize(recs(400, 0), out=None)  # structure discovery is the default -> a BN
        self.assertEqual(type(bn).__name__, "HeterogeneousBayesianNetwork")
        cert = certify(bn)
        self.assertGreaterEqual(cert.guarantee, Guarantee.GLOBAL)  # CLG/GLM/exp-family factors only
        self.assertEqual(cert.gradient_blocks, [])
        # the CLG factor's least-squares block is unique global
        self.assertTrue(
            any(b.method == "least_squares" and b.guarantee == Guarantee.GLOBAL_UNIQUE for b in cert.blocks)
        )


class LatentCertificateTest(unittest.TestCase):
    def test_mixture_is_stationary_but_m_steps_are_closed_form(self):
        model = optimize(
            [float(np.random.RandomState(i).randn()) for i in range(400)],
            st.MixtureEstimator([st.GaussianEstimator(), st.GaussianEstimator()]),
            max_its=30,
            out=None,
        )
        cert = certify(model)
        self.assertEqual(cert.guarantee, Guarantee.STATIONARY)  # latent structure caps it
        self.assertEqual(cert.gradient_blocks, [])  # but no ADAM: every M-step is closed form
        comp_blocks = [b for b in cert.blocks if b.name.startswith("component")]
        self.assertTrue(comp_blocks and all(b.guarantee == Guarantee.GLOBAL_UNIQUE for b in comp_blocks))

    def test_escape_tested_upgrades_the_em_block(self):
        model = optimize(
            [float(np.random.RandomState(i).randn()) for i in range(200)],
            st.MixtureEstimator([st.GaussianEstimator(), st.GaussianEstimator()]),
            max_its=20,
            out=None,
        )
        plain = certify(model, escape_tested=False)
        tested = certify(model, escape_tested=True)
        self.assertEqual(plain.guarantee, Guarantee.STATIONARY)
        self.assertEqual(tested.guarantee, Guarantee.STATIONARY_ESCAPE_TESTED)
        self.assertTrue(tested.escape_tested)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class GradientAuditTest(unittest.TestCase):
    def test_neural_block_is_isolated_and_pool_eligible(self):
        import torch

        from mixle.models.neural_density import NeuralDensity, build_coupling_flow

        torch.manual_seed(0)
        train = [np.random.RandomState(i).randn(2) for i in range(400)]
        est = st.MixtureEstimator(
            [NeuralDensity(build_coupling_flow(2, layers=4)).estimator(), st.MultivariateGaussianEstimator(dim=2)]
        )
        init = st.MixtureDistribution(
            [
                NeuralDensity(build_coupling_flow(2, layers=4)),
                st.MultivariateGaussianDistribution(np.zeros(2), np.eye(2)),
            ],
            [0.5, 0.5],
        )
        hybrid = optimize(train, est, prev_estimate=init, max_its=4, out=None)
        cert = certify(hybrid)
        self.assertEqual(cert.guarantee, Guarantee.HEURISTIC)  # capped by the one gradient block
        self.assertEqual(len(cert.gradient_blocks), 1)
        self.assertEqual(cert.gradient_blocks[0].placement, "pool_eligible")
        # the classical component stayed closed form -- the audit names the one exception
        self.assertIn("required gradient descent", cert.why_not_adam())


class ProcessClassificationTest(unittest.TestCase):
    def _ip(self):
        from mixle.stats.processes.inhomogeneous_poisson import InhomogeneousPoissonProcessEstimator as IPE

        rng = np.random.RandomState(0)
        data = [np.sort(rng.uniform(0, 10, rng.poisson(20))).tolist() for _ in range(30)]
        return optimize(data, IPE(num_bins=5, t_max=10.0), out=None, max_its=5)

    def _hawkes(self):
        from mixle.stats.processes.hawkes_process import HawkesProcessEstimator as HE

        rng = np.random.RandomState(1)
        data = [np.sort(rng.uniform(0, 10, rng.poisson(15))).tolist() for _ in range(30)]
        return optimize(data, HE(window=10.0), out=None, max_its=5)

    def test_inhomogeneous_poisson_is_global_unique_closed_form(self):
        block = certify(self._ip()).blocks[0]
        self.assertEqual(block.guarantee, Guarantee.GLOBAL_UNIQUE)  # closed-form per-bin rate MLE
        self.assertEqual(block.method, "closed_form_counts")
        self.assertFalse(block.gradient)

    def test_hawkes_is_stationary_non_convex_em(self):
        block = certify(self._hawkes()).blocks[0]
        self.assertEqual(block.guarantee, Guarantee.STATIONARY)  # non-convex, EM stationary point
        self.assertEqual(block.method, "em_branching")
        self.assertIn("non-convex", block.reason)

    def test_neither_process_used_gradient_descent(self):
        self.assertIn("No gradient descent", certify(self._ip()).why_not_adam())
        self.assertIn("No gradient descent", certify(self._hawkes()).why_not_adam())

    def test_renewal_inherits_its_interarrival_guarantee(self):
        from mixle.stats.processes.renewal_process import RenewalProcessEstimator as RPE

        rng = np.random.RandomState(0)
        data = [np.cumsum(rng.exponential(1.0, rng.poisson(12) + 1)).tolist() for _ in range(40)]
        model = optimize(data, RPE(st.ExponentialEstimator(), window=15.0), out=None, max_its=5)
        block = certify(model).blocks[0]
        # exponential inter-arrivals are exp-family -> the renewal MLE is GLOBAL_UNIQUE too
        self.assertEqual(block.guarantee, Guarantee.GLOBAL_UNIQUE)
        self.assertIn("renewal_mle", block.method)


class FacadeTest(unittest.TestCase):
    def test_model_fit_attaches_a_certificate(self):
        from mixle import Model

        m = Model(st.GaussianEstimator()).fit([float(np.random.RandomState(i).randn()) for i in range(200)])
        self.assertIsNotNone(m.certificate)
        self.assertEqual(m.certificate.guarantee, Guarantee.GLOBAL_UNIQUE)

    def test_plan_estimation_is_the_prefit_alias(self):
        model = optimize(
            [float(np.random.RandomState(i).randn()) for i in range(200)], st.GaussianEstimator(), out=None
        )
        self.assertEqual(plan_estimation(model).guarantee, certify(model).guarantee)


if __name__ == "__main__":
    unittest.main()
