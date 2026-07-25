"""EstimationCertificate: per-block method + guarantee ladder, the why-not-ADAM audit."""

import unittest

import numpy as np

import mixle.stats as st
from mixle.inference import Guarantee, VerificationReceipt, certify, optimize, plan_estimation

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class GuaranteeLadderTest(unittest.TestCase):
    def test_ladder_is_ordered(self):
        self.assertLess(Guarantee.UNVERIFIED, Guarantee.HEURISTIC)
        self.assertLess(Guarantee.HEURISTIC, Guarantee.STATIONARY)
        self.assertLess(Guarantee.STATIONARY, Guarantee.STATIONARY_ESCAPE_TESTED)
        self.assertLess(Guarantee.STATIONARY_ESCAPE_TESTED, Guarantee.GLOBAL)
        self.assertLess(Guarantee.GLOBAL, Guarantee.GLOBAL_UNIQUE)


class ClosedFormCertificateTest(unittest.TestCase):
    def test_exp_family_composite_is_global_unique_with_no_gradient(self):
        rows = [(float(np.random.RandomState(i).randn()), int(np.random.RandomState(i).poisson(3))) for i in range(300)]
        model = optimize(rows, st.CompositeEstimator((st.GaussianEstimator(), st.PoissonEstimator())), out=None)
        cert = certify(model, data=rows)
        self.assertEqual(cert.guarantee, Guarantee.GLOBAL_UNIQUE)
        self.assertEqual(len(cert.blocks), 2)
        self.assertEqual(cert.gradient_blocks, [])
        self.assertIn("No gradient descent", cert.why_not_adam())

    def test_single_exp_family_leaf(self):
        model = optimize(
            [float(np.random.RandomState(i).randn()) for i in range(200)], st.GaussianEstimator(), out=None
        )
        data = [float(np.random.RandomState(i).randn()) for i in range(200)]
        cert = certify(model, data=data)
        self.assertEqual(cert.guarantee, Guarantee.GLOBAL_UNIQUE)
        self.assertEqual(len(cert.blocks), 1)

    def test_nominal_family_without_data_is_not_a_certificate(self):
        data = [float(np.random.RandomState(i).randn()) for i in range(200)]
        model = optimize(data, st.GaussianEstimator(), out=None)
        cert = certify(model)
        self.assertEqual(cert.guarantee, Guarantee.UNVERIFIED)
        self.assertEqual(cert.blocks[0].candidate_guarantee, Guarantee.GLOBAL_UNIQUE)
        self.assertTrue(cert.blocks[0].proof_obligations)

    def test_well_conditioned_but_unrelated_data_cannot_certify_fitted_parameters(self):
        fitted_on = list(np.random.RandomState(0).normal(0.0, 1.0, 200))
        unrelated = list(np.random.RandomState(1).normal(10.0, 3.0, 200))
        model = optimize(fitted_on, st.GaussianEstimator(), out=None)
        self.assertEqual(certify(model, data=unrelated).guarantee, Guarantee.UNVERIFIED)

    def test_degenerate_gaussian_data_does_not_prove_identification(self):
        data = [2.0] * 20
        model = optimize(data, st.GaussianEstimator(), out=None)
        cert = certify(model, data=data)
        self.assertEqual(cert.guarantee, Guarantee.UNVERIFIED)
        self.assertIn("identified_parameters", {item.check for item in cert.blocks[0].proof_obligations})

    def test_discovered_bayesian_network_is_closed_form(self):
        def recs(n, seed):
            r = np.random.RandomState(seed)
            out = []
            for _ in range(n):
                plan = ["free", "pro"][r.randint(0, 2)]
                usage = float({"free": 5.0, "pro": 30.0}[plan] + 3.0 * r.randn())
                out.append((plan, usage))
            return out

        rows = recs(400, 0)
        bn = optimize(rows, out=None)  # structure discovery is the default -> a BN
        self.assertEqual(type(bn).__name__, "HeterogeneousBayesianNetwork")
        cert = certify(bn, data=rows)
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
        self.assertEqual(cert.guarantee, Guarantee.UNVERIFIED)
        self.assertEqual(cert.gradient_blocks, [])  # but no ADAM: every M-step is closed form
        comp_blocks = [b for b in cert.blocks if b.name.startswith("component")]
        self.assertTrue(comp_blocks and all(b.candidate_guarantee == Guarantee.GLOBAL_UNIQUE for b in comp_blocks))

    def test_escape_requires_a_structured_receipt(self):
        model = optimize(
            [float(np.random.RandomState(i).randn()) for i in range(200)],
            st.MixtureEstimator([st.GaussianEstimator(), st.GaussianEstimator()]),
            max_its=20,
            out=None,
        )
        plain = certify(model)
        with self.assertRaisesRegex(ValueError, "not evidence"):
            certify(model, escape_tested=True)
        receipt = VerificationReceipt(
            receipt_id="restart-comparison-1",
            block="mixture",
            guarantee=Guarantee.STATIONARY_ESCAPE_TESTED,
            checks=("optimizer_converged", "saddle_escape_compared"),
            source="test optimizer report",
            evidence={"converged": True, "starts": 4, "same_objective": True},
        )
        tested = certify(model, receipts=[receipt])
        self.assertEqual(plain.guarantee, Guarantee.UNVERIFIED)
        self.assertEqual(tested.blocks[0].guarantee, Guarantee.STATIONARY_ESCAPE_TESTED)
        self.assertTrue(tested.escape_tested)

    def test_incomplete_receipt_does_not_upgrade_the_block(self):
        model = optimize(
            [float(np.random.RandomState(i).randn()) for i in range(200)],
            st.MixtureEstimator([st.GaussianEstimator(), st.GaussianEstimator()]),
            max_its=20,
            out=None,
        )
        receipt = VerificationReceipt(
            receipt_id="restart-assertion-only",
            block="mixture",
            guarantee=Guarantee.STATIONARY_ESCAPE_TESTED,
            checks=("saddle_escape_compared",),
            source="test incomplete report",
            evidence={"starts": 4},
        )
        self.assertEqual(certify(model, receipts=[receipt]).blocks[0].guarantee, Guarantee.UNVERIFIED)


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
        self.assertEqual(cert.guarantee, Guarantee.UNVERIFIED)
        self.assertEqual(cert.gradient_blocks[0].candidate_guarantee, Guarantee.HEURISTIC)
        self.assertEqual(len(cert.gradient_blocks), 1)
        self.assertEqual(cert.gradient_blocks[0].placement, "pool_eligible")
        # the classical component stayed closed form -- the audit names the one exception
        self.assertIn("required gradient descent", cert.why_not_adam())


class ProcessClassificationTest(unittest.TestCase):
    def _ip(self):
        from mixle.stats.processes.inhomogeneous_poisson import InhomogeneousPoissonProcessEstimator as IPE

        rng = np.random.RandomState(0)
        data = [np.sort(rng.uniform(0, 10, rng.poisson(20))).tolist() for _ in range(30)]
        return optimize(data, IPE(num_bins=5, t_max=10.0), out=None, max_its=5), data

    def _hawkes(self):
        from mixle.stats.processes.hawkes_process import HawkesProcessEstimator as HE

        rng = np.random.RandomState(1)
        data = [np.sort(rng.uniform(0, 10, rng.poisson(15))).tolist() for _ in range(30)]
        return optimize(data, HE(window=10.0), out=None, max_its=5)

    def test_inhomogeneous_poisson_is_global_unique_closed_form(self):
        model, data = self._ip()
        block = certify(model, data=data).blocks[0]
        self.assertEqual(block.guarantee, Guarantee.GLOBAL_UNIQUE)  # closed-form per-bin rate MLE
        self.assertEqual(block.method, "closed_form_counts")
        self.assertFalse(block.gradient)

    def test_hawkes_is_stationary_non_convex_em(self):
        model = self._hawkes()
        block = certify(model).blocks[0]
        self.assertEqual(block.guarantee, Guarantee.UNVERIFIED)
        self.assertEqual(block.candidate_guarantee, Guarantee.STATIONARY)
        self.assertEqual(block.method, "em_branching")
        self.assertIn("non-convex", block.reason)

    def test_ctmc_with_an_unexposed_state_is_not_identified(self):
        from mixle.stats.processes.ctmc import ContinuousTimeMarkovChainDistribution

        model = ContinuousTimeMarkovChainDistribution(np.zeros((2, 2)), horizon=5.0)
        cert = certify(model, data=[(0, 5.0, [])])
        self.assertEqual(cert.guarantee, Guarantee.UNVERIFIED)
        self.assertEqual(cert.blocks[0].candidate_guarantee, Guarantee.GLOBAL_UNIQUE)

    def test_neither_process_used_gradient_descent(self):
        model, data = self._ip()
        self.assertIn("No gradient descent", certify(model, data=data).why_not_adam())
        self.assertIn("No gradient descent", certify(self._hawkes()).why_not_adam())

    def test_renewal_inherits_its_interarrival_guarantee(self):
        from mixle.stats.processes.renewal_process import RenewalProcessEstimator as RPE

        rng = np.random.RandomState(0)
        data = [np.cumsum(rng.exponential(1.0, rng.poisson(12) + 1)).tolist() for _ in range(40)]
        model = optimize(data, RPE(st.ExponentialEstimator(), window=15.0), out=None, max_its=5)
        block = certify(model).blocks[0]
        self.assertEqual(block.guarantee, Guarantee.UNVERIFIED)
        self.assertEqual(block.candidate_guarantee, Guarantee.GLOBAL_UNIQUE)
        self.assertIn("renewal_mle", block.method)


class ScheduleTest(unittest.TestCase):
    """Planner v2 (A3): block-coordinate schedules for latent models."""

    def test_fully_observed_model_is_one_shot(self):
        from mixle.inference import schedule

        g = optimize([float(np.random.RandomState(i).randn()) for i in range(100)], st.GaussianEstimator(), out=None)
        s = schedule(g)
        self.assertFalse(s.latent)
        self.assertEqual([p.repeat for p in s.passes], ["once"])
        self.assertIn("one-shot", s.describe())

    def test_mixture_schedules_the_em_loop_explicitly(self):
        from mixle.inference import schedule

        data = [float(x) for x in np.random.RandomState(0).normal(0, 1, 200)]
        m = optimize(data, st.MixtureEstimator([st.GaussianEstimator(), st.GaussianEstimator()]), out=None, max_its=5)
        s = schedule(m)
        self.assertTrue(s.latent)
        kinds = [p.kind for p in s.per_round]
        self.assertEqual(kinds[0], "estep")  # E-step first
        self.assertEqual(kinds.count("mstep"), 2)  # one closed-form M-step per component, per round
        self.assertTrue(all(p.placement == "local" for p in s.passes))  # nothing here needs a pool
        self.assertIn("EM loop", s.describe())

    def test_bn_schedules_independent_per_factor_passes(self):
        from mixle.inference import learn_bayesian_network, schedule

        rows = [(["a", "b"][i % 2], float(i % 2 * 10 + np.random.RandomState(i).randn())) for i in range(100)]
        s = schedule(learn_bayesian_network(rows, max_parents=1))
        self.assertFalse(s.latent)
        self.assertEqual(len(s.passes), 2)  # one pass per factor, no loop


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
