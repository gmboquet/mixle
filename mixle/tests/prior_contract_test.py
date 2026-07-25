"""Propriety and provenance contracts for public prior and conjugate helpers."""

import unittest

import numpy as np

from mixle.inference.mcmc.conjugate import ImproperPosteriorError, sample_conjugate_posterior
from mixle.inference.priors import (
    BetaPrior,
    DirichletPrior,
    GammaPrior,
    ImproperPriorReceipt,
    NormalGammaPrior,
    improper,
)
from mixle.stats import GaussianDistribution, GaussianEstimator
from mixle.stats.bayes.normal_gamma import NormalGammaDistribution


class PriorContractTestCase(unittest.TestCase):
    def test_public_prior_defaults_and_positive_hyperparameters_are_proper(self):
        normal_gamma = NormalGammaPrior()
        self.assertGreater(normal_gamma.kappa, 0.0)
        self.assertGreater(normal_gamma.alpha, 0.0)
        self.assertGreater(normal_gamma.beta, 0.0)
        self.assertTrue(normal_gamma.as_dict()["proper"])
        estimator = GaussianEstimator(prior=normal_gamma)
        self.assertIsInstance(estimator.get_prior(), NormalGammaDistribution)
        self.assertEqual(
            estimator.get_prior().get_parameters(),
            (normal_gamma.mu0, normal_gamma.kappa, normal_gamma.alpha, normal_gamma.beta),
        )

        for prior in (
            BetaPrior(1.0, 2.0),
            GammaPrior(1.0, 2.0),
            DirichletPrior([1.0, 2.0]),
        ):
            with self.subTest(prior=type(prior).__name__):
                self.assertTrue(prior.as_dict()["proper"])
                self.assertIsNone(prior.as_dict()["improper_receipt"])

    def test_proper_priors_reject_zero_negative_and_nonfinite_hyperparameters(self):
        factories = (
            lambda value: NormalGammaPrior(kappa=value),
            lambda value: NormalGammaPrior(alpha=value),
            lambda value: NormalGammaPrior(beta=value),
            lambda value: BetaPrior(value, 1.0),
            lambda value: BetaPrior(1.0, value),
            lambda value: GammaPrior(value, 1.0),
            lambda value: GammaPrior(1.0, value),
            lambda value: DirichletPrior([1.0, value]),
        )
        for value in (0.0, -1.0, float("nan"), float("inf")):
            for factory in factories:
                with self.subTest(value=value, factory=factory), self.assertRaises(ValueError):
                    factory(value)
        with self.assertRaises(ValueError):
            NormalGammaPrior(mu0=float("nan"))
        with self.assertRaises(ValueError):
            DirichletPrior([])
        with self.assertRaises(ValueError):
            DirichletPrior([[1.0], [2.0]])

    def test_improper_limits_require_and_serialize_a_nonempty_receipt(self):
        with self.assertRaises(ValueError):
            ImproperPriorReceipt("")
        acknowledgement = improper("Haldane limit for a sensitivity analysis")
        prior = BetaPrior(0.0, 0.0, improper_receipt=acknowledgement)
        payload = prior.as_dict()
        self.assertFalse(payload["proper"])
        self.assertEqual(payload["improper_receipt"]["status"], "improper_limit_acknowledged")
        self.assertIn("sensitivity", payload["improper_receipt"]["rationale"])

        NormalGammaPrior(kappa=0.0, beta=0.0, improper_receipt=acknowledgement)
        GammaPrior(0.0, 0.0, improper_receipt=acknowledgement)
        DirichletPrior([0.0, 0.0], improper_receipt=acknowledgement)
        with self.assertRaisesRegex(ValueError, "every hyperparameter defines a proper prior"):
            BetaPrior(1.0, 1.0, improper_receipt=acknowledgement)
        with self.assertRaisesRegex(ValueError, "cannot use the analytic conjugate estimator"):
            GaussianEstimator(
                prior=NormalGammaPrior(
                    kappa=0.0,
                    beta=0.0,
                    improper_receipt=acknowledgement,
                )
            )

    def test_stats_normal_gamma_rejects_invalid_construction_and_mutation(self):
        for params in (
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0, 0.0),
            (float("nan"), 1.0, 1.0, 1.0),
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                NormalGammaDistribution(*params)
        prior = NormalGammaDistribution(0.0, 1.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            prior.set_parameters((0.0, 0.0, 1.0, 1.0))

    def test_conjugate_default_is_proper_receipted_and_does_not_mutate_input(self):
        prototype = GaussianDistribution(0.0, 1.0)
        result = sample_conjugate_posterior(prototype, [-1.0, 0.0, 2.0], draws=3, seed=1)
        self.assertIsNone(prototype.get_prior())
        self.assertEqual(result.receipt.prior_source, "weak_proper_default")
        self.assertEqual(result.receipt.prior_status, "proper")
        self.assertEqual(result.receipt.posterior_status, "proper")
        self.assertFalse(result.receipt.input_mutated)
        self.assertEqual(len(result.samples), 3)

    def test_conjugate_improper_limit_needs_receipt_and_reports_posterior_status(self):
        prior = NormalGammaDistribution(0.0, 1.0, 1.0, 1.0)
        prototype = GaussianDistribution(0.0, 1.0, prior=prior)
        # Simulate a legacy/deserialized improper limit that predates constructor validation.
        prior.lam = 0.0
        prior.b = 0.0
        with self.assertRaisesRegex(ValueError, "requires an explicit"):
            sample_conjugate_posterior(prototype, [-1.0, 0.0, 2.0], draws=2, seed=2)

        acknowledgement = improper("legacy Jeffreys-limit migration")
        result = sample_conjugate_posterior(
            prototype,
            [-1.0, 0.0, 2.0],
            draws=2,
            seed=2,
            improper_receipt=acknowledgement,
        )
        self.assertEqual(result.receipt.prior_status, "improper")
        self.assertEqual(result.receipt.posterior_status, "proper")
        self.assertEqual(result.receipt.improper_prior["status"], "improper_limit_acknowledged")

        with self.assertRaises(ImproperPosteriorError) as caught:
            sample_conjugate_posterior(
                prototype,
                [],
                draws=0,
                seed=2,
                improper_receipt=acknowledgement,
            )
        self.assertEqual(caught.exception.receipt.posterior_status, "invalid_or_improper")

    def test_conjugate_draw_count_is_an_integer_control(self):
        with self.assertRaises(ValueError):
            sample_conjugate_posterior(GaussianDistribution(0.0, 1.0), [0.0], draws=1.5)
        with self.assertRaises(ValueError):
            sample_conjugate_posterior(GaussianDistribution(0.0, 1.0), [0.0], draws=np.int64(-1))


if __name__ == "__main__":
    unittest.main()
