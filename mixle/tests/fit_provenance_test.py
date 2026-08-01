"""A fitted distribution says how it was fitted (MXR-080-1190 / MXR-080-1202).

``density_semantics()`` already distinguishes an exact density from a bound or an unnormalized factor.
It says nothing about *fitting*: before this, a ``GaussianDistribution`` returned by ``optimize`` was
byte-identical to one written by hand, so a consumer could not tell a converged fit from one that hit
its iteration cap, nor learn that a variance had been clamped or a covariance jitter-healed.
"""

import unittest

import numpy as np

from mixle.inference import optimize
from mixle.stats import (
    GaussianDistribution,
    GaussianEstimator,
    MixtureEstimator,
    MultivariateGaussianDistribution,
)
from mixle.stats.compute.pdist import FitProvenance


def _normal(n=200, seed=0):
    return list(np.random.RandomState(seed).normal(size=n))


class ProvenancePresenceTest(unittest.TestCase):
    def test_a_hand_built_distribution_has_no_fit_provenance(self):
        # None means "no fit produced this", which is different from a fit that produced no receipt.
        self.assertIsNone(GaussianDistribution(0.0, 1.0).fit_provenance())

    def test_a_fitted_distribution_carries_one(self):
        fitted = optimize(_normal(), GaussianEstimator(), max_its=30)
        provenance = fitted.fit_provenance()
        self.assertIsInstance(provenance, FitProvenance)
        self.assertEqual(provenance.estimator, "GaussianEstimator")
        self.assertEqual(provenance.objective, "mle")
        self.assertIn("em", provenance.algorithm)

    def test_the_receipt_is_json_compatible(self):
        record = optimize(_normal(), GaussianEstimator(), max_its=30).fit_provenance().as_dict()
        import json

        self.assertEqual(json.loads(json.dumps(record))["estimator"], "GaussianEstimator")


class ConvergenceTest(unittest.TestCase):
    def test_a_converged_fit_reports_convergence_below_its_cap(self):
        provenance = optimize(_normal(), GaussianEstimator(), max_its=50, delta=1e-9).fit_provenance()
        self.assertTrue(provenance.converged)
        self.assertLess(provenance.iterations, provenance.max_iterations)
        self.assertFalse(provenance.is_approximate())

    def test_a_capped_fit_reports_that_it_never_converged(self):
        # Two iterations of a two-component mixture on 200 points cannot reach a 1e-12 gain.
        estimator = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        fitted = optimize(_normal(), estimator, max_its=2, delta=1e-12, rng=np.random.RandomState(1))
        provenance = fitted.fit_provenance()
        self.assertFalse(provenance.converged)
        self.assertEqual(provenance.iterations, provenance.max_iterations)
        self.assertTrue(provenance.is_approximate())  # a stopped-early fit is an approximation

    def test_the_receipt_records_the_data_it_consumed(self):
        provenance = optimize(_normal(n=137), GaussianEstimator(), max_its=10).fit_provenance()
        self.assertEqual(provenance.n_observations, 137)


class NumericalRepairTest(unittest.TestCase):
    def test_an_unrepaired_fit_records_no_repairs(self):
        self.assertEqual(optimize(_normal(), GaussianEstimator(), max_its=30).fit_provenance().repairs, ())

    def test_a_bound_variance_floor_is_recorded(self):
        # Constant data implies zero variance; the floor keeps the density finite but the returned
        # parameter is no longer the one the data implied.
        provenance = optimize([1.0] * 60, GaussianEstimator(), max_its=5).fit_provenance()
        self.assertTrue(any("variance-floored" in repair for repair in provenance.repairs))
        self.assertTrue(provenance.is_approximate())

    def test_a_jitter_healed_covariance_is_recorded_on_the_distribution(self):
        singular = np.outer([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])  # rank 1 in three dimensions
        healed = MultivariateGaussianDistribution([0.0, 0.0, 0.0], singular)
        self.assertTrue(any("covariance-jitter-healed" in repair for repair in healed.numerical_repairs()))

    def test_a_positive_definite_covariance_needs_no_repair(self):
        clean = MultivariateGaussianDistribution([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(clean.numerical_repairs(), ())


class ProvenanceContractTest(unittest.TestCase):
    def test_repairs_alone_make_a_fit_approximate(self):
        converged_but_repaired = FitProvenance(
            algorithm="em",
            estimator="GaussianEstimator",
            objective="mle",
            iterations=5,
            max_iterations=50,
            converged=True,
            repairs=("variance-floored(0 -> 1e-08)",),
        )
        self.assertTrue(converged_but_repaired.is_approximate())

    def test_a_clean_converged_fit_is_not_approximate(self):
        clean = FitProvenance(
            algorithm="em",
            estimator="GaussianEstimator",
            objective="mle",
            iterations=5,
            max_iterations=50,
            converged=True,
        )
        self.assertFalse(clean.is_approximate())

    def test_with_fit_provenance_rejects_a_non_receipt(self):
        with self.assertRaisesRegex(TypeError, "FitProvenance"):
            GaussianDistribution(0.0, 1.0).with_fit_provenance({"converged": True})


if __name__ == "__main__":
    unittest.main()
