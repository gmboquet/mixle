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


class ImpossibleReceiptTest(unittest.TestCase):
    """A receipt exists to be trusted about the run it describes, so it must be consistent."""

    def _make(self, **overrides):
        kwargs = dict(
            algorithm="em",
            estimator="GaussianEstimator",
            objective="mle",
            iterations=3,
            max_iterations=50,
            converged=True,
        )
        kwargs.update(overrides)
        return FitProvenance(**kwargs)

    def test_iterations_cannot_exceed_the_cap(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed max_iterations"):
            self._make(iterations=99, max_iterations=5)

    def test_convergence_after_zero_iterations_is_refused(self):
        with self.assertRaisesRegex(ValueError, "zero iterations"):
            self._make(iterations=0, converged=True)

    def test_an_anonymous_algorithm_is_refused(self):
        with self.assertRaisesRegex(ValueError, "non-empty name"):
            self._make(algorithm="   ")

    def test_a_negative_iteration_count_is_refused(self):
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            self._make(iterations=-1)

    def test_a_non_finite_objective_becomes_an_absent_measurement(self):
        # The first EM iteration has no baseline, so its gain is legitimately infinite -- but that is
        # not strict JSON, and a receipt that cannot serialize is a receipt that gets dropped.
        provenance = self._make(objective_gain=float("inf"), final_objective=float("-inf"))
        self.assertIsNone(provenance.objective_gain)
        self.assertIsNone(provenance.final_objective)


class ProvenanceRoundTripTest(unittest.TestCase):
    """A persisted fitted model keeps its receipt (MXR-080-1190/1202)."""

    def test_provenance_survives_dict_and_json_round_trips(self):
        fitted = optimize(_normal(), GaussianEstimator(), max_its=30)
        original = fitted.fit_provenance()
        self.assertEqual(GaussianDistribution.from_dict(fitted.to_dict()).fit_provenance(), original)
        self.assertEqual(GaussianDistribution.from_json(fitted.to_json()).fit_provenance(), original)

    def test_a_hand_built_model_round_trips_without_a_receipt(self):
        plain = GaussianDistribution(0.0, 1.0)
        self.assertIsNone(GaussianDistribution.from_dict(plain.to_dict()).fit_provenance())

    def test_a_forged_receipt_is_refused_on_decode(self):
        from mixle.utils.serialization import SerializationError

        payload = optimize(_normal(), GaussianEstimator(), max_its=30).to_dict()
        payload["fit_provenance"]["iterations"] = 9999  # more iterations than its own cap
        with self.assertRaises(SerializationError):
            GaussianDistribution.from_dict(payload)

    def test_provenance_is_not_part_of_model_identity(self):
        # A fingerprint identifies PARAMETERS. Two runs landing on the same parameters are the same
        # model however many iterations each took.
        from mixle.data.hashing import model_hash

        fitted = optimize(_normal(), GaussianEstimator(), max_its=30)
        plain = GaussianDistribution(fitted.mu, fitted.sigma2)
        self.assertEqual(model_hash(fitted), model_hash(plain))


if __name__ == "__main__":
    unittest.main()
