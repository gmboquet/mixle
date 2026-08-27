"""Release-wave-3 regressions for the multivariate Gaussian (B5, B6, and a mixture shape error).

B5: the estimator's default covariance ridge ``eps = max(min_covar, ridge * trace/d)`` was scaled
by the MEAN diagonal variance, so on heterogeneous-unit data it could inflate the smallest
variances far beyond its nominal ~1e-6 relative size while nothing recorded the repair. Any
materially binding regularizer must be disclosed in the estimator docstring and recorded through
``numerical_repairs()`` / ``fit_provenance()`` -- and stay silent on fits it does not bind. (Since
T1-05 the default ridge is priced per coordinate, ``eps_i = ridge * cov_ii``, so the DEFAULT no
longer binds on heterogeneous units at all; the disclosure tests below pin the cases that still
do: an explicit absolute ``min_covar``, and the rank-deficient rescue.)

B6: the serialized state embedded scipy's raw ``cho_factor`` tuple, whose ``lower``-flag Python type
and unused-triangle content are scipy-version implementation details (they changed at scipy 1.18).
A deployed artifact therefore refused to load under a different in-range scipy, both directions.
Serialization must be parameter-only, with legacy full-state artifacts still readable.

t3-minor: ``GaussianMixtureDistribution`` raised a bare ``IndexError`` on 1-D component means; it
must name the expected ``(K, d)`` shape instead (fix lives in ``mixle/stats/latent``).
"""

import json
import tempfile
import unittest

import numpy as np
import scipy.linalg

import mixle
from mixle.stats.latent.gaussian_mixture import GaussianMixtureDistribution
from mixle.stats.multivariate.multivariate_gaussian import (
    MultivariateGaussianDistribution,
    MultivariateGaussianEstimator,
)
from mixle.utils.serialization import (
    OBJECT_SCHEMA_VERSION,
    TAG,
    SerializationError,
    from_serializable,
    to_serializable,
)

_MVN_TID = "mixle.stats.multivariate.multivariate_gaussian.MultivariateGaussianDistribution"


def _heterogeneous_data(n=203, seed=42):
    """Columns with wildly different units (~3200, ~3.2, ~1.5 scales) -- the B5 failure regime."""
    rng = np.random.default_rng(seed)
    x = np.column_stack(
        [
            3000.0 + 3200.0 * rng.standard_normal(n),
            4.0 + 3.2 * rng.standard_normal(n),
            5.8 + 1.5 * rng.standard_normal(n),
        ]
    )
    return [tuple(v) for v in x], x


class _EmulatedChoFactorConvention:
    """Temporarily make ``scipy.linalg.cho_factor`` return one specific historical convention.

    scipy <= 1.17 returns ``(factor with input scratch in the unused triangle, bool flag)``;
    scipy >= 1.18 returns ``(factor with the unused triangle zeroed, 0-d ndarray flag)``. Emulating
    each explicitly makes the cross-version test independent of which scipy runs the suite.
    """

    def __init__(self, style):
        assert style in ("pre118", "post118")
        self.style = style

    def __enter__(self):
        self._real = scipy.linalg.cho_factor

        def emulated(a, *args, **kwargs):
            c, lower = self._real(a, *args, **kwargs)
            a_arr = np.asarray(a, dtype=float)
            if self.style == "post118":
                c = np.tril(c) if lower else np.triu(c)
                return c, np.asarray(bool(lower))
            keep = np.tril(c) if lower else np.triu(c)
            scratch = np.triu(a_arr, 1) if lower else np.tril(a_arr, -1)
            return keep + scratch, bool(lower)

        scipy.linalg.cho_factor = emulated
        return self

    def __exit__(self, *exc):
        scipy.linalg.cho_factor = self._real
        return False


class RidgeDisclosureTest(unittest.TestCase):
    """B5: a materially binding default ridge must be visible; a nominal one must stay silent."""

    def test_default_ridge_no_longer_binds_on_heterogeneous_units(self):
        # UPDATED for T1-05: this test used to pin the DEFECT half of B5 -- the trace-mean default
        # ridge binding materially on heterogeneous-unit data (smallest variance inflated >1.5x) and
        # being disclosed. The ridge is now priced per coordinate (eps_i = ridge * cov_ii), so on
        # the same data the default is a uniform ~1e-6 relative perturbation: nothing binds, and
        # there is nothing to disclose. The disclosure contract itself is pinned on the case that
        # still binds, in the companion test below.
        data, x = _heterogeneous_data()
        model = mixle.inference.fit(data, MultivariateGaussianEstimator(dim=3), max_its=50, delta=None)
        self.assertEqual(model.numerical_repairs(), ())
        empirical = np.diag(np.cov(x.T, bias=True))
        np.testing.assert_allclose(np.diag(np.asarray(model.covar)), empirical, rtol=1e-5)

    def test_material_explicit_floor_is_recorded_in_repairs_and_provenance(self):
        # The B5 disclosure contract on the same heterogeneous data: a regularizer that DOES bind
        # materially (here an explicit absolute min_covar dominating the smallest variance) must be
        # visible in numerical_repairs() and fit_provenance().
        data, x = _heterogeneous_data()
        model = mixle.inference.fit(data, MultivariateGaussianEstimator(dim=3, min_covar=10.0), max_its=50, delta=None)
        self.assertTrue(
            any(repair.startswith("covariance-ridged(") for repair in model.numerical_repairs()),
            model.numerical_repairs(),
        )
        provenance = model.fit_provenance()
        self.assertIsNotNone(provenance)
        self.assertTrue(any("covariance-ridged" in repair for repair in provenance.repairs), provenance.repairs)
        # and the floor really did bind: the smallest fitted variance exceeds the empirical one
        empirical = np.diag(np.cov(x.T, bias=True))
        self.assertGreater(float(np.diag(np.asarray(model.covar)).min()), 1.5 * float(empirical.min()))

    def test_ridge_zero_recovers_the_mle_and_records_nothing(self):
        data, x = _heterogeneous_data()
        model = mixle.inference.fit(data, MultivariateGaussianEstimator(dim=3, ridge=0.0), max_its=50, delta=None)
        self.assertEqual(
            [r for r in model.numerical_repairs() if "covariance-ridged" in r],
            [],
        )
        empirical = np.cov(x.T, bias=True)
        # exact up to the documented min_covar=1e-8 absolute floor
        np.testing.assert_allclose(np.asarray(model.covar), empirical, rtol=1e-9, atol=2e-8)

    def test_scale_homogeneous_fit_stays_silent(self):
        # the design regime: same-unit coordinates, ridge is a ~1e-6 relative perturbation
        rng = np.random.default_rng(7)
        data = [tuple(v) for v in rng.standard_normal((200, 3))]
        model = mixle.inference.fit(data, MultivariateGaussianEstimator(dim=3), max_its=30, delta=None)
        self.assertEqual(model.numerical_repairs(), ())

    def test_zero_scatter_ridge_is_recorded_as_a_degenerate_rescue(self):
        # one observation -> zero scatter matrix -> the ridge manufactures every variance
        est = MultivariateGaussianEstimator(dim=2)
        acc = est.accumulator_factory().make()
        acc.update(np.array([1.0, 2.0]), 1.0, None)
        fitted = est.estimate(None, acc.value())
        self.assertTrue(
            any("onto a non-positive variance" in repair for repair in fitted.numerical_repairs()),
            fitted.numerical_repairs(),
        )

    def test_empty_statistics_identity_prior_stays_silent(self):
        fitted = MultivariateGaussianEstimator(dim=2).estimate(None, (None, None, 0.0))
        self.assertEqual([r for r in fitted.numerical_repairs() if "covariance-ridged" in r], [])

    def test_ridge_is_disclosed_in_the_class_docstring(self):
        # the tester read ``__doc__`` (as docs sites render it) and found a bare one-liner; the
        # regularization and its escape hatch must be discoverable there, not only in __init__.
        doc = MultivariateGaussianEstimator.__doc__
        self.assertIn("ridge", doc)
        # UPDATED for T1-05: the pinned formula token was "trace" while the ridge was trace-scaled;
        # the ridge is now per-coordinate, and the docstring must state the formula actually applied.
        self.assertIn("ridge * cov_ii", doc)
        self.assertIn("ridge=0.0", doc)
        self.assertIn("numerical_repairs", doc)
        # the measured-false claim must be gone everywhere it appeared
        self.assertNotIn("Bias is negligible at the defaults", doc)
        self.assertNotIn("Bias is negligible at the defaults", MultivariateGaussianEstimator.__init__.__doc__)


class CholeskyNormalizationTest(unittest.TestCase):
    """B6 (state hygiene): the stored factor must not carry scipy-version-specific bytes."""

    def test_chol_flag_is_plain_bool_and_scratch_triangle_is_zeroed(self):
        for style in ("pre118", "post118"):
            with _EmulatedChoFactorConvention(style):
                dist = MultivariateGaussianDistribution([1.0, -1.0], [[2.0, 0.4], [0.4, 1.0]])
            factor, lower = dist.chol
            self.assertIs(type(lower), bool, style)
            scratch = np.tril(factor, -1) if not lower else np.triu(factor, 1)
            self.assertTrue(np.array_equal(scratch, np.zeros_like(scratch)), style)

    def test_scoring_is_unaffected_by_normalization(self):
        from scipy.stats import multivariate_normal

        dist = MultivariateGaussianDistribution([1.0, -1.0], [[2.0, 0.4], [0.4, 1.0]])
        probe = np.array([0.3, 0.2])
        self.assertAlmostEqual(
            dist.log_density(probe),
            float(multivariate_normal.logpdf(probe, mean=[1.0, -1.0], cov=[[2.0, 0.4], [0.4, 1.0]])),
            places=12,
        )


class SerializationPortabilityTest(unittest.TestCase):
    """B6: parameter-only artifacts, identical across scipy conventions, legacy form still readable."""

    def _dist(self):
        return MultivariateGaussianDistribution([1.0, -1.0], [[2.0, 0.4], [0.4, 1.0]], name="n", keys="k")

    def test_serialized_state_is_parameter_only(self):
        payload = to_serializable(self._dist())
        fields = sorted(pair[0] for pair in payload["state"]["items"])
        self.assertEqual(fields, ["covar", "keys", "mu", "name", "prior"])

    def test_round_trip_preserves_scoring(self):
        dist = self._dist()
        loaded = from_serializable(json.loads(json.dumps(to_serializable(dist))))
        probe = np.array([0.3, 0.2])
        self.assertEqual(loaded.log_density(probe), dist.log_density(probe))
        self.assertEqual(loaded.name, "n")
        self.assertEqual(loaded.keys, "k")

    def test_serialized_bytes_are_identical_across_cho_factor_conventions(self):
        payloads = {}
        for style in ("pre118", "post118"):
            with _EmulatedChoFactorConvention(style):
                dist = MultivariateGaussianDistribution([1.0, -1.0], [[2.0, 0.4], [0.4, 1.0]])
                payloads[style] = json.dumps(to_serializable(dist), sort_keys=True)
        self.assertEqual(payloads["pre118"], payloads["post118"])

    def test_artifact_saved_under_either_convention_loads_under_the_real_scipy(self):
        probe = np.array([0.3, 0.2])
        want = self._dist().log_density(probe)
        for style in ("pre118", "post118"):
            with _EmulatedChoFactorConvention(style):
                payload = json.loads(json.dumps(to_serializable(self._dist())))
            loaded = from_serializable(payload)  # decoded by the environment's real scipy
            self.assertEqual(loaded.log_density(probe), want, style)

    def test_legacy_full_state_artifacts_still_load_in_both_conventions(self):
        # early-0.8.0 artifacts persisted the whole __dict__, chol included, in whichever
        # convention the writing scipy used. Both variants must decode, and the decoded factor
        # must come out normalized (recomputed, not adopted).
        dist = self._dist()
        probe = np.array([0.3, 0.2])
        want = dist.log_density(probe)
        covar = np.asarray(dist.covar)
        legacy_chols = {
            "pre118": (np.triu(dist.chol[0]) + np.tril(covar, -1), False),
            "post118": (np.triu(dist.chol[0]), np.asarray(False)),
        }
        for style, chol in legacy_chols.items():
            state = dict(dist.__dict__)
            state.pop("_numerical_repairs", None)
            state["chol"] = chol
            envelope = {
                TAG: "object",
                "type": _MVN_TID,
                "schema_version": OBJECT_SCHEMA_VERSION,
                "state": to_serializable(state),
            }
            loaded = from_serializable(json.loads(json.dumps(envelope)))
            self.assertEqual(loaded.log_density(probe), want, style)
            self.assertIs(type(loaded.chol[1]), bool, style)

    def test_univariate_mixture_of_mvn_components_round_trips_across_conventions(self):
        # the tester's gmm1d: a 1-D Gaussian mixture whose components are 1x1 MVNs
        probe = np.array([0.5])
        with _EmulatedChoFactorConvention("post118"):
            mix = GaussianMixtureDistribution(mu=[[0.0], [3.0]], sig2=[[1.0], [2.0]], w=[0.4, 0.6])
            payload = json.loads(json.dumps(to_serializable(mix)))
            want = mix.log_density(probe)
        loaded = from_serializable(payload)
        self.assertAlmostEqual(loaded.log_density(probe), want, places=15)

    def test_model_deploy_then_load_across_conventions(self):
        # the exact reported pipeline: Model.fit -> deploy under one scipy, Model.load under another
        rng = np.random.default_rng(0)
        data = [list(map(float, row)) for row in rng.normal(size=(200, 2))]
        with tempfile.TemporaryDirectory() as tmp:
            art = tmp + "/artifact"
            with _EmulatedChoFactorConvention("post118"):
                model = mixle.Model(
                    MultivariateGaussianDistribution(mu=[0.0, 0.0], covar=[[1.0, 0.0], [0.0, 1.0]])
                ).fit(data)
                model.deploy(art)
            loaded = mixle.Model.load(art)  # the environment's real scipy convention
            enc = loaded.fitted.dist_to_encoder().seq_encode(np.asarray(data))
            reference = model.fitted.dist_to_encoder().seq_encode(np.asarray(data))
            np.testing.assert_array_equal(loaded.fitted.seq_log_density(enc), model.fitted.seq_log_density(reference))

    def test_setstate_rejects_unknown_fields_by_name(self):
        payload = to_serializable(self._dist())
        payload["state"]["items"].append(["injected", True])
        with self.assertRaisesRegex(SerializationError, "injected"):
            from_serializable(json.loads(json.dumps(payload)))

    def test_setstate_rejects_missing_parameters_by_name(self):
        payload = to_serializable(self._dist())
        payload["state"]["items"] = [pair for pair in payload["state"]["items"] if pair[0] != "covar"]
        with self.assertRaisesRegex(SerializationError, "missing required parameters: covar"):
            from_serializable(json.loads(json.dumps(payload)))

    def test_setstate_rejects_a_dim_inconsistent_legacy_state(self):
        dist = self._dist()
        state = dict(dist.__dict__)
        state.pop("_numerical_repairs", None)
        state["dim"] = 5
        envelope = {
            TAG: "object",
            "type": _MVN_TID,
            "schema_version": OBJECT_SCHEMA_VERSION,
            "state": to_serializable(state),
        }
        with self.assertRaisesRegex(SerializationError, "dim=5"):
            from_serializable(json.loads(json.dumps(envelope)))

    def test_invalid_parameters_fail_with_a_serialization_error(self):
        payload = to_serializable(self._dist())
        for pair in payload["state"]["items"]:
            if pair[0] == "covar":
                pair[1] = to_serializable(np.array([[-1.0, 0.0], [0.0, -1.0]]))
        with self.assertRaises(SerializationError):
            from_serializable(json.loads(json.dumps(payload)))


class GaussianMixtureShapeErrorTest(unittest.TestCase):
    """t3-minor: 1-D component means must raise a shaped ValueError, not a bare IndexError."""

    def test_1d_component_means_raise_a_value_error_naming_the_expected_shape(self):
        with self.assertRaisesRegex(ValueError, r"shape \(K, d\)"):
            GaussianMixtureDistribution(mu=[55.0, 80.0], sig2=[36.0, 36.0], w=[0.5, 0.5])

    def test_2d_component_means_still_construct(self):
        mix = GaussianMixtureDistribution(mu=[[55.0], [80.0]], sig2=[[36.0], [36.0]], w=[0.5, 0.5])
        self.assertEqual(mix.dim, 1)


if __name__ == "__main__":
    unittest.main()
