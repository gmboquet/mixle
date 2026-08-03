"""MXR-080-1901: exact limiting laws, exact design geometry, and typed controls in ``mixle.doe``.

Each test names the sub-defect it reproduces. The finding, verbatim:

    "EI/PI/feasibility collapse every standard deviation below a threshold rather than evaluate the
    limiting law; tolerant pseudocomponent sums admit invalid blends; modality counts truncate;
    negative/non-finite policy weights pass; propagators overwrite outputs; Hessian calibration does
    not prove its positive-definite claim; result records can contradict controls. Establish exact
    design geometry and typed abstention/approximation evidence."
"""

import unittest

import numpy as np
from scipy.stats import norm

from mixle.doe.bayesopt import expected_improvement, probability_of_improvement
from mixle.doe.calibrate import _is_positive_definite
from mixle.doe.constrained import probability_of_feasibility
from mixle.doe.distillation import cross_modal_distillation_design, distillation_design
from mixle.doe.entropy import max_value_entropy_search
from mixle.doe.mixture import to_pseudocomponents
from mixle.doe.propagate import _PROPAGATORS, propagate, register_propagator


def _legacy_batch_ei(mean, std, best):
    """The exact inline EI ``propose_local_penalization`` used before MXR-080-1901 (minimization)."""
    signed = best - mean
    z = signed / np.maximum(std, 1e-12)
    return np.maximum(signed, 0.0) * norm.cdf(z) + std * norm.pdf(z)


class BatchExpectedImprovementLawTest(unittest.TestCase):
    """REPRODUCED: ``batch.propose_local_penalization``'s inline EI used the wrong closed form AND
    substituted a clamped sigma for the sigma -> 0 limit."""

    def test_legacy_inline_ei_overstated_non_improving_candidates(self):
        # The defect, reproduced against the old formula: EI is `improve*Phi(z) + sigma*phi(z)`, but the
        # inline copy used `max(improve, 0)*Phi(z)`, deleting the negative term for candidates that do
        # not improve on the incumbent.
        mean, std, best = np.array([2.0]), np.array([1.0]), 1.0
        legacy = _legacy_batch_ei(mean, std, best)[0]
        exact = expected_improvement(mean, std, best)[0]
        self.assertAlmostEqual(exact, 0.08331547, places=7)
        self.assertAlmostEqual(legacy, 0.24197072, places=7)
        self.assertGreater(legacy / exact, 2.9)  # 2.9x overstated

    def test_legacy_inline_ei_reordered_candidates(self):
        # The inflation is z-dependent, so it is not a monotone reparameterization: it changes the
        # ARGMAX, which is what the batch driver selects on. Candidate 0 does not improve on the
        # incumbent (signed = -1) but is uncertain; candidate 1 has a small guaranteed improvement.
        mean = np.array([2.0, 0.9])
        std = np.array([1.0, 0.05])
        best = 1.0
        np.testing.assert_allclose(_legacy_batch_ei(mean, std, best), [0.24197072, 0.10042454])
        np.testing.assert_allclose(expected_improvement(mean, std, best), [0.08331547, 0.10042454])
        self.assertEqual(int(np.argmax(_legacy_batch_ei(mean, std, best))), 0)  # picks the wrong point
        self.assertEqual(int(np.argmax(expected_improvement(mean, std, best))), 1)

    def test_clamped_sigma_understated_the_deterministic_limit(self):
        # The sigma -> 0 limit of EI is exactly max(improve, 0). Dividing by max(std, 1e-12) evaluates
        # the finite-sigma formula at a fictitious sigma instead. `best=0` / `mean=-improve` makes
        # `best - mean` come out exactly `improve` in float64, so the ratios below measure the
        # acquisition rather than the representation error of building the input.
        for improve, expected_ratio in ((1e-13, 0.539828), (1e-12, 0.841345)):
            legacy = _legacy_batch_ei(np.array([-improve]), np.array([0.0]), 0.0)[0]
            self.assertAlmostEqual(legacy / improve, expected_ratio, places=5)
        # The canonical acquisition returns the exact limit for a deterministic candidate.
        for improve in (1e-13, 1e-12, 1e-6, 1.0):
            got = expected_improvement(np.array([-improve]), np.array([0.0]), 0.0)[0]
            self.assertAlmostEqual(got / improve, 1.0, places=12)

    def test_local_penalization_now_uses_the_canonical_acquisition(self):
        # Ties the unit-level law back to the driver: the module-level function it now calls is the
        # canonical one, so the two can no longer diverge.
        from mixle.doe import batch as batch_module

        self.assertIs(batch_module.expected_improvement, expected_improvement)


class DeterministicLimitNegativeControlTest(unittest.TestCase):
    """NEGATIVE CONTROL: EI / PI / probability-of-feasibility already evaluate the sigma -> 0 limiting
    law rather than a clamped stand-in. Verified, not changed -- these lock that in."""

    def test_ei_at_zero_sigma_is_the_guaranteed_improvement(self):
        ei = expected_improvement(np.array([0.5, 1.5]), np.array([0.0, 0.0]), 1.0)
        np.testing.assert_allclose(ei, [0.5, 0.0])  # max(improve, 0), not 0 and not sigma*phi(0)

    def test_ei_is_continuous_into_the_limit(self):
        # The limit is a continuous extension of the finite-sigma formula, not a discontinuous patch.
        improve = 0.25
        for sigma in (1e-8, 1e-10, 1e-11):
            got = expected_improvement(np.array([1.0 - improve]), np.array([sigma]), 1.0)[0]
            self.assertAlmostEqual(got, improve, places=7)

    def test_pi_at_zero_sigma_is_the_indicator(self):
        pi = probability_of_improvement(np.array([0.5, 1.5, 1.0]), np.zeros(3), 1.0)
        np.testing.assert_allclose(pi, [1.0, 0.0, 0.0])  # improve > 0 strictly

    def test_feasibility_at_zero_sigma_is_the_indicator(self):
        pf = probability_of_feasibility(np.array([[-1.0], [1.0], [0.0]]), np.zeros((3, 1)))
        np.testing.assert_allclose(pf, [1.0, 0.0, 1.0])  # c <= 0 is feasible, so 0.0 counts


class MesDeterministicCandidateTest(unittest.TestCase):
    """REPRODUCED: ``max_value_entropy_search`` floored std to 1e-9, so a deterministic candidate was
    credited with log(2) ~= 0.693 nats of information it cannot carry."""

    def test_zero_std_candidate_carries_no_information(self):
        # Candidate 0 is deterministic; y* is clamped to mu.max() == 1.0, exactly as sample_max_values
        # does, which drove gamma to 0 and the old floored formula to log(2).
        info = max_value_entropy_search(np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0]))
        self.assertEqual(info[0], 0.0)
        self.assertGreater(info[1], 0.0)
        # The spurious value used to be the LARGEST merit present, so MES proposed the one candidate
        # guaranteed to teach it nothing.
        self.assertEqual(int(np.argmax(info)), 1)

    def test_tiny_but_nonzero_std_still_uses_the_exact_closed_form(self):
        # Scoped fix: 1e-15 is a legitimate posterior sd, not a stand-in for zero, so it is NOT zeroed.
        info = max_value_entropy_search(np.array([0.0, 1.0]), np.array([0.0, 1e-15]), np.array([1.0, 2.0, 3.0]))
        self.assertTrue(np.all(np.isfinite(info)))
        self.assertEqual(info[0], 0.0)
        self.assertGreater(info[1], 0.0)

    def test_well_posed_moments_are_unchanged(self):
        info = max_value_entropy_search(np.array([0.0, 0.5]), np.array([0.5, 0.5]), np.array([1.0, 2.0]))
        self.assertTrue(np.all(np.isfinite(info)))
        self.assertTrue(np.all(info >= 0.0))


class PseudocomponentSimplexGeometryTest(unittest.TestCase):
    """REPRODUCED: ``to_pseudocomponents``' row-sum check used ``np.allclose``, whose DEFAULT rtol=1e-5
    is ADDED to the passed atol -- so the declared 1e-9 tolerance was really ~1e-5."""

    def test_row_sum_five_thousand_times_the_declared_tolerance_is_rejected(self):
        bad = np.array([[0.5 + 5e-6, 0.5]])  # sums to 1.000005
        self.assertAlmostEqual(float(bad.sum(axis=1)[0]), 1.000005, places=9)
        with self.assertRaisesRegex(ValueError, "each row summing to 1"):
            to_pseudocomponents(bad, [0.1, 0.1])

    def test_the_admitted_blend_used_to_break_the_documented_output_guarantee(self):
        # Why it mattered: the function documents "Rows still sum to 1", and the admitted row produced
        # an output off by ~4e-6 -- four orders of magnitude past the constant that claims to bound it.
        bad = np.array([[0.5 + 5e-6, 0.5]])
        legacy_out = np.asarray([0.1, 0.1]) + (1.0 - 0.2) * bad
        self.assertAlmostEqual(float(legacy_out.sum(axis=1)[0]) - 1.0, 4e-6, places=9)

    def test_genuine_simplex_points_and_roundoff_still_pass(self):
        # Guard-overreach check: the tolerance that IS declared must still be honoured.
        np.testing.assert_allclose(to_pseudocomponents(np.array([[0.5, 0.5]]), [0.1, 0.1]), [[0.5, 0.5]])
        within = np.array([[0.5 + 4e-10, 0.5]])  # inside the real 1e-9 tolerance
        out = to_pseudocomponents(within, [0.1, 0.1])
        self.assertTrue(np.all(np.isfinite(out)))
        # A degenerate-but-valid vertex blend still maps onto the constrained simplex.
        np.testing.assert_allclose(to_pseudocomponents(np.array([[1.0, 0.0]]), [0.2, 0.3]), [[0.7, 0.3]])

    def test_the_error_names_the_offending_rows(self):
        blends = np.array([[0.5, 0.5], [0.7, 0.7]])
        with self.assertRaisesRegex(ValueError, r"row\(s\) \[1\]"):
            to_pseudocomponents(blends, [0.1, 0.1])


class HessianPositiveDefiniteClaimTest(unittest.TestCase):
    """REPRODUCED: ``calibrate`` documents that ``theta_standard_error`` degrades to nan when the
    Hessian "is not ... positive-definite", but only checked invertibility plus a non-negative diagonal
    of the theta block -- neither of which implies positive-definiteness."""

    def test_indefinite_hessian_that_the_old_check_accepted_is_now_refused(self):
        hessian = np.array([[-1.0, 2.0], [2.0, -1.0]])
        np.testing.assert_allclose(np.linalg.eigvalsh(hessian), [-3.0, 1.0])  # indefinite: a saddle
        # The old check: inv() does not raise, and the inverse's diagonal is entirely positive.
        theta_var = np.diag(np.linalg.inv(hessian))
        self.assertTrue(np.all(np.isfinite(theta_var)) and np.all(theta_var >= 0))
        np.testing.assert_allclose(theta_var, [1 / 3, 1 / 3])
        # So it used to report a confident standard error of ~0.577 where it had promised nan.
        self.assertAlmostEqual(float(np.sqrt(theta_var[0])), 0.5773502691896258)
        self.assertFalse(_is_positive_definite(hessian))

    def test_genuinely_positive_definite_hessians_still_produce_a_standard_error(self):
        # Guard-overreach check: a real observed-information matrix must still pass.
        self.assertTrue(_is_positive_definite(np.array([[2.0, 0.0], [0.0, 3.0]])))
        self.assertTrue(_is_positive_definite(np.array([[4.0, 1.0], [1.0, 4.0]])))
        self.assertTrue(_is_positive_definite(np.array([[1e-6]])))  # near-singular but still PD

    def test_negative_definite_singular_and_non_finite_hessians_are_refused(self):
        self.assertFalse(_is_positive_definite(np.array([[-2.0, 0.0], [0.0, -3.0]])))
        self.assertFalse(_is_positive_definite(np.array([[1.0, 1.0], [1.0, 1.0]])))  # singular
        self.assertFalse(_is_positive_definite(np.array([[np.nan, 0.0], [0.0, 1.0]])))
        self.assertFalse(_is_positive_definite(np.array([[np.inf, 0.0], [0.0, 1.0]])))
        self.assertFalse(_is_positive_definite(np.array([[1.0, 2.0], [0.0, 1.0]])))  # asymmetric


class ModalityCountTruncationTest(unittest.TestCase):
    """REPRODUCED: ``cross_modal_distillation_design`` ran ``int(min_modalities)``, silently truncating
    a fractional modality count and accepting a bool."""

    def setUp(self):
        rng = np.random.RandomState(0)
        self.mf = {name: rng.normal(size=(8, 3)) for name in ("a", "b", "c")}
        self.mf["b"][:4, 0] = np.nan  # rows 0-3 have 2 modalities, rows 4-7 have 3

    def _picked(self, **kwargs):
        return sorted(cross_modal_distillation_design(self.mf, 3, seed=0, **kwargs).indices.tolist())

    def test_fractional_modality_count_is_rejected_not_truncated(self):
        strict = self._picked(min_modalities=3)
        self.assertEqual(strict, [5, 6, 7])  # only the 3-modality rows are eligible
        with self.assertRaisesRegex(ValueError, "min_modalities must be an exact integer"):
            cross_modal_distillation_design(self.mf, 3, min_modalities=2.9, seed=0)

    def test_bool_is_rejected_as_a_modality_count(self):
        with self.assertRaises(TypeError):
            cross_modal_distillation_design(self.mf, 3, min_modalities=True, seed=0)

    def test_exact_integers_including_integral_floats_still_work(self):
        # Guard-overreach check: 3 and 3.0 are the same exact count and must both be accepted.
        self.assertEqual(self._picked(min_modalities=3), self._picked(min_modalities=3.0))
        self.assertEqual(len(self._picked(min_modalities=1)), 3)

    def test_nonpositive_modality_count_still_rejected(self):
        with self.assertRaises(ValueError):
            cross_modal_distillation_design(self.mf, 3, min_modalities=0, seed=0)


class PolicyWeightValidationTest(unittest.TestCase):
    """REPRODUCED: negative and non-finite merit weights passed unchecked into
    ``distillation_design``, inverting the documented direction of the term (and being recorded as
    legitimate in the result), or blaming the candidate pool for the caller's bad control."""

    def setUp(self):
        rng = np.random.RandomState(0)
        self.x = rng.normal(size=(10, 3))
        self.uncertainty = rng.uniform(size=10)

    def test_negative_weight_used_to_invert_the_documented_direction(self):
        with self.assertRaisesRegex(ValueError, "uncertainty_weight must be non-negative"):
            distillation_design(self.x, 3, uncertainty=self.uncertainty, uncertainty_weight=-5.0, seed=0)

    def test_result_record_can_no_longer_contradict_the_control_that_produced_it(self):
        # Why it mattered: the old call succeeded, returned the three LEAST uncertain candidates, and
        # recorded metadata["weights"]["uncertainty"] == -5.0 as though that were a valid design.
        design = distillation_design(
            self.x, 3, uncertainty=self.uncertainty, uncertainty_weight=1.0, diversity_weight=0.0, seed=0
        )
        self.assertEqual(design.metadata["weights"]["uncertainty"], 1.0)
        picked = self.uncertainty[design.indices]
        self.assertGreaterEqual(float(picked.min()), float(np.sort(self.uncertainty)[-3]))

    def test_non_finite_weights_name_the_weight_not_the_candidate_pool(self):
        for name, value in (
            ("uncertainty_weight", np.nan),
            ("diversity_weight", np.inf),
            ("cost_weight", -np.inf),
            ("task_coverage_weight", np.nan),
            ("modality_coverage_weight", np.inf),
            ("preference_weight", np.nan),
        ):
            with self.subTest(weight=name):
                with self.assertRaisesRegex(ValueError, f"{name} must be finite"):
                    distillation_design(self.x, 3, uncertainty=self.uncertainty, seed=0, **{name: value})

    def test_alignment_weight_is_validated_too(self):
        rng = np.random.RandomState(1)
        mf = {name: rng.normal(size=(6, 2)) for name in ("a", "b")}
        with self.assertRaisesRegex(ValueError, "alignment_weight must be non-negative"):
            cross_modal_distillation_design(mf, 2, alignment_weight=-1.0, seed=0)

    def test_zero_and_positive_weights_still_accepted(self):
        # Guard-overreach check: zero is the documented way to switch a term off, and the existing
        # distillation tests rely on it. Only negative and non-finite are refused.
        design = distillation_design(
            self.x,
            3,
            uncertainty=self.uncertainty,
            uncertainty_weight=0.0,
            diversity_weight=1.0,
            task_coverage_weight=0.0,
            cost_weight=0.0,
            seed=0,
        )
        self.assertEqual(len(design.indices), 3)

    def test_bool_weight_is_refused_as_a_mis_passed_flag(self):
        with self.assertRaises(ValueError):
            distillation_design(self.x, 3, uncertainty=self.uncertainty, diversity_weight=True, seed=0)


class PropagatorRegistryTest(unittest.TestCase):
    """REPRODUCED: ``register_propagator`` did a bare ``_PROPAGATORS[name] = fn``, so re-registering a
    built-in name silently changed the output of every existing ``propagate(method=...)`` call."""

    def setUp(self):
        self._saved = dict(_PROPAGATORS)
        self.addCleanup(lambda: (_PROPAGATORS.clear(), _PROPAGATORS.update(self._saved)))

    def test_re_registering_a_builtin_name_is_refused(self):
        def impostor(func, mean, cov, *, n, quantiles, seed):
            return {"mean": np.array([999.0]), "std": np.array([0.0])}

        with self.assertRaisesRegex(ValueError, "already registered"):
            register_propagator("montecarlo")(impostor)
        # And the built-in still produces the real answer, not the impostor's.
        result = propagate(lambda x: x[:, 0], np.array([0.0]), np.array([[1.0]]), n=256, seed=0)
        self.assertLess(abs(float(np.asarray(result["mean"]))), 0.5)
        self.assertNotEqual(float(np.asarray(result["mean"])), 999.0)

    def test_registering_a_new_name_still_works(self):
        # Guard-overreach check: the extension point stays open; only silent REPLACEMENT is refused.
        @register_propagator("mxr_1901_probe")
        def probe(func, mean, cov, *, n, quantiles, seed):
            return {"mean": np.array([1.0]), "std": np.array([2.0])}

        self.assertIn("mxr_1901_probe", _PROPAGATORS)
        out = propagate(lambda x: x[:, 0], np.array([0.0]), np.array([[1.0]]), method="mxr_1901_probe")
        np.testing.assert_allclose(out["std"], [2.0])

    def test_malformed_registrations_are_refused(self):
        with self.assertRaises(ValueError):
            register_propagator("")(lambda *a, **k: {})
        with self.assertRaises(ValueError):
            register_propagator(123)(lambda *a, **k: {})
        with self.assertRaises(TypeError):
            register_propagator("mxr_1901_not_callable")(object())


if __name__ == "__main__":
    unittest.main()
