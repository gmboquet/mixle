"""Focused contract tests for canonical causal semantics."""

import unittest

from mixle.causal import (
    AssumptionStatus,
    CausalAssumption,
    CausalContractError,
    CausalEvidenceKind,
    Estimand,
    IdentificationResult,
    IdentificationStatus,
    InterventionSpec,
    canonical_json,
    semantic_id,
)


class CausalContractTest(unittest.TestCase):
    def setUp(self):
        self.estimand = Estimand("ate", "dose", "response", "eligible adults", "E[Y(1)-Y(0)]")
        self.exchangeability = CausalAssumption("a1", "exchangeability", "no unmeasured confounding")

    def test_estimand_identity_and_round_trip_are_deterministic(self):
        restored = Estimand.from_dict(self.estimand.as_dict())
        self.assertEqual(restored, self.estimand)
        self.assertEqual(restored.identity, self.estimand.identity)

    def test_identified_result_requires_expression_and_live_assumptions(self):
        result = IdentificationResult(
            self.estimand.identity,
            IdentificationStatus.IDENTIFIED,
            (self.exchangeability,),
            CausalEvidenceKind.INTERVENTION,
            identifying_expression="E[Y|do(dose=1)]-E[Y|do(dose=0)]",
        )
        self.assertEqual(IdentificationResult.from_dict(result.as_dict()), result)
        failed = CausalAssumption("a1", "exchangeability", "no unmeasured confounding", status=AssumptionStatus.FAILED)
        with self.assertRaisesRegex(CausalContractError, "failed assumptions"):
            IdentificationResult(
                self.estimand.identity,
                IdentificationStatus.IDENTIFIED,
                (failed,),
                CausalEvidenceKind.ASSOCIATION,
                identifying_expression="invalid",
            )

    def test_partial_and_not_identified_results_cannot_fabricate_certainty(self):
        partial = IdentificationResult(
            self.estimand.identity,
            IdentificationStatus.PARTIALLY_IDENTIFIED,
            (self.exchangeability,),
            CausalEvidenceKind.ASSOCIATION,
            lower_bound=-0.2,
            upper_bound=0.7,
        )
        self.assertEqual(partial.as_dict()["status"], "partially_identified")
        with self.assertRaisesRegex(CausalContractError, "diagnostic"):
            IdentificationResult(
                self.estimand.identity,
                IdentificationStatus.NOT_IDENTIFIED,
                (self.exchangeability,),
                CausalEvidenceKind.ASSOCIATION,
            )

    def test_identification_refuses_unresolved_assumptions_and_prediction_evidence(self):
        # MXR-080-1675: only FAILED assumptions were rejected and evidence_kind was unconstrained, so an
        # "identified" record could rest on a CHALLENGED exchangeability assumption with
        # evidence_kind="prediction" and the purely observational expression E[Y|X].
        challenged = CausalAssumption(
            id="a1",
            kind="exchangeability",
            statement="no unmeasured confounding",
            status=AssumptionStatus.CHALLENGED,
        )
        with self.assertRaisesRegex(CausalContractError, "challenged"):
            IdentificationResult(
                estimand_ref=self.estimand.identity,
                status=IdentificationStatus.IDENTIFIED,
                assumptions=(challenged,),
                evidence_kind=CausalEvidenceKind.ASSOCIATION,
                identifying_expression="E[Y|X]",
            )
        for status in (IdentificationStatus.IDENTIFIED, IdentificationStatus.PARTIALLY_IDENTIFIED):
            with self.subTest(status=repr(status)), self.assertRaisesRegex(CausalContractError, "prediction-only"):
                IdentificationResult(
                    estimand_ref=self.estimand.identity,
                    status=status,
                    assumptions=(self.exchangeability,),
                    evidence_kind=CausalEvidenceKind.PREDICTION,
                    identifying_expression="E[Y|X]" if status is IdentificationStatus.IDENTIFIED else None,
                    lower_bound=None if status is IdentificationStatus.IDENTIFIED else -0.2,
                    upper_bound=None if status is IdentificationStatus.IDENTIFIED else 0.7,
                )
        failed = CausalAssumption(
            id="a1", kind="exchangeability", statement="no unmeasured confounding", status=AssumptionStatus.FAILED
        )
        with self.assertRaisesRegex(CausalContractError, "failed assumptions"):
            IdentificationResult(
                estimand_ref=self.estimand.identity,
                status=IdentificationStatus.PARTIALLY_IDENTIFIED,
                assumptions=(failed,),
                evidence_kind=CausalEvidenceKind.ASSOCIATION,
                lower_bound=-0.2,
                upper_bound=0.7,
            )
        # an unresolved assumption is still perfectly reportable -- as a downgraded status
        downgraded = IdentificationResult(
            estimand_ref=self.estimand.identity,
            status=IdentificationStatus.NOT_IDENTIFIED,
            assumptions=(challenged,),
            evidence_kind=CausalEvidenceKind.PREDICTION,
            diagnostics=("exchangeability challenged",),
        )
        self.assertEqual(downgraded.as_dict()["status"], "not_identified")

    def test_semantic_identity_does_not_erase_distinct_mapping_keys(self):
        # MXR-080-1676: str(key) inside a dict comprehension let distinct keys overwrite before hashing,
        # so semantic_id({1: "first", "1": "last"}) equaled semantic_id({"1": "last"}) and the canonical
        # payload retained only the later value.
        with self.assertRaisesRegex(CausalContractError, "string mapping keys"):
            semantic_id({1: "first", "1": "last"})
        with self.assertRaises(CausalContractError):
            canonical_json({"x": float("nan")})
        self.assertNotEqual(semantic_id({"1": "first"}), semantic_id({"1": "last"}))
        self.assertEqual(canonical_json({"b": 1, "a": 2}), '{"a":2,"b":1}')

    def test_intervention_range_and_authority_are_distinct(self):
        intervention = InterventionSpec("dose-1", "dose", 0.5, 0.0, 1.0, None, ("interlock",))
        self.assertFalse(intervention.authorized)
        with self.assertRaisesRegex(CausalContractError, "safe range"):
            InterventionSpec("dose-2", "dose", 2.0, 0.0, 1.0, "authority://1")


if __name__ == "__main__":
    unittest.main()
