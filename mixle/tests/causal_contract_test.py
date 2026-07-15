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

    def test_intervention_range_and_authority_are_distinct(self):
        intervention = InterventionSpec("dose-1", "dose", 0.5, 0.0, 1.0, None, ("interlock",))
        self.assertFalse(intervention.authorized)
        with self.assertRaisesRegex(CausalContractError, "safe range"):
            InterventionSpec("dose-2", "dose", 2.0, 0.0, 1.0, "authority://1")


if __name__ == "__main__":
    unittest.main()
