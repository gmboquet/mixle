"""check_factuality (B3): ground an answer's claims against the substrate, per-claim, with citations."""

import unittest

from mixle.substrate import Substrate, check_factuality
from mixle.substrate.factuality import (  # white-box: MXR-080-0258
    Corroboration,
    FactualityReceipt,
    _default_corroborates,
)


def _kb():
    s = Substrate()
    s.add(kind="text", text="Refunds are processed within 30 days of a written request.")
    s.add(kind="text", text="Enterprise support is staffed 24 hours a day, 7 days a week.")
    return s


def _drug_kb():
    s = Substrate()
    s.add(kind="text", text="The drug cures no cancer and does not work.")
    return s


class FactualityTest(unittest.TestCase):
    def test_supported_claim_is_cited(self):
        rec = check_factuality(_kb(), "Refunds are processed within 30 days.")
        self.assertIsInstance(rec, FactualityReceipt)
        self.assertTrue(rec.verdicts[0].supported)
        self.assertFalse(rec.verdicts[0].contradicted)
        self.assertTrue(rec.verdicts[0].citations)  # carries the citing item
        self.assertTrue(rec.is_grounded())

    def test_fabricated_claim_is_flagged_unsupported(self):
        rec = check_factuality(_kb(), "Free accounts include a dedicated account manager.")
        self.assertFalse(rec.verdicts[0].supported)
        self.assertEqual(rec.verdicts[0].citations, [])
        self.assertFalse(rec.is_grounded())

    def test_unrelated_evidence_is_unverified_not_contradicted(self):
        # Control for MXR-080-0258: evidence that shares no content with the claim must never be
        # reported as a contradiction -- it is silent on the claim, which is weaker than disagreeing.
        rec = check_factuality(_kb(), "The moon is made of green cheese.")
        self.assertFalse(rec.verdicts[0].supported)
        self.assertFalse(rec.verdicts[0].contradicted)
        self.assertEqual(rec.verdicts[0].contradictions, [])

    def test_mixed_answer_grounded_fraction(self):
        ans = "Refunds are processed within 30 days. Free accounts include a dedicated account manager."
        rec = check_factuality(_kb(), ans)
        self.assertEqual(len(rec.verdicts), 2)
        self.assertEqual(rec.grounded_fraction, 0.5)
        self.assertEqual(len(rec.unsupported()), 1)

    def test_min_score_guards_against_noise(self):
        # a high floor rejects weak matches, so a loosely-related claim goes unsupported
        rec = check_factuality(_kb(), "Support exists.", min_score=0.9)
        self.assertFalse(rec.verdicts[0].supported)

    def test_as_dict_is_serializable(self):
        rec = check_factuality(_kb(), "Refunds are processed within 30 days.")
        d = rec.as_dict()
        self.assertIn("grounded_fraction", d)
        self.assertEqual(d["n_claims"], 1)
        self.assertIn("n_contradicted", d)
        self.assertEqual(d["n_contradicted"], 0)

    # -- MXR-080-0259 (Critical): an empty verdict list used to report grounded_fraction == 1.0 and
    # is_grounded() == True -- a perfect factuality result for an answer with nothing assessed. --------
    def test_mxr_080_0259_empty_answer_is_unknown_not_grounded(self):
        rec = check_factuality(_kb(), "")
        self.assertIsNone(rec.grounded_fraction)  # unknown, not a vacuous 1.0
        self.assertEqual(rec.verdicts, [])
        self.assertFalse(rec.is_grounded())  # fails closed: nothing was verified

    def test_mxr_080_0259_evasive_answer_with_no_assessable_claims_fails_closed(self):
        # No fragment has >= 2 words, so sentence_claims extracts nothing -- an evasive non-answer,
        # not literally empty, must be treated the same as one: unknown, never "fully grounded".
        rec = check_factuality(_kb(), "Hm. Well.")
        self.assertEqual(rec.verdicts, [])
        self.assertIsNone(rec.grounded_fraction)
        self.assertFalse(rec.is_grounded())
        self.assertFalse(rec.is_grounded(threshold=0.0))  # fails closed no matter how low the bar is

    # -- MXR-080-0258 (Critical): lexical overlap alone used to mark a direct contradiction as fully
    # SUPPORTED (score 1.0) -- the audit's own example: "the drug cures cancer" was corroborated by
    # "the drug cures no cancer and does not work" purely because almost every word overlaps. ----------
    def test_mxr_080_0258_drug_cancer_contradiction_is_not_marked_supported(self):
        rec = check_factuality(_drug_kb(), "The drug cures cancer.")
        verdict = rec.verdicts[0]
        self.assertFalse(verdict.supported)  # must NOT be certified as supported
        self.assertTrue(verdict.contradicted)  # the substrate actively disagrees, not merely silent
        self.assertTrue(verdict.contradictions)  # the contradicting item is attached as provenance
        self.assertEqual(rec.contradicted(), [verdict])  # surfaced at the receipt level too
        self.assertFalse(rec.is_grounded())
        self.assertEqual(rec.grounded_fraction, 0.0)

    def test_mxr_080_0258_default_corroborates_is_entailment_aware(self):  # white-box
        # The exact adversarial pair from the audit: heavy lexical overlap, opposite polarity.
        self.assertEqual(
            _default_corroborates("The drug cures no cancer and does not work.", "The drug cures cancer."),
            Corroboration.CONTRADICTED,
        )
        # Genuine positive support: overlap, no polarity disagreement -- must still work.
        self.assertEqual(
            _default_corroborates(
                "Refunds are processed within 30 days of a written request.",
                "Refunds are processed within 30 days.",
            ),
            Corroboration.SUPPORTED,
        )
        # Unrelated evidence: no overlap at all is UNVERIFIED -- neither a false SUPPORTED nor a false
        # CONTRADICTED, because it simply isn't evidence either way.
        self.assertEqual(
            _default_corroborates("The moon orbits the earth.", "The drug cures cancer."),
            Corroboration.UNVERIFIED,
        )
        # Symmetric negation: both texts negate the same content the same way -- agreement, not
        # contradiction, so a genuinely negative claim can still be genuinely supported.
        self.assertEqual(
            _default_corroborates(
                "Clinical trials show the drug does not cure cancer.",
                "The drug does not cure cancer.",
            ),
            Corroboration.SUPPORTED,
        )

    def test_legacy_bool_corroborator_is_still_accepted(self):
        # Pre-0258 callers passed `(evidence, claim) -> bool`. True -> supported, False -> unverified
        # (never silently promoted to "contradicted", which a bare bool cannot express).
        rec = check_factuality(_drug_kb(), "The drug cures cancer.", corroborates=lambda ev, cl: True)
        self.assertTrue(rec.verdicts[0].supported)
        self.assertFalse(rec.verdicts[0].contradicted)

        rec2 = check_factuality(_drug_kb(), "The drug cures cancer.", corroborates=lambda ev, cl: False)
        self.assertFalse(rec2.verdicts[0].supported)
        self.assertFalse(rec2.verdicts[0].contradicted)
        self.assertEqual(rec2.contradicted(), [])


if __name__ == "__main__":
    unittest.main()
