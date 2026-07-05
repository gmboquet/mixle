"""check_factuality (B3): ground an answer's claims against the substrate, per-claim, with citations."""

import unittest

from mixle.substrate import Substrate, check_factuality
from mixle.substrate.factuality import FactualityReceipt


def _kb():
    s = Substrate()
    s.add(kind="text", text="Refunds are processed within 30 days of a written request.")
    s.add(kind="text", text="Enterprise support is staffed 24 hours a day, 7 days a week.")
    return s


class FactualityTest(unittest.TestCase):
    def test_supported_claim_is_cited(self):
        rec = check_factuality(_kb(), "Refunds are processed within 30 days.")
        self.assertIsInstance(rec, FactualityReceipt)
        self.assertTrue(rec.verdicts[0].supported)
        self.assertTrue(rec.verdicts[0].citations)  # carries the citing item
        self.assertTrue(rec.is_grounded())

    def test_fabricated_claim_is_flagged_unsupported(self):
        rec = check_factuality(_kb(), "Free accounts include a dedicated account manager.")
        self.assertFalse(rec.verdicts[0].supported)
        self.assertEqual(rec.verdicts[0].citations, [])
        self.assertFalse(rec.is_grounded())

    def test_mixed_answer_grounded_fraction(self):
        ans = "Refunds are processed within 30 days. Free accounts include a dedicated account manager."
        rec = check_factuality(_kb(), ans)
        self.assertEqual(len(rec.verdicts), 2)
        self.assertEqual(rec.grounded_fraction, 0.5)
        self.assertEqual(len(rec.unsupported()), 1)

    def test_empty_answer_is_vacuously_grounded(self):
        rec = check_factuality(_kb(), "")
        self.assertEqual(rec.grounded_fraction, 1.0)
        self.assertEqual(rec.verdicts, [])

    def test_min_score_guards_against_noise(self):
        # a high floor rejects weak matches, so a loosely-related claim goes unsupported
        rec = check_factuality(_kb(), "Support exists.", min_score=0.9)
        self.assertFalse(rec.verdicts[0].supported)

    def test_as_dict_is_serializable(self):
        rec = check_factuality(_kb(), "Refunds are processed within 30 days.")
        d = rec.as_dict()
        self.assertIn("grounded_fraction", d)
        self.assertEqual(d["n_claims"], 1)


if __name__ == "__main__":
    unittest.main()
