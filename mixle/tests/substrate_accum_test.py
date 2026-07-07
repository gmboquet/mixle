"""The knowledge-accumulation flywheel (workstream ACCUM-a): assimilating calibrated knowledge should
raise solve-rate on a held-out set, WITH NO MODEL RETRAINING -- and the gain must be attributable to
the store growing, and immune to being inflated by low-credence assertions."""

import unittest

from mixle.substrate.accum import QAItem, measure_flywheel
from mixle.substrate.belief import MODEL_ASSERTION, Claim, assimilate
from mixle.substrate.core import Substrate

_QUESTIONS = [
    QAItem(question="capital of Fooland", answer="Foo City"),
    QAItem(question="capital of Barland", answer="Bar Town"),
]

_FACTS = {
    "capital of Fooland": ("Fooland's capital is Foo City", "Foo City"),
    "capital of Barland": ("Barland's capital is Bar Town", "Bar Town"),
}


def _answer_from_context(question: str, context: list[str]) -> str:
    """A fixed, never-retrained 'model': answers only what appears verbatim in retrieved context."""
    fact = _FACTS.get(question)
    if fact is None:
        return "unknown"
    claim_text, answer = fact
    return answer if claim_text in context else "unknown"


def _assimilate_strong_batch(sub: Substrate) -> list[str]:
    ids = []
    for claim_text, _answer in _FACTS.values():
        belief = assimilate(
            sub, Claim(text=claim_text), {"source_id": "gazetteer-2026", "tier": "held_out_truth", "weight": 1.0}
        )
        ids.append(belief.id)
    return ids


def _assimilate_weak_batch(sub: Substrate) -> list[str]:
    ids = []
    for claim_text, _answer in _FACTS.values():
        belief = assimilate(
            sub, Claim(text=claim_text), {"source_id": "self-assertion", "tier": MODEL_ASSERTION, "weight": 1.0}
        )
        ids.append(belief.id)
    return ids


class FlywheelTest(unittest.TestCase):
    def test_assimilating_calibrated_knowledge_raises_solve_rate_with_no_retraining(self):
        sub = Substrate()
        report = measure_flywheel(sub, _QUESTIONS, _answer_from_context, _assimilate_strong_batch, min_credence=0.6)

        self.assertEqual(report.before.solve_rate, 0.0)
        self.assertEqual(report.before.grounded_fraction, 0.0)
        self.assertEqual(report.after.solve_rate, 1.0)
        self.assertEqual(report.after.grounded_fraction, 1.0)

    def test_the_improvement_is_attributed_to_the_new_knowledge_not_something_else(self):
        sub = Substrate()
        report = measure_flywheel(sub, _QUESTIONS, _answer_from_context, _assimilate_strong_batch, min_credence=0.6)

        self.assertTrue(report.attribution_confirmed)
        # withholding exactly the newly-assimilated beliefs from retrieval erases the gain
        self.assertEqual(report.withheld.solve_rate, report.before.solve_rate)

    def test_low_credence_assertions_do_not_inflate_the_measured_improvement(self):
        sub = Substrate()
        report = measure_flywheel(sub, _QUESTIONS, _answer_from_context, _assimilate_weak_batch, min_credence=0.6)

        # MODEL_ASSERTION-only evidence is capped at 0.5 credence -- below the 0.6 retrieval threshold,
        # so these claims are never retrieved and the measured solve-rate does not move at all.
        self.assertEqual(report.before.solve_rate, report.after.solve_rate)
        self.assertEqual(report.after.solve_rate, 0.0)
        self.assertFalse(report.attribution_confirmed)  # no real improvement to attribute

    def test_a_lower_credence_threshold_lets_weak_assertions_through_but_still_capped(self):
        sub = Substrate()
        report = measure_flywheel(sub, _QUESTIONS, _answer_from_context, _assimilate_weak_batch, min_credence=0.3)

        # with the threshold below the model-assertion cap (0.5), the weak claims ARE retrievable
        self.assertEqual(report.after.solve_rate, 1.0)
        self.assertTrue(report.attribution_confirmed)


if __name__ == "__main__":
    unittest.main()
