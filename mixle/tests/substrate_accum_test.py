"""The knowledge-accumulation flywheel (workstream ACCUM-a): assimilating calibrated knowledge should
raise solve-rate on a held-out set, WITH NO MODEL RETRAINING -- and the gain must be attributable to
the store growing, and immune to being inflated by low-credence assertions."""

import unittest

from mixle.substrate.accum import QAItem, measure_flywheel
from mixle.substrate.belief import MODEL_ASSERTION, Claim, assimilate, retrieve_beliefs
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
    # MXR-080-0243: a held_out_truth claim only earns its strength once source_id resolves to a real
    # substrate item (a receipt) -- register the gazetteer once, before the first claim cites it.
    if sub.get("gazetteer-2026") is None:
        sub.add(kind="text", text="gazetteer-2026: authoritative capitals reference", id="gazetteer-2026")
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


def _assimilate_update_existing_and_create_new(sub: Substrate) -> list[str]:
    """MXR-080-0246 fixture: UPDATES a belief the test has already assimilated into ``sub`` before
    calling ``measure_flywheel`` (same claim text, so :func:`assimilate` finds it by key and appends
    more evidence to the SAME belief id), and separately creates one genuinely NEW belief. Exercises the
    exact shape the finding names: ``assimilate_batch`` returning an id that was already visible to the
    ``before`` pass, mixed in with an id that was not."""
    if sub.get("second-receipt") is None:
        sub.add(kind="text", text="second-receipt: another authoritative reference", id="second-receipt")
    fact0_claim, fact1_claim = (text for text, _answer in _FACTS.values())
    updated = assimilate(
        sub, Claim(text=fact0_claim), {"source_id": "second-receipt", "tier": "held_out_truth", "weight": 0.2}
    )
    created = assimilate(
        sub, Claim(text=fact1_claim), {"source_id": "second-receipt", "tier": "held_out_truth", "weight": 1.0}
    )
    return [updated.id, created.id]


class FlywheelWithholdingIsDeltaLevelTest(unittest.TestCase):
    """MXR-080-0246 (Critical): ``assimilate_batch`` may UPDATE a belief that already existed -- and was
    already visible to the ``before`` pass -- rather than create one from nothing. The withheld pass
    must remove only the batch's OWN contribution to that belief, not the belief outright, or it also
    destroys pre-existing evidence the batch never touched and understates the true baseline."""

    def test_withheld_preserves_a_pre_existing_beliefs_own_evidence_after_the_batch_updates_it(self):
        sub = Substrate()
        sub.add(kind="text", text="first-receipt: an authoritative reference", id="first-receipt")
        fact0_claim = _FACTS["capital of Fooland"][0]
        # A belief that already exists, with REAL, valid, pre-existing evidence -- before any batch
        # this test runs even starts.
        pre_existing = assimilate(
            sub, Claim(text=fact0_claim), {"source_id": "first-receipt", "tier": "held_out_truth", "weight": 1.0}
        )

        report = measure_flywheel(
            sub, _QUESTIONS, _answer_from_context, _assimilate_update_existing_and_create_new, min_credence=0.6
        )

        # before: the pre-existing belief already answers "capital of Fooland"; "capital of Barland"
        # has no MATCHING belief, but retrieve_beliefs has no relevance cutoff -- with only one belief
        # in the whole store, it is still the (irrelevant) top-1 result for either query, so both
        # questions count as "grounded" even though only one is actually answered correctly.
        self.assertEqual(report.before.solve_rate, 0.5)
        self.assertEqual(report.before.grounded_fraction, 1.0)
        # after: assimilate_batch touched both -- updated the first, created the second.
        self.assertEqual(report.after.solve_rate, 1.0)
        # THE regression (MXR-080-0246): withholding must reproduce the TRUE pre-batch baseline exactly
        # -- "capital of Fooland" stays answerable from `pre_existing`'s OWN evidence, which this batch
        # only added to, never removed. Before the fix this belief's id was fully excluded from
        # retrieval (it IS in assimilate_batch's returned ids), destroying that pre-existing evidence
        # too and driving withheld.solve_rate to 0.0 -- BELOW the true baseline of 0.5.
        self.assertEqual(report.withheld.solve_rate, report.before.solve_rate)
        self.assertEqual(report.withheld.solve_rate, 0.5)
        self.assertEqual(report.withheld.grounded_fraction, report.before.grounded_fraction)
        # correctly confirmed -- and now for the right reason: the withheld gap (after - withheld) is
        # 0.5, matching that only "capital of Barland" (the genuinely new belief) is really attributable
        # to the batch, not the full after-before gap of 1.0 a belief-level removal would have shown.
        self.assertTrue(report.attribution_confirmed)

        # sanity-check the fixture itself: assimilate_batch really did UPDATE `pre_existing` in place
        # (same id, evidence appended) rather than create a duplicate belief for the same claim text.
        [reloaded] = [b for b in retrieve_beliefs(sub, fact0_claim, k=10) if b.claim.text == fact0_claim]
        self.assertEqual(reloaded.id, pre_existing.id)
        self.assertEqual(len(reloaded.evidence_history), 2)

    def test_sub_reflects_the_true_post_assimilation_state_after_measuring(self):
        """The withheld pass's rollback of an updated belief is an internal device for taking ONE
        measurement -- it must never leak past measure_flywheel. Afterward `sub` must show the belief
        with ALL its evidence, the batch's included, not the rolled-back snapshot."""
        sub = Substrate()
        sub.add(kind="text", text="first-receipt: an authoritative reference", id="first-receipt")
        fact0_claim = _FACTS["capital of Fooland"][0]
        original = assimilate(
            sub, Claim(text=fact0_claim), {"source_id": "first-receipt", "tier": "held_out_truth", "weight": 1.0}
        )
        self.assertEqual(len(original.evidence_history), 1)

        measure_flywheel(
            sub, _QUESTIONS, _answer_from_context, _assimilate_update_existing_and_create_new, min_credence=0.6
        )

        [final] = [b for b in retrieve_beliefs(sub, fact0_claim, k=10) if b.id == original.id]
        self.assertEqual(len(final.evidence_history), 2)  # the original entry PLUS the batch's own


if __name__ == "__main__":
    unittest.main()
