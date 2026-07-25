"""Claim-level UQ — reliability of the information inside a response (mixle.reason.llm)."""

import unittest

import numpy as np

from mixle.reason import LLMUncertainty, content_overlap, sentence_claims
from mixle.reason.llm import (  # white-box: MXR-080-0295
    Corroboration,
    _default_claim_corroborator,
    _negated_content_words,
    information_corroborator,
)


class ExtractorTest(unittest.TestCase):
    def test_sentence_claims_splits_atomic_units(self):
        text = "The tower is 300 meters tall. It was built in 1889! Where is it?"
        claims = sentence_claims(text)
        self.assertEqual(len(claims), 3)
        self.assertIn("The tower is 300 meters tall.", claims)

    def test_content_overlap_corroboration(self):
        self.assertTrue(content_overlap("the eiffel tower is 300 meters tall", "tower is 300 meters tall"))
        self.assertFalse(content_overlap("the tower is in paris", "the tower is 300 meters tall"))


class MockClaimLLM:
    """Responses always contain the same TRUE claims, plus one FABRICATED claim that differs every
    call (a hallucination that won't corroborate across samples)."""

    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)
        self.true = ["The tower is 300 meters tall.", "The tower was built in 1889."]
        self.fake_cities = ["lyon", "berlin", "cairo", "oslo", "lima", "tokyo", "madrid", "rome"]

    def __call__(self, prompt):
        city = self.fake_cities[self.rng.randint(len(self.fake_cities))]
        fabricated = f"The tower is located in {city}."
        # order stable; the fabricated claim varies each call
        return " ".join([*self.true, fabricated])


class ClaimUQTest(unittest.TestCase):
    def test_flags_the_fabricated_claim(self):
        uq = LLMUncertainty(MockClaimLLM(seed=1), n=12)
        info = uq.assess_claims("Tell me about the tower.", threshold=0.5)
        by_claim = {c.claim: c for c in info.claims}
        # the two true claims recur across every resample -> high support, reliable
        self.assertGreater(by_claim["The tower is 300 meters tall."].support, 0.9)
        self.assertGreater(by_claim["The tower was built in 1889."].support, 0.9)
        # the fabricated 'located in <city>' claim differs every time -> low support, flagged
        fab = [c for c in info.claims if "located in" in c.claim][0]
        self.assertLess(fab.support, 0.3)
        self.assertFalse(fab.reliable)
        self.assertIn(fab, info.fabricated)

    def test_overall_reliability_between_zero_and_one(self):
        uq = LLMUncertainty(MockClaimLLM(seed=2), n=10)
        info = uq.assess_claims("Tell me about the tower.")
        self.assertTrue(0.0 <= info.reliability <= 1.0)
        # two of three claims are solid, one fabricated -> reliability in a sensible mid-high band
        self.assertGreater(info.reliability, 0.55)
        self.assertLess(info.reliability, 0.8)

    def test_all_reliable_when_response_is_consistent(self):
        # a model that always says the exact same thing -> every claim fully corroborated
        uq = LLMUncertainty(lambda p: "Water boils at 100 C. Ice melts at 0 C.", n=6)
        info = uq.assess_claims("physics facts")
        self.assertTrue(all(c.reliable for c in info.claims))
        self.assertAlmostEqual(info.reliability, 1.0, places=6)
        self.assertEqual(info.fabricated, [])

    def test_custom_extractor_and_corroborator(self):
        # plug in domain-specific claim extraction + entailment
        uq = LLMUncertainty(lambda p: "X", n=4)
        info = uq.assess_claims(
            "q",
            extract=lambda text: ["claim-1", "claim-2"],
            corroborates=lambda sample, claim: claim == "claim-1",
            threshold=0.5,
        )
        d = {c.claim: c for c in info.claims}
        self.assertTrue(d["claim-1"].reliable)
        self.assertFalse(d["claim-2"].reliable)


class DefaultClaimCorroboratorTest(unittest.TestCase):  # white-box: MXR-080-0295
    """Coverage for the negation-aware default corroborator, mirroring
    mixle.substrate.factuality's MXR-080-0258 test pattern (same adversarial pair)."""

    def test_negated_resample_sharing_claim_vocabulary_is_contradicted_not_supported(self):
        # The audit's exact adversarial pattern (mirrored from the substrate/factuality.py precedent):
        # a resample that shares almost every content word with a claim but negates it. Pre-fix, both
        # content_overlap and information_corroborator -- lexical-overlap-only -- treat this as full
        # corroboration; the negation-aware default must not.
        claim = "The drug cures cancer."
        negated = "The drug cures no cancer and does not work."
        self.assertTrue(content_overlap(negated, claim, threshold=0.5))  # candidacy still clears
        self.assertTrue(information_corroborator([claim, negated])(negated, claim))  # old default: True
        verdict = _default_claim_corroborator([claim, negated])(negated, claim)
        self.assertEqual(verdict, Corroboration.CONTRADICTED)

    def test_genuine_support_with_no_negation_still_supported(self):
        # Guard against the fix over-correcting to "everything is contradicted".
        samples = ["The tower is 300 meters tall.", "The tower is 300 meters tall, built in 1889."]
        verdict = _default_claim_corroborator(samples)(samples[1], samples[0])
        self.assertEqual(verdict, Corroboration.SUPPORTED)

    def test_unrelated_sample_is_unverified_not_contradicted(self):
        samples = ["The tower is 300 meters tall.", "The moon orbits the earth."]
        verdict = _default_claim_corroborator(samples)(samples[1], samples[0])
        self.assertEqual(verdict, Corroboration.UNVERIFIED)

    def test_symmetric_negation_is_still_supported(self):
        # Both texts negate the same shared content the same way -- agreement, not disagreement, so a
        # genuinely negative claim can still be genuinely corroborated (not "any negation anywhere
        # fails the claim").
        claim = "The drug does not cure cancer."
        sample = "Clinical trials show the drug does not cure cancer."
        verdict = _default_claim_corroborator([claim, sample])(sample, claim)
        self.assertEqual(verdict, Corroboration.SUPPORTED)

    def test_legacy_bool_corroborator_still_accepted_by_assess_claims(self):
        # A pre-0295 caller-supplied `corroborates=(sample, claim) -> bool` must keep working.
        uq = LLMUncertainty(lambda p: "X", n=4)
        info = uq.assess_claims(
            "q",
            extract=lambda text: ["claim-1"],
            corroborates=lambda sample, claim: True,
            threshold=0.5,
        )
        self.assertTrue(info.claims[0].reliable)
        self.assertEqual(info.claims[0].contradicted, 0.0)

    def test_contraction_negation_is_detected(self):
        # n't-contractions must survive tokenization ("doesn't" -> "does not") for the negation check
        # to see them at all.
        self.assertIn("work", _negated_content_words("The drug doesn't work."))


class ContradictedClaimEndToEndTest(unittest.TestCase):  # MXR-080-0295
    def test_resample_that_contradicts_the_primary_answer_is_not_reliable(self):
        responses = iter(
            [
                "The drug cures cancer.",
                "The drug cures no cancer and does not work.",
                "The drug cures no cancer and does not work.",
                "The drug cures no cancer and does not work.",
            ]
        )
        uq = LLMUncertainty(lambda p: next(responses), n=4)
        info = uq.assess_claims("Does the drug cure cancer?", threshold=0.5)
        claim = info.claims[0]
        self.assertEqual(claim.support, 0.0)
        self.assertEqual(claim.contradicted, 1.0)
        self.assertFalse(claim.reliable)
        self.assertIn(claim, info.fabricated)

    def test_zero_extractable_claims_reports_unassessed_not_perfect_reliability(self):
        # MXR-080-0295 (Critical): an empty/evasive primary response used to report reliability == 1.0
        # (vacuously "fully reliable"). No fragment in "Hm. Well." has >= 2 words, so sentence_claims
        # extracts nothing -- this must be UNASSESSED (None), not a perfect score, and is_reliable()
        # must fail closed regardless of threshold.
        uq = LLMUncertainty(lambda p: "Hm. Well.", n=4)
        info = uq.assess_claims("evasive question")
        self.assertEqual(info.claims, [])
        self.assertIsNone(info.reliability)
        self.assertFalse(info.is_reliable())
        self.assertFalse(info.is_reliable(threshold=0.0))  # fails closed no matter how low the bar is

    def test_literally_empty_response_is_also_unassessed(self):
        uq = LLMUncertainty(lambda p: "", n=4)
        info = uq.assess_claims("q")
        self.assertEqual(info.claims, [])
        self.assertIsNone(info.reliability)
        self.assertFalse(info.is_reliable())


class AssessClaimsTupleGenerationTest(unittest.TestCase):  # MXR-080-0296
    def test_tuple_generations_are_unpacked_before_claim_extraction(self):
        # A (text, logprob) generator used to hand the raw tuple straight to extract()/corroborates(),
        # producing garbage claims from the tuple's str() repr instead of the actual sentences.
        text = "The tower is 300 meters tall. It was built in 1889."
        uq = LLMUncertainty(lambda p: (text, -0.25), n=3)
        info = uq.assess_claims("about the tower")
        claims = [c.claim for c in info.claims]
        self.assertEqual(claims, ["The tower is 300 meters tall.", "It was built in 1889."])
        self.assertTrue(all("(" not in c and "-0.25" not in c for c in claims))


if __name__ == "__main__":
    unittest.main()
