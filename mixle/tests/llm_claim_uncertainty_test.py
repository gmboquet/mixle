"""Claim-level UQ — reliability of the information inside a response (mixle.reason.llm)."""

import unittest

import numpy as np

from mixle.reason import LLMUncertainty, content_overlap, sentence_claims


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


if __name__ == "__main__":
    unittest.main()
