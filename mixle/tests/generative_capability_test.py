"""Extraction-student capture profiles (workstream D6): F1-against-gold, not exact-match agreement.

The load-bearing claim: exact-match agreement (the base capture_profile's notion) understates a
partially-correct extractor's real capability -- a student that gets most fields right under a
corruption looks the same as one that gets none right, once you demand an exact dict match. Field-level
F1 against a fixed gold reference does not have that blind spot. Built with a real distilled
ExtractionIO student, not a hand-rolled stand-in.
"""

import unittest

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")

from mixle.task.capability import (  # noqa: E402
    CapabilitySuite,
    capture_profile,
    keyboard_typo_corruption,
)
from mixle.task.extract import distill_extractor  # noqa: E402
from mixle.task.generative_capability import (  # noqa: E402
    extractive_capture_profile,
    validate_extraction_schema,
)

NAMES = ["alice", "bob", "carol", "david", "erin", "frank", "grace", "henry"]
TEMPLATES = [
    "hi my name is {n} nice to meet you",
    "hello, i am {n} and i like tea",
    "{n} said hello to the team today",
    "please welcome {n} to the call",
]


def _teacher(texts):
    out = []
    for t in texts:
        low = t.lower()
        found = next((n for n in NAMES if n in low), None)
        out.append({"name": found} if found else {})
    return out


def _corpus(n_per=18, seed=0):
    rng = np.random.RandomState(seed)
    texts = []
    for name in NAMES:
        for _ in range(n_per):
            texts.append(TEMPLATES[rng.randint(len(TEMPLATES))].format(n=name))
    rng.shuffle(texts)
    return texts


class ValidateExtractionSchemaTest(unittest.TestCase):
    def test_complete_and_grounded_record_passes(self):
        check = validate_extraction_schema({"name": "alice"}, "hi my name is alice", ["name"])
        self.assertTrue(check["complete"])
        self.assertTrue(check["grounded"])
        self.assertEqual(check["missing"], [])
        self.assertEqual(check["ungrounded"], [])

    def test_missing_field_fails_completeness(self):
        check = validate_extraction_schema({}, "hi there", ["name"])
        self.assertFalse(check["complete"])
        self.assertIn("name", check["missing"])

    def test_hallucinated_value_fails_groundedness(self):
        # "zach" is not a substring of the source text -- a hallucinated (or corrupted-span) value.
        check = validate_extraction_schema({"name": "zach"}, "hi my name is alice", ["name"])
        self.assertFalse(check["grounded"])
        self.assertIn("name", check["ungrounded"])


class ExtractiveCaptureProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.texts = _corpus()
        cls.student = distill_extractor(_teacher, cls.texts, ["name"], epochs=80, seed=0)

    def test_clean_f1_is_high_and_schema_valid(self):
        suite = CapabilitySuite()
        profile = extractive_capture_profile(self.student, _teacher, self.texts, suite, fields=["name"])
        self.assertGreater(profile["clean_f1"], 0.85)
        # ExtractionIO always decodes a span of the text it was called on -- grounded by construction.
        self.assertEqual(profile["schema_validity"]["student"], 1.0)
        self.assertEqual(profile["schema_validity"]["teacher"], 1.0)

    def test_profile_shows_measurable_degradation_under_corruption(self):
        suite = CapabilitySuite(corruptions={"typo_40": keyboard_typo_corruption(0.4, seed=0)})
        profile = extractive_capture_profile(self.student, _teacher, self.texts, suite, fields=["name"])
        self.assertLess(profile["corruptions"]["typo_40"]["student_f1"], profile["clean_f1"])

    def test_agreement_with_teacher_can_mask_true_performance_f1_against_gold_does_not(self):
        # Same fixture, scored two ways under a heavy corruption. capture_profile's base metric is
        # student-vs-TEACHER agreement on the corrupted input: if a corruption is severe enough that
        # BOTH sides degrade to the same (wrong) extraction -- e.g. both fail to find the name and
        # return {} -- the two "agree" perfectly even though neither is actually right. F1-against-a-
        # fixed-gold does not have that blind spot: it is checked against the true answer, not against
        # whatever the teacher also happens to say under the same corruption.
        suite = CapabilitySuite(corruptions={"typo_70": keyboard_typo_corruption(0.7, seed=1)})
        extractive = extractive_capture_profile(self.student, _teacher, self.texts, suite, fields=["name"])
        base = capture_profile(self.student, _teacher, self.texts, suite)

        f1_score = extractive["corruptions"]["typo_70"]["student_f1"]
        exact_match_score = base["corruptions"]["typo_70"]
        # heavy corruption: student and teacher both largely fail the same way (agreement looks fine)
        self.assertGreater(exact_match_score, 0.9)
        # ... but true field-level performance against the fixed gold is genuinely poor -- the gap
        # base capture_profile's agreement number cannot see.
        self.assertLess(f1_score, exact_match_score - 0.3)

    def test_invariance_uses_f1_not_binary_violation(self):
        suite = CapabilitySuite(invariances={"case_jitter": lambda t: t.swapcase()})
        profile = extractive_capture_profile(self.student, _teacher, self.texts, suite, fields=["name"])
        self.assertIn("case_jitter", profile["invariances"])
        self.assertIn("student_f1", profile["invariances"]["case_jitter"])
        self.assertIn("teacher_f1", profile["invariances"]["case_jitter"])


if __name__ == "__main__":
    unittest.main()
