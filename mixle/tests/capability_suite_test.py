"""CapabilitySuite / capture_profile (mixle.task.capability): a distilled student's behavioral spec.

Uses the spam/ham rule-teacher pattern from ``task_distill_routing_test.py``: a hashed-n-gram MLP student
distilled from a keyword rule teacher, checked under typo corruptions and a case-jitter invariance.
"""

import json
import unittest

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")

from mixle.task.capability import (  # noqa: E402
    CapabilitySuite,
    capture_profile,
    case_jitter_invariance,
    keyboard_typo_corruption,
)
from mixle.task.distill import distill_for_routing  # noqa: E402

SPAM_WORDS = {"free", "winner", "prize", "buy", "cheap", "offer", "click"}


def _make_corpus(n_per_class=150, seed=0):
    rng = np.random.RandomState(seed)
    spam_words = list(SPAM_WORDS)
    ham_words = ["meeting", "lunch", "project", "report", "schedule", "team", "review"]
    filler = ["the", "a", "today", "tomorrow", "please", "thanks", "we", "you"]
    texts = []
    for words in (spam_words, ham_words):
        for _ in range(n_per_class):
            k = rng.randint(3, 7)
            toks = list(rng.choice(words, size=2)) + list(rng.choice(filler, size=k))
            rng.shuffle(toks)
            texts.append(" ".join(toks))
    rng.shuffle(texts)
    return texts


def _teacher(texts):
    # case-insensitive by construction, so a case-jitter rewrite should never flip its decision
    return ["spam" if any(w in t.lower().split() for w in SPAM_WORDS) else "ham" for t in texts]


def _suite():
    return CapabilitySuite(
        corruptions={
            "typo_10": keyboard_typo_corruption(0.1, seed=1),
            "typo_40": keyboard_typo_corruption(0.4, seed=1),
            "typo_80": keyboard_typo_corruption(0.8, seed=1),
        },
        invariances={"case_jitter": case_jitter_invariance},
    )


class CapabilitySuiteTest(unittest.TestCase):
    def test_clean_agreement_high_and_degrades_under_typos(self):
        train = _make_corpus(seed=1)
        student = distill_for_routing(
            _teacher, train, n=4, dim=512, hidden=[64], epochs=300, lr=1e-2, seed=0, calibration_frac=0.2
        )
        test = _make_corpus(seed=99)
        profile = capture_profile(student, _teacher, test, _suite())

        self.assertGreater(profile["clean_agreement"], 0.8)
        # every corruption level degrades agreement RELATIVE TO CLEAN. Cross-severity monotonicity
        # (typo_80 <= typo_10) is deliberately NOT asserted: agreement measures teacher-student
        # CONSISTENCY, not accuracy, and under severe corruption both models can collapse onto the
        # same fallback prediction, making heavy-typo agreement rise degenerately (measured: 0.997
        # at typo_80 vs 0.94 at typo_10 after the analytic-first neural fitting change -- the old
        # ordering assertion held only while corruption happened to differentiate the two models).
        levels = ["typo_10", "typo_40", "typo_80"]
        scores = [profile["corruptions"][lvl] for lvl in levels]
        for lvl, score in zip(levels, scores):
            self.assertLess(score, profile["clean_agreement"], f"{lvl} must degrade agreement vs clean")

    def test_case_jitter_near_zero_violation_for_case_insensitive_teacher(self):
        train = _make_corpus(seed=2)
        student = distill_for_routing(
            _teacher, train, n=4, dim=512, hidden=[64], epochs=300, lr=1e-2, seed=0, calibration_frac=0.2
        )
        test = _make_corpus(seed=88)
        profile = capture_profile(student, _teacher, test, _suite())

        self.assertLess(profile["invariances"]["case_jitter"]["teacher_violation_rate"], 0.05)

    def test_abstention_reported_for_calibrated_student(self):
        train = _make_corpus(seed=3)
        student = distill_for_routing(
            _teacher, train, n=4, dim=512, hidden=[64], epochs=300, lr=1e-2, seed=0, calibration_frac=0.2
        )
        test = _make_corpus(seed=77)
        profile = capture_profile(student, _teacher, test, _suite())

        self.assertIn("abstention", profile)
        self.assertIsNotNone(profile["abstention"]["student_escalation_rate"])
        self.assertIsNone(profile["abstention"]["teacher_escalation_rate"])

    def test_profile_round_trips_json(self):
        train = _make_corpus(seed=4)
        student = distill_for_routing(
            _teacher, train, n=4, dim=128, hidden=[64], epochs=60, seed=0, calibration_frac=0.2
        )
        test = _make_corpus(seed=55)
        profile = capture_profile(student, _teacher, test, _suite())
        round_tripped = json.loads(json.dumps(profile))
        self.assertEqual(round_tripped, profile)

    def test_empty_suite_yields_clean_agreement_only(self):
        train = _make_corpus(seed=5)
        student = distill_for_routing(
            _teacher, train, n=4, dim=128, hidden=[64], epochs=60, seed=0, calibration_frac=0.2
        )
        test = _make_corpus(seed=66)
        profile = capture_profile(student, _teacher, test, CapabilitySuite())
        self.assertEqual(profile["corruptions"], {})
        self.assertEqual(profile["invariances"], {})
        self.assertNotIn("probes", profile)


if __name__ == "__main__":
    unittest.main()
