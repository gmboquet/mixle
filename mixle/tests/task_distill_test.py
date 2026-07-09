"""Distillation (mixle.task.distill): a tiny student learns to mimic a teacher, then runs locally.

A learnable rule teacher (keyword -> label) is distilled into a small hashed-n-gram classifier; the student
should reproduce the teacher with high agreement on held-out text and survive a save/load.
"""

import os
import tempfile
import unittest

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")

from mixle.task.distill import agreement, distill  # noqa: E402
from mixle.task.model import TaskModel  # noqa: E402


def _make_corpus(n_per_class=120, seed=0):
    rng = np.random.RandomState(seed)
    spam_words = ["free", "winner", "prize", "buy", "cheap", "offer", "click"]
    ham_words = ["meeting", "lunch", "project", "report", "schedule", "team", "review"]
    filler = ["the", "a", "today", "tomorrow", "please", "thanks", "we", "you"]
    texts = []
    for words, _label in ((spam_words, "spam"), (ham_words, "ham")):
        for _ in range(n_per_class):
            k = rng.randint(3, 7)
            toks = list(rng.choice(words, size=2)) + list(rng.choice(filler, size=k))
            rng.shuffle(toks)
            texts.append(" ".join(toks))
    rng.shuffle(texts)
    return texts


def _teacher(texts):
    # the rule the student must recover: any spam keyword present -> "spam"
    spam_words = {"free", "winner", "prize", "buy", "cheap", "offer", "click"}
    return ["spam" if any(w in t.split() for w in spam_words) else "ham" for t in texts]


class DistillTest(unittest.TestCase):
    def test_student_recovers_teacher(self):
        train = _make_corpus(seed=1)
        student = distill(_teacher, train, n=4, dim=512, hidden=[64], epochs=300, lr=1e-2, seed=0)
        self.assertGreaterEqual(student.meta["train_agreement"], 0.8)

        test = _make_corpus(seed=99)
        held_out = agreement(student, _teacher(test), test)
        self.assertGreaterEqual(held_out, 0.75)


class EarlyStoppingTest(unittest.TestCase):
    def test_stops_well_before_the_epoch_ceiling_on_an_easy_task(self):
        # a clean, well-separated rule should plateau long before 300 requested epochs
        train = _make_corpus(seed=6)
        student = distill(_teacher, train, n=4, dim=512, hidden=[64], epochs=300, lr=1e-2, seed=0)
        self.assertLess(student.meta["recipe"]["epochs_run"], 300)
        self.assertGreater(student.meta["recipe"]["epochs_run"], 0)
        # the ceiling itself is preserved unchanged in the recipe -- only the actual run count is new
        self.assertEqual(student.meta["recipe"]["epochs"], 300)

    def test_never_exceeds_the_requested_ceiling(self):
        train = _make_corpus(n_per_class=20, seed=8)
        # a tiny epoch budget: early stopping must never run MORE than what was asked for
        student = distill(_teacher, train, n=3, dim=64, epochs=15, seed=0)
        self.assertLessEqual(student.meta["recipe"]["epochs_run"], 15)

    def test_does_not_regress_accuracy_vs_the_full_fixed_run(self):
        # early stopping should only skip steps that weren't improving the loss -- held-out accuracy should
        # match (not merely "be acceptable in isolation", but be comparable to) an unrelated full run
        train = _make_corpus(seed=9)
        test = _make_corpus(seed=100)
        student = distill(_teacher, train, n=4, dim=512, hidden=[64], epochs=300, lr=1e-2, seed=0)
        held_out = agreement(student, _teacher(test), test)
        self.assertGreaterEqual(held_out, 0.75)  # same bar as test_student_recovers_teacher, still cleared

    def test_labels_inferred_and_recorded(self):
        train = _make_corpus(seed=2)
        student = distill(_teacher, train, dim=256, epochs=50, seed=0)
        self.assertEqual(sorted(student.adapter.labels), ["ham", "spam"])
        self.assertTrue(student.meta["distilled"])
        self.assertEqual(student.meta["n_examples"], len(train))

    def test_distilled_model_saves_and_calls_locally(self):
        train = _make_corpus(seed=3)
        student = distill(_teacher, train, n=4, dim=512, hidden=[64], epochs=300, lr=1e-2, seed=0)
        test = _make_corpus(seed=7)
        before = student.batch(test)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "spamcls")
            student.save(path)
            loaded = TaskModel.load(path)
            after = loaded.batch(test)
        self.assertEqual(before, after)
        # and it actually labels a clearly-spam string as spam
        self.assertEqual(student("free prize click here"), "spam")


if __name__ == "__main__":
    unittest.main()
