"""Calibrated task model (mixle.task.calibrate): conformal sets give an honest answer-vs-escalate decision.

Coverage of the conformal set must hold on held-out data; the escalate decision and the threshold must survive
a save/load so a loaded model decides identically.
"""

import os
import tempfile
import unittest

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")

from mixle.task.calibrate import ESCALATE, CalibratedTaskModel  # noqa: E402
from mixle.task.distill import distill  # noqa: E402


def _make_corpus(n_per_class=120, seed=0):
    rng = np.random.RandomState(seed)
    spam = ["free", "winner", "prize", "buy", "cheap", "offer", "click"]
    ham = ["meeting", "lunch", "project", "report", "schedule", "team", "review"]
    filler = ["the", "a", "today", "please", "thanks", "we", "you"]
    texts = []
    for words in (spam, ham):
        for _ in range(n_per_class):
            toks = list(rng.choice(words, size=2)) + list(rng.choice(filler, size=rng.randint(3, 7)))
            rng.shuffle(toks)
            texts.append(" ".join(toks))
    rng.shuffle(texts)
    return texts


def _teacher(texts):
    spam = {"free", "winner", "prize", "buy", "cheap", "offer", "click"}
    return ["spam" if any(w in t.split() for w in spam) else "ham" for t in texts]


def _calibrated(alpha=0.1, seed=0):
    train, cal = _make_corpus(seed=seed), _make_corpus(seed=seed + 50)
    student = distill(_teacher, train, n=4, dim=512, hidden=[64], epochs=250, lr=1e-2, seed=0)
    model = CalibratedTaskModel(student, alpha=alpha).calibrate(cal, _teacher(cal))
    return model


class CoverageTest(unittest.TestCase):
    def test_held_out_set_coverage(self):
        alpha = 0.1
        model = _calibrated(alpha=alpha, seed=1)
        test = _make_corpus(seed=777)
        truth = _teacher(test)
        sets = model.predict_sets(test)
        covered = np.mean([t in s for s, t in zip(sets, truth)])
        self.assertGreaterEqual(covered, 1.0 - alpha - 0.05)  # finite-sample slack

    def test_escalation_rate_in_unit_interval(self):
        model = _calibrated(seed=2)
        r = model.escalation_rate(_make_corpus(seed=321))
        self.assertTrue(0.0 <= r <= 1.0)


class DecisionTest(unittest.TestCase):
    def test_decide_singleton_or_escalate(self):
        model = _calibrated(seed=3)
        for text in _make_corpus(seed=9)[:20]:
            d = model.decide(text)
            s = model.predict_set(text)
            if len(s) == 1:
                self.assertEqual(d, s[0])
            else:
                self.assertIs(d, ESCALATE)


class PersistenceTest(unittest.TestCase):
    def test_save_load_preserves_decisions(self):
        model = _calibrated(seed=4)
        test = _make_corpus(seed=11)
        before = model.batch_decide(test)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cal")
            model.save(path)
            loaded = CalibratedTaskModel.load(path)
            self.assertEqual(loaded.alpha, model.alpha)
            self.assertAlmostEqual(loaded.qhat, model.qhat, places=9)
            after = loaded.batch_decide(test)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
