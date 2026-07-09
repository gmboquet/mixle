"""Soft-label distillation (mixle.task.distill_soft): matching a frontier teacher's probability vector,
not just its argmax, transfers the teacher's dark knowledge -- the student's SOFT distribution ends up
closer to the teacher's than a hard-label student's does.
"""

import unittest

import numpy as np
import pytest

pytest.importorskip("torch")

from mixle.task.distill import distill_from_labels  # noqa: E402
from mixle.task.distill_soft import (  # noqa: E402
    distill_from_soft_labels,
    distill_soft,
    soft_agreement,
)

_LABELS = ["A", "B", "C"]
# A is distinct; B and C share most of their vocabulary, so a text is genuinely ambiguous between them
# -- the teacher's split probability over B vs C is the dark knowledge a hard label throws away.
_BANK = {
    "A": ["alpha", "apex", "atom", "arc", "axis"],
    "B": ["beta", "band", "blue", "bard"],
    "C": ["gamma", "core", "cave", "card"],
    "shared_BC": ["signal", "node", "cell", "wave", "field"],
}


def _oracle_probs(text: str) -> np.ndarray:
    """A fixed soft 'teacher': class scores from word-bank counts, softmax -> an informative distribution
    that splits mass between B and C on shared vocabulary."""
    words = text.split()
    score = np.zeros(3)
    for w in words:
        if w in _BANK["A"]:
            score[0] += 2.0
        if w in _BANK["B"]:
            score[1] += 1.6
        if w in _BANK["C"]:
            score[2] += 1.6
        if w in _BANK["shared_BC"]:  # shared vocab lifts BOTH B and C -> confusable
            score[1] += 0.7
            score[2] += 0.7
    e = np.exp(score - score.max())
    return e / e.sum()


def _corpus(n, seed):
    rng = np.random.RandomState(seed)
    texts = []
    for _ in range(n):
        cls = rng.choice(3)
        own = _BANK[_LABELS[cls]]
        pool = own + _BANK["shared_BC"]
        k = rng.randint(3, 7)
        texts.append(" ".join(rng.choice(pool, size=k)))
    probs = np.array([_oracle_probs(t) for t in texts])
    return texts, probs


class SoftDistillTest(unittest.TestCase):
    def setUp(self):
        self.train_texts, self.train_probs = _corpus(300, seed=0)
        self.test_texts, self.test_probs = _corpus(120, seed=99)

    def test_soft_student_matches_teacher_distribution_better_than_a_hard_student(self):
        soft = distill_from_soft_labels(
            self.train_texts, self.train_probs, labels=_LABELS, temperature=3.0, epochs=400, seed=0
        )
        hard_labels = [_LABELS[i] for i in np.argmax(self.train_probs, axis=1)]
        hard = distill_from_labels(self.train_texts, hard_labels, labels=_LABELS, epochs=400, seed=0)

        soft_kl = soft_agreement(soft, self.test_probs, self.test_texts)
        hard_kl = soft_agreement(hard, self.test_probs, self.test_texts)
        # the soft student's distribution is closer to the teacher's on held-out text
        self.assertLess(soft_kl, hard_kl)

    def test_student_probabilities_are_well_formed(self):
        soft = distill_from_soft_labels(self.train_texts, self.train_probs, labels=_LABELS, epochs=200, seed=0)
        p = np.asarray(soft.adapter.proba_batch(soft.model, self.test_texts), dtype=float)
        self.assertEqual(p.shape, (len(self.test_texts), 3))
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)

    def test_argmax_agreement_stays_high(self):
        soft = distill_from_soft_labels(self.train_texts, self.train_probs, labels=_LABELS, epochs=400, seed=0)
        pred = soft.batch(self.test_texts)
        teacher_hard = [_LABELS[i] for i in np.argmax(self.test_probs, axis=1)]
        acc = np.mean([p == t for p, t in zip(pred, teacher_hard)])
        self.assertGreater(acc, 0.75)

    def test_deterministic_given_seed(self):
        a = distill_from_soft_labels(self.train_texts, self.train_probs, labels=_LABELS, epochs=100, seed=7)
        b = distill_from_soft_labels(self.train_texts, self.train_probs, labels=_LABELS, epochs=100, seed=7)
        pa = a.adapter.proba_batch(a.model, self.test_texts)
        pb = b.adapter.proba_batch(b.model, self.test_texts)
        np.testing.assert_allclose(pa, pb, atol=1e-6)

    def test_distill_soft_calls_the_probability_teacher_once(self):
        calls = {"n": 0}

        def teacher_proba(texts):
            calls["n"] += 1
            return np.array([_oracle_probs(t) for t in texts])

        student = distill_soft(teacher_proba, self.train_texts, labels=_LABELS, epochs=100, seed=0)
        self.assertEqual(calls["n"], 1)  # one batched query, not one per example
        self.assertEqual(student.meta["soft"], True)

    def test_bad_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            distill_from_soft_labels(self.train_texts, self.train_probs, labels=_LABELS, temperature=0.0)
        with self.assertRaises(ValueError):
            distill_from_soft_labels(self.train_texts, self.train_probs, labels=_LABELS, hard_weight=1.5)
        with self.assertRaises(ValueError):
            distill_from_soft_labels(self.train_texts[:5], self.train_probs, labels=_LABELS)  # length mismatch


if __name__ == "__main__":
    unittest.main()
