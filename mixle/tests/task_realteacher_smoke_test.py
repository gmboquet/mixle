"""Real-teacher distillation smoke test: guard against a distiller that fails to learn.

Most task tests use synthetic RNG teachers whose rule is trivially recoverable, so a broken distiller that
just predicted the majority class could still pass. This test distills a teacher whose decision boundary a
no-op student cannot imitate, and asserts the student's held-out agreement clears a bar well above the
majority-class baseline -- so a distiller that failed to learn fails the test.

Primary path: a REAL sklearn ``LogisticRegression`` teacher over TF-IDF on the (offline-cached) 20-newsgroups
corpus. Fallback (if that corpus is not available offline): a deterministic non-trivial teacher whose label
is a keyword-count parity, which is not the majority class -- so the same "beats the baseline" assertion still
has teeth. Skips cleanly if neither path's dependencies are present.
"""

import unittest
from collections import Counter

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("sklearn")

from mixle.task.distill import agreement, distill  # noqa: E402


def _majority_fraction(labels) -> float:
    return max(Counter(labels).values()) / len(labels)


def _real_newsgroups_teacher():
    """Return ``(train_texts, test_texts, teacher, labels)`` for a real sklearn teacher, or ``None`` if offline.

    The teacher is a TF-IDF + LogisticRegression classifier trained on the real 20-newsgroups labels; distilling
    it forces the student to reconstruct a genuine learned decision boundary, not an RNG rule.
    """
    try:
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None

    cats = ["rec.sport.baseball", "sci.space"]
    strip = ("headers", "footers", "quotes")
    try:
        # download_if_missing=False => raise (not hang on the network) when the corpus is not cached.
        tr = fetch_20newsgroups(subset="train", categories=cats, remove=strip, download_if_missing=False)
        te = fetch_20newsgroups(subset="test", categories=cats, remove=strip, download_if_missing=False)
    except Exception:
        return None

    vec = TfidfVectorizer(max_features=5000, stop_words="english")
    clf = LogisticRegression(max_iter=1000).fit(vec.fit_transform(tr.data), tr.target)

    def teacher(texts):
        return [cats[i] for i in clf.predict(vec.transform(texts))]

    rng = np.random.RandomState(0)
    idx = rng.permutation(len(tr.data))[:600]
    train_texts = [tr.data[i] for i in idx]
    return train_texts, list(te.data), teacher, cats


def _synthetic_nontrivial_teacher():
    """Fallback: a deterministic teacher whose rule (keyword-count parity) a majority-class student would fail.

    The corpus is built so ``"odd"`` is the minority class (~1/3): a distiller that collapsed to the majority
    label would score only that minority fraction, well under the assertion bar, so a broken distiller fails.
    """
    kw = ["alpha", "beta", "gamma", "delta"]
    filler = ["the", "a", "and", "of", "to", "in", "it", "on"]
    rng = np.random.RandomState(0)

    def make(n, seed):
        r = np.random.RandomState(seed)
        texts = []
        for _ in range(n):
            m = int(r.randint(0, 4))  # number of DISTINCT keywords: 0..3
            toks = list(r.choice(kw, size=m, replace=False)) if m else []
            toks += list(r.choice(filler, size=int(r.randint(4, 9))))
            r.shuffle(toks)
            texts.append(" ".join(toks))
        return texts

    def teacher(texts):
        out = []
        for t in texts:
            present = sum(1 for w in kw if w in t.split())
            out.append("odd" if present % 2 == 1 else "even")  # parity of distinct-keyword count
        return out

    train_texts = make(700, seed=1)
    test_texts = make(300, seed=99)
    return train_texts, test_texts, teacher, ["even", "odd"]


class RealTeacherSmokeTest(unittest.TestCase):
    def _run(self, train_texts, test_texts, teacher, labels):
        student = distill(
            teacher, train_texts, labels=list(labels), n=2, dim=2048, hidden=[128], epochs=250, lr=1e-2, seed=0
        )
        # the student must actually fit the teacher on the training set (rules out a no-op student)
        self.assertGreaterEqual(student.meta["train_agreement"], 0.85)

        teacher_test = teacher(test_texts)
        held = agreement(student, teacher_test, test_texts)
        baseline = _majority_fraction(teacher_test)
        # a meaningful bar: clear the majority-class baseline by a wide margin AND be high in absolute terms.
        self.assertGreater(
            held,
            baseline + 0.15,
            msg=f"held-out agreement {held:.3f} did not beat majority baseline {baseline:.3f} by 0.15",
        )
        self.assertGreaterEqual(held, 0.75, msg=f"held-out agreement {held:.3f} below absolute bar 0.75")

    def test_real_sklearn_teacher_is_distilled_above_baseline(self):
        real = _real_newsgroups_teacher()
        if real is None:
            self.skipTest("20-newsgroups not available offline and sklearn text stack missing")
        self._run(*real)

    def test_nontrivial_teacher_is_distilled_above_baseline(self):
        # Always runs (no dataset dependency): a deterministic non-majority rule a broken distiller cannot fake.
        self._run(*_synthetic_nontrivial_teacher())


if __name__ == "__main__":
    unittest.main()
