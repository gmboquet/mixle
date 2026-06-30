"""Density gate (mixle.task.density): a real p(x) flags out-of-distribution inputs the softmax can't see.

A diagonal-Gaussian mixture is fit over in-distribution features; clearly novel inputs should score below the
calibrated floor while in-distribution inputs mostly clear it, and the gate must round-trip through its spec.
"""

import unittest

import numpy as np
import pytest

pytest.importorskip("torch")  # the gate itself is torch-free, but it ships with the torch-task suite


def _id_corpus(n=300, seed=0):
    rng = np.random.RandomState(seed)
    words = ["meeting", "lunch", "project", "report", "schedule", "team", "review", "today", "please"]
    return [" ".join(rng.choice(words, size=rng.randint(4, 8))) for _ in range(n)]


def _ood_inputs():
    # nothing like the training vocabulary: digits, other alphabets, symbols
    return ["1234567890 0987 5555", "ΩΨΔ λβγ ξζ", "!!!??? @#$%^&*()", "zzzzz qqqqq xxxxx"]


class DensityGateTest(unittest.TestCase):
    def _gate(self, seed=0):
        from mixle.task.density import DensityGate
        from mixle.task.model import HashedNGram

        return DensityGate(HashedNGram(n=3, dim=48, seed=1)).fit(_id_corpus(seed=seed), n_components=3, seed=0)

    def test_ood_scores_below_in_distribution(self):
        gate = self._gate(seed=1)
        id_ld = gate.log_density(_id_corpus(seed=99))
        ood_ld = gate.log_density(_ood_inputs())
        self.assertLess(float(np.median(ood_ld)), float(np.median(id_ld)))

    def test_flags_ood_not_in_distribution(self):
        gate = self._gate(seed=2)
        # most clearly-novel inputs flag OOD
        self.assertGreaterEqual(np.mean([gate.is_ood(t) for t in _ood_inputs()]), 0.75)
        # in-distribution inputs mostly do not (the floor is the 2% quantile)
        self.assertLessEqual(float(np.mean(gate.ood_mask(_id_corpus(seed=7)))), 0.15)

    def test_spec_round_trip(self):
        from mixle.task.density import DensityGate

        gate = self._gate(seed=3)
        texts = _id_corpus(seed=5)[:10] + _ood_inputs()
        before = gate.log_density(texts)
        clone = DensityGate.from_spec(gate.to_spec())
        after = clone.log_density(texts)
        self.assertAlmostEqual(clone.log_threshold, gate.log_threshold, places=9)
        self.assertTrue(np.allclose(before, after, atol=1e-6))


class CalibratedWithDensityTest(unittest.TestCase):
    def test_density_gate_raises_escalation_on_ood(self):
        from mixle.task.calibrate import ESCALATE, CalibratedTaskModel
        from mixle.task.density import DensityGate
        from mixle.task.distill import distill
        from mixle.task.model import HashedNGram

        rng = np.random.RandomState(0)
        spam = ["free", "winner", "prize", "buy", "cheap", "offer", "click"]
        ham = ["meeting", "lunch", "project", "report", "schedule", "team", "review"]
        filler = ["the", "a", "today", "please", "thanks"]

        def corpus(seed):
            r = np.random.RandomState(seed)
            out = []
            for w in (spam, ham):
                for _ in range(120):
                    toks = list(r.choice(w, size=2)) + list(r.choice(filler, size=r.randint(3, 6)))
                    r.shuffle(toks)
                    out.append(" ".join(toks))
            r.shuffle(out)
            return out

        def teacher(texts):
            s = set(spam)
            return ["spam" if any(w in t.split() for w in s) else "ham" for t in texts]

        train, cal = corpus(1), corpus(2)
        student = distill(teacher, train, n=4, dim=512, hidden=[64], epochs=200, seed=0)
        gate = DensityGate(HashedNGram(n=3, dim=48, seed=1)).fit(train, n_components=3, seed=0)
        model = CalibratedTaskModel(student, alpha=0.1, density_gate=gate).calibrate(cal, teacher(cal))

        # a clearly out-of-distribution input escalates even if the classifier would have been "confident"
        self.assertIs(model.decide("ΩΨΔ 1234 !!!"), ESCALATE)
        # escalation rate with the gate is at least the conformal-only rate
        bare = CalibratedTaskModel(student, alpha=0.1, qhat=model.qhat)
        self.assertGreaterEqual(model.escalation_rate(corpus(3)), bare.escalation_rate(corpus(3)) - 1e-9)
        _ = rng  # silence unused


if __name__ == "__main__":
    unittest.main()
