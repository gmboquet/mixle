"""Routing-ready distillation (mixle.task.distill): distill_for_routing / distill_records_for_routing.

One call should take a teacher + raw data straight to a decide()-able CalibratedTaskModel -- with a proper,
disjoint calibration split handled internally -- so it drops directly into Cascade/Router with no separate
calibration step for the caller to remember (or get wrong via calibrating on the training data itself).
"""

import unittest

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")

from mixle.task.calibrate import ESCALATE, CalibratedTaskModel  # noqa: E402
from mixle.task.cascade import Cascade  # noqa: E402
from mixle.task.distill import (  # noqa: E402
    distill_for_routing,
    distill_from_labels_for_routing,
    distill_records_for_routing,
)
from mixle.task.economics import CostModel  # noqa: E402


def _make_corpus(n_per_class=150, seed=0):
    rng = np.random.RandomState(seed)
    spam_words = ["free", "winner", "prize", "buy", "cheap", "offer", "click"]
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
    spam_words = {"free", "winner", "prize", "buy", "cheap", "offer", "click"}
    return ["spam" if any(w in t.split() for w in spam_words) else "ham" for t in texts]


def _make_records(n_per_class=80, seed=0):
    rng = np.random.RandomState(seed)
    records = []
    for base, label in ((10.0, "low"), (90.0, "high")):
        for _ in range(n_per_class):
            records.append((float(base + rng.normal(0, 3)), label))
    rng.shuffle(records)
    return [r[0] for r in records], [r[1] for r in records]


class DistillForRoutingTest(unittest.TestCase):
    def test_returns_a_calibrated_decideable_model(self):
        train = _make_corpus(seed=1)
        calibrated = distill_for_routing(
            _teacher, train, n=4, dim=512, hidden=[64], epochs=300, lr=1e-2, seed=0, calibration_frac=0.2
        )
        self.assertIsInstance(calibrated, CalibratedTaskModel)
        self.assertIsNotNone(calibrated.qhat)

        test = _make_corpus(seed=99)
        decisions = [calibrated.decide(t) for t in test]
        # every decision is either a real label or the escalate sentinel -- never a crash, never something else
        for d in decisions:
            self.assertTrue(d is ESCALATE or d in calibrated.labels)
        # a well-separated rule (spam keyword present) should mostly be confidently decided, not escalated
        rate = calibrated.escalation_rate(test)
        self.assertGreater(rate, 0.0)
        self.assertLess(rate, 0.6)

    def test_label_set_spans_both_split_sides_even_if_skewed(self):
        # a tiny, heavily class-imbalanced sample: the shared label_list must still cover both classes
        train = _make_corpus(n_per_class=10, seed=2)
        calibrated = distill_for_routing(_teacher, train, dim=128, epochs=40, seed=0, calibration_frac=0.3)
        self.assertEqual(sorted(calibrated.labels), ["ham", "spam"])

    def test_deterministic_given_seed(self):
        train = _make_corpus(seed=3)
        a = distill_for_routing(_teacher, train, dim=128, epochs=40, seed=7, calibration_frac=0.25)
        b = distill_for_routing(_teacher, train, dim=128, epochs=40, seed=7, calibration_frac=0.25)
        self.assertAlmostEqual(a.qhat, b.qhat, places=9)
        test = _make_corpus(seed=44)
        self.assertEqual([a.decide(t) for t in test], [b.decide(t) for t in test])

    def test_invalid_calibration_frac_raises(self):
        train = _make_corpus(n_per_class=5, seed=4)
        with self.assertRaises(ValueError):
            distill_for_routing(_teacher, train, calibration_frac=0.0)
        with self.assertRaises(ValueError):
            distill_for_routing(_teacher, train, calibration_frac=1.0)
        with self.assertRaises(ValueError):
            distill_from_labels_for_routing(train, _teacher(train), calibration_frac=1.5)

    def test_calibration_frac_too_small_a_sample_raises(self):
        # 3 examples, calibration_frac requesting effectively all of them for calibration -> no training data left
        tiny = ["free money now", "team meeting today", "click for your prize"]
        with self.assertRaises(ValueError):
            distill_from_labels_for_routing(tiny, _teacher(tiny), calibration_frac=0.99)

    def test_plugs_directly_into_cascade_with_no_manual_calibration_glue(self):
        train = _make_corpus(seed=5)
        calibrated = distill_for_routing(
            _teacher, train, n=4, dim=512, hidden=[64], epochs=300, lr=1e-2, seed=0, calibration_frac=0.2
        )
        test = _make_corpus(seed=77)
        cascade = Cascade(calibrated, _teacher, cost=CostModel(c_local=0.00001, c_frontier=0.01))
        served = cascade.serve(test)
        self.assertEqual(len(served), len(test))
        self.assertTrue(all(label in ("spam", "ham") for label in served))
        report = cascade.report()
        self.assertEqual(report["n_requests"], len(test))
        # the escalated fraction paid the teacher; the rest were free-ish -- realized cost must beat frontier-only
        self.assertLess(report["realized_cost"], report["frontier_only_cost"])
        # harvested examples are exactly the escalated ones, ready to feed back into another distill() round
        harvested_texts, harvested_labels = cascade.harvested()
        self.assertEqual(len(harvested_texts), report["n_escalated"])
        self.assertEqual(len(harvested_labels), report["n_escalated"])


class DistillRecordsForRoutingTest(unittest.TestCase):
    def test_returns_a_calibrated_decideable_record_model(self):
        records, labels = _make_records(seed=1)
        teacher = dict(zip(records, labels))  # exact rule: record value -> its assigned label

        def record_teacher(batch):
            return [teacher[r] for r in batch]

        calibrated = distill_records_for_routing(record_teacher, records, dim=64, hidden=[16], epochs=100, seed=0)
        self.assertIsInstance(calibrated, CalibratedTaskModel)
        self.assertIsNotNone(calibrated.qhat)
        decisions = [calibrated.decide(r) for r in records[:20]]
        for d in decisions:
            self.assertTrue(d is ESCALATE or d in calibrated.labels)


class DensityGateRoutingTest(unittest.TestCase):
    """CARD B1-a: density_gate=True on the *_for_routing family escalates inputs a softmax can't see are OOD."""

    def _ood_text(self, seed: int) -> str:
        # long random-unicode-ish text sharing no vocabulary with the spam/ham corpus
        rng = np.random.RandomState(seed)
        return " ".join("".join(chr(rng.randint(0x3B1, 0x3C9)) for _ in range(8)) for _ in range(12))

    def test_gate_on_escalates_ood_input(self):
        train = _make_corpus(seed=10)
        gated = distill_for_routing(
            _teacher,
            train,
            n=4,
            dim=512,
            hidden=[64],
            epochs=300,
            lr=1e-2,
            seed=0,
            calibration_frac=0.2,
            density_gate=True,
        )
        self.assertIsNotNone(gated.density_gate)
        ood = self._ood_text(0)
        self.assertIs(gated.decide(ood), ESCALATE)

    def test_escalation_rate_gate_on_gte_off(self):
        train = _make_corpus(seed=11)
        ungated = distill_for_routing(
            _teacher,
            train,
            n=4,
            dim=512,
            hidden=[64],
            epochs=300,
            lr=1e-2,
            seed=0,
            calibration_frac=0.2,
            density_gate=False,
        )
        gated = distill_for_routing(
            _teacher,
            train,
            n=4,
            dim=512,
            hidden=[64],
            epochs=300,
            lr=1e-2,
            seed=0,
            calibration_frac=0.2,
            density_gate=True,
        )
        mixed = _make_corpus(seed=88) + [self._ood_text(i) for i in range(20)]
        self.assertGreaterEqual(gated.escalation_rate(mixed), ungated.escalation_rate(mixed))

    def test_deterministic_given_seed(self):
        train = _make_corpus(seed=12)
        a = distill_for_routing(
            _teacher,
            train,
            n=4,
            dim=128,
            hidden=[32],
            epochs=60,
            seed=7,
            calibration_frac=0.25,
            density_gate=True,
        )
        b = distill_for_routing(
            _teacher,
            train,
            n=4,
            dim=128,
            hidden=[32],
            epochs=60,
            seed=7,
            calibration_frac=0.25,
            density_gate=True,
        )
        self.assertAlmostEqual(a.density_gate.log_threshold, b.density_gate.log_threshold, places=9)
        test = _make_corpus(seed=44) + [self._ood_text(1)]
        self.assertEqual([a.decide(t) for t in test], [b.decide(t) for t in test])


if __name__ == "__main__":
    unittest.main()
