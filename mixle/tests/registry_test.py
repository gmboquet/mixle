"""Registry (mixle.registry): a dir-backed catalog of registered task models, queried by capability/fingerprint.

Card REG-a (workstream J2): register writes a real task-artifact directory + a JSON index entry; find_for and
tier_stack read the index back, including in a fresh Registry instance pointed at the same dir.
"""

import tempfile
import unittest

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")

from mixle.registry import Registry  # noqa: E402
from mixle.task.calibrate import ESCALATE  # noqa: E402
from mixle.task.distill import distill_for_routing  # noqa: E402
from mixle.task.router import Router  # noqa: E402


def _spam_teacher(texts):
    words = {"free", "winner", "prize"}
    return ["spam" if any(w in t.split() for w in words) else "ham" for t in texts]


def _billing_teacher(texts):
    words = {"invoice", "overdue", "payment"}
    return ["billing" if any(w in t.split() for w in words) else "other" for t in texts]


def _toy_model(teacher, vocab, seed):
    texts = [f"{a} {b} filler {c}" for a in vocab for b in vocab for c in ["x", "y", "z"]]
    return distill_for_routing(teacher, texts, dim=64, hidden=[16], epochs=40, seed=seed, calibration_frac=0.3)


class RegistryTest(unittest.TestCase):
    def test_find_for_matches_capability_not_the_other(self):
        with tempfile.TemporaryDirectory() as d:
            reg = Registry(d)
            spam_model = _toy_model(_spam_teacher, ["free", "winner", "prize", "meeting", "lunch"], seed=0)
            billing_model = _toy_model(_billing_teacher, ["invoice", "overdue", "payment", "meeting", "lunch"], seed=1)
            reg.register(spam_model, capabilities=["spam_filter"], fingerprint=[0.0, 0.0, 0.0, 0.0, 0.0], cost=0.01)
            reg.register(
                billing_model, capabilities=["billing_router"], fingerprint=[9.0, 9.0, 9.0, 9.0, 9.0], cost=0.02
            )

            spam_matches = reg.find_for("spam_filter")
            self.assertEqual(len(spam_matches), 1)
            self.assertEqual(spam_matches[0].capabilities, ["spam_filter"])

            billing_matches = reg.find_for("billing_router")
            self.assertEqual(len(billing_matches), 1)
            self.assertEqual(billing_matches[0].capabilities, ["billing_router"])

            near_spam = reg.find_for([0.1, 0.0, 0.0, 0.0, 0.0])
            self.assertEqual(len(near_spam), 1)
            self.assertEqual(near_spam[0].capabilities, ["spam_filter"])

    def test_tier_stack_cheapest_first_frontier_last(self):
        with tempfile.TemporaryDirectory() as d:
            reg = Registry(d)
            cheap = _toy_model(_spam_teacher, ["free", "winner", "prize", "meeting", "lunch"], seed=2)
            pricier = _toy_model(_spam_teacher, ["free", "winner", "prize", "meeting", "lunch"], seed=3)
            reg.register(pricier, capabilities=["spam_filter"], cost=0.05)
            reg.register(cheap, capabilities=["spam_filter"], cost=0.01)

            def frontier(texts):
                # Router calls the frontier as a BATCHED callable (texts -> [label]) and does its own
                # single-item wrap/unwrap (see Router.__call__) -- matching Cascade._teacher_label's
                # convention of "teacher is the raw batched function". A frontier that already unwraps
                # to one text in/out double-wraps and breaks (a list-of-one-string gets .split()'d).
                return _spam_teacher(texts)

            stack = reg.tier_stack("spam_filter", frontier=frontier, costs=[0.01, 0.05, 1.0])

            self.assertEqual(len(stack), 3)
            self.assertEqual([c for _, _, c in stack], [0.01, 0.05, 1.0])
            self.assertEqual(stack[-1][0], "frontier")
            self.assertIs(stack[-1][1], frontier)
            for name, model, _cost in stack[:-1]:
                self.assertTrue(hasattr(model, "decide"))

            # the exact shape Router's constructor wants: cheapest calibrated tiers, frontier fallback last
            router = Router(tiers=stack)
            decisions = [router(t) for t in ["free prize now", "team meeting today"]]
            for d_ in decisions:
                self.assertTrue(d_ is ESCALATE or isinstance(d_, str))

    def test_round_trips_through_a_fresh_registry_instance(self):
        with tempfile.TemporaryDirectory() as d:
            first = Registry(d)
            model = _toy_model(_spam_teacher, ["free", "winner", "prize", "meeting", "lunch"], seed=4)
            entry = first.register(model, capabilities=["spam_filter"], fingerprint=[1.0, 2.0, 3.0], cost=0.02)

            reopened = Registry(d)
            self.assertEqual(len(reopened.find_for("spam_filter")), 1)
            reloaded = reopened.load(entry.entry_id)
            self.assertEqual(reloaded.decide("free prize now"), model.decide("free prize now"))


if __name__ == "__main__":
    unittest.main()
