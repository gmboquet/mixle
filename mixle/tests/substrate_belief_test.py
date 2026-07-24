"""KNOW-a: harvest -> assimilate calibrated belief, anti-laundering, revision/retract, replay."""

import math
import random
import unittest

from mixle.substrate import Substrate
from mixle.substrate.belief import (
    MODEL_ASSERTION_CAP,
    Claim,
    EvidenceEntry,
    assimilate,
    credence_from_history,
    harvest_knowledge,
    retract,
)


class HarvestKnowledgeTest(unittest.TestCase):
    def test_splits_atomic_claims_with_provenance(self):
        text = "The rate is 5%. It rose from 3% last year."
        claims = harvest_knowledge(text, source={"model": "teacher-v1", "confidence": 0.9})
        self.assertEqual(len(claims), 2)
        for c in claims:
            self.assertIsInstance(c, Claim)
            self.assertEqual(c.produced_by, {"model": "teacher-v1", "confidence": 0.9})
        self.assertIn("5%", claims[0].text)


class CredenceTierTest(unittest.TestCase):
    def test_strong_source_beats_model_assertion_and_cap_holds(self):
        strong_sub = Substrate()
        strong = assimilate(
            strong_sub,
            Claim(text="The Eiffel Tower is in Paris.", produced_by={"model": "m"}),
            {"source_id": "doc-1", "tier": "real_measurement", "direction": "+", "weight": 1.0},
        )
        self.assertGreater(strong.credence, 0.9)

        weak_sub = Substrate()
        weak = assimilate(
            weak_sub,
            Claim(text="The Eiffel Tower is in Paris.", produced_by={"model": "m"}),
            {"source_id": "self-assert", "tier": "model_assertion", "direction": "+", "weight": 1.0},
        )
        self.assertLessEqual(weak.credence, MODEL_ASSERTION_CAP)
        self.assertGreater(strong.credence, weak.credence)


class AntiLaunderingTest(unittest.TestCase):
    def test_self_reference_contributes_zero(self):
        sub = Substrate()
        claim = Claim(text="X causes Y.", produced_by={"model": "m"})
        b1 = assimilate(
            sub, claim, {"source_id": "self-assert", "tier": "model_assertion", "direction": "+", "weight": 1.0}
        )
        before = b1.credence

        # cite the belief's own id as "evidence" for itself, at a strong declared tier
        b2 = assimilate(sub, claim, {"source_id": b1.id, "tier": "held_out_truth", "direction": "+", "weight": 1.0})
        self.assertEqual(b2.credence, before)
        self.assertLessEqual(b2.credence, MODEL_ASSERTION_CAP)

    def test_ungrounded_peer_in_same_batch_contributes_zero(self):
        sub = Substrate()
        claim = Claim(text="X causes Y.", produced_by={"model": "m"})
        b1 = assimilate(
            sub, claim, {"source_id": "self-assert", "tier": "model_assertion", "direction": "+", "weight": 1.0}
        )
        before = b1.credence

        # a second claim whose ONLY support is also a bare model assertion (not independently grounded)
        peer = assimilate(
            sub,
            Claim(text="Z is true because X causes Y.", produced_by={"model": "m"}),
            {"source_id": "self-assert-2", "tier": "model_assertion", "direction": "+", "weight": 1.0},
        )
        self.assertLessEqual(peer.credence, MODEL_ASSERTION_CAP)

        # laundering attempt: cite the ungrounded peer as if it were solid evidence
        laundered = assimilate(
            sub, claim, {"source_id": peer.id, "tier": "real_measurement", "direction": "+", "weight": 1.0}
        )
        self.assertEqual(laundered.credence, before)
        self.assertLessEqual(laundered.credence, MODEL_ASSERTION_CAP)

    def test_citing_an_independently_grounded_belief_is_not_laundering(self):
        sub = Substrate()
        grounded = assimilate(
            sub,
            Claim(text="The rate is 5%.", produced_by={"model": "m"}),
            {"source_id": "doc-1", "tier": "held_out_truth", "direction": "+", "weight": 1.0},
        )
        self.assertGreater(grounded.credence, 0.5)

        downstream = assimilate(
            sub,
            Claim(text="Therefore the estimate holds.", produced_by={"model": "m"}),
            {"source_id": grounded.id, "tier": "held_out_truth", "direction": "+", "weight": 1.0},
        )
        self.assertGreater(downstream.credence, 0.5)

    def test_multi_hop_laundering_ring_is_rejected(self):
        """A -> B -> C -> back to A: each individual hop looks like a legitimate citation of a
        grounded belief, but the whole ring's only real support traces back to A's own claim. A
        citing C (closing the ring) must be rejected exactly like a direct self-citation would be --
        the one-hop check `_launders` used to do is not enough to catch this."""
        sub = Substrate()
        a = assimilate(
            sub,
            Claim(text="claim A", produced_by={"model": "m"}),
            {"source_id": "ext-doc-1", "tier": "real_measurement", "direction": "+", "weight": 1.0},
        )
        before = a.credence
        b = assimilate(
            sub,
            Claim(text="claim B", produced_by={"model": "m"}),
            {"source_id": a.id, "tier": "real_measurement", "direction": "+", "weight": 1.0},
        )
        self.assertGreater(b.credence, 0.5)  # legitimate: A is genuinely grounded
        c = assimilate(
            sub,
            Claim(text="claim C", produced_by={"model": "m"}),
            {"source_id": b.id, "tier": "real_measurement", "direction": "+", "weight": 1.0},
        )
        self.assertGreater(c.credence, 0.5)  # legitimate: B is genuinely grounded (via A)

        a_revised = assimilate(
            sub,
            Claim(text="claim A", produced_by={"model": "m"}),
            {"source_id": c.id, "tier": "real_measurement", "direction": "+", "weight": 1.0},
        )
        self.assertEqual(a_revised.credence, before)  # the ring closes -- rejected, credence unchanged


class RevisionAndRetractTest(unittest.TestCase):
    def test_contradicting_evidence_lowers_credence(self):
        sub = Substrate()
        claim = Claim(text="The rate is 5%.", produced_by={"model": "m"})
        supported = assimilate(
            sub, claim, {"source_id": "doc-1", "tier": "held_out_truth", "direction": "+", "weight": 1.0}
        )
        high = supported.credence

        revised = assimilate(
            sub, claim, {"source_id": "doc-2", "tier": "held_out_truth", "direction": "-", "weight": 1.0}
        )
        self.assertLess(revised.credence, high)

    def test_retract_lowers_dependents_and_cascades(self):
        sub = Substrate()
        base = assimilate(
            sub,
            Claim(text="The rate is 5%.", produced_by={"model": "m"}),
            {"source_id": "doc-1", "tier": "held_out_truth", "direction": "+", "weight": 1.0},
        )
        self.assertGreater(base.credence, 0.9)

        downstream = assimilate(
            sub,
            Claim(text="Therefore the estimate holds.", produced_by={"model": "m"}),
            {"source_id": base.id, "tier": "held_out_truth", "direction": "+", "weight": 1.0},
        )
        self.assertGreater(downstream.credence, 0.9)

        changed = retract(sub, "doc-1")
        changed_ids = {c.id for c in changed}

        # base lost its only support -> back to neutral, no longer independently grounded
        self.assertIn(base.id, changed_ids)
        rebased = next(c for c in changed if c.id == base.id)
        self.assertAlmostEqual(rebased.credence, 0.5, places=6)

        # cascade: downstream cited base, which is no longer grounded, so its citation is now zeroed too
        self.assertIn(downstream.id, changed_ids)
        redown = next(c for c in changed if c.id == downstream.id)
        self.assertAlmostEqual(redown.credence, 0.5, places=6)


class TraceableHistoryTest(unittest.TestCase):
    def test_replay_reproduces_stored_credence(self):
        sub = Substrate()
        b = assimilate(
            sub,
            Claim(text="Revenue grew 12% year over year.", produced_by={"model": "m"}),
            [
                {"source_id": "doc-1", "tier": "held_out_truth", "direction": "+", "weight": 1.0},
                {"source_id": "doc-2", "tier": "simulation", "direction": "+", "weight": 0.6},
                {"source_id": "assistant-1", "tier": "model_assertion", "direction": "+", "weight": 1.0},
            ],
        )
        replayed = credence_from_history(b.evidence_history)
        self.assertAlmostEqual(replayed, b.credence, places=9)


class CalibrationTest(unittest.TestCase):
    def test_coarse_reliability_across_evidence_profiles(self):
        rng = random.Random(0)
        sub = Substrate()
        profiles = {
            "strong": [{"source_id": "s", "tier": "real_measurement", "direction": "+", "weight": 1.0}],
            "medium": [{"source_id": "s", "tier": "held_out_truth", "direction": "+", "weight": 1.0}],
            "weak": [{"source_id": "s", "tier": "simulation", "direction": "+", "weight": 0.6}],
        }
        n_per_profile = 60
        results: dict[str, list[bool]] = {name: [] for name in profiles}
        target: dict[str, float] = {}

        for name, evidence in profiles.items():
            target[name] = credence_from_history([EvidenceEntry(**e) for e in evidence])
            for i in range(n_per_profile):
                truth = rng.random() < target[name]
                claim = Claim(text=f"{name}-claim-{i}", produced_by={"model": "m"})
                evidence_i = [dict(e, source_id=f"{e['source_id']}-{name}-{i}") for e in evidence]
                b = assimilate(sub, claim, evidence_i)
                self.assertAlmostEqual(b.credence, target[name], places=9)
                results[name].append(truth)

        for name, truths in results.items():
            rate = sum(truths) / len(truths)
            self.assertLess(abs(rate - target[name]), 0.15, msg=f"{name}: rate={rate} target={target[name]}")


class DirectionValidationTest(unittest.TestCase):
    """MXR-080-0242: direction is a closed enum ('+'/'-'). Before this was validated, any other string
    silently fell through the `direction == "+"` check as if it were "-", corrupting the credence math
    with contradicting evidence nobody actually submitted."""

    def test_unrecognized_direction_string_is_rejected_by_assimilate(self):
        sub = Substrate()
        claim = Claim(text="The bridge is closed.", produced_by={"model": "m"})
        with self.assertRaises(ValueError):
            assimilate(
                sub,
                claim,
                {"source_id": "self-assert", "tier": "model_assertion", "direction": "positive", "weight": 1.0},
            )

    def test_direct_construction_rejects_bad_direction(self):
        with self.assertRaises(ValueError):
            EvidenceEntry(source_id="x", tier="model_assertion", direction="up")

    def test_both_valid_directions_are_still_accepted(self):
        EvidenceEntry(source_id="x", tier="model_assertion", direction="+")
        EvidenceEntry(source_id="x", tier="model_assertion", direction="-")


class EvidenceNumericValidationTest(unittest.TestCase):
    """MXR-080-0242: weight and time must be finite (weight also non-negative) -- a NaN weight used to
    silently produce a NaN credence that would then be persisted and used for ranking."""

    def test_nan_weight_is_rejected_by_assimilate(self):
        sub = Substrate()
        claim = Claim(text="The reservoir is full.", produced_by={"model": "m"})
        with self.assertRaises(ValueError):
            assimilate(sub, claim, {"source_id": "self-assert", "tier": "model_assertion", "weight": float("nan")})

    def test_infinite_weight_is_rejected_by_assimilate(self):
        sub = Substrate()
        claim = Claim(text="The reservoir is full.", produced_by={"model": "m"})
        with self.assertRaises(ValueError):
            assimilate(sub, claim, {"source_id": "self-assert", "tier": "model_assertion", "weight": float("inf")})

    def test_negative_weight_is_rejected_by_assimilate(self):
        sub = Substrate()
        claim = Claim(text="The reservoir is full.", produced_by={"model": "m"})
        with self.assertRaises(ValueError):
            assimilate(sub, claim, {"source_id": "self-assert", "tier": "model_assertion", "weight": -1.0})

    def test_nan_time_is_rejected_by_assimilate(self):
        sub = Substrate()
        claim = Claim(text="The reservoir is full.", produced_by={"model": "m"})
        with self.assertRaises(ValueError):
            assimilate(sub, claim, {"source_id": "self-assert", "tier": "model_assertion", "time": float("nan")})

    def test_direct_construction_also_rejects_nan_weight(self):
        # __post_init__ guards EVERY construction path, not just assimilate's -- a directly-built
        # entry (a test, or _from_item deserializing a stored one) gets the same guarantee.
        with self.assertRaises(ValueError):
            EvidenceEntry(source_id="x", tier="model_assertion", weight=float("nan"))

    def test_direct_construction_also_rejects_infinite_time(self):
        with self.assertRaises(ValueError):
            EvidenceEntry(source_id="x", tier="model_assertion", time=float("inf"))


class NumericallyStableLogisticTest(unittest.TestCase):
    """MXR-080-0242: a very large but FINITE weight must not overflow the naive 1/(1+exp(-x)) logistic
    -- finiteness validation alone does not catch this (1e300 is finite); only a numerically stable
    sigmoid does."""

    def test_large_negative_evidence_does_not_crash_and_saturates_toward_zero(self):
        sub = Substrate()
        claim = Claim(text="The volcano is dormant.", produced_by={"model": "m"})
        b = assimilate(
            sub, claim, {"source_id": "self-assert", "tier": "model_assertion", "direction": "-", "weight": 1e300}
        )
        self.assertTrue(math.isfinite(b.credence))
        self.assertAlmostEqual(b.credence, 0.0, places=9)

    def test_large_positive_evidence_does_not_crash_and_saturates_toward_one(self):
        sub = Substrate()
        claim = Claim(text="The comet will return.", produced_by={"model": "m"})
        b = assimilate(
            sub, claim, {"source_id": "doc-huge", "tier": "real_measurement", "direction": "+", "weight": 1e300}
        )
        self.assertTrue(math.isfinite(b.credence))
        self.assertAlmostEqual(b.credence, 1.0, places=9)

    def test_replay_of_a_large_weight_history_is_also_stable(self):
        # credence_from_history is the pure replay path -- must be equally stable independent of assimilate.
        history = [EvidenceEntry(source_id="s", tier="real_measurement", direction="-", weight=1e250)]
        credence = credence_from_history(history)
        self.assertTrue(math.isfinite(credence))
        self.assertAlmostEqual(credence, 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
