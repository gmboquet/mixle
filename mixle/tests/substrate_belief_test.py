"""KNOW-a: harvest -> assimilate calibrated belief, anti-laundering, revision/retract, replay."""

import math
import random
import unittest

from mixle.substrate import PUBLIC, AccessPolicy, Substrate
from mixle.substrate.belief import (
    MODEL_ASSERTION_CAP,
    Claim,
    EvidenceEntry,
    assimilate,
    credence_from_history,
    harvest_knowledge,
    retract,
)


def _register(sub: Substrate, source_id: str, *, scope: str = "local") -> None:
    """Register ``source_id`` as a real, resolvable substrate item -- MXR-080-0243 requires a strong
    evidence tier to resolve to something real (a receipt) before it earns its claimed strength; a bare,
    never-registered string source_id no longer earns credence on its own say-so."""
    sub.add(kind="text", text=f"evidence receipt for {source_id}", id=source_id, scope=scope)


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
        _register(strong_sub, "doc-1")
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
        _register(sub, "doc-1")
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
        _register(sub, "ext-doc-1")
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
        _register(sub, "doc-1")
        _register(sub, "doc-2")
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
        _register(sub, "doc-1")
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
                for e in evidence_i:
                    _register(sub, e["source_id"])  # MXR-080-0243: a receipt, not just a claimed tier
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
        _register(sub, "doc-huge")
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


class EvidenceVerificationTest(unittest.TestCase):
    """MXR-080-0243: a claimed evidence tier is not itself evidence. A strong tier only earns its
    claimed strength when source_id resolves to something real; a fabricated, unresolved, or omitted
    source_id must be represented as unverified (model-assertion strength) instead."""

    def test_fabricated_source_id_does_not_earn_strong_credence(self):
        sub = Substrate()
        claim = Claim(text="The bridge load limit is 40 tons.", produced_by={"model": "m"})
        b = assimilate(
            sub,
            claim,
            {"source_id": "totally-fabricated-nonexistent-id-9182", "tier": "real_measurement", "weight": 1.0},
        )
        self.assertLessEqual(b.credence, MODEL_ASSERTION_CAP)

    def test_missing_source_id_does_not_earn_strong_credence(self):
        sub = Substrate()
        claim = Claim(text="The bridge load limit is 40 tons.", produced_by={"model": "m"})
        b = assimilate(sub, claim, {"tier": "real_measurement", "weight": 1.0})  # no source_id at all
        self.assertLessEqual(b.credence, MODEL_ASSERTION_CAP)

    def test_registered_source_id_does_earn_strong_credence(self):
        sub = Substrate()
        _register(sub, "genuine-doc")
        claim = Claim(text="The bridge load limit is 40 tons.", produced_by={"model": "m"})
        b = assimilate(sub, claim, {"source_id": "genuine-doc", "tier": "real_measurement", "weight": 1.0})
        self.assertGreater(b.credence, 0.9)

    def test_model_assertion_tier_needs_no_receipt(self):
        sub = Substrate()
        claim = Claim(text="The bridge load limit is 40 tons.", produced_by={"model": "m"})
        b = assimilate(sub, claim, {"source_id": "unregistered-whatever", "tier": "model_assertion", "weight": 1.0})
        self.assertLessEqual(b.credence, MODEL_ASSERTION_CAP)

    def test_verified_flag_is_recorded_on_the_entry(self):
        sub = Substrate()
        _register(sub, "genuine-doc-2")
        claim = Claim(text="Water boils at 100C at sea level.", produced_by={"model": "m"})
        b = assimilate(sub, claim, {"source_id": "genuine-doc-2", "tier": "real_measurement", "weight": 1.0})
        self.assertTrue(b.evidence_history[0].verified)

        claim2 = Claim(text="Water boils at 90C at sea level.", produced_by={"model": "m"})
        b2 = assimilate(sub, claim2, {"source_id": "fabricated-xyz-999", "tier": "real_measurement", "weight": 1.0})
        self.assertFalse(b2.evidence_history[0].verified)


class CrossScopeEvidenceLaunderingTest(unittest.TestCase):
    """MXR-080-0244: source_id must be resolved through an authorized, scope-respecting view -- never a
    raw, scope-blind sub.get -- at the first hop AND at every transitive hop of a multi-belief proof
    chain. Same adversarial shape as the core.py MXR-080-0237 fix: two scopes, distinguishable content,
    a caller confined to one scope who must not be able to use, or learn anything from, the other
    scope's private beliefs."""

    def test_team_b_cannot_borrow_team_as_private_grounding(self):
        sub = Substrate()
        # team-a genuinely grounds a belief in its OWN private scope, from a receipt only team-a can see.
        sub.add(kind="text", text="team-a's confidential sensor reading", id="team-a-secret-reading", scope="team-a")
        team_a_belief = assimilate(
            sub,
            Claim(text="The reactor core temperature is nominal.", produced_by={"model": "m"}),
            {"source_id": "team-a-secret-reading", "tier": "real_measurement", "weight": 1.0},
            scope="team-a",
        )
        self.assertGreater(team_a_belief.credence, 0.9)  # genuinely grounded, within team-a's own scope

        # team-b (no grant to read team-a) cites team-a's belief id directly, claiming real_measurement.
        team_b_belief = assimilate(
            sub,
            Claim(text="Team B's unrelated claim riding on team A's reading.", produced_by={"model": "m"}),
            {"source_id": team_a_belief.id, "tier": "real_measurement", "weight": 1.0},
            scope="team-b",
        )
        # must NOT inherit team-a's grounding -- capped exactly like citing a nonexistent id would be.
        self.assertLessEqual(team_b_belief.credence, MODEL_ASSERTION_CAP)
        self.assertFalse(team_b_belief.evidence_history[0].verified)

    def test_grounded_and_ungrounded_team_a_beliefs_are_indistinguishable_from_team_b(self):
        """The disclosure angle: team-b must not be able to tell, from the resulting credence, whether
        the team-a belief id it cited was actually grounded -- both must look identical (capped)."""
        sub = Substrate()
        sub.add(kind="text", text="team-a's real receipt", id="team-a-real-receipt", scope="team-a")
        grounded_a = assimilate(
            sub,
            Claim(text="Team A claim ONE, genuinely grounded.", produced_by={"model": "m"}),
            {"source_id": "team-a-real-receipt", "tier": "real_measurement", "weight": 1.0},
            scope="team-a",
        )
        ungrounded_a = assimilate(
            sub,
            Claim(text="Team A claim TWO, only ever self-asserted.", produced_by={"model": "m"}),
            {"source_id": "team-a-self-assert", "tier": "model_assertion", "weight": 1.0},
            scope="team-a",
        )
        self.assertGreater(grounded_a.credence, 0.9)
        self.assertLessEqual(ungrounded_a.credence, MODEL_ASSERTION_CAP)

        via_grounded = assimilate(
            sub,
            Claim(text="Team B cites A's grounded claim.", produced_by={"model": "m"}),
            {"source_id": grounded_a.id, "tier": "real_measurement", "weight": 1.0},
            scope="team-b",
        )
        via_ungrounded = assimilate(
            sub,
            Claim(text="Team B cites A's ungrounded claim.", produced_by={"model": "m"}),
            {"source_id": ungrounded_a.id, "tier": "real_measurement", "weight": 1.0},
            scope="team-b",
        )
        # indistinguishable: team-b gets the SAME (capped) result regardless of A's internal grounding.
        self.assertEqual(via_grounded.credence, via_ungrounded.credence)
        self.assertLessEqual(via_grounded.credence, MODEL_ASSERTION_CAP)

    def test_transitive_proof_chain_stays_scoped_at_every_hop(self):
        """team-a: X (grounded) <- Y (cites X, legitimate WITHIN team-a). team-b cites Y directly, two
        hops from team-a's real receipt -- must be denied at every hop, not just the first."""
        sub = Substrate()
        sub.add(kind="text", text="team-a's root receipt", id="team-a-root-receipt", scope="team-a")
        x = assimilate(
            sub,
            Claim(text="Team A root claim X.", produced_by={"model": "m"}),
            {"source_id": "team-a-root-receipt", "tier": "real_measurement", "weight": 1.0},
            scope="team-a",
        )
        y = assimilate(
            sub,
            Claim(text="Team A downstream claim Y, citing X.", produced_by={"model": "m"}),
            {"source_id": x.id, "tier": "real_measurement", "weight": 1.0},
            scope="team-a",
        )
        self.assertGreater(y.credence, 0.9)  # legitimate WITHIN team-a: y really is grounded via x

        team_b_via_y = assimilate(
            sub,
            Claim(text="Team B claim citing A's Y, two hops from A's real receipt.", produced_by={"model": "m"}),
            {"source_id": y.id, "tier": "real_measurement", "weight": 1.0},
            scope="team-b",
        )
        self.assertLessEqual(team_b_via_y.credence, MODEL_ASSERTION_CAP)

    def test_explicit_grant_legitimately_restores_cross_scope_grounding(self):
        """The mechanism is a real access gate, not a blanket cross-scope deny: a caller-supplied
        AccessPolicy granting team-b read access to team-a correctly lets team-b use team-a's grounding
        -- reusing spaces.AccessPolicy's own grant semantics, not a parallel concept."""
        sub = Substrate()
        sub.add(kind="text", text="team-a's shared-by-grant receipt", id="team-a-grant-receipt", scope="team-a")
        team_a_belief = assimilate(
            sub,
            Claim(text="Team A claim, later shared by explicit grant.", produced_by={"model": "m"}),
            {"source_id": "team-a-grant-receipt", "tier": "real_measurement", "weight": 1.0},
            scope="team-a",
        )
        policy = AccessPolicy().grant_read("team-b", "team-a")
        team_b_belief = assimilate(
            sub,
            Claim(text="Team B claim citing A's belief, WITH an explicit read grant.", produced_by={"model": "m"}),
            {"source_id": team_a_belief.id, "tier": "real_measurement", "weight": 1.0},
            scope="team-b",
            policy=policy,
        )
        self.assertGreater(team_b_belief.credence, 0.9)

    def test_public_scope_grounding_is_usable_from_any_scope(self):
        sub = Substrate()
        sub.add(kind="text", text="a publicly shared receipt", id="public-receipt", scope=PUBLIC)
        public_belief = assimilate(
            sub,
            Claim(text="A publicly grounded claim.", produced_by={"model": "m"}),
            {"source_id": "public-receipt", "tier": "real_measurement", "weight": 1.0},
            scope=PUBLIC,
        )
        team_b_belief = assimilate(
            sub,
            Claim(text="Team B cites the public claim.", produced_by={"model": "m"}),
            {"source_id": public_belief.id, "tier": "real_measurement", "weight": 1.0},
            scope="team-b",
        )
        self.assertGreater(team_b_belief.credence, 0.9)


if __name__ == "__main__":
    unittest.main()
