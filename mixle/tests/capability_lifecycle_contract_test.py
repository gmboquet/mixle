"""Focused contract checks for independent capability lifecycle dimensions."""

import unittest
from datetime import UTC, datetime, timedelta

from mixle.capability_lifecycle import (
    AuthorizationDecision,
    AuthorizationOutcome,
    AuthorizationStatus,
    CapabilityIdentity,
    CapabilityLifecycle,
    CapabilityMaturity,
    EpistemicStanding,
    EvaluationState,
    LifecycleTransitionError,
    OperationalState,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


class CapabilityLifecycleContractTest(unittest.TestCase):
    def setUp(self):
        self.identity = CapabilityIdentity("capability.mesh.solve", "1.2.0", "sha256:abc")

    def test_dimensions_transition_independently_and_round_trip(self):
        lifecycle = CapabilityLifecycle(self.identity, updated_at=T0)
        lifecycle = lifecycle.evolve(
            maturity=CapabilityMaturity.CANDIDATE,
            operational=OperationalState.AVAILABLE,
            evaluation=EvaluationState.RUNNING,
            epistemic=EpistemicStanding.HYPOTHESIS,
            at=T0 + timedelta(minutes=1),
        )
        lifecycle = lifecycle.evolve(
            evaluation=EvaluationState.PASSED,
            epistemic=EpistemicStanding.CORROBORATED,
            at=T0 + timedelta(minutes=2),
        )
        lifecycle = lifecycle.evolve(
            maturity=CapabilityMaturity.VALIDATED,
            at=T0 + timedelta(minutes=3),
        )
        lifecycle = lifecycle.evolve(
            evaluation=EvaluationState.STALE,
            operational=OperationalState.DEGRADED,
            at=T0 + timedelta(minutes=4),
        )
        self.assertEqual(lifecycle.maturity, CapabilityMaturity.VALIDATED)
        self.assertEqual(lifecycle.evaluation, EvaluationState.STALE)
        self.assertEqual(lifecycle.operational, OperationalState.DEGRADED)
        self.assertEqual(CapabilityLifecycle.from_dict(lifecycle.as_dict()), lifecycle)

    def test_success_does_not_promote_and_promotion_requires_passed_evidence(self):
        lifecycle = CapabilityLifecycle(self.identity, updated_at=T0).evolve(
            maturity=CapabilityMaturity.CANDIDATE,
            evaluation=EvaluationState.RUNNING,
            at=T0 + timedelta(minutes=1),
        )
        lifecycle = lifecycle.evolve(evaluation=EvaluationState.PASSED, at=T0 + timedelta(minutes=2))
        self.assertEqual(lifecycle.maturity, CapabilityMaturity.CANDIDATE)

        unevaluated = CapabilityLifecycle(self.identity, updated_at=T0).evolve(
            maturity=CapabilityMaturity.CANDIDATE,
            at=T0 + timedelta(minutes=1),
        )
        with self.assertRaisesRegex(LifecycleTransitionError, "requires a passed evaluation"):
            unevaluated.evolve(maturity=CapabilityMaturity.VALIDATED, at=T0 + timedelta(minutes=2))

    def test_authorization_is_scoped_expiring_and_revocable(self):
        decision = AuthorizationDecision(
            decision_id="auth-1",
            capability=self.identity,
            outcome=AuthorizationOutcome.GRANTED,
            issued_by="safety-board",
            scopes=frozenset({"sandbox"}),
            decided_at=T0,
            expires_at=T0 + timedelta(hours=2),
        )
        lifecycle = CapabilityLifecycle(self.identity, authorization=decision, updated_at=T0 + timedelta(minutes=1))
        self.assertTrue(lifecycle.allows("sandbox"))
        self.assertFalse(lifecycle.allows("production"))
        self.assertEqual(decision.status_at(T0 + timedelta(hours=3)), AuthorizationStatus.EXPIRED)

        revoked = decision.revoke(by="safety-board", at=T0 + timedelta(hours=1), reason="new hazard")
        self.assertEqual(revoked.status_at(T0 + timedelta(hours=1)), AuthorizationStatus.REVOKED)
        self.assertEqual(AuthorizationDecision.from_dict(revoked.as_dict()), revoked)

    def test_directly_constructed_string_outcomes_are_canonicalized_not_trusted(self):
        # MXR-080-1677: a str-valued outcome used to survive __post_init__ untouched, so the
        # identity comparisons in status_at() fell through and reported a denial as GRANTED.
        denied = AuthorizationDecision(
            decision_id="auth-denied",
            capability=self.identity,
            outcome="denied",
            issued_by="safety-board",
            scopes=frozenset({"run"}),
            decided_at=T0,
        )
        self.assertIs(denied.outcome, AuthorizationOutcome.DENIED)
        self.assertEqual(denied.status_at(T0), AuthorizationStatus.DENIED)
        self.assertFalse(denied.allows("run", at=T0))
        self.assertEqual(denied.as_dict()["outcome"], "denied")
        self.assertEqual(AuthorizationDecision.from_dict(denied.as_dict()), denied)

        granted = AuthorizationDecision(
            decision_id="auth-granted",
            capability=self.identity,
            outcome="granted",
            issued_by="safety-board",
            scopes=frozenset({"run"}),
            decided_at=T0,
        )
        self.assertIs(granted.outcome, AuthorizationOutcome.GRANTED)
        self.assertTrue(granted.allows("run", at=T0))
        self.assertIsNotNone(granted.revoke(by="safety-board", at=T0 + timedelta(hours=1)).revoked_at)

        with self.assertRaises(ValueError):
            AuthorizationDecision(
                decision_id="auth-bogus",
                capability=self.identity,
                outcome="probably",
                issued_by="safety-board",
                scopes=frozenset({"run"}),
                decided_at=T0,
            )

    def _record(self, **overrides):
        record = {
            "capability": self.identity.as_dict(),
            "maturity": "concept",
            "operational": "unavailable",
            "evaluation": "unevaluated",
            "epistemic": "unassessed",
            "authorization": None,
            "revision": 0,
            "updated_at": "2026-01-01T00:00:00Z",
        }
        record.update(overrides)
        return record

    def test_direct_snapshots_canonicalize_states_and_cannot_forge_a_promotion(self):
        # MXR-080-1678: string dimensions used to be stored verbatim, and nothing required a
        # validated/supported snapshot to have been evaluated at all, so a direct construction
        # bypassed every promotion gate and only failed later inside as_dict().
        lifecycle = CapabilityLifecycle(
            self.identity,
            maturity="candidate",
            operational="available",
            evaluation="running",
            epistemic="hypothesis",
            updated_at=T0,
        )
        self.assertIs(lifecycle.maturity, CapabilityMaturity.CANDIDATE)
        self.assertIs(lifecycle.operational, OperationalState.AVAILABLE)
        self.assertIs(lifecycle.evaluation, EvaluationState.RUNNING)
        self.assertIs(lifecycle.epistemic, EpistemicStanding.HYPOTHESIS)
        self.assertEqual(lifecycle.as_dict()["maturity"], "candidate")

        with self.assertRaisesRegex(ValueError, "cannot be unevaluated"):
            CapabilityLifecycle(
                self.identity,
                maturity="supported",
                operational="available",
                evaluation="unevaluated",
                updated_at=T0,
            )
        with self.assertRaises(ValueError):
            CapabilityLifecycle(self.identity, maturity="extremely-supported", updated_at=T0)

        # A promotion that was legitimately earned survives a later stale/failed evaluation:
        # the dimensions stay independent once the gate itself has been passed.
        earned = (
            CapabilityLifecycle(self.identity, updated_at=T0)
            .evolve(maturity=CapabilityMaturity.CANDIDATE, evaluation=EvaluationState.RUNNING, at=T0)
            .evolve(evaluation=EvaluationState.PASSED, at=T0)
            .evolve(maturity=CapabilityMaturity.VALIDATED, at=T0)
            .evolve(evaluation=EvaluationState.STALE, at=T0)
        )
        self.assertIs(earned.evaluation, EvaluationState.STALE)
        self.assertEqual(CapabilityLifecycle.from_dict(earned.as_dict()), earned)

    def test_revision_is_validated_before_conversion(self):
        # MXR-080-1678: from_dict() ran int() first, so a persisted -0.5 silently became 0.
        for bad in (-0.5, 2.5, True, "3", float("nan")):
            with self.assertRaises(ValueError):
                CapabilityLifecycle.from_dict(self._record(revision=bad))
        with self.assertRaises(ValueError):
            CapabilityLifecycle(self.identity, revision=True, updated_at=T0)
        self.assertEqual(CapabilityLifecycle.from_dict(self._record(revision=3)).revision, 3)

    def test_illegal_transitions_and_cross_identity_authorization_fail(self):
        lifecycle = CapabilityLifecycle(self.identity, updated_at=T0)
        with self.assertRaisesRegex(LifecycleTransitionError, "concept -> supported"):
            lifecycle.evolve(maturity=CapabilityMaturity.SUPPORTED, at=T0 + timedelta(minutes=1))
        other = AuthorizationDecision(
            decision_id="auth-other",
            capability=CapabilityIdentity("other", "1"),
            outcome=AuthorizationOutcome.GRANTED,
            issued_by="owner",
            scopes=frozenset({"sandbox"}),
            decided_at=T0,
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            CapabilityLifecycle(self.identity, authorization=other, updated_at=T0)

    def test_existing_substrate_governance_adapts_without_changing_legacy_behavior(self):
        from mixle.substrate import Governance, Substrate, approve, authorization_decision, propose

        substrate = Substrate()
        item = substrate.add(kind="artifact", text="candidate", scope="team")
        governance = Governance().grant("reviewer", "org", by="root")
        self.assertEqual(propose(substrate, [item], to="org", by="author"), [item])
        self.assertTrue(approve(substrate, item, by="reviewer", governance=governance))
        decision = authorization_decision(
            substrate,
            item,
            capability_id=self.identity.capability_id,
            version=self.identity.version,
            digest=self.identity.digest,
        )
        self.assertEqual(decision.outcome, AuthorizationOutcome.GRANTED)
        self.assertTrue(decision.allows("org", at=datetime.now(UTC)))


if __name__ == "__main__":
    unittest.main()
