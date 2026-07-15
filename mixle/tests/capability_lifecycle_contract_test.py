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
        governance = Governance().grant("reviewer", "org")
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
