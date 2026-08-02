"""A receipt must be bound to what it describes (MXR-080-1874).

Five typed-runtime and planning receipts were annotation-only: each carried the fields an auditor
would read but held none of them to the fact they assert. The shared failure is that a receipt
records a *decision*, and a decision that can be restated after the fact -- or that can contradict
its own evidence at construction -- is not evidence of anything.

Each test below is one reproduced construction, not a screening result. What is deliberately NOT
checked is recorded beside it: a scheduler may legitimately overspend a soft budget, and a rejected
commit legitimately leaves the version vector where it found it.
"""

import json
import unittest

from mixle.experimental.typed_runtime.cache import InvalidationReceipt
from mixle.experimental.typed_runtime.contracts import ArtifactKind, ObjectiveKind
from mixle.experimental.typed_runtime.replay import ReplayStepReceipt
from mixle.experimental.typed_runtime.scheduler import ScheduleReceipt
from mixle.experimental.typed_runtime.transaction import CanaryVerdict, CommitReceipt, CommitStatus
from mixle.inference.planning import Guarantee, VerificationReceipt


def _versions(model_version: int) -> dict:
    return {"model_version": model_version, "node_versions": {"n": 0}}


def _commit(**overrides) -> CommitReceipt:
    fields = dict(
        commit_id="commit-0",
        batch_id="batch-0",
        proposal_ids=("p0",),
        status=CommitStatus.ACCEPTED,
        reason="canary-accepted",
        versions_before=_versions(0),
        versions_after=_versions(1),
        canary=CanaryVerdict(accepted=True, reason="measured"),
        run_id="run",
        model_id="model",
    )
    fields.update(overrides)
    return CommitReceipt(**fields)


class AcceptedCommitTest(unittest.TestCase):
    """``ACCEPTED`` is the strongest claim the class makes and was the one status nothing checked."""

    def test_the_producer_shaped_acceptance_constructs(self):
        self.assertTrue(_commit().accepted)

    def test_an_acceptance_with_no_canary_is_refused(self):
        with self.assertRaisesRegex(ValueError, "must carry the canary verdict"):
            _commit(canary=None)

    def test_an_acceptance_its_own_canary_rejected_is_refused(self):
        with self.assertRaisesRegex(ValueError, "its own canary rejected"):
            _commit(canary=CanaryVerdict(accepted=False, reason="regression"))

    def test_an_acceptance_that_did_not_advance_the_model_is_refused(self):
        with self.assertRaisesRegex(ValueError, "model_version did not advance"):
            _commit(versions_after=_versions(0))

    def test_a_rejection_still_needs_neither_canary_nor_transition(self):
        # A preflight rejection happens before any measurement and moves nothing; requiring either
        # here would refuse the receipt the coordinator actually emits.
        receipt = _commit(
            status=CommitStatus.REJECTED, reason="frozen-node:n", canary=None, versions_after=_versions(0)
        )
        self.assertFalse(receipt.accepted)


class NestedMutabilityTest(unittest.TestCase):
    """The previous fix severed only the TOP-level alias; a version vector is two levels deep."""

    def test_a_nested_version_map_is_detached_from_the_caller(self):
        versions = _versions(0)
        receipt = _commit(
            status=CommitStatus.REJECTED, reason="no", canary=None, versions_before=versions, versions_after=versions
        )
        versions["node_versions"]["n"] = 99
        self.assertEqual(receipt.versions_before["node_versions"]["n"], 0)

    def test_a_nested_version_map_is_read_only(self):
        receipt = _commit()
        with self.assertRaises(TypeError):
            receipt.versions_after["node_versions"]["n"] = 5

    def test_as_dict_still_returns_plain_json_containers(self):
        payload = _commit().as_dict()
        self.assertIsInstance(payload["versions_after"]["node_versions"], dict)
        json.dumps(payload)

    def test_canary_metrics_are_detached_and_read_only(self):
        metrics = {"loss": 1.0}
        verdict = CanaryVerdict(accepted=True, reason="measured", metrics=metrics)
        metrics["loss"] = 99.0
        self.assertEqual(verdict.metrics["loss"], 1.0)
        with self.assertRaises(TypeError):
            verdict.metrics["loss"] = 2.0


class ScheduleReceiptTest(unittest.TestCase):
    def _receipt(self, **overrides) -> ScheduleReceipt:
        fields = dict(
            round_index=0,
            model_version=0,
            target_objective=ObjectiveKind.MLE,
            selected_nodes=(),
            ranked_nodes=(),
            eligible_nodes=(),
            forced_starvation=(),
            bootstrap_nodes=(),
            lower_confidence_bounds={},
            effective_costs={},
            priorities={},
            invalidation_costs={},
            skipped={},
            rejected_evidence={},
            budget=1.0,
            spent=0.5,
        )
        fields.update(overrides)
        return ScheduleReceipt(**fields)

    def test_a_baseline_receipt_constructs(self):
        self.assertEqual(self._receipt().budget_overrun, 0.0)

    def test_a_negative_round_or_version_is_refused(self):
        for field in ("round_index", "model_version"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "must be a non-negative integer"):
                    self._receipt(**{field: -5})

    def test_a_negative_budget_or_spend_is_refused(self):
        for field in ("budget", "spent"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    self._receipt(**{field: -1.0})

    def test_overspending_a_soft_budget_is_still_allowed(self):
        # Deliberately not refused: a fairness-forced selection may exceed its soft budget, which is
        # exactly what budget_overrun exists to report.
        self.assertAlmostEqual(self._receipt(budget=1.0, spent=3.0).budget_overrun, 2.0)

    def test_selecting_a_node_that_was_not_eligible_is_refused(self):
        with self.assertRaisesRegex(ValueError, "eligible set does not contain"):
            self._receipt(selected_nodes=("ghost",))

    def test_selecting_and_skipping_the_same_node_is_refused(self):
        with self.assertRaisesRegex(ValueError, "both selected and skipped"):
            self._receipt(selected_nodes=("n",), eligible_nodes=("n",), skipped={"n": "budget"})


class ReplayStepTest(unittest.TestCase):
    def test_matched_true_with_mismatches_is_refused(self):
        with self.assertRaisesRegex(ValueError, "matched=True while listing mismatches"):
            ReplayStepReceipt(0, "commit-0", True, ("status", "reason"))

    def test_the_two_consistent_verdicts_still_construct(self):
        self.assertTrue(ReplayStepReceipt(0, "commit-0", True, ()).matched)
        self.assertFalse(ReplayStepReceipt(0, "commit-0", False, ("status",)).matched)

    def test_a_non_boolean_verdict_is_refused(self):
        with self.assertRaisesRegex(TypeError, "must be a Boolean verdict"):
            ReplayStepReceipt(0, "commit-0", "yes", ())


class InvalidationReceiptTest(unittest.TestCase):
    def _receipt(self, **overrides) -> InvalidationReceipt:
        fields = dict(
            source_nodes=("n",),
            written_artifact=ArtifactKind.PARAMETERS,
            invalidated_nodes=("n",),
            removed_entries=(),
            generations={"n": 1},
        )
        fields.update(overrides)
        return InvalidationReceipt(**fields)

    def test_a_baseline_receipt_constructs(self):
        json.dumps(self._receipt().as_dict())

    def test_a_string_artifact_is_refused_where_as_dict_needs_an_enum(self):
        with self.assertRaisesRegex(TypeError, "must be an ArtifactKind"):
            self._receipt(written_artifact="not-an-artifact")

    def test_a_bare_string_is_not_a_node_list(self):
        with self.assertRaisesRegex(TypeError, "must be a sequence of node ids"):
            self._receipt(source_nodes="ab")

    def test_generations_are_detached_from_the_caller(self):
        generations = {"n": 1}
        receipt = self._receipt(generations=generations)
        generations["n"] = 99
        self.assertEqual(receipt.generations["n"], 1)

    def test_removing_an_entry_outside_the_invalidated_set_is_refused(self):
        with self.assertRaisesRegex(ValueError, "invalidated_nodes does not contain"):
            self._receipt(removed_entries=(("ghost", ArtifactKind.PARAMETERS),))


class VerificationReceiptTest(unittest.TestCase):
    def _receipt(self, **overrides) -> VerificationReceipt:
        fields = dict(
            receipt_id="v0",
            block="component[0]",
            guarantee=Guarantee.STATIONARY,
            checks=("gradient_norm",),
            source="test",
            evidence={"gradient_norm": 1e-9},
        )
        fields.update(overrides)
        return VerificationReceipt(**fields)

    def test_a_baseline_receipt_constructs(self):
        json.dumps(self._receipt().evidence_as_dict())

    def test_evidence_an_audit_cannot_read_is_refused(self):
        with self.assertRaisesRegex(TypeError, "neither immutable nor JSON-expressible"):
            self._receipt(evidence={"session": object()})

    def test_nested_evidence_is_detached_from_the_caller(self):
        evidence = {"trace": {"norms": [1.0, 2.0]}}
        receipt = self._receipt(evidence=evidence)
        evidence["trace"]["norms"].append(3.0)
        self.assertEqual(receipt.evidence["trace"]["norms"], (1.0, 2.0))

    def test_a_non_finite_measurement_is_refused_because_the_receipt_must_serialize(self):
        with self.assertRaisesRegex(ValueError, "must be finite to serialize"):
            self._receipt(evidence={"gain": float("inf")})

    def test_an_integer_guarantee_is_refused_where_the_ladder_needs_a_label(self):
        with self.assertRaisesRegex(TypeError, "must be a Guarantee"):
            self._receipt(guarantee=2)


if __name__ == "__main__":
    unittest.main()
