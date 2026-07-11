"""Deterministic and tolerance replay for proposal/commit logs."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from mixle.experimental.typed_runtime.proposal import ProposalBatch, payload_fingerprint
from mixle.experimental.typed_runtime.transaction import CommitReceipt, TransactionalCoordinator


class ReplayMode(StrEnum):
    """State comparison guarantee expected from a backend."""

    BITWISE = "bitwise"
    TOLERANCE = "tolerance"


@dataclass(frozen=True)
class ReplayEntry:
    """One immutable batch, expected receipt, and optional resulting state."""

    batch: ProposalBatch
    expected_receipt: CommitReceipt
    expected_state: Any = field(default=None, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        """Return metadata without serializing arbitrary expected state."""

        return {
            "batch": self.batch.as_dict(),
            "expected_receipt": self.expected_receipt.as_dict(),
            "has_expected_state": self.expected_state is not None,
            "expected_state_fingerprint": (
                payload_fingerprint(self.expected_state) if self.expected_state is not None else None
            ),
        }


@dataclass
class ReplayLog:
    """Append-only sequence of proposal batches and terminal commit receipts."""

    entries: list[ReplayEntry] = field(default_factory=list)

    def record(self, batch: ProposalBatch, receipt: CommitReceipt, *, expected_state: Any = None) -> None:
        """Record detached replay inputs so later caller mutation cannot rewrite history."""

        if receipt.batch_id != batch.batch_id:
            raise ValueError("commit receipt does not belong to the proposal batch.")
        self.entries.append(ReplayEntry(copy.deepcopy(batch), receipt, copy.deepcopy(expected_state)))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible replay manifest."""

        return {"entries": [entry.as_dict() for entry in self.entries]}


@dataclass(frozen=True)
class ReplayStepReceipt:
    """Comparison result for one replayed transaction."""

    index: int
    commit_id: str
    matched: bool
    mismatches: tuple[str, ...]
    actual_receipt: CommitReceipt | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible replay comparison."""

        return {
            "index": self.index,
            "commit_id": self.commit_id,
            "matched": self.matched,
            "mismatches": list(self.mismatches),
            "actual_receipt": self.actual_receipt.as_dict() if self.actual_receipt is not None else None,
        }


@dataclass(frozen=True)
class ReplayReport:
    """Whole-log replay verdict with explicit numeric tolerance."""

    mode: ReplayMode
    absolute_tolerance: float
    relative_tolerance: float
    steps: tuple[ReplayStepReceipt, ...]

    @property
    def matched(self) -> bool:
        """Whether every recorded transaction replayed under the selected guarantee."""

        return all(step.matched for step in self.steps)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible replay report."""

        return {
            "mode": self.mode.value,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "matched": self.matched,
            "steps": [step.as_dict() for step in self.steps],
        }


StateProbe = Callable[[], Any]


def _numeric_close(left: Any, right: Any, *, atol: float, rtol: float) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(np.allclose(np.asarray(left), np.asarray(right), atol=atol, rtol=rtol, equal_nan=False))
        except (TypeError, ValueError):
            return False
    if isinstance(left, (int, float, np.number)) and isinstance(right, (int, float, np.number)):
        return math.isclose(float(left), float(right), abs_tol=atol, rel_tol=rtol)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            _numeric_close(left[key], right[key], atol=atol, rtol=rtol) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_numeric_close(a, b, atol=atol, rtol=rtol) for a, b in zip(left, right))
    return left == right


def _canary_mismatches(expected: CommitReceipt, actual: CommitReceipt, *, atol: float, rtol: float) -> list[str]:
    if expected.canary is None or actual.canary is None:
        return [] if expected.canary is actual.canary else ["canary-presence"]
    mismatches = []
    if expected.canary.accepted != actual.canary.accepted or expected.canary.reason != actual.canary.reason:
        mismatches.append("canary-verdict")
    numeric = (
        "objective_before",
        "objective_after",
        "lower_confidence_gain",
        "confidence_level",
    )
    for name in numeric:
        left = getattr(expected.canary, name)
        right = getattr(actual.canary, name)
        if left is None or right is None:
            if left is not right:
                mismatches.append("canary-%s" % name)
        elif not math.isclose(left, right, abs_tol=atol, rel_tol=rtol):
            mismatches.append("canary-%s" % name)
    if expected.canary.sample_count != actual.canary.sample_count:
        mismatches.append("canary-sample-count")
    if not _numeric_close(expected.canary.metrics, actual.canary.metrics, atol=atol, rtol=rtol):
        mismatches.append("canary-metrics")
    return mismatches


def replay_log(
    log: ReplayLog,
    coordinator: TransactionalCoordinator,
    *,
    mode: ReplayMode = ReplayMode.BITWISE,
    state_probe: StateProbe | None = None,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> ReplayReport:
    """Replay every batch and compare semantic receipts plus resulting state."""

    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("replay tolerances must be non-negative.")
    if mode is ReplayMode.TOLERANCE and state_probe is None:
        raise ValueError("tolerance replay requires a state_probe.")
    steps: list[ReplayStepReceipt] = []
    for index, entry in enumerate(log.entries):
        expected = entry.expected_receipt
        input_mismatches = [
            "proposal-payload-mutated:%s" % proposal.proposal_id
            for proposal in entry.batch.proposals
            if payload_fingerprint(proposal.payload) != proposal.payload_hash
        ]
        if input_mismatches:
            steps.append(ReplayStepReceipt(index, expected.commit_id, False, tuple(input_mismatches)))
            break

        actual = coordinator.commit(entry.batch, commit_id=expected.commit_id)
        mismatches = list(input_mismatches)
        semantic = (
            ("status", expected.status, actual.status),
            ("reason", expected.reason, actual.reason),
            ("versions-before", expected.versions_before, actual.versions_before),
            ("versions-after", expected.versions_after, actual.versions_after),
            ("invalidated-nodes", expected.invalidated_nodes, actual.invalidated_nodes),
            ("rollback-verified", expected.rollback_verified, actual.rollback_verified),
            ("proposal-ids", expected.proposal_ids, actual.proposal_ids),
        )
        mismatches.extend(name for name, left, right in semantic if left != right)
        mismatches.extend(
            _canary_mismatches(
                expected,
                actual,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
            )
        )

        if mode is ReplayMode.BITWISE:
            if expected.participant_fingerprints_after != actual.participant_fingerprints_after:
                mismatches.append("participant-state-fingerprint")
            if entry.expected_state is not None and state_probe is not None:
                if payload_fingerprint(entry.expected_state) != payload_fingerprint(state_probe()):
                    mismatches.append("probed-state-fingerprint")
        else:
            if entry.expected_state is None:
                mismatches.append("missing-expected-tolerance-state")
            elif not _numeric_close(
                entry.expected_state,
                state_probe(),
                atol=absolute_tolerance,
                rtol=relative_tolerance,
            ):
                mismatches.append("probed-state-outside-tolerance")
        steps.append(ReplayStepReceipt(index, expected.commit_id, not mismatches, tuple(mismatches), actual))

    return ReplayReport(mode, absolute_tolerance, relative_tolerance, tuple(steps))


__all__ = [
    "ReplayEntry",
    "ReplayLog",
    "ReplayMode",
    "ReplayReport",
    "ReplayStepReceipt",
    "StateProbe",
    "replay_log",
]
