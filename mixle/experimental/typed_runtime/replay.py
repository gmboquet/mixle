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


class ReplayStatus(StrEnum):
    """Whole-log replay outcome; an empty log is explicitly not run."""

    MATCHED = "matched"
    MISMATCHED = "mismatched"
    NOT_RUN = "not_run"


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

        if not isinstance(batch, ProposalBatch) or not isinstance(receipt, CommitReceipt):
            raise TypeError("replay records require a ProposalBatch and CommitReceipt.")
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

    def __post_init__(self) -> None:
        """Bind the verdict to the list that is supposed to justify it (MXR-080-1874).

        ``matched`` and ``mismatches`` are the same fact stated twice, and nothing held them
        together: ``ReplayStepReceipt(0, "commit-0", True, ("status", "reason"))`` constructed and
        reported a matched replay while enumerating what did not match. ``ReplayReport.status`` reads
        ``matched`` alone, so that receipt made a whole-log replay report MATCHED over its own
        evidence. ``replay_log`` passes ``not mismatches``, so this refuses nothing it produces.
        """
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError(f"replay step index must be a non-negative integer, got {self.index!r}.")
        if not isinstance(self.commit_id, str) or not self.commit_id:
            raise ValueError("replay step commit_id must be a non-empty string.")
        if not isinstance(self.matched, bool):
            raise TypeError(f"replay step matched must be a Boolean verdict, got {type(self.matched).__name__}.")
        if isinstance(self.mismatches, (str, bytes)) or not isinstance(self.mismatches, (tuple, list)):
            raise TypeError(
                f"replay step mismatches must be a sequence of names, got {type(self.mismatches).__name__}."
            )
        object.__setattr__(self, "mismatches", tuple(self.mismatches))
        if any(not isinstance(name, str) or not name.strip() for name in self.mismatches):
            raise ValueError(f"replay step mismatches must each name a comparison, got {list(self.mismatches)!r}.")
        if self.matched and self.mismatches:
            raise ValueError(
                f"replay step {self.commit_id} reports matched=True while listing mismatches "
                f"{list(self.mismatches)}; a replay matched exactly when nothing differed."
            )
        if self.actual_receipt is not None and not isinstance(self.actual_receipt, CommitReceipt):
            raise TypeError(
                f"replay step actual_receipt must be a CommitReceipt, got {type(self.actual_receipt).__name__}."
            )

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

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ReplayMode):
            raise TypeError("replay report mode must be ReplayMode.")
        if (
            not math.isfinite(self.absolute_tolerance)
            or not math.isfinite(self.relative_tolerance)
            or self.absolute_tolerance < 0.0
            or self.relative_tolerance < 0.0
        ):
            raise ValueError("replay-report tolerances must be finite and non-negative.")
        if not isinstance(self.steps, tuple) or any(not isinstance(step, ReplayStepReceipt) for step in self.steps):
            raise TypeError("replay report steps must be a tuple of ReplayStepReceipt values.")

    @property
    def status(self) -> ReplayStatus:
        """Distinguish successful evidence from mismatch and absent evidence."""

        if not self.steps:
            return ReplayStatus.NOT_RUN
        if all(step.matched for step in self.steps):
            return ReplayStatus.MATCHED
        return ReplayStatus.MISMATCHED

    @property
    def ran(self) -> bool:
        """Whether at least one replay step was attempted."""

        return self.status is not ReplayStatus.NOT_RUN

    @property
    def matched(self) -> bool:
        """Whether every recorded transaction replayed under the selected guarantee."""

        return self.status is ReplayStatus.MATCHED

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible replay report."""

        return {
            "mode": self.mode.value,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "status": self.status.value,
            "ran": self.ran,
            "matched": self.matched,
            "steps": [step.as_dict() for step in self.steps],
        }


StateProbe = Callable[[], Any]


def _valid_fingerprints(values: Any) -> bool:
    # Mapping, not dict: CommitReceipt's fingerprint maps are read-only views now that a receipt
    # cannot be rewritten after the fact (MXR-080-1865), and MappingProxyType is not a dict subclass.
    # Testing the concrete type made every replay report missing fingerprints it was actually given.
    return (
        isinstance(values, Mapping)
        and bool(values)
        and all(isinstance(name, str) and name and isinstance(value, str) and value for name, value in values.items())
    )


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

    if not isinstance(log, ReplayLog) or not isinstance(coordinator, TransactionalCoordinator):
        raise TypeError("replay_log requires a ReplayLog and TransactionalCoordinator.")
    if not isinstance(mode, ReplayMode):
        raise TypeError("replay mode must be ReplayMode.")
    if (
        not math.isfinite(absolute_tolerance)
        or not math.isfinite(relative_tolerance)
        or absolute_tolerance < 0.0
        or relative_tolerance < 0.0
    ):
        raise ValueError("replay tolerances must be finite and non-negative.")
    if mode is ReplayMode.TOLERANCE and state_probe is None:
        raise ValueError("tolerance replay requires a state_probe.")
    steps: list[ReplayStepReceipt] = []
    for index, entry in enumerate(log.entries):
        if not isinstance(entry, ReplayEntry):
            raise TypeError("replay log entries must be ReplayEntry values.")
        if entry.expected_receipt.batch_id != entry.batch.batch_id:
            raise ValueError("replay entry receipt does not belong to its proposal batch.")
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
            if not _valid_fingerprints(expected.participant_fingerprints_before) or not _valid_fingerprints(
                expected.participant_fingerprints_after
            ):
                mismatches.append("missing-expected-participant-fingerprints")
            if not _valid_fingerprints(actual.participant_fingerprints_before) or not _valid_fingerprints(
                actual.participant_fingerprints_after
            ):
                mismatches.append("missing-actual-participant-fingerprints")
            if expected.participant_fingerprints_before != actual.participant_fingerprints_before:
                mismatches.append("participant-state-fingerprint-before")
            if expected.participant_fingerprints_after != actual.participant_fingerprints_after:
                mismatches.append("participant-state-fingerprint")
            if entry.expected_state is not None:
                if state_probe is None:
                    mismatches.append("missing-state-probe")
                elif payload_fingerprint(entry.expected_state) != payload_fingerprint(state_probe()):
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
    "ReplayStatus",
    "ReplayStepReceipt",
    "StateProbe",
    "replay_log",
]
