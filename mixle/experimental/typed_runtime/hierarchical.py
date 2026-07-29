"""Hierarchical island-proposal admission, merge, and transactional commit."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mixle.experimental.typed_runtime.proposal import (
    PayloadMerger,
    ProposalBatch,
    ProposalPacket,
    merge_same_node_proposals,
    payload_fingerprint,
)
from mixle.experimental.typed_runtime.staleness import (
    CorrectionResult,
    StalenessAction,
    StalenessPolicy,
    StalenessReceipt,
    assess_staleness,
    shrink_proposal,
)
from mixle.experimental.typed_runtime.transaction import CommitReceipt, TransactionalCoordinator

CorrectionProvider = Callable[[ProposalPacket, StalenessReceipt], CorrectionResult]


@dataclass(frozen=True)
class HierarchicalRoundReceipt:
    """Admission, merge, rejection, and commit result for one outer round."""

    round_id: str
    input_proposal_ids: tuple[str, ...]
    staleness: tuple[StalenessReceipt, ...]
    admitted_proposal_ids: tuple[str, ...]
    merged_proposals: dict[str, tuple[str, ...]]
    rejected: dict[str, str]
    commit: CommitReceipt | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible hierarchical receipt."""

        return {
            "round_id": self.round_id,
            "input_proposal_ids": list(self.input_proposal_ids),
            "staleness": [receipt.as_dict() for receipt in self.staleness],
            "admitted_proposal_ids": list(self.admitted_proposal_ids),
            "merged_proposals": {key: list(value) for key, value in self.merged_proposals.items()},
            "rejected": dict(self.rejected),
            "commit": self.commit.as_dict() if self.commit is not None else None,
        }


class HierarchicalProposalCoordinator:
    """Turn local-island proposals into one canonical outer transaction."""

    def __init__(
        self,
        coordinator: TransactionalCoordinator,
        *,
        default_staleness_policy: StalenessPolicy | None = None,
        node_staleness_policies: Mapping[str, StalenessPolicy] | None = None,
        payload_mergers: Mapping[str, PayloadMerger] | None = None,
        correction_provider: CorrectionProvider | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.default_staleness_policy = default_staleness_policy or StalenessPolicy()
        self.node_staleness_policies = dict(node_staleness_policies or {})
        self.payload_mergers = dict(payload_mergers or {})
        self.correction_provider = correction_provider
        self.receipts: list[HierarchicalRoundReceipt] = []

    def submit(
        self,
        round_id: str,
        proposals: Sequence[ProposalPacket],
    ) -> HierarchicalRoundReceipt:
        """Admit, transform, merge, and commit one set of island proposals."""

        if not round_id:
            raise ValueError("hierarchical round_id must be non-empty.")
        rows = tuple(proposals)
        if len({proposal.proposal_id for proposal in rows}) != len(rows):
            raise ValueError("hierarchical input proposal ids must be unique.")
        staleness_receipts: list[StalenessReceipt] = []
        rejected: dict[str, str] = {}
        admitted: list[ProposalPacket] = []
        for proposal in rows:
            try:
                contract = self.coordinator.graph.node(proposal.node_id).contract
            except KeyError:
                rejected[proposal.proposal_id] = "unknown-node"
                continue
            policy = self.node_staleness_policies.get(proposal.node_id, self.default_staleness_policy)
            correction = None
            receipt = assess_staleness(proposal, contract, self.coordinator.versions, policy)
            if receipt.reason == "missing-drift-correction":
                if self.correction_provider is None:
                    staleness_receipts.append(receipt)
                    rejected[proposal.proposal_id] = "missing-correction-provider"
                    continue
                try:
                    correction = self.correction_provider(proposal, receipt)
                    if not isinstance(correction, CorrectionResult):
                        raise TypeError("correction provider must return CorrectionResult.")
                    receipt = assess_staleness(
                        proposal,
                        contract,
                        self.coordinator.versions,
                        policy,
                        correction=correction,
                    )
                except Exception as error:  # noqa: BLE001 - isolate one untrusted provider
                    staleness_receipts.append(receipt)
                    rejected[proposal.proposal_id] = "correction-failed:%s:%s" % (
                        type(error).__name__,
                        error,
                    )
                    continue
            staleness_receipts.append(receipt)
            if not receipt.accepted:
                rejected[proposal.proposal_id] = receipt.reason
                continue
            if receipt.action in (StalenessAction.SHRINK, StalenessAction.CORRECT):
                try:
                    proposal = shrink_proposal(
                        proposal,
                        receipt,
                        proposal_id="rebased:%s:v%d" % (proposal.proposal_id, self.coordinator.versions.model_version),
                        correction=correction,
                    )
                except (TypeError, ValueError) as error:
                    rejected[proposal.proposal_id] = "stale-transform-failed:%s" % error
                    continue
            admitted.append(proposal)

        grouped: dict[str, list[ProposalPacket]] = {}
        for proposal in admitted:
            grouped.setdefault(proposal.node_id, []).append(proposal)
        canonical: list[ProposalPacket] = []
        merged: dict[str, tuple[str, ...]] = {}
        for node_id, node_rows in sorted(grouped.items()):
            if len(node_rows) == 1:
                canonical.append(node_rows[0])
                continue
            contract = self.coordinator.graph.node(node_id).contract
            merged_id = "merged:%s" % payload_fingerprint(tuple(sorted(row.proposal_id for row in node_rows)))[:16]
            try:
                merged_proposal = merge_same_node_proposals(
                    node_rows,
                    merged_proposal_id=merged_id,
                    merge_law=contract.merge_law,
                    payload_merger=self.payload_mergers.get(node_id),
                )
            except (TypeError, ValueError) as error:
                for row in node_rows:
                    rejected[row.proposal_id] = "merge-failed:%s" % error
                continue
            canonical.append(merged_proposal)
            merged[merged_id] = tuple(sorted(row.proposal_id for row in node_rows))

        commit = None
        if canonical:
            batch_id = "hierarchical:%s" % round_id
            commit = self.coordinator.commit(ProposalBatch(batch_id, tuple(canonical)))
        receipt = HierarchicalRoundReceipt(
            round_id,
            tuple(proposal.proposal_id for proposal in rows),
            tuple(staleness_receipts),
            tuple(proposal.proposal_id for proposal in canonical),
            merged,
            rejected,
            commit,
        )
        self.receipts.append(receipt)
        return receipt


__all__ = ["CorrectionProvider", "HierarchicalProposalCoordinator", "HierarchicalRoundReceipt"]
