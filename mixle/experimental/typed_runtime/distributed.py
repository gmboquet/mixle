"""Lower statistically typed update contracts into communication declarations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from mixle.experimental.typed_runtime.contracts import MergeLaw, UpdateKind
from mixle.experimental.typed_runtime.graph import UpdateGraph
from mixle.utils.parallel.training_contracts import (
    CollectiveKind,
    DistributedUpdate,
    ParallelAxis,
    ParallelPlan,
    PayloadKind,
    StateLayout,
)

_GRADIENT_UPDATES = {
    UpdateKind.FIRST_ORDER,
    UpdateKind.PRECONDITIONED,
    UpdateKind.PROXIMAL,
}
_STATISTIC_UPDATES = {
    UpdateKind.EXACT_CLOSED_FORM,
    UpdateKind.GENERALIZED_EM,
    UpdateKind.COORDINATE,
    UpdateKind.MONTE_CARLO,
}


@dataclass(frozen=True)
class CollectiveNumericsEvidence:
    """Observed backend determinism and error for one exact collective identity."""

    evidence_id: str
    update_id: str
    collective: CollectiveKind
    mesh_axes: tuple[ParallelAxis, ...]
    backend: str
    dtype: str
    deterministic: bool
    maximum_absolute_error: float
    maximum_relative_error: float
    sample_count: int
    ordering_fingerprint: str

    def __post_init__(self) -> None:
        strings = (
            self.evidence_id,
            self.update_id,
            self.backend,
            self.dtype,
            self.ordering_fingerprint,
        )
        if any(not isinstance(value, str) or not value for value in strings):
            raise ValueError("collective numerical evidence identities must be non-empty.")
        errors = (self.maximum_absolute_error, self.maximum_relative_error)
        if any(not math.isfinite(value) or value < 0.0 for value in errors):
            raise ValueError("collective numerical error bounds must be finite and non-negative.")
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count < 1:
            raise ValueError("collective numerical evidence requires a positive integer sample_count.")
        if len(set(self.mesh_axes)) != len(self.mesh_axes):
            raise ValueError("collective numerical evidence mesh axes must be unique.")


def _data_collective(plan: ParallelPlan, *, statistics: bool) -> tuple[CollectiveKind, tuple[ParallelAxis, ...]]:
    axes = tuple(axis for axis in (ParallelAxis.DP_REPLICATE, ParallelAxis.DP_SHARD) if plan.size(axis) > 1)
    if not axes:
        return CollectiveKind.NONE, ()
    if statistics:
        return CollectiveKind.ALL_REDUCE, axes
    if plan.dp_shard > 1:
        return CollectiveKind.REDUCE_SCATTER, axes
    return CollectiveKind.ALL_REDUCE, axes


def _lower_update(
    *,
    update_id: str,
    payload: PayloadKind,
    collective: CollectiveKind,
    axes: tuple[ParallelAxis, ...],
    state_layout: StateLayout,
    contract_exact: bool,
    note: str,
    evidence: Mapping[str, CollectiveNumericsEvidence],
    used_evidence: set[str],
) -> DistributedUpdate:
    observed = evidence.get(update_id)
    if observed is not None:
        if observed.update_id != update_id:
            raise ValueError("collective evidence key does not match its update_id.")
        if observed.collective is not collective or observed.mesh_axes != axes:
            raise ValueError("collective evidence does not match the planned collective and mesh axes.")
        used_evidence.add(update_id)
    guaranteed_exact = contract_exact and collective is CollectiveKind.NONE
    notes = [note]
    if contract_exact and not guaranteed_exact:
        notes.append("model contract is exact; distributed numerical exactness is not guaranteed")
    return DistributedUpdate(
        node_id=update_id,
        payload=payload,
        collective=collective,
        mesh_axes=axes,
        state_layout=state_layout,
        exact=guaranteed_exact,
        notes=tuple(notes),
        contract_exact=contract_exact,
        determinism_observed=observed.deterministic if observed is not None else None,
        maximum_absolute_error=observed.maximum_absolute_error if observed is not None else None,
        maximum_relative_error=observed.maximum_relative_error if observed is not None else None,
        numerics_evidence_id=observed.evidence_id if observed is not None else None,
        numerics_sample_count=observed.sample_count if observed is not None else 0,
    )


def plan_distributed_updates(
    graph: UpdateGraph,
    plan: ParallelPlan,
    *,
    numerics_evidence: Mapping[str, CollectiveNumericsEvidence] | None = None,
) -> tuple[DistributedUpdate, ...]:
    """Produce an auditable collective plan for every compiled update node.

    The result deliberately distinguishes additive sufficient statistics from
    gradients.  It also emits model-axis transfers separately, so a scheduler
    cannot mistake an EP token exchange or PP activation send for gradient
    synchronization.
    """

    evidence = dict(numerics_evidence or {})
    used_evidence: set[str] = set()
    updates: list[DistributedUpdate] = []
    for node_id in graph.topological_order():
        contract = graph.node(node_id).contract
        kind = contract.update_kind
        if kind is UpdateKind.FROZEN:
            updates.append(
                _lower_update(
                    update_id=node_id,
                    payload=PayloadKind.PARAMETER,
                    collective=CollectiveKind.NONE,
                    axes=(),
                    state_layout=StateLayout.REPLICATED,
                    contract_exact=contract.exact,
                    note="frozen state has no distributed update",
                    evidence=evidence,
                    used_evidence=used_evidence,
                )
            )
            continue

        statistics = kind in _STATISTIC_UPDATES and contract.merge_law not in {
            MergeLaw.NON_MERGEABLE,
            MergeLaw.REPLICATED,
        }
        if statistics:
            collective, axes = _data_collective(plan, statistics=True)
            payload = PayloadKind.SUFFICIENT_STATISTIC
        elif kind in _GRADIENT_UPDATES:
            collective, axes = _data_collective(plan, statistics=False)
            payload = PayloadKind.GRADIENT
        elif kind is UpdateKind.MESSAGE_PASSING:
            collective, axes = CollectiveKind.CUSTOM, tuple(plan.active_axes)
            payload = PayloadKind.MESSAGE
        else:
            collective, axes = (
                CollectiveKind.BROADCAST,
                tuple(axis for axis in plan.active_axes if axis in {ParallelAxis.DP_REPLICATE, ParallelAxis.DP_SHARD}),
            )
            payload = PayloadKind.PARAMETER

        state_layout = StateLayout.SHARDED if plan.dp_shard > 1 else StateLayout.REPLICATED
        updates.append(
            _lower_update(
                update_id=node_id,
                payload=payload,
                collective=collective,
                axes=axes,
                state_layout=state_layout,
                contract_exact=contract.exact,
                note="derived from %s/%s" % (kind.value, contract.merge_law.value),
                evidence=evidence,
                used_evidence=used_evidence,
            )
        )

        model_axis_updates = (
            (ParallelAxis.TP, PayloadKind.ACTIVATION, CollectiveKind.ALL_REDUCE, "tensor-parallel sublayer"),
            (ParallelAxis.PP, PayloadKind.ACTIVATION, CollectiveKind.P2P, "pipeline stage boundary"),
            (ParallelAxis.CP, PayloadKind.KV_BLOCK, CollectiveKind.ALL_GATHER, "context-parallel attention"),
            (ParallelAxis.EP, PayloadKind.TOKEN, CollectiveKind.ALL_TO_ALL, "expert dispatch/combine"),
            (ParallelAxis.ETP, PayloadKind.ACTIVATION, CollectiveKind.ALL_REDUCE, "expert tensor parallel"),
        )
        for axis, axis_payload, axis_collective, note in model_axis_updates:
            if plan.size(axis) > 1 and payload in {PayloadKind.GRADIENT, PayloadKind.PARAMETER}:
                update_id = "%s:%s" % (node_id, axis.value)
                updates.append(
                    _lower_update(
                        update_id=update_id,
                        payload=axis_payload,
                        collective=axis_collective,
                        axes=(axis,),
                        state_layout=(
                            StateLayout.EXPERT_LOCAL
                            if axis in {ParallelAxis.EP, ParallelAxis.ETP}
                            else StateLayout.PIPELINE_LOCAL
                            if axis is ParallelAxis.PP
                            else StateLayout.SHARDED
                        ),
                        contract_exact=contract.exact,
                        note=note,
                        evidence=evidence,
                        used_evidence=used_evidence,
                    )
                )
    unused = sorted(set(evidence) - used_evidence)
    if unused:
        raise ValueError("collective numerical evidence did not match planned updates: %s" % ", ".join(unused))
    return tuple(updates)


__all__ = ["CollectiveNumericsEvidence", "plan_distributed_updates"]
