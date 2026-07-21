"""Lower statistically typed update contracts into communication declarations."""

from __future__ import annotations

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


def _data_collective(plan: ParallelPlan, *, statistics: bool) -> tuple[CollectiveKind, tuple[ParallelAxis, ...]]:
    axes = tuple(axis for axis in (ParallelAxis.DP_REPLICATE, ParallelAxis.DP_SHARD) if plan.size(axis) > 1)
    if not axes:
        return CollectiveKind.NONE, ()
    if statistics:
        return CollectiveKind.ALL_REDUCE, axes
    if plan.dp_shard > 1:
        return CollectiveKind.REDUCE_SCATTER, axes
    return CollectiveKind.ALL_REDUCE, axes


def plan_distributed_updates(graph: UpdateGraph, plan: ParallelPlan) -> tuple[DistributedUpdate, ...]:
    """Produce an auditable collective plan for every compiled update node.

    The result deliberately distinguishes additive sufficient statistics from
    gradients.  It also emits model-axis transfers separately, so a scheduler
    cannot mistake an EP token exchange or PP activation send for gradient
    synchronization.
    """

    updates: list[DistributedUpdate] = []
    for node_id in graph.topological_order():
        contract = graph.node(node_id).contract
        kind = contract.update_kind
        if kind is UpdateKind.FROZEN:
            updates.append(
                DistributedUpdate(
                    node_id=node_id,
                    payload=PayloadKind.PARAMETER,
                    collective=CollectiveKind.NONE,
                    mesh_axes=(),
                    state_layout=StateLayout.REPLICATED,
                    exact=True,
                    notes=("frozen state has no distributed update",),
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
            DistributedUpdate(
                node_id=node_id,
                payload=payload,
                collective=collective,
                mesh_axes=axes,
                state_layout=state_layout,
                exact=contract.exact,
                notes=("derived from %s/%s" % (kind.value, contract.merge_law.value),),
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
                updates.append(
                    DistributedUpdate(
                        node_id="%s:%s" % (node_id, axis.value),
                        payload=axis_payload,
                        collective=axis_collective,
                        mesh_axes=(axis,),
                        state_layout=(
                            StateLayout.EXPERT_LOCAL
                            if axis in {ParallelAxis.EP, ParallelAxis.ETP}
                            else StateLayout.PIPELINE_LOCAL
                            if axis is ParallelAxis.PP
                            else StateLayout.SHARDED
                        ),
                        exact=contract.exact,
                        notes=(note,),
                    )
                )
    return tuple(updates)


__all__ = ["plan_distributed_updates"]
