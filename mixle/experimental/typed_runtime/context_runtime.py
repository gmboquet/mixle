"""Closed-loop effective-context construction from actions through bounded materialization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mixle.experimental.typed_runtime.context_execution import ContextActionExecutor
from mixle.experimental.typed_runtime.context_ir import (
    ContextAction,
    ContextActionKind,
    ContextActionReceipt,
    ContextGraph,
)
from mixle.experimental.typed_runtime.context_materializer import (
    MaterializationPolicy,
    MaterializedContext,
    materialize_context,
)
from mixle.experimental.typed_runtime.context_scheduler import (
    ContextScheduleDecision,
    ValueOfInformationScheduler,
)

ContextActionProvider = Callable[
    [ContextGraph, tuple[ContextActionReceipt, ...]],
    tuple[ContextAction, ...],
]


@dataclass(frozen=True)
class EffectiveContextRun:
    """Every scheduler decision and actual action receipt through stopping."""

    decisions: tuple[ContextScheduleDecision, ...]
    action_receipts: tuple[ContextActionReceipt, ...]
    stopping_reason: str
    final_graph_version: int

    @property
    def completed_actions(self) -> tuple[ContextActionReceipt, ...]:
        """Receipts excluding the zero-cost STOP marker."""

        return tuple(row for row in self.action_receipts if row.action.kind is not ContextActionKind.STOP)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible closed-loop receipt."""

        return {
            "decisions": [decision.as_dict() for decision in self.decisions],
            "action_receipts": [receipt.as_dict() for receipt in self.action_receipts],
            "stopping_reason": self.stopping_reason,
            "final_graph_version": self.final_graph_version,
        }


class EffectiveContextRuntime:
    """Run value-of-information context actions until an explicit stopping rule."""

    def __init__(
        self,
        graph: ContextGraph,
        scheduler: ValueOfInformationScheduler,
        executor: ContextActionExecutor,
        action_provider: ContextActionProvider,
    ) -> None:
        if executor.graph is not graph:
            raise ValueError("context runtime graph and executor graph must be the same object.")
        self.graph = graph
        self.scheduler = scheduler
        self.executor = executor
        self.action_provider = action_provider
        self.last_run: EffectiveContextRun | None = None

    def run(self, *, maximum_iterations: int = 100) -> EffectiveContextRun:
        """Construct/revisit context until VOI stops or the hard loop guard fires."""

        if maximum_iterations < 1:
            raise ValueError("maximum_iterations must be positive.")
        decisions = []
        receipts: list[ContextActionReceipt] = []
        stopping_reason = "maximum-iterations"
        for _ in range(maximum_iterations):
            actions = self.action_provider(self.graph, tuple(receipts))
            decision = self.scheduler.choose(tuple(actions), self.graph)
            decisions.append(decision)
            receipt = self.executor.execute(decision.selected)
            receipts.append(receipt)
            if decision.stopped:
                stopping_reason = decision.stopping_reason or "scheduler-stop"
                break
            self.scheduler.record(receipt)
        else:
            stop = ContextAction(
                "stop:max-iterations:v%d" % self.graph.version,
                ContextActionKind.STOP,
            )
            receipts.append(self.executor.execute(stop))

        result = EffectiveContextRun(tuple(decisions), tuple(receipts), stopping_reason, self.graph.version)
        self.last_run = result
        return result

    def materialize(
        self,
        relevance: dict[str, float],
        policy: MaterializationPolicy,
        *,
        source_horizon_tokens: int | None = None,
        required_node_ids: tuple[str, ...] = (),
    ) -> MaterializedContext:
        """Materialize the last run with its measured action costs."""

        if self.last_run is None:
            raise RuntimeError("run context construction before materialization.")
        rows = self.last_run.completed_actions
        return materialize_context(
            self.graph,
            relevance,
            policy,
            source_horizon_tokens=source_horizon_tokens,
            required_node_ids=required_node_ids,
            context_actions=len(rows),
            retrieval_actions=sum(
                row.action.kind in (ContextActionKind.RETRIEVE, ContextActionKind.EXPAND_SOURCE) for row in rows
            ),
            generation_actions=sum(
                row.action.kind
                in (
                    ContextActionKind.GENERATE_HYPOTHESIS,
                    ContextActionKind.GENERATE_QUERY,
                    ContextActionKind.SUMMARIZE,
                )
                for row in rows
            ),
            verification_actions=sum(row.action.kind is ContextActionKind.VERIFY for row in rows),
            tool_calls=sum(row.tool_calls for row in rows),
            latency_seconds=sum(row.latency_seconds for row in rows),
            monetary_cost=sum(row.monetary_cost for row in rows),
            stopped_reason=self.last_run.stopping_reason,
        )


__all__ = ["ContextActionProvider", "EffectiveContextRun", "EffectiveContextRuntime"]
