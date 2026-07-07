"""``System`` -- the facade every subsystem eventually plugs into (workstream J1/J8).

Three verbs, nothing else: ``answer`` (serve a query), ``ingest`` (turn a model output into stored,
credence-weighted knowledge), ``improve`` (spend a budget making the system better). This is
deliberately a THIN SHELL: in this card ``answer`` just routes straight to the configured teacher and
wraps the reply in a minimal receipt; ``ingest`` writes what it can with whatever pieces already exist
(the belief store from workstream E when it is importable, a plain substrate item otherwise -- never a
hard import of a card that may not be built yet); ``improve`` honestly reports there is nothing to
improve until an orchestrator/router/registry registers into it. Later cards (REG-a the registry,
SPEND-a the budget ledger, FAULT-a degraded modes, SCORE-a the scorecard) extend these same three verbs
rather than adding new ones.

SPEND-a wires a real :class:`~mixle.spend.Spend` ledger into ``answer``: ``budget`` is a hard ceiling
measured in :meth:`~mixle.spend.Spend.total_units` -- a request that cannot afford even the cheapest
answer path is refused (with the shortfall named on the receipt), never silently served over budget.
Every successful call's incremental spend is added to :attr:`System.total_spend`, and both the
incremental and running totals ride on the receipt.

FAULT-a wires two named degraded modes (:mod:`mixle.fault`) into the same two verbs: ``answer`` falls
back to captured+store-only reasoning when the teacher raises (``teacher_down``); ``ingest`` falls back
to acknowledging-without-accumulating when the store write itself raises (``store_down``). Both flag
``degraded_mode``/``degraded_reason`` on the returned receipt/report rather than silently serving worse.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mixle.fault import with_fallback
from mixle.spend import Spend
from mixle.task.llm import LLM, OpenAICompatLLM


@dataclass
class SystemConfig:
    """Everything a :class:`System` needs to run. Secrets (endpoints, keys) come from the environment,
    never hardcoded -- see :meth:`from_env`."""

    teacher: LLM | Callable[..., str]
    registry_dir: str | None = None
    store: Any = None  # a mixle.substrate.Substrate handle, or None (ingest/retrieval degrade honestly)
    default_budget: int = 1
    scope: str = "local"

    @classmethod
    def from_env(cls, *, store: Any = None, registry_dir: str | None = None) -> SystemConfig:
        """Build a config whose teacher is an :class:`OpenAICompatLLM` sourced entirely from env vars.

        Reads ``MIXLE_TEACHER_BASE_URL`` (required), ``MIXLE_TEACHER_MODEL`` (required), and the
        optional ``MIXLE_TEACHER_API_KEY``. Raises ``ValueError`` naming the missing variable rather
        than silently constructing a broken teacher.
        """
        base_url = os.environ.get("MIXLE_TEACHER_BASE_URL")
        model = os.environ.get("MIXLE_TEACHER_MODEL")
        if not base_url:
            raise ValueError("SystemConfig.from_env needs MIXLE_TEACHER_BASE_URL set")
        if not model:
            raise ValueError("SystemConfig.from_env needs MIXLE_TEACHER_MODEL set")
        teacher = OpenAICompatLLM(base_url, model, api_key=os.environ.get("MIXLE_TEACHER_API_KEY"))
        return cls(teacher=teacher, registry_dir=registry_dir, store=store)


@dataclass
class Query:
    """The typed problem contract for :meth:`System.answer`.

    Field names align with the ``mixle-knowledge`` ``ContextPacket``/manifest contracts (``task``,
    ``expected_output`` <-> ``expected_output_schema``, ``scope``) so a ``Query`` can be built directly
    from one when a caller already holds a manifest.
    """

    text: str
    task: str = ""
    fingerprint: Any = None
    expected_output: dict[str, Any] | None = None
    scope: str = "local"


def _complete(teacher: LLM | Callable[..., str], prompt: str) -> str:
    if hasattr(teacher, "complete"):
        return teacher.complete(prompt)
    return teacher(prompt)


class System:
    """Constructed from a :class:`SystemConfig`; exposes ``answer``/``ingest``/``improve``."""

    def __init__(self, config: SystemConfig) -> None:
        self.config = config
        self.total_spend = Spend()

    def answer(self, query: Query, *, budget: int | None = None) -> tuple[str | None, dict[str, Any]]:
        """Thin shell: route straight to the teacher, wrap the reply in a minimal H-style receipt.

        ``budget`` is a hard ceiling (:class:`~mixle.spend.Spend.total_units`): if it cannot afford even
        one frontier call, the request is refused -- ``reply`` is ``None`` and the receipt names the exact
        ``shortfall`` -- rather than silently answering over budget. A served answer's cost is added to
        :attr:`total_spend`, which every receipt also carries as ``total_spend``.

        If the teacher call itself raises, this falls back to ``teacher_down`` degraded mode: answer from
        the store alone (a plain retrieval over ``config.store``) when one is configured and has anything
        relevant, flagging ``degraded_mode="teacher_down"`` on the receipt; if there is no store (or
        nothing relevant in it), the failure is reported honestly (``status="failed"``), never masked as a
        normal answer.
        """
        requested = self.config.default_budget if budget is None else int(budget)
        cost = Spend(frontier_calls=1)
        if requested < cost.total_units():
            return None, {
                "produced_by": None,
                "status": "refused",
                "reason": "budget insufficient for one frontier call",
                "budget": requested,
                "shortfall": cost.total_units() - requested,
                "spend": Spend().to_dict(),
                "total_spend": self.total_spend.to_dict(),
                "captured": False,
                "task": query.task,
            }

        def _call_teacher() -> str:
            return _complete(self.config.teacher, query.text)

        def _teacher_down_fallback(exc: Exception) -> str:
            if self.config.store is not None:
                from mixle.substrate.retrieve import retrieve

                hits = retrieve(self.config.store, query.text, k=3, scope=self.config.scope)
                texts = [it.text for it in hits.items if it.text]
                if texts:
                    return "[degraded: store-only] " + " ".join(texts)
            raise RuntimeError(f"teacher unavailable ({exc}) and no usable store to fall back on") from exc

        try:
            result = with_fallback(_call_teacher, _teacher_down_fallback, mode="teacher_down")
        except Exception as exc:
            return None, {
                "produced_by": None,
                "status": "failed",
                "reason": str(exc),
                "budget": requested,
                "spend": Spend().to_dict(),
                "total_spend": self.total_spend.to_dict(),
                "captured": False,
                "task": query.task,
            }
        actual_cost = Spend() if result.degraded else cost
        self.total_spend = self.total_spend + actual_cost
        receipt = {
            "produced_by": "store" if result.degraded else "teacher",
            "status": "answered",
            "spend": actual_cost.to_dict(),
            "total_spend": self.total_spend.to_dict(),
            "budget": requested,
            "captured": False,  # no local model has captured this capability yet (workstream D)
            "task": query.task,
            **result.to_receipt_fields(),
        }
        return result.value, receipt

    def ingest(self, model_output: str, *, source: dict[str, Any]) -> dict[str, Any]:
        """Turn a model output into stored knowledge. Uses the belief store (workstream E, KNOW-a) when
        it is importable; otherwise degrades to a plain substrate item rather than hard-depending on a
        card that may not be built yet.

        If the store write itself raises (the store is down/unreachable, as opposed to KNOW-a simply not
        being installed), this falls back to ``store_down`` degraded mode: the model output is
        acknowledged but NOT accumulated, and the report is flagged ``degraded_mode="store_down"`` rather
        than silently reporting success or losing the output without a trace.
        """
        if self.config.store is None:
            return {"status": "no_store", "assimilated": False}

        def _write() -> dict[str, Any]:
            try:
                from mixle.substrate.belief import assimilate, harvest_knowledge
            except ImportError:
                return self._ingest_fallback(model_output, source=source)
            claims = harvest_knowledge(model_output, source=source)
            items = [assimilate(self.config.store, claim, []) for claim in claims]
            return {"status": "ok", "n_claims": len(claims), "items": items}

        def _store_down_fallback(exc: Exception) -> dict[str, Any]:
            return {"status": "degraded_no_accumulation", "assimilated": False}

        result = with_fallback(_write, _store_down_fallback, mode="store_down")
        return {**result.value, **result.to_receipt_fields()} if result.degraded else result.value

    def _ingest_fallback(self, model_output: str, *, source: dict[str, Any]) -> dict[str, Any]:
        from mixle.substrate.core import SubstrateItem

        item = SubstrateItem(
            kind="text",
            text=model_output,
            provenance=dict(source),
            scope=self.config.scope,
            tags=["model_assertion", "unassimilated"],
        )
        self.config.store.put(item)
        return {"status": "ok_fallback", "assimilated": False, "item_id": item.id}

    def improve(self, budget: int) -> dict[str, Any]:
        """Stub until an orchestrator/router/registry registers into the system: nothing to improve yet."""
        return {
            "status": "nothing_to_improve",
            "reason": "no improvement subsystem registered yet",
            "budget": int(budget),
        }
