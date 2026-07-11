"""Facade for answering, ingesting knowledge, and improving a Mixle system.

The facade exposes three verbs: ``answer`` serves a query, ``ingest`` stores a
model output as credence-weighted knowledge, and ``improve`` spends a budget on
measured improvement. The shell is deliberately thin: ``answer`` routes to the
configured teacher and attaches a receipt, ``ingest`` writes through the
available store boundary, and ``improve`` promotes harvested answers into an
explicit captured cache.

The :class:`~mixle.spend.Spend` ledger treats ``budget`` as a hard ceiling
measured in :meth:`~mixle.spend.Spend.total_units`. A request that cannot afford
the minimum-cost answer path is refused with the shortfall named on the receipt.
Successful calls add incremental spend to :attr:`System.total_spend`, and
receipts carry both incremental and running totals.

Named degraded modes from :mod:`mixle.fault` use the same verbs. ``answer`` can
fall back to captured or store-only reasoning when the teacher raises
(``teacher_down``), and ``ingest`` can acknowledge without accumulating when a
store write raises (``store_down``). Both paths flag ``degraded_mode`` and
``degraded_reason`` on the returned receipt or report.

The cold-start loop harvests teacher-produced answers. ``improve`` promotes the
harvest into a verbatim captured cache; ``answer`` checks that cache before
spending. Capture only promotes after an explicit ``improve()`` call, so
measured savings are attributable to improvement rather than implicit caching.
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
    """Configuration required to run a :class:`System`.

    Secrets such as endpoints and keys are read from the environment by
    :meth:`from_env`; they are not hardcoded in the config object.
    """

    teacher: LLM | Callable[..., str]
    registry_dir: str | None = None
    store: Any = None  # a mixle.substrate.Substrate handle, or None (ingest/retrieval return degraded receipts)
    default_budget: int = 1
    scope: str = "local"

    @classmethod
    def from_env(cls, *, store: Any = None, registry_dir: str | None = None) -> SystemConfig:
        """Build a config whose teacher is an :class:`OpenAICompatLLM` sourced entirely from env vars.

        Reads ``MIXLE_TEACHER_BASE_URL`` (required), ``MIXLE_TEACHER_MODEL`` (required), and the
        optional ``MIXLE_TEACHER_API_KEY``. Raises ``ValueError`` naming the missing variable rather
        than silently constructing an unusable teacher.
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

    ``task`` and ``expected_output`` align with the ``mixle-knowledge``
    ``ContextPacket`` contract's ``task`` and ``expected_output_schema`` fields
    (see :meth:`from_knowledge_dict`). ``scope`` is a ``Query``-level routing
    boundary and is not inferred from the packet.
    """

    text: str
    task: str = ""
    fingerprint: Any = None
    expected_output: dict[str, Any] | None = None
    scope: str = "local"

    @classmethod
    def from_knowledge_dict(cls, packet: dict[str, Any], *, scope: str = "local") -> Query:
        """Build a ``Query`` from a mixle-knowledge-shaped ``ContextPacket`` dict.

        ``text`` comes from ``payload["rendered"]``. ``task`` and
        ``expected_output`` map from the packet's ``task`` and
        ``expected_output_schema``. ``scope`` is supplied by the caller.
        """
        payload = packet.get("payload") or {}
        return cls(
            text=str(payload.get("rendered", "")),
            task=str(packet.get("task", "")),
            expected_output=packet.get("expected_output_schema") or None,
            scope=scope,
        )


def _complete(teacher: LLM | Callable[..., str], prompt: str) -> str:
    if hasattr(teacher, "complete"):
        return teacher.complete(prompt)
    return teacher(prompt)


class System:
    """Constructed from a :class:`SystemConfig`; exposes ``answer``/``ingest``/``improve``."""

    def __init__(self, config: SystemConfig) -> None:
        self.config = config
        self.total_spend = Spend()
        self._harvest: dict[tuple[str, str, str], str] = {}
        self._captured: dict[tuple[str, str, str], str] = {}

    def answer(self, query: Query, *, budget: int | None = None) -> tuple[str | None, dict[str, Any]]:
        """Thin shell: route straight to the teacher, wrap the reply in a minimal H-style receipt.

        Checks the captured cache first (see :meth:`improve`): an exact repeat of a query (same text,
        task, AND scope -- two queries that merely share text but differ in task/scope are different
        questions and must not share a cache entry) already promoted by a prior ``improve()`` call is
        served free, no budget spent, ``captured=True``.

        ``budget`` is a hard ceiling (:class:`~mixle.spend.Spend.total_units`): if it cannot afford even
        one frontier call, the request is refused -- ``reply`` is ``None`` and the receipt names the exact
        ``shortfall`` -- rather than silently answering over budget. A served answer's cost is added to
        :attr:`total_spend`, which every receipt also carries as ``total_spend``.

        If the teacher call itself raises, this falls back to ``teacher_down`` degraded mode: answer from
        the store alone (a plain retrieval over ``config.store``) when one is configured and has anything
        relevant, flagging ``degraded_mode="teacher_down"`` on the receipt; if there is no store (or
        nothing relevant in it), the failure is reported explicitly (``status="failed"``), never masked as a
        normal answer.
        """
        cache_key = (query.text, query.task, query.scope)
        if cache_key in self._captured:
            return self._captured[cache_key], {
                "produced_by": "captured",
                "status": "answered",
                "spend": Spend().to_dict(),
                "total_spend": self.total_spend.to_dict(),
                "budget": self.config.default_budget if budget is None else int(budget),
                "captured": True,
                "task": query.task,
                "degraded_mode": None,
                "degraded_reason": None,
            }

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
        except Exception as exc:  # noqa: BLE001
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
        if not result.degraded:
            self._harvest[cache_key] = result.value
        receipt = {
            "produced_by": "store" if result.degraded else "teacher",
            "status": "answered",
            "spend": actual_cost.to_dict(),
            "total_spend": self.total_spend.to_dict(),
            "budget": requested,
            "captured": False,  # no local model has captured this capability yet
            "task": query.task,
            **result.to_receipt_fields(),
        }
        return result.value, receipt

    def ingest(self, model_output: str, *, source: dict[str, Any]) -> dict[str, Any]:
        """Turn a model output into stored knowledge.

        Uses the belief store when it is importable; otherwise records a plain
        substrate item rather than requiring optional knowledge-substrate
        components.

        If the store write raises, this falls back to ``store_down`` degraded
        mode. The model output is acknowledged but not accumulated, and the
        report is flagged with ``degraded_mode="store_down"``.
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
        """Promote every harvested (query, reply) pair from :meth:`answer` into the captured cache.

        Reports that there is nothing to improve when nothing has been
        harvested yet. Otherwise this is the cold-start capture step: after
        this call, a repeat of a captured query is answered from the local cache
        (see :meth:`answer`).
        """
        if not self._harvest:
            return {
                "status": "nothing_to_improve",
                "reason": "no improvement subsystem registered yet",
                "budget": int(budget),
            }
        n_captured = len(self._harvest)
        self._captured.update(self._harvest)
        self._harvest.clear()
        return {
            "status": "captured",
            "reason": f"promoted {n_captured} harvested (query, reply) pair(s) into the captured cache",
            "budget": int(budget),
            "n_captured": n_captured,
        }
