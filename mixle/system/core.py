"""Facade for answering, ingesting knowledge, and improving a Mixle system.

Part of :mod:`mixle.system` (formerly the flat top-level ``mixle.system`` module; now
``mixle.system.core``, alongside its former siblings ``mixle.spend``, ``mixle.fault``,
``mixle.scorecard``, ``mixle.meta``, and ``mixle.registry`` -- one cohesive local-application-facade
package instead of six cross-importing top-level files). Import from :mod:`mixle.system` directly
(``from mixle.system import System``); the old flat names still work via a deprecation shim.

The facade exposes three verbs: ``answer`` serves a query, ``ingest`` stores a
model output as credence-weighted knowledge, and ``improve`` spends a budget on
measured improvement. The shell is deliberately thin: ``answer`` routes to the
configured teacher and attaches a receipt, ``ingest`` writes through the
available store boundary, and ``improve`` promotes harvested answers into an
explicit captured cache.

The :class:`~mixle.system.spend.Spend` ledger treats ``budget`` as a hard ceiling
measured in :meth:`~mixle.system.spend.Spend.total_units`. A request that cannot afford
the minimum-cost answer path is refused with the shortfall named on the receipt.
Successful calls add incremental spend to :attr:`System.total_spend`, and
receipts carry both incremental and running totals.

Named degraded modes from :mod:`mixle.system.fault` use the same verbs. ``answer`` can
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

from mixle.system.fault import with_fallback
from mixle.system.spend import Spend, _count
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
    """Call the teacher and hold it to the string answer contract it promises.

    The provider boundary is untrusted: a teacher returning ``None`` or a dict used to be charged as a
    successful frontier call, reported ``status="answered"``, harvested, promoted by :meth:`System.improve`
    and then served from the captured cache forever. In the ``None`` case the public return was
    indistinguishable from a refusal or a failure unless every caller also read the receipt. An answer
    that does not satisfy the contract is a failed call, and ``ValueError`` (a failing dependency,
    not a defect in this process) routes it into exactly the same ``teacher_down`` store-fallback /
    ``status="failed"`` path a teacher that raises already takes.
    """
    reply = teacher.complete(prompt) if hasattr(teacher, "complete") else teacher(prompt)
    if not isinstance(reply, str):
        raise ValueError(f"teacher must return a string answer, got {type(reply).__name__}")
    if not reply.strip():
        raise ValueError("teacher returned an empty answer")
    return reply


# System.improve()'s cost model: one flat unit per (query, reply) pair promoted from the harvest into
# the captured cache. Promoting a pair is a pure cache write, not an external call, so this is
# deliberately a separate, simpler ledger than Spend (which counts real frontier/oracle calls); replace
# this constant with a real per-item cost function if promotion ever needs a richer model.
_PROMOTE_COST_PER_ITEM = 1


class System:
    """Constructed from a :class:`SystemConfig`; exposes ``answer``/``ingest``/``improve``."""

    def __init__(self, config: SystemConfig) -> None:
        self.config = config
        self.total_spend = Spend()
        self._harvest: dict[tuple[str, str, str], str] = {}
        self._captured: dict[tuple[str, str, str], str] = {}

    def answer(
        self, query: Query, *, budget: int | None = None, read_only: bool = False
    ) -> tuple[str | None, dict[str, Any]]:
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
        the store alone (a plain retrieval over ``config.store``, scoped to the QUERY's own ``scope`` --
        never ``config.scope``) when one is configured and has anything relevant, flagging
        ``degraded_mode="teacher_down"`` on the receipt; if there is no store (or nothing relevant in
        it), the failure is reported explicitly (``status="failed"``), never masked as a normal answer.

        ``read_only=True`` runs the identical routing/answering logic but takes a genuine snapshot: the
        call is guaranteed not to promote into :attr:`_harvest` (so a later :meth:`improve` can never
        capture it) and not to accumulate into :attr:`total_spend` -- no trace of the call is left for
        the system's future behavior to depend on. This is what
        :func:`~mixle.system.scorecard.evaluate` uses so that scoring a system against a held-out
        question set can never itself teach the system the held-out answers: an evaluation call must
        never be able to change what a later call to this same method returns. Every other observable
        (the reply, the receipt fields a scorer reads, whether the call degraded) is identical to a
        normal call, since a snapshot must still faithfully measure the real answering path.

        ``budget`` is validated as an exact, non-Boolean, nonnegative call count -- the same
        :func:`~mixle.system.spend._count` contract :class:`~mixle.system.spend.Spend` holds its own
        dimensions to, since the ceiling is compared directly against
        :meth:`~mixle.system.spend.Spend.total_units` (MXR-080-1902). ``int(budget)`` TRUNCATED
        instead: ``budget=1.9`` silently became a one-call ceiling and ``budget=True`` became one
        too, and the receipt then reported the truncated number as if it were what the caller
        asked for. A negative budget is a caller error and raises, matching :meth:`improve`, rather
        than being served back as an ordinary "refused, shortfall N" receipt that reads like a
        legitimately under-funded request.
        """
        if budget is not None:
            budget = _count("budget", budget)
        cache_key = (query.text, query.task, query.scope)
        if cache_key in self._captured:
            return self._captured[cache_key], {
                "produced_by": "captured",
                "status": "answered",
                "spend": Spend().to_dict(),
                "total_spend": self.total_spend.to_dict(),
                "budget": self.config.default_budget if budget is None else budget,
                "captured": True,
                "task": query.task,
                "degraded_mode": None,
                "degraded_reason": None,
                "read_only": read_only,
            }

        requested = self.config.default_budget if budget is None else budget
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
                "read_only": read_only,
            }

        def _call_teacher() -> str:
            return _complete(self.config.teacher, query.text)

        def _teacher_down_fallback(exc: Exception) -> str:
            if self.config.store is not None:
                from mixle.substrate.retrieve import retrieve

                # the query's OWN scope, never config.scope: a degraded fallback must stay inside the
                # same tenant/evidence boundary the query itself declared, not silently widen to the
                # system-wide default (see Query.scope's docstring -- scope is a query-level boundary).
                hits = retrieve(self.config.store, query.text, k=3, scope=query.scope)
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
                "read_only": read_only,
            }
        actual_cost = Spend() if result.degraded else cost
        if not read_only:
            # the only two places this method mutates persistent state -- guarded together so a
            # read_only call is a genuine snapshot: it cannot grow total_spend, and (this being the
            # important half) it cannot land in _harvest, so a later improve() can never promote
            # something this call answered.
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
            "read_only": read_only,
            **result.to_receipt_fields(),
        }
        return result.value, receipt

    def ingest(self, model_output: str, *, source: dict[str, Any]) -> dict[str, Any]:
        """Turn a model output into stored knowledge.

        Uses the belief store when it is importable; otherwise records a plain
        substrate item rather than requiring optional knowledge-substrate
        components.

        Every route writes under :attr:`SystemConfig.scope`, and the effective scope is named on the
        report. The belief path used to call ``assimilate()`` without its ``scope`` argument, so it
        defaulted to ``"local"`` no matter how the system was configured -- a system configured for
        ``"tenant-A"`` ingested two claims and got back two apparently successful ``BeliefItem``s whose
        scope, and whose stored substrate scope, were both ``"local"``. The import-fallback path *did*
        use ``config.scope``, so which boundary a write landed in depended on whether an optional
        module happened to be importable.

        If the store write raises before anything is committed, this falls back to ``store_down``
        degraded mode: the model output is acknowledged but not accumulated, and the report is flagged
        with ``degraded_mode="store_down"``.

        Claims are assimilated one at a time and there is no transaction, so a failure partway through
        leaves the earlier claims stored. That case is reported as ``status="partial_accumulation"``
        with the exact committed items, not as ``degraded_no_accumulation``: denying durable state
        invites a retry that duplicates the committed evidence, or reasoning from a receipt that says
        nothing was written when something was.
        """
        if self.config.store is None:
            return {"status": "no_store", "assimilated": False, "scope": self.config.scope}

        def _write() -> dict[str, Any]:
            try:
                from mixle.substrate.belief import assimilate, harvest_knowledge
            except ImportError:
                return self._ingest_fallback(model_output, source=source)
            claims = harvest_knowledge(model_output, source=source)
            items: list[Any] = []
            for claim in claims:
                try:
                    items.append(assimilate(self.config.store, claim, [], scope=self.config.scope))
                except Exception as exc:  # noqa: BLE001 -- partial commits are reported, not degraded away
                    if not items:
                        raise  # nothing committed yet: an ordinary store_down degradation
                    return {
                        "status": "partial_accumulation",
                        "assimilated": True,
                        "n_claims": len(claims),
                        "n_committed": len(items),
                        "items": items,
                        "committed_ids": [getattr(item, "id", None) for item in items],
                        "failed_claim_index": len(items),
                        "reason": str(exc),
                        "degraded_mode": "store_down",
                        "scope": self.config.scope,
                    }
            return {"status": "ok", "n_claims": len(claims), "items": items, "scope": self.config.scope}

        def _store_down_fallback(exc: Exception) -> dict[str, Any]:
            return {"status": "degraded_no_accumulation", "assimilated": False, "scope": self.config.scope}

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
        return {"status": "ok_fallback", "assimilated": False, "item_id": item.id, "scope": self.config.scope}

    def improve(self, budget: int) -> dict[str, Any]:
        """Promote harvested (query, reply) pairs from :meth:`answer` into the captured cache, up to
        ``budget``.

        Promotion is costed at a flat :data:`_PROMOTE_COST_PER_ITEM` (currently 1) unit per (query,
        reply) pair -- the only cost model defined for this step, since promoting a harvested pair is a
        pure cache write, not an external call like the frontier/oracle spend :meth:`answer` meters.
        ``budget`` is a hard ceiling on how many pairs THIS call may promote, the same hard-ceiling
        spirit as ``answer``'s budget: a negative budget is a caller error and raises ``ValueError``
        rather than being silently treated as zero or unlimited.

        Candidates are promoted in harvest order (the order :meth:`answer` first harvested them --
        harvested pairs carry no other recency or confidence signal to rank by, so this is the simplest
        available, documented priority). Whatever does not fit the budget is left in the harvest for a
        later, better-funded call rather than being discarded.

        Reports that there is nothing to improve when nothing has been harvested yet. Otherwise this is
        the cold-start capture step: after this call, a repeat of a captured query is answered from the
        local cache (see :meth:`answer`). The return value reports both the requested ceiling
        (``budget``) and the realized spend (``realized_spend``) so a caller can tell a fully-funded
        round from a partial one.

        ``budget`` is an exact, non-Boolean, nonnegative promotion count
        (:func:`~mixle.system.spend._count`, MXR-080-1902). ``int(budget)`` truncated first and only
        then rejected a negative, so ``improve(1.9)`` quietly promoted one pair against a budget the
        caller wrote as nearly two, ``improve(True)`` promoted one, and ``improve(-0.5)`` truncated
        to ``0`` and returned an ordinary "insufficient_budget" report instead of naming the error
        that the same call written as ``improve(-1)`` raises for.
        """
        budget = _count("improve budget", budget)
        if not self._harvest:
            return {
                "status": "nothing_to_improve",
                "reason": "no improvement subsystem registered yet",
                "budget": budget,
                "realized_spend": 0,
                "n_captured": 0,
                "n_skipped": 0,
            }
        affordable = budget // _PROMOTE_COST_PER_ITEM
        candidates = list(self._harvest.items())[:affordable]
        for key, _value in candidates:
            self._captured[key] = self._harvest.pop(key)
        n_captured = len(candidates)
        realized_spend = n_captured * _PROMOTE_COST_PER_ITEM
        n_skipped = len(self._harvest)
        if n_captured:
            reason = f"promoted {n_captured} harvested (query, reply) pair(s) into the captured cache"
        else:
            reason = f"budget {budget} affords 0 of {n_skipped} harvested pair(s) at {_PROMOTE_COST_PER_ITEM} unit/pair"
        return {
            "status": "captured" if n_captured else "insufficient_budget",
            "reason": reason,
            "budget": budget,
            "realized_spend": realized_spend,
            "n_captured": n_captured,
            "n_skipped": n_skipped,
        }
