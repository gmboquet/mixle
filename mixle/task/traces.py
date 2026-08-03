"""Harvest stored agent-session traces for task distillation.

The mixle-agent server persists every conversation (``~/.mixle-agent/conversations/*.json``) with the
full message stream, including the ``tool_use`` blocks the frontier model emitted. Those traces can
serve as deterministic teachers for agentic distillation workflows::

    traces = harvest_agent_traces()                       # or (dir=...) for a custom store
    tools  = traces.tool_specs()                           # ToolSpecs inferred from observed usage
    tc     = distill_tool_caller(traces.call_teacher(), traces.requests(), tools)
    gp     = sft_planner(traces.plan_teacher(), traces.requests(min_steps=1), tools)

Each trace pairs a user request with the ordered tool calls the assistant made
before the next user turn, plus the final text reply. Tool specs are inferred
from observed usage: a tool's argument set is the union of keys ever passed,
and ``required`` is the keys present in every observed call. The teachers are
lookup tables over harvested requests; they do not call a model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mixle.task.toolcall import ToolSpec
from mixle.utils.immutable import detach_receipt_container

_DEFAULT_DIR = Path.home() / ".mixle-agent" / "conversations"


class TraceFormatError(ValueError):
    """Raised when a stored trace row violates the conversation schema."""


class AmbiguousTraceError(LookupError):
    """Raised when one request text has multiple conflicting harvested executions."""


@dataclass(frozen=True)
class TraceRejection:
    """A file or row rejected during harvesting, with enough location to repair it."""

    source: str
    location: str
    reason: str


@dataclass
class AgentTrace:
    """One request, ordered tool calls, and final text reply."""

    request: str
    plan: list[dict]  # [{"tool": name, "args": {...}}, ...] in execution order
    reply: str = ""
    conversation_id: str = ""
    trace_id: str = ""
    # MXR-080-1892: True when this turn's transcript recorded tool results and EVERY harvested step was
    # bound to a non-error one -- i.e. the plan is a verified successful execution rather than an
    # unaudited list of attempts. Vacuously True for a turn with no tool calls. False means the
    # transcript carried no outcomes for these calls, so nothing here claims they worked.
    outcomes_verified: bool = False

    def __post_init__(self) -> None:
        # The plan arrived as a caller-owned list of dicts and was stored by reference, so a consumer
        # that edited a step (or a nested ``args`` value handed out by the teachers) rewrote the
        # harvested corpus itself. Detached rather than frozen: plans are JSON-serialized and
        # type-tested by the distillers, so list/dict must stay list/dict (MXR-080-1892).
        self.plan = detach_receipt_container(self.plan)


@dataclass
class AgentTraces:
    """The harvested corpus plus the teacher views the distillers consume."""

    traces: list[AgentTrace] = field(default_factory=list)
    rejections: list[TraceRejection] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.traces)

    def requests(self, *, min_steps: int = 0) -> list[str]:
        """The request texts (optionally only those whose plan has at least ``min_steps`` calls)."""
        return [t.request for t in self.traces if len(t.plan) >= min_steps]

    def by_id(self, trace_id: str) -> AgentTrace:
        """Return one execution by its stable harvested ID."""
        matches = [trace for trace in self.traces if trace.trace_id == trace_id]
        if not matches:
            raise KeyError(trace_id)
        if len(matches) > 1:
            raise AmbiguousTraceError(f"duplicate trace ID {trace_id!r}")
        return matches[0]

    def tool_specs(self) -> list[ToolSpec]:
        """Infer tool specs from observed argument usage."""
        seen: dict[str, list[dict]] = {}
        for t in self.traces:
            for step in t.plan:
                seen.setdefault(step["tool"], []).append(dict(step.get("args") or {}))
        specs = []
        for name in sorted(seen):
            calls = seen[name]
            union = sorted({k for c in calls for k in c})
            required = sorted(k for k in union if all(k in c for c in calls))
            specs.append(ToolSpec(name, union, required))
        return specs

    def _by_request(self) -> dict[str, list[AgentTrace]]:
        table: dict[str, list[AgentTrace]] = {}
        for trace in self.traces:
            table.setdefault(trace.request, []).append(trace)
        return table

    @staticmethod
    def _unique_execution(candidates: list[AgentTrace], request: str) -> AgentTrace | None:
        if not candidates:
            return None
        signatures = {
            json.dumps({"plan": trace.plan, "reply": trace.reply}, sort_keys=True, default=repr) for trace in candidates
        }
        if len(signatures) != 1:
            ids = [trace.trace_id for trace in candidates]
            raise AmbiguousTraceError(f"request {request!r} has conflicting executions {ids!r}; select by trace ID")
        return candidates[0]

    def call_teacher(self) -> Any:
        """Return a ``distill_tool_caller`` teacher over the first tool call."""
        table = self._by_request()

        def teacher(r: Any) -> Any:
            if isinstance(r, list):
                return [teacher(x) for x in r]
            request = str(r)
            t = self._unique_execution(table.get(request, []), request)
            if t is None or not t.plan:
                return {"tool": None, "args": {}}
            first = t.plan[0]
            # deep, not dict(): a one-level copy still shares every nested args value with the
            # harvested corpus, so a consumer editing one rewrote the trace (MXR-080-1892).
            return {"tool": first["tool"], "args": detach_receipt_container(first.get("args") or {})}

        return teacher

    def plan_teacher(self) -> Any:
        """Return a planner teacher over the full harvested tool-call plan."""
        table = self._by_request()

        def teacher(r: Any) -> Any:
            if isinstance(r, list):
                return [teacher(x) for x in r]
            request = str(r)
            t = self._unique_execution(table.get(request, []), request)
            # deep, not dict(): see call_teacher (MXR-080-1892).
            return [detach_receipt_container(s) for s in t.plan] if t is not None else []

        return teacher


def _reject(
    rejections: list[TraceRejection] | None,
    *,
    source: str,
    location: str,
    reason: str,
) -> None:
    rejection = TraceRejection(source=source, location=location, reason=reason)
    if rejections is None:
        raise TraceFormatError(f"{source}:{location}: {reason}")
    rejections.append(rejection)


def _content_blocks(
    message: Any,
    *,
    source: str,
    message_index: int,
    rejections: list[TraceRejection] | None,
) -> list[dict[str, Any]]:
    if not isinstance(message, dict):
        _reject(rejections, source=source, location=f"messages[{message_index}]", reason="message must be an object")
        return []
    content = message.get("content", [])
    if not isinstance(content, list):
        _reject(
            rejections,
            source=source,
            location=f"messages[{message_index}].content",
            reason="content must be an array",
        )
        return []
    valid: list[dict[str, Any]] = []
    for block_index, block in enumerate(content):
        if not isinstance(block, dict):
            _reject(
                rejections,
                source=source,
                location=f"messages[{message_index}].content[{block_index}]",
                reason="content block must be an object",
            )
            continue
        valid.append(block)
    return valid


def _text_of(
    message: Any,
    *,
    source: str = "conversation",
    message_index: int = 0,
    rejections: list[TraceRejection] | None = None,
) -> str:
    parts: list[str] = []
    for block_index, block in enumerate(
        _content_blocks(message, source=source, message_index=message_index, rejections=rejections)
    ):
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if not isinstance(text, str):
            _reject(
                rejections,
                source=source,
                location=f"messages[{message_index}].content[{block_index}].text",
                reason="text block value must be a string",
            )
            continue
        parts.append(text)
    return " ".join(parts).strip()


def _tool_uses(
    message: Any,
    *,
    source: str = "conversation",
    message_index: int = 0,
    rejections: list[TraceRejection] | None = None,
) -> list[tuple[dict, str | None, str]]:
    """Validated ``tool_use`` blocks as ``(step, tool_use_id, location)``.

    The id and location are returned alongside the step -- rather than folded into it -- so
    :func:`parse_conversation` can bind each call to its result without putting an id in the plan the
    distillers train on. An id in the plan would make two otherwise identical executions of the same
    request compare as conflicting in :meth:`AgentTraces._unique_execution` (MXR-080-1892).
    """
    uses: list[tuple[dict, str | None, str]] = []
    for block_index, block in enumerate(
        _content_blocks(message, source=source, message_index=message_index, rejections=rejections)
    ):
        if block.get("type") != "tool_use":
            continue
        name = block.get("name")
        raw_input = block.get("input", {})
        location = f"messages[{message_index}].content[{block_index}]"
        if not isinstance(name, str) or not name.isidentifier():
            _reject(rejections, source=source, location=location, reason="tool_use name must be an identifier")
            continue
        if not isinstance(raw_input, dict):
            _reject(rejections, source=source, location=location, reason="tool_use input must be an object")
            continue
        if any(not isinstance(key, str) or not key.isidentifier() for key in raw_input):
            _reject(rejections, source=source, location=location, reason="tool_use argument keys must be identifiers")
            continue
        use_id = block.get("id")
        uses.append(({"tool": name, "args": dict(raw_input)}, use_id if isinstance(use_id, str) else None, location))
    return uses


def _tool_results(
    message: Any,
    *,
    source: str = "conversation",
    message_index: int = 0,
    rejections: list[TraceRejection] | None = None,
) -> dict[str, bool]:
    """``tool_use_id -> is_error`` for every ``tool_result`` block in one message.

    A result whose ``is_error`` is not a bool is treated as an error rather than a success: an
    unparseable outcome is not evidence that the call worked (MXR-080-1892). Deliberately NOT checked:
    the result's ``content`` -- a successful call may legitimately return anything, including nothing.
    """
    results: dict[str, bool] = {}
    for block_index, block in enumerate(
        _content_blocks(message, source=source, message_index=message_index, rejections=rejections)
    ):
        if block.get("type") != "tool_result":
            continue
        use_id = block.get("tool_use_id")
        if not isinstance(use_id, str) or not use_id:
            _reject(
                rejections,
                source=source,
                location=f"messages[{message_index}].content[{block_index}]",
                reason="tool_result must carry a string tool_use_id",
            )
            continue
        flag = block.get("is_error", False)
        results[use_id] = True if not isinstance(flag, bool) else flag
    return results


def _has_tool_result(message: Any) -> bool:
    """Whether a message carries at least one ``tool_result`` block (validation deferred to
    :func:`_tool_results`; this is only the turn-boundary question)."""
    content = message.get("content") if isinstance(message, dict) else None
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "tool_result" for block in content
    )


def parse_conversation(
    doc: dict,
    *,
    source_id: str = "",
    rejections: list[TraceRejection] | None = None,
) -> list[AgentTrace]:
    """Split one stored conversation into request-to-tool-plan traces."""
    source = source_id or "conversation"
    if not isinstance(doc, dict):
        raise TraceFormatError(f"{source}: document must be an object")
    out: list[AgentTrace] = []
    convo_id = str(doc.get("id", ""))
    raw_messages = doc.get("messages", [])
    if not isinstance(raw_messages, list):
        raise TraceFormatError(f"{source}: messages must be an array")
    messages = list(raw_messages)
    i = 0
    while i < len(messages):
        m = messages[i]
        if not isinstance(m, dict):
            _reject(rejections, source=source, location=f"messages[{i}]", reason="message must be an object")
            i += 1
            continue
        request = _text_of(m, source=source, message_index=i, rejections=rejections)
        if m.get("role") != "user" or not request:
            i += 1
            continue
        attempted: list[tuple[dict, str | None, str]] = []
        outcomes: dict[str, bool] = {}
        saw_outcomes = False
        reply = ""
        j = i + 1
        while j < len(messages):
            mj = messages[j]
            if not isinstance(mj, dict):
                _reject(rejections, source=source, location=f"messages[{j}]", reason="message must be an object")
                j += 1
                continue
            if mj.get("role") == "user":
                # A user turn carrying tool_result blocks is the transcript handing results BACK to the
                # assistant, not a new request. Treating it as a turn boundary truncated the plan at the
                # first tool call and silently discarded everything after it -- including the successful
                # retry that followed a failed call (MXR-080-1892). Its outcomes are harvested either
                # way; only a turn that also carries request text ends this trace.
                if _has_tool_result(mj):
                    outcomes.update(_tool_results(mj, source=source, message_index=j, rejections=rejections))
                    saw_outcomes = True
                    if not _text_of(mj, source=source, message_index=j, rejections=rejections):
                        j += 1
                        continue
                break
            if mj.get("role") == "assistant":
                attempted.extend(_tool_uses(mj, source=source, message_index=j, rejections=rejections))
                text = _text_of(mj, source=source, message_index=j, rejections=rejections)
                if text:
                    reply = text
            j += 1

        # Bind every attempt to its recorded outcome. A transcript that records outcomes at all is one
        # this module can audit, so an unbound or errored call is rejected into the ledger rather than
        # taught as a correct plan step. A transcript with no tool_result blocks anywhere records no
        # outcomes -- the pre-existing shape of most stored conversations -- so its calls are still
        # harvested, but the trace says so via ``outcomes_verified=False`` instead of implying success.
        plan: list[dict] = []
        for step, use_id, location in attempted:
            if not saw_outcomes:
                plan.append(step)
                continue
            if use_id is None or use_id not in outcomes:
                _reject(
                    rejections,
                    source=source,
                    location=location,
                    reason="tool_use has no bound tool_result in a transcript that records outcomes",
                )
                continue
            if outcomes[use_id]:
                _reject(rejections, source=source, location=location, reason="tool_use returned an error result")
                continue
            plan.append(step)

        stable_source = source_id or convo_id or "conversation"
        out.append(
            AgentTrace(
                request=request,
                plan=plan,
                reply=reply,
                conversation_id=convo_id,
                trace_id=f"{stable_source}:{i}",
                outcomes_verified=saw_outcomes or not attempted,
            )
        )
        i = j
    return out


def harvest_agent_traces(directory: str | Path | None = None) -> AgentTraces:
    """Read stored conversations, preserving traces and a ledger of rejected files/rows."""
    root = Path(directory) if directory is not None else _DEFAULT_DIR
    traces: list[AgentTrace] = []
    rejections: list[TraceRejection] = []
    if root.is_dir():
        for p in sorted(root.glob("*.json")):
            try:
                document = json.loads(p.read_text())
                traces.extend(parse_conversation(document, source_id=p.name, rejections=rejections))
            except (OSError, ValueError) as exc:
                rejections.append(TraceRejection(source=str(p), location="document", reason=str(exc)))
    return AgentTraces(traces=traces, rejections=rejections)
