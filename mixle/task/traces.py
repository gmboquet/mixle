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

_DEFAULT_DIR = Path.home() / ".mixle-agent" / "conversations"


@dataclass
class AgentTrace:
    """One request, ordered tool calls, and final text reply."""

    request: str
    plan: list[dict]  # [{"tool": name, "args": {...}}, ...] in execution order
    reply: str = ""
    conversation_id: str = ""


@dataclass
class AgentTraces:
    """The harvested corpus plus the teacher views the distillers consume."""

    traces: list[AgentTrace] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.traces)

    def requests(self, *, min_steps: int = 0) -> list[str]:
        """The request texts (optionally only those whose plan has at least ``min_steps`` calls)."""
        return [t.request for t in self.traces if len(t.plan) >= min_steps]

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

    def _by_request(self) -> dict[str, AgentTrace]:
        return {t.request: t for t in self.traces}

    def call_teacher(self) -> Any:
        """Return a ``distill_tool_caller`` teacher over the first tool call."""
        table = self._by_request()

        def teacher(r: Any) -> Any:
            if isinstance(r, list):
                return [teacher(x) for x in r]
            t = table.get(str(r))
            if t is None or not t.plan:
                return {"tool": None, "args": {}}
            first = t.plan[0]
            return {"tool": first["tool"], "args": dict(first.get("args") or {})}

        return teacher

    def plan_teacher(self) -> Any:
        """Return a planner teacher over the full harvested tool-call plan."""
        table = self._by_request()

        def teacher(r: Any) -> Any:
            if isinstance(r, list):
                return [teacher(x) for x in r]
            t = table.get(str(r))
            return [dict(s) for s in t.plan] if t is not None else []

        return teacher


def _text_of(message: dict) -> str:
    return " ".join(b.get("text", "") for b in message.get("content", []) if b.get("type") == "text").strip()


def _tool_uses(message: dict) -> list[dict]:
    return [
        {"tool": str(b.get("name")), "args": {k: v for k, v in (b.get("input") or {}).items()}}
        for b in message.get("content", [])
        if b.get("type") == "tool_use" and b.get("name")
    ]


def parse_conversation(doc: dict) -> list[AgentTrace]:
    """Split one stored conversation into request-to-tool-plan traces."""
    out: list[AgentTrace] = []
    convo_id = str(doc.get("id", ""))
    messages = list(doc.get("messages", []))
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.get("role") != "user" or not _text_of(m):
            i += 1
            continue
        request = _text_of(m)
        plan: list[dict] = []
        reply = ""
        j = i + 1
        while j < len(messages) and messages[j].get("role") != "user":
            if messages[j].get("role") == "assistant":
                plan.extend(_tool_uses(messages[j]))
                text = _text_of(messages[j])
                if text:
                    reply = text
            j += 1
        out.append(AgentTrace(request=request, plan=plan, reply=reply, conversation_id=convo_id))
        i = j
    return out


def harvest_agent_traces(directory: str | Path | None = None) -> AgentTraces:
    """Read every stored mixle-agent conversation and return the trace corpus (skips unreadable files)."""
    root = Path(directory) if directory is not None else _DEFAULT_DIR
    traces: list[AgentTrace] = []
    if root.is_dir():
        for p in sorted(root.glob("*.json")):
            try:
                traces.extend(parse_conversation(json.loads(p.read_text())))
            except (OSError, ValueError):
                continue  # a corrupt file is skipped, never fatal
    return AgentTraces(traces=traces)
