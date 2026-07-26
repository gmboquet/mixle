"""Distill single-step tool calling into a calibrated local model.

Tool calling decomposes into two problems the task spine already solves, composed under one gate:

  * **which tool** (or none) -- a calibrated classification student (:func:`~mixle.task.solve.solve`
    over the request text: conformal answer-or-escalate + optional OOD gate);
  * **the arguments** -- one token-level extractor per tool (:func:`~mixle.task.extract.distill_extractor`),
    distilled from the teacher's own argument fills.

``teacher(request) -> {"tool": name, "args": {...}}`` (or ``{"tool": None}``)
can be a frontier LLM behind :mod:`mixle.task.llm`, an agent loop, or a rule.
The returned :class:`ToolCaller` emits a call only when the selector is
conformally confident and every required argument extracts. Anything else
escalates to the teacher and is harvested for the next distillation round.

This covers single-step function calling. Multi-step planning is handled by the
planner distillation surfaces.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from mixle.task.extract import distill_extractor
from mixle.task.solve import Solution, solve

_NO_TOOL = "__none__"


@dataclass
class ToolSpec:
    """One callable tool: its name and the argument fields to extract from the request text."""

    name: str
    args: list[str]
    required: list[str] | None = None  # defaults to all args

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("tool name must be a valid non-empty identifier")
        if (
            not isinstance(self.args, list)
            or any(not isinstance(arg, str) or not arg.isidentifier() for arg in self.args)
            or len(set(self.args)) != len(self.args)
        ):
            raise ValueError(f"tool {self.name!r} arguments must be unique identifiers")
        self.args = list(self.args)
        if self.required is not None:
            if (
                not isinstance(self.required, list)
                or any(not isinstance(arg, str) for arg in self.required)
                or len(set(self.required)) != len(self.required)
                or any(arg not in self.args for arg in self.required)
            ):
                raise ValueError(f"tool {self.name!r} required arguments must be a unique subset of args")
            self.required = list(self.required)

    @property
    def required_args(self) -> list[str]:
        """Return required argument names, defaulting to all declared arguments."""
        return list(self.required) if self.required is not None else list(self.args)


def _tool_spec_map(tools: Sequence[ToolSpec], *, reserved: Sequence[str] = ()) -> dict[str, ToolSpec]:
    specs: dict[str, ToolSpec] = {}
    for spec in tools:
        if not isinstance(spec, ToolSpec):
            raise TypeError("tools must contain ToolSpec instances")
        if spec.name in reserved:
            raise ValueError(f"tool name {spec.name!r} is reserved")
        if spec.name in specs:
            raise ValueError(f"duplicate tool name {spec.name!r}")
        specs[spec.name] = spec
    if not specs:
        raise ValueError("at least one tool specification is required")
    return specs


def _validated_args(spec: ToolSpec, raw_args: Any) -> dict[str, Any]:
    if not isinstance(raw_args, dict):
        raise ValueError(f"arguments for tool {spec.name!r} must be a dictionary")
    unknown = set(raw_args) - set(spec.args)
    if unknown:
        raise ValueError(f"tool {spec.name!r} received undeclared arguments {sorted(unknown)!r}")
    missing = [name for name in spec.required_args if name not in raw_args]
    if missing:
        raise ValueError(f"tool {spec.name!r} is missing required arguments {missing!r}")
    return {name: raw_args[name] for name in spec.args if name in raw_args}


def _validated_call(call: Any, specs: dict[str, ToolSpec]) -> dict[str, Any]:
    if not isinstance(call, dict):
        raise ValueError("tool call must be a dictionary")
    name = call.get("tool")
    if name is None:
        args = call.get("args", {})
        if args not in ({}, None):
            raise ValueError("a no-tool call cannot carry arguments")
        return {"tool": None, "args": {}}
    if name not in specs:
        raise ValueError(f"tool call names undeclared tool {name!r}")
    return {"tool": name, "args": _validated_args(specs[name], call.get("args", {}))}


@dataclass
class ToolCaller:
    """Distilled function caller with calibrated selection and argument extraction."""

    selector: Solution
    extractors: dict[str, Any]
    tools: dict[str, ToolSpec]
    teacher: Callable[[str], dict]
    selection_agreement: float
    n_requests: int = 0
    n_escalated: int = 0
    harvested: list[tuple[str, dict]] = field(default_factory=list)

    def try_local(self, request: str) -> dict[str, Any] | None:
        """Return the local decision, or ``None`` when the request must escalate.

        This method does not call the teacher.
        """
        tool = self.selector.cascade.model.decide(request)
        if tool is not None and tool != _NO_TOOL and tool in self.tools:
            spec = self.tools[tool]
            if not spec.args:
                return {"tool": tool, "args": {}}
            if tool not in self.extractors:
                return None
            args = self.extractors[tool](request)
            try:
                return {"tool": tool, "args": _validated_args(spec, args)}
            except ValueError:
                return None
        if tool == _NO_TOOL:
            return {"tool": None, "args": {}}
        return None

    def __call__(self, request: str) -> dict[str, Any]:
        """Return ``{"tool", "args", "escalate"}``; escalations carry the teacher's call and are harvested."""
        self.n_requests += 1
        local = self.try_local(request)
        if local is not None:
            return {**local, "escalate": False}
        self.n_escalated += 1
        out = _validated_call(self.teacher(request), self.tools)
        self.harvested.append((request, out))
        return {**out, "escalate": True}

    def report(self) -> dict[str, Any]:
        """Return serving counts, escalation rate, and selector agreement diagnostics."""
        return {
            "selection_agreement": round(self.selection_agreement, 4),
            "requests": self.n_requests,
            "escalated": self.n_escalated,
            "escalation_rate": (self.n_escalated / self.n_requests) if self.n_requests else 0.0,
            "harvested_traces": len(self.harvested),
        }

    def save(self, path: str) -> str:
        """Persist selector, per-tool extractors, and tool specs."""
        import json
        from pathlib import Path

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.selector.save(str(out / "selector"))
        for name, ex in self.extractors.items():
            ex.save(str(out / "extractors" / name))
        manifest = {
            "kind": "toolcaller/v1",
            "tools": {n: {"args": t.args, "required": t.required} for n, t in self.tools.items()},
            "extractors": sorted(self.extractors),
            "selection_agreement": self.selection_agreement,
        }
        (out / "toolcaller.json").write_text(json.dumps(manifest, indent=2))
        return str(out)

    @classmethod
    def load(cls, path: str, teacher: Callable[[str], dict], *, device: str = "cpu") -> ToolCaller:
        """Reconstitute a serving ToolCaller from :meth:`save` output plus the teacher fallback."""
        import json
        from pathlib import Path

        from mixle.task.model import TaskModel
        from mixle.task.solve import Solution

        p = Path(path)
        manifest = json.loads((p / "toolcaller.json").read_text())
        selector = Solution.load(
            str(p / "selector"), lambda batch: [teacher(r).get("tool") or _NO_TOOL for r in batch], device=device
        )
        extractors = {
            name: TaskModel.load(str(p / "extractors" / name), device=device) for name in manifest["extractors"]
        }
        tools = _tool_spec_map(
            [ToolSpec(n, list(t["args"]), t.get("required")) for n, t in manifest["tools"].items()],
            reserved=(_NO_TOOL,),
        )
        return cls(
            selector=selector,
            extractors=extractors,
            tools=tools,
            teacher=teacher,
            selection_agreement=float(manifest.get("selection_agreement", float("nan"))),
        )


def distill_tool_caller(
    teacher: Callable[[str], dict],
    requests: Sequence[str],
    tools: Sequence[ToolSpec],
    *,
    seed: int = 0,
    selector_kw: dict | None = None,
    extractor_kw: dict | None = None,
) -> ToolCaller:
    """Distill the teacher's function-calling into a local selector plus per-tool argument extractors.

    Args:
        teacher: ``teacher(request) -> {"tool": name-or-None, "args": {field: value}}`` — the frontier
            LLM / agent / rule currently doing the calling. It labels everything; it remains the fallback.
        requests: example request texts covering the tools.
        tools: the tool specs (names + argument fields; ``required`` defaults to all).
        selector_kw / extractor_kw: knobs forwarded to :func:`solve` and :func:`distill_extractor`.
    """
    reqs = [str(r) for r in requests]
    if len(reqs) < 8:
        raise ValueError("distill_tool_caller needs at least 8 example requests")
    specs = _tool_spec_map(tools, reserved=(_NO_TOOL,))
    calls = [_validated_call(teacher(r), specs) for r in reqs]

    # 1) Tool selection: a calibrated classification student over the request text.
    call_by_req = dict(zip(reqs, calls))

    def select_teacher(r: str) -> str:
        got = call_by_req.get(r)
        got = got if got is not None else teacher(r)
        return got.get("tool") or _NO_TOOL

    selector = solve(select_teacher, reqs, seed=seed, **(selector_kw or {}))

    # 2) Arguments: one extractor per tool, trained on that tool's requests with the teacher's fills.
    extractors: dict[str, Any] = {}
    for name, spec in specs.items():
        rows = [(r, c) for r, c in zip(reqs, calls) if c.get("tool") == name]
        if len(rows) < 8 or not spec.args:
            continue  # too little data escalates; an argless tool needs selection alone

        def make_arg_teacher(table: dict) -> Callable[[Any], Any]:
            def arg_teacher(text: Any) -> Any:
                if isinstance(text, list):  # distill probes the teacher batched first
                    return [arg_teacher(t) for t in text]
                got = table.get(text)
                return dict((got or teacher(text)).get("args") or {})

            return arg_teacher

        extractors[name] = distill_extractor(
            make_arg_teacher(dict(rows)), [r for r, _ in rows], spec.args, seed=seed, **(extractor_kw or {})
        )

    return ToolCaller(
        selector=selector,
        extractors=extractors,
        tools=specs,
        teacher=teacher,
        selection_agreement=float(selector.holdout_agreement),
    )
