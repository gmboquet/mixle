"""``distill_planner`` trains local models to decompose requests into tool steps.

The plan representation is an autoregressive chain of calibrated tool calls ending in
``STOP``. The teacher (an LLM, an agent loop, or a rule) shows plans for example requests; each
trace flattens into ``(context, next-call)`` pairs where the context is the request plus the steps taken
so far. "Predict the next call" is the problem :mod:`~mixle.task.toolcall` already solves:
a conformal selector for which tool comes next (``STOP`` is just another action) and a per-tool extractor for
its arguments, both reading the rendered context.

The safety contract is stepwise: a step is emitted only when the selector is confident, the
required arguments extract, and, when an ``execute`` map is given, the call actually runs. Any failure
escalates the whole request to the teacher; a partially executed guessed plan is not returned as local success,
and the escalation is harvested as a fresh trace for the next distillation round.

    teacher(request) -> [{"tool": ..., "args": {...}}, ...]      # the plan
    planner = distill_planner(teacher, requests, tools)
    planner(request)                                             # {"plan", "escalate"}
    planner(request, execute={"lookup": fn, ...})                # + per-step "results", verified

This is template-oriented decomposition. For free-form generated plans, use the
trace-SFT planner on the same trace format.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from mixle.task.extract import distill_extractor
from mixle.task.solve import solve
from mixle.task.toolcall import ToolSpec, _tool_spec_map, _validated_args, _validated_call

_STOP = "__stop__"


def _render(request: str, steps: Sequence[dict]) -> str:
    """The context the students read: the request plus the steps taken so far."""
    if not steps:
        return f"{request} [plan so far: none]"
    done = "; ".join(f"{s['tool']}({', '.join(f'{k}={v}' for k, v in (s.get('args') or {}).items())})" for s in steps)
    return f"{request} [plan so far: {done}]"


def _validated_plan(raw_plan: Any, tools: dict[str, ToolSpec], max_steps: int) -> list[dict[str, Any]]:
    if isinstance(raw_plan, (str, bytes)) or not isinstance(raw_plan, Sequence):
        raise ValueError("plan must be a sequence of tool-call dictionaries")
    if len(raw_plan) > max_steps:
        raise ValueError(f"plan contains {len(raw_plan)} steps, exceeding max_steps={max_steps}")
    plan = [_validated_call(step, tools) for step in raw_plan]
    if any(step["tool"] is None for step in plan):
        raise ValueError("planner plans cannot contain a no-tool step; use an empty plan to stop")
    return plan


def _validate_execution_policy(plan: Sequence[dict[str, Any]], execute: dict[str, Callable[..., Any]]) -> None:
    if not isinstance(execute, dict):
        raise TypeError("execute must be a dictionary of tool names to callables")
    missing = sorted({step["tool"] for step in plan if not callable(execute.get(step["tool"]))})
    if missing:
        raise ValueError(f"execute has no callable implementation for tools {missing}")


def _execute_plan(plan: list[dict[str, Any]], execute: dict[str, Callable[..., Any]]) -> dict[str, Any]:
    """Execute an already-validated plan once, returning an explicit partial-state receipt on failure."""
    _validate_execution_policy(plan, execute)
    results: list[Any] = []
    for index, step in enumerate(plan):
        try:
            results.append(execute[step["tool"]](**step["args"]))
        except Exception as exc:  # noqa: BLE001 - external tool failures become auditable state
            return {
                "plan": plan,
                "results": results,
                "partial": True,
                "committed_steps": index,
                "failed_step": index,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
    return {"plan": plan, "results": results}


@dataclass
class Planner:
    """A distilled decomposer: emit verified steps until ``STOP``, or escalate the whole problem."""

    selector: Any
    extractors: dict[str, Any]
    tools: dict[str, ToolSpec]
    teacher: Callable[[str], list[dict]]
    plan_agreement: float
    max_steps: int = 8
    n_requests: int = 0
    n_escalated: int = 0
    harvested: list[tuple[str, list[dict]]] = field(default_factory=list)
    n_partial: int = 0

    def try_plan(self, request: str, *, execute: dict[str, Callable[..., Any]] | None = None) -> dict[str, Any] | None:
        """The local decomposition alone: a complete verified plan, or ``None`` (= must escalate).

        This method does not call the teacher."""
        steps: list[dict] = []
        for _ in range(self.max_steps):
            ctx = _render(request, steps)
            tool = self.selector.cascade.model.decide(ctx)
            if tool == _STOP:
                return _execute_plan(steps, execute) if execute is not None else {"plan": steps, "results": []}
            if tool is None or tool not in self.tools:
                return None  # unsure which step comes next
            spec = self.tools[tool]
            if spec.args:
                if tool not in self.extractors:
                    return None
                try:
                    args = _validated_args(spec, self.extractors[tool](ctx))
                except ValueError:
                    return None
            else:
                args = {}
            step = {"tool": tool, "args": args}
            steps.append(step)
        return None  # max_steps without STOP

    def __call__(self, request: str, *, execute: dict[str, Callable[..., Any]] | None = None) -> dict[str, Any]:
        """Plan (and optionally execute) step by step; any uncertain/malformed/failing step escalates."""
        self.n_requests += 1
        local = self.try_plan(request, execute=execute)
        if local is not None:
            if local.get("partial"):
                self.n_escalated += 1
                self.n_partial += 1
                return {**local, "escalate": True}
            return {**local, "escalate": False}
        self.n_escalated += 1
        plan = _validated_plan(self.teacher(request), self.tools, self.max_steps)
        self.harvested.append((request, plan))
        if execute is None:
            return {"plan": [dict(p) for p in plan], "escalate": True}
        executed = _execute_plan(plan, execute)
        if executed.get("partial"):
            self.n_partial += 1
        return {**executed, "escalate": True}

    def report(self) -> dict[str, Any]:
        """Return plan agreement, escalation, and harvested-trace metrics."""
        return {
            "plan_agreement": round(self.plan_agreement, 4),
            "requests": self.n_requests,
            "escalated": self.n_escalated,
            "escalation_rate": (self.n_escalated / self.n_requests) if self.n_requests else 0.0,
            "harvested_traces": len(self.harvested),
            "partial_executions": self.n_partial,
        }

    def save(self, path: str) -> str:
        """Persist selector + per-tool extractors + specs as one artifact directory; :meth:`load` restores."""
        import json
        from pathlib import Path

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.selector.save(str(out / "selector"))
        for name, ex in self.extractors.items():
            ex.save(str(out / "extractors" / name))
        manifest = {
            "kind": "planner/v1",
            "tools": {n: {"args": t.args, "required": t.required} for n, t in self.tools.items()},
            "extractors": sorted(self.extractors),
            "plan_agreement": self.plan_agreement,
            "max_steps": self.max_steps,
        }
        (out / "planner.json").write_text(json.dumps(manifest, indent=2))
        return str(out)

    @classmethod
    def load(cls, path: str, teacher: Callable[[str], list[dict]], *, device: str = "cpu") -> Planner:
        """Reconstitute a serving Planner from :meth:`save` output plus the teacher fallback."""
        import json
        from pathlib import Path

        from mixle.task.model import TaskModel
        from mixle.task.solve import Solution

        p = Path(path)
        manifest = json.loads((p / "planner.json").read_text())
        selector = Solution.load(str(p / "selector"), lambda batch: [_STOP for _ in batch], device=device)
        extractors = {
            name: TaskModel.load(str(p / "extractors" / name), device=device) for name in manifest["extractors"]
        }
        tools = _tool_spec_map(
            [ToolSpec(n, list(t["args"]), t.get("required")) for n, t in manifest["tools"].items()],
            reserved=(_STOP,),
        )
        return cls(
            selector=selector,
            extractors=extractors,
            tools=tools,
            teacher=teacher,
            plan_agreement=float(manifest.get("plan_agreement", float("nan"))),
            max_steps=int(manifest.get("max_steps", 8)),
        )


def distill_planner(
    teacher: Callable[[str], list[dict]],
    requests: Sequence[str],
    tools: Sequence[ToolSpec],
    *,
    holdout: float = 0.2,
    seed: int = 0,
    max_steps: int = 8,
    selector_kw: dict | None = None,
    extractor_kw: dict | None = None,
    behavior_verifier: Callable[[str, list[dict], list[dict]], bool] | None = None,
) -> Planner:
    """Distill the teacher's multi-step plans into next-step students (see module docstring).

    Plan-level verification is measured on held-out requests the students never trained on: a plan
    agrees when every step's tool and required arguments match the teacher's plan exactly, in order.
    """
    if not callable(teacher):
        raise TypeError("teacher must be callable")
    if not 0.0 < holdout < 1.0:
        raise ValueError("holdout must be in (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    reqs = [str(r) for r in requests]
    if len(reqs) < 8:
        raise ValueError("distill_planner needs at least 8 example requests")
    specs = _tool_spec_map(tools, reserved=(_STOP,))

    import numpy as np

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(reqs))
    n_hold = max(2, int(round(len(reqs) * holdout)))
    if n_hold >= len(reqs):
        raise ValueError("holdout leaves no planner training requests")
    hold = [reqs[i] for i in order[:n_hold]]
    train = [reqs[i] for i in order[n_hold:]]

    traces = [(request, _validated_plan(teacher(request), specs, max_steps)) for request in train]

    # flatten traces into (context -> next tool) and (context -> args) supervision
    contexts: list[str] = []
    next_tool: dict[str, str] = {}
    per_tool_rows: dict[str, list[tuple[str, dict]]] = {name: [] for name in specs}
    for r, plan in traces:
        for i in range(len(plan) + 1):
            ctx = _render(r, plan[:i])
            contexts.append(ctx)
            target = plan[i]["tool"] if i < len(plan) else _STOP
            if ctx in next_tool and next_tool[ctx] != target:
                raise ValueError("duplicate planner context has conflicting next-tool supervision")
            if i < len(plan):
                next_tool[ctx] = target
                per_tool_rows[plan[i]["tool"]].append((ctx, dict(plan[i].get("args") or {})))
            else:
                next_tool[ctx] = target

    def select_teacher(c: Any) -> Any:
        if isinstance(c, list):
            return [select_teacher(x) for x in c]
        return next_tool[c]

    selector = solve(select_teacher, contexts, seed=seed, **(selector_kw or {}))

    extractors: dict[str, Any] = {}
    for name, rows in per_tool_rows.items():
        if len(rows) < 8 or not specs[name].args:
            continue

        def make_arg_teacher(table: dict) -> Callable[[Any], Any]:
            def arg_teacher(text: Any) -> Any:
                if isinstance(text, list):
                    return [arg_teacher(t) for t in text]
                return dict(table[text])

            return arg_teacher

        table: dict[str, dict[str, Any]] = {}
        for context, arguments in rows:
            if context in table and table[context] != arguments:
                raise ValueError(f"duplicate planner context has conflicting arguments for tool {name!r}")
            table[context] = arguments
        extractors[name] = distill_extractor(
            make_arg_teacher(table), list(table), specs[name].args, seed=seed, **(extractor_kw or {})
        )

    planner = Planner(
        selector=selector,
        extractors=extractors,
        tools=specs,
        teacher=teacher,
        plan_agreement=float("nan"),
        max_steps=int(max_steps),
    )

    # plan-level holdout verification (students never saw these requests)
    agree = 0
    for r in hold:
        want = _validated_plan(teacher(r), specs, max_steps)
        got = planner.try_plan(r)
        ok = got is not None and got["plan"] == want
        if ok and behavior_verifier is not None:
            ok = bool(behavior_verifier(r, got["plan"], want))
        agree += int(ok)
    planner.plan_agreement = agree / len(hold)
    planner.n_requests = 0  # verification calls don't count as live traffic
    planner.n_escalated = 0
    planner.harvested.clear()
    return planner
