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
from mixle.task.toolcall import ToolSpec

_STOP = "__stop__"


def _render(request: str, steps: Sequence[dict]) -> str:
    """The context the students read: the request plus the steps taken so far."""
    if not steps:
        return f"{request} [plan so far: none]"
    done = "; ".join(f"{s['tool']}({', '.join(f'{k}={v}' for k, v in (s.get('args') or {}).items())})" for s in steps)
    return f"{request} [plan so far: {done}]"


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

    def try_plan(self, request: str, *, execute: dict[str, Callable[..., Any]] | None = None) -> dict[str, Any] | None:
        """The local decomposition alone: a complete verified plan, or ``None`` (= must escalate).

        This method does not call the teacher."""
        steps: list[dict] = []
        results: list[Any] = []
        for _ in range(self.max_steps):
            ctx = _render(request, steps)
            tool = self.selector.cascade.model.decide(ctx)
            if tool == _STOP:
                return {"plan": steps, "results": results}
            if tool is None or tool not in self.extractors:
                return None  # unsure which step comes next
            args = self.extractors[tool](ctx)
            spec = self.tools[tool]
            if not all(args.get(a) for a in spec.required_args):
                return None  # cannot fill the step's required arguments
            step = {"tool": tool, "args": {k: v for k, v in args.items() if k in spec.args}}
            if execute is not None:
                try:
                    results.append(execute[tool](**step["args"]))
                except Exception:  # noqa: BLE001 - a failing step is exactly what must escalate
                    return None
            steps.append(step)
        return None  # max_steps without STOP

    def __call__(self, request: str, *, execute: dict[str, Callable[..., Any]] | None = None) -> dict[str, Any]:
        """Plan (and optionally execute) step by step; any uncertain/malformed/failing step escalates."""
        self.n_requests += 1
        local = self.try_plan(request, execute=execute)
        if local is not None:
            return {**local, "escalate": False}
        self.n_escalated += 1
        plan = self.teacher(request)
        self.harvested.append((request, plan))
        out: dict[str, Any] = {"plan": [dict(p) for p in plan], "escalate": True}
        if execute is not None:
            out["results"] = [execute[p["tool"]](**(p.get("args") or {})) for p in plan]
        return out

    def report(self) -> dict[str, Any]:
        """Return plan agreement, escalation, and harvested-trace metrics."""
        return {
            "plan_agreement": round(self.plan_agreement, 4),
            "requests": self.n_requests,
            "escalated": self.n_escalated,
            "escalation_rate": (self.n_escalated / self.n_requests) if self.n_requests else 0.0,
            "harvested_traces": len(self.harvested),
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
        tools = {n: ToolSpec(n, list(t["args"]), t.get("required")) for n, t in manifest["tools"].items()}
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
) -> Planner:
    """Distill the teacher's multi-step plans into next-step students (see module docstring).

    Plan-level verification is measured on held-out requests the students never trained on: a plan
    agrees when every step's tool and required arguments match the teacher's plan exactly, in order.
    """
    reqs = [str(r) for r in requests]
    if len(reqs) < 8:
        raise ValueError("distill_planner needs at least 8 example requests")
    specs = {t.name: t for t in tools}

    import numpy as np

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(reqs))
    n_hold = max(2, int(round(len(reqs) * holdout)))
    hold = [reqs[i] for i in order[:n_hold]]
    train = [reqs[i] for i in order[n_hold:]]

    traces = {r: list(teacher(r)) for r in train}
    for plan in traces.values():
        for step in plan:
            if step.get("tool") not in specs:
                raise ValueError(f"teacher plan uses tool {step.get('tool')!r} not in the provided specs")

    # flatten traces into (context -> next tool) and (context -> args) supervision
    contexts: list[str] = []
    next_tool: dict[str, str] = {}
    per_tool_rows: dict[str, list[tuple[str, dict]]] = {name: [] for name in specs}
    for r, plan in traces.items():
        for i in range(len(plan) + 1):
            ctx = _render(r, plan[:i])
            contexts.append(ctx)
            if i < len(plan):
                next_tool[ctx] = plan[i]["tool"]
                per_tool_rows[plan[i]["tool"]].append((ctx, dict(plan[i].get("args") or {})))
            else:
                next_tool[ctx] = _STOP

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

        extractors[name] = distill_extractor(
            make_arg_teacher(dict(rows)), [c for c, _ in rows], specs[name].args, seed=seed, **(extractor_kw or {})
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
        want = list(teacher(r))
        got = planner(r)
        ok = (not got["escalate"]) and len(got["plan"]) == len(want)
        if ok:
            for g, w in zip(got["plan"], want):
                spec = specs[w["tool"]]
                if g["tool"] != w["tool"] or any(
                    str(g["args"].get(a)) != str((w.get("args") or {}).get(a)) for a in spec.required_args
                ):
                    ok = False
                    break
        agree += int(ok)
    planner.plan_agreement = agree / len(hold)
    planner.n_requests = 0  # verification calls don't count as live traffic
    planner.n_escalated = 0
    planner.harvested.clear()
    return planner
