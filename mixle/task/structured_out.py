"""``solve_structured`` -- replace rigid code that returns a dict, field by field.

The structured-output shape of the solve loop: ``teacher(x) -> {"field": value, ...}`` with a consistent
schema (an enricher, a triager, a quote builder). Rather than inventing new machinery, each output field
decomposes onto the shape that already carries guarantees:

  * a categorical/string field -> a :func:`~mixle.task.solve.solve` classifier (conformal singleton);
  * a numeric field -> a :func:`~mixle.task.regress.solve_regression` student (conformal interval +
    the caller's ``tol`` precision rule -- required per numeric field).

The composition rule is strict: the input is answered locally only when **every** field's sub-solution
answers locally; one unsure field escalates the whole request to the teacher (harvested), so a
locally-returned dict never contains a guessed field. The teacher is called exactly once per training
example -- per-field sub-teachers are lookups over that single pass.

``improve()`` pushes each harvested ``(input, dict)`` down into every field's own harvest buffer and
runs each sub-solution's anti-regression improve. No structured-level OOD gate yet (the classifier
fields' own gates are off here to avoid redundant vetoes) -- noted, not hidden.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from mixle.task.regress import RegressionSolution, solve_regression
from mixle.task.solve import Solution, _label_with, solve


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


@dataclass
class StructuredSolution:
    """Per-field calibrated students in front of the dict-valued routine they replace."""

    fields_cat: dict[str, Solution]
    fields_num: dict[str, RegressionSolution]
    teacher: Callable[..., Any]
    n_requests: int = 0
    n_escalated: int = 0
    harvested_inputs: list = field(default_factory=list)
    harvested_outputs: list = field(default_factory=list)

    @property
    def schema(self) -> dict[str, str]:
        """Return each output field's inferred kind: ``categorical`` or ``numeric``."""
        return {**{k: "categorical" for k in self.fields_cat}, **{k: "numeric" for k in self.fields_num}}

    def try_local(self, x: Any) -> dict[str, Any] | None:
        """The fully-decided output dict, or ``None`` when ANY field is unsure (= must escalate)."""
        out: dict[str, Any] = {}
        for key, sub in self.fields_cat.items():
            label = sub.cascade.model.decide(x)
            if label is None:
                return None
            out[key] = label
        for key, sub in self.fields_num.items():
            if not sub.answers_locally:
                return None
            out[key] = float(sub._predict([x])[0])
        return out

    def decide(self, x: Any) -> dict[str, Any] | None:
        """Return the local structured-output decision, or ``None`` when the example should escalate."""
        return self.try_local(x)

    def __call__(self, x: Any) -> dict[str, Any]:
        """Answer locally when every field is confident; otherwise call and harvest the teacher output."""
        self.n_requests += 1
        local = self.try_local(x)
        if local is not None:
            return local
        self.n_escalated += 1
        got = dict(_label_with(self.teacher, [x])[0])
        self.harvested_inputs.append(x)
        self.harvested_outputs.append(got)
        return got

    def report(self) -> dict[str, Any]:
        """Return per-field calibration details and aggregate serving/harvest counts."""
        per_field: dict[str, Any] = {}
        for key, sub in self.fields_cat.items():
            per_field[key] = {"kind": "categorical", "holdout_agreement": round(sub.holdout_agreement, 4)}
        for key, sub in self.fields_num.items():
            per_field[key] = {"kind": "numeric", "qhat": round(float(sub.qhat), 6), "tol": sub.tol}
        return {
            "fields": per_field,
            "requests": self.n_requests,
            "escalated": self.n_escalated,
            "escalation_rate": (self.n_escalated / self.n_requests) if self.n_requests else 0.0,
            "harvested": len(self.harvested_outputs),
        }

    def save(self, path: str) -> str:
        """Persist every field's sub-artifact under one directory; :meth:`load` restores the whole schema."""
        import json
        from pathlib import Path

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        for key, sub in self.fields_cat.items():
            sub.save(str(out / "cat" / key))
        for key, sub in self.fields_num.items():
            sub.save(str(out / "num" / key))
        (out / "structured.json").write_text(
            json.dumps({"kind": "structured/v1", "cat": sorted(self.fields_cat), "num": sorted(self.fields_num)})
        )
        return str(out)

    @classmethod
    def load(cls, path: str, teacher: Callable[..., Any], *, device: str = "cpu") -> StructuredSolution:
        """Reconstitute a serving StructuredSolution (fields serve locally; escalation runs ``teacher``)."""
        import json
        from pathlib import Path

        from mixle.task.regress import RegressionSolution as _RS
        from mixle.task.solve import Solution as _S

        p = Path(path)
        manifest = json.loads((p / "structured.json").read_text())

        def _never(*_a: Any, **_k: Any) -> Any:  # sub-teachers are never consulted on the serving path
            raise RuntimeError("structured sub-fields serve locally; escalation goes through the parent teacher")

        fields_cat = {k: _S.load(str(p / "cat" / k), _never, device=device) for k in manifest["cat"]}
        fields_num = {k: _RS.load(str(p / "num" / k), _never, device=device) for k in manifest["num"]}
        return cls(fields_cat=fields_cat, fields_num=fields_num, teacher=teacher)

    def improve(self) -> bool:
        """Push the harvested dicts down into every field's buffer; each sub improves anti-regressively."""
        if not self.harvested_inputs:
            return False
        promoted = False
        for key, sub in self.fields_cat.items():
            sub.cascade.stats.escalated_texts.extend(self.harvested_inputs)
            sub.cascade.stats.escalated_labels.extend(str(o.get(key)) for o in self.harvested_outputs)
            promoted = bool(sub.improve()) or promoted
        for key, sub in self.fields_num.items():
            sub.harvested_inputs.extend(self.harvested_inputs)
            sub.harvested_ys.extend(float(o.get(key)) for o in self.harvested_outputs)
            promoted = bool(sub.improve()) or promoted
        self.harvested_inputs.clear()
        self.harvested_outputs.clear()
        return promoted


def solve_structured(
    teacher: Callable[..., Any],
    inputs: Sequence[Any],
    *,
    tol: dict[str, float] | float | None = None,
    alpha: float = 0.1,
    prelabeled: tuple[Sequence[Any], Sequence[dict]] | None = None,
    seed: int = 0,
    **sub_kw: Any,
) -> StructuredSolution:
    """Replace a dict-valued routine with per-field calibrated students (see module docstring).

    Args:
        teacher: ``teacher(x) -> dict`` with a consistent schema; called once per example input.
        inputs: example inputs (text or dict/tuple records).
        tol: the precision requirement for numeric fields — a scalar for all, or ``{field: tol}``.
            Required when the schema has numeric fields.
        alpha: shared miscoverage level for every field's calibration.
        prelabeled: already-teacher-labeled ``(inputs, output_dicts)`` — typically harvested
            escalations from a serving deployment — fanned down into every field's TRAINING split
            only, never calibration (each sub-solution's guarantee stays a fresh split of ``inputs``).
            The schema stays authoritative from the ``inputs`` pass; a pair missing a field is simply
            skipped for that field.
        **sub_kw: knobs forwarded to every sub-solve (``epochs``, ``hidden``, ``dim``, …).
    """
    items = list(inputs)
    if len(items) < 12:
        raise ValueError("solve_structured needs at least 12 example inputs")
    outs = [dict(o) for o in _label_with(teacher, items)]
    keys = sorted({k for o in outs for k in o})
    if not keys:
        raise ValueError("the teacher produced empty dicts on the example inputs")

    pre_in: list = []
    pre_outs: list[dict] = []
    if prelabeled is not None:
        pre_in = list(prelabeled[0])
        pre_outs = [dict(o) for o in prelabeled[1]]
        if len(pre_in) != len(pre_outs):
            raise ValueError("prelabeled inputs and output dicts must have equal length")

    numeric = {k for k in keys if all(_is_number(o.get(k)) for o in outs)}
    if numeric and tol is None:
        raise ValueError(f"numeric output fields {sorted(numeric)} need tol= (a scalar or per-field dict)")
    tol_of = (lambda k: float(tol[k])) if isinstance(tol, dict) else (lambda k: float(tol))  # type: ignore[index,arg-type]

    table = {repr(x): o for x, o in zip(items, outs)}

    def _field_pre(key: str, cast: Callable[[Any], Any]) -> tuple[list, list] | None:
        pairs = [(x, cast(o[key])) for x, o in zip(pre_in, pre_outs) if key in o]
        return ([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else None

    fields_cat: dict[str, Solution] = {}
    fields_num: dict[str, RegressionSolution] = {}
    for key in keys:
        if key in numeric:
            fields_num[key] = solve_regression(
                lambda x, _k=key: float(table[repr(x)][_k]),
                items,
                tol=tol_of(key),
                alpha=alpha,
                prelabeled=_field_pre(key, float),
                seed=seed,
                **sub_kw,
            )
        else:
            fields_cat[key] = solve(
                lambda x, _k=key: str(table[repr(x)][_k]),
                items,
                alpha=alpha,
                ood=None,
                prelabeled=_field_pre(key, str),
                seed=seed,
                **sub_kw,
            )
    return StructuredSolution(fields_cat=fields_cat, fields_num=fields_num, teacher=teacher)
