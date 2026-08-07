"""``solve_structured`` -- replace rigid code that returns a dict with one joint coverage contract.

The structured-output shape of the solve loop: ``teacher(x) -> {"field": value, ...}`` with a consistent
schema (an enricher, a triager, a quote builder). Rather than inventing new machinery, each output field
decomposes onto the shape that already carries guarantees:

  * a categorical/string field -> a :func:`~mixle.task.solve.solve` classifier (conformal singleton);
  * a numeric field -> a :func:`~mixle.task.regress.solve_regression` student (conformal interval +
    the caller's ``tol`` precision rule -- required per numeric field).

The composition rule is strict: a separate held-out slice calibrates the maximum nonconformity across
all fields. The input is answered locally only when every categorical prediction set is a singleton and
every numeric interval meets its tolerance. Thus the complete record, rather than each field
marginally, has the advertised ``1 - alpha`` split-conformal coverage contract.

**Scope of that statement.** The joint ``1 - alpha`` coverage is finite-sample and MARGINAL over the
calibration draw and the query jointly, under exchangeability of the calibration rows and incoming
traffic. It is NOT an accuracy guarantee conditional on answering locally: serving conditions on every
field simultaneously clearing its singleton/tolerance gate, and coverage conditional on that event is
not controlled -- answered-slice quality is a measurement (the per-field ``holdout_agreement`` values
in ``report()``), never a guarantee. Distribution shift or any other break of exchangeability voids
the statement silently; re-measure on drifted traffic. ``report()`` carries this scope
machine-readably in ``coverage_contract_scope``.

``improve()`` pushes each harvested ``(input, dict)`` down into every field's own harvest buffer and
runs each sub-solution's anti-regression improve. No structured-level OOD gate yet (the classifier
fields' own gates are off here to avoid redundant vetoes) -- noted, not hidden.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

import numpy as np

from mixle.task._ledger import conformal_scope
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
    joint_qhat: float = float("inf")
    alpha: float = 0.1
    numeric_tolerances: dict[str, float] = field(default_factory=dict)
    calibration_receipt: dict[str, Any] = field(default_factory=dict)
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
            labels = list(sub.cascade.model.task.adapter.labels)
            probabilities = np.asarray(
                sub.cascade.model.task.adapter.proba_batch(sub.cascade.model.task.model, [x]), dtype=np.float64
            )
            if (
                probabilities.shape != (1, len(labels))
                or not np.all(np.isfinite(probabilities))
                or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
            ):
                raise ValueError(f"categorical field {key!r} returned invalid probabilities")
            admissible = [label for label, p in zip(labels, probabilities[0]) if 1.0 - p <= self.joint_qhat]
            if len(admissible) != 1:
                return None
            out[key] = admissible[0]
        for key, sub in self.fields_num.items():
            tolerance = self.numeric_tolerances[key]
            if not np.isfinite(self.joint_qhat) or self.joint_qhat * tolerance > tolerance:
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
        got = _validated_outputs(_label_with(self.teacher, [x]), self.schema)[0]
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
            "coverage_contract": "joint_structured",
            "coverage_contract_scope": conformal_scope(
                "finite-sample marginal joint coverage of the complete output record"
            ),
            "joint_qhat": self.joint_qhat,
            "alpha": self.alpha,
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
            json.dumps(
                {
                    "kind": "structured/v2",
                    "cat": sorted(self.fields_cat),
                    "num": sorted(self.fields_num),
                    "joint_qhat": self.joint_qhat,
                    "alpha": self.alpha,
                    "numeric_tolerances": self.numeric_tolerances,
                    "calibration_receipt": self.calibration_receipt,
                }
            )
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
        return cls(
            fields_cat=fields_cat,
            fields_num=fields_num,
            teacher=teacher,
            joint_qhat=float(manifest.get("joint_qhat", float("inf"))),
            alpha=float(manifest.get("alpha", 0.1)),
            numeric_tolerances={k: float(v) for k, v in manifest.get("numeric_tolerances", {}).items()},
            calibration_receipt=dict(manifest.get("calibration_receipt", {})),
        )

    def improve(self) -> bool:
        """Refuse unsafe adaptive reuse of the joint calibration slice.

        Re-solve with the harvested rows as ``prelabeled=`` and a fresh base sample. That path creates a
        new untouched joint calibration slice; silently recycling the old one would void coverage.
        """
        if not self.harvested_inputs:
            return False
        raise RuntimeError(
            "structured improvement requires re-solving with fresh base inputs and "
            "prelabeled=(harvested_inputs, harvested_outputs)"
        )


def _validated_outputs(values: Sequence[Any], schema: dict[str, str]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    expected = set(schema)
    for row in values:
        if not isinstance(row, dict):
            raise ValueError(f"structured teacher rows must be dictionaries, got {type(row).__name__}")
        if set(row) != expected:
            raise ValueError(f"structured output schema mismatch: expected {sorted(expected)}, got {sorted(row)}")
        normalized: dict[str, Any] = {}
        for key, kind in schema.items():
            value = row[key]
            if kind == "numeric":
                if not _is_number(value) or not np.isfinite(value):
                    raise ValueError(f"numeric structured field {key!r} must be finite")
                normalized[key] = float(value)
            else:
                if isinstance(value, (dict, list, tuple, set)):
                    raise ValueError(f"categorical structured field {key!r} must be scalar")
                normalized[key] = str(value)
        outputs.append(normalized)
    return outputs


def _joint_scores(
    fields_cat: dict[str, Solution],
    fields_num: dict[str, RegressionSolution],
    inputs: Sequence[Any],
    outputs: Sequence[dict[str, Any]],
    tolerances: dict[str, float],
) -> np.ndarray:
    scores = np.zeros(len(inputs), dtype=np.float64)
    for key, sub in fields_cat.items():
        labels = list(sub.cascade.model.task.adapter.labels)
        probabilities = np.asarray(
            sub.cascade.model.task.adapter.proba_batch(sub.cascade.model.task.model, list(inputs)), dtype=np.float64
        )
        if probabilities.shape != (len(inputs), len(labels)) or not np.all(np.isfinite(probabilities)):
            raise ValueError(f"categorical field {key!r} returned invalid calibration probabilities")
        index = {label: j for j, label in enumerate(labels)}
        try:
            field_scores = np.asarray([1.0 - probabilities[i, index[str(row[key])]] for i, row in enumerate(outputs)])
        except KeyError as exc:
            raise ValueError(f"joint calibration observed an unknown label for field {key!r}") from exc
        scores = np.maximum(scores, field_scores)
    for key, sub in fields_num.items():
        predictions = sub._predict(list(inputs))
        residual = np.abs(np.asarray([row[key] for row in outputs], dtype=np.float64) - predictions)
        scores = np.maximum(scores, residual / tolerances[key])
    if scores.shape != (len(inputs),) or not np.all(np.isfinite(scores)):
        raise ValueError("joint structured calibration scores must be finite and aligned")
    return scores


def solve_structured(
    teacher: Callable[..., Any],
    inputs: Sequence[Any],
    *,
    schema: dict[str, str] | None = None,
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
    if not callable(teacher):
        raise TypeError("teacher must be callable")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    items = list(inputs)
    if len(items) < 12:
        raise ValueError("solve_structured needs at least 12 example inputs")
    raw_outputs = _label_with(teacher, items)
    if len(raw_outputs) != len(items):
        raise ValueError("teacher must return exactly one structured output per input")
    if schema is None:
        if not raw_outputs or not isinstance(raw_outputs[0], dict) or not raw_outputs[0]:
            raise ValueError("the teacher must produce a nonempty dictionary schema")
        schema = {str(key): "numeric" if _is_number(value) else "categorical" for key, value in raw_outputs[0].items()}
    else:
        schema = dict(schema)
        if not schema or any(not isinstance(key, str) or not key for key in schema):
            raise ValueError("schema keys must be nonempty strings")
        if any(kind not in ("categorical", "numeric") for kind in schema.values()):
            raise ValueError("schema values must be 'categorical' or 'numeric'")
    outs = _validated_outputs(raw_outputs, schema)
    keys = sorted(schema)
    if not keys:
        raise ValueError("the teacher produced empty dicts on the example inputs")

    pre_in: list = []
    pre_outs: list[dict] = []
    if prelabeled is not None:
        pre_in = list(prelabeled[0])
        pre_outs = _validated_outputs(prelabeled[1], schema)
        if len(pre_in) != len(pre_outs):
            raise ValueError("prelabeled inputs and output dicts must have equal length")

    numeric = {key for key, field_kind in schema.items() if field_kind == "numeric"}
    if numeric and tol is None:
        raise ValueError(f"numeric output fields {sorted(numeric)} need tol= (a scalar or per-field dict)")
    if isinstance(tol, dict):
        missing = numeric - set(tol)
        if missing:
            raise ValueError(f"tol is missing numeric fields {sorted(missing)}")
        numeric_tolerances = {key: float(tol[key]) for key in numeric}
    else:
        numeric_tolerances = {key: float(tol) for key in numeric}  # type: ignore[arg-type]
    if any(not np.isfinite(value) or value <= 0.0 for value in numeric_tolerances.values()):
        raise ValueError("numeric structured tolerances must be finite and positive")

    order = np.random.RandomState(seed).permutation(len(items))
    n_joint_cal = int(np.ceil(1.0 / alpha)) - 1
    if n_joint_cal < 1 or len(items) - n_joint_cal < 12:
        raise ValueError(
            f"solve_structured needs at least {n_joint_cal + 12} inputs for joint calibration and sub-model fitting"
        )
    joint_idx, sub_idx = order[:n_joint_cal], order[n_joint_cal:]
    joint_inputs = [items[i] for i in joint_idx]
    joint_outputs = [outs[i] for i in joint_idx]
    sub_inputs = [items[i] for i in sub_idx]
    sub_outputs = [outs[i] for i in sub_idx]

    def _field_pre(key: str, cast: Callable[[Any], Any]) -> tuple[list, list] | None:
        pairs = [(x, cast(o[key])) for x, o in zip(pre_in, pre_outs)]
        return ([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else None

    def _known_teacher(values: list[Any]) -> Callable[[Any], Any]:
        def lookup(batch: Any) -> list[Any]:
            if isinstance(batch, list) and len(batch) == len(values):
                return list(values)
            raise ValueError("structured field teacher is defined only for the preserved training rows")

        return lookup

    fields_cat: dict[str, Solution] = {}
    fields_num: dict[str, RegressionSolution] = {}
    for key in keys:
        if key in numeric:
            values = [float(output[key]) for output in sub_outputs]
            fields_num[key] = solve_regression(
                _known_teacher(values),
                sub_inputs,
                tol=numeric_tolerances[key],
                alpha=alpha,
                prelabeled=_field_pre(key, float),
                seed=seed,
                **sub_kw,
            )
        else:
            values = [str(output[key]) for output in sub_outputs]
            fields_cat[key] = solve(
                _known_teacher(values),
                sub_inputs,
                alpha=alpha,
                ood=None,
                prelabeled=_field_pre(key, str),
                seed=seed,
                **sub_kw,
            )
    joint_scores = _joint_scores(fields_cat, fields_num, joint_inputs, joint_outputs, numeric_tolerances)
    rank = int(np.ceil((len(joint_scores) + 1) * (1.0 - alpha)))
    if rank < 1 or rank > len(joint_scores):
        raise ValueError("joint structured calibration slice is too small for the requested alpha")
    joint_qhat = float(np.sort(joint_scores)[rank - 1])
    return StructuredSolution(
        fields_cat=fields_cat,
        fields_num=fields_num,
        teacher=teacher,
        joint_qhat=joint_qhat,
        alpha=float(alpha),
        numeric_tolerances=numeric_tolerances,
        calibration_receipt={
            "contract": "joint_structured",
            "calibration_count": len(joint_inputs),
            "calibration_indices": [int(i) for i in joint_idx],
            "calibration_sha256": sha256(repr(list(zip(joint_inputs, joint_outputs))).encode("utf-8")).hexdigest(),
        },
    )
