"""Compose two models or teachers while carrying a typed per-stage evidence ledger.

``compose`` chains two models/teachers ``a: x -> y`` and ``b: y -> z`` into one
callable ``x -> z`` while preserving a per-stage evidence ledger that can be
reused by composition and belief-walk workflows.

``a`` and ``b`` are any callables -- a :class:`~mixle.task.model.TaskModel`, a
:class:`~mixle.task.calibrate.CalibratedTaskModel`, a teacher LLM, or a plain function. A stage that
additionally exposes ``.score(input) -> float`` (a log-confidence or log-density) contributes that
number only when the caller declares its statistical type. Unrelated units are never added: a
combination rule is explicit and rejects incompatible evidence kinds.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

_EVIDENCE_KINDS = {"log_evidence", "probability", "confidence", "reward", "unscored"}
_COMBINATION_RULES = {"none", "sum_log_evidence", "product_probability"}


@dataclass(frozen=True)
class EvidenceValue:
    """One typed scalar. Its kind determines which composition rules may consume it."""

    kind: str
    value: float | None

    def __post_init__(self) -> None:
        if self.kind not in _EVIDENCE_KINDS:
            raise ValueError(f"unsupported evidence kind {self.kind!r}")
        if self.kind == "unscored":
            if self.value is not None:
                raise ValueError("unscored evidence must have value=None")
            return
        if self.value is None or not math.isfinite(self.value):
            raise ValueError("scored evidence must be finite")
        if self.kind in ("probability", "confidence") and not 0.0 <= self.value <= 1.0:
            raise ValueError(f"{self.kind} evidence must be in [0, 1]")


@dataclass(frozen=True)
class StageReceipt:
    name: str
    output: Any
    evidence: EvidenceValue


def _combine(evidence: list[EvidenceValue], rule: str) -> float | None:
    if rule not in _COMBINATION_RULES:
        raise ValueError(f"unsupported evidence combination rule {rule!r}")
    scored = [item for item in evidence if item.kind != "unscored"]
    if rule == "none":
        if scored:
            raise ValueError("combination_rule='none' cannot receive scored evidence")
        return None
    if rule == "sum_log_evidence":
        if any(item.kind != "log_evidence" for item in scored) or len(scored) != len(evidence):
            raise ValueError("sum_log_evidence requires every stage to declare log_evidence")
        return float(sum(item.value for item in scored if item.value is not None))
    if any(item.kind != "probability" for item in scored) or len(scored) != len(evidence):
        raise ValueError("product_probability requires every stage to declare probability")
    return float(math.prod(item.value for item in scored if item.value is not None))


def _receipt_digest(answer: Any, intermediate: Any, stages: list[StageReceipt], rule: str, total: Any) -> str:
    payload = (
        repr(answer),
        repr(intermediate),
        [(stage.name, repr(stage.output), stage.evidence.kind, stage.evidence.value) for stage in stages],
        rule,
        total,
    )
    return sha256(repr(payload).encode("utf-8")).hexdigest()


@dataclass
class ComposedAnswer:
    """A composed ``x -> z`` answer plus the per-stage receipt that attributes it to both stages."""

    answer: Any
    intermediate: Any
    stages: list[StageReceipt]
    combination_rule: str
    combined_evidence: float | None
    receipt_sha256: str

    def check(self, tol: float = 1e-9) -> bool:
        """Validate evidence units, the declared algebra, and the receipt integrity digest."""
        try:
            expected = _combine([stage.evidence for stage in self.stages], self.combination_rule)
        except ValueError:
            return False
        total_matches = (
            expected is None
            and self.combined_evidence is None
            or expected is not None
            and self.combined_evidence is not None
            and abs(expected - self.combined_evidence) <= tol
        )
        digest = _receipt_digest(
            self.answer, self.intermediate, self.stages, self.combination_rule, self.combined_evidence
        )
        return bool(total_matches and digest == self.receipt_sha256)


def _stage_evidence(stage: Callable[..., Any], stage_input: Any, declared_kind: str | None) -> EvidenceValue:
    if declared_kind is None:
        return EvidenceValue("unscored", None)
    if declared_kind not in _EVIDENCE_KINDS - {"unscored"}:
        raise ValueError(f"unsupported declared evidence kind {declared_kind!r}")
    if hasattr(stage, "evidence"):
        raw = stage.evidence(stage_input)
        if isinstance(raw, EvidenceValue):
            if raw.kind != declared_kind:
                raise ValueError(f"stage returned {raw.kind!r}, but composition declares {declared_kind!r}")
            return raw
        value = raw
    elif hasattr(stage, "score"):
        value = stage.score(stage_input)
    elif hasattr(stage, "confidence"):
        value = stage.confidence(stage_input)
    else:
        raise ValueError("a scored composition stage must expose evidence(), score(), or confidence()")
    return EvidenceValue(declared_kind, float(value))


class ComposedModel:
    """Chain ``a: x -> y`` then ``b: y -> z`` as one callable ``x -> z``.

    ``composed(x)`` returns the bare answer ``z`` (so a ``ComposedModel`` can stand in anywhere a plain
    teacher callable is expected -- including as the ``a`` or ``b`` of another ``compose()``, chaining
    further). ``composed.answer(x)`` returns the ledger-carrying :class:`ComposedAnswer` instead.
    """

    def __init__(
        self,
        a: Callable[[Any], Any],
        b: Callable[[Any], Any],
        *,
        name_a: str = "stage_a",
        name_b: str = "stage_b",
        evidence_a: str | None = None,
        evidence_b: str | None = None,
        combination_rule: str = "none",
    ) -> None:
        if not callable(a) or not callable(b):
            raise TypeError("composition stages must be callable")
        self.a = a
        self.b = b
        self.name_a = str(name_a)
        self.name_b = str(name_b)
        if not self.name_a or not self.name_b or self.name_a == self.name_b:
            raise ValueError("composition stage names must be nonempty and distinct")
        self.evidence_a = evidence_a
        self.evidence_b = evidence_b
        self.combination_rule = combination_rule
        # Validate the declared algebra before any stage can execute.
        declared = [
            EvidenceValue(evidence_a, 0.5 if evidence_a in ("probability", "confidence") else 0.0)
            if evidence_a
            else EvidenceValue("unscored", None),
            EvidenceValue(evidence_b, 0.5 if evidence_b in ("probability", "confidence") else 0.0)
            if evidence_b
            else EvidenceValue("unscored", None),
        ]
        _combine(declared, combination_rule)

    def __call__(self, x: Any) -> Any:
        return self.b(self.a(x))

    def answer(self, x: Any) -> ComposedAnswer:
        """Return the composed answer with each stage's contribution record."""
        y = self.a(x)
        z = self.b(y)
        evidence_a = _stage_evidence(self.a, x, self.evidence_a)
        evidence_b = _stage_evidence(self.b, y, self.evidence_b)
        stages = [StageReceipt(self.name_a, y, evidence_a), StageReceipt(self.name_b, z, evidence_b)]
        combined = _combine([evidence_a, evidence_b], self.combination_rule)
        digest = _receipt_digest(z, y, stages, self.combination_rule, combined)
        return ComposedAnswer(
            answer=z,
            intermediate=y,
            stages=stages,
            combination_rule=self.combination_rule,
            combined_evidence=combined,
            receipt_sha256=digest,
        )


def compose(
    a: Callable[[Any], Any],
    b: Callable[[Any], Any],
    *,
    name_a: str = "stage_a",
    name_b: str = "stage_b",
    evidence_a: str | None = None,
    evidence_b: str | None = None,
    combination_rule: str = "none",
) -> ComposedModel:
    """Chain ``a: x -> y`` and ``b: y -> z`` into one ledger-carrying ``x -> z`` callable."""
    return ComposedModel(
        a,
        b,
        name_a=name_a,
        name_b=name_b,
        evidence_a=evidence_a,
        evidence_b=evidence_b,
        combination_rule=combination_rule,
    )


__all__ = ["ComposedAnswer", "ComposedModel", "EvidenceValue", "StageReceipt", "compose"]
