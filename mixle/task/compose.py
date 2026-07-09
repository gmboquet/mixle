"""Compose two models or teachers while carrying a per-stage ledger.

``compose`` chains two models/teachers ``a: x -> y`` and ``b: y -> z`` into one
callable ``x -> z`` while preserving a per-stage evidence ledger that can be
reused by composition and belief-walk workflows.

``a`` and ``b`` are any callables -- a :class:`~mixle.task.model.TaskModel`, a
:class:`~mixle.task.calibrate.CalibratedTaskModel`, a teacher LLM, or a plain function. A stage that
additionally exposes ``.score(input) -> float`` (a log-confidence or log-density) contributes that
number to the ledger; a stage without one contributes ``0.0`` ("unscored", never fabricated).
The ledger is purely additive by construction -- ``ComposedAnswer.check()`` asserts
``sum(contributions) == total_contribution`` exactly, the same identity
:mod:`mixle.inference.explain` uses for a single model's evidence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class ComposedAnswer:
    """A composed ``x -> z`` answer plus the per-stage receipt that attributes it to both stages."""

    answer: Any
    intermediate: Any
    stages: list[tuple[str, Any, float]]  # (stage_name, stage_output, contribution)
    total_contribution: float

    def check(self, tol: float = 1e-9) -> bool:
        """``sum(contributions) == total_contribution`` -- the ledger is exact by construction."""
        return abs(sum(c for _, _, c in self.stages) - self.total_contribution) <= tol


def _stage_contribution(stage: Callable[..., Any], stage_input: Any) -> float:
    if hasattr(stage, "score"):
        return float(stage.score(stage_input))
    if hasattr(stage, "confidence"):
        return float(stage.confidence(stage_input))
    return 0.0


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
    ) -> None:
        self.a = a
        self.b = b
        self.name_a = str(name_a)
        self.name_b = str(name_b)

    def __call__(self, x: Any) -> Any:
        return self.b(self.a(x))

    def answer(self, x: Any) -> ComposedAnswer:
        """Return the composed answer with each stage's contribution record."""
        y = self.a(x)
        z = self.b(y)
        contribution_a = _stage_contribution(self.a, x)
        contribution_b = _stage_contribution(self.b, y)
        stages = [(self.name_a, y, contribution_a), (self.name_b, z, contribution_b)]
        return ComposedAnswer(
            answer=z,
            intermediate=y,
            stages=stages,
            total_contribution=contribution_a + contribution_b,
        )


def compose(
    a: Callable[[Any], Any],
    b: Callable[[Any], Any],
    *,
    name_a: str = "stage_a",
    name_b: str = "stage_b",
) -> ComposedModel:
    """Chain ``a: x -> y`` and ``b: y -> z`` into one ledger-carrying ``x -> z`` callable."""
    return ComposedModel(a, b, name_a=name_a, name_b=name_b)
