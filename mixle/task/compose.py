"""``compose`` -- chain two models/teachers into one ledger-carrying callable (CARD COMPOSE-a).

Neither stage answers the composite question alone: stage ``a`` maps ``x -> y``, stage ``b`` maps
``y -> z``; only the composition answers ``x -> z``. ``ComposedModel`` is that chain as one callable,
and its ``answer(x)`` emits a receipt attributing the final answer to both stages' own
input/output -- the workstream-H discipline applied to a composition of arbitrary callables (not just
fitted mixle distributions, which is what :func:`mixle.inference.explain.explain` decomposes).

    pipeline = compose(classify_reading, recommend_action, name_a="classify", name_b="recommend")
    answer, receipt = pipeline.answer(87.0)
    receipt["stages"]     # [{"name": "classify", "input": 87.0, "output": "high"},
                           #  {"name": "recommend", "input": "high", "output": "escalate"}]

This is the substrate AMPLIFY-a's "2-teacher composition" chains, and (extended to more than two
stages) the belief-walk (F3) transport chain: a multi-hop path is a longer composition of the same
shape, each hop's own input/output named in the receipt.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class ComposedModel:
    """Two callables chained ``x -> y -> z``; ``answer`` returns ``z`` plus a per-stage receipt."""

    stage_a: Callable[[Any], Any]
    stage_b: Callable[[Any], Any]
    name_a: str = "stage_a"
    name_b: str = "stage_b"

    def __call__(self, x: Any) -> Any:
        return self.answer(x)[0]

    def answer(self, x: Any) -> tuple[Any, dict[str, Any]]:
        """Run both stages on ``x``; return ``(z, receipt)`` with the receipt naming each stage's own
        input/output -- an H-style attribution for a plain callable composition, not just a fitted
        distribution's margin."""
        y = self.stage_a(x)
        z = self.stage_b(y)
        receipt = {
            "answer": z,
            "stages": [
                {"name": self.name_a, "input": x, "output": y},
                {"name": self.name_b, "input": y, "output": z},
            ],
        }
        return z, receipt


def compose(
    a: Callable[[Any], Any],
    b: Callable[[Any], Any],
    *,
    name_a: str = "stage_a",
    name_b: str = "stage_b",
) -> ComposedModel:
    """Chain ``a: x -> y`` and ``b: y -> z`` into one callable ``ComposedModel`` answering ``x -> z``."""
    return ComposedModel(stage_a=a, stage_b=b, name_a=name_a, name_b=name_b)


__all__ = ["ComposedModel", "compose"]
