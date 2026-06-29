"""Shared fit-result base for the model-layer experiment helpers.

Several model helpers (``random_graph``, ``knowledge_graph``, ``grammar``,
``partially_observable_markov_decision_process``, ...) report a fit as the same
small ``(model, history)`` record, optionally with a held-out
``validation_history``.  They share this one generic base rather than each
redefining the dataclass, so the common shape lives in a single place while each
module keeps its own public class name (and narrows the ``model`` type via the
generic parameter).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

ModelT = TypeVar("ModelT")


@dataclass
class FitResult(Generic[ModelT]):
    """Fitted ``model`` plus its training log-likelihood ``history``.

    ``validation_history`` holds the held-out log-likelihood trace when a
    validation set was supplied during fitting, else ``None``.
    """

    model: ModelT
    history: list[float]
    validation_history: list[float] | None = None
