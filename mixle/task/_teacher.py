"""Resolve a user-supplied teacher's calling convention once, instead of on every batch.

``solve``, ``solve_regression`` and ``distill_text_generative`` all accept a teacher that may be
*per-item* (``teacher(x) -> label``) or *batched* (``teacher([x, ...]) -> [label, ...]``), and
neither convention is declared anywhere. It therefore has to be discovered by calling -- and a
teacher is typically the expensive thing in the loop: a paid API client, a legacy service, a
stateful process. Discovering it by offering the whole batch and falling back to per-item calls on
any surprise costs a per-item teacher ``len(items) + 1`` invocations, and nothing remembered the
answer, so that surcharge was paid again on every single call. On a demoted :class:`Solution`,
which routes every request to the teacher, that is a permanent 2x on the serving path.

:class:`TeacherCaller` runs that same discovery once and then remembers it. The whole batch is
still what gets offered first, deliberately: some callers pass a teacher that answers only for one
exact batch -- ``structured_out``'s precomputed-label replay is one -- and probing with a slice
would break them. Set a ``batched`` attribute on the callable (or pass ``batched=``) to declare the
convention and skip discovery entirely; worth doing for a teacher whose labels are themselves
tuples or lists, the one shape that discovery cannot tell apart from a batch.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class TeacherCaller:
    """A ``list -> list`` view over a teacher of either calling convention.

    ``batched`` may be given explicitly; otherwise it is read from a ``batched`` attribute on the
    callable, and failing that discovered on first use and kept.
    """

    __slots__ = ("_batched", "teacher")

    def __init__(self, teacher: Callable[..., Any], *, batched: bool | None = None) -> None:
        if batched is None:
            declared = getattr(teacher, "batched", None)
            batched = declared if isinstance(declared, bool) else None
        self.teacher = teacher
        self._batched = batched

    @property
    def batched(self) -> bool | None:
        """The resolved convention, or ``None`` while still undiscovered."""
        return self._batched

    def __call__(self, items: list) -> list:
        """Label every element of ``items``, in order."""
        if not items:
            return []
        return self._labeled(items) if self._batched is not None else self._resolve(items)

    def one(self, x: Any) -> Any:
        """Label a single item."""
        return self([x])[0]

    def _labeled(self, items: list) -> list:
        """Label with the already-resolved convention -- no probing, no silent second attempt.

        Once the convention is known, a raise from the teacher is a teacher failure, not evidence
        about its signature, so it propagates instead of triggering a full per-item retry.
        """
        if self._batched:
            return self._checked(self.teacher(items), items)
        return [self.teacher(x) for x in items]

    def _resolve(self, items: list) -> list:
        """Discover the convention from the first real labeling pass, and remember it."""
        try:
            out = self.teacher(items)
        except Exception:  # noqa: BLE001 - how a teacher rejects a batch shape is not part of any contract
            self._batched = False
            return [self.teacher(x) for x in items]
        if isinstance(out, (list, tuple)):
            # a sequence answer to a batch means batched; a *wrong-length* one is a broken batched
            # teacher rather than a per-item one, so it is reported instead of silently relabeled
            self._batched = True
            return self._checked(out, items)
        self._batched = False
        return [self.teacher(x) for x in items]

    @staticmethod
    def _checked(out: Any, items: list) -> list:
        if not isinstance(out, (list, tuple)) or len(out) != len(items):
            n = len(out) if isinstance(out, (list, tuple)) else type(out).__name__
            raise ValueError(
                f"batched teacher returned {n} for {len(items)} items; it must return exactly one label per item"
            )
        return list(out)


def as_batch_view(teacher: Callable[..., Any]) -> TeacherCaller:
    """A strict ``list -> list`` teacher view (e.g. for cascade escalation)."""
    return teacher if isinstance(teacher, TeacherCaller) else TeacherCaller(teacher)
