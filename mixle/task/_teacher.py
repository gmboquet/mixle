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

TEACHER_MODES = ("auto", "batch", "item")
"""How a teacher wants to be called. ``"auto"`` discovers it; the others declare it."""


def _teacher_name(teacher: Any) -> str:
    return getattr(teacher, "__name__", None) or type(teacher).__name__


def batched_from_mode(teacher_mode: str) -> bool | None:
    """Map a public ``teacher_mode`` onto :class:`TeacherCaller`'s ``batched`` flag."""
    if teacher_mode not in TEACHER_MODES:
        raise ValueError(f"teacher_mode must be one of {TEACHER_MODES}, got {teacher_mode!r}")
    return None if teacher_mode == "auto" else teacher_mode == "batch"


class TeacherCaller:
    """A ``list -> list`` view over a teacher of either calling convention.

    ``batched`` may be given explicitly; otherwise it is read from a ``batched`` attribute on the
    callable, and failing that discovered on first use and kept.
    """

    __slots__ = ("_batched", "last_probe_error", "teacher")

    def __init__(self, teacher: Callable[..., Any], *, batched: bool | None = None) -> None:
        if batched is None:
            declared = getattr(teacher, "batched", None)
            batched = declared if isinstance(declared, bool) else None
        self.teacher = teacher
        self._batched = batched
        # Set here so it is always readable, not only after a probe failure has happened to occur.
        self.last_probe_error: BaseException | None = None

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
        """Discover the convention from the first real labeling pass, and remember it.

        Every exception is absorbed here, and that is a deliberate choice rather than an oversight.
        A per-item callable handed a list raises whatever its own body raises -- ``TypeError`` from
        a bad index, ``ValueError`` from a shape check, ``KeyError`` from a lookup -- and real
        teachers in this repository do all three. There is no exception type that separates "does
        not accept a list" from "ran and failed"; I tried ``TypeError`` only, and then boundary-only
        ``TypeError``, and each broke a different set of legitimate teachers (30 tests, then 2 more).

        What the breadth costs is real and is the reason ``teacher_mode`` exists: a teacher whose
        backend is down looks exactly like a per-item teacher, so it gets called ``len(items)`` more
        times. Declaring ``teacher_mode="batch"`` or ``"item"`` skips this path entirely and is the
        right answer for anything metered or stateful. When the per-item retry fails too, the
        discarded probe error is attached as a note so the first failure is never simply lost.
        """
        try:
            out = self.teacher(items)
        except Exception as probe_error:  # noqa: BLE001 - see the docstring: no type discriminates here
            self._batched = False
            try:
                resolved = [self.teacher(x) for x in items]
            except Exception as exc:
                exc.add_note(
                    f"the batch-call probe on {_teacher_name(self.teacher)} first failed with: "
                    f"{probe_error!r} -- pass teacher_mode='item' or 'batch' to skip this discovery"
                )
                raise
            # A retry that SUCCEEDS used to return with the probe error discarded, which is how
            # MXR-080-0686 hid: a batch teacher whose body raised (a backend down, say) was
            # indistinguishable from a genuine per-item teacher, so the values came back as if the
            # first failure had not happened. The two cases really are indistinguishable -- a per-item
            # lambda handed a list raises just as readily -- so this cannot warn without firing on every
            # ordinary per-item teacher, which was measured and is why there is no warning here. What it
            # can do is stop throwing the evidence away: the probe error is retained on the caller for a
            # supervisor or a telemetry pass to read. teacher_mode='batch' remains the way to make a
            # failure inside a batch teacher raise instead of being rediscovered as a convention.
            self.last_probe_error = probe_error
            return resolved
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


def as_batch_view(teacher: Callable[..., Any], teacher_mode: str = "auto") -> TeacherCaller:
    """A strict ``list -> list`` teacher view (e.g. for cascade escalation).

    ``teacher_mode`` declares the calling convention -- ``"batch"``, ``"item"``, or ``"auto"``
    to discover it. Declaring skips discovery entirely, which is what a teacher that keeps
    state, costs money per call, or rejects a batch from inside its own body should do.
    """
    if isinstance(teacher, TeacherCaller):
        return teacher
    return TeacherCaller(teacher, batched=batched_from_mode(teacher_mode))
