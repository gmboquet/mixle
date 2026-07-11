"""Transactional snapshots for mutable model state used inside EM updates.

Most mixle distributions are immutable value objects: an M-step returns a new
distribution and leaves the previous iterate untouched.  Torch-backed leaves are
the important exception.  Their estimators update a shared ``nn.Module`` in
place, so an objective gate needs to snapshot that state before proposing a step
and restore it when the proposal is rejected.

The helpers here deliberately recognize the small ``state_dict`` /
``load_state_dict`` protocol instead of importing torch.  This keeps torch an
optional dependency and also works for compatible stateful modules.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


def _is_stateful_module(obj: Any) -> bool:
    return (
        callable(getattr(obj, "state_dict", None))
        and callable(getattr(obj, "load_state_dict", None))
        and callable(getattr(obj, "parameters", None))
    )


def _children(obj: Any):
    if isinstance(obj, dict):
        yield from obj.keys()
        yield from obj.values()
    elif isinstance(obj, (list, tuple, set, frozenset)):
        yield from obj
    elif hasattr(obj, "__dict__"):
        yield from vars(obj).values()


def stateful_modules(*roots: Any) -> tuple[Any, ...]:
    """Return unique mutable modules reachable from ``roots``.

    Traversal stops at a module boundary.  Walking through a torch module's own
    object graph would be both expensive and redundant because ``state_dict`` is
    already the module's complete transactional state.
    """

    found: list[Any] = []
    seen: set[int] = set()
    stack = list(roots)
    while stack:
        obj = stack.pop()
        if obj is None or isinstance(obj, (str, bytes, bytearray, int, float, complex, bool)):
            continue
        ident = id(obj)
        if ident in seen:
            continue
        seen.add(ident)
        if _is_stateful_module(obj):
            found.append(obj)
            continue
        stack.extend(_children(obj))
    return tuple(found)


def has_mutable_state(*roots: Any) -> bool:
    """Whether any torch-like mutable module is reachable from ``roots``."""

    return bool(stateful_modules(*roots))


def _clone_state(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    clone = getattr(value, "clone", None)
    if callable(detach) and callable(clone):
        return detach().clone()
    return copy.deepcopy(value)


@dataclass
class MutableStateSnapshot:
    """Restorable state for all mutable modules reachable from a model tree."""

    entries: tuple[tuple[Any, dict[str, Any], bool | None], ...]

    @classmethod
    def capture(cls, *roots: Any) -> MutableStateSnapshot:
        entries = []
        for module in stateful_modules(*roots):
            state = {key: _clone_state(value) for key, value in module.state_dict().items()}
            entries.append((module, state, getattr(module, "training", None)))
        return cls(tuple(entries))

    @property
    def active(self) -> bool:
        return bool(self.entries)

    def restore(self) -> None:
        for module, state, training in self.entries:
            module.load_state_dict(state)
            if training is not None and callable(getattr(module, "train", None)):
                module.train(training)
