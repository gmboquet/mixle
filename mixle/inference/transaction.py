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
import types
from dataclasses import dataclass
from typing import Any

import numpy as np


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


def _algorithm_children(obj: Any):
    if isinstance(obj, dict):
        yield from obj.keys()
        yield from obj.values()
    elif isinstance(obj, (list, tuple, set, frozenset)):
        yield from obj
    else:
        if hasattr(obj, "__dict__"):
            yield from vars(obj).values()
        for cls in type(obj).__mro__:
            slots = cls.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for name in slots:
                if name in {"__dict__", "__weakref__"}:
                    continue
                attr_name = name
                if name.startswith("__") and not name.endswith("__"):
                    attr_name = f"_{cls.__name__.lstrip('_')}{name}"
                if hasattr(obj, attr_name):
                    yield getattr(obj, attr_name)


def _algorithm_objects(root: Any, preserved: set[int]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    objects: list[Any] = []
    rngs: list[Any] = []
    seen: set[int] = set()
    stack = [root]
    atomic = (
        str,
        bytes,
        bytearray,
        int,
        float,
        complex,
        bool,
        type(None),
        np.ndarray,
        np.generic,
        types.BuiltinFunctionType,
        types.BuiltinMethodType,
        types.FunctionType,
        types.MethodType,
        types.ModuleType,
        type,
    )
    while stack:
        obj = stack.pop()
        if isinstance(obj, atomic):
            continue
        ident = id(obj)
        if ident in seen or ident in preserved:
            continue
        seen.add(ident)
        if isinstance(obj, (np.random.RandomState, np.random.Generator)):
            rngs.append(obj)
            continue
        if _is_stateful_module(obj):
            continue
        if hasattr(obj, "__dict__") or any("__slots__" in cls.__dict__ for cls in type(obj).__mro__):
            objects.append(obj)
        stack.extend(_algorithm_children(obj))
    return tuple(objects), tuple(rngs)


@dataclass
class AlgorithmStateSnapshot:
    """In-place snapshot of a strategy object graph and every reachable NumPy RNG."""

    dict_entries: tuple[tuple[Any, dict[str, Any]], ...]
    slot_entries: tuple[tuple[Any, str, Any], ...]
    rng_entries: tuple[tuple[Any, Any], ...]
    preserved: tuple[Any, ...]

    @classmethod
    def capture(cls, strategy: Any, *preserve: Any) -> AlgorithmStateSnapshot:
        preserved = {id(value) for value in preserve}
        objects, rngs = _algorithm_objects(strategy, preserved)
        memo = {id(value): value for value in (*objects, *rngs, *preserve)}
        dict_entries = []
        slot_entries = []
        for obj in objects:
            if hasattr(obj, "__dict__"):
                dict_entries.append((obj, copy.deepcopy(vars(obj), memo)))
            for owner in type(obj).__mro__:
                slots = owner.__dict__.get("__slots__", ())
                if isinstance(slots, str):
                    slots = (slots,)
                for name in slots:
                    if name in {"__dict__", "__weakref__"}:
                        continue
                    attr_name = name
                    if name.startswith("__") and not name.endswith("__"):
                        attr_name = f"_{owner.__name__.lstrip('_')}{name}"
                    if hasattr(obj, attr_name):
                        slot_entries.append((obj, attr_name, copy.deepcopy(getattr(obj, attr_name), memo)))
        rng_entries = []
        for rng in rngs:
            if isinstance(rng, np.random.RandomState):
                rng_entries.append((rng, copy.deepcopy(rng.get_state())))
            else:
                rng_entries.append((rng, copy.deepcopy(rng.bit_generator.state)))
        return cls(tuple(dict_entries), tuple(slot_entries), tuple(rng_entries), tuple(preserve))

    def restore(self) -> None:
        """Restore captured objects without replacing their externally visible identities."""
        identities = [obj for obj, _ in self.dict_entries]
        identities.extend(obj for obj, _, _ in self.slot_entries)
        identities.extend(rng for rng, _ in self.rng_entries)
        memo = {id(value): value for value in (*identities, *self.preserved)}
        for obj, state in self.dict_entries:
            vars(obj).clear()
            vars(obj).update(copy.deepcopy(state, memo))
        for obj, name, value in self.slot_entries:
            setattr(obj, name, copy.deepcopy(value, memo))
        for rng, state in self.rng_entries:
            if isinstance(rng, np.random.RandomState):
                rng.set_state(copy.deepcopy(state))
            else:
                rng.bit_generator.state = copy.deepcopy(state)
