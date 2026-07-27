"""Shared validation and bounded-failure contracts for stopping-time Markov models."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

DEFAULT_TERMINAL_STEP_CAP = 1_000_000


class HiddenMarkovNonterminationError(RuntimeError):
    """Raised when bounded stopping-time sampling does not reach its terminal event."""

    def __init__(self, *, mode: str, max_steps: int, last_state: int | None = None) -> None:
        self.mode = str(mode)
        self.max_steps = int(max_steps)
        self.last_state = last_state
        state = "" if last_state is None else " (last state %d)" % last_state
        super().__init__(
            "hidden Markov %s sampling did not terminate within %d steps%s" % (self.mode, self.max_steps, state)
        )


def validated_state_ids(
    values: Iterable[Any] | None,
    n_states: int,
    *,
    context: str,
    field: str,
) -> frozenset[int] | None:
    """Return exact in-range state IDs without lossy integer coercion."""
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise TypeError("%s %s must be an iterable of exact integer state IDs" % (context, field))
    try:
        raw = list(values)
    except TypeError as exc:
        raise TypeError("%s %s must be an iterable of exact integer state IDs" % (context, field)) from exc
    if not raw:
        raise ValueError("%s %s cannot be empty" % (context, field))
    states: set[int] = set()
    for value in raw:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError("%s %s IDs must be exact integers, got %r" % (context, field, value))
        state = int(value)
        if state < 0 or state >= n_states:
            raise ValueError("%s %s ID %d is outside [0, %d)" % (context, field, state, n_states))
        states.add(state)
    return frozenset(states)


def validated_terminal_states(values: Iterable[Any] | None, n_states: int, *, context: str) -> frozenset[int] | None:
    """Return exact in-range terminal state IDs without lossy integer coercion."""
    return validated_state_ids(values, n_states, context=context, field="terminal_states")


def validated_terminal_values(values: Iterable[Any] | None, *, context: str) -> frozenset[Any] | None:
    """Return an owned non-empty set of terminal emissions."""
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise TypeError("%s terminal_values must be a collection, not a string scalar" % context)
    try:
        result = frozenset(values)
    except TypeError as exc:
        raise TypeError("%s terminal_values must contain hashable values" % context) from exc
    if not result:
        raise ValueError("%s terminal_values cannot be empty" % context)
    return result


def validate_terminal_reachability(
    initial: Any,
    transitions: Any,
    terminal_states: frozenset[int] | None,
    *,
    context: str,
) -> None:
    """Require every initially reachable state to have a positive-probability path to termination."""
    if terminal_states is None:
        return
    initial_array = np.asarray(initial, dtype=np.float64)
    transition_array = np.asarray(transitions, dtype=np.float64)

    reachable = initial_array > 0.0
    frontier = list(np.flatnonzero(reachable))
    while frontier:
        state = frontier.pop()
        for child in np.flatnonzero(transition_array[state] > 0.0):
            if not reachable[child]:
                reachable[child] = True
                frontier.append(int(child))

    can_terminate = np.zeros(len(initial_array), dtype=bool)
    can_terminate[list(terminal_states)] = True
    frontier = list(terminal_states)
    while frontier:
        state = frontier.pop()
        for parent in np.flatnonzero(transition_array[:, state] > 0.0):
            if not can_terminate[parent]:
                can_terminate[parent] = True
                frontier.append(int(parent))

    trapped = np.flatnonzero(reachable & ~can_terminate)
    if trapped.size:
        raise ValueError(
            "%s has reachable non-terminal states with no path to termination: %s" % (context, trapped.tolist())
        )


def validated_terminal_step_cap(value: Any) -> int:
    """Return an exact positive stopping-time sampling bound."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("terminal sampling max_steps must be an exact positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError("terminal sampling max_steps must be positive")
    return result


def require_terminal_reached(
    reached: bool,
    *,
    mode: str,
    max_steps: int,
    last_state: int | None = None,
) -> None:
    """Raise the typed bounded-sampling failure when the stop event was not reached."""
    if not reached:
        raise HiddenMarkovNonterminationError(mode=mode, max_steps=max_steps, last_state=last_state)
