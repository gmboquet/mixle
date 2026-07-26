"""Independent step/token/observation clocks for multi-rate update nodes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Integral, Real
from typing import Any

from mixle.experimental.typed_runtime.contracts import CounterSemantics


@dataclass(frozen=True)
class ClockProgress:
    """Monotone run progress against which node cadences are evaluated."""

    step: int = 0
    tokens: int = 0
    observations: float = 0.0
    counter_semantics: CounterSemantics = CounterSemantics.CUMULATIVE

    def __post_init__(self) -> None:
        integers = (self.step, self.tokens)
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in integers):
            raise ValueError("clock progress must be non-negative.")
        if (
            isinstance(self.observations, bool)
            or not isinstance(self.observations, Real)
            or not math.isfinite(float(self.observations))
            or self.observations < 0.0
        ):
            raise ValueError("clock progress observations must be finite and non-negative.")
        if self.counter_semantics is not CounterSemantics.CUMULATIVE:
            raise ValueError("clock progress counters must use cumulative semantics.")

    def as_dict(self) -> dict[str, int | float]:
        """Return a JSON-compatible progress vector."""

        return {
            "step": self.step,
            "tokens": self.tokens,
            "observations": self.observations,
            "counter_semantics": self.counter_semantics.value,
        }


@dataclass(frozen=True)
class UpdateCadence:
    """Any-trigger cadence with an optional hard staleness bound."""

    every_steps: int | None = None
    every_tokens: int | None = None
    every_observations: float | None = None
    max_staleness_steps: int | None = None

    def __post_init__(self) -> None:
        values = (self.every_steps, self.every_tokens, self.every_observations, self.max_staleness_steps)
        if all(value is None for value in values):
            raise ValueError("an update cadence requires at least one trigger.")
        for name in ("every_steps", "every_tokens", "max_staleness_steps"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer when supplied.")
        if self.every_observations is not None and (
            isinstance(self.every_observations, bool)
            or not isinstance(self.every_observations, Real)
            or not math.isfinite(float(self.every_observations))
            or self.every_observations <= 0.0
        ):
            raise ValueError("every_observations must be finite and positive when supplied.")


class ClockTrigger(StrEnum):
    """Reason an update clock became due."""

    STEP = "step"
    TOKENS = "tokens"
    OBSERVATIONS = "observations"
    STALENESS_BOUND = "staleness_bound"
    NEVER_COMMITTED = "never_committed"


@dataclass(frozen=True)
class ClockDecision:
    """Due/not-due decision and elapsed work since the last commit."""

    node_id: str
    due: bool
    triggers: tuple[ClockTrigger, ...]
    progress: ClockProgress
    last_commit: ClockProgress | None
    commit_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("clock decisions require a non-empty node_id.")
        if not isinstance(self.due, bool):
            raise TypeError("clock decision due must be boolean.")
        if (
            not isinstance(self.triggers, tuple)
            or any(not isinstance(trigger, ClockTrigger) for trigger in self.triggers)
            or len(set(self.triggers)) != len(self.triggers)
        ):
            raise ValueError("clock decision triggers must be unique ClockTrigger values.")
        if self.due != bool(self.triggers):
            raise ValueError("clock decision due flag must agree with its triggers.")
        if not isinstance(self.progress, ClockProgress):
            raise TypeError("clock decision progress must be ClockProgress.")
        if self.last_commit is not None and not isinstance(self.last_commit, ClockProgress):
            raise TypeError("clock decision last_commit must be ClockProgress or None.")
        if isinstance(self.commit_count, bool) or not isinstance(self.commit_count, Integral) or self.commit_count < 0:
            raise ValueError("clock decision commit_count must be a non-negative integer.")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible clock decision."""

        return {
            "node_id": self.node_id,
            "due": self.due,
            "triggers": [trigger.value for trigger in self.triggers],
            "progress": self.progress.as_dict(),
            "last_commit": self.last_commit.as_dict() if self.last_commit is not None else None,
            "commit_count": self.commit_count,
        }


@dataclass
class MultiRateUpdateClocks:
    """Stateful cadence registry that never permits progress to move backward."""

    cadences: dict[str, UpdateCadence]
    _last_commit: dict[str, ClockProgress] = field(default_factory=dict, init=False, repr=False)
    _commit_count: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _last_progress: ClockProgress = field(default_factory=ClockProgress, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.cadences, dict) or not self.cadences:
            raise ValueError("multi-rate clocks require non-empty node ids and cadences.")
        if any(not isinstance(node_id, str) or not node_id for node_id in self.cadences):
            raise ValueError("multi-rate clocks require non-empty node ids and cadences.")
        if any(not isinstance(cadence, UpdateCadence) for cadence in self.cadences.values()):
            raise TypeError("multi-rate clock values must be UpdateCadence instances.")
        self.cadences = dict(self.cadences)

    def _validate_progress(self, progress: ClockProgress) -> None:
        if not isinstance(progress, ClockProgress):
            raise TypeError("clock progress must be ClockProgress.")
        previous = self._last_progress
        if (
            progress.step < previous.step
            or progress.tokens < previous.tokens
            or progress.observations < previous.observations
        ):
            raise ValueError("clock progress cannot move backward.")
        self._last_progress = progress

    def evaluate(self, progress: ClockProgress, node_ids: tuple[str, ...] | None = None) -> tuple[ClockDecision, ...]:
        """Evaluate selected or all registered clocks without marking a commit."""

        self._validate_progress(progress)
        selected = tuple(self.cadences) if node_ids is None else node_ids
        unknown = sorted(set(selected) - set(self.cadences))
        if unknown:
            raise KeyError("unknown update clocks: %s" % ", ".join(unknown))
        decisions = []
        for node_id in selected:
            cadence = self.cadences[node_id]
            last = self._last_commit.get(node_id)
            count = self._commit_count.get(node_id, 0)
            triggers: list[ClockTrigger] = []
            if last is None:
                triggers.append(ClockTrigger.NEVER_COMMITTED)
            else:
                step_delta = progress.step - last.step
                token_delta = progress.tokens - last.tokens
                observation_delta = progress.observations - last.observations
                if cadence.every_steps is not None and step_delta >= cadence.every_steps:
                    triggers.append(ClockTrigger.STEP)
                if cadence.every_tokens is not None and token_delta >= cadence.every_tokens:
                    triggers.append(ClockTrigger.TOKENS)
                if cadence.every_observations is not None and observation_delta >= cadence.every_observations:
                    triggers.append(ClockTrigger.OBSERVATIONS)
                if cadence.max_staleness_steps is not None and step_delta >= cadence.max_staleness_steps:
                    triggers.append(ClockTrigger.STALENESS_BOUND)
            decisions.append(ClockDecision(node_id, bool(triggers), tuple(triggers), progress, last, count))
        return tuple(decisions)

    def mark_committed(self, node_ids: tuple[str, ...], progress: ClockProgress) -> None:
        """Advance selected clocks after a successful transaction only."""

        self._validate_progress(progress)
        unknown = sorted(set(node_ids) - set(self.cadences))
        if unknown:
            raise KeyError("unknown update clocks: %s" % ", ".join(unknown))
        for node_id in dict.fromkeys(node_ids):
            self._last_commit[node_id] = progress
            self._commit_count[node_id] = self._commit_count.get(node_id, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        """Return replay-relevant clock state."""

        return {
            "last_progress": self._last_progress.as_dict(),
            "nodes": {
                node_id: {
                    "last_commit": self._last_commit[node_id].as_dict() if node_id in self._last_commit else None,
                    "commit_count": self._commit_count.get(node_id, 0),
                }
                for node_id in self.cadences
            },
        }


__all__ = ["ClockDecision", "ClockProgress", "ClockTrigger", "MultiRateUpdateClocks", "UpdateCadence"]
