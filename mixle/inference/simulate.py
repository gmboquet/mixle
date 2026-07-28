"""Turn a fitted model into a reusable simulator with scenarios.

A fitted generative model already has sampling behavior. :func:`simulate`
packages that behavior into a :class:`Simulator` that can produce baseline
synthetic data and, for learned Bayesian networks, named intervention
scenarios.

Non-graph models simulate from their baseline distribution. Interventions
require the causal structure exposed by
:class:`~mixle.inference.HeterogeneousBayesianNetwork`.

Every draw count is an exact positive integer and every run returns exactly that
many records: a Monte Carlo precision claim, an effect comparison, or a receipt
that says ``n`` trials were run is only worth what the realized sample size is,
so an underproducing generator raises :class:`IncompleteSimulationError` rather
than silently shrinking the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class IncompleteSimulationError(RuntimeError):
    """A generator produced a different number of records than the experiment requested.

    ``records`` holds what it did produce and ``requested`` the size that was asked for, so a caller can
    decide what to do with a short run instead of summarizing it as though it were the full one.
    """

    def __init__(self, message: str, records: list[Any], requested: int) -> None:
        super().__init__(message)
        self.records = records
        self.requested = requested


def _exact_draws(n: Any) -> int:
    """``n`` as an exact positive integer draw count.

    ``bool`` is rejected on purpose (``True`` would run a one-trial experiment), as are fractional counts
    (``int(2.9)`` silently ran two) and non-positive ones (a zero-trial run summarizes to ``nan``).
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise TypeError(f"number of draws must be an exact integer, got {n!r}")
    count = int(n)
    if count < 1:
        raise ValueError(f"number of draws must be positive, got {count}")
    return count


@dataclass
class Scenario:
    """A named simulation condition: which fields are clamped to which values (an intervention)."""

    name: str
    interventions: dict[int, Any] = field(default_factory=dict)


class Simulator:
    """A fitted model packaged as a data generator, runnable under a baseline or named scenarios."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self._is_bn = hasattr(model, "factors") and hasattr(model, "order")
        self.scenarios: dict[str, Scenario] = {}

    def scenario(self, name: str, interventions: dict[int, Any]) -> Simulator:
        """Register a named intervention scenario (requires a learned Bayesian network to apply)."""
        if interventions and not self._is_bn:
            raise TypeError("intervention scenarios need a HeterogeneousBayesianNetwork (do-operator)")
        self.scenarios[name] = Scenario(name, dict(interventions))
        return self

    def run(
        self, n: int = 100, *, scenario: str | None = None, interventions: dict[int, Any] | None = None, seed: int = 0
    ) -> list[Any]:
        """Generate exactly ``n`` synthetic records under the baseline, a registered ``scenario``, or ad-hoc
        ``interventions``.

        ``n`` must be an exact positive integer. Raises :class:`IncompleteSimulationError` if the underlying
        generator yields a different number of records than requested.
        """
        draws = _exact_draws(n)
        iv = dict(interventions or {})
        if scenario is not None:
            if scenario not in self.scenarios:
                raise KeyError(f"no scenario named {scenario!r}; register it with .scenario(...)")
            iv.update(self.scenarios[scenario].interventions)
        if iv:
            if not self._is_bn:
                raise TypeError("interventions need a HeterogeneousBayesianNetwork")
            from mixle.inference.causal import do

            gen = do(self.model, iv)
            out = gen.sample(draws, seed=seed) if _accepts_seed(gen.sample) else gen.sample(draws)
        else:
            out = self.model.sampler(seed=seed).sample(draws)
        rows = out if isinstance(out, list) else list(out)
        if len(rows) != draws:
            raise IncompleteSimulationError(
                f"simulation requested {draws} records but the generator produced {len(rows)}; "
                "an experiment size cannot be reported as run when it was not",
                rows,
                draws,
            )
        return rows

    def outcome_mean(self, field_index: int, *, scenario: str | None = None, n: int = 2000, seed: int = 0) -> float:
        """The mean of a numeric field over exactly ``n`` draws -- the quantity to compare across conditions.

        The run is size-checked first (see :meth:`run`), so the returned mean is always over ``n``
        observations rather than over however many the generator happened to yield.
        """
        rows = self.run(n, scenario=scenario, seed=seed)
        return float(np.mean([float(r[field_index]) for r in rows]))

    def compare(
        self, field_index: int, scenario_a: str | None, scenario_b: str | None, *, n: int = 4000, seed: int = 0
    ) -> float:
        """``mean(field | scenario_a) - mean(field | scenario_b)`` -- the simulated effect of A vs B."""
        return self.outcome_mean(field_index, scenario=scenario_a, n=n, seed=seed) - self.outcome_mean(
            field_index, scenario=scenario_b, n=n, seed=seed
        )


def _accepts_seed(fn: Any) -> bool:
    import inspect

    try:
        return "seed" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def simulate(model: Any) -> Simulator:
    """Package a fitted ``model`` as a :class:`Simulator` (see module docstring)."""
    return Simulator(model)
