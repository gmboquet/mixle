"""Turn a fitted model into a reusable simulator with scenarios.

A fitted generative model already has sampling behavior. :func:`simulate`
packages that behavior into a :class:`Simulator` that can produce baseline
synthetic data and, for learned Bayesian networks, named intervention
scenarios.

Non-graph models simulate from their baseline distribution. Interventions
require the causal structure exposed by
:class:`~mixle.inference.HeterogeneousBayesianNetwork`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


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
        """Generate ``n`` synthetic records under the baseline, a registered ``scenario``, or ad-hoc ``interventions``."""
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
            return list(gen.sample(int(n), seed=seed)) if _accepts_seed(gen.sample) else list(gen.sample(int(n)))
        sampler = self.model.sampler(seed=seed)
        out = sampler.sample(int(n))
        return list(out) if not isinstance(out, list) else out

    def outcome_mean(self, field_index: int, *, scenario: str | None = None, n: int = 2000, seed: int = 0) -> float:
        """The mean of a numeric field under a scenario -- the quantity to compare across conditions."""
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
